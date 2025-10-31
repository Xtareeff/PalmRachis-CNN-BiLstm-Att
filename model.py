import os, csv
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Model: ResNet18 -> rachis-wise sequence (avg over width) -> BiLSTM + Attention
# with AUTO-ORIENTATION (base->tip) heuristic
# -------------------------
class PalmRachis_BiLSTM_Attn(nn.Module):
    """
    CNN-BiLSTM model with Attention for Palm Rachis analysis.
    The CNN backbone (ResNet18) extracts features, which are then averaged
    across the width to form a sequence (rachis-wise). A BiLSTM processes
    this sequence, and an Attention mechanism pools the sequence output.
    Includes an auto-orientation heuristic to ensure sequence consistency.
    """
    def __init__(self, num_classes=9, hidden_dim=256, num_layers=1, dropout=0.5,
                 freeze_until=6, auto_orient=True):
        super().__init__()
        self.auto_orient = auto_orient

        # ResNet18 backbone (remove last two layers: AvgPool and FC)
        base = models.resnet18(weights=None)                 # offline-safe
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # [B, 512, Hf, Wf]
        
        # Freeze a few early layers
        for i, (_, p) in enumerate(self.backbone.named_parameters()):
            if i < freeze_until:
                p.requires_grad = False

        self.hidden_dim = hidden_dim
        # BiLSTM: input_size is 512 (from ResNet18 feature maps)
        self.bilstm = nn.LSTM(
            input_size=512, hidden_size=hidden_dim,
            num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout
        )
        # Attention layer: input_size is 2*hidden_dim (from BiLSTM output)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

        # for XAI/debug
        self.last_feat_hw = None           # (Hf, Wf)
        self.last_attn_weights = None      # [B, Hf]
        self.last_flipped_mask = None      # [B, 1, 1] True if sequence flipped this forward

    def forward(self, x):
        f = self.backbone(x)               # [B, 512, Hf, Wf]
        B, C, Hf, Wf = f.shape
        self.last_feat_hw = (Hf, Wf)

        # --- Build rachis-wise sequence: average across width (keep vertical steps) ---
        seq = f.mean(dim=3)                # [B, 512, Hf]
        seq = seq.permute(0, 2, 1)         # [B, Hf, 512]  (top->bottom by default)

        # --- AUTO ORIENTATION to ensure sequence starts at the BASE ---
        if self.auto_orient and Hf >= 4:
            # Compare mean activation in top vs bottom quarter
            top_q    = f[:, :, :Hf//4, :].abs().mean(dim=(1, 2, 3))     # [B]
            bottom_q = f[:, :, -Hf//4:, :].abs().mean(dim=(1, 2, 3))    # [B]
            # If bottom stronger -> base likely at bottom -> flip to bring base to the front of the sequence
            need_flip = (bottom_q > top_q).view(B, 1, 1)                 # [B,1,1] boolean
            self.last_flipped_mask = need_flip.detach()
            # Flip the sequence if needed
            seq = torch.where(need_flip, seq.flip(dims=[1]), seq)
        else:
            self.last_flipped_mask = torch.zeros((B,1,1), dtype=torch.bool, device=seq.device)

        # --- BiLSTM + attention over rachis steps (base->tip) ---
        h, _ = self.bilstm(seq)            # [B, Hf, 2*hidden]
        # Compute attention weights
        attn = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # [B, Hf]
        self.last_attn_weights = attn.detach()

        # Apply attention to get context vector
        ctx = torch.sum(h * attn.unsqueeze(-1), dim=1)         # [B, 2*hidden]
        
        # Classifier layers
        z = torch.relu(self.fc1(ctx))
        z = self.dropout(z)
        return self.fc2(z)


# =====================================
# TRAIN FUNCTION
# =====================================

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3,
                amp_device="cpu", use_amp=False, out_dir="."):
    """
    Trains the model, logs metrics, and saves the best checkpoint.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-5)

    # Scheduler (Reduce LR on plateau monitoring val_acc)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_path = os.path.join(out_dir, "best_model.pth")
    os.makedirs(out_dir, exist_ok=True)
    log_csv = os.path.join(out_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            run_loss += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
            total += bs

        train_loss = run_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_loss_accum, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                v_loss_accum += loss.item() * xb.size(0)
                v_corr += (logits.argmax(1) == yb).sum().item()
                v_tot += xb.size(0)

        val_loss = v_loss_accum / max(1, v_tot)
        val_acc = v_corr / max(1, v_tot)

        # record LR
        current_lr = optimizer.param_groups[0]['lr']

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)
        hist["lr"].append(current_lr)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

        # Scheduler step (monitoring validation accuracy)
        try:
            scheduler.step(val_acc)
        except Exception:
            # fallback: if scheduler expects loss, use val_loss
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"[CKPT] Saved best to {best_path} (val_acc={best_val_acc:.4f})")

    # Plot training curves
    try:
        ep = range(1, len(hist["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(ep, hist["train_loss"], label="train", linewidth=2)
        plt.plot(ep, hist["val_loss"], label="val", linestyle="--", linewidth=2, color="orange")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(ep, hist["train_acc"], label="train", linewidth=2)
        plt.plot(ep, hist["val_acc"], label="val", linestyle="--", linewidth=2, color="green")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_val_curves_true.png"), dpi=150)
        plt.close()
        print("[PLOT] Saved true accuracy/loss curves.")
    except Exception as e:
        print(f"Could not save curves: {e}")

    # also save hist as numpy for later
    np.save(os.path.join(out_dir, "train_hist.npy"), hist)

    return best_path, best_val_acc, hist


# =====================================
# EVALUATION FUNCTION
# =====================================

def evaluate(model, loader, device, class_names: List[str], amp_device="cpu", use_amp=False, out_dir="."):
    """
    Evaluates the model, prints classification report, plots confusion matrix and ROC curves.
    """
    model.eval()
    y_true, y_pred = [], []
    y_scores = []  # softmax probabilities per sample
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())

    # confusion & report
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # plot & save confusion matrix
    fig, ax = plt.subplots(figsize=(0.7*len(class_names)+4, 0.7*len(class_names)+4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    # --- Multiclass ROC (One-vs-Rest) ---
    try:
        n_classes = len(class_names)
        # binarize the true labels
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))  # shape (n_samples, n_classes)
        y_scores_arr = np.array(y_scores)  # shape (n_samples, n_classes)

        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_arr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average and macro-average
        # micro
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores_arr.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # macro-average: aggregate all fpr points
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC (AUC = {roc_auc["micro"]:.3f})', color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC (AUC = {roc_auc["macro"]:.3f})', color='navy', linestyle=':', linewidth=2)

        colors = plt.cm.get_cmap('tab10', n_classes)
        for i, cname in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], color=colors(i), lw=1.5,
                     label=f'ROC {cname} (AUC = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True)
        roc_path = os.path.join(out_dir, "roc_multiclass.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"[ROC] Saved multiclass ROC to: {roc_path}")
    except Exception as e:
        print(f"Could not compute ROC: {e}")

    # return metrics
    acc = accuracy_score(y_true, y_pred)
    return {"acc": acc, "cm": cm, "y_true": y_true, "y_pred": y_pred, "y_scores": y_scores}
