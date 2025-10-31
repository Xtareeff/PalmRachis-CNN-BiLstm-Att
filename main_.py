import os, random, warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Import modules
from preprocessing import get_dataloaders
from model import PalmRachis_BiLSTM_Attn, train_model, evaluate
from xai import export_xai_panels

# Suppress PIL UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    DATA_DIR: str = "../input/pam-leaf-dataa/Processed"  # <-- Path of Palm Dataset
    OUT_DIR:  str = "./palm_rachis_out"                  # <-- Path of output Folder
    IMG_SIZE: int = 128
    BATCH: int = 16
    EPOCHS: int = 2
    LR: float = 1e-4
    VAL_SPLIT: float = 0.3
    SEED: int = 42
    NUM_WORKERS: int = 2
    EXPORT_XAI: int = 15        # number of XAI panels to export

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    print("DATA_DIR:", cfg.DATA_DIR)
    print("OUT_DIR :", cfg.OUT_DIR)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = (amp_device == "cuda")
    print(f"Device: {device} | AMP: {'ON' if USE_AMP else 'OFF'}")

    pin_mem = (device.type == "cuda")

    # 1. Get DataLoaders
    train_loader, val_loader, class_names = get_dataloaders(
        cfg.DATA_DIR, batch_size=cfg.BATCH, img_size=cfg.IMG_SIZE,
        val_split=cfg.VAL_SPLIT, num_workers=cfg.NUM_WORKERS, use_pin=pin_mem,
        seed=cfg.SEED
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    out_run = os.path.join(cfg.OUT_DIR, "run_rachis_attn")
    os.makedirs(out_run, exist_ok=True)

    # 2. Initialize Model
    model = PalmRachis_BiLSTM_Attn(num_classes=len(class_names)).to(device)

    # 3. Train Model
    print("\n--- Starting Training ---")
    best_path, best_val, hist = train_model(
        model, train_loader, val_loader, device,
        epochs=cfg.EPOCHS, lr=cfg.LR,
        amp_device=amp_device, use_amp=USE_AMP, out_dir=out_run
    )
    
    # 4. Evaluate Model (reload best weights)
    print("\n--- Starting Evaluation ---")
    model.load_state_dict(torch.load(best_path, map_location=device))
    metrics = evaluate(model, val_loader, device, class_names,
                       amp_device=amp_device, use_amp=USE_AMP, out_dir=out_run)
    print("Eval accuracy:", metrics["acc"])

    # 5. Export XAI Panels
    if cfg.EXPORT_XAI and cfg.EXPORT_XAI > 0:
        print("\n--- Exporting XAI Panels ---")
        xai_dir = os.path.join(out_run, "xai")
        export_xai_panels(model, val_loader, xai_dir, max_samples=cfg.EXPORT_XAI,
                          amp_device=amp_device, use_amp=USE_AMP)

    print("\nDone.")
