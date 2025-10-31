import os

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Grad-CAM + Attention timeline (XAI)
# -------------------------
def get_last_conv(model: nn.Module):
    """Finds the last Conv2d layer in the model's backbone."""
    last = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def denorm_to_uint8(img_chw):
    """Denormalizes a PyTorch tensor (C, H, W) to an 8-bit numpy array (H, W, C)."""
    x = img_chw.detach().cpu().permute(1,2,0).numpy()
    # Assuming normalization was: (x - 0.5) / 0.5
    x = (x*0.5) + 0.5
    x = np.clip(x, 0, 1)
    return (x*255).round().astype(np.uint8)

def gradcam_for_pred(model, img_tensor, target_layer):
    """
    Computes the Grad-CAM heatmap for the predicted class.
    
    Args:
        model: The PyTorch model (must have a forward pass that uses target_layer).
        img_tensor: The input image tensor (C, H, W).
        target_layer: The last convolutional layer for feature extraction.
        
    Returns:
        A tuple (cam_heatmap_01, predicted_class_index).
    """
    device = next(model.parameters()).device
    x = img_tensor.unsqueeze(0).to(device).clone().requires_grad_(True)
    model.zero_grad()
    feats = None
    grads = None

    # Hooks to capture features and gradients
    def fwd_hook(_, __, out): 
        nonlocal feats; feats = out
    def bwd_hook(_, grad_in, grad_out):
        nonlocal grads; grads = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    logits = model(x)
    pred = int(torch.argmax(logits, dim=1))
    score = logits[0, pred]
    score.backward(retain_graph=True)

    assert feats is not None and grads is not None, "Grad-CAM hooks failed."
    
    # Global Average Pooling of gradients
    w = grads.mean(dim=(2,3), keepdim=True)        # [B,C,1,1]
    
    # Compute the weighted feature map and apply ReLU
    cam = torch.relu((w * feats).sum(dim=1, keepdim=True))  # [B,1,Hf,Wf]
    
    # Upsample the heatmap to the size of the input image
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    
    # Normalize the heatmap to [0, 1]
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove(); h2.remove()
    return cam, pred

def to_color_map(h01):
    """Converts a [0, 1] grayscale heatmap to a JET colormap image."""
    h8 = (np.clip(h01,0,1)*255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

def export_xai_panels(model, loader, out_dir, max_samples=8, amp_device="cpu", use_amp=False):
    """
    Generates and exports XAI panels (Grad-CAM overlay and Attention timeline).
    
    NOTE: The model's forward pass must populate `model.last_attn_weights`.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    target_layer = get_last_conv(model)
    assert target_layer is not None, "No Conv2d for Grad-CAM."

    saved = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        for i in range(xb.size(0)):
            if saved >= max_samples:
                return
                
            img = xb[i]
            
            # 1. Attention Weights (requires a forward pass)
            with torch.no_grad():
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    # Forward pass to populate model.last_attn_weights
                    _ = model(img.unsqueeze(0))
                attn = model.last_attn_weights[0].cpu().numpy()  # [Hf]
            
            # 2. Grad-CAM Heatmap
            cam01, pred = gradcam_for_pred(model, img, target_layer)

            # 3. Create Grad-CAM overlay image
            rgb = denorm_to_uint8(img.cpu())
            cam_rgb = to_color_map(cam01)
            overlay = cv2.addWeighted(rgb.astype(np.float32), 0.6, cam_rgb.astype(np.float32), 0.4, 0)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # 4. Attention timeline plot
            plt.figure(figsize=(6,2.2))
            plt.plot(np.arange(len(attn)), attn, marker='o')
            plt.xlabel("Rachis step (base->tip)"); plt.ylabel("Attention")
            plt.title(f"Attention timeline (pred={pred})")
            plt.tight_layout()
            attn_path = os.path.join(out_dir, f"attn_{saved:03d}.png")
            plt.savefig(attn_path, dpi=150); plt.close()

            # 5. Side-by-side panel (Original + Overlay)
            H, W = rgb.shape[:2]
            pad = 10
            canvas = Image.new('RGB', (W*2 + 3*pad, H + 2*pad), (255,255,255))
            canvas.paste(Image.fromarray(rgb), (pad, pad))
            canvas.paste(Image.fromarray(overlay), (W + 2*pad, pad))
            panel_path = os.path.join(out_dir, f"xai_{saved:03d}_pred-{pred}.png")
            canvas.save(panel_path)

            print(f"[XAI] Saved: {panel_path}  &  {attn_path}")
            saved += 1
