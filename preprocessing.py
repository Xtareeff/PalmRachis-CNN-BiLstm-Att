import os, math, random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- Configuration (CFG is assumed to be imported or defined elsewhere for full context) ---

def stratified_split_indices(dataset: datasets.ImageFolder, val_split=0.2, seed=42):
    """
    Splits dataset indices into training and validation sets in a stratified manner.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    idx = np.arange(len(dataset))
    tr, va = [], []
    for cls in np.unique(targets):
        cls_idx = idx[targets == cls]
        rng.shuffle(cls_idx)
        n_val = int(math.ceil(val_split * len(cls_idx)))
        va.extend(cls_idx[:n_val].tolist()); tr.extend(cls_idx[n_val:].tolist())
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

def get_dataloaders(data_dir, batch_size=32, img_size=128, val_split=0.2, num_workers=2, use_pin=False, seed=42):
    """
    Creates ImageFolder datasets and DataLoaders for training and validation.
    Performs stratified splitting.
    """
    # Define transformations
    train_tf = transforms.Compose([
        transforms.Lambda(lambda im: im.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Lambda(lambda im: im.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    # Base dataset for splitting
    base_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
    classes = base_train.classes
    
    # Stratified split
    tr_idx, va_idx = stratified_split_indices(base_train, val_split=val_split, seed=seed)
    train_ds = Subset(base_train, tr_idx)

    # Base dataset for validation (using only validation transforms)
    base_val = datasets.ImageFolder(root=data_dir, transform=val_tf)
    val_ds = Subset(base_val, va_idx)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin)
    return train_loader, val_loader, classes
