"""
utils.py

Shared helpers: training/evaluation loops, checkpointing, benchmarking,
and plotting utilities used by train.py and evaluate.py.
"""

import copy
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss, running_correct, n_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        n_samples += imgs.size(0)
        pbar.set_postfix(loss=running_loss / n_samples, acc=running_correct / n_samples)

    return running_loss / n_samples, running_correct / n_samples


@torch.no_grad()
def evaluate_model(model, loader, criterion, device) -> Tuple[float, float]:
    """Evaluate a model on a loader. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss, running_correct, n_samples = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        n_samples += imgs.size(0)

    return running_loss / n_samples, running_correct / n_samples


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference over a loader. Returns (preds, labels) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_model(
    model, train_loader, val_loader, epochs, lr, weight_decay,
    model_name, device, patience=5,
) -> Tuple[nn.Module, Dict]:
    """Full training loop with Adam, ReduceLROnPlateau, and early stopping.

    Restores the best validation-loss checkpoint before returning.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"[{model_name}] Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def benchmark_inference(model, device, img_size=64, n_warmup=20, n_runs=200, batch_size=1):
    """Measure per-image inference latency (mean/std/p95 in ms) and FPS."""
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)

    for _ in range(n_warmup):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    timings = np.array(timings) * 1000  # ms
    return {
        "mean_ms": timings.mean(),
        "std_ms": timings.std(),
        "p95_ms": np.percentile(timings, 95),
        "fps": 1000.0 / timings.mean(),
    }


def save_checkpoint(model, path: str) -> None:
    """Save a model's state_dict, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device) -> nn.Module:
    """Load a state_dict into a model from disk."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def plot_confusion_matrix(preds, labels, model_name, num_classes=43, save_path=None):
    """Plot and optionally save a row-normalized confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, cmap="Blues", square=True, cbar_kws={"label": "Proportion"})
    plt.title(f"Normalized Confusion Matrix — {model_name}", fontsize=14)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return cm


def plot_learning_curves(histories, model_names, save_path=None):
    """Plot loss and accuracy curves for one or more training histories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, name in zip(histories, model_names):
        axes[0].plot(hist["train_loss"], label=f"{name} (train)", linestyle="--")
        axes[0].plot(hist["val_loss"], label=f"{name} (val)")
        axes[1].plot(hist["train_acc"], label=f"{name} (train)", linestyle="--")
        axes[1].plot(hist["val_acc"], label=f"{name} (val)")

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()