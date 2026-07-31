"""
train.py

CLI training entrypoint for the CS 535 traffic sign classification project.

Usage:
    python train.py --model efficientnet_b0 --epochs 10 --batch-size 128 --lr 1e-4
    python train.py --model resnet18 --epochs 10 --lr 1e-4
    python train.py --model cnn --epochs 15 --lr 1e-3
    python train.py --model efficientnet_b0 --no-augmentation   # ablation arm
"""

import argparse
import os

import torch

from dataset import cfg, set_seed, download_gtsrb, build_dataloaders
from models import build_model, count_parameters
from utils import train_model, evaluate_model, save_checkpoint
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GTSRB traffic sign classifier.")
    parser.add_argument("--model", type=str, required=True,
                         choices=["cnn", "resnet18", "efficientnet_b0"],
                         help="Which model architecture to train.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Number of epochs (defaults to per-model config value).")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None,
                         help="Learning rate (defaults to per-model config value).")
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--no-augmentation", action="store_true",
                         help="Train without data augmentation (ablation arm).")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--checkpoint-dir", type=str, default=cfg.CHECKPOINT_DIR)
    return parser.parse_args()


DEFAULT_EPOCHS = {"cnn": cfg.EPOCHS_CNN, "resnet18": cfg.EPOCHS_RESNET, "efficientnet_b0": cfg.EPOCHS_EFFNET}
DEFAULT_LR = {"cnn": cfg.LR_CNN, "resnet18": cfg.LR_RESNET, "efficientnet_b0": cfg.LR_EFFNET}


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = args.epochs or DEFAULT_EPOCHS[args.model]
    lr = args.lr or DEFAULT_LR[args.model]

    cfg.BATCH_SIZE = args.batch_size
    download_gtsrb(cfg)
    loaders, _ = build_dataloaders(cfg)

    train_loader = loaders["train_no_aug"] if args.no_augmentation else loaders["train"]
    val_loader = loaders["val"]

    model = build_model(args.model).to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {args.model} | Total params: {total:,} | Trainable: {trainable:,}")

    suffix = "_no_aug" if args.no_augmentation else ""
    model_name = f"{args.model}{suffix}"

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=args.weight_decay,
        model_name=model_name,
        device=device,
        patience=args.patience,
    )

    ckpt_path = os.path.join(args.checkpoint_dir, f"{model_name}.pt")
    save_checkpoint(model, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, loaders["test"], criterion, device)
    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()