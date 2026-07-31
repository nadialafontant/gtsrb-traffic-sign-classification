"""
models.py

Model definitions for the CS 535 traffic sign classification project:
  - BaselineCNN: a compact CNN trained from scratch
  - ResNet18: ImageNet-pretrained, fine-tuned end-to-end (transfer learning)
  - EfficientNet-B0: ImageNet-pretrained, fine-tuned end-to-end (transfer learning)
"""

import torch
import torch.nn as nn
from torchvision import models

from dataset import cfg


class BaselineCNN(nn.Module):
    """A compact 6-conv-layer CNN trained from scratch (non-transfer baseline)."""

    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes: int = cfg.NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """Build an ImageNet-pretrained ResNet18 with a custom classifier head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes: int = cfg.NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """Build an ImageNet-pretrained EfficientNet-B0 with a custom classifier head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


MODEL_REGISTRY = {
    "cnn": lambda: BaselineCNN(),
    "resnet18": lambda: build_resnet18(),
    "efficientnet_b0": lambda: build_efficientnet_b0(),
}


def build_model(name: str) -> nn.Module:
    """Factory: build a model by name ('cnn', 'resnet18', 'efficientnet_b0')."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}.")
    return MODEL_REGISTRY[name]()


def count_parameters(model: nn.Module):
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        m = build_model(name)
        total, trainable = count_parameters(m)
        print(f"{name}: total={total:,} trainable={trainable:,}")