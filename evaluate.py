"""
evaluate.py

CLI evaluation entrypoint for the CS 535 traffic sign classification project.
Loads a trained checkpoint and reports test accuracy, macro F1, a confusion
matrix, a classification report, and inference latency benchmarks.

Usage:
    python evaluate.py --model efficientnet_b0 --checkpoint checkpoints/efficientnet_b0.pt
    python evaluate.py --model cnn --checkpoint checkpoints/cnn.pt --benchmark
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from dataset import cfg, set_seed, download_gtsrb, build_dataloaders
from models import build_model
from utils import (
    evaluate_model,
    get_predictions,
    load_checkpoint,
    plot_confusion_matrix,
    benchmark_inference,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GTSRB classifier.")
    parser.add_argument("--model", type=str, required=True,
                         choices=["cnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a saved model checkpoint (.pt file).")
    parser.add_argument("--figure-dir", type=str, default=cfg.FIGURE_DIR)
    parser.add_argument("--results-dir", type=str, default=cfg.RESULTS_DIR)
    parser.add_argument("--benchmark", action="store_true",
                         help="Also run inference latency benchmarking.")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    download_gtsrb(cfg)
    loaders, _ = build_dataloaders(cfg)
    test_loader = loaders["test"]

    model = build_model(args.model)
    model = load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    preds, labels = get_predictions(model, test_loader, device)
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")

    # Confusion matrix
    cm_path = os.path.join(args.figure_dir, f"confusion_matrix_{args.model}.png")
    plot_confusion_matrix(preds, labels, args.model, num_classes=cfg.NUM_CLASSES, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Classification report
    class_names = [str(i) for i in range(cfg.NUM_CLASSES)]
    report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(args.results_dir, f"classification_report_{args.model}.csv")
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    # Optional inference benchmark
    if args.benchmark:
        stats = benchmark_inference(model, device, img_size=cfg.IMG_SIZE)
        print(f"Inference: {stats['mean_ms']:.2f} ms/img (p95={stats['p95_ms']:.2f} ms), "
              f"{stats['fps']:.1f} FPS")
        bench_df = pd.DataFrame([{"Model": args.model, **stats}])
        bench_path = os.path.join(args.results_dir, f"inference_benchmark_{args.model}.csv")
        bench_df.to_csv(bench_path, index=False)
        print(f"Benchmark saved to {bench_path}")


if __name__ == "__main__":
    main()