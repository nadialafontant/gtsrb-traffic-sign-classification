# Real-Time Road Sign Classification for Autonomous Driving Using Transfer Learning

CS 535 — Deep Learning for Computer Vision | Final Project

A comparative study of three CNN architectures — a from-scratch baseline CNN,
ResNet18, and EfficientNet-B0 (both fine-tuned via transfer learning) — for
43-class traffic sign classification on the **German Traffic Sign Recognition
Benchmark (GTSRB)**, evaluated on accuracy, per-class F1, and inference
latency relevant to real-time autonomous driving deployment.

## Results Summary

| Model | Test Accuracy | Macro F1 | Inference (ms/img) | FPS |
|---|---|---|---|---|
| CNN (from scratch) | 96.43% | 0.957 | 1.11 | 899 |
| ResNet18 (transfer learning) | 95.72% | 0.928 | 2.76 | 362 |
| EfficientNet-B0 (transfer learning) | 93.83% | 0.899 | 8.73 | 115 |

**Key finding:** despite being the largest, ImageNet-pretrained backbone,
EfficientNet-B0 underperformed both simpler models on this task — likely
because its compound-scaled architecture is tuned for 224×224 inputs, while
GTSRB images here are resized to 64×64, and because GTSRB's centered,
low-resolution sign crops are an easier task than ImageNet, reducing the
advantage of a heavier pretrained backbone. See the full report for
discussion. Data augmentation improved EfficientNet-B0 test accuracy by
+1.64 points (93.83% vs. 92.19% without augmentation).

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── dataset.py          # GTSRB loading, transforms, config, train/val/test splits
├── models.py            # BaselineCNN, ResNet18, EfficientNet-B0 builders
├── train.py              # Training entrypoint (CLI)
├── evaluate.py         # Evaluation, confusion matrix, benchmarking (CLI)
├── utils.py              # Train/eval loops, checkpointing, plotting helpers
├── notebooks/
│   └── GTSRB_Traffic_Sign_Classification.ipynb
├── checkpoints/         # Saved model weights (generated)
├── figures/                # Generated plots (generated)
└── results/                 # Metrics, CSVs, JSON exports (generated)
```

## Setup

```bash
git clone <repo-url>
cd gtsrb-traffic-sign-classification
pip install -r requirements.txt
```

## Usage

### Train a model

```bash
python train.py --model efficientnet_b0 --epochs 10 --batch-size 128 --lr 1e-4
python train.py --model resnet18 --epochs 10 --lr 1e-4
python train.py --model cnn --epochs 15 --lr 1e-3

# Ablation arm: train without data augmentation
python train.py --model efficientnet_b0 --no-augmentation
```

Checkpoints are saved to `checkpoints/<model_name>.pt`.

### Evaluate a trained checkpoint

```bash
python evaluate.py --model efficientnet_b0 --checkpoint checkpoints/efficientnet_b0.pt --benchmark
```

Produces test-set accuracy/F1, a normalized confusion matrix, a
classification report, and (with `--benchmark`) inference latency stats —
saved to `figures/` and `results/`.

## Dataset

[GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) — 43 traffic sign
classes, ~39,000 training images, ~12,600 test images. Downloaded
automatically via `torchvision.datasets.GTSRB` on first run (no manual
download or API key required).

## Models

- **BaselineCNN** — 6-conv-layer CNN trained from scratch, no pretraining.
- **ResNet18** — ImageNet-pretrained, fully fine-tuned, custom classifier head.
- **EfficientNet-B0** — ImageNet-pretrained, fully fine-tuned, custom classifier head.

All models trained with Adam, `ReduceLROnPlateau` scheduling, and early
stopping (patience=5) on validation loss.

## Reproducing Results

Random seed fixed at `42` across NumPy, PyTorch, and CUDA for
reproducibility. Exact configuration in `dataset.py::Config`.

## Author

Nadia Lafontant — CS 491, Southern Illinois University Carbondale, Summer 2026

## Acknowledgments

- GTSRB dataset: Stallkamp et al., "Man vs. Computer: Benchmarking Machine
  Learning Algorithms for Traffic Sign Recognition," Neural Networks, 2012.
- Pretrained weights: torchvision model zoo (ImageNet1K).
