# CURE-TSR: Traffic Sign Recognition with CNN

A PyTorch-based training and evaluation pipeline for traffic sign classification on the [CURE-TSR dataset](https://github.com/olivesgatech/CURE-TSR), with calibration via temperature scaling.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Calibration](#calibration)
- [Citation](#citation)

---

## Overview

This notebook trains a small CNN to classify traffic signs into 14 categories using the CURE-TSR dataset. It also evaluates model confidence calibration using reliability diagrams and Expected Calibration Error (ECE), and applies temperature scaling to explore the affects of calibration on both real and synthetic (Unreal Engine) test sets.

---

## Dataset

**CURE-TSR** (Challenging Unreal and Real Environments for Traffic Sign Recognition) contains over 2 million traffic sign images under a variety of challenging conditions. The dataset contains 14 classes of traffic signs.

---

## Requirements

- Python 3.8+
- PyTorch (CUDA-enabled recommended)
- torchvision
- NumPy
- Pillow
- matplotlib
- tqdm
- TensorFlow
- scipy

---

## Installation

```bash
pip install torch torchvision numpy pillow matplotlib tqdm tensorflow scipy
```

---

## Configuration

All hyperparameters and paths are set in the **Configuration** cell of the notebook. No command-line arguments are required.

| Parameter | Default | Description |
|---|---|---|
| `DATA_DIR` | `/storage/ice1/shared/d-pace_community/makerspace-datasets/AVs/CURE-TSR` | Root path to the CURE-TSR dataset. Will be different if not using Georgia Tech's AI Makerspace|
| `BATCH_SIZE` | 256 | Training/evaluation batch size |
| `EPOCHS` | 80 | Total training epochs |
| `LR` | 0.1 | Initial learning rate (decays ×0.1 every 30 epochs) |
| `MOMENTUM` | 0.9 | SGD momentum |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `WORKERS` | 4 | DataLoader worker threads |
| `RESUME` | `''` | Path to checkpoint to resume from (if empty, start from scratch) |
| `EVALUATE_ONLY` | `False` | Skip training and run evaluation only |
| `DEBUG` | `True` | Disable disk logging and checkpoint saving |
| `SAVE_DIR` | `'CNN_iter'` | Subdirectory name for checkpoints and logs |

---

## Usage

1. **Set `DATA_DIR`** in the Configuration cell to your local CURE-TSR path. No adjustment is needed if running on the Georgia Tech AI Makerspace.
2. **Run all cells** sequentially. The notebook is self-contained.
3. To **evaluate a pre-trained model**, set `RESUME` to your checkpoint path and `EVALUATE_ONLY = True`.
4. To **save checkpoints and logs**, set `DEBUG = False`.

Training outputs reliability diagram images:
- `reliability_before.png` — calibration on real test set before temperature scaling
- `reliability_comparison.png` — before/after comparison on real test set
- `unreal_reliability_before.png` — calibration on Unreal test set

---

## Project Structure

The notebook merges four logical modules from the CURE-TSR GitHub linked above:

| Module | Notebook Section | Description |
|---|---|---|
| `utils.py` | §3, §7 | Dataset class, image loaders, normalization transforms |
| `models.py` | §4, §8 | CNN (`Net`) and baseline (`SoftmaxClassifier`) definitions |
| `logger.py` | §5 | TensorBoard logger (no-op fallback if TF unavailable) |
| `train.py` | §6, §10, §11 | Training loop, evaluation, checkpointing, LR scheduling |

Calibration code (§ after §11) is standalone and runs after training completes.

---

## Model Architecture

**`Net`** — a small CNN with 2 convolutional layers and 3 fully connected layers:

```
Input (3×28×28)
→ Conv(3→6, k=5) + ReLU + MaxPool(2)
→ Conv(6→16, k=5) + ReLU + MaxPool(2)
→ FC(256→120) + ReLU
→ FC(120→84) + ReLU
→ FC(84→14)
```

**`SoftmaxClassifier`** — a linear baseline (flattened 28×28×3 → 14 classes).

Preprocessing: images are resized to 28×28, converted to tensors, L2-normalized, and per-channel standardized.

---

## Calibration

After training, the notebook measures and corrects **confidence calibration**:

1. **Reliability diagrams** plot model accuracy vs. confidence.
2. **ECE (Expected Calibration Error)** quantifies the gap between ideal reliability and measured reliability.
3. **Temperature scaling** fits a single scalar `T` (via L-BFGS on NLL) to rescale logits — improving calibration without retraining.

Calibration is evaluated separately on:
- The **real** test split (`Real_Test/ChallengeFree`)
- The **synthetic** test split (`Unreal_Test`) using the temperature learned from the real split

---

## Citation

If you use CURE-TSR, please cite the original dataset paper:

```bibtex
@article{temel2019cure,
  title     = {CURE-TSR: Challenging unreal and real environments for traffic sign recognition},
  author    = {Temel, Dogancan and Alshawi, Tariq and Chen, Min-Hung and AlRegib, Ghassan},
  journal   = {arXiv preprint arXiv:1712.02463},
  year      = {2019}
}
```

