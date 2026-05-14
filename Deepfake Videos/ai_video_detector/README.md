# Deepfake Video Detection System

A dual-branch deep learning system that detects AI-generated video manipulation by analyzing both spatial (per-frame) and temporal (across-frame) forensic cues.

## How It Works

The system processes a video through 4 stages:

1. **Face Extraction** — MTCNN detects and crops the largest face from 15 equidistant frames
2. **Spatial Analysis** — A MobileNetV3-Large backbone with CBAM attention extracts 960-dimensional feature vectors from each face crop
3. **Temporal Analysis** — Feature-level difference maps (frame N minus frame N-1) highlight micro-flickering. These are concatenated with spatial features (1920 dims total) and fed to a bidirectional GRU with packed sequences (padding-aware)
4. **Classification** — A dropout-regularized MLP produces a binary real/fake probability

## Architecture

```
Input Video (15 frames)
    │
    ├─→ MTCNN Face Extraction (offline preprocessing)
    │
    ├─→ MobileNetV3 + CBAM Attention  →  960-dim spatial features
    │
    ├─→ Feature Difference Maps        →  960-dim temporal features
    │
    ├─→ Concatenate (1920-dim)  →  Bi-GRU (packed, padding-aware)
    │
    └─→ MLP Classifier  →  P(fake)
```

## Project Structure

```
ai_video_detector/
├── main.py                 # Unified CLI: prepare | train | test | export | serve
├── config.py               # Centralized constants (image size, sequence length, etc.)
├── preprocess_dataset.py   # Offline MTCNN face extraction → .jpg sequences
├── train.py                # Training loop with AMP, gradient clipping, F1 tracking
├── test.py                 # Evaluation with ROC/PR curves, threshold tuning
├── export_onnx.py          # ONNX export for production inference
├── models/
│   ├── spatial_model.py    # MobileNetV3 + CBAM spatial feature extractor
│   ├── temporal_model.py   # Bi-directional GRU with pack_padded_sequence
│   └── fusion.py           # Spatial-temporal fusion with frame-validity masking
├── utils/
│   ├── dataset.py          # PyTorch Dataset with ReplayCompose augmentations
│   └── preprocessing.py    # MTCNN face extraction utility
└── web/
    ├── app.py              # FastAPI backend with ONNX inference
    └── static/             # Glassmorphic frontend (HTML/CSS/JS)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Preprocess dataset (extract faces — run once, takes a few hours)
python main.py prepare

# 2. Train the model
python main.py train --epochs 30

# 3. Evaluate on test set (generates ROC/PR curves)
python main.py test

# 4. Export to ONNX for production
python main.py export

# 5. Launch web interface
python main.py serve
```

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| Offline face extraction | Eliminates MTCNN bottleneck during training (hours → minutes) |
| Feature-level difference maps | Zero-cost alternative to optical flow for temporal cues |
| Frame-validity mask | GRU ignores zero-padded frames via `pack_padded_sequence` |
| Dynamic `pos_weight` | Automatically handles class imbalance from actual data counts |
| Mixed precision (AMP) | Halves VRAM usage, enables training on consumer GPUs |
| Gradient accumulation | Simulates batch_size=16 on GPUs that only fit batch_size=4 |
| ONNX export | 2-3x faster inference than PyTorch, deployable anywhere |

## Tech Stack

- **Training**: PyTorch 2.x, Albumentations, MTCNN (facenet-pytorch)
- **Inference**: ONNX Runtime (GPU-accelerated)
- **Serving**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS with glassmorphic design

## Evaluation Outputs

After running `python main.py test`, the `logs/` directory will contain:
- `test_metrics.json` — Structured metrics with optimal threshold
- `confusion_matrix.png` — Visual confusion matrix
- `roc_curve.png` — ROC curve with AUC score
- `pr_curve.png` — Precision-Recall curve
- `training_evaluation_plots.png` — Loss, accuracy, and F1 over epochs

## Dataset

The system expects the FaceForensics++ dataset (or similar) organized as:
```
dataset/
├── train/
│   ├── real/    # .mp4 files
│   └── fake/    # .mp4 files
├── val/
└── test/
```

Run `python main.py prepare` to convert this into extracted face sequences under `dataset_preprocessed/`.
