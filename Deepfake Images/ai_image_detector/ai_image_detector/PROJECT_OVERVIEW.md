# AI Image Detector - Project Overview

## What Is This?

A production-ready AI image detection system that determines whether an image is real (camera-captured) or AI-generated (deepfake, GAN, diffusion model output). It uses an **ensemble of 6 independent analyzers** to make robust predictions.

---

## Architecture

### 6-Analyzer Ensemble

| Analyzer | Method | What It Detects |
|----------|--------|-----------------|
| **Deep Learning (70%)** | EfficientNet-B0 CNN | Semantic features learned from 140K face images |
| **Frequency / FFT (6%)** | 2D Fourier Transform | Spectral artifacts in AI-generated images |
| **Noise Pattern (6%)** | High-pass filtering | Difference between camera sensor noise vs synthetic noise |
| **Pixel Statistics (6%)** | Color histograms | Unnatural saturation, contrast, brightness distributions |
| **Metadata / EXIF (6%)** | EXIF + ICC parsing | Missing camera data, AI software tags, stripped profiles |
| **ELA (6%)** | JPEG re-compression | Error level uniformity indicating manipulation |

### Score Combination
- Model 70% + 5 analyzers at 6% each
- Final score: 0-100 (≥70 = Real, 50-69 = Uncertain, <50 = AI-Generated)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, Flask 3.0, Flask-Limiter |
| **ML** | PyTorch 2.5.1+cu121 (CUDA GPU), EfficientNet-B0 |
| **Image Processing** | OpenCV, Pillow, NumPy, SciPy |
| **Evaluation** | Matplotlib, scikit-learn (ROC, confusion matrix) |
| **Frontend** | HTML5, CSS3, Vanilla JS, Chart.js 4.4 |
| **Deployment** | Docker, Gunicorn |

---

## Project Structure

```
ai_image_detector/
├── api/routes.py                  # Flask API + rate limiting + ensemble
├── core/
│   ├── config.py                  # Centralized configuration
│   ├── analyzers/                 # 5 physics-based analyzers
│   │   ├── ela_analyzer.py
│   │   ├── frequency_analyzer.py
│   │   ├── noise_analyzer.py
│   │   ├── metadata_analyzer.py
│   │   └── pixel_analyzer.py
│   └── ml/                        # Machine learning
│       ├── train_faces.py         # Training (140K Faces dataset)
│       └── train_model.py         # Training (CIFAKE dataset)
├── frontend/index.html            # Web UI with Chart.js charts
├── models/
│   ├── best_model.pth             # Trained model checkpoint
│   └── plots/                     # Auto-generated evaluation plots
├── datasets/faces/                # 140K Real & Fake Faces (256x256)
├── run.py                         # Application entry point
├── Dockerfile                     # Production container
└── requirements.txt               # Dependencies
```

---

## Quick Start

```powershell
# From: ai_image_detector/ai_image_detector/
$env:PYTHONPATH = "."

# 1. Train model (GPU recommended, ~20-30 min)
py -3.12 core/ml/train_faces.py

# 2. Start server
py -3.12 run.py

# 3. Open frontend/index.html in browser
```

---

## Frontend Features

After analyzing an image, the UI displays:
- **Final Score** with classification (Real / AI-Generated / Uncertain)
- **Animated Score Bars** for all 6 analyzers
- **Bar Chart** — Individual analyzer score comparison
- **Pie Chart** — Classification probability breakdown
- **Radar Chart** — Component analysis spider web
- **Detailed Metrics** — Per-analyzer technical breakdown

---

## Training

**Dataset**: 140K Real & Fake Faces (256×256, Kaggle)
- 100K training images (50K real + 50K fake)
- 20K validation, 20K test
- Binary classification: Real vs Fake

**Auto-Generated Evaluation Plots**:
- Accuracy & Loss curves
- Confusion Matrix
- ROC Curve with AUC
- Dataset Distribution pie chart

---

## Security

- Rate limiting (5 req/min on /api/analyze)
- Decompression bomb protection (100M pixel limit)
- In-memory processing (minimal disk I/O)
- UUID temp filenames
- Input validation (file type + decode check)

---

## Key Design Decisions

1. **Ensemble over single model** — Physics-based analyzers provide robustness when the ML model is uncertain
2. **Adaptive weighting** — Model weight scales with image resolution to handle domain shift
3. **Binary classification** — Real/Fake is more practical and accurate than multi-class generator attribution
4. **EfficientNet-B0** — Best accuracy/speed tradeoff for the RTX 3050 GPU
5. **Chart.js visualizations** — Intuitive, interactive result presentation

---

*Built with PyTorch, Flask, and Chart.js*
