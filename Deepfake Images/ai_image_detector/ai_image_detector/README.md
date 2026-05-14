# AI Image Detector

A multi-layer AI image detection system that determines whether an image is **real** or **AI-generated**. Combines deep learning with physics-based signal analysis for robust detection.

## Features

### 6-Analyzer Ensemble System

| # | Analyzer | Method | Weight |
|---|----------|--------|--------|
| 1 | **Deep Learning** | EfficientNet-B0 (trained on 140K faces) | 70% |
| 2 | **Frequency Analysis** | 2D FFT spectrum analysis | 6% |
| 3 | **Noise Pattern** | Camera noise vs synthetic noise detection | 6% |
| 4 | **Pixel Statistics** | Color distribution & histogram analysis | 6% |
| 5 | **Metadata (EXIF/ICC)** | Camera tags, ICC profiles, AI software | 6% |
| 6 | **Error Level Analysis** | JPEG re-compression artifact detection | 6% |

### Frontend Visualizations (Chart.js)
- Animated score bars for all 6 analyzers
- Bar chart of individual analyzer scores
- Pie chart of classification probabilities
- Radar chart of component analysis

### Security
- Rate limiting (5 req/min)
- Decompression bomb protection
- In-memory processing
- UUID temp filenames

---

## Requirements

- **Python 3.12** (with CUDA PyTorch for GPU)
- **NVIDIA GPU** recommended (RTX 3050 or better)
- PyTorch 2.5.1+cu121
- Flask 3.0

## Installation

### 1. Install Dependencies

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download Dataset

Download the [140K Real & Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset from Kaggle and extract to `datasets/faces/`.

Or use the Kaggle API:
```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\<you>\.kaggle"
py -3.12 -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); api.dataset_download_files('xhlulu/140k-real-and-fake-faces', path='datasets/faces', unzip=True)"
```

### 3. Train the Model

```powershell
$env:PYTHONPATH = "."
py -3.12 core/ml/train_faces.py
```

Training on RTX 3050: ~20-30 minutes for 3 epochs.

Auto-generates evaluation plots in `models/plots/`:
- Accuracy & Loss curves
- Confusion Matrix
- ROC Curve with AUC score
- Dataset Distribution chart

### 4. Run the Server

```powershell
$env:PYTHONPATH = "."
py -3.12 run.py
```

Server starts at `http://localhost:5000`

### 5. Open Frontend

Open `frontend/index.html` in your browser. Drag & drop or click to upload an image.

---

## How It Works

### Analysis Pipeline

```
Image Upload
    |
    v
[Parallel Execution via ThreadPoolExecutor]
    |
    +-- Frequency Analyzer (FFT spectrum)
    +-- Noise Analyzer (high-pass filtering)
    +-- Pixel Analyzer (color statistics)
    +-- Metadata Analyzer (EXIF/ICC)
    +-- ELA Analyzer (JPEG re-compression)
    +-- Deep Learning Model (EfficientNet-B0)
    |
    v
Adaptive Weighted Score Combination
    |
    v
Final Classification:
    >= 70: Real Image (High Confidence)
    50-69: Uncertain (Medium Confidence)
    < 50:  AI-Generated (High Confidence)
```

### Adaptive Weighting

The model weight adjusts based on image resolution to handle domain shift between training data (256×256 faces) and arbitrary input images. Physics-based analyzers (FFT, noise, pixel, metadata, ELA) work at any resolution.

---

## Project Structure

```
ai_image_detector/
├── api/
│   └── routes.py                 # Flask API + rate limiting + ensemble
├── core/
│   ├── config.py                 # Configuration
│   ├── analyzers/
│   │   ├── ela_analyzer.py       # Error Level Analysis
│   │   ├── frequency_analyzer.py # FFT frequency analysis
│   │   ├── noise_analyzer.py     # Noise pattern analysis
│   │   ├── metadata_analyzer.py  # EXIF/ICC metadata
│   │   └── pixel_analyzer.py     # Pixel statistics
│   └── ml/
│       ├── train_faces.py        # Training (140K Faces)
│       └── train_model.py        # Training (CIFAKE, legacy)
├── frontend/
│   └── index.html                # Web UI + Chart.js
├── models/
│   ├── best_model.pth            # Trained model
│   └── plots/                    # Evaluation plots
├── datasets/
│   └── faces/                    # 140K Real & Fake Faces
├── run.py                        # Entry point
├── Dockerfile                    # Production container
├── ARCHITECTURE.txt              # Architecture diagram
├── PROJECT_OVERVIEW.md           # Project overview
└── requirements.txt              # Dependencies
```

---

## API Reference

### POST /api/analyze

Upload an image for analysis.

**Request**: `multipart/form-data` with `image` field

**Response**:
```json
{
  "success": true,
  "image_info": { "filename": "...", "format": "JPEG", "width": 1024, "height": 768 },
  "analysis": {
    "final_result": {
      "final_score": 78.5,
      "classification": "Real Image",
      "confidence_level": "High",
      "component_scores": {
        "model": 85.2, "frequency": 72.1, "noise": 68.4,
        "pixel": 74.3, "metadata": 80.0, "ela": 65.7
      }
    },
    "model_prediction": { "class": "Real", "confidence": 85.2, "probabilities": {} },
    "frequency_analysis": { "...": "..." },
    "noise_analysis": { "...": "..." },
    "pixel_analysis": { "...": "..." },
    "metadata_analysis": { "...": "..." },
    "ela_analysis": { "...": "..." }
  }
}
```

### GET /api/health

Health check endpoint.

---

## Training Details

### Dataset
- **Name**: 140K Real & Fake Faces
- **Source**: [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Size**: ~1.5 GB (zip), ~3.7 GB extracted
- **Resolution**: 256×256 pixels
- **Classes**: Real (0), Fake (1)
- **Split**: 100K train / 20K valid / 20K test

### Model
- **Architecture**: EfficientNet-B0 with custom classifier head
- **Parameters**: ~5.3M
- **Input**: 256×256 RGB
- **Output**: 2 classes (Real, Fake)
- **Training**: Adam optimizer, CrossEntropy loss, Mixed Precision (AMP)

---

## Deployment

### Docker
```bash
docker build -t ai-detector .
docker run -p 5000:5000 ai-detector
```

Uses Gunicorn with 4 workers in production.

---

## Limitations

1. **Face-focused model**: Trained on face images; may be less accurate on non-face images (landscapes, objects). Physics-based analyzers still work for these.
2. **Evolving AI**: New generators may produce images the model hasn't seen.
3. **Compression artifacts**: Heavy JPEG compression can confuse the ELA analyzer.
4. **Metadata stripping**: Social media often removes EXIF data (handled with neutral scoring).

---

*Built with PyTorch, Flask, EfficientNet, and Chart.js*
