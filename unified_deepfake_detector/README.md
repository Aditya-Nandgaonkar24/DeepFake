# Unified Deepfake Detector

This project combines the original image and video deepfake detectors into one runnable app.

## What Was Merged

- `image_service/` contains the Flask image analyzer, image feature analyzers, and image model weights.
- `video_service/` contains the video CLI/training code, ONNX inference utilities, model definitions, and ONNX model.
- `app.py` is the single FastAPI entry point.
- `static/` contains the combined browser UI.

Large raw datasets, preprocessed frame folders, virtual environments, git internals, and generated logs were intentionally left out of the merged project.

## Run

```powershell
cd C:\Users\adity\Desktop\practice\Deepfake\unified_deepfake_detector
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## API

- `GET /api/health` checks the unified host and model availability.
- `POST /image-service/api/analyze` accepts an `image` multipart upload.
- `POST /api/video/analyze` accepts a `video` multipart upload.

## Notes

- The image service still uses its original Flask internals and is mounted under `/image-service`.
- The video analyzer loads `video_service/detector_model.onnx` lazily on the first video request.
- If you have an NVIDIA GPU, you can replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`.
