import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.wsgi import WSGIMiddleware


BASE_DIR = Path(__file__).resolve().parent
IMAGE_SERVICE_DIR = BASE_DIR / "image_service"
VIDEO_SERVICE_DIR = BASE_DIR / "video_service"
STATIC_DIR = BASE_DIR / "static"

# The original projects use top-level imports such as `from core...` and
# `from config...`, so their service folders must be import roots.
for service_dir in (VIDEO_SERVICE_DIR, IMAGE_SERVICE_DIR):
    service_path = str(service_dir)
    if service_path not in sys.path:
        sys.path.insert(0, service_path)


from image_service.run import app as image_flask_app  # noqa: E402
from config import (  # noqa: E402
    INFERENCE_THRESHOLD,
    INFERENCE_UNCERTAINTY_MARGIN,
    MAX_UPLOAD_SIZE_MB,
    SEQUENCE_LENGTH,
    USE_AUDIO_BRANCH,
)


app = FastAPI(title="Unified Deepfake Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/image-service", WSGIMiddleware(image_flask_app))


_video_model = None
_video_extractor = None
_video_device = None
_video_init_error = None


def _load_video_runtime():
    global _video_model, _video_extractor, _video_device, _video_init_error
    if _video_model is not None and _video_extractor is not None:
        return _video_model, _video_extractor, _video_device
    if _video_init_error is not None:
        raise RuntimeError(_video_init_error)

    weights_path = VIDEO_SERVICE_DIR / "logs" / "best_model.pth"
    if not weights_path.exists():
        _video_init_error = f"PyTorch weights not found at {weights_path}"
        raise RuntimeError(_video_init_error)

    try:
        import torch
        from models.fusion import SpatialTemporalFusion
        from utils.preprocessing import FaceExtractor

        _video_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[System] Loading PyTorch video model on {_video_device}...")
        _video_model = SpatialTemporalFusion(
            seq_length=SEQUENCE_LENGTH,
            use_audio=USE_AUDIO_BRANCH,
        )
        _video_model.load_state_dict(
            torch.load(str(weights_path), map_location=_video_device)
        )
        _video_model.to(_video_device)
        _video_model.eval()
        print(f"[OK] Video model loaded from {weights_path}")

        _video_extractor = FaceExtractor(device=_video_device)
        return _video_model, _video_extractor, _video_device
    except Exception as exc:
        _video_init_error = str(exc)
        raise RuntimeError(_video_init_error) from exc


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "image_service": "mounted",
        "video_model_exists": (VIDEO_SERVICE_DIR / "logs" / "best_model.pth").exists(),
        "video_model_loaded": _video_model is not None,
        "video_init_error": _video_init_error,
    }


@app.post("/api/video/analyze")
async def analyze_video(video: UploadFile = File(...)):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Upload must be a video file.")

    try:
        model, extractor, device = _load_video_runtime()
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            file_size = 0
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit.",
                    )
                temp_file.write(chunk)

        from utils.inference import analyze_video_with_pytorch

        result = analyze_video_with_pytorch(
            temp_path,
            extractor,
            model,
            device,
            threshold=INFERENCE_THRESHOLD,
            uncertainty_margin=INFERENCE_UNCERTAINTY_MARGIN,
        )
        if not result.get("success"):
            return JSONResponse(
                status_code=422,
                content={"error": result.get("error", "Video analysis failed.")},
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

