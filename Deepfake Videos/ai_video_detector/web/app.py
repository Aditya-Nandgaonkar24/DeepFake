import os
import uuid
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import sys

# Resolve all paths relative to this file, not CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_DIR)

from utils.preprocessing import FaceExtractor
from utils.inference import analyze_video_with_onnx
from config import (
    MAX_UPLOAD_SIZE_MB,
    INFERENCE_THRESHOLD,
    INFERENCE_UNCERTAINTY_MARGIN,
)

app = FastAPI(title="Deepfake UI API Server Framework")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

ONNX_MODEL_PATH = os.path.join(PROJECT_DIR, "detector_model.onnx")
ort_session = None

if os.path.exists(ONNX_MODEL_PATH):
    providers = ort.get_available_providers()
    print(f"Available ONNX Runtime Hardware Engines: {providers}")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
else:
    print(f"Server Alert! No pre-trained ONNX structure detected at {ONNX_MODEL_PATH}. Export pipeline not run yet.")

# CPU fallback: use GPU if available, otherwise gracefully use CPU
import torch
_device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = FaceExtractor(device=_device)
print(f"Face extraction device: {_device}")

# Serve index.html at root
@app.get("/")
async def serve_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    if not ort_session:
        return JSONResponse(status_code=500, content={"error": "Deepfake ONNX brain missing. Please train model and run `python export_onnx.py`."})
    
    # Inference safeguard: validate file type
    if not video.content_type or not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Upload must be a video file.")
    
    temp_video_path = None
    try:
        # UUID prevents race condition on concurrent uploads
        temp_video_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4().hex}_{video.filename}")
        
        # Stream upload to disk in chunks to avoid loading entire file into memory
        file_size = 0
        with open(temp_video_path, "wb") as buffer:
            while chunk := await video.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    buffer.close()
                    os.remove(temp_video_path)
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")
                buffer.write(chunk)
        
        result = analyze_video_with_onnx(
            temp_video_path,
            extractor,
            ort_session,
            threshold=INFERENCE_THRESHOLD,
            uncertainty_margin=INFERENCE_UNCERTAINTY_MARGIN,
        )
        if not result.get("success"):
            return {"error": result.get("error", "Inference failed.")}
        return result
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Guaranteed temp file cleanup
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
