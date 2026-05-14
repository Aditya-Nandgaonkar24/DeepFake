import csv
import json
import os

import onnxruntime as ort
import torch

from config import (
    INFERENCE_THRESHOLD,
    INFERENCE_UNCERTAINTY_MARGIN,
    ONNX_MODEL_NAME,
    LOG_DIR,
)
from utils.inference import analyze_video_with_onnx
from utils.preprocessing import FaceExtractor


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def list_video_files(path):
    if os.path.isfile(path):
        return [path]

    collected = []
    for root, _, files in os.walk(path):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in VIDEO_EXTENSIONS:
                collected.append(os.path.join(root, file_name))
    return sorted(collected)


def analyze_path(
    path,
    model_path=ONNX_MODEL_NAME,
    output_path=None,
    threshold=INFERENCE_THRESHOLD,
    uncertainty_margin=INFERENCE_UNCERTAINTY_MARGIN,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at '{model_path}'. Run export first.")

    os.makedirs(LOG_DIR, exist_ok=True)
    video_files = list_video_files(path)
    if not video_files:
        raise FileNotFoundError(f"No video files found under '{path}'.")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    extractor = FaceExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for video_path in video_files:
        result = analyze_video_with_onnx(
            video_path,
            extractor,
            session,
            threshold=threshold,
            uncertainty_margin=uncertainty_margin,
        )
        results.append(result)
        label = result.get("prediction", "ERROR")
        confidence = result.get("confidence")
        if confidence is None:
            print(f"[FAIL] {os.path.basename(video_path)} -> {result.get('error', 'unknown error')}")
        else:
            print(f"[{label}] {os.path.basename(video_path)} -> fake_prob={confidence:.4f}")

    if output_path is None:
        output_path = os.path.join(LOG_DIR, "analysis_report.json")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_name",
                "video_path",
                "success",
                "prediction",
                "decision_band",
                "confidence",
                "threshold",
                "uncertainty_margin",
                "distance_from_threshold",
                "num_detected_faces",
                "frame_coverage_ratio",
                "warnings",
                "error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "video_name": row.get("video_name"),
                    "video_path": row.get("video_path"),
                    "success": row.get("success"),
                    "prediction": row.get("prediction"),
                    "decision_band": row.get("decision_band"),
                    "confidence": row.get("confidence"),
                    "threshold": row.get("threshold"),
                    "uncertainty_margin": row.get("uncertainty_margin"),
                    "distance_from_threshold": row.get("distance_from_threshold"),
                    "num_detected_faces": row.get("num_detected_faces"),
                    "frame_coverage_ratio": row.get("frame_coverage_ratio"),
                    "warnings": ",".join(row.get("warnings", [])),
                    "error": row.get("error"),
                }
            )

    summary = {
        "total_videos": len(results),
        "successful": sum(1 for row in results if row.get("success")),
        "real": sum(1 for row in results if row.get("prediction") == "REAL"),
        "fake": sum(1 for row in results if row.get("prediction") == "FAKE"),
        "uncertain": sum(1 for row in results if row.get("prediction") == "UNCERTAIN"),
        "errors": sum(1 for row in results if not row.get("success")),
        "json_report": output_path,
        "csv_report": csv_path,
    }
    print(json.dumps(summary, indent=2))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to a video file or directory of videos")
    parser.add_argument("--model", type=str, default=ONNX_MODEL_NAME)
    parser.add_argument("--output", type=str, default=os.path.join(LOG_DIR, "analysis_report.json"))
    parser.add_argument("--threshold", type=float, default=INFERENCE_THRESHOLD)
    parser.add_argument("--uncertainty_margin", type=float, default=INFERENCE_UNCERTAINTY_MARGIN)
    args = parser.parse_args()

    analyze_path(
        args.path,
        model_path=args.model,
        output_path=args.output,
        threshold=args.threshold,
        uncertainty_margin=args.uncertainty_margin,
    )
