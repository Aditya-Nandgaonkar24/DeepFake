import os
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from utils.audio import extract_audio_features_from_video
from config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    SEQUENCE_LENGTH,
    AUDIO_N_MELS,
    INFERENCE_THRESHOLD,
    INFERENCE_UNCERTAINTY_MARGIN,
    TIMELINE_SEGMENT_THRESHOLD,
    TIMELINE_TOP_K_FRAMES,
)


def sigmoid(logit):
    return 1.0 / (1.0 + np.exp(-float(logit)))


def normalize_faces(faces, image_size=IMAGE_SIZE):
    normalized_frames = []
    for face in faces:
        face_resized = cv2.resize(face, (image_size, image_size))
        img_float = face_resized.astype(np.float32) / 255.0
        img_float = (img_float - list(NORMALIZE_MEAN)) / list(NORMALIZE_STD)
        normalized_frames.append(np.transpose(img_float, (2, 0, 1)))
    return normalized_frames


def build_sequence_inputs(faces, sequence_length=SEQUENCE_LENGTH):
    num_real_faces = len(faces)
    normalized_frames = normalize_faces(faces)

    while len(normalized_frames) < sequence_length:
        normalized_frames.append(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

    frame_mask = np.zeros(sequence_length, dtype=bool)
    frame_mask[:num_real_faces] = True
    batch_input = np.array(normalized_frames, dtype=np.float32)
    return batch_input, frame_mask, num_real_faces


def classify_probability(probability, threshold=INFERENCE_THRESHOLD, uncertainty_margin=INFERENCE_UNCERTAINTY_MARGIN):
    delta = float(probability) - float(threshold)
    if abs(delta) <= uncertainty_margin:
        label = "UNCERTAIN"
    elif probability >= threshold:
        label = "FAKE"
    else:
        label = "REAL"

    if delta <= -(uncertainty_margin * 2):
        band = "clear_real"
    elif delta < -uncertainty_margin:
        band = "lean_real"
    elif delta < uncertainty_margin:
        band = "borderline"
    elif delta < uncertainty_margin * 2:
        band = "lean_fake"
    else:
        band = "clear_fake"

    return {
        "prediction": label,
        "decision_band": band,
        "distance_from_threshold": round(delta, 4),
        "uncertainty_margin": float(uncertainty_margin),
    }


def summarize_timeline(
    timeline: Iterable[float],
    frame_mask,
    suspicious_threshold=TIMELINE_SEGMENT_THRESHOLD,
    top_k=TIMELINE_TOP_K_FRAMES,
):
    timeline = [float(x) for x in timeline]
    valid_indices = [idx for idx, is_valid in enumerate(frame_mask) if is_valid]
    valid_scores = [{"frame_index": idx, "score": round(timeline[idx], 4)} for idx in valid_indices]
    top_frames = sorted(valid_scores, key=lambda item: item["score"], reverse=True)[:top_k]

    suspicious_segments = []
    current_segment = None
    for idx in valid_indices:
        score = timeline[idx]
        if score >= suspicious_threshold:
            if current_segment is None:
                current_segment = {
                    "start_frame": idx,
                    "end_frame": idx,
                    "peak_score": score,
                }
            else:
                current_segment["end_frame"] = idx
                current_segment["peak_score"] = max(current_segment["peak_score"], score)
        elif current_segment is not None:
            suspicious_segments.append(current_segment)
            current_segment = None

    if current_segment is not None:
        suspicious_segments.append(current_segment)

    for segment in suspicious_segments:
        start = segment["start_frame"]
        end = segment["end_frame"]
        scores = [timeline[i] for i in range(start, end + 1)]
        segment["mean_score"] = round(float(np.mean(scores)), 4)
        segment["peak_score"] = round(float(segment["peak_score"]), 4)
        segment["length"] = end - start + 1

    return {
        "top_frames": top_frames,
        "suspicious_segments": suspicious_segments,
    }


def build_inference_warnings(num_real_faces, sequence_length, decision_band):
    warnings = []
    if num_real_faces == 0:
        warnings.append("no_faces_detected")
    elif num_real_faces < max(4, sequence_length // 3):
        warnings.append("low_face_coverage")

    if decision_band == "borderline":
        warnings.append("borderline_decision")

    return warnings


def run_onnx_sequence_inference(
    ort_session,
    frames_batch,
    frame_mask,
    audio_features=None,
):
    input_names = [inp.name for inp in ort_session.get_inputs()]
    ort_inputs = {input_names[0]: np.expand_dims(frames_batch, axis=0)}
    if len(input_names) >= 2:
        ort_inputs[input_names[1]] = np.expand_dims(frame_mask, axis=0)
    if len(input_names) >= 3:
        if audio_features is None:
            audio_features = np.zeros((SEQUENCE_LENGTH, AUDIO_N_MELS), dtype=np.float32)
        ort_inputs[input_names[2]] = np.expand_dims(audio_features, axis=0)
    logits = ort_session.run(None, ort_inputs)[0]
    return float(logits.flat[0]), ort_inputs, input_names


def analyze_video_with_onnx(
    video_path,
    extractor,
    ort_session,
    threshold=INFERENCE_THRESHOLD,
    uncertainty_margin=INFERENCE_UNCERTAINTY_MARGIN,
):
    faces = extractor.extract_faces_from_video(video_path, num_frames=SEQUENCE_LENGTH)
    if not faces:
        return {
            "success": False,
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "error": "No valid human faces detected in video.",
        }

    batch_input, frame_mask, num_real_faces = build_sequence_inputs(faces)
    audio_features = None
    input_names = [inp.name for inp in ort_session.get_inputs()]
    if len(input_names) >= 3:
        audio_features = extract_audio_features_from_video(
            video_path,
            sequence_length=SEQUENCE_LENGTH,
            n_mels=AUDIO_N_MELS,
        )

    raw_logit, ort_inputs, input_names = run_onnx_sequence_inference(
        ort_session,
        batch_input,
        frame_mask,
        audio_features=audio_features,
    )
    probability = sigmoid(raw_logit)

    per_frame_scores = []
    for idx in range(SEQUENCE_LENGTH):
        if not frame_mask[idx]:
            per_frame_scores.append(0.0)
            continue

        masked_input = batch_input.copy()
        masked_input[idx] = 0.0
        dropout_mask = frame_mask.copy()
        dropout_mask[idx] = False
        masked_logit, _, _ = run_onnx_sequence_inference(
            ort_session,
            masked_input,
            dropout_mask,
            audio_features=audio_features,
        )
        masked_prob = sigmoid(masked_logit)
        contribution = abs(probability - masked_prob)
        per_frame_scores.append(round(min(contribution * 10 + probability * 0.5, 0.95), 4))

    classification = classify_probability(
        probability,
        threshold=threshold,
        uncertainty_margin=uncertainty_margin,
    )
    timeline_summary = summarize_timeline(per_frame_scores, frame_mask)
    warnings = build_inference_warnings(num_real_faces, SEQUENCE_LENGTH, classification["decision_band"])

    return {
        "success": True,
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "prediction": classification["prediction"],
        "decision_band": classification["decision_band"],
        "confidence": probability,
        "prob_fake": probability,
        "prob_real": round(1.0 - probability, 4),
        "distance_from_threshold": classification["distance_from_threshold"],
        "threshold": float(threshold),
        "uncertainty_margin": classification["uncertainty_margin"],
        "timeline": per_frame_scores,
        "top_frames": timeline_summary["top_frames"],
        "suspicious_segments": timeline_summary["suspicious_segments"],
        "num_detected_faces": int(num_real_faces),
        "frame_coverage_ratio": round(num_real_faces / SEQUENCE_LENGTH, 4),
        "warnings": warnings,
    }
