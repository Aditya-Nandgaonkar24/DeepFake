import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

from config import (
    PREPROCESS_CANDIDATE_MULTIPLIER,
    PREPROCESS_MIN_CANDIDATE_FRAMES,
    PREPROCESS_FACE_MARGIN_RATIO,
    PREPROCESS_MOUTH_HEIGHT_RATIO,
    PREPROCESS_MIN_FACE_AREA_RATIO,
    PREPROCESS_MIN_DETECTION_CONFIDENCE,
)


class FaceExtractor:
    def __init__(self, device="cpu"):
        self.mtcnn = MTCNN(
            keep_all=True,
            select_largest=False,
            post_process=False,
            device=device,
            min_face_size=60,
        )

    @staticmethod
    def _safe_crop(frame_rgb, box):
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        return frame_rgb[y1:y2, x1:x2]

    @staticmethod
    def _expand_box(box, frame_shape, margin_ratio):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        mx = bw * margin_ratio
        my = bh * margin_ratio
        return np.array(
            [
                max(0.0, x1 - mx),
                max(0.0, y1 - my),
                min(float(w), x2 + mx),
                min(float(h), y2 + my),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _mouth_box(face_box, frame_shape, mouth_height_ratio):
        x1, y1, x2, y2 = face_box
        h, _ = frame_shape[:2]
        face_height = y2 - y1
        mouth_top = y2 - face_height * mouth_height_ratio
        return np.array(
            [
                x1,
                max(0.0, mouth_top),
                x2,
                min(float(h), y2),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _blur_score(image_rgb):
        if image_rgb.size == 0:
            return 0.0
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _quality_score(probability, blur_score, area_ratio):
        blur_term = min(1.0, np.log1p(max(blur_score, 0.0)) / 6.0)
        area_term = min(1.0, area_ratio / 0.08)
        return float(probability * 0.5 + blur_term * 0.3 + area_term * 0.2)

    def _choose_primary_face(self, boxes, probs, frame_shape):
        if boxes is None or probs is None:
            return None

        frame_area = float(frame_shape[0] * frame_shape[1])
        candidates = []
        for box, prob in zip(boxes, probs):
            if box is None or prob is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
            if prob < PREPROCESS_MIN_DETECTION_CONFIDENCE:
                continue
            if area_ratio < PREPROCESS_MIN_FACE_AREA_RATIO:
                continue
            score = float(prob) + area_ratio
            candidates.append((score, np.array([x1, y1, x2, y2], dtype=np.float32), float(prob), float(area_ratio)))

        if not candidates:
            return None

        _, box, prob, area_ratio = max(candidates, key=lambda item: item[0])
        return box, prob, area_ratio

    def _read_candidate_frames(self, video_path, sequence_length):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return []

        candidate_count = min(
            frame_count,
            max(sequence_length * PREPROCESS_CANDIDATE_MULTIPLIER, PREPROCESS_MIN_CANDIDATE_FRAMES),
        )
        indices = set(np.linspace(0, frame_count - 1, candidate_count, dtype=int))

        frames = []
        current_frame = 0
        while cap.isOpened():
            ok = cap.grab()
            if not ok:
                break
            if current_frame in indices:
                ok, frame_bgr = cap.retrieve()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(
                    {
                        "frame_index": int(current_frame),
                        "frame_rgb": frame_rgb,
                        "pil_image": Image.fromarray(frame_rgb),
                    }
                )
            current_frame += 1

        cap.release()
        return frames

    def _select_temporal_quality_records(self, records, sequence_length):
        if not records:
            return []

        records = sorted(records, key=lambda item: item["frame_index"])
        bins = np.array_split(np.arange(len(records)), min(sequence_length, len(records)))

        selected = []
        used_indices = set()
        for bin_indices in bins:
            if len(bin_indices) == 0:
                continue
            local_records = [records[int(i)] for i in bin_indices]
            best = max(local_records, key=lambda item: item["quality_score"])
            selected.append(best)
            used_indices.add(best["frame_index"])

        if len(selected) < sequence_length:
            remaining = [item for item in records if item["frame_index"] not in used_indices]
            remaining.sort(key=lambda item: item["quality_score"], reverse=True)
            selected.extend(remaining[: max(0, sequence_length - len(selected))])

        selected = sorted(selected, key=lambda item: item["frame_index"])[:sequence_length]
        return selected

    def extract_faces_from_video(self, video_path, num_frames=15):
        records = self.extract_face_records(video_path, num_frames=num_frames)
        return [record["face"] for record in records]

    def extract_face_records(self, video_path, num_frames=15):
        candidate_frames = self._read_candidate_frames(video_path, num_frames)
        if not candidate_frames:
            return []

        pil_images = [item["pil_image"] for item in candidate_frames]
        try:
            boxes_batch, probs_batch = self.mtcnn.detect(pil_images)
        except Exception:
            boxes_batch, probs_batch = [], []
            for image in pil_images:
                boxes, probs = self.mtcnn.detect(image)
                boxes_batch.append(boxes)
                probs_batch.append(probs)

        records = []
        for frame_data, boxes, probs in zip(candidate_frames, boxes_batch, probs_batch):
            chosen = self._choose_primary_face(boxes, probs, frame_data["frame_rgb"].shape)
            if chosen is None:
                continue

            box, probability, area_ratio = chosen
            face_box = self._expand_box(box, frame_data["frame_rgb"].shape, PREPROCESS_FACE_MARGIN_RATIO)
            mouth_box = self._mouth_box(face_box, frame_data["frame_rgb"].shape, PREPROCESS_MOUTH_HEIGHT_RATIO)
            face_crop = self._safe_crop(frame_data["frame_rgb"], face_box)
            mouth_crop = self._safe_crop(frame_data["frame_rgb"], mouth_box)
            if face_crop.size == 0 or mouth_crop.size == 0:
                continue

            blur_score = self._blur_score(face_crop)
            quality_score = self._quality_score(probability, blur_score, area_ratio)
            records.append(
                {
                    "frame_index": frame_data["frame_index"],
                    "face": face_crop,
                    "mouth": mouth_crop,
                    "detection_confidence": round(probability, 4),
                    "blur_score": round(blur_score, 4),
                    "face_area_ratio": round(area_ratio, 4),
                    "quality_score": round(quality_score, 4),
                    "face_box": [round(float(v), 2) for v in face_box.tolist()],
                    "mouth_box": [round(float(v), 2) for v in mouth_box.tolist()],
                }
            )

        return self._select_temporal_quality_records(records, num_frames)
