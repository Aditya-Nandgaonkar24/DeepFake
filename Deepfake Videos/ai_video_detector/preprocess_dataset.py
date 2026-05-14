import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.audio import extract_audio_features_from_video
from utils.preprocessing import FaceExtractor
from config import DATASET_RAW, DATASET_PREPROCESSED, SEQUENCE_LENGTH, AUDIO_N_MELS


def _existing_outputs_ready(out_folder):
    if not os.path.exists(out_folder):
        return False
    has_faces = any(name.startswith("face_") and name.endswith(".jpg") for name in os.listdir(out_folder))
    has_mouths = any(name.startswith("mouth_") and name.endswith(".jpg") for name in os.listdir(out_folder))
    has_audio = os.path.exists(os.path.join(out_folder, "audio_features.npy"))
    has_metadata = os.path.exists(os.path.join(out_folder, "metadata.json"))
    return has_faces and has_mouths and has_audio and has_metadata


def main():
    source_dir = DATASET_RAW
    target_dir = DATASET_PREPROCESSED

    print("[INFO] Initializing Offline Face Extraction Pipeline v2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FaceExtractor(device=device)
    print(f"[INFO] Face Extractor running on: {device}")

    skipped_videos = []
    processed_count = 0

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            s_dir = os.path.join(source_dir, split, cls)
            if not os.path.exists(s_dir):
                continue

            t_dir = os.path.join(target_dir, split, cls)
            os.makedirs(t_dir, exist_ok=True)

            videos = sorted(f for f in os.listdir(s_dir) if f.endswith(".mp4"))
            if not videos:
                continue

            print(f"\n---> Processing Folder [{split}/{cls}] | Total Videos: {len(videos)}")

            for video_name in tqdm(videos):
                v_path = os.path.join(s_dir, video_name)
                v_id = video_name.rsplit(".", 1)[0]
                out_folder = os.path.join(t_dir, v_id)
                os.makedirs(out_folder, exist_ok=True)

                if _existing_outputs_ready(out_folder):
                    continue

                records = extractor.extract_face_records(v_path, num_frames=SEQUENCE_LENGTH)
                if not records:
                    skipped_videos.append(
                        {
                            "split": split,
                            "class": cls,
                            "video": video_name,
                            "reason": "no_valid_faces",
                        }
                    )
                    continue

                for old_name in os.listdir(out_folder):
                    if old_name.startswith(("face_", "mouth_")) and old_name.endswith(".jpg"):
                        try:
                            os.remove(os.path.join(out_folder, old_name))
                        except OSError:
                            pass

                for idx, record in enumerate(records):
                    Image.fromarray(record["face"]).save(os.path.join(out_folder, f"face_{idx:03d}.jpg"))
                    Image.fromarray(record["mouth"]).save(os.path.join(out_folder, f"mouth_{idx:03d}.jpg"))

                audio_features = extract_audio_features_from_video(
                    v_path,
                    sequence_length=SEQUENCE_LENGTH,
                    n_mels=AUDIO_N_MELS,
                )
                np.save(os.path.join(out_folder, "audio_features.npy"), audio_features)

                metadata = {
                    "video_name": video_name,
                    "split": split,
                    "class": cls,
                    "sequence_length_target": SEQUENCE_LENGTH,
                    "selected_frames": [
                        {
                            "frame_index": int(record["frame_index"]),
                            "detection_confidence": record["detection_confidence"],
                            "blur_score": record["blur_score"],
                            "face_area_ratio": record["face_area_ratio"],
                            "quality_score": record["quality_score"],
                            "face_box": record["face_box"],
                            "mouth_box": record["mouth_box"],
                        }
                        for record in records
                    ],
                    "num_selected_frames": len(records),
                    "mean_detection_confidence": round(float(np.mean([r["detection_confidence"] for r in records])), 4),
                    "mean_blur_score": round(float(np.mean([r["blur_score"] for r in records])), 4),
                    "mean_face_area_ratio": round(float(np.mean([r["face_area_ratio"] for r in records])), 4),
                    "mean_quality_score": round(float(np.mean([r["quality_score"] for r in records])), 4),
                    "frame_coverage_ratio": round(len(records) / SEQUENCE_LENGTH, 4),
                }

                with open(os.path.join(out_folder, "metadata.json"), "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

                processed_count += 1

    if skipped_videos:
        log_path = os.path.join(target_dir, "skipped_videos.json")
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(skipped_videos, handle, indent=2)
        print(f"\n[WARNING] {len(skipped_videos)} videos had no usable faces. See: {log_path}")

    print(f"\n[SUCCESS] Dataset extraction complete! Newly processed videos: {processed_count}")


if __name__ == "__main__":
    main()
