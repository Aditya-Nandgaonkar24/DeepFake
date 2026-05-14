"""
Quick fine-tune: add real/fake videos to the existing trained model.

Usage:
    python finetune_quick.py --real v1.mp4 --real v2.mp4 --fake f1.mp4 --fake f2.mp4
    python finetune_quick.py --rollback

What it does:
  1. Backs up existing best_model.pth and detector_model.onnx
  2. Loads the existing best_model.pth weights
  3. Extracts faces from all videos
  4. Fine-tunes with a very low LR (1e-5)
  5. Saves updated weights and re-exports ONNX
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.fusion import SpatialTemporalFusion
from utils.preprocessing import FaceExtractor
from utils.audio import extract_audio_features_from_video
from config import (
    SEQUENCE_LENGTH, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD,
    AUDIO_N_MELS, USE_AUDIO_BRANCH, ONNX_MODEL_NAME,
)


def extract_and_prepare(video_path, extractor, label):
    """Extract faces from video and prepare as training tensor."""
    faces = extractor.extract_faces_from_video(video_path, num_frames=SEQUENCE_LENGTH)
    if not faces:
        print(f"  [ERROR] No faces found in {video_path}")
        return None

    transform = A.ReplayCompose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2()
    ])

    frames_tensor = torch.zeros((SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE))
    mask = torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool)

    saved_replay = None
    for i, face in enumerate(faces[:SEQUENCE_LENGTH]):
        img_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if saved_replay is None:
            result = transform(image=img_rgb)
            saved_replay = result['replay']
            frames_tensor[i] = result['image']
        else:
            result = A.ReplayCompose.replay(saved_replay, image=img_rgb)
            frames_tensor[i] = result['image']
        mask[i] = True

    audio_features = extract_audio_features_from_video(
        video_path, sequence_length=SEQUENCE_LENGTH, n_mels=AUDIO_N_MELS
    )
    audio_tensor = torch.zeros((SEQUENCE_LENGTH, AUDIO_N_MELS), dtype=torch.float32)
    if audio_features is not None and audio_features.ndim == 2:
        length = min(audio_features.shape[0], SEQUENCE_LENGTH)
        feature_dim = min(audio_features.shape[1], AUDIO_N_MELS)
        audio_tensor[:length, :feature_dim] = torch.from_numpy(
            audio_features[:length, :feature_dim].astype(np.float32)
        )

    label_tensor = torch.tensor(label, dtype=torch.float32)
    return frames_tensor, label_tensor, mask, audio_tensor


def backup_originals(base_dir):
    """Back up original weights and ONNX model before fine-tuning."""
    import shutil
    backup_dir = os.path.join(base_dir, "backup_before_finetune")
    os.makedirs(backup_dir, exist_ok=True)

    files_to_backup = [
        (os.path.join(base_dir, "logs", "best_model.pth"), "best_model.pth"),
        (os.path.join(base_dir, "detector_model.onnx"), "detector_model.onnx"),
    ]

    for src, name in files_to_backup:
        dst = os.path.join(backup_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [BACKUP] {name} -> {backup_dir}")
        else:
            print(f"  [SKIP]   {name} not found, nothing to backup")

    print(f"[OK] Backups saved to: {backup_dir}\n")
    return backup_dir


def rollback(base_dir):
    """Restore original weights and ONNX from backup."""
    import shutil
    backup_dir = os.path.join(base_dir, "backup_before_finetune")
    if not os.path.exists(backup_dir):
        print("[ERROR] No backup found! Cannot rollback.")
        return False

    restores = [
        ("best_model.pth", os.path.join(base_dir, "logs", "best_model.pth")),
        ("detector_model.onnx", os.path.join(base_dir, "detector_model.onnx")),
    ]

    for name, dst in restores:
        src = os.path.join(backup_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [RESTORED] {name}")
        else:
            print(f"  [SKIP]     {name} not in backup")

    print("\n[DONE] Rollback complete! Restart the server to use the original model.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick fine-tune with real/fake videos")
    parser.add_argument("--real", action="append", default=[], help="Path to a REAL video (can specify multiple)")
    parser.add_argument("--fake", action="append", default=[], help="Path to a FAKE video (can specify multiple)")
    parser.add_argument("--weights", default="logs/best_model.pth",
                        help="Existing model weights to fine-tune from")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (very low to avoid catastrophic forgetting)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of gradient steps")
    parser.add_argument("--rollback", action="store_true",
                        help="Undo fine-tuning: restore original model from backup")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── ROLLBACK MODE ──
    if args.rollback:
        print("[INFO] Rolling back to original model...")
        rollback(base_dir)
        return

    # ── FINE-TUNE MODE ──
    if not args.real or not args.fake:
        parser.error("Need at least one --real and one --fake video for fine-tuning")

    weights_path = args.weights if os.path.isabs(args.weights) else os.path.join(base_dir, args.weights)

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found at {weights_path}")
        sys.exit(1)

    # Back up originals first
    print("[INFO] Backing up original model files...")
    backup_originals(base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load model
    print(f"[INFO] Loading model from {weights_path}")
    model = SpatialTemporalFusion(
        seq_length=SEQUENCE_LENGTH,
        freeze_spatial=False,
        use_audio=USE_AUDIO_BRANCH,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("[OK] Model loaded\n")

    # 2. Extract faces from all videos
    print("[INFO] Extracting faces from videos...")
    extractor = FaceExtractor(device=str(device))

    samples = []
    for path in args.real:
        print(f"  Processing REAL: {os.path.basename(path)}")
        sample = extract_and_prepare(path, extractor, label=0.0)
        if sample:
            samples.append(sample)

    for path in args.fake:
        print(f"  Processing FAKE: {os.path.basename(path)}")
        sample = extract_and_prepare(path, extractor, label=1.0)
        if sample:
            samples.append(sample)

    if len(samples) < 2:
        print("[ERROR] Need at least 2 valid samples. Aborting.")
        print("[INFO] Run with --rollback to restore original model.")
        sys.exit(1)

    num_real = sum(1 for s in samples if s[1].item() == 0.0)
    num_fake = sum(1 for s in samples if s[1].item() == 1.0)
    print(f"\n[INFO] Ready: {num_real} real + {num_fake} fake = {len(samples)} total samples")

    # Stack into batch
    frames_batch = torch.stack([s[0] for s in samples]).to(device)
    labels_batch = torch.stack([s[1] for s in samples]).to(device)
    masks_batch = torch.stack([s[2] for s in samples]).to(device)
    audio_batch = torch.stack([s[3] for s in samples]).to(device)

    # 3. Fine-tune
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    print(f"[INFO] Fine-tuning for {args.steps} steps (lr={args.lr})...\n")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        with torch.amp.autocast(str(device)):
            logits = model(frames_batch, mask=masks_batch, audio_features=audio_batch)
            loss = criterion(logits, labels_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            probs = torch.sigmoid(logits).detach()
            prob_strs = "  ".join(
                f"{'R' if labels_batch[i].item() == 0 else 'F'}:{probs[i].item():.4f}"
                for i in range(len(samples))
            )
            print(f"  Step {step:3d} | loss={loss.item():.4f} | {prob_strs}")

    # Final check
    model.eval()
    with torch.no_grad():
        logits = model(frames_batch, mask=masks_batch, audio_features=audio_batch)
        probs = torch.sigmoid(logits)
        print(f"\n{'='*60}")
        print(f"[RESULT] After fine-tuning:")
        for i in range(len(samples)):
            kind = "Real" if labels_batch[i].item() == 0 else "Fake"
            p = probs[i].item()
            verdict = "FAKE" if p >= 0.33 else "REAL"
            print(f"  {kind} video -> fake_prob = {p:.4f}  ({verdict})")
        print(f"{'='*60}")

    # 4. Save weights
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Weights saved to {save_path}")

    # 5. Re-export ONNX
    print("[INFO] Re-exporting ONNX model...")
    from export_onnx import export_to_onnx
    export_to_onnx(model_path=save_path, out_path=ONNX_MODEL_NAME, use_audio=USE_AUDIO_BRANCH)
    print("\n[DONE] Fine-tuning complete! Restart the server to use the updated model.")
    print("[TIP]  To undo: python finetune_quick.py --rollback")


if __name__ == "__main__":
    main()
