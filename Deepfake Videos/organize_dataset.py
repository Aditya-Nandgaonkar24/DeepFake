import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = Path(r"C:\Users\adity\Desktop\practice\Deepfake\FaceForensics++_C23")
target_dir = Path(r"C:\Users\adity\Desktop\practice\Deepfake\ai_video_detector\dataset")

# Configurations
splits = ['train', 'val', 'test']
classes = ['real', 'fake']
split_ratios = [0.8, 0.1, 0.1]

# Ensure root dataset folders exist
for split in splits:
    for cls in classes:
        os.makedirs(target_dir / split / cls, exist_ok=True)

def gather_videos(base_path):
    videos = []
    if base_path.exists():
        for f in base_path.rglob("*.mp4"):
            videos.append(f)
    return videos

print("Gathering real videos...")
real_videos = gather_videos(source_dir / "original")

print("Gathering fake videos...")
fake_videos = []
fake_folders = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
for name in fake_folders:
    fake_videos.extend(gather_videos(source_dir / name))

# Shuffle randomly to break chronological/alphabetical biases
random.seed(42)
random.shuffle(real_videos)
random.shuffle(fake_videos)

print(f"Found {len(real_videos)} REAL videos and {len(fake_videos)} FAKE videos.")

def distribute_files(file_list, class_name):
    total = len(file_list)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    t_vids = file_list[:train_end]
    v_vids = file_list[train_end:val_end]
    test_vids = file_list[val_end:]
    
    print(f"-> Moving {class_name.upper()}: {len(t_vids)} train, {len(v_vids)} val, {len(test_vids)} test.")
    
    for splits_list, split_name in zip([t_vids, v_vids, test_vids], splits):
        for f in splits_list:
            new_name = f.parent.name + "_" + f.name
            dest = target_dir / split_name / class_name / new_name
            shutil.move(str(f), str(dest))

print("Executing fast-move operation into Train/Val/Test architecture...")
distribute_files(real_videos, "real")
distribute_files(fake_videos, "fake")
print("Success! Workspace securely structured for deep learning mapping!")
