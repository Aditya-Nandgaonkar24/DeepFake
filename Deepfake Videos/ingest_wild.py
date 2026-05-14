import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    base_dir = r"c:\Users\adity\Desktop\practice\Deepfake"
    target_base = os.path.join(base_dir, "ai_video_detector", "dataset_preprocessed")
    target_sequence_len = 15
    
    split_map = {
        'train': 'train',
        'test': 'test',
        'valid': 'val'
    }
    classes = ['real', 'fake']
    
    total_videos = 0
    total_images_processed = 0

    print("Beginning WildDeepfake PNG to JPEG Sequence Conversion...\n")
    
    for src_split, dst_split in split_map.items():
        for cls in classes:
            src_path = os.path.join(base_dir, src_split, cls)
            if not os.path.exists(src_path):
                continue
                
            dst_path = os.path.join(target_base, dst_split, cls)
            os.makedirs(dst_path, exist_ok=True)
            
            files = os.listdir(src_path)
            png_files = [f for f in files if f.endswith('.png')]
            if len(png_files) == 0:
                continue
                
            # Group files by video_id (assumes format videoID_frameID.png)
            videos = {}
            for f in png_files:
                parts = f.split('_')
                if len(parts) >= 2:
                    vid_id = parts[0]
                    if vid_id not in videos:
                        videos[vid_id] = []
                    videos[vid_id].append(f)
            
            print(f"[{src_split}/{cls}] Found {len(videos)} unique videos.")
            
            for vid_id, frames in tqdm(videos.items()):
                # Sort frames chronologically if possible by the integer frame_id
                try:
                    frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                except ValueError:
                    frames.sort()
                
                # Subsample to exactly sequence_length equidistant frames
                if len(frames) > target_sequence_len:
                    indices = np.linspace(0, len(frames) - 1, target_sequence_len, dtype=int)
                    selected_frames = [frames[i] for i in indices]
                else:
                    selected_frames = frames
                    
                target_video_folder = os.path.join(dst_path, f"wild_{vid_id}")
                
                # Skip if already processed
                if os.path.exists(target_video_folder) and len(os.listdir(target_video_folder)) > 0:
                    continue
                    
                os.makedirs(target_video_folder, exist_ok=True)
                
                for i, frame_filename in enumerate(selected_frames):
                    src_file = os.path.join(src_path, frame_filename)
                    # Convert to JPG to match expected dataset.py pipeline
                    try:
                        img = Image.open(src_file).convert('RGB')
                        dst_file = os.path.join(target_video_folder, f"face_{i:03d}.jpg")
                        img.save(dst_file, format='JPEG', quality=95)
                        total_images_processed += 1
                    except Exception as e:
                        pass
                        
                total_videos += 1

    print(f"\nConversion Complete! Ingested {total_videos} videos ({total_images_processed} JPEG frames).")

if __name__ == "__main__":
    main()
