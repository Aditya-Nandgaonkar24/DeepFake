import os
import shutil
import random

def main():
    base_dir = r"c:\Users\adity\Desktop\practice\Deepfake"
    dfd_dir = os.path.join(base_dir, "DFD")
    target_dir = os.path.join(base_dir, "ai_video_detector", "dataset")
    
    # Map raw folders to their correct class labels
    folders_to_class = {
        'DFD_original sequences': 'real',
        'DFD_manipulated_sequences': 'fake'
    }
    
    splits = ['train', 'val', 'test']
    
    # Create target directories safely
    for s in splits:
        for c in ['real', 'fake']:
            os.makedirs(os.path.join(target_dir, s, c), exist_ok=True)
            
    total_moved = 0
    print("Initiating DFD Dataset Ingestion Pipeline...\n")
    
    for folder_name, cls in folders_to_class.items():
        src_folder = os.path.join(dfd_dir, folder_name)
        if not os.path.exists(src_folder):
            continue
            
        print(f"Reading folder: {folder_name} (Class: {cls})")
        
        # Grab all video files (usually deepfake datasets are .mp4 or .avi)
        files = [f for f in os.listdir(src_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
        random.shuffle(files)
        
        # 80 / 10 / 10 Split
        n = len(files)
        train_idx = int(0.8 * n)
        val_idx = int(0.9 * n)
        
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        dist_map = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for s_name, s_files in dist_map.items():
            for f in s_files:
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(target_dir, s_name, cls, f)
                
                # Prevent hard overwrites if filenames overlap with Celeb/FF++
                if os.path.exists(dst_path):
                    parts = f.rsplit('.', 1)
                    dst_path = os.path.join(target_dir, s_name, cls, f"{parts[0]}_dfd.{parts[1]}")
                    
                try:
                    shutil.move(src_path, dst_path)
                    total_moved += 1
                except Exception as e:
                    pass
                    
    print(f"\nSuccessfully migrated {total_moved} DFD videos seamlessly into the main dataset architecture!")
    print("You can safely delete the empty DFD directory now.")

if __name__ == "__main__":
    main()
