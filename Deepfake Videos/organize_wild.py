import os
import shutil

def main():
    base_dir = r"c:\Users\adity\Desktop\practice\Deepfake"
    target_dir = os.path.join(base_dir, "ai_video_detector", "dataset")
    
    # Map their directory names to our directory names
    split_map = {
        'train': 'train',
        'test': 'test',
        'valid': 'val'
    }
    
    classes = ['real', 'fake']
    
    total_moved = 0
    
    for src_split, dst_split in split_map.items():
        for cls in classes:
            src_path = os.path.join(base_dir, src_split, cls)
            dst_path = os.path.join(target_dir, dst_split, cls)
            
            # Ensure target directory exists
            os.makedirs(dst_path, exist_ok=True)
            
            if os.path.exists(src_path):
                files = os.listdir(src_path)
                for file in files:
                    if file.endswith('.mp4'):
                        full_src = os.path.join(src_path, file)
                        full_dst = os.path.join(dst_path, file)
                        
                        # Handle potential filename collisions (extremely rare but possible across datasets)
                        if os.path.exists(full_dst):
                            parts = file.rsplit('.', 1)
                            new_name = f"{parts[0]}_wild.{parts[1]}"
                            full_dst = os.path.join(dst_path, new_name)
                            
                        try:
                            shutil.move(full_src, full_dst)
                            total_moved += 1
                        except Exception as e:
                            print(f"Skipping {file}: {e}")
                            
    print(f"WildDeepfake successfully mapped! Total videos migrated: {total_moved}")
    print("You can now safely delete the empty train, test, and valid folders from the Desktop.")

if __name__ == "__main__":
    main()
