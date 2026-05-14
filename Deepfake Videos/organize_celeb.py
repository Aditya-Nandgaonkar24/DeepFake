import os
import shutil
import random

def main():
    base_dir = r"c:\Users\adity\Desktop\practice\Deepfake"
    target_dir = os.path.join(base_dir, "ai_video_detector", "dataset")
    
    # Target structure
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)
            
    # Read test split
    test_files = set()
    test_list_path = os.path.join(base_dir, "List_of_testing_videos.txt")
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    test_files.add(parts[1])
                    
    # Gather all available files
    all_real = []
    all_fake = []
    
    for r_dir in ['Celeb-real', 'YouTube-real']:
        d_path = os.path.join(base_dir, r_dir)
        if os.path.exists(d_path):
            files = os.listdir(d_path)
            for f in files:
                all_real.append(os.path.join(r_dir, f))
                
    c_fake = os.path.join(base_dir, 'Celeb-synthesis')
    if os.path.exists(c_fake):
        files = os.listdir(c_fake)
        for f in files:
            all_fake.append(os.path.join('Celeb-synthesis', f))
            
    # Move Test Files First
    print(f"Assigning {len(test_files)} official test files...")
    for rel_path in test_files:
        src = os.path.join(base_dir, rel_path)
        if os.path.exists(src):
            cls = 'real' if 'real' in rel_path else 'fake'
            dst = os.path.join(target_dir, 'test', cls, os.path.basename(rel_path))
            try:
                shutil.move(src, dst)
            except Exception as e:
                pass
            
            # Remove from tracking lists so we don't try to move again
            if cls == 'real' and rel_path in all_real:
                all_real.remove(rel_path)
            elif cls == 'fake' and rel_path in all_fake:
                all_fake.remove(rel_path)

    # Randomly shuffle and split remaining 80/20 train/val
    print("Splitting remaining files into Train (80%) and Val (20%)...")
    random.shuffle(all_real)
    random.shuffle(all_fake)
    
    def distribute(files, cls_type):
        split_idx = int(len(files) * 0.8)
        train_list = files[:split_idx]
        val_list = files[split_idx:]
        
        for p in train_list:
            src = os.path.join(base_dir, p)
            if os.path.exists(src):
                shutil.move(src, os.path.join(target_dir, 'train', cls_type, os.path.basename(p)))
                
        for p in val_list:
            src = os.path.join(base_dir, p)
            if os.path.exists(src):
                shutil.move(src, os.path.join(target_dir, 'val', cls_type, os.path.basename(p)))
                
    distribute(all_real, 'real')
    distribute(all_fake, 'fake')
    
    print("Celeb-DF successfully mapped into ai_video_detector architecture!")

if __name__ == "__main__":
    main()
