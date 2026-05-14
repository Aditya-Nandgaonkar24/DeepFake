import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, AUDIO_N_MELS

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, sequence_length=15, transform=None, phase='train'):
        """
        data_dir: Path to 'dataset_preprocessed' containing splits and 'real'/'fake' folders.
        sequence_length: Number of frames to extract per item.
        transform: Albumentations composition.
        """
        self.sequence_length = sequence_length
        self.phase = phase
        self.video_paths = []  # These will now point to directories of extracted images
        self.labels = []
        
        # Traverse real videos (label 0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for folder in sorted(os.listdir(real_dir)):  # Deterministic ordering
                f_path = os.path.join(real_dir, folder)
                if os.path.isdir(f_path):
                    self.video_paths.append(f_path)
                    self.labels.append(0.0) # Float for BCEWithLogits
                    
        # Traverse fake videos (label 1)
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for folder in sorted(os.listdir(fake_dir)):  # Deterministic ordering
                f_path = os.path.join(fake_dir, folder)
                if os.path.isdir(f_path):
                    self.video_paths.append(f_path)
                    self.labels.append(1.0)
                    
        # Fix v2#3: Use ReplayCompose for temporally consistent augmentations
        # This ensures the same random transforms are applied across all frames in a sequence
        if transform is None:
            if phase == 'train':
                self.transform = A.ReplayCompose([
                    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.ImageCompression(quality_range=(60, 100), p=0.5),
                    A.GaussNoise(p=0.3),
                    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ToTensorV2()
                ])
            else:
                self.transform = A.ReplayCompose([
                    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        folder_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Output tensor structure (Sequence Length, Channels, Height, Width)
        frames_tensor = torch.zeros((self.sequence_length, 3, IMAGE_SIZE, IMAGE_SIZE))
        # Frame-validity mask: True for real frames, False for zero-padding
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        audio_tensor = torch.zeros((self.sequence_length, AUDIO_N_MELS), dtype=torch.float32)
        
        # Load preprocessed frames in chronological order
        face_files = sorted(
            [
                f for f in os.listdir(folder_path)
                if f.startswith('face_') and f.endswith('.jpg')
            ]
        )
        valid_frames = min(len(face_files), self.sequence_length)
        
        # Apply first frame's augmentation, then replay on remaining frames
        saved_replay = None
        for i in range(valid_frames):
            img_path = os.path.join(folder_path, face_files[i])
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                if saved_replay is None:
                    result = self.transform(image=img_rgb)
                    saved_replay = result['replay']
                    frames_tensor[i] = result['image']
                else:
                    result = A.ReplayCompose.replay(saved_replay, image=img_rgb)
                    frames_tensor[i] = result['image']
                mask[i] = True  # Mark this frame as valid

        audio_features_path = os.path.join(folder_path, "audio_features.npy")
        if os.path.exists(audio_features_path):
            try:
                audio_features = np.load(audio_features_path)
                audio_features = np.asarray(audio_features, dtype=np.float32)
                if audio_features.ndim == 2:
                    feature_dim = min(audio_features.shape[1], AUDIO_N_MELS)
                    length = min(audio_features.shape[0], self.sequence_length)
                    audio_tensor[:length, :feature_dim] = torch.from_numpy(audio_features[:length, :feature_dim])
            except Exception:
                pass

        return frames_tensor, torch.tensor(label, dtype=torch.float32), mask, audio_tensor
