import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F

from config import AUDIO_SAMPLE_RATE, AUDIO_N_MELS, SEQUENCE_LENGTH

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    from moviepy import VideoFileClip
except Exception:
    VideoFileClip = None


def _resample_if_needed(waveform, sample_rate, target_sample_rate):
    if sample_rate == target_sample_rate:
        return waveform
    if torchaudio is None:
        return waveform
    resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
    return resampler(waveform)


def _waveform_to_segment_features(waveform, sample_rate, sequence_length, n_mels):
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = _resample_if_needed(waveform, sample_rate, AUDIO_SAMPLE_RATE)
    waveform = waveform.float()

    if waveform.numel() == 0:
        return np.zeros((sequence_length, n_mels), dtype=np.float32)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)
    mel = torch.log(mel.clamp_min(1e-6))
    mel = mel.squeeze(0)

    if mel.ndim != 2 or mel.shape[1] == 0:
        return np.zeros((sequence_length, n_mels), dtype=np.float32)

    mel = mel.unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(mel, (n_mels, sequence_length)).squeeze(0)
    features = pooled.transpose(0, 1).contiguous()
    return features.cpu().numpy().astype(np.float32)


def extract_audio_features_from_video(video_path, sequence_length=SEQUENCE_LENGTH, n_mels=AUDIO_N_MELS):
    if torchaudio is None:
        warnings.warn("torchaudio is unavailable; audio features will be zeroed.")
        return np.zeros((sequence_length, n_mels), dtype=np.float32)

    if VideoFileClip is None:
        warnings.warn("moviepy is unavailable; please run `pip install moviepy` to extract audio safely on Windows.")
        return np.zeros((sequence_length, n_mels), dtype=np.float32)

    try:
        # Utilize MoviePy architecture to completely sidestep Torchaudio's native 
        # failure on Windows MP4 container formats without FFMPEG binaries installed
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                return np.zeros((sequence_length, n_mels), dtype=np.float32)
                
            audio_array = clip.audio.to_soundarray(fps=AUDIO_SAMPLE_RATE)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1) # Mono Conversion
                
            # Restructure back into PyTorch compatible shapes [Channels, Frames]
            waveform = torch.from_numpy(audio_array).unsqueeze(0).float()
            sample_rate = AUDIO_SAMPLE_RATE
            
        return _waveform_to_segment_features(waveform, sample_rate, sequence_length, n_mels)
    except Exception as exc:
        warnings.warn(f"Audio extraction failed for '{os.path.basename(video_path)}': {exc}")
        return np.zeros((sequence_length, n_mels), dtype=np.float32)
