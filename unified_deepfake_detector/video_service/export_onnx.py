import torch
from models.fusion import SpatialTemporalFusion
from config import SEQUENCE_LENGTH, IMAGE_SIZE, AUDIO_N_MELS, USE_AUDIO_BRANCH

def export_to_onnx(model_path="logs/best_model.pth", out_path="detector_model.onnx", use_audio=USE_AUDIO_BRANCH):
    print("Loading PyTorch architecture...")
    model = SpatialTemporalFusion(seq_length=SEQUENCE_LENGTH, use_audio=use_audio)
    
    # Fail loudly if weights can't be loaded
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded weights from {model_path} successfully.")
    except FileNotFoundError:
        print(f"ERROR: Weight file not found at '{model_path}'. Train the model first.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load weights: {e}")
        print("Aborting export. A model with random weights is useless for inference.")
        return
    
    model.eval()

    # Create dummy inputs matching the model signature
    dummy_frames = torch.randn(1, SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_mask = torch.ones(1, SEQUENCE_LENGTH, dtype=torch.bool)
    dummy_audio = torch.zeros(1, SEQUENCE_LENGTH, AUDIO_N_MELS, dtype=torch.float32)
    
    print(f"Exporting to ONNX format: {out_path} ...")
    torch.onnx.export(
        model, 
        (dummy_frames, dummy_mask, dummy_audio),
        out_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_video', 'mask', 'audio_features'],
        output_names=['logits'],
        dynamic_axes={
            'input_video': {0: 'batch_size'}, 
            'mask': {0: 'batch_size'},
            'audio_features': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print("ONNX export complete!")

if __name__ == "__main__":
    export_to_onnx()
