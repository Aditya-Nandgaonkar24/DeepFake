import torch
import torch.nn as nn
from models.spatial_model import SpatialFeatureExtractor
from models.temporal_model import TemporalSequenceModel
from config import TEMPORAL_INPUT_DIM, GRU_HIDDEN_DIM, GRU_NUM_LAYERS, AUDIO_N_MELS, AUDIO_EMBED_DIM

class SpatialTemporalFusion(nn.Module):
    def __init__(self, seq_length=15, freeze_spatial=True, use_audio=False):
        super(SpatialTemporalFusion, self).__init__()
        self.seq_length = seq_length
        self.use_audio = use_audio
        
        self.spatial_extractor = SpatialFeatureExtractor(freeze_blocks=freeze_spatial)
        if self.use_audio:
            self.audio_projector = nn.Sequential(
                nn.LayerNorm(AUDIO_N_MELS),
                nn.Linear(AUDIO_N_MELS, AUDIO_EMBED_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
            )

        temporal_input_dim = TEMPORAL_INPUT_DIM + (AUDIO_EMBED_DIM if self.use_audio else 0)
        self.temporal_model = TemporalSequenceModel(
            input_dim=temporal_input_dim,
            hidden_dim=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(GRU_HIDDEN_DIM * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None, audio_features=None):
        b, seq_len, c, h, w = x.shape
        x = x.reshape(b * seq_len, c, h, w)
        
        # Spatial Processing (Outputs 960 channels)
        spatial_features = self.spatial_extractor(x)
        spatial_features = spatial_features.reshape(b, seq_len, -1) 
        
        # Zero out spatial features at padded positions so diff maps are clean
        if mask is not None:
            # mask shape: (batch, seq_len) — True for real frames, False for padding
            spatial_features = spatial_features * mask.unsqueeze(-1).float()
        
        # Feature-Level Difference Maps
        # Highlights micro-flickering at near-zero computational cost
        diff_features = torch.zeros_like(spatial_features)
        diff_features[:, 1:, :] = spatial_features[:, 1:, :] - spatial_features[:, :-1, :]
        
        # Merge spatial and drift arrays (960 + 960 = 1920 dimensions)
        combined_features = torch.cat([spatial_features, diff_features], dim=2)

        if self.use_audio:
            if audio_features is None:
                audio_features = combined_features.new_zeros((b, seq_len, AUDIO_N_MELS))
            if mask is not None:
                audio_features = audio_features * mask.unsqueeze(-1).float()
            audio_embeddings = self.audio_projector(audio_features)
            combined_features = torch.cat([combined_features, audio_embeddings], dim=2)
        
        # Compute actual sequence lengths from mask for pack_padded_sequence
        if mask is not None:
            lengths = mask.sum(dim=1)  # (batch,) — count of True values per sequence
        else:
            lengths = None
        
        temporal_features = self.temporal_model(combined_features, lengths=lengths)
        out = self.classifier(temporal_features)
        
        return out.squeeze(1)
