import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import TEMPORAL_INPUT_DIM, GRU_HIDDEN_DIM, GRU_NUM_LAYERS

class TemporalSequenceModel(nn.Module):
    def __init__(self, input_dim=TEMPORAL_INPUT_DIM, hidden_dim=GRU_HIDDEN_DIM, num_layers=GRU_NUM_LAYERS):
        super(TemporalSequenceModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            # Pack sequences so GRU ignores zero-padded frames entirely
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed)
            # We only need hidden states, not unpacked output
        else:
            _, hidden = self.gru(x)
        
        # Properly extract both forward and backward final hidden states
        # hidden shape: (num_layers * 2, batch, hidden_dim) for bidirectional
        forward_final = hidden[-2]   # (batch, hidden_dim)
        backward_final = hidden[-1]  # (batch, hidden_dim)
        final_state = torch.cat([forward_final, backward_final], dim=1)
        return final_state
