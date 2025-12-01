import torch
import torch.nn as nn
from typing import Tuple


class BioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        encoder_config = config['model']['encoder']['bio']
        
        self.input_dim = encoder_config['input_dim']
        self.hidden_dim = encoder_config['hidden_dim']
        self.num_layers = encoder_config['num_layers']
        self.dropout = encoder_config['dropout']
        self.bidirectional = encoder_config['bidirectional']
        
        encoder_type = encoder_config['type'].lower()
        
        if encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif encoder_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x)
        
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.layer_norm(hidden)
        
        return hidden
    
    def get_output_dim(self) -> int:
        return self.output_dim