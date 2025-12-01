import torch
import torch.nn as nn
from typing import Tuple


class LSTMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        baseline_config = config['model']['baseline']
        
        if 'input_dim' in baseline_config:
            input_dim = baseline_config['input_dim']
        else:
            input_dim = config['model']['encoder']['cps']['input_dim']
        
        hidden_dim = baseline_config['hidden_dim']
        num_layers = baseline_config['num_layers']
        bidirectional = baseline_config['bidirectional']
        dropout = baseline_config['dropout']
        
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        latent_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(input_dim, input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        
        y, _ = self.decoder(z)
        
        output = self.output_layer(y)
        
        return output
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y, _ = self.decoder(z)
        output = self.output_layer(y)
        return output
    
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        
        x_recon = self.forward(x)
        
        if reduction == 'mean':
            error = torch.mean((x - x_recon) ** 2, dim=(1, 2))
        elif reduction == 'sum':
            error = torch.sum((x - x_recon) ** 2, dim=(1, 2))
        elif reduction == 'none':
            error = (x - x_recon) ** 2
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return error