import torch
import torch.nn as nn


class CPSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['model']['encoder']['cps']['input_dim']
        hidden_dim = config['model']['encoder']['cps']['hidden_dim']
        num_layers = config['model']['encoder']['cps']['num_layers']
        dropout = config['model']['encoder']['cps']['dropout']
        bidirectional = config['model']['encoder']['cps']['bidirectional']
        
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
    
    def forward(self, x):
        output, hidden = self.rnn(x)
        
        if self.config['model']['encoder']['cps']['bidirectional']:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return hidden


class BioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['model']['encoder']['bio']['input_dim']
        hidden_dim = config['model']['encoder']['bio']['hidden_dim']
        num_layers = config['model']['encoder']['bio']['num_layers']
        dropout = config['model']['encoder']['bio']['dropout']
        bidirectional = config['model']['encoder']['bio']['bidirectional']
        
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
    
    def forward(self, x):
        output, hidden = self.rnn(x)
        
        if self.config['model']['encoder']['bio']['bidirectional']:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return hidden


class BehaviorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['model']['encoder']['beh']['input_dim']
        hidden_dim = config['model']['encoder']['beh']['hidden_dim']
        num_layers = config['model']['encoder']['beh']['num_layers']
        dropout = config['model']['encoder']['beh']['dropout']
        bidirectional = config['model']['encoder']['beh']['bidirectional']
        
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
    
    def forward(self, x):
        output, hidden = self.rnn(x)
        
        if self.config['model']['encoder']['beh']['bidirectional']:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return hidden