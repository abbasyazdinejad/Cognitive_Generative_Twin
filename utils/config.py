import yaml
from pathlib import Path
from typing import Dict, Any
import torch


class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
        self._setup_device()
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        required_sections = ['project', 'data', 'model', 'training', 'evaluation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def _setup_paths(self):
        root = Path(self.config['project']['root_dir'])
        
        self.paths = {
            'root': root,
            'data': root / self.config['project']['data_dir'],
            'raw': root / self.config['data']['raw_dir'],
            'processed': root / self.config['data']['processed_dir'],
            'interim': root / self.config['data']['interim_dir'],
            'experiments': root / self.config['project']['exp_dir'],
            'results': root / self.config['project']['results_dir']
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self):
        device_config = self.config['project']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_config)
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getitem__(self, key: str):
        return self.get(key)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


def set_seed(seed: int):
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False