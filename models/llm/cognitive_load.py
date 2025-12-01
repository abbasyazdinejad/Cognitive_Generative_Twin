import torch
import numpy as np
from typing import Dict


class CognitiveLoadEstimator:
    def __init__(self, config):
        self.config = config
        self.alpha_variance = 0.4
        self.alpha_magnitude = 0.3
        self.alpha_stress = 0.3
    
    def estimate_load(
        self,
        z: torch.Tensor,
        stress: torch.Tensor
    ) -> torch.Tensor:
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        z_np = z.detach().cpu().numpy()
        
        z_variance = np.var(z_np, axis=-1)
        z_magnitude = np.linalg.norm(z_np, axis=-1)
        
        stress_np = stress.detach().cpu().numpy() if isinstance(stress, torch.Tensor) else stress
        
        if stress_np.ndim == 0:
            stress_np = np.array([stress_np])
        
        load_score = (
            self.alpha_variance * z_variance +
            self.alpha_magnitude * z_magnitude +
            self.alpha_stress * stress_np
        )
        
        if len(load_score) > 0:
            load_min = load_score.min()
            load_max = load_score.max()
            
            if load_max - load_min > 1e-8:
                load_normalized = (load_score - load_min) / (load_max - load_min)
            else:
                load_normalized = np.zeros_like(load_score)
        else:
            load_normalized = np.array([0.0])
        
        return torch.FloatTensor(load_normalized)
    
    def categorize_load(self, load_score: float) -> str:
        if load_score < 0.33:
            return 'low'
        elif load_score < 0.67:
            return 'medium'
        else:
            return 'high'
    
    def batch_estimate_load(
        self,
        z_batch: torch.Tensor,
        stress_batch: torch.Tensor
    ) -> torch.Tensor:
        
        return self.estimate_load(z_batch, stress_batch)
    
    def estimate_load_with_context(
        self,
        z: torch.Tensor,
        stress: torch.Tensor,
        context: Dict
    ) -> Dict[str, any]:
        
        load_score = self.estimate_load(z, stress)
        
        load_value = float(load_score[0]) if load_score.dim() > 0 else float(load_score)
        category = self.categorize_load(load_value)
        
        return {
            'load_score': load_value,
            'category': category,
            'stress_level': int(stress.item()) if isinstance(stress, torch.Tensor) else int(stress),
            'latent_variance': float(np.var(z.detach().cpu().numpy())),
            'latent_magnitude': float(np.linalg.norm(z.detach().cpu().numpy()))
        }