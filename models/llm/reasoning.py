import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from .ollama_interface import OllamaClient, PromptBuilder


class LLMReasoningModule:
    def __init__(self, config):
        self.config = config
        self.ollama_client = OllamaClient(config)
        self.prompt_builder = PromptBuilder()
    
    def _compute_z_stats(self, z: torch.Tensor) -> Dict[str, float]:
        z_np = z.detach().cpu().numpy()
        
        return {
            'mean': float(np.mean(z_np)),
            'std': float(np.std(z_np)),
            'max': float(np.max(z_np)),
            'min': float(np.min(z_np))
        }
    
    def generate_rationale(
        self,
        z: torch.Tensor,
        severity: int,
        stress: int,
        metadata: Dict,
        model_name: Optional[str] = None
    ) -> Dict:
        
        z_stats = self._compute_z_stats(z)
        
        prompt = self.prompt_builder.build_rationale_prompt(
            z_stats, severity, stress, metadata
        )
        
        result = self.ollama_client.generate(prompt, model=model_name)
        
        return {
            'rationale': result['response'],
            'latency_ms': result['latency_ms'],
            'model': result['model'],
            'success': result['success']
        }
    
    def generate_counterfactual(
        self,
        z: torch.Tensor,
        severity_current: int,
        severity_new: int,
        stress_current: int,
        stress_new: int,
        model_name: Optional[str] = None
    ) -> Dict:
        
        if model_name is None:
            model_name = self.config['model']['llm']['models'][2]
        
        prompt = self.prompt_builder.build_counterfactual_prompt(
            severity_current, severity_new, stress_current, stress_new
        )
        
        result = self.ollama_client.generate(prompt, model=model_name)
        
        return {
            'counterfactual': result['response'],
            'latency_ms': result['latency_ms'],
            'model': result['model'],
            'success': result['success']
        }
    
    def generate_causal_explanation(
        self,
        z: torch.Tensor,
        severity: int,
        stress: int,
        metadata: Dict,
        model_name: Optional[str] = None
    ) -> Dict:
        
        if model_name is None:
            model_name = self.config['model']['llm']['models'][2]
        
        prompt = self.prompt_builder.build_causal_explanation_prompt(
            severity, stress, metadata
        )
        
        result = self.ollama_client.generate(prompt, model=model_name)
        
        return {
            'explanation': result['response'],
            'latency_ms': result['latency_ms'],
            'model': result['model'],
            'success': result['success']
        }
    
    def generate_adaptive_summary(
        self,
        z: torch.Tensor,
        severity: int,
        stress: int,
        context: str,
        model_name: Optional[str] = None
    ) -> Dict:
        
        z_stats = self._compute_z_stats(z)
        
        prompt = self.prompt_builder.build_adaptive_summary_prompt(
            z_stats, severity, stress, context
        )
        
        result = self.ollama_client.generate(prompt, model=model_name)
        
        return {
            'summary': result['response'],
            'latency_ms': result['latency_ms'],
            'model': result['model'],
            'success': result['success']
        }
    
    def compare_models(
        self,
        z: torch.Tensor,
        severity: int,
        stress: int,
        metadata: Dict
    ) -> Dict[str, Dict]:
        
        z_stats = self._compute_z_stats(z)
        
        prompt = self.prompt_builder.build_rationale_prompt(
            z_stats, severity, stress, metadata
        )
        
        results = self.ollama_client.compare_models(prompt)
        
        return results
    
    def batch_generate_rationales(
        self,
        z_batch: torch.Tensor,
        severity_batch: torch.Tensor,
        stress_batch: torch.Tensor,
        metadata_list: List[Dict],
        model_name: Optional[str] = None
    ) -> List[Dict]:
        
        results = []
        
        for i in range(len(z_batch)):
            z = z_batch[i]
            severity = int(severity_batch[i].item())
            stress = int(stress_batch[i].item())
            metadata = metadata_list[i] if i < len(metadata_list) else {}
            
            result = self.generate_rationale(z, severity, stress, metadata, model_name)
            results.append(result)
        
        return results
    
    def batch_generate_counterfactuals(
        self,
        z_batch: torch.Tensor,
        severity_current_batch: torch.Tensor,
        severity_new_batch: torch.Tensor,
        stress_current_batch: torch.Tensor,
        stress_new_batch: torch.Tensor,
        model_name: Optional[str] = None
    ) -> List[Dict]:
        
        results = []
        
        for i in range(len(z_batch)):
            z = z_batch[i]
            sev_curr = int(severity_current_batch[i].item())
            sev_new = int(severity_new_batch[i].item())
            stress_curr = int(stress_current_batch[i].item())
            stress_new = int(stress_new_batch[i].item())
            
            result = self.generate_counterfactual(
                z, sev_curr, sev_new, stress_curr, stress_new, model_name
            )
            results.append(result)
        
        return results