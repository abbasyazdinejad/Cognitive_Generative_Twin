import ollama
import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple


class LLMReasoningModule:
    def __init__(self, config):
        self.config = config
        self.models = config['model']['llm']['models']
        self.temperature = config['model']['llm']['temperature']
        self.max_tokens = config['model']['llm']['max_tokens']
        self.top_k = config['model']['llm']['top_k']
        self.top_p = config['model']['llm']['top_p']
        
        self._ensure_models_available()
    
    def _ensure_models_available(self):
        available_models = [m['name'] for m in ollama.list()['models']]
        
        for model in self.models:
            if model not in available_models:
                print(f"Pulling model: {model}")
                ollama.pull(model)
    
    def _build_rationale_prompt(self, z, severity, stress, metadata):
        z_np = z.detach().cpu().numpy()
        z_stats = {
            'mean': float(np.mean(z_np)),
            'std': float(np.std(z_np)),
            'max': float(np.max(z_np)),
            'min': float(np.min(z_np))
        }
        
        severity_map = {0: 'normal', 1: 'low', 2: 'medium', 3: 'high'}
        stress_map = {0: 'baseline', 1: 'low', 2: 'medium', 3: 'high'}
        
        severity_str = severity_map.get(int(severity), 'unknown')
        stress_str = stress_map.get(int(stress), 'unknown')
        
        prompt = f"""You are an industrial cybersecurity AI assistant analyzing a cyber-physical system incident.

System Context:
- Incident Severity: {severity_str}
- Operator Cognitive Stress: {stress_str}
- Timestamp: {metadata.get('t_event', 'N/A')}
- System State Summary: Mean={z_stats['mean']:.3f}, Std={z_stats['std']:.3f}

Task: Provide a concise technical rationale (2-3 sentences) for the current system state and operator condition.

Rationale:"""
        
        return prompt
    
    def _build_counterfactual_prompt(self, z, severity_current, severity_new, stress_current, stress_new):
        severity_map = {0: 'normal', 1: 'low', 2: 'medium', 3: 'high'}
        stress_map = {0: 'baseline', 1: 'low', 2: 'medium', 3: 'high'}
        
        sev_curr_str = severity_map.get(int(severity_current), 'unknown')
        sev_new_str = severity_map.get(int(severity_new), 'unknown')
        stress_curr_str = stress_map.get(int(stress_current), 'unknown')
        stress_new_str = stress_map.get(int(stress_new), 'unknown')
        
        prompt = f"""You are analyzing an industrial control system incident.

Current State:
- Incident Severity: {sev_curr_str}
- Operator Stress: {stress_curr_str}

Hypothetical Change:
- New Incident Severity: {sev_new_str}
- Expected Operator Stress: {stress_new_str}

Question: How would changing incident severity from {sev_curr_str} to {sev_new_str} affect:
1. Operator stress trajectory
2. Response time
3. Decision accuracy

Provide a brief counterfactual analysis (2-3 sentences).

Analysis:"""
        
        return prompt
    
    def _build_causal_explanation_prompt(self, z, severity, stress, metadata):
        severity_map = {0: 'normal', 1: 'low', 2: 'medium', 3: 'high'}
        stress_map = {0: 'baseline', 1: 'low', 2: 'medium', 3: 'high'}
        
        severity_str = severity_map.get(int(severity), 'unknown')
        stress_str = stress_map.get(int(stress), 'unknown')
        
        prompt = f"""You are an industrial cybersecurity expert.

Observed State:
- Incident Severity: {severity_str}
- Operator Stress: {stress_str}
- Attack Type: {metadata.get('attack_type', 'unknown')}

Task: Explain the causal relationship between the incident severity and operator stress level. What factors contribute to this relationship?

Explanation:"""
        
        return prompt
    
    def generate_rationale(self, z, severity, stress, metadata, model_name=None):
        if model_name is None:
            model_name = self.models[0]
        
        prompt = self._build_rationale_prompt(z, severity, stress, metadata)
        
        start_time = time.time()
        
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
                'top_k': self.top_k,
                'top_p': self.top_p
            }
        )
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'rationale': response['response'],
            'latency_ms': latency,
            'model': model_name
        }
    
    def generate_counterfactual(self, z, severity_current, severity_new, stress_current, stress_new, model_name=None):
        if model_name is None:
            model_name = self.models[2]
        
        prompt = self._build_counterfactual_prompt(z, severity_current, severity_new, stress_current, stress_new)
        
        start_time = time.time()
        
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
                'top_k': self.top_k,
                'top_p': self.top_p
            }
        )
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'counterfactual': response['response'],
            'latency_ms': latency,
            'model': model_name
        }
    
    def generate_causal_explanation(self, z, severity, stress, metadata, model_name=None):
        if model_name is None:
            model_name = self.models[2]
        
        prompt = self._build_causal_explanation_prompt(z, severity, stress, metadata)
        
        start_time = time.time()
        
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
                'top_k': self.top_k,
                'top_p': self.top_p
            }
        )
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'explanation': response['response'],
            'latency_ms': latency,
            'model': model_name
        }
    
    def compare_models(self, z, severity, stress, metadata):
        results = {}
        
        for model in self.models:
            result = self.generate_rationale(z, severity, stress, metadata, model_name=model)
            results[model] = result
        
        return results
    
    def batch_generate_rationales(self, z_batch, severity_batch, stress_batch, metadata_list):
        results = []
        
        for i in range(len(z_batch)):
            z = z_batch[i]
            severity = severity_batch[i]
            stress = stress_batch[i]
            metadata = metadata_list[i] if i < len(metadata_list) else {}
            
            result = self.generate_rationale(z, severity, stress, metadata)
            results.append(result)
        
        return results


class CognitiveLoadEstimator:
    def __init__(self, config):
        self.config = config
    
    def estimate_load(self, z, stress):
        z_np = z.detach().cpu().numpy()
        
        z_variance = np.var(z_np, axis=-1)
        z_magnitude = np.linalg.norm(z_np, axis=-1)
        
        load_score = 0.4 * z_variance + 0.3 * z_magnitude + 0.3 * stress.cpu().numpy()
        
        load_normalized = (load_score - load_score.min()) / (load_score.max() - load_score.min() + 1e-8)
        
        return torch.FloatTensor(load_normalized)
    
    def categorize_load(self, load_score):
        if load_score < 0.33:
            return 'low'
        elif load_score < 0.67:
            return 'medium'
        else:
            return 'high'