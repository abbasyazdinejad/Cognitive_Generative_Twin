import ollama
import time
from typing import Dict, List, Optional


class OllamaClient:
    def __init__(self, config):
        self.config = config
        self.models = config['model']['llm']['models']
        self.temperature = config['model']['llm']['temperature']
        self.max_tokens = config['model']['llm']['max_tokens']
        self.top_k = config['model']['llm']['top_k']
        self.top_p = config['model']['llm']['top_p']
        
        self._ensure_models_available()
    
    def _ensure_models_available(self):
        available_models = []
        try:
            model_list = ollama.list()
            available_models = [m['name'] for m in model_list.get('models', [])]
        except Exception as e:
            print(f"Warning: Could not list Ollama models: {e}")
            return
        
        for model in self.models:
            if model not in available_models:
                print(f"Pulling model: {model}")
                try:
                    ollama.pull(model)
                except Exception as e:
                    print(f"Error pulling model {model}: {e}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        
        if model is None:
            model = self.models[0]
        
        if temperature is None:
            temperature = self.temperature
        
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        start_time = time.time()
        
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_k': self.top_k,
                    'top_p': self.top_p
                }
            )
            
            latency = (time.time() - start_time) * 1000
            
            return {
                'response': response['response'],
                'latency_ms': latency,
                'model': model,
                'success': True
            }
        
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                'response': '',
                'latency_ms': latency,
                'model': model,
                'success': False,
                'error': str(e)
            }
    
    def batch_generate(
        self,
        prompts: List[str],
        model: Optional[str] = None
    ) -> List[Dict]:
        
        results = []
        
        for prompt in prompts:
            result = self.generate(prompt, model)
            results.append(result)
        
        return results
    
    def compare_models(
        self,
        prompt: str
    ) -> Dict[str, Dict]:
        
        results = {}
        
        for model in self.models:
            result = self.generate(prompt, model)
            results[model] = result
        
        return results


class PromptBuilder:
    def __init__(self):
        self.severity_map = {
            0: 'normal',
            1: 'low',
            2: 'medium',
            3: 'high'
        }
        
        self.stress_map = {
            0: 'baseline',
            1: 'low',
            2: 'medium',
            3: 'high'
        }
    
    def build_rationale_prompt(
        self,
        z_stats: Dict,
        severity: int,
        stress: int,
        metadata: Dict
    ) -> str:
        
        severity_str = self.severity_map.get(severity, 'unknown')
        stress_str = self.stress_map.get(stress, 'unknown')
        
        prompt = f"""You are an industrial cybersecurity AI assistant analyzing a cyber-physical system incident.

System Context:
- Incident Severity: {severity_str}
- Operator Cognitive Stress: {stress_str}
- Timestamp: {metadata.get('t_event', 'N/A')}
- System State Summary: Mean={z_stats['mean']:.3f}, Std={z_stats['std']:.3f}

Task: Provide a concise technical rationale (2-3 sentences) for the current system state and operator condition.

Rationale:"""
        
        return prompt
    
    def build_counterfactual_prompt(
        self,
        severity_current: int,
        severity_new: int,
        stress_current: int,
        stress_new: int
    ) -> str:
        
        sev_curr_str = self.severity_map.get(severity_current, 'unknown')
        sev_new_str = self.severity_map.get(severity_new, 'unknown')
        stress_curr_str = self.stress_map.get(stress_current, 'unknown')
        stress_new_str = self.stress_map.get(stress_new, 'unknown')
        
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
    
    def build_causal_explanation_prompt(
        self,
        severity: int,
        stress: int,
        metadata: Dict
    ) -> str:
        
        severity_str = self.severity_map.get(severity, 'unknown')
        stress_str = self.stress_map.get(stress, 'unknown')
        
        prompt = f"""You are an industrial cybersecurity expert.

Observed State:
- Incident Severity: {severity_str}
- Operator Stress: {stress_str}
- Attack Type: {metadata.get('attack_type', 'unknown')}

Task: Explain the causal relationship between the incident severity and operator stress level. What factors contribute to this relationship?

Explanation:"""
        
        return prompt
    
    def build_adaptive_summary_prompt(
        self,
        z_stats: Dict,
        severity: int,
        stress: int,
        context: str
    ) -> str:
        
        severity_str = self.severity_map.get(severity, 'unknown')
        stress_str = self.stress_map.get(stress, 'unknown')
        
        prompt = f"""You are an AI assistant helping security operators understand system behavior.

System State:
- Incident Severity: {severity_str}
- Operator Cognitive Load: {stress_str}
- Context: {context}

Task: Provide a brief adaptive summary that helps the operator understand the evolving CPS-cognition relationship.

Summary:"""
        
        return prompt