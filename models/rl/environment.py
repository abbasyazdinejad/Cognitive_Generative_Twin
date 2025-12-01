import numpy as np
from typing import Dict, Tuple


class SOCEnvironment:
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.max_steps = config['training']['rl']['timesteps_per_episode']
        
        self.operator_state = {
            'fatigue': 0.0,
            'attention': 1.0,
            'stress': 0.0,
            'experience': 0.7
        }
        
        self.base_accuracy = 0.9
        self.rng = np.random.default_rng(config['project']['seed'])
    
    def reset(self) -> Dict:
        self.current_step = 0
        
        self.operator_state = {
            'fatigue': 0.0,
            'attention': 1.0,
            'stress': 0.0,
            'experience': self.rng.uniform(0.6, 0.95)
        }
        
        return self._get_state()
    
    def step(
        self,
        action: Dict,
        cognitive_load: float
    ) -> Tuple[Dict, float, bool, Dict]:
        
        alert_grouping = action['alert_grouping']
        viz_density = action['viz_density']
        prioritization = action['prioritization']
        
        if alert_grouping == 2:
            self.operator_state['fatigue'] += 0.02
        elif alert_grouping == 0:
            self.operator_state['attention'] -= 0.01
        else:
            self.operator_state['attention'] += 0.005
        
        if viz_density == 2:
            self.operator_state['stress'] += 0.03
            self.operator_state['fatigue'] += 0.015
        elif viz_density == 0:
            self.operator_state['attention'] -= 0.02
        else:
            self.operator_state['stress'] += 0.01
        
        if prioritization == 2:
            self.operator_state['attention'] += 0.01
        elif prioritization == 0:
            self.operator_state['attention'] -= 0.015
        
        self.operator_state['fatigue'] = np.clip(self.operator_state['fatigue'], 0.0, 1.0)
        self.operator_state['attention'] = np.clip(self.operator_state['attention'], 0.0, 1.0)
        self.operator_state['stress'] = np.clip(self.operator_state['stress'], 0.0, 1.0)
        
        accuracy = self._compute_accuracy(cognitive_load)
        workload = self._compute_workload(cognitive_load, viz_density, alert_grouping)
        situational_awareness = self._compute_situational_awareness(alert_grouping, prioritization)
        
        reward = self._compute_reward(accuracy, workload, situational_awareness)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'accuracy': accuracy,
            'workload': workload,
            'situational_awareness': situational_awareness,
            'operator_state': self.operator_state.copy()
        }
        
        return self._get_state(), reward, done, info
    
    def _compute_accuracy(self, cognitive_load: float) -> float:
        base_accuracy = self.base_accuracy * self.operator_state['experience']
        
        fatigue_penalty = self.operator_state['fatigue'] * 0.3
        attention_factor = self.operator_state['attention']
        stress_penalty = self.operator_state['stress'] * 0.2
        
        accuracy = base_accuracy * attention_factor - fatigue_penalty - stress_penalty
        
        if cognitive_load > 0.7:
            accuracy *= 0.9
        
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        return float(accuracy)
    
    def _compute_workload(
        self,
        cognitive_load: float,
        viz_density: int,
        alert_grouping: int
    ) -> float:
        
        base_workload = cognitive_load
        
        viz_factor = 1.0 + viz_density * 0.2
        alert_factor = 1.0 + (2 - alert_grouping) * 0.1
        
        workload = base_workload * viz_factor * alert_factor
        
        workload = np.clip(workload, 0.0, 1.0)
        
        return float(workload)
    
    def _compute_situational_awareness(
        self,
        alert_grouping: int,
        prioritization: int
    ) -> float:
        
        base_sa = self.operator_state['attention']
        
        alert_factor = 1.0 - alert_grouping * 0.1
        priority_factor = 1.0 + prioritization * 0.15
        fatigue_penalty = self.operator_state['fatigue'] * 0.2
        
        sa = base_sa * alert_factor * priority_factor - fatigue_penalty
        
        sa = np.clip(sa, 0.0, 1.0)
        
        return float(sa)
    
    def _compute_reward(
        self,
        accuracy: float,
        workload: float,
        situational_awareness: float
    ) -> float:
        
        alpha1 = 1.0
        alpha2 = 0.5
        alpha3 = 0.3
        
        reward = alpha1 * accuracy - alpha2 * workload + alpha3 * situational_awareness
        
        return float(reward)
    
    def _get_state(self) -> Dict:
        return {
            'step': self.current_step,
            'operator_state': self.operator_state.copy()
        }
    
    def render(self):
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Operator State: {self.operator_state}")