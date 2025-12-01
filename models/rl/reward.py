import numpy as np
from typing import Dict


class RewardComputer:
    def __init__(self, config):
        self.config = config
        self.alpha_accuracy = 1.0
        self.alpha_workload = 0.5
        self.alpha_sa = 0.3
    
    def compute_reward(
        self,
        accuracy: float,
        workload: float,
        situational_awareness: float
    ) -> float:
        
        reward = (
            self.alpha_accuracy * accuracy -
            self.alpha_workload * workload +
            self.alpha_sa * situational_awareness
        )
        
        return float(reward)
    
    def compute_shaped_reward(
        self,
        accuracy: float,
        workload: float,
        situational_awareness: float,
        previous_state: Dict,
        current_state: Dict
    ) -> float:
        
        base_reward = self.compute_reward(accuracy, workload, situational_awareness)
        
        attention_improvement = current_state['attention'] - previous_state['attention']
        fatigue_increase = current_state['fatigue'] - previous_state['fatigue']
        stress_increase = current_state['stress'] - previous_state['stress']
        
        shaping_bonus = (
            0.1 * attention_improvement -
            0.15 * fatigue_increase -
            0.1 * stress_increase
        )
        
        total_reward = base_reward + shaping_bonus
        
        return float(total_reward)
    
    def compute_batch_rewards(
        self,
        accuracies: np.ndarray,
        workloads: np.ndarray,
        situational_awarenesses: np.ndarray
    ) -> np.ndarray:
        
        rewards = (
            self.alpha_accuracy * accuracies -
            self.alpha_workload * workloads +
            self.alpha_sa * situational_awarenesses
        )
        
        return rewards