import torch
import numpy as np
from typing import Dict, List
from ..data.augmentation import DataAugmenter


class RobustnessMetrics:
    def __init__(self, config):
        self.config = config
        self.augmenter = DataAugmenter(config['project']['seed'])
        
        eval_config = config['evaluation']['robustness']
        self.noise_levels = eval_config['noise_levels']
        self.dropout_rates = eval_config['dropout_rates']
        self.num_trials = eval_config['num_trials']
    
    def evaluate_noise_robustness(
        self,
        model,
        data: torch.Tensor,
        labels: torch.Tensor,
        metric_fn
    ) -> Dict[str, List[float]]:
        
        results = {
            'noise_levels': [],
            'accuracies': [],
            'std_devs': []
        }
        
        for noise_level in self.noise_levels:
            trial_accuracies = []
            
            for trial in range(self.num_trials):
                noisy_data = self.augmenter.add_gaussian_noise(
                    data.cpu().numpy(),
                    noise_level,
                    fraction_features=0.3
                )
                
                noisy_data_tensor = torch.FloatTensor(noisy_data)
                
                accuracy = metric_fn(model, noisy_data_tensor, labels)
                trial_accuracies.append(accuracy)
            
            results['noise_levels'].append(noise_level)
            results['accuracies'].append(np.mean(trial_accuracies))
            results['std_devs'].append(np.std(trial_accuracies))
        
        return results
    
    def evaluate_dropout_robustness(
        self,
        model,
        data: torch.Tensor,
        labels: torch.Tensor,
        metric_fn
    ) -> Dict[str, List[float]]:
        
        results = {
            'dropout_rates': [],
            'accuracies': [],
            'std_devs': []
        }
        
        for dropout_rate in self.dropout_rates:
            trial_accuracies = []
            
            for trial in range(self.num_trials):
                dropped_data = self.augmenter.apply_dropout(
                    data.cpu().numpy(),
                    dropout_rate
                )
                
                dropped_data_tensor = torch.FloatTensor(dropped_data)
                
                accuracy = metric_fn(model, dropped_data_tensor, labels)
                trial_accuracies.append(accuracy)
            
            results['dropout_rates'].append(dropout_rate)
            results['accuracies'].append(np.mean(trial_accuracies))
            results['std_devs'].append(np.std(trial_accuracies))
        
        return results
    
    def compute_retention_rate(
        self,
        baseline_accuracy: float,
        corrupted_accuracy: float
    ) -> float:
        
        if baseline_accuracy == 0:
            return 0.0
        
        retention = (corrupted_accuracy / baseline_accuracy) * 100
        return float(retention)
    
    def evaluate_combined_robustness(
        self,
        model,
        data: torch.Tensor,
        labels: torch.Tensor,
        metric_fn
    ) -> Dict[str, any]:
        
        model.eval()
        
        with torch.no_grad():
            baseline_accuracy = metric_fn(model, data, labels)
        
        noise_results = self.evaluate_noise_robustness(model, data, labels, metric_fn)
        dropout_results = self.evaluate_dropout_robustness(model, data, labels, metric_fn)
        
        noise_retention_rates = [
            self.compute_retention_rate(baseline_accuracy, acc)
            for acc in noise_results['accuracies']
        ]
        
        dropout_retention_rates = [
            self.compute_retention_rate(baseline_accuracy, acc)
            for acc in dropout_results['accuracies']
        ]
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'noise_robustness': {
                'noise_levels': noise_results['noise_levels'],
                'accuracies': noise_results['accuracies'],
                'std_devs': noise_results['std_devs'],
                'retention_rates': noise_retention_rates
            },
            'dropout_robustness': {
                'dropout_rates': dropout_results['dropout_rates'],
                'accuracies': dropout_results['accuracies'],
                'std_devs': dropout_results['std_devs'],
                'retention_rates': dropout_retention_rates
            },
            'average_noise_retention': np.mean(noise_retention_rates),
            'average_dropout_retention': np.mean(dropout_retention_rates),
            'overall_robustness_score': (np.mean(noise_retention_rates) + np.mean(dropout_retention_rates)) / 2
        }