import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


class OperationalMetrics:
    def __init__(self, config):
        self.config = config
        
        eval_config = config['evaluation']['operational']
        self.num_simulations = eval_config['num_simulations']
        self.attack_windows = eval_config['attack_windows']
        self.num_analysts = eval_config['num_analysts']
        
        self.rng = np.random.default_rng(config['project']['seed'])
    
    def compute_ttfj(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        
        attack_indices = np.where(true_labels == 1)[0]
        
        if len(attack_indices) == 0:
            return float('inf')
        
        first_attack_idx = attack_indices[0]
        
        threshold = np.percentile(anomaly_scores, 95)
        
        detected_indices = np.where(anomaly_scores > threshold)[0]
        
        detection_during_attack = detected_indices[detected_indices >= first_attack_idx]
        
        if len(detection_during_attack) == 0:
            return float('inf')
        
        first_detection_idx = detection_during_attack[0]
        
        ttfj = timestamps[first_detection_idx] - timestamps[first_attack_idx]
        
        return float(ttfj)
    
    def compute_error_rate(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        
        errors = np.sum(predictions != true_labels)
        error_rate = errors / len(true_labels)
        
        return float(error_rate)
    
    def compute_escalation_quality(
        self,
        anomaly_scores: np.ndarray,
        severities: np.ndarray,
        true_labels: np.ndarray,
        top_k: int = 10
    ) -> float:
        
        if len(anomaly_scores) < top_k:
            top_k = len(anomaly_scores)
        
        top_k_indices = np.argsort(anomaly_scores)[-top_k:]
        
        top_k_severities = severities[top_k_indices]
        top_k_labels = true_labels[top_k_indices]
        
        high_severity_count = np.sum(top_k_severities >= 2)
        true_attack_count = np.sum(top_k_labels == 1)
        
        quality_score = (high_severity_count + true_attack_count) / (2 * top_k)
        
        return float(quality_score)
    
    def simulate_analyst_workload(
        self,
        anomaly_scores: np.ndarray,
        dashboard_actions: List[Dict],
        cognitive_loads: np.ndarray
    ) -> Dict[str, float]:
        
        num_alerts = len(anomaly_scores)
        
        avg_cognitive_load = np.mean(cognitive_loads)
        
        alert_grouping_levels = [action['alert_grouping'] for action in dashboard_actions]
        avg_grouping = np.mean(alert_grouping_levels)
        
        viz_density_levels = [action['viz_density'] for action in dashboard_actions]
        avg_viz_density = np.mean(viz_density_levels)
        
        base_workload = num_alerts * 0.1
        
        grouping_factor = 1.0 - (avg_grouping * 0.3)
        viz_factor = 1.0 + (avg_viz_density * 0.2)
        cognitive_factor = 1.0 + (avg_cognitive_load * 0.5)
        
        total_workload = base_workload * grouping_factor * viz_factor * cognitive_factor
        
        time_to_triage = total_workload * 2.5
        
        return {
            'total_workload': float(total_workload),
            'time_to_triage_minutes': float(time_to_triage),
            'avg_cognitive_load': float(avg_cognitive_load),
            'avg_grouping_level': float(avg_grouping),
            'avg_viz_density': float(avg_viz_density)
        }
    
    def compute_detection_metrics(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        
        roc_auc = roc_auc_score(true_labels, anomaly_scores)
        avg_precision = average_precision_score(true_labels, anomaly_scores)
        
        precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
        
        return {
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'best_f1': float(best_f1),
            'best_threshold': float(best_threshold)
        }
    
    def evaluate_operational_performance(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        severities: np.ndarray,
        timestamps: np.ndarray,
        dashboard_actions: List[Dict],
        cognitive_loads: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, any]:
        
        ttfj = self.compute_ttfj(anomaly_scores, true_labels, timestamps)
        
        error_rate = self.compute_error_rate(predictions, true_labels)
        
        escalation_quality = self.compute_escalation_quality(
            anomaly_scores, severities, true_labels, top_k=10
        )
        
        workload_metrics = self.simulate_analyst_workload(
            anomaly_scores, dashboard_actions, cognitive_loads
        )
        
        detection_metrics = self.compute_detection_metrics(anomaly_scores, true_labels)
        
        return {
            'ttfj': ttfj,
            'error_rate': error_rate,
            'escalation_quality': escalation_quality,
            'workload': workload_metrics,
            'detection': detection_metrics
        }