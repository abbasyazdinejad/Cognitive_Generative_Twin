import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
from tqdm import tqdm
from .fidelity import FidelityMetrics
from .robustness import RobustnessMetrics
from .operational import OperationalMetrics
from .interpretability import InterpretabilityMetrics


class EvaluationSuite:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        self.fidelity_metrics = FidelityMetrics(config)
        self.robustness_metrics = RobustnessMetrics(config)
        self.operational_metrics = OperationalMetrics(config)
        self.interpretability_metrics = InterpretabilityMetrics(config)
        
        self.results_dir = Path(config['project']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_fidelity(
        self,
        model,
        test_loader,
        model_type: str = 'cgt'
    ) -> Dict[str, float]:
        
        print("\n=== Evaluating Fidelity Metrics ===")
        
        model.eval()
        
        all_real_data = []
        all_generated_data = []
        all_real_features = []
        all_generated_features = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Computing fidelity"):
                if model_type == 'cgt':
                    cps_data = batch['cps'].to(self.device)
                    bio_data = batch['bio'].to(self.device)
                    beh_data = batch['beh'].to(self.device)
                    severity = batch['severity'].to(self.device)
                    stress = batch['stress'].to(self.device)
                    
                    outputs = model(cps_data, bio_data, beh_data, severity, stress, 
                                  use_diffusion=True, generate_explanations=False)
                    
                    real_features = outputs['encodings']['cps']
                    gen_features = outputs['reconstructions']['cps']
                    
                    all_real_data.append(cps_data.cpu())
                    all_generated_data.append(cps_data.cpu())
                    all_real_features.append(real_features.cpu())
                    all_generated_features.append(gen_features.cpu())
                
                else:
                    x = batch[0].to(self.device)
                    x_recon = model(x)
                    
                    all_real_data.append(x.cpu())
                    all_generated_data.append(x_recon.cpu())
                    all_real_features.append(x.cpu())
                    all_generated_features.append(x_recon.cpu())
        
        real_data = torch.cat(all_real_data, dim=0)
        generated_data = torch.cat(all_generated_data, dim=0)
        real_features = torch.cat(all_real_features, dim=0)
        generated_features = torch.cat(all_generated_features, dim=0)
        
        fidelity_results = self.fidelity_metrics.compute_all_fidelity_metrics(
            real_data, generated_data, real_features, generated_features
        )
        
        print(f"DTW Distance: {fidelity_results['dtw']:.4f}")
        print(f"MMD: {fidelity_results['mmd']:.4f}")
        print(f"Discriminator Accuracy: {fidelity_results['discriminator_accuracy']:.4f}")
        print(f"MSE: {fidelity_results['mse']:.6f}")
        print(f"MAE: {fidelity_results['mae']:.6f}")
        
        return fidelity_results
    
    def evaluate_robustness(
        self,
        model,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        model_type: str = 'cgt'
    ) -> Dict[str, any]:
        
        print("\n=== Evaluating Robustness Metrics ===")
        
        def compute_accuracy(model, data, labels):
            model.eval()
            
            with torch.no_grad():
                if model_type == 'cgt':
                    batch_size = 32
                    errors = []
                    
                    for i in range(0, len(data), batch_size):
                        batch_data = data[i:i+batch_size].to(self.device)
                        batch_labels = labels[i:i+batch_size]
                        
                        dummy_bio = torch.randn(len(batch_data), 256, 8).to(self.device)
                        dummy_beh = torch.randn(len(batch_data), 256, 12).to(self.device)
                        dummy_severity = torch.zeros(len(batch_data), dtype=torch.long).to(self.device)
                        dummy_stress = torch.zeros(len(batch_data), dtype=torch.long).to(self.device)
                        
                        outputs = model(batch_data, dummy_bio, dummy_beh, 
                                      dummy_severity, dummy_stress, use_diffusion=False)
                        
                        recon = outputs['reconstructions']['cps']
                        orig = outputs['encodings']['cps']
                        
                        batch_errors = torch.mean((orig - recon) ** 2, dim=1).cpu().numpy()
                        errors.extend(batch_errors)
                    
                    errors = np.array(errors)
                    threshold = np.percentile(errors, 95)
                    predictions = (errors > threshold).astype(int)
                    accuracy = np.mean(predictions == labels.cpu().numpy())
                
                else:
                    data = data.to(self.device)
                    recon = model(data)
                    
                    errors = torch.mean((data - recon) ** 2, dim=(1, 2)).cpu().numpy()
                    threshold = np.percentile(errors, 95)
                    predictions = (errors > threshold).astype(int)
                    accuracy = np.mean(predictions == labels.cpu().numpy())
            
            return float(accuracy)
        
        robustness_results = self.robustness_metrics.evaluate_combined_robustness(
            model, test_data, test_labels, compute_accuracy
        )
        
        print(f"Baseline Accuracy: {robustness_results['baseline_accuracy']:.4f}")
        print(f"Average Noise Retention: {robustness_results['average_noise_retention']:.2f}%")
        print(f"Average Dropout Retention: {robustness_results['average_dropout_retention']:.2f}%")
        print(f"Overall Robustness Score: {robustness_results['overall_robustness_score']:.2f}%")
        
        return robustness_results
    
    def evaluate_operational(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        severities: np.ndarray,
        timestamps: np.ndarray,
        dashboard_actions: List[Dict],
        cognitive_loads: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, any]:
        
        print("\n=== Evaluating Operational Metrics ===")
        
        operational_results = self.operational_metrics.evaluate_operational_performance(
            anomaly_scores, true_labels, severities, timestamps,
            dashboard_actions, cognitive_loads, predictions
        )
        
        print(f"Time to First Judgment (TTFJ): {operational_results['ttfj']:.2f}")
        print(f"Error Rate: {operational_results['error_rate']:.4f}")
        print(f"Escalation Quality: {operational_results['escalation_quality']:.4f}")
        print(f"ROC-AUC: {operational_results['detection']['roc_auc']:.4f}")
        print(f"Average Precision: {operational_results['detection']['average_precision']:.4f}")
        
        return operational_results
    
    def evaluate_interpretability(
        self,
        explanations: Dict[str, List[str]],
        severities: np.ndarray,
        stresses: np.ndarray
    ) -> Dict[str, float]:
        
        print("\n=== Evaluating Interpretability Metrics ===")
        
        interpretability_results = self.interpretability_metrics.evaluate_explanation_quality(
            explanations, severities, stresses
        )
        
        print(f"Average Clarity: {interpretability_results['avg_clarity']:.4f}")
        print(f"Average Relevance: {interpretability_results['avg_relevance']:.4f}")
        print(f"Average Usefulness: {interpretability_results['avg_usefulness']:.4f}")
        print(f"CF Consistency: {interpretability_results['cf_consistency']:.4f}")
        print(f"Overall Quality: {interpretability_results['overall_quality']:.4f}")
        
        return interpretability_results
    
    def run_full_evaluation(
        self,
        model,
        test_loader,
        model_type: str = 'cgt',
        generate_explanations: bool = True
    ) -> Dict[str, any]:
        
        print("\n" + "="*60)
        print("RUNNING FULL EVALUATION SUITE")
        print("="*60)
        
        results = {}
        
        fidelity_results = self.evaluate_fidelity(model, test_loader, model_type)
        results['fidelity'] = fidelity_results
        
        test_data_list = []
        test_labels_list = []
        
        for batch in test_loader:
            if model_type == 'cgt':
                test_data_list.append(batch['cps'])
                test_labels_list.append(batch['severity'])
            else:
                test_data_list.append(batch[0])
                test_labels_list.append(batch[1])
        
        test_data = torch.cat(test_data_list, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)
        
        robustness_results = self.evaluate_robustness(model, test_data, test_labels, model_type)
        results['robustness'] = robustness_results
        
        if model_type == 'cgt' and generate_explanations:
            anomaly_scores = []
            true_labels = []
            severities = []
            stresses = []
            cognitive_loads = []
            dashboard_actions = []
            explanations_rationales = []
            explanations_counterfactuals = []
            
            model.eval()
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Generating operational data"):
                    cps_data = batch['cps'].to(self.device)
                    bio_data = batch['bio'].to(self.device)
                    beh_data = batch['beh'].to(self.device)
                    severity = batch['severity'].to(self.device)
                    stress = batch['stress'].to(self.device)
                    
                    outputs = model(cps_data, bio_data, beh_data, severity, stress,
                                  use_diffusion=True, generate_explanations=True,
                                  adapt_dashboard=True)
                    
                    recon = outputs['reconstructions']['cps']
                    orig = outputs['encodings']['cps']
                    errors = torch.mean((orig - recon) ** 2, dim=1).cpu().numpy()
                    
                    anomaly_scores.extend(errors)
                    true_labels.extend((severity > 0).cpu().numpy())
                    severities.extend(severity.cpu().numpy())
                    stresses.extend(stress.cpu().numpy())
                    cognitive_loads.extend(outputs['cognitive_load'].cpu().numpy())
                    
                    if 'dashboard_action' in outputs:
                        for _ in range(len(cps_data)):
                            dashboard_actions.append(outputs['dashboard_action'])
                    
                    if 'explanations' in outputs:
                        explanations_rationales.append(outputs['explanations']['rationale']['rationale'])
                        explanations_counterfactuals.append(outputs['explanations']['counterfactual']['counterfactual'])
            
            anomaly_scores = np.array(anomaly_scores)
            true_labels = np.array(true_labels)
            severities = np.array(severities)
            stresses = np.array(stresses)
            cognitive_loads = np.array(cognitive_loads)
            
            timestamps = np.arange(len(anomaly_scores))
            
            threshold = np.percentile(anomaly_scores, 95)
            predictions = (anomaly_scores > threshold).astype(int)
            
            operational_results = self.evaluate_operational(
                anomaly_scores, true_labels, severities, timestamps,
                dashboard_actions, cognitive_loads, predictions
            )
            results['operational'] = operational_results
            
            if explanations_rationales:
                explanations = {
                    'rationales': explanations_rationales,
                    'counterfactuals': explanations_counterfactuals
                }
                
                interpretability_results = self.evaluate_interpretability(
                    explanations, severities[:len(explanations_rationales)],
                    stresses[:len(explanations_rationales)]
                )
                results['interpretability'] = interpretability_results
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        return results
    
    def save_results(self, results: Dict, filename: str = 'evaluation_results.json'):
        output_path = self.results_dir / filename
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def generate_summary_report(self, results: Dict) -> str:
        report = []
        report.append("\n" + "="*60)
        report.append("EVALUATION SUMMARY REPORT")
        report.append("="*60 + "\n")
        
        if 'fidelity' in results:
            report.append("FIDELITY METRICS:")
            report.append(f"  DTW Distance: {results['fidelity']['dtw']:.4f}")
            report.append(f"  MMD: {results['fidelity']['mmd']:.4f}")
            report.append(f"  Discriminator Accuracy: {results['fidelity']['discriminator_accuracy']:.4f}")
            report.append(f"  MSE: {results['fidelity']['mse']:.6f}")
            report.append("")
        
        if 'robustness' in results:
            report.append("ROBUSTNESS METRICS:")
            report.append(f"  Baseline Accuracy: {results['robustness']['baseline_accuracy']:.4f}")
            report.append(f"  Avg Noise Retention: {results['robustness']['average_noise_retention']:.2f}%")
            report.append(f"  Avg Dropout Retention: {results['robustness']['average_dropout_retention']:.2f}%")
            report.append(f"  Overall Robustness: {results['robustness']['overall_robustness_score']:.2f}%")
            report.append("")
        
        if 'operational' in results:
            report.append("OPERATIONAL METRICS:")
            report.append(f"  TTFJ: {results['operational']['ttfj']:.2f}")
            report.append(f"  Error Rate: {results['operational']['error_rate']:.4f}")
            report.append(f"  Escalation Quality: {results['operational']['escalation_quality']:.4f}")
            report.append(f"  ROC-AUC: {results['operational']['detection']['roc_auc']:.4f}")
            report.append(f"  Average Precision: {results['operational']['detection']['average_precision']:.4f}")
            report.append("")
        
        if 'interpretability' in results:
            report.append("INTERPRETABILITY METRICS:")
            report.append(f"  Avg Clarity: {results['interpretability']['avg_clarity']:.4f}")
            report.append(f"  Avg Relevance: {results['interpretability']['avg_relevance']:.4f}")
            report.append(f"  Avg Usefulness: {results['interpretability']['avg_usefulness']:.4f}")
            report.append(f"  CF Consistency: {results['interpretability']['cf_consistency']:.4f}")
            report.append(f"  Overall Quality: {results['interpretability']['overall_quality']:.4f}")
            report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)