import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device
from src.models.cgt import CognitiveGenerativeTwin
from src.data.loaders import create_dataloaders
from src.evaluation.suite import EvaluationSuite


def evaluate_ablation(
    config,
    device,
    test_loader,
    checkpoint_path,
    ablation_name,
    use_diffusion,
    logger
):
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION: {ablation_name}")
    logger.info(f"{'='*60}")
    
    cgt_model = CognitiveGenerativeTwin(config.config, device)
    
    if checkpoint_path.exists():
        cgt_model.load(str(checkpoint_path))
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using randomly initialized model")
    
    evaluation_suite = EvaluationSuite(config.config, device)
    
    logger.info(f"\nEvaluating with use_diffusion={use_diffusion}...")
    
    cgt_model.eval()
    
    all_real_data = []
    all_generated_data = []
    all_real_features = []
    all_generated_features = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 50:
                break
            
            cps_data = batch['cps'].to(device)
            bio_data = batch['bio'].to(device)
            beh_data = batch['beh'].to(device)
            severity = batch['severity'].to(device)
            stress = batch['stress'].to(device)
            
            outputs = cgt_model(
                cps_data, bio_data, beh_data, severity, stress,
                use_diffusion=use_diffusion,
                generate_explanations=False,
                adapt_dashboard=False
            )
            
            real_features = outputs['encodings']['cps']
            gen_features = outputs['reconstructions']['cps']
            
            all_real_data.append(cps_data.cpu())
            all_generated_data.append(cps_data.cpu())
            all_real_features.append(real_features.cpu())
            all_generated_features.append(gen_features.cpu())
    
    real_data = torch.cat(all_real_data, dim=0)
    generated_data = torch.cat(all_generated_data, dim=0)
    real_features = torch.cat(all_real_features, dim=0)
    generated_features = torch.cat(all_generated_features, dim=0)
    
    fidelity_results = evaluation_suite.fidelity_metrics.compute_all_fidelity_metrics(
        real_data, generated_data, real_features, generated_features
    )
    
    logger.info(f"\nFidelity Results:")
    logger.info(f"  DTW: {fidelity_results['dtw']:.4f}")
    logger.info(f"  MMD: {fidelity_results['mmd']:.4f}")
    logger.info(f"  Discriminator Accuracy: {fidelity_results['discriminator_accuracy']:.4f}")
    logger.info(f"  MSE: {fidelity_results['mse']:.6f}")
    
    return {
        'ablation': ablation_name,
        'use_diffusion': use_diffusion,
        'fidelity': fidelity_results
    }


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'ablations' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('ablation_study', log_dir)
    
    logger.info("="*60)
    logger.info("ABLATION STUDY: CGT COMPONENTS")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    
    cps_data = np.load(processed_dir / 'processed_cps.npz', allow_pickle=True)
    bio_data = np.load(processed_dir / 'processed_bio.npz', allow_pickle=True)
    beh_data = np.load(processed_dir / 'processed_beh.npz', allow_pickle=True)
    
    cps_dict = {
        'windows': cps_data['windows'],
        'labels': cps_data['labels'],
        'severity': cps_data['severity']
    }
    
    bio_dict = {
        'windows': bio_data['windows'],
        'labels': bio_data['labels'],
        'stress': bio_data['stress']
    }
    
    beh_dict = {
        'windows': beh_data['windows'],
        'labels': beh_data['labels'],
        'workload': beh_data['workload']
    }
    
    _, _, test_loader = create_dataloaders(
        cps_dict, bio_dict, beh_dict, config.config, use_alignment=True
    )
    
    checkpoint_path = config.paths['experiments'] / 'checkpoints' / 'cgt_best.pt'
    
    ablation_studies = [
        {
            'name': 'Full CGT (with Diffusion)',
            'use_diffusion': True
        },
        {
            'name': 'CGT without Diffusion (CVAE only)',
            'use_diffusion': False
        }
    ]
    
    all_results = []
    
    for study in ablation_studies:
        result = evaluate_ablation(
            config,
            device,
            test_loader,
            checkpoint_path,
            study['name'],
            study['use_diffusion'],
            logger
        )
        all_results.append(result)
    
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*60)
    
    for result in all_results:
        logger.info(f"\n{result['ablation']}:")
        logger.info(f"  DTW: {result['fidelity']['dtw']:.4f}")
        logger.info(f"  MMD: {result['fidelity']['mmd']:.4f}")
        logger.info(f"  Disc Acc: {result['fidelity']['discriminator_accuracy']:.4f}")
        logger.info(f"  MSE: {result['fidelity']['mse']:.6f}")
    
    if len(all_results) >= 2:
        full_cgt = all_results[0]['fidelity']
        no_diffusion = all_results[1]['fidelity']
        
        logger.info("\n" + "-"*60)
        logger.info("IMPROVEMENT WITH DIFFUSION:")
        logger.info("-"*60)
        
        dtw_improvement = ((no_diffusion['dtw'] - full_cgt['dtw']) / no_diffusion['dtw']) * 100
        mmd_improvement = ((no_diffusion['mmd'] - full_cgt['mmd']) / no_diffusion['mmd']) * 100
        
        logger.info(f"DTW Reduction: {dtw_improvement:.2f}%")
        logger.info(f"MMD Reduction: {mmd_improvement:.2f}%")
    
    results_path = config.paths['results'] / 'ablation_results.json'
    
    serializable_results = []
    for result in all_results:
        serializable_result = {
            'ablation': result['ablation'],
            'use_diffusion': result['use_diffusion'],
            'fidelity': {k: float(v) for k, v in result['fidelity'].items()}
        }
        serializable_results.append(serializable_result)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nAblation results saved to: {results_path}")
    
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()