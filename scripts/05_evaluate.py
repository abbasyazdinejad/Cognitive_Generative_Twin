import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device
from src.models.cgt import CognitiveGenerativeTwin
from src.models.generative.baseline import LSTMAE
from src.data.loaders import create_dataloaders, create_baseline_dataloaders
from src.evaluation.suite import EvaluationSuite


def evaluate_baseline():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'evaluation' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('evaluation_baseline', log_dir)
    
    logger.info("="*60)
    logger.info("EVALUATING BASELINE LSTM-AE")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    cps_data = np.load(processed_dir / 'processed_cps.npz', allow_pickle=True)
    
    cps_dict = {
        'windows': cps_data['windows'],
        'labels': cps_data['labels'],
        'severity': cps_data['severity']
    }
    
    logger.info(f"\nLoaded CPS data: {cps_dict['windows'].shape}")
    
    _, _, test_loader = create_baseline_dataloaders(
        cps_dict, config.config, dataset_type='cps'
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    baseline_config = config.config['model']['baseline'].copy()
    baseline_config['input_dim'] = cps_dict['windows'].shape[-1]
    
    temp_config = config.config.copy()
    temp_config['model']['baseline'] = baseline_config
    
    model = LSTMAE(temp_config).to(device)
    
    checkpoint_path = config.paths['experiments'] / 'baseline_lstm_ae' / 'checkpoints' / 'baseline_best.pt'
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please run 02_train_baseline.py first")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    
    evaluation_suite = EvaluationSuite(config.config, device)
    
    results = evaluation_suite.run_full_evaluation(
        model, test_loader, model_type='baseline', generate_explanations=False
    )
    
    evaluation_suite.save_results(results, filename='baseline_evaluation_results.json')
    
    summary = evaluation_suite.generate_summary_report(results)
    print(summary)
    
    logger.info("\n" + "="*60)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info("="*60)


def evaluate_cgt():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'evaluation' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('evaluation_cgt', log_dir)
    
    logger.info("="*60)
    logger.info("EVALUATING COGNITIVE GENERATIVE TWIN")
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
    
    logger.info(f"\nLoaded CPS data: {cps_dict['windows'].shape}")
    logger.info(f"Loaded Bio data: {bio_dict['windows'].shape}")
    logger.info(f"Loaded Behavior data: {beh_dict['windows'].shape}")
    
    _, _, test_loader = create_dataloaders(
        cps_dict, bio_dict, beh_dict, config.config, use_alignment=True
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    cgt_model = CognitiveGenerativeTwin(config.config, device)
    
    checkpoint_path = config.paths['experiments'] / 'checkpoints' / 'cgt_best.pt'
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please run 03_train_cgt.py first")
        return
    
    cgt_model.load(str(checkpoint_path))
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    
    rl_checkpoint_path = config.paths['experiments'] / 'rl_checkpoints' / 'rl_agent_best.pt'
    
    if rl_checkpoint_path.exists():
        cgt_model.rl_agent.load(str(rl_checkpoint_path))
        logger.info(f"Loaded RL agent from: {rl_checkpoint_path}")
    else:
        logger.warning("RL agent checkpoint not found - using untrained agent")
    
    evaluation_suite = EvaluationSuite(config.config, device)
    
    logger.info("\n--- Running Full Evaluation (with LLM explanations) ---")
    logger.info("Note: This may take a while due to LLM generation...")
    
    results = evaluation_suite.run_full_evaluation(
        cgt_model, test_loader, model_type='cgt', generate_explanations=True
    )
    
    evaluation_suite.save_results(results, filename='cgt_evaluation_results.json')
    
    summary = evaluation_suite.generate_summary_report(results)
    print(summary)
    
    logger.info("\n" + "="*60)
    logger.info("CGT EVALUATION COMPLETE")
    logger.info("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CGT or Baseline model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['baseline', 'cgt', 'both'],
        default='both',
        help='Which model to evaluate'
    )
    
    args = parser.parse_args()
    
    if args.model in ['baseline', 'both']:
        print("\n" + "="*60)
        print("STARTING BASELINE EVALUATION")
        print("="*60 + "\n")
        evaluate_baseline()
    
    if args.model in ['cgt', 'both']:
        print("\n" + "="*60)
        print("STARTING CGT EVALUATION")
        print("="*60 + "\n")
        evaluate_cgt()


if __name__ == '__main__':
    main()