import sys
import argparse
from pathlib import Path

from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device

import torch
import numpy as np


def run_preprocessing(config, logger):
    from src.data.preprocessing import (
        SWaTPreprocessor,
        WADIPreprocessor,
        WESADPreprocessor,
        SWELLPreprocessor
    )
    
    logger.info("="*60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n--- Processing SWaT Dataset ---")
    try:
        swat_processor = SWaTPreprocessor(config.config)
        swat_data = swat_processor.process()
        
        logger.info(f"SWaT Train Windows: {swat_data['train_windows'].shape}")
        logger.info(f"SWaT Test Windows: {swat_data['test_windows'].shape}")
        
        np.savez(
            processed_dir / 'swat_processed.npz',
            train_windows=swat_data['train_windows'],
            train_labels=swat_data['train_labels'],
            train_severity=swat_data['train_severity'],
            test_windows=swat_data['test_windows'],
            test_labels=swat_data['test_labels'],
            test_severity=swat_data['test_severity'],
            mean=swat_data['mean'],
            std=swat_data['std'],
            feature_names=swat_data['feature_names']
        )
        
        logger.info("✓ SWaT processed successfully")
        
    except Exception as e:
        logger.error(f"✗ Error processing SWaT: {e}")
        raise
    
    logger.info("\n--- Processing WADI Dataset ---")
    try:
        wadi_processor = WADIPreprocessor(config.config)
        wadi_data = wadi_processor.process()
        
        logger.info(f"WADI Windows: {wadi_data['windows'].shape}")
        
        np.savez(
            processed_dir / 'wadi_processed.npz',
            windows=wadi_data['windows'],
            labels=wadi_data['labels'],
            severity=wadi_data['severity'],
            mean=wadi_data['mean'],
            std=wadi_data['std'],
            feature_names=wadi_data['feature_names']
        )
        
        logger.info("✓ WADI processed successfully")
        
    except Exception as e:
        logger.error(f"✗ Error processing WADI: {e}")
        raise
    
    logger.info("\n--- Processing WESAD Dataset ---")
    try:
        wesad_processor = WESADPreprocessor(config.config)
        wesad_data = wesad_processor.process()
        
        logger.info(f"WESAD Windows: {wesad_data['windows'].shape}")
        
        np.savez(
            processed_dir / 'wesad_processed.npz',
            windows=wesad_data['windows'],
            labels=wesad_data['labels'],
            stress=wesad_data['stress'],
            mean=wesad_data['mean'],
            std=wesad_data['std']
        )
        
        logger.info("✓ WESAD processed successfully")
        
    except Exception as e:
        logger.error(f"✗ Error processing WESAD: {e}")
        raise
    
    logger.info("\n--- Processing SWELL Dataset ---")
    try:
        swell_processor = SWELLPreprocessor(config.config)
        swell_data = swell_processor.process()
        
        if len(swell_data['windows']) > 0:
            logger.info(f"SWELL Windows: {swell_data['windows'].shape}")
            
            np.savez(
                processed_dir / 'swell_processed.npz',
                windows=swell_data['windows'],
                labels=swell_data['labels'],
                workload=swell_data['workload'],
                mean=swell_data['mean'],
                std=swell_data['std']
            )
            
            logger.info("✓ SWELL processed successfully")
        else:
            logger.warning("! SWELL dataset is empty")
        
    except Exception as e:
        logger.warning(f"! Error processing SWELL: {e}")
        swell_data = {'windows': np.array([]), 'labels': np.array([]), 'workload': np.array([])}
    
    logger.info("\n--- Creating Combined Datasets ---")
    
    cps_train = swat_data['train_windows']
    cps_test = swat_data['test_windows']
    cps_train_labels = swat_data['train_labels']
    cps_test_labels = swat_data['test_labels']
    cps_train_severity = swat_data['train_severity']
    cps_test_severity = swat_data['test_severity']
    
    if len(wadi_data['windows']) > 0:
        cps_all = np.vstack([cps_train, cps_test, wadi_data['windows']])
        cps_labels = np.concatenate([cps_train_labels, cps_test_labels, wadi_data['labels']])
        cps_severity = np.concatenate([cps_train_severity, cps_test_severity, wadi_data['severity']])
    else:
        cps_all = np.vstack([cps_train, cps_test])
        cps_labels = np.concatenate([cps_train_labels, cps_test_labels])
        cps_severity = np.concatenate([cps_train_severity, cps_test_severity])
    
    np.savez(
        processed_dir / 'processed_cps.npz',
        windows=cps_all,
        labels=cps_labels,
        severity=cps_severity,
        mean=swat_data['mean'],
        std=swat_data['std']
    )
    
    logger.info(f"✓ Combined CPS: {cps_all.shape}")
    
    if len(wesad_data['windows']) > 0:
        np.savez(
            processed_dir / 'processed_bio.npz',
            windows=wesad_data['windows'],
            labels=wesad_data['labels'],
            stress=wesad_data['stress'],
            mean=wesad_data['mean'],
            std=wesad_data['std']
        )
        logger.info(f"✓ Bio data: {wesad_data['windows'].shape}")
    
    if len(swell_data['windows']) > 0:
        np.savez(
            processed_dir / 'processed_beh.npz',
            windows=swell_data['windows'],
            labels=swell_data['labels'],
            workload=swell_data['workload'],
            mean=swell_data['mean'],
            std=swell_data['std']
        )
        logger.info(f"✓ Behavior data: {swell_data['windows'].shape}")
    else:
        bio_windows = wesad_data['windows']
        dummy_beh_windows = np.random.randn(len(bio_windows), 256, 12).astype(np.float32)
        dummy_beh_labels = np.zeros(len(bio_windows))
        dummy_beh_workload = np.random.randint(0, 3, len(bio_windows))
        
        np.savez(
            processed_dir / 'processed_beh.npz',
            windows=dummy_beh_windows,
            labels=dummy_beh_labels,
            workload=dummy_beh_workload,
            mean=np.zeros(12),
            std=np.ones(12)
        )
        logger.info("✓ Dummy behavior data created")
    
    logger.info("\n✓ Preprocessing complete!")


def run_baseline_training(config, device, logger):
    from src.models.generative.baseline import LSTMAE
    from src.data.loaders import create_baseline_dataloaders
    from src.training.trainer_baseline import BaselineTrainer
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: BASELINE TRAINING (LSTM-AE)")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    cps_data = np.load(processed_dir / 'processed_cps.npz', allow_pickle=True)
    
    cps_dict = {
        'windows': cps_data['windows'],
        'labels': cps_data['labels'],
        'severity': cps_data['severity']
    }
    
    logger.info(f"Loaded CPS data: {cps_dict['windows'].shape}")
    
    train_loader, val_loader, test_loader = create_baseline_dataloaders(
        cps_dict, config.config, dataset_type='cps'
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    baseline_config = config.config['model']['baseline'].copy()
    baseline_config['input_dim'] = cps_dict['windows'].shape[-1]
    
    temp_config = config.config.copy()
    temp_config['model']['baseline'] = baseline_config
    
    model = LSTMAE(temp_config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    log_dir = config.paths['experiments'] / 'baseline_lstm_ae' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = BaselineTrainer(model, config.config, device, log_dir)
    
    epochs = config.config['training']['baseline']['epochs']
    
    history = trainer.train(train_loader, val_loader, epochs)
    
    logger.info(f"\n✓ Baseline training complete! Best val loss: {trainer.best_val_loss:.6f}")


def run_cgt_training(config, device, logger):
    from src.models.cgt import CognitiveGenerativeTwin
    from src.data.loaders import create_dataloaders
    from src.training.trainer_generative import GenerativeTrainer
    
    logger.info("\n" + "="*60)
    logger.info("STEP 3: CGT TRAINING")
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
    
    logger.info(f"CPS: {cps_dict['windows'].shape}")
    logger.info(f"Bio: {bio_dict['windows'].shape}")
    logger.info(f"Behavior: {beh_dict['windows'].shape}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        cps_dict, bio_dict, beh_dict, config.config, use_alignment=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    cgt_model = CognitiveGenerativeTwin(config.config, device)
    
    total_params = sum(p.numel() for p in cgt_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    
    log_dir = config.paths['experiments'] / 'cgt_full' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = GenerativeTrainer(cgt_model, config.config, device, log_dir)
    
    epochs = config.config['training']['generative']['epochs']
    
    history = trainer.train(train_loader, val_loader, epochs)
    
    logger.info(f"\n✓ CGT training complete! Best val loss: {trainer.best_val_loss:.4f}")


def run_rl_training(config, device, logger):
    from src.models.rl.dashboard_agent import DashboardAdapter
    from src.training.trainer_rl import RLTrainer
    
    logger.info("\n" + "="*60)
    logger.info("STEP 4: RL TRAINING")
    logger.info("="*60)
    
    rl_agent = DashboardAdapter(config.config)
    
    policy_params = sum(p.numel() for p in rl_agent.policy_net.parameters() if p.requires_grad)
    value_params = sum(p.numel() for p in rl_agent.value_net.parameters() if p.requires_grad)
    
    logger.info(f"Policy parameters: {policy_params:,}")
    logger.info(f"Value parameters: {value_params:,}")
    
    trainer = RLTrainer(rl_agent, config.config, device)
    
    history = trainer.train()
    
    avg_reward = np.mean(history['episode_rewards'][-100:])
    logger.info(f"\n✓ RL training complete! Avg reward (last 100): {avg_reward:.3f}")


def run_evaluation(config, device, logger):
    from src.models.cgt import CognitiveGenerativeTwin
    from src.models.generative.baseline import LSTMAE
    from src.data.loaders import create_dataloaders, create_baseline_dataloaders
    from src.evaluation.suite import EvaluationSuite
    
    logger.info("\n" + "="*60)
    logger.info("STEP 5: EVALUATION")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    
    logger.info("\n--- Evaluating Baseline ---")
    
    cps_data = np.load(processed_dir / 'processed_cps.npz', allow_pickle=True)
    
    cps_dict = {
        'windows': cps_data['windows'],
        'labels': cps_data['labels'],
        'severity': cps_data['severity']
    }
    
    _, _, test_loader_baseline = create_baseline_dataloaders(
        cps_dict, config.config, dataset_type='cps'
    )
    
    baseline_config = config.config['model']['baseline'].copy()
    baseline_config['input_dim'] = cps_dict['windows'].shape[-1]
    
    temp_config = config.config.copy()
    temp_config['model']['baseline'] = baseline_config
    
    baseline_model = LSTMAE(temp_config).to(device)
    
    baseline_checkpoint = config.paths['experiments'] / 'baseline_lstm_ae' / 'checkpoints' / 'baseline_best.pt'
    
    if baseline_checkpoint.exists():
        checkpoint = torch.load(baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ Loaded baseline checkpoint")
        
        evaluation_suite = EvaluationSuite(config.config, device)
        
        baseline_results = evaluation_suite.run_full_evaluation(
            baseline_model, test_loader_baseline, model_type='baseline', generate_explanations=False
        )
        
        evaluation_suite.save_results(baseline_results, filename='baseline_evaluation_results.json')
        
        logger.info("✓ Baseline evaluation complete")
    else:
        logger.warning("! Baseline checkpoint not found - skipping baseline evaluation")
    
    logger.info("\n--- Evaluating CGT ---")
    
    bio_data = np.load(processed_dir / 'processed_bio.npz', allow_pickle=True)
    beh_data = np.load(processed_dir / 'processed_beh.npz', allow_pickle=True)
    
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
    
    _, _, test_loader_cgt = create_dataloaders(
        cps_dict, bio_dict, beh_dict, config.config, use_alignment=True
    )
    
    cgt_model = CognitiveGenerativeTwin(config.config, device)
    
    cgt_checkpoint = config.paths['experiments'] / 'checkpoints' / 'cgt_best.pt'
    
    if cgt_checkpoint.exists():
        cgt_model.load(str(cgt_checkpoint))
        logger.info(f"✓ Loaded CGT checkpoint")
        
        rl_checkpoint = config.paths['experiments'] / 'rl_checkpoints' / 'rl_agent_best.pt'
        if rl_checkpoint.exists():
            cgt_model.rl_agent.load(str(rl_checkpoint))
            logger.info(f"✓ Loaded RL agent checkpoint")
        
        evaluation_suite = EvaluationSuite(config.config, device)
        
        cgt_results = evaluation_suite.run_full_evaluation(
            cgt_model, test_loader_cgt, model_type='cgt', generate_explanations=True
        )
        
        evaluation_suite.save_results(cgt_results, filename='cgt_evaluation_results.json')
        
        summary = evaluation_suite.generate_summary_report(cgt_results)
        print(summary)
        
        logger.info("✓ CGT evaluation complete")
    else:
        logger.warning("! CGT checkpoint not found - skipping CGT evaluation")


def main():
    parser = argparse.ArgumentParser(description='CGT: Cognitive Generative Twin for Industrial CPS')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', 'preprocess', 'baseline', 'cgt', 'rl', 'evaluate'],
        default='all',
        help='Which stage to run'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = Config(config_path)
    
    if args.device != 'auto':
        config.config['project']['device'] = args.device
    
    if args.seed is not None:
        config.config['project']['seed'] = args.seed
    
    seed = config.config['project']['seed']
    set_seed(seed)
    
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'main_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('main', log_dir)
    
    logger.info("\n" + "="*60)
    logger.info("COGNITIVE GENERATIVE TWIN (CGT)")
    logger.info("Industrial Cyber-Physical Systems Security")
    logger.info("="*60)
    logger.info(f"\nConfiguration: {config_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Stage: {args.stage}")
    
    try:
        if args.stage in ['all', 'preprocess']:
            run_preprocessing(config, logger)
        
        if args.stage in ['all', 'baseline']:
            run_baseline_training(config, device, logger)
        
        if args.stage in ['all', 'cgt']:
            run_cgt_training(config, device, logger)
        
        if args.stage in ['all', 'rl']:
            run_rl_training(config, device, logger)
        
        if args.stage in ['all', 'evaluate']:
            run_evaluation(config, device, logger)
        
        logger.info("\n" + "="*60)
        logger.info("ALL STAGES COMPLETE!")
        logger.info("="*60)
        
        logger.info("\nResults saved to:")
        logger.info(f"  - Experiments: {config.paths['experiments']}")
        logger.info(f"  - Results: {config.paths['results']}")
        logger.info(f"  - Figures: {config.paths['results'] / 'figures'}")
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()