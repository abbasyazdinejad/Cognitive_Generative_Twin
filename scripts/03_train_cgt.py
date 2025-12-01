import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device
from src.models.cgt import CognitiveGenerativeTwin
from src.data.loaders import create_dataloaders
from src.training.trainer_generative import GenerativeTrainer
from src.utils.visualization import plot_training_curves


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'cgt_full' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('cgt_training', log_dir)
    
    logger.info("="*60)
    logger.info("TRAINING COGNITIVE GENERATIVE TWIN")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    
    logger.info("\n--- Loading Processed Data ---")
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
    
    logger.info(f"CPS Windows: {cps_dict['windows'].shape}")
    logger.info(f"Bio Windows: {bio_dict['windows'].shape}")
    logger.info(f"Behavior Windows: {beh_dict['windows'].shape}")
    
    logger.info("\n--- Creating DataLoaders with Alignment ---")
    train_loader, val_loader, test_loader = create_dataloaders(
        cps_dict, bio_dict, beh_dict, config.config, use_alignment=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    logger.info("\n--- Creating CGT Model ---")
    cgt_model = CognitiveGenerativeTwin(config.config, device)
    
    total_params = sum(p.numel() for p in cgt_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    
    encoder_params = (
        sum(p.numel() for p in cgt_model.cps_encoder.parameters() if p.requires_grad) +
        sum(p.numel() for p in cgt_model.bio_encoder.parameters() if p.requires_grad) +
        sum(p.numel() for p in cgt_model.beh_encoder.parameters() if p.requires_grad)
    )
    cvae_params = sum(p.numel() for p in cgt_model.cvae.parameters() if p.requires_grad)
    diffusion_params = sum(p.numel() for p in cgt_model.diffusion.parameters() if p.requires_grad)
    
    logger.info(f"  Encoders: {encoder_params:,}")
    logger.info(f"  CVAE: {cvae_params:,}")
    logger.info(f"  Diffusion: {diffusion_params:,}")
    
    logger.info("\n--- Starting Training ---")
    
    trainer = GenerativeTrainer(cgt_model, config.config, device, log_dir)
    
    epochs = config.config['training']['generative']['epochs']
    
    history = trainer.train(train_loader, val_loader, epochs)
    
    logger.info("\n--- Training Complete ---")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    logger.info("\n--- Plotting Training Curves ---")
    figures_dir = config.paths['results'] / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(history, save_path=figures_dir / 'cgt_training_curves.png')
    
    logger.info(f"Training curves saved to: {figures_dir / 'cgt_training_curves.png'}")
    
    logger.info("\n--- Saving Training History ---")
    import json
    
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    
    history_path = config.paths['results'] / 'cgt_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    logger.info(f"Training history saved to: {history_path}")
    
    logger.info("\n" + "="*60)
    logger.info("CGT TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()