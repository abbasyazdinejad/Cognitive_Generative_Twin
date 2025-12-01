import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device
from src.models.generative.baseline import LSTMAE
from src.data.loaders import create_baseline_dataloaders
from src.training.trainer_baseline import BaselineTrainer
from src.utils.visualization import plot_training_curves


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'baseline_lstm_ae' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('baseline_training', log_dir)
    
    logger.info("="*60)
    logger.info("TRAINING BASELINE LSTM-AE")
    logger.info("="*60)
    
    processed_dir = config.paths['processed']
    
    logger.info("\n--- Loading Processed Data ---")
    cps_data = np.load(processed_dir / 'processed_cps.npz', allow_pickle=True)
    
    cps_dict = {
        'windows': cps_data['windows'],
        'labels': cps_data['labels'],
        'severity': cps_data['severity']
    }
    
    logger.info(f"CPS Windows: {cps_dict['windows'].shape}")
    logger.info(f"CPS Labels: {cps_dict['labels'].shape}")
    
    logger.info("\n--- Creating DataLoaders ---")
    train_loader, val_loader, test_loader = create_baseline_dataloaders(
        cps_dict, config.config, dataset_type='cps'
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    logger.info("\n--- Creating Model ---")
    
    baseline_config = config.config['model']['baseline'].copy()
    baseline_config['input_dim'] = cps_dict['windows'].shape[-1]
    
    temp_config = config.config.copy()
    temp_config['model']['baseline'] = baseline_config
    
    model = LSTMAE(temp_config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    logger.info("\n--- Starting Training ---")
    
    trainer = BaselineTrainer(model, config.config, device, log_dir)
    
    epochs = config.config['training']['baseline']['epochs']
    
    history = trainer.train(train_loader, val_loader, epochs)
    
    logger.info("\n--- Training Complete ---")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    
    logger.info("\n--- Plotting Training Curves ---")
    figures_dir = config.paths['results'] / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Baseline LSTM-AE Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / 'baseline_training_curves.png', dpi=300, bbox_inches='tight')
    logger.info(f"Training curves saved to: {figures_dir / 'baseline_training_curves.png'}")
    
    logger.info("\n--- Evaluating on Test Set ---")
    
    model.load_state_dict(
        torch.load(
            trainer.checkpoint_dir / 'baseline_best.pt',
            map_location=device
        )['model_state_dict']
    )
    
    test_metrics = trainer.validate(test_loader, epoch=0)
    
    logger.info(f"Test Loss: {test_metrics['loss']:.6f}")
    
    logger.info("\n--- Computing Reconstruction Errors ---")
    
    model.eval()
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            labels = batch[1]
            
            errors = model.reconstruction_error(x, reduction='mean')
            
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    threshold = np.percentile(all_errors, 95)
    predictions = (all_errors > threshold).astype(int)
    
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    
    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_errors)
        avg_precision = average_precision_score(all_labels, all_errors)
        accuracy = accuracy_score(all_labels, predictions)
        
        logger.info(f"\n--- Test Set Performance ---")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Accuracy (95th percentile threshold): {accuracy:.4f}")
        
        results = {
            'test_loss': test_metrics['loss'],
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'accuracy': float(accuracy),
            'threshold': float(threshold)
        }
    else:
        logger.warning("Only one class in test set - skipping classification metrics")
        results = {
            'test_loss': test_metrics['loss'],
            'threshold': float(threshold)
        }
    
    import json
    results_path = config.paths['results'] / 'baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    logger.info("\n" + "="*60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()