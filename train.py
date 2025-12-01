import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from utils.helpers import set_seed, load_config, save_config, create_experiment_dir, plot_training_history
from data.preprocessing import SWaTPreprocessor, WADIPreprocessor, WESADPreprocessor, SWELLPreprocessor
from data.loaders import create_dataloaders
from models.encoders import CPSEncoder, BioEncoder, BehaviorEncoder
from models.cvae import ConditionalVAE
from models.generative.diffusion import DiffusionModel
from training.trainer_generative import CGTTrainer


def main(args):
    config = load_config(args.config)
    
    set_seed(config['project']['seed'])
    
    device = torch.device(config['project']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    exp_dir = create_experiment_dir(config['project']['exp_dir'], args.exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    save_config(config, exp_dir / 'config.yaml')
    
    print("\n=== Preprocessing Datasets ===")
    
    print("Processing SWaT...")
    swat_processor = SWaTPreprocessor(config)
    swat_data = swat_processor.process()
    
    print("Processing WADI...")
    wadi_processor = WADIPreprocessor(config)
    wadi_data = wadi_processor.process()
    
    cps_data = {
        'windows': swat_data['windows'],
        'severity': swat_data['severity']
    }
    
    print("Processing WESAD...")
    wesad_processor = WESADPreprocessor(config)
    wesad_data = wesad_processor.process()
    
    bio_data = {
        'windows': wesad_data['windows'],
        'stress': wesad_data['stress']
    }
    
    print("Processing SWELL-KW...")
    swell_processor = SWELLPreprocessor(config)
    swell_data = swell_processor.process()
    
    beh_data = {
        'windows': swell_data['windows'],
        'workload': swell_data['workload']
    }
    
    print("\n=== Creating DataLoaders ===")
    train_loader, val_loader, test_loader = create_dataloaders(cps_data, bio_data, beh_data, config)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n=== Initializing Models ===")
    
    cps_encoder = CPSEncoder(config)
    bio_encoder = BioEncoder(config)
    beh_encoder = BehaviorEncoder(config)
    cvae = ConditionalVAE(config)
    diffusion = DiffusionModel(config)
    
    print(f"CPS Encoder parameters: {sum(p.numel() for p in cps_encoder.parameters())}")
    print(f"Bio Encoder parameters: {sum(p.numel() for p in bio_encoder.parameters())}")
    print(f"Behavior Encoder parameters: {sum(p.numel() for p in beh_encoder.parameters())}")
    print(f"CVAE parameters: {sum(p.numel() for p in cvae.parameters())}")
    print(f"Diffusion parameters: {sum(p.numel() for p in diffusion.parameters())}")
    
    config['project']['checkpoint_dir'] = str(exp_dir / 'checkpoints')
    
    trainer = CGTTrainer(config, cps_encoder, bio_encoder, beh_encoder, cvae, diffusion, device)
    
    print("\n=== Training ===")
    epochs = config['training']['generative']['epochs']
    
    history = trainer.train(train_loader, val_loader, epochs)
    
    print("\n=== Saving Results ===")
    
    import json
    with open(exp_dir / 'results' / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    plot_training_history(history, exp_dir / 'visualizations' / 'training_curves.png')
    
    print(f"\n✓ Training completed!")
    print(f"Results saved to: {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CGT Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--exp_name', type=str, default='cgt_exp_001', help='Experiment name')
    
    args = parser.parse_args()
    main(args)