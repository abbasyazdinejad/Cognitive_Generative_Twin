import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional


class GenerativeTrainer:
    def __init__(self, cgt_model, config, device, log_dir: Optional[Path] = None):
        self.cgt = cgt_model
        self.config = config
        self.device = device
        
        training_config = config['training']['generative']
        
        self.optimizer = optim.AdamW(
            list(self.cgt.cps_encoder.parameters()) +
            list(self.cgt.bio_encoder.parameters()) +
            list(self.cgt.beh_encoder.parameters()) +
            list(self.cgt.cvae.parameters()) +
            list(self.cgt.diffusion.parameters()),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['epochs'],
            eta_min=1e-6
        )
        
        self.grad_clip = training_config['grad_clip']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = training_config['early_stopping_patience']
        
        self.checkpoint_dir = Path(config['project']['exp_dir']) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.cgt.cps_encoder.train()
        self.cgt.bio_encoder.train()
        self.cgt.beh_encoder.train()
        self.cgt.cvae.train()
        self.cgt.diffusion.train()
        
        total_loss = 0.0
        total_cvae_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        total_diffusion_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
        
        for batch_idx, batch in enumerate(pbar):
            cps_data = batch['cps'].to(self.device)
            bio_data = batch['bio'].to(self.device)
            beh_data = batch['beh'].to(self.device)
            severity = batch['severity'].to(self.device)
            stress = batch['stress'].to(self.device)
            
            self.optimizer.zero_grad()
            
            losses = self.cgt.train_step(cps_data, bio_data, beh_data, severity, stress)
            
            loss = losses['total_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.cgt.cps_encoder.parameters()) +
                list(self.cgt.bio_encoder.parameters()) +
                list(self.cgt.beh_encoder.parameters()) +
                list(self.cgt.cvae.parameters()) +
                list(self.cgt.diffusion.parameters()),
                self.grad_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_cvae_loss += losses['cvae_loss'].item()
            total_recon_loss += losses['recon_loss'].item()
            total_kld_loss += losses['kld_loss'].item()
            total_diffusion_loss += losses['diffusion_loss'].item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'cvae': losses['cvae_loss'].item(),
                'diff': losses['diffusion_loss'].item()
            })
            
            if self.writer and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/CVAE_Loss', losses['cvae_loss'].item(), global_step)
                self.writer.add_scalar('Train/Diffusion_Loss', losses['diffusion_loss'].item(), global_step)
        
        num_batches = len(train_loader)
        
        return {
            'loss': total_loss / num_batches,
            'cvae_loss': total_cvae_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kld_loss': total_kld_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches
        }
    
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        self.cgt.cps_encoder.eval()
        self.cgt.bio_encoder.eval()
        self.cgt.beh_encoder.eval()
        self.cgt.cvae.eval()
        self.cgt.diffusion.eval()
        
        total_loss = 0.0
        total_cvae_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        total_diffusion_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} - Validation'):
                cps_data = batch['cps'].to(self.device)
                bio_data = batch['bio'].to(self.device)
                beh_data = batch['beh'].to(self.device)
                severity = batch['severity'].to(self.device)
                stress = batch['stress'].to(self.device)
                
                losses = self.cgt.train_step(cps_data, bio_data, beh_data, severity, stress)
                
                total_loss += losses['total_loss'].item()
                total_cvae_loss += losses['cvae_loss'].item()
                total_recon_loss += losses['recon_loss'].item()
                total_kld_loss += losses['kld_loss'].item()
                total_diffusion_loss += losses['diffusion_loss'].item()
        
        num_batches = len(val_loader)
        
        return {
            'loss': total_loss / num_batches,
            'cvae_loss': total_cvae_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kld_loss': total_kld_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches
        }
    
    def train(self, train_loader, val_loader, epochs: int):
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_cvae_loss': [],
            'val_cvae_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kld_loss': [],
            'val_kld_loss': [],
            'train_diffusion_loss': [],
            'val_diffusion_loss': []
        }
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            self.scheduler.step()
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_cvae_loss'].append(train_metrics['cvae_loss'])
            history['val_cvae_loss'].append(val_metrics['cvae_loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])
            history['val_recon_loss'].append(val_metrics['recon_loss'])
            history['train_kld_loss'].append(train_metrics['kld_loss'])
            history['val_kld_loss'].append(val_metrics['kld_loss'])
            history['train_diffusion_loss'].append(train_metrics['diffusion_loss'])
            history['val_diffusion_loss'].append(val_metrics['diffusion_loss'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train CVAE: {train_metrics['cvae_loss']:.4f}, Val CVAE: {val_metrics['cvae_loss']:.4f}")
            print(f"Train Diffusion: {train_metrics['diffusion_loss']:.4f}, Val Diffusion: {val_metrics['diffusion_loss']:.4f}")
            
            if self.writer:
                self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, 'best')
                print(f"✓ Best model saved with val_loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
        
        if self.writer:
            self.writer.close()
        
        return history
    
    def save_checkpoint(self, epoch: int, name: str):
        checkpoint_path = self.checkpoint_dir / f'cgt_{name}.pt'
        self.cgt.save(str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        self.cgt.load(path)
        print(f"Checkpoint loaded: {path}")