import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional


class BaselineTrainer:
    def __init__(self, model, config, device, log_dir: Optional[Path] = None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        training_config = config['training']['baseline']
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.0)
        )
        
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config['project']['exp_dir']) / 'baseline_checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
        
        for batch in pbar:
            if isinstance(batch, dict):
                x = batch['data'].to(self.device)
            else:
                x = batch[0].to(self.device)
            
            self.optimizer.zero_grad()
            
            x_recon = self.model(x)
            
            loss = self.criterion(x_recon, x)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} - Validation'):
                if isinstance(batch, dict):
                    x = batch['data'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                x_recon = self.model(x)
                
                loss = self.criterion(x_recon, x)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            
            print(f"Train Loss: {train_metrics['loss']:.6f}, Val Loss: {val_metrics['loss']:.6f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best')
                print(f"✓ Best model saved with val_loss: {self.best_val_loss:.6f}")
        
        return history
    
    def save_checkpoint(self, name: str):
        checkpoint_path = self.checkpoint_dir / f'baseline_{name}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']