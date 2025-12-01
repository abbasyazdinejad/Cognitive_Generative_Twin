import torch
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Tuple
import torch.nn as nn


class FidelityMetrics:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_dtw(
        self,
        real_sequences: np.ndarray,
        generated_sequences: np.ndarray
    ) -> float:
        
        dtw_distances = []
        
        num_samples = min(len(real_sequences), len(generated_sequences))
        
        for i in range(num_samples):
            real_seq = real_sequences[i]
            gen_seq = generated_sequences[i]
            
            distance, _ = fastdtw(real_seq, gen_seq, dist=euclidean)
            dtw_distances.append(distance)
        
        avg_dtw = np.mean(dtw_distances)
        
        return float(avg_dtw)
    
    def compute_mmd(
        self,
        real_features: torch.Tensor,
        generated_features: torch.Tensor,
        kernel: str = 'rbf',
        bandwidth: float = 1.0
    ) -> float:
        
        if isinstance(real_features, np.ndarray):
            real_features = torch.FloatTensor(real_features)
        if isinstance(generated_features, np.ndarray):
            generated_features = torch.FloatTensor(generated_features)
        
        real_features = real_features.view(real_features.size(0), -1)
        generated_features = generated_features.view(generated_features.size(0), -1)
        
        def kernel_matrix(x, y, bandwidth):
            x_size = x.size(0)
            y_size = y.size(0)
            
            xx = x.unsqueeze(1).expand(x_size, y_size, x.size(1))
            yy = y.unsqueeze(0).expand(x_size, y_size, y.size(1))
            
            diff = xx - yy
            squared_dist = torch.sum(diff ** 2, dim=2)
            
            if kernel == 'rbf':
                K = torch.exp(-squared_dist / (2 * bandwidth ** 2))
            else:
                K = squared_dist
            
            return K
        
        K_xx = kernel_matrix(real_features, real_features, bandwidth)
        K_yy = kernel_matrix(generated_features, generated_features, bandwidth)
        K_xy = kernel_matrix(real_features, generated_features, bandwidth)
        
        m = real_features.size(0)
        n = generated_features.size(0)
        
        mmd = (K_xx.sum() / (m * m) + K_yy.sum() / (n * n) - 2 * K_xy.sum() / (m * n))
        
        return float(mmd.item())
    
    def train_discriminator(
        self,
        real_features: torch.Tensor,
        generated_features: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 64
    ) -> nn.Module:
        
        real_features = real_features.view(real_features.size(0), -1)
        generated_features = generated_features.view(generated_features.size(0), -1)
        
        input_dim = real_features.size(1)
        
        discriminator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        criterion = nn.BCELoss()
        
        real_features = real_features.to(self.device)
        generated_features = generated_features.to(self.device)
        
        real_labels = torch.ones(real_features.size(0), 1).to(self.device)
        fake_labels = torch.zeros(generated_features.size(0), 1).to(self.device)
        
        discriminator.train()
        
        for epoch in range(epochs):
            num_batches = max(1, real_features.size(0) // batch_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, real_features.size(0))
                
                real_batch = real_features[start_idx:end_idx]
                real_batch_labels = real_labels[start_idx:end_idx]
                
                gen_start = batch_idx * batch_size
                gen_end = min((batch_idx + 1) * batch_size, generated_features.size(0))
                gen_batch = generated_features[gen_start:gen_end]
                gen_batch_labels = fake_labels[gen_start:gen_end]
                
                optimizer.zero_grad()
                
                real_pred = discriminator(real_batch)
                real_loss = criterion(real_pred, real_batch_labels)
                
                fake_pred = discriminator(gen_batch)
                fake_loss = criterion(fake_pred, gen_batch_labels)
                
                loss = real_loss + fake_loss
                loss.backward()
                optimizer.step()
        
        return discriminator
    
    def discriminator_accuracy(
        self,
        real_features: torch.Tensor,
        generated_features: torch.Tensor
    ) -> float:
        
        discriminator = self.train_discriminator(real_features, generated_features)
        
        discriminator.eval()
        
        real_features = real_features.view(real_features.size(0), -1).to(self.device)
        generated_features = generated_features.view(generated_features.size(0), -1).to(self.device)
        
        with torch.no_grad():
            real_pred = discriminator(real_features)
            fake_pred = discriminator(generated_features)
            
            real_correct = (real_pred > 0.5).float().sum().item()
            fake_correct = (fake_pred <= 0.5).float().sum().item()
            
            total_correct = real_correct + fake_correct
            total_samples = real_features.size(0) + generated_features.size(0)
            
            accuracy = total_correct / total_samples
        
        return float(accuracy)
    
    def compute_reconstruction_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        
        mse = torch.mean((original - reconstructed) ** 2).item()
        mae = torch.mean(torch.abs(original - reconstructed)).item()
        
        original_flat = original.view(original.size(0), -1)
        reconstructed_flat = reconstructed.view(reconstructed.size(0), -1)
        
        cosine_sim = nn.functional.cosine_similarity(original_flat, reconstructed_flat, dim=1)
        avg_cosine_sim = cosine_sim.mean().item()
        
        return {
            'mse': mse,
            'mae': mae,
            'cosine_similarity': avg_cosine_sim
        }
    
    def compute_all_fidelity_metrics(
        self,
        real_data: torch.Tensor,
        generated_data: torch.Tensor,
        real_features: torch.Tensor,
        generated_features: torch.Tensor
    ) -> Dict[str, float]:
        
        real_np = real_data.cpu().numpy()
        gen_np = generated_data.cpu().numpy()
        
        dtw = self.compute_dtw(real_np, gen_np)
        
        mmd = self.compute_mmd(real_features, generated_features)
        
        disc_acc = self.discriminator_accuracy(real_features, generated_features)
        
        recon_metrics = self.compute_reconstruction_error(real_data, generated_data)
        
        return {
            'dtw': dtw,
            'mmd': mmd,
            'discriminator_accuracy': disc_acc,
            'mse': recon_metrics['mse'],
            'mae': recon_metrics['mae'],
            'cosine_similarity': recon_metrics['cosine_similarity']
        }