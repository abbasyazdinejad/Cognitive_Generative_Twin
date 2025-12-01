import torch
import numpy as np
import random
import yaml
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(load_path):
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_recon'], label='Train')
    axes[0, 1].plot(history['val_recon'], label='Val')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['train_kld'], label='Train')
    axes[1, 0].plot(history['val_kld'], label='Val')
    axes[1, 0].set_title('KLD Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['train_diff'], label='Train')
    axes[1, 1].plot(history['val_diff'], label='Val')
    axes[1, 1].set_title('Diffusion Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_space(z, labels, save_path):
    from sklearn.decomposition import PCA
    
    z_np = z.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_np, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Severity/Stress Level')
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reconstruction(original, reconstructed, save_path, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        axes[i, 0].plot(original[i])
        axes[i, 0].set_title(f'Original Sample {i+1}')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(reconstructed[i])
        axes[i, 1].set_title(f'Reconstructed Sample {i+1}')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path):
    models = list(metrics_dict.keys())
    metric_names = list(metrics_dict[models[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names[:4]):
        values = [metrics_dict[model][metric] for model in models]
        
        axes[idx].bar(models, values)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylabel('Value')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_robustness_results(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    noise_keys = sorted([k for k in results.keys() if 'noise' in k])
    dropout_keys = sorted([k for k in results.keys() if 'dropout' in k])
    
    if noise_keys:
        noise_values = [results[k] for k in noise_keys]
        axes[0].plot(noise_keys, noise_values, marker='o')
        axes[0].set_title('Robustness to CPS Noise')
        axes[0].set_xlabel('Noise Configuration')
        axes[0].set_ylabel('Relative Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
    
    if dropout_keys:
        dropout_values = [results[k] for k in dropout_keys]
        axes[1].plot(dropout_keys, dropout_values, marker='o', color='orange')
        axes[1].set_title('Robustness to Channel Dropout')
        axes[1].set_xlabel('Dropout Configuration')
        axes[1].set_ylabel('Relative Accuracy')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_experiment_dir(base_dir, exp_name):
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    (exp_dir / 'visualizations').mkdir(exist_ok=True)
    
    return exp_dir


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count