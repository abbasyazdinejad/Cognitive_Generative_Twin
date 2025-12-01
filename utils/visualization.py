import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def plot_training_curves(
    history: Dict,
    save_path: Optional[Path] = None
):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_cvae_loss'], label='Train CVAE', linewidth=2)
    axes[0, 1].plot(history['val_cvae_loss'], label='Val CVAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('CVAE Loss')
    axes[0, 1].set_title('CVAE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['train_recon_loss'], label='Train Recon', linewidth=2)
    axes[1, 0].plot(history['val_recon_loss'], label='Val Recon', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Reconstruction Loss')
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['train_diffusion_loss'], label='Train Diffusion', linewidth=2)
    axes[1, 1].plot(history['val_diffusion_loss'], label='Val Diffusion', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Diffusion Loss')
    axes[1, 1].set_title('Diffusion Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_latent_space(
    latent_vectors: torch.Tensor,
    labels: np.ndarray,
    method: str = 'pca',
    save_path: Optional[Path] = None
):
    
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.cpu().numpy()
    
    latent_2d = latent_vectors.reshape(latent_vectors.shape[0], -1)
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(latent_2d)
        title = f'Latent Space Visualization (PCA)'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(latent_2d)
        title = f'Latent Space Visualization (t-SNE)'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                   c=[color], label=f'Class {label}', 
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space plot saved to: {save_path}")
    
    plt.show()


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    num_samples: int = 3,
    save_path: Optional[Path] = None
):
    
    num_samples = min(num_samples, len(original))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        axes[i, 0].plot(original[i], linewidth=1.5)
        axes[i, 0].set_title(f'Original Sample {i+1}')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Feature Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(reconstructed[i], linewidth=1.5, color='orange')
        axes[i, 1].set_title(f'Reconstructed Sample {i+1}')
        axes[i, 1].set_xlabel('Time Step')
        axes[i, 1].set_ylabel('Feature Value')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction plot saved to: {save_path}")
    
    plt.show()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[Path] = None
):
    
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC-PR curves saved to: {save_path}")
    
    plt.show()


def plot_robustness_curves(
    robustness_results: Dict,
    save_path: Optional[Path] = None
):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    noise_levels = robustness_results['noise_robustness']['noise_levels']
    noise_accuracies = robustness_results['noise_robustness']['accuracies']
    noise_std = robustness_results['noise_robustness']['std_devs']
    
    axes[0].plot(noise_levels, noise_accuracies, 'o-', linewidth=2, markersize=8)
    axes[0].fill_between(noise_levels, 
                         np.array(noise_accuracies) - np.array(noise_std),
                         np.array(noise_accuracies) + np.array(noise_std),
                         alpha=0.3)
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Robustness to Gaussian Noise')
    axes[0].grid(True, alpha=0.3)
    
    dropout_rates = robustness_results['dropout_robustness']['dropout_rates']
    dropout_accuracies = robustness_results['dropout_robustness']['accuracies']
    dropout_std = robustness_results['dropout_robustness']['std_devs']
    
    axes[1].plot(dropout_rates, dropout_accuracies, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].fill_between(dropout_rates,
                         np.array(dropout_accuracies) - np.array(dropout_std),
                         np.array(dropout_accuracies) + np.array(dropout_std),
                         alpha=0.3, color='orange')
    axes[1].set_xlabel('Dropout Rate')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Robustness to Channel Dropout')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness curves saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()