import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device
from src.models.cgt import CognitiveGenerativeTwin
from src.data.loaders import create_dataloaders
from src.utils.visualization import (
    plot_latent_space,
    plot_reconstruction,
    plot_roc_pr_curves,
    plot_robustness_curves
)


def generate_latent_space_visualizations(config, device, test_loader, cgt_model, figures_dir, logger):
    logger.info("\n--- Generating Latent Space Visualizations ---")
    
    cgt_model.eval()
    
    all_latents = []
    all_severities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 100:
                break
            
            cps_data = batch['cps'].to(device)
            bio_data = batch['bio'].to(device)
            beh_data = batch['beh'].to(device)
            severity = batch['severity'].to(device)
            stress = batch['stress'].to(device)
            
            outputs = cgt_model(cps_data, bio_data, beh_data, severity, stress,
                              use_diffusion=False, generate_explanations=False)
            
            all_latents.append(outputs['latent'].cpu())
            all_severities.append(severity.cpu())
    
    latents = torch.cat(all_latents, dim=0)
    severities = torch.cat(all_severities, dim=0).numpy()
    
    plot_latent_space(
        latents, severities, method='pca',
        save_path=figures_dir / 'latent_space_pca.png'
    )
    logger.info("✓ PCA latent space visualization saved")
    
    plot_latent_space(
        latents, severities, method='tsne',
        save_path=figures_dir / 'latent_space_tsne.png'
    )
    logger.info("✓ t-SNE latent space visualization saved")


def generate_reconstruction_visualizations(config, device, test_loader, cgt_model, figures_dir, logger):
    logger.info("\n--- Generating Reconstruction Visualizations ---")
    
    cgt_model.eval()
    
    batch = next(iter(test_loader))
    
    cps_data = batch['cps'].to(device)
    bio_data = batch['bio'].to(device)
    beh_data = batch['beh'].to(device)
    severity = batch['severity'].to(device)
    stress = batch['stress'].to(device)
    
    with torch.no_grad():
        outputs = cgt_model(cps_data, bio_data, beh_data, severity, stress,
                          use_diffusion=True, generate_explanations=False)
    
    original = cps_data.cpu().numpy()[:3]
    
    h_cps = outputs['encodings']['cps']
    
    with torch.no_grad():
        h_cps_recon, _, _ = cgt_model.decode_latent(
            outputs['latent'][:3], severity[:3], stress[:3]
        )
    
    reconstructed_features = h_cps_recon.cpu().numpy()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    for i in range(3):
        axes[i, 0].plot(original[i, :, 0], linewidth=1.5, label='Feature 0')
        axes[i, 0].set_title(f'Original Sample {i+1}')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(reconstructed_features[i], linewidth=1.5, color='orange', label='Reconstructed')
        axes[i, 1].set_title(f'Reconstructed Features {i+1}')
        axes[i, 1].set_xlabel('Feature Index')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Reconstruction comparison saved")
    plt.close()


def generate_roc_pr_visualizations(config, device, test_loader, cgt_model, figures_dir, logger):
    logger.info("\n--- Generating ROC and PR Curves ---")
    
    cgt_model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            cps_data = batch['cps'].to(device)
            bio_data = batch['bio'].to(device)
            beh_data = batch['beh'].to(device)
            severity = batch['severity'].to(device)
            stress = batch['stress'].to(device)
            
            outputs = cgt_model(cps_data, bio_data, beh_data, severity, stress,
                              use_diffusion=False, generate_explanations=False)
            
            recon = outputs['reconstructions']['cps']
            orig = outputs['encodings']['cps']
            
            errors = torch.mean((orig - recon) ** 2, dim=1).cpu().numpy()
            labels = (severity > 0).cpu().numpy()
            
            all_scores.extend(errors)
            all_labels.extend(labels)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) > 1:
        plot_roc_pr_curves(
            all_labels, all_scores,
            save_path=figures_dir / 'roc_pr_curves.png'
        )
        logger.info("✓ ROC and PR curves saved")
    else:
        logger.warning("Only one class in test set - skipping ROC/PR curves")


def generate_robustness_visualizations(config, figures_dir, logger):
    logger.info("\n--- Generating Robustness Visualizations ---")
    
    results_path = config.paths['results'] / 'cgt_evaluation_results.json'
    
    if not results_path.exists():
        logger.warning(f"Evaluation results not found: {results_path}")
        logger.warning("Please run 05_evaluate.py first")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if 'robustness' in results:
        plot_robustness_curves(
            results['robustness'],
            save_path=figures_dir / 'robustness_curves.png'
        )
        logger.info("✓ Robustness curves saved")
    else:
        logger.warning("Robustness results not found in evaluation results")


def generate_comparison_table(config, figures_dir, logger):
    logger.info("\n--- Generating Comparison Table ---")
    
    baseline_path = config.paths['results'] / 'baseline_evaluation_results.json'
    cgt_path = config.paths['results'] / 'cgt_evaluation_results.json'
    
    if not baseline_path.exists() or not cgt_path.exists():
        logger.warning("Evaluation results not complete")
        logger.warning("Please run 05_evaluate.py first")
        return
    
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    with open(cgt_path, 'r') as f:
        cgt_results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    metrics = ['DTW', 'MMD', 'Disc. Acc.', 'MSE']
    baseline_values = [
        baseline_results['fidelity']['dtw'],
        baseline_results['fidelity']['mmd'],
        baseline_results['fidelity']['discriminator_accuracy'],
        baseline_results['fidelity']['mse']
    ]
    cgt_values = [
        cgt_results['fidelity']['dtw'],
        cgt_results['fidelity']['mmd'],
        cgt_results['fidelity']['discriminator_accuracy'],
        cgt_results['fidelity']['mse']
    ]
    
    table_data = []
    for metric, baseline_val, cgt_val in zip(metrics, baseline_values, cgt_values):
        if metric == 'Disc. Acc.':
            improvement = ((0.5 - abs(cgt_val - 0.5)) / (0.5 - abs(baseline_val - 0.5)) - 1) * 100
        else:
            improvement = ((baseline_val - cgt_val) / baseline_val) * 100
        
        table_data.append([
            metric,
            f'{baseline_val:.4f}',
            f'{cgt_val:.4f}',
            f'{improvement:+.2f}%'
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Baseline', 'CGT', 'Improvement'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    for i in range(len(metrics) + 1):
        if i == 0:
            table[(i, 0)].set_facecolor('#40466e')
            table[(i, 1)].set_facecolor('#40466e')
            table[(i, 2)].set_facecolor('#40466e')
            table[(i, 3)].set_facecolor('#40466e')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
            table[(i, 2)].set_text_props(weight='bold', color='white')
            table[(i, 3)].set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#f0f0f0')
                table[(i, 2)].set_facecolor('#f0f0f0')
                table[(i, 3)].set_facecolor('#f0f0f0')
    
    plt.title('Baseline vs CGT Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(figures_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Comparison table saved")
    plt.close()


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'figures' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('figure_generation', log_dir)
    
    logger.info("="*60)
    logger.info("GENERATING PAPER FIGURES")
    logger.info("="*60)
    
    figures_dir = config.paths['results'] / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    _, _, test_loader = create_dataloaders(
        cps_dict, bio_dict, beh_dict, config.config, use_alignment=True
    )
    
    checkpoint_path = config.paths['experiments'] / 'checkpoints' / 'cgt_best.pt'
    
    if not checkpoint_path.exists():
        logger.error(f"CGT checkpoint not found: {checkpoint_path}")
        logger.error("Please run 03_train_cgt.py first")
        return
    
    cgt_model = CognitiveGenerativeTwin(config.config, device)
    cgt_model.load(str(checkpoint_path))
    logger.info(f"Loaded CGT checkpoint from: {checkpoint_path}")
    
    generate_latent_space_visualizations(config, device, test_loader, cgt_model, figures_dir, logger)
    
    generate_reconstruction_visualizations(config, device, test_loader, cgt_model, figures_dir, logger)
    
    generate_roc_pr_visualizations(config, device, test_loader, cgt_model, figures_dir, logger)
    
    generate_robustness_visualizations(config, figures_dir, logger)
    
    generate_comparison_table(config, figures_dir, logger)
    
    logger.info("\n" + "="*60)
    logger.info(f"ALL FIGURES SAVED TO: {figures_dir}")
    logger.info("="*60)
    
    logger.info("\nGenerated figures:")
    for fig_file in sorted(figures_dir.glob('*.png')):
        logger.info(f"  - {fig_file.name}")


if __name__ == '__main__':
    main()