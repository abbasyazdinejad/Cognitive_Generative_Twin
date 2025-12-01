import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import Config, set_seed
from src.utils.logging import get_logger
from src.utils.device import get_device
from src.models.rl.dashboard_agent import DashboardAdapter
from src.training.trainer_rl import RLTrainer


def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = Config(config_path)
    
    set_seed(config.config['project']['seed'])
    
    device = get_device(config.config['project']['device'])
    
    log_dir = config.paths['experiments'] / 'cgt_full' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('rl_training', log_dir)
    
    logger.info("="*60)
    logger.info("TRAINING RL DASHBOARD ADAPTER")
    logger.info("="*60)
    
    logger.info("\n--- Creating RL Agent ---")
    rl_agent = DashboardAdapter(config.config)
    
    policy_params = sum(p.numel() for p in rl_agent.policy_net.parameters() if p.requires_grad)
    value_params = sum(p.numel() for p in rl_agent.value_net.parameters() if p.requires_grad)
    total_params = policy_params + value_params
    
    logger.info(f"Policy Network Parameters: {policy_params:,}")
    logger.info(f"Value Network Parameters: {value_params:,}")
    logger.info(f"Total Parameters: {total_params:,}")
    
    logger.info("\n--- Training Configuration ---")
    rl_config = config.config['training']['rl']
    logger.info(f"Episodes: {rl_config['episodes']}")
    logger.info(f"Timesteps per Episode: {rl_config['timesteps_per_episode']}")
    logger.info(f"Update Frequency: {rl_config['update_frequency']}")
    logger.info(f"Buffer Size: {rl_config['buffer_size']}")
    logger.info(f"Batch Size: {rl_config['batch_size']}")
    
    logger.info("\n--- Starting RL Training ---")
    
    trainer = RLTrainer(rl_agent, config.config, device)
    
    history = trainer.train()
    
    logger.info("\n--- Training Complete ---")
    
    avg_reward_last_100 = np.mean(history['episode_rewards'][-100:])
    logger.info(f"Average Reward (last 100 episodes): {avg_reward_last_100:.3f}")
    
    logger.info("\n--- Plotting Training Curves ---")
    figures_dir = config.paths['results'] / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['episode_rewards'], alpha=0.6, linewidth=0.5)
    
    window = 50
    if len(history['episode_rewards']) >= window:
        smoothed_rewards = np.convolve(
            history['episode_rewards'],
            np.ones(window)/window,
            mode='valid'
        )
        axes[0, 0].plot(range(window-1, len(history['episode_rewards'])),
                       smoothed_rewards, linewidth=2, color='red',
                       label=f'Moving Average ({window})')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['episode_steps'], linewidth=1)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    non_zero_policy_losses = [loss for loss in history['policy_losses'] if loss > 0]
    if non_zero_policy_losses:
        axes[1, 0].plot(non_zero_policy_losses, linewidth=1)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    non_zero_value_losses = [loss for loss in history['value_losses'] if loss > 0]
    if non_zero_value_losses:
        axes[1, 1].plot(non_zero_value_losses, linewidth=1, color='orange')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'rl_training_curves.png', dpi=300, bbox_inches='tight')
    logger.info(f"RL training curves saved to: {figures_dir / 'rl_training_curves.png'}")
    
    plt.close()
    
    logger.info("\n--- Saving Training History ---")
    import json
    
    history_serializable = {
        'episode_rewards': [float(r) for r in history['episode_rewards']],
        'episode_steps': [int(s) for s in history['episode_steps']],
        'policy_losses': [float(l) for l in history['policy_losses']],
        'value_losses': [float(l) for l in history['value_losses']],
        'entropies': [float(e) for e in history['entropies']]
    }
    
    history_path = config.paths['results'] / 'rl_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    logger.info(f"RL training history saved to: {history_path}")
    
    logger.info("\n--- Computing Statistics ---")
    logger.info(f"Total Episodes: {len(history['episode_rewards'])}")
    logger.info(f"Mean Reward: {np.mean(history['episode_rewards']):.3f}")
    logger.info(f"Std Reward: {np.std(history['episode_rewards']):.3f}")
    logger.info(f"Max Reward: {np.max(history['episode_rewards']):.3f}")
    logger.info(f"Min Reward: {np.min(history['episode_rewards']):.3f}")
    
    logger.info("\n" + "="*60)
    logger.info("RL TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()