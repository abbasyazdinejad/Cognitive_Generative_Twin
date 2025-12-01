import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from ..models.rl.environment import SOCEnvironment
from ..models.rl.dashboard_agent import DashboardAdapter


class RLTrainer:
    def __init__(self, rl_agent: DashboardAdapter, config, device):
        self.rl_agent = rl_agent
        self.config = config
        self.device = device
        
        self.env = SOCEnvironment(config)
        
        rl_config = config['training']['rl']
        self.episodes = rl_config['episodes']
        self.timesteps_per_episode = rl_config['timesteps_per_episode']
        self.update_frequency = rl_config['update_frequency']
        
        self.checkpoint_dir = Path(config['project']['exp_dir']) / 'rl_checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_latent_sample(self, batch_size: int = 1):
        latent_dim = self.config['model']['cvae']['latent_dim']
        z = torch.randn(batch_size, latent_dim)
        return z
    
    def train_episode(self, episode: int) -> Dict[str, float]:
        env_state = self.env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(self.timesteps_per_episode):
            z = self.generate_latent_sample(1)[0]
            cognitive_load = np.random.uniform(0.3, 0.9)
            
            state = self.rl_agent._state_from_latent(z, cognitive_load)
            
            action, action_idx, action_probs = self.rl_agent.select_action(
                z, cognitive_load, deterministic=False
            )
            
            next_env_state, reward, done, info = self.env.step(action, cognitive_load)
            
            z_next = self.generate_latent_sample(1)[0]
            cognitive_load_next = np.random.uniform(0.3, 0.9)
            next_state = self.rl_agent._state_from_latent(z_next, cognitive_load_next)
            
            self.rl_agent.store_transition(
                state, action_idx, reward, next_state, done, action_probs
            )
            
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        if episode % self.update_frequency == 0:
            update_metrics = self.rl_agent.update()
        else:
            update_metrics = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            }
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'policy_loss': update_metrics['policy_loss'],
            'value_loss': update_metrics['value_loss'],
            'entropy': update_metrics['entropy']
        }
    
    def train(self):
        history = {
            'episode_rewards': [],
            'episode_steps': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        best_reward = float('-inf')
        
        pbar = tqdm(range(1, self.episodes + 1), desc='RL Training')
        
        for episode in pbar:
            metrics = self.train_episode(episode)
            
            history['episode_rewards'].append(metrics['episode_reward'])
            history['episode_steps'].append(metrics['episode_steps'])
            history['policy_losses'].append(metrics['policy_loss'])
            history['value_losses'].append(metrics['value_loss'])
            history['entropies'].append(metrics['entropy'])
            
            pbar.set_postfix({
                'reward': f"{metrics['episode_reward']:.3f}",
                'steps': metrics['episode_steps']
            })
            
            if episode % 10 == 0:
                avg_reward = np.mean(history['episode_rewards'][-10:])
                print(f"\nEpisode {episode}: Avg Reward (last 10): {avg_reward:.3f}")
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_checkpoint('best')
                    print(f"✓ Best RL agent saved with avg_reward: {best_reward:.3f}")
            
            if episode % 50 == 0:
                self.save_checkpoint(f'episode_{episode}')
        
        return history
    
    def save_checkpoint(self, name: str):
        checkpoint_path = self.checkpoint_dir / f'rl_agent_{name}.pt'
        self.rl_agent.save(str(checkpoint_path))
    
    def load_checkpoint(self, path: str):
        self.rl_agent.load(path)