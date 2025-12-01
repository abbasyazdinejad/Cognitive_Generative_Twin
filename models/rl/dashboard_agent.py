import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_norm1(self.fc1(state)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_norm1(self.fc1(state)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        value = self.fc3(x)
        return value


class DashboardAdapter:
    def __init__(self, config):
        self.config = config
        
        rl_config = config['model']['rl']
        
        latent_dim = config['model']['cvae']['latent_dim']
        state_dim = latent_dim + 1
        
        self.action_space = {
            'alert_grouping': [0, 1, 2],
            'viz_density': [0, 1, 2],
            'prioritization': [0, 1, 2]
        }
        
        action_dim = (
            len(self.action_space['alert_grouping']) *
            len(self.action_space['viz_density']) *
            len(self.action_space['prioritization'])
        )
        
        hidden_dim = rl_config['hidden_dim']
        self.learning_rate = rl_config['learning_rate']
        self.gamma = rl_config['gamma']
        self.gae_lambda = rl_config['gae_lambda']
        self.clip_epsilon = rl_config['clip_epsilon']
        self.value_coef = rl_config['value_coef']
        self.entropy_coef = rl_config['entropy_coef']
        
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.learning_rate
        )
        
        self.memory = deque(maxlen=config['training']['rl']['buffer_size'])
        
        self.action_dim = action_dim
        self.state_dim = state_dim
    
    def _state_from_latent(
        self,
        z: torch.Tensor,
        cognitive_load: float
    ) -> torch.Tensor:
        
        if isinstance(z, torch.Tensor):
            z_flat = z.view(-1)
        else:
            z_flat = torch.FloatTensor(z).view(-1)
        
        if isinstance(cognitive_load, torch.Tensor):
            load_tensor = cognitive_load.view(-1)
        else:
            load_tensor = torch.FloatTensor([cognitive_load])
        
        state = torch.cat([z_flat, load_tensor])
        
        return state
    
    def _decode_action(self, action_idx: int) -> Dict[str, int]:
        num_alert = len(self.action_space['alert_grouping'])
        num_viz = len(self.action_space['viz_density'])
        num_prior = len(self.action_space['prioritization'])
        
        alert_idx = action_idx // (num_viz * num_prior)
        viz_idx = (action_idx % (num_viz * num_prior)) // num_prior
        prior_idx = action_idx % num_prior
        
        return {
            'alert_grouping': self.action_space['alert_grouping'][alert_idx],
            'viz_density': self.action_space['viz_density'][viz_idx],
            'prioritization': self.action_space['prioritization'][prior_idx]
        }
    
    def select_action(
        self,
        z: torch.Tensor,
        cognitive_load: float,
        deterministic: bool = False
    ) -> Tuple[Dict, int, torch.Tensor]:
        
        state = self._state_from_latent(z, cognitive_load)
        
        with torch.no_grad():
            action_probs = self.policy_net(state.unsqueeze(0))
        
        if deterministic:
            action_idx = torch.argmax(action_probs).item()
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()
        
        action = self._decode_action(action_idx)
        
        return action, action_idx, action_probs.squeeze()
    
    def compute_reward(
        self,
        action: Dict,
        accuracy: float,
        workload: float,
        situational_awareness: float
    ) -> float:
        
        alpha1 = 1.0
        alpha2 = 0.5
        alpha3 = 0.3
        
        reward = alpha1 * accuracy - alpha2 * workload + alpha3 * situational_awareness
        
        return reward
    
    def store_transition(
        self,
        state: torch.Tensor,
        action_idx: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        action_probs: torch.Tensor
    ):
        
        self.memory.append({
            'state': state,
            'action': action_idx,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'action_probs': action_probs
        })
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        advantages = []
        gae = 0
        
        rewards = rewards.cpu()
        values = values.cpu()
        next_values = next_values.cpu()
        dones = dones.cpu()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[-1]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, float]:
        
        if batch_size is None:
            batch_size = self.config['training']['rl']['batch_size']
        
        if epochs is None:
            epochs = self.config['training']['rl']['train_epochs']
        
        if len(self.memory) < batch_size:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            }
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.stack([item['state'] for item in batch])
        actions = torch.LongTensor([item['action'] for item in batch])
        rewards = torch.FloatTensor([item['reward'] for item in batch])
        next_states = torch.stack([item['next_state'] for item in batch])
        dones = torch.FloatTensor([item['done'] for item in batch])
        old_action_probs = torch.stack([item['action_probs'] for item in batch])
        
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
        
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(epochs):
            action_probs = self.policy_net(states)
            values_pred = self.value_net(states).squeeze()
            
            action_probs_selected = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            old_action_probs_selected = old_action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            ratio = action_probs_selected / (old_action_probs_selected + 1e-8)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values_pred, returns)
            
            dist = torch.distributions.Categorical(action_probs)
            entropy = dist.entropy().mean()
            
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        return {
            'policy_loss': total_policy_loss / epochs,
            'value_loss': total_value_loss / epochs,
            'entropy': total_entropy / epochs
        }
    
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])