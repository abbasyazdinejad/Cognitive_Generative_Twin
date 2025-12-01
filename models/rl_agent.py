import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DashboardAdapter:
    def __init__(self, config):
        self.config = config
        
        latent_dim = config['model']['cvae']['latent_dim']
        state_dim = latent_dim + 1
        
        self.action_space = {
            'alert_grouping': [0, 1, 2],
            'viz_density': [0, 1, 2],
            'prioritization': [0, 1, 2]
        }
        
        action_dim = len(self.action_space['alert_grouping']) * len(self.action_space['viz_density']) * len(self.action_space['prioritization'])
        
        hidden_dim = config['model']['rl']['hidden_dim']
        self.learning_rate = config['model']['rl']['learning_rate']
        self.gamma = config['model']['rl']['gamma']
        self.gae_lambda = config['model']['rl']['gae_lambda']
        self.clip_epsilon = config['model']['rl']['clip_epsilon']
        self.value_coef = config['model']['rl']['value_coef']
        self.entropy_coef = config['model']['rl']['entropy_coef']
        
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        
        self.memory = deque(maxlen=10000)
        
        self.action_dim = action_dim
    
    def _state_from_latent(self, z, cognitive_load):
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
    
    def _decode_action(self, action_idx):
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
    
    def select_action(self, z, cognitive_load, deterministic=False):
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
    
    def compute_reward(self, action, accuracy, workload, situational_awareness):
        alpha1 = 1.0
        alpha2 = 0.5
        alpha3 = 0.3
        
        reward = alpha1 * accuracy - alpha2 * workload + alpha3 * situational_awareness
        
        return reward
    
    def store_transition(self, state, action_idx, reward, next_state, done, action_probs):
        self.memory.append({
            'state': state,
            'action': action_idx,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'action_probs': action_probs
        })
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        
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
    
    def update(self, batch_size=64, epochs=10):
        if len(self.memory) < batch_size:
            return
        
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
        
        for _ in range(epochs):
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
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }