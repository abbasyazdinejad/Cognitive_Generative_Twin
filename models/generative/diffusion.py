import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        diffusion_config = config['model']['diffusion']
        
        self.timesteps = diffusion_config['timesteps']
        self.beta_start = diffusion_config['beta_start']
        self.beta_end = diffusion_config['beta_end']
        self.schedule = diffusion_config['schedule']
        
        if self.schedule == 'linear':
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.schedule == 'cosine':
            steps = torch.linspace(0, self.timesteps, self.timesteps + 1)
            alphas_cumprod = torch.cos(((steps / self.timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        latent_dim = config['model']['cvae']['latent_dim']
        hidden_dim = diffusion_config['hidden_dim']
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t.float().unsqueeze(-1) / self.timesteps)
        
        x = torch.cat([z, t_emb], dim=1)
        noise_pred = self.noise_predictor(x)
        
        return noise_pred
    
    def add_noise(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if noise is None:
            noise = torch.randn_like(z)
        
        device = z.device
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noisy_z = sqrt_alpha_cumprod_t * z + sqrt_one_minus_alpha_cumprod_t * noise
        
        return noisy_z, noise
    
    def ddpm_sample(
        self,
        z: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        
        if num_steps is None:
            num_steps = self.timesteps
        
        device = z.device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        
        z_t = z
        
        for t in reversed(range(num_steps)):
            t_batch = torch.full((z.shape[0],), t, device=device, dtype=torch.long)
            
            noise_pred = self(z_t, t_batch)
            
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            mean = self.sqrt_recip_alphas[t] * (z_t - beta_t / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred)
            
            if t > 0:
                noise = torch.randn_like(z_t)
                variance = self.posterior_variance[t]
                z_t = mean + torch.sqrt(variance) * noise
            else:
                z_t = mean
        
        return z_t
    
    def ddim_sample(
        self,
        z: torch.Tensor,
        num_steps: Optional[int] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        
        if num_steps is None:
            num_steps = self.config['model']['diffusion']['sampling_steps']
        
        device = z.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        z_t = z
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(z.shape[0])
            
            noise_pred = self(z_t, t_batch)
            
            alpha_t = self.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            
            pred_x0 = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred
            
            noise = torch.randn_like(z_t) if i < len(timesteps) - 1 else torch.zeros_like(z_t)
            
            z_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return z_t
    
    def sample(self, z: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        return self.ddim_sample(z, num_steps)
    
    def loss_function(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        
        self.betas = self.betas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        noise = torch.randn_like(z)
        z_noisy, _ = self.add_noise(z, t, noise)
        
        noise_pred = self(z_noisy, t)
        
        loss = F.mse_loss(noise_pred, noise)
        
        return loss