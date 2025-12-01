import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .encoders import CPSEncoder, BioEncoder, BehaviorEncoder
from .generative.cvae import ConditionalVAE
from .generative.diffusion import DiffusionModel
from .llm.reasoning import LLMReasoningModule
from .llm.cognitive_load import CognitiveLoadEstimator
from .rl.dashboard_agent import DashboardAdapter


class CognitiveGenerativeTwin(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.cps_encoder = CPSEncoder(config).to(device)
        self.bio_encoder = BioEncoder(config).to(device)
        self.beh_encoder = BehaviorEncoder(config).to(device)
        
        self.cvae = ConditionalVAE(config).to(device)
        self.diffusion = DiffusionModel(config).to(device)
        
        self.llm_module = LLMReasoningModule(config)
        self.cognitive_load_estimator = CognitiveLoadEstimator(config)
        self.rl_agent = DashboardAdapter(config)
    
    def encode(
        self,
        cps_data: torch.Tensor,
        bio_data: torch.Tensor,
        beh_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        h_cps = self.cps_encoder(cps_data)
        h_bio = self.bio_encoder(bio_data)
        h_beh = self.beh_encoder(beh_data)
        
        return h_cps, h_bio, h_beh
    
    def generate_latent(
        self,
        h_cps: torch.Tensor,
        h_bio: torch.Tensor,
        h_beh: torch.Tensor,
        severity: torch.Tensor,
        stress: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        mu, logvar = self.cvae.encode(h_cps, h_bio, h_beh, severity, stress)
        z = self.cvae.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def refine_latent(self, z: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        z_refined = self.diffusion.sample(z, num_steps)
        return z_refined
    
    def decode_latent(
        self,
        z: torch.Tensor,
        severity: torch.Tensor,
        stress: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        h_cps_recon, h_bio_recon, h_beh_recon = self.cvae.decode(z, severity, stress)
        return h_cps_recon, h_bio_recon, h_beh_recon
    
    def forward(
        self,
        cps_data: torch.Tensor,
        bio_data: torch.Tensor,
        beh_data: torch.Tensor,
        severity: torch.Tensor,
        stress: torch.Tensor,
        use_diffusion: bool = True,
        generate_explanations: bool = False,
        adapt_dashboard: bool = False
    ) -> Dict:
        
        self.cps_encoder.eval()
        self.bio_encoder.eval()
        self.beh_encoder.eval()
        self.cvae.eval()
        self.diffusion.eval()
        
        with torch.no_grad():
            h_cps, h_bio, h_beh = self.encode(cps_data, bio_data, beh_data)
            
            z, mu, logvar = self.generate_latent(h_cps, h_bio, h_beh, severity, stress)
            
            if use_diffusion:
                z_refined = self.refine_latent(z)
            else:
                z_refined = z
            
            h_cps_recon, h_bio_recon, h_beh_recon = self.decode_latent(z_refined, severity, stress)
            
            cognitive_load = self.cognitive_load_estimator.estimate_load(z_refined, stress)
        
        output = {
            'latent': z_refined,
            'latent_raw': z,
            'mu': mu,
            'logvar': logvar,
            'encodings': {
                'cps': h_cps,
                'bio': h_bio,
                'beh': h_beh
            },
            'reconstructions': {
                'cps': h_cps_recon,
                'bio': h_bio_recon,
                'beh': h_beh_recon
            },
            'cognitive_load': cognitive_load
        }
        
        if generate_explanations:
            metadata = {'t_event': 0, 'attack_type': 'unknown'}
            
            rationale = self.llm_module.generate_rationale(
                z_refined[0], int(severity[0].item()), int(stress[0].item()), metadata
            )
            
            sev_new = max(0, int(severity[0].item()) - 1)
            stress_new = max(0, int(stress[0].item()) - 1)
            
            counterfactual = self.llm_module.generate_counterfactual(
                z_refined[0],
                int(severity[0].item()),
                sev_new,
                int(stress[0].item()),
                stress_new
            )
            
            output['explanations'] = {
                'rationale': rationale,
                'counterfactual': counterfactual
            }
        
        if adapt_dashboard:
            dashboard_action, action_idx, action_probs = self.rl_agent.select_action(
                z_refined[0], float(cognitive_load[0].item()), deterministic=True
            )
            
            output['dashboard_action'] = dashboard_action
            output['action_idx'] = action_idx
            output['action_probs'] = action_probs
        
        return output
    
    def train_step(
        self,
        cps_data: torch.Tensor,
        bio_data: torch.Tensor,
        beh_data: torch.Tensor,
        severity: torch.Tensor,
        stress: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        h_cps, h_bio, h_beh = self.encode(cps_data, bio_data, beh_data)
        
        h_cps_recon, h_bio_recon, h_beh_recon, mu, logvar, z = self.cvae(
            h_cps, h_bio, h_beh, severity, stress
        )
        
        cvae_loss, recon_loss, kld_loss = self.cvae.loss_function(
            h_cps, h_bio, h_beh,
            h_cps_recon, h_bio_recon, h_beh_recon,
            mu, logvar
        )
        
        diffusion_loss = self.diffusion.loss_function(z)
        
        total_loss = cvae_loss + diffusion_loss
        
        return {
            'total_loss': total_loss,
            'cvae_loss': cvae_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'diffusion_loss': diffusion_loss,
            'latent': z
        }
    
    def save(self, path: str):
        torch.save({
            'cps_encoder': self.cps_encoder.state_dict(),
            'bio_encoder': self.bio_encoder.state_dict(),
            'beh_encoder': self.beh_encoder.state_dict(),
            'cvae': self.cvae.state_dict(),
            'diffusion': self.diffusion.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.cps_encoder.load_state_dict(checkpoint['cps_encoder'])
        self.bio_encoder.load_state_dict(checkpoint['bio_encoder'])
        self.beh_encoder.load_state_dict(checkpoint['beh_encoder'])
        self.cvae.load_state_dict(checkpoint['cvae'])
        self.diffusion.load_state_dict(checkpoint['diffusion'])