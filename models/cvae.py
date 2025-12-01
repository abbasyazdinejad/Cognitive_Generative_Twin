import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.latent_dim = config['model']['cvae']['latent_dim']
        self.condition_dim = config['model']['cvae']['condition_dim']
        self.beta = config['model']['cvae']['beta']
        
        cps_encoder_dim = config['model']['encoder']['cps']['hidden_dim'] * 2
        bio_encoder_dim = config['model']['encoder']['bio']['hidden_dim'] * 2
        beh_encoder_dim = config['model']['encoder']['beh']['hidden_dim'] * 2
        
        encoder_input_dim = cps_encoder_dim + bio_encoder_dim + beh_encoder_dim + self.condition_dim * 2
        
        encoder_hidden = config['model']['cvae']['encoder_hidden']
        
        encoder_layers = []
        prev_dim = encoder_input_dim
        for hidden_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_hidden[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden[-1], self.latent_dim)
        
        decoder_hidden = config['model']['cvae']['decoder_hidden']
        decoder_input_dim = self.latent_dim + self.condition_dim * 2
        
        decoder_layers = []
        prev_dim = decoder_input_dim
        for hidden_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.decoder_common = nn.Sequential(*decoder_layers)
        
        self.decoder_cps = nn.Linear(decoder_hidden[-1], cps_encoder_dim)
        self.decoder_bio = nn.Linear(decoder_hidden[-1], bio_encoder_dim)
        self.decoder_beh = nn.Linear(decoder_hidden[-1], beh_encoder_dim)
        
        self.condition_embedding = nn.Embedding(4, self.condition_dim)
    
    def encode(self, h_cps, h_bio, h_beh, severity, stress):
        severity_emb = self.condition_embedding(severity.squeeze())
        stress_emb = self.condition_embedding(stress.squeeze())
        
        x = torch.cat([h_cps, h_bio, h_beh, severity_emb, stress_emb], dim=1)
        
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, severity, stress):
        severity_emb = self.condition_embedding(severity.squeeze())
        stress_emb = self.condition_embedding(stress.squeeze())
        
        x = torch.cat([z, severity_emb, stress_emb], dim=1)
        
        h = self.decoder_common(x)
        
        h_cps_recon = self.decoder_cps(h)
        h_bio_recon = self.decoder_bio(h)
        h_beh_recon = self.decoder_beh(h)
        
        return h_cps_recon, h_bio_recon, h_beh_recon
    
    def forward(self, h_cps, h_bio, h_beh, severity, stress):
        mu, logvar = self.encode(h_cps, h_bio, h_beh, severity, stress)
        z = self.reparameterize(mu, logvar)
        h_cps_recon, h_bio_recon, h_beh_recon = self.decode(z, severity, stress)
        
        return h_cps_recon, h_bio_recon, h_beh_recon, mu, logvar, z
    
    def loss_function(self, h_cps, h_bio, h_beh, h_cps_recon, h_bio_recon, h_beh_recon, mu, logvar):
        recon_loss_cps = F.mse_loss(h_cps_recon, h_cps, reduction='sum')
        recon_loss_bio = F.mse_loss(h_bio_recon, h_bio, reduction='sum')
        recon_loss_beh = F.mse_loss(h_beh_recon, h_beh, reduction='sum')
        
        recon_loss = recon_loss_cps + recon_loss_bio + recon_loss_beh
        
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kld_loss
        
        return total_loss, recon_loss, kld_loss