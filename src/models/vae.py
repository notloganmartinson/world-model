import torch
import torch.nn as nn
from typing import Tuple, Dict

class VAE(nn.Module):
    """
    A production-grade Variational Autoencoder (VAE) for compressing macroeconomic states.
    Updated for 7D input and 4D latent space for institutional-grade compression.
    """
    def __init__(self, input_dim=7, hidden_dim=128, latent_dim=4):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) while maintaining backprop."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z) -> torch.Tensor:
        """Maps latent vector back to reconstructed input."""
        return self.decoder(z)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: Encode -> Reparameterize -> Decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0) -> Dict[str, torch.Tensor]:
    """
    Computes the beta-VAE loss.
    Loss = Recon_Loss + (beta * KLD_Loss)
    """
    # MSE for reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + (beta * kld_loss)
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kld_loss': kld_loss
    }
