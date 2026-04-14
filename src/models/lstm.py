import torch
import torch.nn as nn
from typing import Tuple

class StochasticRSSM(nn.Module):
    """
    A stochastic Recurrent State Space Model (RSSM) designed to model the 
    macroeconomy's forward transition dynamics as a probability distribution.
    Updated for 4D 'Mood' latent space.
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, output_size: int = 4):
        super(StochasticRSSM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # nn.GRU is more parameter-efficient than LSTM for 6GB VRAM hardware
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Branching heads for stochastic modeling
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logvar = nn.Linear(hidden_size, output_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Sequence of latent vectors, shape (batch, seq_len, 4)
            h (torch.Tensor): Initial hidden state
        Returns:
            Tuple: (mu, logvar, next_h) for the next latent state
        """
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, next_h = self.gru(x, h)
        
        # Final time step hidden state
        h_last = out[:, -1, :]
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        
        return mu, logvar, next_h

    def hallucinate(self, z_start: torch.Tensor, steps: int, noise_scale: float = 0.05) -> torch.Tensor:
        """
        Autoregressively generate synthetic economic futures (Monte Carlo hallucination).
        Includes small 'grounding noise' to prevent drifting into unphysical extreme states.
        """
        self.eval()
        hallucinated_sequence = []
        current_seq = z_start.clone()
        
        # Initial hidden state for the recursive loop
        h = None
        
        with torch.no_grad():
            for _ in range(steps):
                mu, logvar, h = self.forward(current_seq, h)
                
                # Sample next state z_{t+1} ~ N(mu, sigma^2)
                z_next = self.reparameterize(mu, logvar) # (1, 4)
                
                # Inject grounding noise (Elite Addition)
                z_next += torch.randn_like(z_next) * noise_scale
                
                hallucinated_sequence.append(z_next.squeeze(0))
                
                # Update sequence for next step (autoregressive)
                current_seq = z_next.unsqueeze(1) # (1, 1, 4)
        
        return torch.stack(hallucinated_sequence)
