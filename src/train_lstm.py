import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.lstm import StochasticRSSM
import os
from typing import Tuple

def create_sliding_window_dataset(data: torch.Tensor, window_size: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms a single sequence into a dataset of windows and targets.
    Example: WindowSize=30, Input=latent_data[0:30], Target=latent_data[30]
    """
    inputs = []
    targets = []
    # Loop until the target (i+window_size) would exceed data length
    for i in range(len(data) - window_size):
        inputs.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return torch.stack(inputs), torch.stack(targets)

def train_lstm():
    # 1. Setup Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    print(f"Training M-Model (Stochastic RSSM) on: {device}")

    # 2. Load Latent Data (8D)
    latent_path = "src/data/latent_economy.pt"
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent data not found at {latent_path}. Ensure train_vae.py has run.")
    
    latent_data = torch.load(latent_path, map_location=device)
    print(f"Loaded latent economy data: {latent_data.shape}")

    # 3. Create Sliding Window Dataset (30 days window)
    window_size = 30
    X, y = create_sliding_window_dataset(latent_data, window_size)
    print(f"Window dataset ready. Input shape: {X.shape}, Target shape: {y.shape}")

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 4. Initialize Model & Optimizer
    # Updated: input_size=4, hidden_size=128, num_layers=2, output_size=4
    model = StochasticRSSM(input_size=4, hidden_size=128, num_layers=2, output_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop
    epochs = 100
    model.train()
    print(f"Starting M-Model training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        batch_losses = []
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: get predicted distribution
            mu, logvar, _ = model(batch_X)
            
            # Gaussian Negative Log-Likelihood (NLL) Loss
            # loss = 0.5 * sum(logvar + ((z_target - mu)^2 / exp(logvar)))
            loss = 0.5 * torch.sum(logvar + ((batch_y - mu).pow(2)) / torch.exp(logvar))
            
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients during BPTT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        # Logging every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = sum(batch_losses) / len(batch_losses)
            print(f"Epoch {epoch:3d} | Gaussian NLL Loss: {avg_loss:.6f}")

    # 6. Save Model Weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_weights.pth")
    print("\nTraining Complete. Model saved to models/lstm_weights.pth")

if __name__ == "__main__":
    train_lstm()
