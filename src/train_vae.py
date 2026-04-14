import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.data.fetcher import DataFetcher
from src.models.vae import VAE, vae_loss_function
import os
import numpy as np

def train_vae():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training V-Model on: {device}")

    # 2. Load Real Data
    data_path = "src/data/macro_data_daily_train.pt"
    if not os.path.exists(data_path):
        print("Fetching fresh data and applying firewall...")
        fetcher = DataFetcher()
        df_raw = fetcher.fetch_data()
        df_norm = fetcher.get_normalized_tensor(df_raw)
        # Split but only keep training for this script
        train_data, oos_data = fetcher.split_data(df_norm, split_date="2018-12-31")
        torch.save(train_data, data_path)
        torch.save(oos_data, "src/data/macro_data_daily_oos.pt")
        data = train_data
    else:
        print(f"Loading existing training data from {data_path}")
        data = torch.load(data_path, map_location="cpu") # Initial load to CPU

    # 3. Setup DataLoader with Optimization (pin_memory for DMA transfer)
    is_cuda = device.type == "cuda"
    dataset = TensorDataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=True, 
        pin_memory=is_cuda
    )

    # 4. Initialize Model & Optimizer
    # Updated: input_dim=7, hidden_dim=128, latent_dim=4 (Elite Compression)
    model = VAE(input_dim=7, hidden_dim=128, latent_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop with KL Annealing
    epochs = 100
    annealing_steps = 50
    model.train()
    print(f"Starting beta-VAE training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        total_epoch_loss = 0
        # Linear KL Annealing: Beta starts at 0, increases to 1 by annealing_steps
        beta = min(1.0, (epoch - 1) / (annealing_steps - 1)) if annealing_steps > 1 else 1.0
        
        for batch_idx, (x,) in enumerate(dataloader):
            # non_blocking=True for asynchronous memory transfer
            x = x.to(device, non_blocking=is_cuda)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(x)
            loss_dict = vae_loss_function(recon_batch, x, mu, logvar, beta=beta)
            
            loss = loss_dict['loss']
            loss.backward()
            total_epoch_loss += loss.item()
            optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:3d} | Beta: {beta:.2f} | Avg Loss: {avg_loss:.6f}")

    # 6. Save Model Weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/vae_weights.pth")
    print(f"\nModel weights saved to models/vae_weights.pth")

    # 7. Generate Latent Economy (8D compression)
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        mu, _ = model.encode(data)
        torch.save(mu.cpu(), "src/data/latent_economy.pt")
        print(f"Latent coordinates (Shape: {mu.shape}) saved to src/data/latent_economy.pt")

    # 8. Visualization (Projecting 8D to 2D for monitoring)
    # Using simple slicing or first two principal components (here we just take first 2 dimensions)
    mu_np = mu.cpu().numpy()
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_np[:, 0], mu_np[:, 1], alpha=0.5, s=2, c=range(len(mu_np)), cmap='viridis')
    plt.colorbar(scatter, label='Time progression (Daily Index)')
    plt.title("Latent Economic Space (8D Regimes Projected to 2D)")
    plt.xlabel("Latent Coordinate 1")
    plt.ylabel("Latent Coordinate 2")
    plt.grid(True, alpha=0.3)
    plt.savefig("src/data/latent_plot.png")
    print("Scatter plot of latent economy saved to src/data/latent_plot.png")

if __name__ == "__main__":
    train_vae()
