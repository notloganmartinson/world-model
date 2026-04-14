import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env.portfolio_env import PortfolioEnv
import os

def train_agent():
    print("--- Phase 4: Controller (Agent Training) ---")
    
    # 1. Instantiate the dreamed environment
    # Every step uses LSTM prediction and VAE decoding
    env = PortfolioEnv(max_steps=252) # episode lasts 1 trading year (252 steps)

    # 2. Setup PPO Agent
    # Updated to use 'cuda' if available for 5M step scaling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Agent on: {device}")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        ent_coef=0.01,
        batch_size=64,
        device=device 
    )

    # 3. Scale Compute with Checkpointing
    total_timesteps = 5000000
    os.makedirs("models/checkpoints", exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_portfolio_model"
    )

    print(f"Training PPO Agent for {total_timesteps} steps with checkpointing...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # 4. Save the trained agent for evaluation/deployment
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_portfolio")
    print("\n[SUCCESS] Agent saved to models/ppo_portfolio.zip")

if __name__ == "__main__":
    train_agent()
