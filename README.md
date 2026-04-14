# Macroeconomic World Model (V-M-C Framework)

An institutional-grade macroeconomic world model designed to simulate stochastic market "dreams" and train reinforcement learning agents for dynamic, risk-adjusted portfolio allocation.

## 🏗 Architecture (V-M-C Iteration 3)

The system is built on a high-fidelity three-tier neural architecture optimized for professional-grade quantitative simulation:

1.  **Vision (V-Model)**: A **$\beta$-Variational Autoencoder ($\beta$-VAE)**.
    *   **Input (7D):** SPY (Equity), VUSTX (Bonds), ^VIX (Volatility), FEDFUNDS (Rates), CPI (Inflation), T10Y2Y (Yield Curve), BAMLH0A0HYM2 (Credit Stress).
    *   **Compression:** Compresses the 7D macroeconomic state into a **4D Latent Space** representing distinct "Economic Regimes."
    *   **$\beta$-Scheduling:** Uses monotonic KL-annealing to prevent posterior collapse and ensure a structured latent representation.

2.  **Memory (M-Model)**: A **Stochastic Recurrent State Space Model (RSSM)**.
    *   **Mechanism:** GRU-based probabilistic engine with **Hidden State Persistence** across timesteps.
    *   **Stochasticity:** Instead of a point prediction, it outputs a Gaussian distribution ($\mu, \sigma^2$) for the next latent state.
    *   **Grounding Noise:** Injects 0.05 Gaussian noise into state transitions to prevent unphysical "dream" drift.
    *   **Dreaming:** Enables **Monte Carlo Hallucinations** of synthetic economic futures by autoregressively sampling from predicted distributions.

3.  **Controller (C-Model)**: A **State-Aware PPO Reinforcement Learning Agent**.
    *   **Proprioception (5D Obs):** The agent sees the 4D latent economic state *plus* its own current portfolio allocation, allowing it to internalize the cost of movement.
    *   **Unit-Correct Friction:** Implements the **Square Root Law of Market Impact** with a 0.05 multiplier, ensuring transaction costs are mathematically significant in Z-score space.
    *   **Mean-Variance Reward:** Trained with a continuous, differentiable reward function: $Reward = Net\_Return - (2.0 \cdot Net\_Return^2)$ to stabilize policy gradients and penalize volatility.

## 🚀 Execution Pipeline

The project is designed to be run sequentially:

1.  **Causal Data Ingestion**: Fetch 30+ years of daily data, apply publication lags (45 days for CPI, 1 day for rates), and perform **Absolute Causal Rolling Z-Score** normalization (252-day window, shifted by 1 day to eliminate look-ahead bias).
    ```bash
    PYTHONPATH=. python src/data/fetcher.py
    ```
2.  **Regime Compression**: Train the $\beta$-VAE to map the 7D macro economy into 4D latent regimes.
    ```bash
    PYTHONPATH=. python src/train_vae.py
    ```
3.  **Physics Engine Training**: Train the Stochastic RSSM to simulate forward transitions in the latent space using Gaussian NLL loss.
    ```bash
    PYTHONPATH=. python src/train_lstm.py
    ```
4.  **Agent Training**: Train the RL agent inside the probabilistic RSSM simulation for **5,000,000 timesteps** with a 60-day horizon and checkpointing.
    ```bash
    PYTHONPATH=. python src/train_agent.py
    ```
5.  **Institutional Evaluation**: Deploy the agent into a real-world **Out-of-Sample (OOS) Firewall** (2019–Present) and benchmark against a passive 60/40 portfolio.
    ```bash
    PYTHONPATH=. python src/evaluate.py
    ```

## 📊 Outputs
- `models/`: Trained weights for the VAE (`vae_weights.pth`), RSSM (`lstm_weights.pth`), and PPO Agent (`ppo_portfolio.zip`).
- `models/checkpoints/`: Model weights saved every 500,000 steps during training.
- `portfolio_evaluation_oos.png`: A professional 2-panel chart comparing the Agent's cumulative returns (net of fees) against a 60/40 benchmark, alongside its dynamic equity exposure over the OOS period.
- `src/data/latent_plot.png`: Visualization of the 4D economic regimes projected into 2D space.

## 🛠 Tech Stack
- **Deep Learning**: PyTorch (Optimized with DMA transfers and asynchronous memory pinning)
- **Reinforcement Learning**: Gymnasium, Stable-Baselines3 (PPO)
- **Data**: yfinance, pandas, FRED API
- **Visualization**: Matplotlib
