# Macroeconomic World Model (V-M-C Framework)

A production-grade macroeconomic world model designed to simulate stochastic market "dreams" and train reinforcement learning agents for dynamic, risk-adjusted portfolio allocation.

## 🏗 Architecture (V-M-C Iteration 2)

The system is built on a high-fidelity three-tier neural architecture optimized for a 6GB VRAM hardware ceiling:

1.  **Vision (V-Model)**: A **$\beta$-Variational Autoencoder ($\beta$-VAE)**.
    *   **Input (7D):** SPY (Equity), VUSTX (Bonds), ^VIX (Volatility), FEDFUNDS (Rates), CPI (Inflation), T10Y2Y (Yield Curve), BAMLH0A0HYM2 (Credit Stress).
    *   **Compression:** Compresses the 7D macroeconomic state into an **8D Latent Space** representing distinct "Economic Regimes."
    *   **$\beta$-Scheduling:** Uses monotonic KL-annealing to prevent posterior collapse and ensure a structured latent representation.

2.  **Memory (M-Model)**: A **Stochastic Recurrent State Space Model (RSSM)**.
    *   **Mechanism:** Replaced deterministic LSTM with a **GRU-based** probabilistic engine.
    *   **Stochasticity:** Instead of a point prediction, it outputs a Gaussian distribution ($\mu, \sigma^2$) for the next latent state.
    *   **Dreaming:** Enables **Monte Carlo Hallucinations** of future economic futures by autoregressively sampling from predicted distributions.

3.  **Controller (C-Model)**: A **PPO Reinforcement Learning Agent**.
    *   **Friction:** Operates under institutional trading constraints, including a **15bps transaction cost** on all turnover.
    *   **Risk-Aversion:** Trained with an asymmetrical reward function (Risk Aversion = 2.0) to prioritize drawdown protection during volatile "dreams."

## 🚀 Execution Pipeline

The project is designed to be run sequentially:

1.  **Causal Data Ingestion**: Fetch 30+ years of daily data, apply publication lags (45 days for CPI, 1 day for rates), and perform **Causal Rolling Z-Score** normalization (252-day window).
    ```bash
    PYTHONPATH=. python src/data/fetcher.py
    ```
2.  **Regime Compression**: Train the $\beta$-VAE to map the 7D macro economy into 8D latent regimes.
    ```bash
    PYTHONPATH=. python src/train_vae.py
    ```
3.  **Physics Engine Training**: Train the Stochastic RSSM to simulate forward transitions in the latent space using Gaussian NLL loss.
    ```bash
    PYTHONPATH=. python src/train_lstm.py
    ```
4.  **Agent Training**: Train the RL agent inside the probabilistic RSSM simulation (250k timesteps).
    ```bash
    PYTHONPATH=. python src/train_agent.py
    ```
5.  **Monte Carlo Evaluation**: Deploy the agent into a 1-year forward hallucination and benchmark against a passive 60/40 portfolio.
    ```bash
    PYTHONPATH=. python src/evaluate.py
    ```

## 📊 Outputs
- `models/`: Trained weights for the VAE (`vae_weights.pth`), RSSM (`lstm_weights.pth`), and PPO Agent (`ppo_portfolio.zip`).
- `portfolio_evaluation.png`: A professional 2-panel chart comparing the Agent's cumulative returns (net of fees) against a 60/40 benchmark, alongside its dynamic equity exposure over a 1-year hallucination.
- `src/data/latent_plot.png`: Visualization of the 8D economic regimes projected into 2D space.

## 🛠 Tech Stack
- **Deep Learning**: PyTorch (Optimized with DMA transfers and asynchronous memory pinning)
- **Reinforcement Learning**: Gymnasium, Stable-Baselines3 (PPO)
- **Data**: yfinance, pandas, FRED API
- **Visualization**: Matplotlib
