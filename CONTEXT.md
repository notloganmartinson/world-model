# PROJECT CONTEXT: Macroeconomic World Model (Institutional POC)

## 1. THE CORE MISSION
Upgrade the V-M-C (Vision-Memory-Controller) architecture to institutional standards. We have successfully implemented a 7D causal data pipeline, an 8D $\beta$-VAE, a stochastic RSSM for Monte Carlo hallucination, and a PPO agent penalized by 15bps transaction friction. 

*The current objective is to enforce Out-of-Sample (OOS) backtesting rigor and implement non-linear market impact before initiating a 5,000,000 timestep compute scaling run.*

## 2. CURRENT FOCUS: OOS FIREWALL & ADVANCED FRICTION
We must update the codebase to ensure the agent is completely blinded to post-2018 data during training, and we must make the trading environment brutally realistic.

**Task 1: The OOS Data Firewall (`src/data/fetcher.py` & Training Scripts)**
* Introduce a strict chronological split. 
* **Training Set:** 1993-01-01 to 2018-12-31. All training scripts (`train_vae.py`, `train_lstm.py`, `train_agent.py`) must ONLY ingest this data.
* **OOS Test Set:** 2019-01-01 to Present. This data is strictly quarantined for the final `evaluate.py` backtest.
* Ensure the 252-day Rolling Z-Score normalization does not leak data across the 2018/2019 boundary.

**Task 2: Non-Linear Market Impact (`src/env/portfolio_env.py`)**
* Flat 15bps transaction costs are unrealistic for institutional block trades.
* Implement the Square Root Law of Market Impact. The slippage penalty must scale non-linearly with the turnover. 
* Formula concept: `slippage = base_bps * sqrt(turnover)`. Small allocation adjustments should incur minimal friction; dumping 100% of the portfolio in a single day should trigger massive slippage penalties.

**Task 3: Compute Scaling (`src/train_agent.py`)**
* Update the PPO agent to train for `5,000,000` timesteps. 
* Implement Checkpointing (`stable_baselines3.common.callbacks.CheckpointCallback`) to save the agent's weights every 500,000 steps, preventing total loss if the server crashes during the massive run.

## 3. ENGINEERING CONSTRAINTS & RULES
* **No Data Leakage:** The 2019-2024 data must not touch the VAE, the RSSM, or the RL Agent during training.
* **Code Quality:** Production-ready PyTorch. Optimize for a 6GB VRAM ceiling. Do not hallucinate external libraries outside of the current stack.
