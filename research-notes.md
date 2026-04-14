# Research Notes: Macroeconomic World Model via V-M-C Architecture
**Project:** Autonomous Portfolio Allocation in Synthetic Latent Environments

## Abstract
This document outlines the architecture, rationale, and theoretical underpinnings of the Macroeconomic World Model project. The objective is to construct a robust Proof-of-Concept (POC) demonstrating advanced agentic AI capabilities in quantitative finance. By moving beyond traditional autoregressive Large Language Models (LLMs) and standard historical backtesting, this project implements a Vision-Memory-Controller (V-M-C) framework to simulate synthetic macroeconomic realities and train a Reinforcement Learning (RL) agent to dynamically optimize a portfolio under shifting economic regimes.

---

## I. Introduction and Theoretical Foundation
The genesis of this architecture maps closely to human cognitive processing—specifically implicit learning and the "Hub and Spoke" model of the brain. Just as the human brain subconsciously compresses multidimensional sensory input into abstract concepts (semantic networks) to predict outcomes, a World Model compresses complex environmental data into a low-dimensional mathematical latent space.

For institutional investors and quantitative analysts, traditional risk assessment relies heavily on historical data regression. However, historical data cannot account for "Black Swan" events or novel economic combinations (e.g., unprecedented inflation coupled with simultaneous tech-sector drawdowns). This architecture solves that limitation by learning the *physics* of the market and generating statistically valid, synthetic future scenarios.

---

## II. Methodology and Architecture (The V-M-C Framework)

### A. Data Engineering & The Forward-Fill Paradigm
**Objective:** Construct a high-volume, normalized dataset representing the U.S. macroeconomy.
* **Original Assets (4D):** SPY (Equity proxy), VUSTX (Bond proxy), FEDFUNDS (Rates), CPIAUCSL (Inflation).
* **Current Expansion (7D):** Added ^VIX (Market Volatility), T10Y2Y (10-Year minus 2-Year Yield Curve Spread), and BAMLH0A0HYM2 (Credit Stress indicator).
* **The Problem:** Deep learning models require massive datasets to avoid overfitting. Monthly macroeconomic data yields less than 300 data points over a 24-year period—mathematically insufficient for a neural network.
* **The Solution:** We adopted a high-resolution, daily-frequency approach. By utilizing VUSTX (established in 1986) instead of TLT (established in 2002), the timeline was extended back to 1993. Monthly macroeconomic indicators (CPI, FEDFUNDS) were *forward-filled* to align with daily trading days.
* **Information Availability (Publication Lags):** To ensure the model reflects the actual information available to a trader at each daily close, we implemented mandatory publication lags for FRED data:
    * **CPIAUCSL (Inflation):** Shifted forward by **45 days** to account for standard BLS release delays.
    * **FEDFUNDS (Interest Rates):** Shifted forward by **1 day** (overnight publication lag).
    * **Market Spreads (T10Y2Y, BAMLH0A0HYM2):** Shifted forward by **1 day** (T+1 reporting).
    * These shifts are applied **before** the final forward-fill (`ffill`), ensuring the agent only "sees" data that would have been publicly available on that date.
* **Result (Iteration 1):** A robust tensor of shape `(8355, 4)`, providing ~8,300 contiguous data points.
* **Normalization (Causal Rolling Z-Score):** To ensure stationarity and prevent look-ahead bias, we replaced global `MinMaxScaler` with a **Causal Rolling Z-Score** normalization.
    * **Window:** 252 trading days (~1 year).
    * **Formula:** $Z_t = (X_t - \mu_{t-252:t}) / \sigma_{t-252:t}$.
    * This ensures that at any point $t$, the normalization only uses information that was historically available, making the features truly stationary for neural network processing without leaking future information.

### B. Vision Model (V-Model): State Compression via VAE
**Objective:** Translate noisy, 7-dimensional market data into a clean, 2-dimensional spatial coordinate system.
* **Implementation:** A Variational Autoencoder (VAE) utilizing the Reparameterization Trick.
* **Rationale:** The VAE forces the data through a bottleneck (the latent space). Instead of memorizing the data, it learns the underlying probability distribution of the economy.
* **Insights Learned:** Upon visualizing the `(8355, 2)` latent coordinates, the "Arrow of Time" emerged organically. The model, without being explicitly programmed with temporal awareness, clustered distinct macroeconomic eras (e.g., the Dot-Com boom, the ZIRP era, post-COVID inflation) into contiguous geometric paths. This proves the economy moves in predictable regime shifts rather than random walks.

### C. Memory Model (M-Model): Autoregressive Simulation via LSTM
**Objective:** Build the "Physics Engine" to predict forward momentum within the latent space.
* **Implementation:** A Long Short-Term Memory (LSTM) network processing 30-day rolling windows.
* **Rationale:** The LSTM serves as the "Dreamer." It analyzes the previous 30 days of latent spatial movement to predict the exact coordinate of Day 31. Because the VAE mapped the economy as a continuous, smooth path, the LSTM achieved an exceptionally low Mean Squared Error (MSE of ~0.00000008).
* **Strategic Value:** Once trained, this model can autoregressively hallucinate endless, synthetic economic timelines that obey the mathematical laws of the real market, completely bypassing the limitations of historical backtesting.

### D. Controller (C-Model): Dynamic Allocation via PPO Agent
**Objective:** Train an autonomous RL agent to navigate the simulated economy.
* **Implementation:** Proximal Policy Optimization (PPO) using `stable-baselines3`, deployed inside a custom `gymnasium.Env`.
* **The Environment:** The custom environment acts as the bridge. At each timestep, the agent receives the current 2D economic coordinate. It outputs a portfolio allocation (0.0 to 1.0, representing the split between Stocks and Bonds).
* **The Reward Function:** The agent is evaluated using a Sharpe Ratio-inspired metric. It is penalized for volatility and drawdowns during simulated recessions, forcing it to learn risk mitigation.
* **Insights Learned:** Utilizing an off-the-shelf PPO algorithm demonstrates enterprise-grade pragmatism. The true Intellectual Property (IP) lies in the custom environment and the World Model's physics engine, not the underlying calculus of the RL optimizer.

---

## III. Key Takeaways & Engineering Philosophy

1. **Tracer Bullet Development:** During the initial data pipeline phase, the temptation to over-optimize feature engineering (e.g., adding MACD or yield curves) was avoided. By pushing a streamlined `(N, 4)` tensor through the entire V-M-C pipeline, we established a complete, modular architecture. Future iterations can scale `input_dim=4` to `input_dim=50` without breaking the core system.
2. **Iterative Scaling (Current):** The data pipeline was expanded from 4 to 7 dimensions (adding VIX, yield spread, and credit stress) and refactored to use logarithmic returns for improved mathematical rigor in asset growth modeling. The V-M-C architecture handled this `input_dim=7` shift seamlessly.
3. **Context Economy:** Using AST (Abstract Syntax Trees) to generate token-optimized repo maps allowed for rapid, agentic development without degrading the LLM's context window during complex architectural builds.
4. **The Nature of Machine "Thought":** The system mimics human intuition (System 1 thinking). Just as a human intuits a concept without words via an abstract constellation of neurons, the VAE maps abstract market conditions into a mathematical constellation. The PPO agent then "acts" on this pre-linguistic, mathematical intuition.

## IV. Strategic Value & Applications
This codebase serves as a **Macroeconomic Stress-Testing Infrastructure**. 

Rather than relying on reactive trading logic, this system compresses global market physics into a generative latent space, allowing quantitative researchers to test portfolio resilience against thousands of synthetic, unrecorded Black Swan events. It demonstrates a shift away from standard predictive analytics toward fully agentic simulation engines.
# Research Notes v2.0: Autonomous Portfolio Allocation in Stochastic Latent Environments

**Project:** Macroeconomic World Model via V-M-C Architecture (Institutional POC)
**Revision:** v2.0 (Transition to Stochastic RSSM & Institutional Friction)

## Abstract
This paper details the architectural evolution of the Macroeconomic World Model from a deterministic, frictionless proof-of-concept to a rigorously causal, stochastic simulation engine. Addressing critical flaws in traditional financial machine learning—namely lookahead bias, deterministic transition assumptions, and frictionless trading environments—this iteration implements a DreamerV3-inspired Vision-Memory-Controller (V-M-C) framework. By utilizing a $ eta$-Variational Autoencoder ($ eta$-VAE) for state compression, a Recurrent State Space Model (RSSM) for stochastic Monte Carlo hallucination, and a friction-penalized Reinforcement Learning (RL) environment, we demonstrate an agent capable of navigating unrecorded, synthetic market regimes while surviving institutional transaction costs.

---

## I. Introduction and the Fallacy of Determinism
The initial iteration of this architecture successfully established the V-M-C pipeline but relied on "toy model" assumptions. Specifically, it suffered from global normalization (lookahead bias) and utilized a deterministic LSTM that predicted exact future coordinate states. In quantitative finance, deterministic point-estimates are epistemically flawed; the market is inherently probabilistic, characterized by dynamic variance and fat-tail risks. 

To bridge the gap between academic exercise and institutional viability, the system required a complete methodological overhaul. The objective shifted from predicting *what* the economy will do, to predicting the *probability distribution* of what the economy *might* do, and forcing an agent to optimize risk-adjusted returns within that uncertainty.

---

## II. Methodological Enhancements

### A. Data Engineering: Causality and State Expansion
**The Problem:** The v1.0 pipeline leaked future information via global `MinMaxScaler` application and the immediate forward-filling of lagging macroeconomic indicators. Furthermore, a 4D state space was insufficient to capture regime-shifting volatility.
**The Solution:**
1.  **State Space Expansion (7D):** We integrated `^VIX` (implied volatility), `T10Y2Y` (yield curve inversion), and `BAMLH0A0HYM2` (high-yield credit spreads) to capture market stress and credit liquidity.
2.  **Mathematical Rigor:** Asset returns (SPY, VUSTX) were converted from simple percentage changes to logarithmic returns ($R_t = \ln(P_t / P_{t-1})$) to ensure statistical additivity.
3.  **Strict Publication Lags:** To ensure strict causality, macroeconomic indicators were delayed to match real-world publication schedules (e.g., CPI lagged by 45 days) before forward-filling.
4.  **Causal Normalization:** Global scaling was entirely replaced with a 252-trading-day Rolling Z-Score. This guarantees that the network only normalizes present data using strictly historical distributions.

### B. Vision Model (V-Model): $ eta$-VAE and KL Annealing
**The Problem:** Scaling the latent space to 8D to compress the new 7D input space introduced the risk of "Posterior Collapse"—a phenomenon where the KL divergence penalty dominates the reconstruction loss, causing the latent space to collapse into a standard normal distribution that encodes zero economic information.
**The Solution:**
* **$ eta$-VAE Implementation:** We introduced a $ eta$ parameter to the loss function (`Loss = Recon_Loss + (beta * KLD_Loss)`).
* **Monotonic KL Annealing:** During training, $ eta$ is linearly scaled from 0.0 to 1.0 over the first 50 epochs. This allows the decoder to learn the complex manifold of the expanded 7D feature space before the Gaussian regularization is fully enforced.
* **Hardware Optimization:** To respect a strict 6GB VRAM ceiling, DMA transfers were optimized using `pin_memory=True` and non-blocking CUDA transfers.

### C. Memory Model (M-Model): The Stochastic RSSM
**The Problem:** The deterministic LSTM ignored market variance. 
**The Solution:**
* **Recurrent State Space Model:** The LSTM was replaced with a stochastic RSSM utilizing GRU cells (for memory efficiency). 
* **Probabilistic Output:** Rather than predicting a flat 8D vector, the GRU branches into dual linear heads, outputting the mean ($\mu$) and log-variance ($\log\sigma^2$) of the *subsequent* latent state.
* **Gaussian NLL Loss:** Mean Squared Error was replaced with Gaussian Negative Log-Likelihood (NLL). This adversarial loss function forces the model to predict wide variances during volatile market regimes (e.g., credit spread blowouts) and tight variances during stable periods, mathematically capturing systemic uncertainty.
* **Monte Carlo Hallucination:** By leveraging the Reparameterization Trick ($z_{t+1} \sim \mathcal{N}(\mu, \sigma^2)$), the engine can now autoregressively "dream" thousands of unique, statistically valid economic timelines.

### D. Controller (C-Model): Institutional Friction
**The Problem:** A PPO agent trained in a frictionless vacuum learns to rapidly oscillate allocations to capture basis points, a strategy that fails instantly under real-world trading costs.
**The Solution:**
* **Transaction Costs:** Implemented a 15 basis point (0.0015) penalty per unit of portfolio turnover.
* **Asymmetrical Downside Penalty:** Replaced absolute return rewards with a Sharpe-inspired risk metric. Negative net returns are subjected to a 2.0x risk aversion multiplier, forcing the agent to prioritize drawdown protection.

---

## III. Empirical Results & Behavioral Analysis
Upon deploying the PPO agent into a 252-day forward-looking hallucination generated by the RSSM, empirical validation confirmed the efficacy of the architecture.

The RSSM successfully hallucinated a severe, unrecorded economic downturn. The agent, navigating this synthetic bear market, significantly outperformed the static 60/40 benchmark, entirely *net* of the 15bps transaction friction. 

**Learned Behaviors and Limitations:**
The introduction of friction caused the agent to adopt an aggressive "Hold" strategy (remaining ~100% long equities) to avoid transaction costs. However, during moments of extreme hallucinated volatility, the agent exhibited abrupt "panic-selling" behavior—dumping equities to near 0% to avoid the asymmetrical downside penalty, before aggressively rebuying. 

While mathematically optimal given its current training scope, this blunt binary behavior indicates under-training. A training envelope of 250,000 timesteps is insufficient for an agent to learn smooth, macro-aware transition curves in a highly stochastic 8D environment.

## V. Scaling and the Out-of-Sample (OOS) Firewall (Revision v3.0)

**Project:** Macroeconomic World Model via V-M-C Architecture (Institutional Rigor)
**Revision:** v3.0 (OOS Data Firewall, Non-Linear Impact, and Compute Scaling)

### A. The OOS Data Firewall & Latent Optimization
**Objective:** Eliminate data leakage and optimize the latent representation for institutional-grade backtesting.
* **Firewall Implementation:** A strict chronological split was enforced at the data ingestion layer. The training set was restricted to 1993-01-01 through 2018-12-31. The 2019-Present period was quarantined as an Out-of-Sample (OOS) test set. 
* **Causal Normalization:** To prevent look-ahead bias, the 252-day Rolling Z-Score normalization was applied to the full dataset *prior* to the split. This ensures the initial window of the OOS set is grounded in the trailing distribution of the training set without leaking future information back to the V-Model or M-Model.
* **4D Latent Compression:** The latent dimension was reduced from 8D to **4D**. This provides a more aggressive compression of the 7D input space, forcing the V-Model to extract more robust "regime-level" features and simplifying the state space for the RL agent.
* **GPU Acceleration:** The entire pipeline was refactored for full GPU (CUDA) support, including the environment's VAE/RSSM inference loop, to support massive compute scaling.

### B. Non-Linear Market Impact
**Objective:** Replace "toy" flat transaction costs with a realistic institutional friction model.
* **Implementation:** We implemented the **Square Root Law of Market Impact**. 
* **Formula:** $Slippage = \gamma \cdot \sqrt{Turnover}$, where $\gamma = 0.0010$ (10bps base multiplier).
* **Rationale:** This model ensures that small rebalances incur minimal friction, while large, abrupt shifts in allocation (e.g., a 0% to 100% "slam") trigger exponentially higher slippage penalties, mimicking the liquidity constraints of a large institutional block trade.

### C. Compute Scaling & Checkpointing
**Objective:** Increase the agent's exposure to the stochastic "dream" environment to stabilize policy gradients.
* **Scaling:** PPO training was increased from 250,000 to **5,000,000 timesteps**.
* **Persistence:** `CheckpointCallback` was integrated to save model weights at 500,000-step intervals, ensuring fault tolerance during long-running compute phases.

---

## VI. Results: The OOS Failure and the "Barcode" Problem

Upon evaluating the agent on the 2019–2024 OOS dataset, the model exhibited a significant failure in allocation logic. Despite the 5M step training run and the introduction of non-linear friction, the agent achieved an **OOS Sharpe Ratio of -0.52**, severely underperforming the 60/40 benchmark (-0.10).

**Key Behavioral Observations:**
1. **High-Frequency Churn ("The Barcode"):** The agent's allocation chart resembles a "barcode," characterized by rapid, near-daily oscillations between 0% and 100% equity exposure. The agent is effectively "churning" the portfolio, incinerating its capital through cumulative transaction fees—a phenomenon described as "death by a thousand cuts."
2. **Inertia Blindness:** The agent demonstrates no concept of "holding" a position. It treats each trading day as an independent optimization problem, failing to account for the path-dependent cost of its previous allocation.
3. **Volatility Induced Panic:** During the March 2020 COVID-19 crash, the agent exhibited extreme "panic-selling," exiting the market at the local bottom and incurring massive slippage costs, only to "barcode" back into equities during the subsequent recovery, missing the primary rebound.

The persistence of this behavior after 5,000,000 steps of training suggests a fundamental architectural gap in how the agent perceives its own state and the cost of movement within the latent economy.

---

## VII. Sprint: Elite Quant Audit & State-Aware Proprioception
**Objective:** Solve "Barcode" churn and OOS failure via state-aware proprioception, continuous reward gradients, and RNN memory persistence.

### A. Observation Proprioception (Solving Inertia Blindness)
*   **The Problem:** The agent was "blind" to its current portfolio position. In `v2.0`, the observation was purely the 4D latent economic state. Without knowing its `previous_weight`, the agent could not calculate the cost of its next action until *after* the reward was processed, leading to erratic "barcode" oscillations.
*   **The Solution:** Expanded the observation space from **4D to 5D**. We now concatenate the agent's current `stock_weight` to the 4D latent vector. This grants the agent "proprioception"—the ability to sense its own position—allowing it to internalize transaction friction as a structural constraint of its state, not just a random penalty.

### B. RNN Memory Persistence (Fixing the Temporal Leak)
*   **The Problem:** An architectural audit revealed that the `PortfolioEnv` was resetting the RSSM's hidden state (`h`) to `None` on every single `step()`. 
*   **The Impact:** This effectively neutralized the "Memory" in the V-M-C framework. The agent was navigating a Markovian "mood" rather than a temporal sequence, causing the "dreams" to lose historical context and drift.
*   **The Fix:** Implemented hidden state persistence. `self.h` is now initialized in `reset()` and passed/updated during each `step()`, ensuring the RSSM maintains a contiguous temporal "memory" throughout the 60-day trading horizon.

### C. Continuous Mean-Variance Reward
*   **The Problem:** The previous 2.0x downside penalty was a non-differentiable step function. This caused the agent to "panic" at the zero-bound, leading to binary all-in/all-out behavior.
*   **The Solution:** Implemented a symmetric, continuous **Mean-Variance Objective**:
    *   $Reward = Net\_Return - (\lambda \cdot Net\_Return^2)$
    *   $\lambda = 2.0$ (Risk Aversion Coefficient).
    *   This provides a smooth, differentiable gradient that penalizes volatility (variance) in both directions, encouraging the agent to seek stable, low-volatility growth paths.

### D. Unit-Correct Friction & Causal Shadowing
*   **Unit-Correct Friction:** In Z-score space (standard deviations), a flat 10bps penalty ($0.0010$) is mathematically microscopic. We scaled the slippage multiplier to **0.05** to ensure transaction costs are significant relative to Z-scored returns.
*   **Absolute Causality:** Fixed a "micro-leak" in the `DataFetcher`. The rolling Z-score was shifted by **1 day** ($df.shift(1)$) to ensure that the normalization parameters (mean/std) for today's data are calculated using *only* information available before today's close.
*   **Grounding Noise:** Synchronized the environment's `step()` logic with the evaluation engine by injecting **0.05 grounding noise** into the latent hallucinations. This prevents the agent from over-optimizing for "perfect" dreams and forces it to build a policy robust to stochastic shocks.

### E. Horizon Stabilization
*   **The Fix:** Reduced the environment's `max_steps` from 252 (1 year) to **60 (1 quarter)**. 
*   **Rationale:** Shortening the "dream" horizon prevents the autoregressive RSSM from drifting into unphysical extreme states (covariate shift) while providing enough temporal depth for the agent to learn quarterly economic cycles.

**Conclusion:** With state-awareness (5D obs), temporal memory (RNN persistence), and unit-correct friction, the agent is now architecturally equipped to solve the "barcode" problem and move toward stable, institutional-grade allocation.

---

## VIII. Sprint: Institutional Pacing & The "Bang-Bang" Loophole
**Objective:** Eliminate binary "Bang-Bang" control by enforcing a Delta-Action Space and Quadratic Market Impact, forcing the agent to learn smooth institutional scaling.

### A. The "Bang-Bang" Loophole & Square Root Friction
*   **The Problem:** Despite prior fixes, the agent achieved a 0.93 OOS Sharpe Ratio by mathematically exploiting the Square Root Law of market impact (`0.05 * math.sqrt(turnover)`). The agent realized that dumping 100% of the portfolio at once was cheaper on a per-unit basis than scaling out smoothly over 10 days. This resulted in binary "Bang-Bang" control (0% or 100% allocations), which is unfeasible for institutional block sizing.
*   **The Solution:** We replaced the square root friction with a **Quadratic Market Impact** formula: `slippage = (0.0050 * turnover) + (5.0 * (turnover ** 2))`. This introduces a devastating penalty for large block trades, making the marginal cost of slippage increase exponentially with trade size.

### B. Delta-Action Space Constraint
*   **The Problem:** The agent's action space allowed it to directly output an absolute target weight between `[0.0, 1.0]`.
*   **The Solution:** Migrated the action space to a bounded delta: `Box(low=-0.1, high=0.1)`. The agent can now only shift its allocation by a maximum of 10% per day (`stock_weight = np.clip(previous_weight + action, 0.0, 1.0)`). This acts as a hard mechanical constraint guaranteeing inertia.

### C. Architectural Audit & Reward Function Overhaul
Prior to a 5M step training run, an architectural audit revealed four critical RL misalignment issues that would have prevented the agent from learning institutional pacing under the new constraints:
1.  **"Fear of Success" Reward Misalignment:** The continuous mean-variance reward (`net_return - 2.0 * (net_return ** 2)`) was actively punishing high absolute returns in Z-score space (e.g., a return of 2.0 yielded a reward of -6.0). We updated the reward to maximize raw `net_return` directly, letting the quadratic transaction costs handle the pacing and risk aversion.
2.  **PPO Action Noise Mismatch:** Stable Baselines3 PPO defaults to an initial exploration standard deviation of `1.0`. With bounds of `[-0.1, 0.1]`, almost all explored actions were clipped to the extremes. We fixed this by initializing PPO with `policy_kwargs=dict(log_std_init=-3.0)`, reducing the initial exploration noise to a scale appropriate for the delta action space.
3.  **Boundary Saturation ("Sticky Edges"):** When the agent hit the 0.0 or 1.0 boundaries and continued to push past them, the environment clipped the values, but the PPO network's internal pre-activations saturated. This caused delayed reactions during regime shifts. We implemented a `boundary_penalty` in the reward function that quadratically penalizes the agent for attempting to allocate beyond 0% or 100%.
4.  **Diluted Friction Calibration:** The initial quadratic coefficient (`0.10`) only resulted in a 15bps maximum penalty given the new 0.1 action limit. We scaled the coefficient to `5.0`, ensuring a max delta step costs a punishing 500bps, mathematically forcing the agent to prefer smooth, incremental execution.

