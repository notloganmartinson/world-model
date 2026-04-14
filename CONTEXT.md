# PROJECT CONTEXT: World Model Audit & Final State Fixes

## 1. THE CORE MISSION
We are executing a final bug-fixing sprint based on an architectural audit before launching our 5,000,000 timestep background run. We need to fix a critical RNN memory leak, eliminate a microscopic look-ahead bias in the data pipeline, and correct a mathematical unit mismatch between Z-scored returns and absolute friction penalties.

## 2. CURRENT FOCUS: AUDIT EXECUTION

**Task 1: RNN Memory Persistence (`src/env/portfolio_env.py`)**
* The environment's RSSM is currently resetting its hidden state to `None` on every step. 
* **Fix:** Initialize `self.h = None` inside `reset()`. Inside `step()`, capture and pass the hidden state: `mu, logvar, self.h = self.rssm(z_t, self.h)`.

**Task 2: Causal Normalization Shadow Bias (`src/data/fetcher.py`)**
* The 252-day rolling Z-score includes the current day, leaking 1/252th of future data.
* **Fix:** Shift the rolling window calculation by 1 day to ensure absolute causality: `rolling_mean = df.shift(1).rolling(window=window_size).mean()`. Apply this to standard deviation as well.

**Task 3: Unit-Correct Friction & Noise (`src/env/portfolio_env.py`)**
* **Noise:** Inject `+ torch.randn_like(z_next) * 0.05` to the next state hallucination inside `step()` to match the evaluation engine.
* **Friction Mismatch:** Because the returns from the VAE/RSSM are in Z-score space (standard deviations), a friction penalty of `0.0010` is mathematically invisible to the agent.
* **Fix:** Update the slippage multiplier from `0.0010` to `0.05` to scale it properly into Z-score space: `slippage = 0.05 * math.sqrt(turnover)`.

## 3. ENGINEERING CONSTRAINTS
* Apply these fixes directly and concisely. 
* Do not alter the Mean-Variance reward structure established in the previous sprint, only update the slippage coefficient.
