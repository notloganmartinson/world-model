# PROJECT CONTEXT: World Model - Deadband Execution Sprint

## 1. THE CORE MISSION
The V-M-C agent successfully learned institutional pacing (scaling in/out) and achieved a strong Out-of-Sample Sharpe Ratio. However, it suffers from "The Chop"—micro-adjusting its allocation by 1% or 2% in response to daily latent space noise, bleeding capital to quadratic friction without capturing trend.

*The objective of this sprint is to implement an Action Confidence Threshold (Deadband Filter). The agent will be mathematically forced to ignore low-conviction signals, eliminating micro-churn and preparing the architecture for live paper trading.*

## 2. CURRENT FOCUS: ELITE DEADBAND FILTER

**Task 1: Environment Confidence Threshold (`src/env/portfolio_env.py`)**
* The agent's raw delta action needs a hard filter before it is applied to the portfolio weight.
* **Fix:** Inside the `step()` function, extract the raw action delta. If the absolute value of the delta is less than a 2% threshold (`0.02`), force the delta to `0.0`. 
* **Logic:** python
  raw_delta = float(action[0])
  if abs(raw_delta) < 0.02:
      raw_delta = 0.0
  target_weight = self.previous_weight + raw_delta
  stock_weight = np.clip(target_weight, 0.0, 1.0)
Task 2: Evaluation Engine Alignment (src/evaluate.py)

The OOS backtest must mirror the exact physics of the training environment to remain valid.

Fix: Update the evaluation loop in evaluate.py to apply the exact same 0.02 deadband filter to the predicted action before calculating turnover and slippage.

3. ENGINEERING CONSTRAINTS
Do not touch the existing friction math, the reward function, or the PPO hyperparameters.

Maintain the boundary_penalty logic exactly as it is. We are simply zeroing out the delta before the target weight is calculated if the signal is too weak.
