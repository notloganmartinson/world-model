# AST REPO MAP: MACROECONOMIC WORLD MODEL
SYSTEM INSTRUCTIONS:
1. LOGIC OMITTED: Functions are NOT empty. Implementations are abstracted for context efficiency.
2. READ/WRITE PROTOCOL: To modify a function, you MUST ask the user to provide the specific file path first. Do NOT hallucinate modifications without the source file.
3. ARCHITECTURE STRICTNESS: We are building a V-M-C (Vision, Memory, Controller) framework. Ensure all data flows sequentially: Fetcher -> VAE (Compression) -> LSTM (Dreamer) -> RL Agent.
4. PRECISION: Use exact class, function, and file names from this map.

---

```python

# generate_map.py
def format_function(node, indent): # Helper to format a function signature, return type, and brief docstring.
def parse_file(filepath): # Parses a Python file and returns its AST skeleton.
def generate_map(root_dir): # Walks the directory and builds the repo map.

# src/train_lstm.py
def create_sliding_window_dataset(data, window_size) -> Tuple[torch.Tensor, torch.Tensor]: # Transforms a single sequence into a dataset of windows and targets.
def train_lstm():

# src/train_agent.py
def train_agent():

# src/train_vae.py
def train_vae():

# src/evaluate.py
def evaluate():

# src/models/lstm.py
class StochasticRSSM: # A stochastic Recurrent State Space Model (RSSM) designed to model the
    def __init__(self, input_size, hidden_size, num_layers, output_size):
    def reparameterize(self, mu, logvar) -> torch.Tensor: # Reparameterization trick: sample from N(mu, sigma^2).
    def forward(self, x, h) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Forward pass.
    def hallucinate(self, z_start, steps) -> torch.Tensor: # Autoregressively generate synthetic economic futures (Monte Carlo hallucination).

# src/models/vae.py
class VAE: # A production-grade Variational Autoencoder (VAE) for compressing macroeconomic states.
    def __init__(self, input_dim, hidden_dim, latent_dim):
    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]: # Maps input to latent distribution parameters.
    def reparameterize(self, mu, logvar) -> torch.Tensor: # Reparameterization trick to sample from N(mu, var) while maintaining backprop.
    def decode(self, z) -> torch.Tensor: # Maps latent vector back to reconstructed input.
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Full forward pass: Encode -> Reparameterize -> Decode.
def vae_loss_function(recon_x, x, mu, logvar, beta) -> Dict[str, torch.Tensor]: # Computes the beta-VAE loss.

# src/env/portfolio_env.py
class PortfolioEnv: # A portfolio environment that lives inside the 'dream' of the Stochastic RSSM M-Model.
    def __init__(self, max_steps):
    def reset(self, seed, options):
    def step(self, action):

# src/data/fetcher.py
class DataFetcher: # Fetches daily SPY/VUSTX/^VIX and monthly/daily FRED data (FEDFUNDS, CPIAUCSL, T10Y2Y, BAMLH0A0HYM2),
    def __init__(self, start_date):
    def fetch_fred_data(self, series_id):
    def fetch_data(self):
    def get_normalized_tensor(self, df):
```
