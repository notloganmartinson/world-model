import yfinance as yf
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import io
import requests
import os

class DataFetcher:
    """
    Fetches daily SPY/VUSTX/^VIX and monthly/daily FRED data (FEDFUNDS, CPIAUCSL, T10Y2Y, BAMLH0A0HYM2),
    merging them into a daily causal Z-score normalized tensor.
    """
    def __init__(self, start_date="1993-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")

    def split_data(self, df, split_date="2018-12-31"):
        """
        Splits the normalized dataframe into Training and OOS Testing sets.
        Training: start_date to split_date
        OOS Testing: split_date + 1 day to end_date
        """
        train_df = df.loc[:split_date]
        # OOS starts from the first available day after split_date
        oos_df = df.loc[split_date:].iloc[1:] 
        
        train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
        oos_tensor = torch.tensor(oos_df.values, dtype=torch.float32)
        
        return train_tensor, oos_tensor

    def fetch_fred_data(self, series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={self.start_date}&coed={self.end_date}"
        response = requests.get(url)
        if response.status_code == 200:
            # Parse dates directly in read_csv, handling FRED's '.' as NaN
            df = pd.read_csv(io.StringIO(response.text), parse_dates=[0], index_col=0, na_values='.')
            return df
        else:
            raise Exception(f"Failed to fetch {series_id} from FRED. Status: {response.status_code}")

    def fetch_data(self):
        # 1. Fetch Daily SPY, VUSTX, and ^VIX
        print(f"Fetching SPY, VUSTX, and ^VIX (Daily) from {self.start_date}...")
        tickers = ["SPY", "VUSTX", "^VIX"]
        data = yf.download(tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        
        # Handle MultiIndex or Flat Index
        if isinstance(data.columns, pd.MultiIndex):
            try:
                stocks_df = data['Close']
            except KeyError:
                stocks_df = data.xs('Close', axis=1, level=0)
        else:
            stocks_df = data
            
        # Refactor equity and bond returns to log returns: ln(Pt / Pt-1)
        # ^VIX is kept as a level (Market Volatility)
        stocks_log_ret = np.log(stocks_df[["SPY", "VUSTX"]] / stocks_df[["SPY", "VUSTX"]].shift(1))
        stocks_combined = pd.concat([stocks_log_ret, stocks_df[["^VIX"]]], axis=1)

        # 2. Fetch FRED Data with Publication Lags
        # We shift data forward to simulate the information available at the daily close
        print("Fetching FRED data with publication lags...")
        fedfunds = self.fetch_fred_data("FEDFUNDS").shift(1, freq='D')         # 1-day lag
        cpiaucsl = self.fetch_fred_data("CPIAUCSL").shift(45, freq='D')        # 45-day lag (approx. BLS release)
        t10y2y = self.fetch_fred_data("T10Y2Y").shift(1, freq='D')             # 1-day lag
        credit_spread = self.fetch_fred_data("BAMLH0A0HYM2").shift(1, freq='D') # 1-day lag
        
        # 3. Merge and Forward Fill
        # The join occurs on the daily trading index. ffill() projects the latest
        # available (lagged) macro data across trading days.
        fred_combined = pd.concat([fedfunds, cpiaucsl, t10y2y, credit_spread], axis=1)
        
        # Join daily stock data with FRED data
        df = stocks_combined.join(fred_combined, how='left')
        
        # Ffill all macro indicators to fill gaps (especially for monthly data)
        df = df.ffill()
        
        # 4. Cleanup initial nans before normalization
        df = df.dropna()
        
        # Ensure column order is consistent
        df = df[["SPY", "VUSTX", "^VIX", "FEDFUNDS", "CPIAUCSL", "T10Y2Y", "BAMLH0A0HYM2"]]
        
        print(f"Dataset merged. Total daily records: {len(df)}")
        return df

    def get_normalized_tensor(self, df):
        # 5. Implement Causal Rolling Z-Score Normalization
        # Using a 252-trading-day window (approx. 1 trading year)
        # Zt = (Xt - mu_trailing) / sigma_trailing
        print("Applying Causal Rolling Z-Score (252-day window)...")
        window_size = 252
        # Shifted by 1 to ensure parameters (mean/std) only use data available BEFORE today
        rolling_mean = df.shift(1).rolling(window=window_size).mean()
        rolling_std = df.shift(1).rolling(window=window_size).std()
        
        # Apply normalization
        df_normalized = (df - rolling_mean) / rolling_std
        
        # 6. Drop the first 252 rows that contain NaNs from the rolling window
        valid_mask = ~df_normalized.isna().any(axis=1)
        df_normalized = df_normalized[valid_mask]
        rolling_mean = rolling_mean[valid_mask]
        rolling_std = rolling_std[valid_mask]
        
        # Returns Normalized Dataframe and the stats to maintain date-index for split_data
        return df_normalized, rolling_mean, rolling_std

if __name__ == "__main__":
    fetcher = DataFetcher()
    df_raw = fetcher.fetch_data()
    df_normalized, df_means, df_stds = fetcher.get_normalized_tensor(df_raw)
    
    # Apply OOS Firewall
    split_date = "2018-12-31"
    train_tensor, oos_tensor = fetcher.split_data(df_normalized, split_date=split_date)
    
    # Split the Rolling Stats for Anchored Un-normalization
    train_means_tensor, _ = fetcher.split_data(df_means, split_date=split_date)
    train_stds_tensor, _ = fetcher.split_data(df_stds, split_date=split_date)

    # Also split the RAW log returns for evaluation (to avoid linear combination of log-rets)
    # We only need SPY and VUSTX for the return calculation
    df_raw_returns = df_raw[["SPY", "VUSTX"]]
    # Align with the normalized index (which dropped the first 252 days)
    df_raw_returns = df_raw_returns.loc[df_normalized.index]
    
    _, oos_raw_tensor = fetcher.split_data(df_raw_returns, split_date=split_date)
    
    # Save Normalization Metadata (Keep as fallback)
    train_means_last = df_means.loc[:split_date].iloc[-1]
    train_stds_last = df_stds.loc[:split_date].iloc[-1]
    norm_metadata = {
        "spy_mean": train_means_last["SPY"],
        "spy_std": train_stds_last["SPY"],
        "vustx_mean": train_means_last["VUSTX"],
        "vustx_std": train_stds_last["VUSTX"]
    }

    print("\nProcessed DataFrame Sample (Daily):")
    print(df_normalized.head())
    
    # Save for Phases 2-5
    os.makedirs("src/data", exist_ok=True)
    torch.save(train_tensor, "src/data/macro_data_daily_train.pt")
    torch.save(train_means_tensor, "src/data/macro_data_daily_train_mean.pt")
    torch.save(train_stds_tensor, "src/data/macro_data_daily_train_std.pt")
    torch.save(oos_tensor, "src/data/macro_data_daily_oos.pt")
    torch.save(oos_raw_tensor, "src/data/macro_data_daily_oos_raw.pt")
    torch.save(norm_metadata, "src/data/norm_metadata.pt")
    
    print("Tensors and Metadata saved to src/data/")
