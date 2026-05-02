"""
data_generator.py
-----------------
Generates a synthetic market demand dataset with:
  - trend, seasonality, noise
  - simulated news sentiment scores (NLP proxy)
Saves to data/market_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)


def generate_market_data(n_days: int = 730, save: bool = True) -> pd.DataFrame:
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    # ── Numerical signal ──────────────────────────────────────────────────────
    trend      = np.linspace(100, 160, n_days)
    weekly_s   = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    yearly_s   = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise      = np.random.normal(0, 5, n_days)
    demand     = trend + weekly_s + yearly_s + noise
    demand     = np.clip(demand, 50, 250)

    # ── Exogenous / market features ───────────────────────────────────────────
    price      = 50 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 90) + np.random.normal(0, 2, n_days)
    promo_flag = (np.random.rand(n_days) > 0.85).astype(int)

    # ── NLP proxy: sentiment score ────────────────────────────────────────────
    # Simulates aggregated daily sentiment from financial news [-1, +1]
    sentiment  = 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 30) + np.random.normal(0, 0.2, n_days)
    sentiment  = np.clip(sentiment, -1, 1)

    df = pd.DataFrame({
        "date":      dates,
        "demand":    demand.round(2),
        "price":     price.round(2),
        "promo":     promo_flag,
        "sentiment": sentiment.round(4),
    })
    df.set_index("date", inplace=True)

    if save:
        out = Path(__file__).parent / "market_data.csv"
        df.to_csv(out)
        print(f"[data_generator] Saved → {out}  ({len(df)} rows)")

    return df


if __name__ == "__main__":
    generate_market_data()
