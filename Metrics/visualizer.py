"""
visualizer.py
-------------
Generates all paper-quality plots:
  1. Raw demand + sentiment time series
  2. Forecast comparison (all models vs actuals)
  3. Feature importance (LR & RF)
  4. Model comparison bar chart (RMSE / MAE)
  5. N-BEATS trend + seasonality decomposition
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PALETTE = {
    "actual": "#1f2937",
    "lr":     "#3b82f6",
    "rf_base":"#f59e0b",
    "rf_hyb": "#10b981",
    "nbeats": "#ef4444",
    "sentiment": "#8b5cf6",
}


# ── 1. Raw data overview ──────────────────────────────────────────────────────
def plot_raw_data(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df.index, df["demand"], color=PALETTE["actual"], lw=1.5, label="Demand")
    axes[0].fill_between(df.index, df["demand"], alpha=0.1, color=PALETTE["actual"])
    axes[0].set_ylabel("Demand")
    axes[0].set_title("Market Demand Signal", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["sentiment"], color=PALETTE["sentiment"], lw=1.2, label="Sentiment")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("Sentiment Score")
    axes[1].set_title("Daily Sentiment Signal (NLP Proxy)", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "01_raw_data.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out}")


# ── 2. Forecast comparison ────────────────────────────────────────────────────
def plot_forecasts(
    test_index: pd.DatetimeIndex,
    y_test:     np.ndarray,
    predictions: dict,         # {"Model Name": np.ndarray}
    title: str = "Forecast Comparison",
    filename: str = "02_forecast_comparison.png",
):
    color_map = {
        "Linear Regression":        PALETTE["lr"],
        "Random Forest (Baseline)": PALETTE["rf_base"],
        "Random Forest (Hybrid)":   PALETTE["rf_hyb"],
        "N-BEATS":                  PALETTE["nbeats"],
    }

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(test_index, y_test, color=PALETTE["actual"], lw=2, label="Actual", zorder=5)

    for name, pred in predictions.items():
        color = color_map.get(name, "gray")
        ax.plot(test_index[:len(pred)], pred, lw=1.5, ls="--", color=color, label=name, alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (scaled)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out}")


# ── 3. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(importance_dict: dict, model_name: str, filename: str):
    items    = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    features = [i[0] for i in items]
    values   = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features[::-1], values[::-1],
                   color=PALETTE["rf_hyb"], edgecolor="white", height=0.7)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out}")


# ── 4. Model comparison bar chart ─────────────────────────────────────────────
def plot_model_comparison(results: list):
    """results = list of dicts with keys: model, RMSE, MAE, R2, MAPE"""
    df  = pd.DataFrame(results).set_index("model")
    metrics = ["RMSE", "MAE", "MAPE"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [PALETTE["lr"], PALETTE["rf_base"], PALETTE["rf_hyb"], PALETTE["nbeats"]]

    for ax, metric in zip(axes, metrics):
        vals = df[metric]
        bars = ax.bar(range(len(vals)), vals.values, color=colors[:len(vals)],
                      edgecolor="white", width=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=20, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = OUT_DIR / "05_model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out}")


# ── 5. N-BEATS decomposition ──────────────────────────────────────────────────
def plot_nbeats_decomposition(
    series:       np.ndarray,
    nbeats_model,                   # NBEATSModel instance
    dates:        pd.DatetimeIndex,
):
    if nbeats_model._backend != "numpy":
        print("[Plot] Decomposition plot only available for NumPy N-BEATS backend.")
        return

    nm       = nbeats_model._np_model
    lookback = nm.lookback
    y_hist   = series[-lookback - nbeats_model.horizon : -nbeats_model.horizon]
    t        = np.arange(lookback, dtype=float)

    trend_fit  = np.column_stack([t ** d for d in range(nm.trend_degree + 1)]) @ nm._trend_coef
    season_fit = nm._fourier_basis(t) @ nm._fourier_coef

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(y_hist, color=PALETTE["actual"], lw=1.5, label="Input Series")
    axes[0].set_title("N-BEATS — Input Signal", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(trend_fit, color=PALETTE["rf_hyb"], lw=2, label="Trend Block")
    axes[1].set_title("N-BEATS — Trend Component", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(season_fit, color=PALETTE["nbeats"], lw=1.5, label="Seasonality Block")
    axes[2].set_title("N-BEATS — Seasonality Component", fontweight="bold")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "04_nbeats_decomposition.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out}")
