"""
nbeats_model.py
---------------
N-BEATS wrapper with two backends:
  1. neuralforecast (PyTorch) — used when available
  2. Pure NumPy fallback — Theta-method style decomposition that mimics the
     trend + seasonality block concept of N-BEATS without any deep-learning
     dependency. Suitable for environments without GPU / PyTorch.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ── Numpy fallback: Trend + Seasonality decomposition ────────────────────────

class _NumpyNBEATS:
    """
    Lightweight N-BEATS-inspired model implemented entirely in NumPy.

    Architecture (mirrors paper):
      Block 1 — Trend block   : fits a polynomial of degree `trend_degree`
      Block 2 — Seasonality   : fits a Fourier series of `n_harmonics` terms
      Residual pass-through   : passes unexplained residual to next block

    Forecast = trend_forecast + seasonality_forecast + residual_mean
    """

    def __init__(
        self,
        horizon: int = 30,
        lookback: int = 60,
        trend_degree: int = 3,
        n_harmonics: int = 3,
        season_period: int = 7,
    ):
        self.horizon       = horizon
        self.lookback      = lookback
        self.trend_degree  = trend_degree
        self.n_harmonics   = n_harmonics
        self.season_period = season_period

        self._trend_coef: np.ndarray  = None
        self._fourier_coef: np.ndarray = None
        self._residual_mean: float    = 0.0

    # ── Fourier basis ─────────────────────────────────────────────────────────
    def _fourier_basis(self, t: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(t))]
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * t / self.season_period))
            cols.append(np.cos(2 * np.pi * k * t / self.season_period))
        return np.column_stack(cols)

    def fit(self, series: np.ndarray) -> "_NumpyNBEATS":
        if len(series) < self.lookback:
            raise ValueError(f"Series too short ({len(series)}); need ≥ {self.lookback} points.")

        y = series[-self.lookback:]
        t = np.arange(len(y), dtype=float)

        # ── Block 1: Trend ────────────────────────────────────────────────────
        A_trend          = np.column_stack([t ** d for d in range(self.trend_degree + 1)])
        self._trend_coef = np.linalg.lstsq(A_trend, y, rcond=None)[0]
        trend_fit        = A_trend @ self._trend_coef
        residual1        = y - trend_fit

        # ── Block 2: Seasonality ──────────────────────────────────────────────
        A_fourier          = self._fourier_basis(t)
        self._fourier_coef = np.linalg.lstsq(A_fourier, residual1, rcond=None)[0]
        season_fit         = A_fourier @ self._fourier_coef
        residual2          = residual1 - season_fit

        # ── Residual block: simple mean ───────────────────────────────────────
        self._residual_mean = float(residual2.mean())
        return self

    def predict(self) -> np.ndarray:
        lookback = self.lookback
        t_future = np.arange(lookback, lookback + self.horizon, dtype=float)

        # Trend forecast
        A_trend_f  = np.column_stack([t_future ** d for d in range(self.trend_degree + 1)])
        trend_fore = A_trend_f @ self._trend_coef

        # Seasonality forecast
        A_four_f   = self._fourier_basis(t_future)
        season_fore = A_four_f @ self._fourier_coef

        return trend_fore + season_fore + self._residual_mean


# ── Public wrapper ────────────────────────────────────────────────────────────

class NBEATSModel:
    """
    Unified N-BEATS wrapper.
    Tries neuralforecast first; falls back to _NumpyNBEATS.
    """

    def __init__(self, horizon: int = 30, max_steps: int = 300):
        self.horizon   = horizon
        self.max_steps = max_steps
        self._backend  = None   # "torch" | "numpy"
        self._nf       = None   # NeuralForecast instance (torch path)
        self._np_model = None   # _NumpyNBEATS instance (numpy path)
        self._train_series: np.ndarray = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, series: np.ndarray, dates: pd.DatetimeIndex = None) -> "NBEATSModel":
        self._train_series = series

        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATS
            from neuralforecast.losses.pytorch import MAE as NF_MAE

            print("[N-BEATS] PyTorch backend detected — using neuralforecast.")

            nbeats_df = pd.DataFrame({
                "unique_id": "series",
                "ds": dates if dates is not None else pd.date_range("2022-01-01", periods=len(series)),
                "y":  series,
            })

            model = NBEATS(
                h=self.horizon,
                input_size=2 * self.horizon,
                loss=NF_MAE(),
                max_steps=self.max_steps,
                stack_types=["trend", "seasonality"],
                n_blocks=[3, 3],
                mlp_units=[[512, 512], [512, 512]],
                scaler_type="standard",
            )
            self._nf = NeuralForecast(models=[model], freq="D")
            self._nf.fit(df=nbeats_df[:-self.horizon])
            self._backend = "torch"

        except Exception as e:
            print(f"[N-BEATS] neuralforecast unavailable ({e}). Using NumPy fallback.")
            lookback = min(max(2 * self.horizon, 60), len(series) - self.horizon)
            self._np_model = _NumpyNBEATS(
                horizon=self.horizon,
                lookback=lookback,
            )
            self._np_model.fit(series[:-self.horizon])
            self._backend = "numpy"

        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self) -> np.ndarray:
        if self._backend == "torch":
            fc = self._nf.predict()
            return fc["NBEATS"].values
        elif self._backend == "numpy":
            return self._np_model.predict()
        else:
            raise RuntimeError("Model not fitted yet.")

    # ── Ground truth for evaluation ───────────────────────────────────────────
    def ground_truth(self) -> np.ndarray:
        return self._train_series[-self.horizon:]
