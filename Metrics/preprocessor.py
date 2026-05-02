"""
preprocessor.py
---------------
Handles:
  - lag / rolling feature engineering
  - train/test split
  - MinMax scaling with inverse capability
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class Preprocessor:
    def __init__(self, target_col: str = "demand", test_size: float = 0.2):
        self.target_col = target_col
        self.test_size  = test_size
        self.scaler     = MinMaxScaler()

    # ── Feature engineering ───────────────────────────────────────────────────
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Scale target
        out["demand_scaled"] = self.scaler.fit_transform(out[[self.target_col]])

        # Lag features
        for lag in [1, 3, 7, 14]:
            out[f"lag_{lag}"] = out["demand_scaled"].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            out[f"roll_mean_{window}"] = out["demand_scaled"].rolling(window).mean()
            out[f"roll_std_{window}"]  = out["demand_scaled"].rolling(window).std()

        # Calendar features
        out["day_of_week"] = out.index.dayofweek
        out["month"]       = out.index.month
        out["quarter"]     = out.index.quarter
        out["is_weekend"]  = (out.index.dayofweek >= 5).astype(int)

        out.dropna(inplace=True)
        return out

    # ── Train / test split ────────────────────────────────────────────────────
    def split(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = df[feature_cols].values
        y = df["demand_scaled"].values

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    # ── Inverse transform ─────────────────────────────────────────────────────
    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()

    # ── Convenience: return numerical feature column names ────────────────────
    @staticmethod
    def num_feature_cols() -> list:
        cols = []
        for lag in [1, 3, 7, 14]:
            cols.append(f"lag_{lag}")
        for window in [7, 14, 30]:
            cols += [f"roll_mean_{window}", f"roll_std_{window}"]
        cols += ["day_of_week", "month", "quarter", "is_weekend", "price", "promo"]
        return cols

    @staticmethod
    def hybrid_feature_cols() -> list:
        return Preprocessor.num_feature_cols() + ["sentiment"]
