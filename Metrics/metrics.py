"""
metrics.py
----------
Common evaluation metrics for regression / forecasting.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    result = {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
    print(
        f"[{model_name:30s}]  RMSE={rmse:.4f}  MAE={mae:.4f}  "
        f"R²={r2:.4f}  MAPE={mape:.2f}%"
    )
    return result
