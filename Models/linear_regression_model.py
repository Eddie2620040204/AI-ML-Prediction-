"""
linear_regression_model.py
--------------------------
Baseline Linear Regression wrapper.
Supports:
  - fit / predict
  - feature importance via coefficients
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LRModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ])
        self.feature_names: list = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        self.pipeline.fit(X_train, y_train)
        if feature_names:
            self.feature_names = feature_names
        print("[LR] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def feature_importance(self) -> dict:
        coefs = self.pipeline.named_steps["lr"].coef_
        return dict(zip(self.feature_names, np.abs(coefs)))
