"""
random_forest_model.py
----------------------
Random Forest Regressor with optional GridSearchCV tuning.
Supports:
  - baseline (numerical) vs hybrid (+ sentiment/NLP) comparison
  - feature importance extraction
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score


class RFModel:
    def __init__(self, tune: bool = True, cv_folds: int = 3):
        self.tune     = tune
        self.cv_folds = cv_folds
        self.model    = None
        self.best_params_: dict = {}
        self.feature_names: list = []

    def _build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        estimator = self._build_estimator()

        if self.tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth":    [5, 10, None],
                "min_samples_split": [2, 5],
            }
            gs = GridSearchCV(
                estimator, param_grid,
                cv=self.cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1, verbose=0,
            )
            gs.fit(X_train, y_train)
            self.model       = gs.best_estimator_
            self.best_params_ = gs.best_params_
            print(f"[RF] Best params: {self.best_params_}")
        else:
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        if feature_names:
            self.feature_names = feature_names
        print("[RF] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> dict:
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp))
