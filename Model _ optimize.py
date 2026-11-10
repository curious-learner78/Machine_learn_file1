"""Automated Model Optimization and Hyperparameter Tuning using Optuna.

Task: Week 5 Task - Model Optimization and Experimentation
Track: Machine Learning Engineer (Yuva Internship Program)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    import lightgbm as lgb
    HAS_OPTUNA_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_OPTUNA_LGB = False


class ModelOptimizer:
    """Bayesian Hyperparameter Optimization Manager using Optuna."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.best_params = {}

    def objective(self, trial: Any) -> float:
        """Optuna objective function for hyperparameter tuning via TimeSeriesSplit CV."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'random_state': self.random_state,
            'verbosity': -1
        }

        tscv = TimeSeriesSplit(n_splits=3)
        rmse_scores = []

        for train_idx, val_idx in tscv.split(self.X):
            X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            if HAS_OPTUNA_LGB:
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    random_state=params['random_state']
                )
                model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)

        return float(np.mean(rmse_scores))

    def run_optimization(self, n_trials: int = 20) -> Dict[str, Any]:
        """Executes the Bayesian optimization study."""
        print(f"\n[INFO] Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params
        print("\n=== OPTIMIZATION COMPLETE ===")
        print(f"Best Validation RMSE: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for key, value in self.best_params.items():
            print(f"  - {key}: {value}")

        return self.best_params


if __name__ == "__main__":
    print("Initializing Model Optimization Script...\n")

    # Generate synthetic time-series telemetry dataset
    np.random.seed(42)
    n_samples = 400

    features = pd.DataFrame({
        'vibration_rolling_mean': np.random.normal(0.5, 0.1, n_samples),
        'temperature_rolling_mean': np.random.normal(70.0, 3.0, n_samples),
        'pressure_rolling_mean': np.random.normal(120.0, 5.0, n_samples),
    })

    rul_target = pd.Series(
        1000 - (features['vibration_rolling_mean'] * 550) + np.random.normal(0, 10, n_samples)
    )

    # Execute Optimizer
    optimizer = ModelOptimizer(features, rul_target, random_state=42)
    best_hyperparameters = optimizer.run_optimization(n_trials=10)
              
