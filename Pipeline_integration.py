
```python
"""End-to-End Integrated Machine Learning Pipeline Execution Script.

Task: Week 6 Task - Final Report and Comprehensive Analysis of ML Project
Track: Machine Learning Engineer (Yuva Internship Program)
"""

import os
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class EndToEndMLPipeline:
  """Integrated Production-Ready Machine Learning Pipeline Controller."""

  def __init__(self, random_state: int = 42):
    self.random_state = random_state
    self.scaler = StandardScaler()
    self.anomaly_model = IsolationForest(
        n_estimators=100, contamination=0.05, random_state=self.random_state
    )
    self.regressor_model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=self.random_state,
    )

  def preprocess_data(
      self, df: pd.DataFrame, feature_cols: list
  ) -> pd.DataFrame:
    """Applies deduplication, forward-fill imputation, and feature scaling."""
    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)
    df_clean.ffill(inplace=True)
    df_clean.bfill(inplace=True)

    # Scaling
    df_clean[feature_cols] = self.scaler.fit_transform(df_clean[feature_cols])
    return df_clean

  def run_end_to_end(
      self, df: pd.DataFrame, feature_cols: list, target_col: str
  ) -> Dict[str, float]:
    """Runs complete ingestion, anomaly detection, regression, and validation loop."""
    print("=" * 80)
    print("RUNNING END-TO-END INTEGRATED ML PIPELINE EXECUTION")
    print("=" * 80 + "\n")

    # Step 1: Preprocess Data
    clean_df = self.preprocess_data(df, feature_cols)
    X = clean_df[feature_cols]
    y = clean_df[target_col]

    # Step 2: Anomaly Detection Scoring
    print("[STEP 1] Executing Isolation Forest Anomaly Detection...")
    self.anomaly_model.fit(X)
    anomaly_flags = np.where(self.anomaly_model.predict(X) == -1, 1, 0)
    print(
        f" -> Identified {np.sum(anomaly_flags)} operational anomaly signals."
    )

    # Step 3: Time-Series CV Evaluation for Regression
    print("\n[STEP 2] Running Time-Series Cross-Validation for RUL Regression...")
    tscv = TimeSeriesSplit(n_splits=4)
    rmse_list, mae_list, r2_list = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), start=1):
      X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
      y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

      self.regressor_model.fit(X_tr, y_tr)
      preds = self.regressor_model.predict(X_val)

      rmse = np.sqrt(mean_squared_error(y_val, preds))
      mae = mean_absolute_error(y_val, preds)
      r2 = r2_score(y_val, preds)

      rmse_list.append(rmse)
      mae_list.append(mae)
      r2_list.append(r2)

      print(
          f" Fold {fold} - RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2 Score:"
          f" {r2:.4f}"
      )

    metrics = {
        "Mean_RMSE": float(np.mean(rmse_list)),
        "Mean_MAE": float(np.mean(mae_list)),
        "Mean_R2": float(np.mean(r2_list)),
    }

    print("\n" + "=" * 80)
    print("FINAL INTEGRATED EVALUATION RESULTS")
    print("=" * 80)
    print(f"Mean CV RMSE : {metrics['Mean_RMSE']:.4f}")
    print(f"Mean CV MAE  : {metrics['Mean_MAE']:.4f}")
    print(f"Mean CV R²   : {metrics['Mean_R2']:.4f}")

    return metrics


if __name__ == "__main__":
  np.random.seed(42)
  records = 300

  sample_data = pd.DataFrame({
      "vibration_x_mean": np.random.normal(0.5, 0.1, records),
      "temperature_c_mean": np.random.normal(70.0, 3.0, records),
      "pressure_psi_mean": np.random.normal(120.0, 4.0, records),
  })

  sample_data["rul_hours"] = pd.Series(
      1000
      - (sample_data["vibration_x_mean"] * 500)
      + np.random.normal(0, 10, records)
  )

  pipeline = EndToEndMLPipeline(random_state=42)
  results = pipeline.run_end_to_end(
      sample_data,
      feature_cols=[
          "vibration_x_mean",
          "temperature_c_mean",
          "pressure_psi_mean",
      ],
      target_col="rul_hours",
  )
