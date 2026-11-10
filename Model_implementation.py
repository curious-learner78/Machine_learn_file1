"""Machine Learning Model Implementation and Documentation Module.

Task: Week 3 Task - Model Implementation and Code Documentation
Internship: Machine Learning Engineer (Yuva Internship Program)
"""

from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd

try:
  import lightgbm as lgb
  HAS_LIGHTGBM = True
except ImportError:
  from sklearn.ensemble import GradientBoostingRegressor

  HAS_LIGHTGBM = False

from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class PredictiveMaintenanceModel:
  """Modular Machine Learning Model Manager for Anomaly Detection and RUL Regression."""

  def __init__(
      self, random_state: int = 42, n_estimators: int = 100, learning_rate: float = 0.05
  ):
    """Initializes the model architecture parameters and internal states."""
    self.random_state = random_state
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate

    # Instantiate anomaly detection model
    self.anomaly_detector = IsolationForest(
        n_estimators=self.n_estimators,
        contamination=0.05,
        random_state=self.random_state,
    )

    # Instantiate RUL regressor based on library availability
    if HAS_LIGHTGBM:
      self.rul_regressor = lgb.LGBMRegressor(
          n_estimators=self.n_estimators,
          learning_rate=self.learning_rate,
          random_state=self.random_state,
          verbosity=-1,
      )
    else:
      self.rul_regressor = GradientBoostingRegressor(
          n_estimators=self.n_estimators,
          learning_rate=self.learning_rate,
          random_state=self.random_state,
      )

  def train_anomaly_detector(self, X: pd.DataFrame) -> np.ndarray:
    """Fits the Isolation Forest anomaly detector and returns operational state predictions."""
    print("[INFO] Training Isolation Forest Anomaly Detector...")
    self.anomaly_detector.fit(X)
    predictions = self.anomaly_detector.predict(X)
    # Convert predictions (-1: anomaly, 1: normal) to binary flags (1: anomaly, 0: normal)
    binary_anomalies = np.where(predictions == -1, 1, 0)
    print(
        f"[INFO] Anomaly detection complete. Identified {np.sum(binary_anomalies)} anomalies."
    )
    return binary_anomalies

  def train_rul_regressor(
      self, X_train: pd.DataFrame, y_train: pd.Series
  ) -> None:
    """Fits the LightGBM/GradientBoosting regressor on RUL target variables."""
    print("[INFO] Fitting RUL Regressor Engine...")
    try:
      self.rul_regressor.fit(X_train, y_train)
      print("[INFO] Model fitting completed successfully.")
    except Exception as e:
      print(f"[ERROR] Model training failed: {str(e)}")
      raise e

  def evaluate_model(
      self, X_test: pd.DataFrame, y_test: pd.Series
  ) -> Dict[str, float]:
    """Evaluates trained regressor model performance metrics."""
    predictions = self.rul_regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {"RMSE": float(rmse), "R2_Score": float(r2)}
    print(f"[METRICS] Test Set RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
    return metrics

  def save_model_artifacts(self, filepath: str = "rul_model.joblib") -> None:
    """Serializes the trained regressor model object to disk."""
    joblib.dump(self.rul_regressor, filepath)
    print(f"[ARTIFACT] Model successfully saved to {filepath}")


if __name__ == "__main__":
  print("Initializing Machine Learning Model Implementation Script...\n")

  # Generate synthetic dataset for model execution testing
  np.random.seed(42)
  num_samples = 200

  features = pd.DataFrame({
      "vibration_x_mean": np.random.normal(0.5, 0.1, num_samples),
      "vibration_x_std": np.random.normal(0.05, 0.01, num_samples),
      "temperature_c_mean": np.random.normal(70.0, 3.0, num_samples),
      "pressure_psi_mean": np.random.normal(120.0, 5.0, num_samples),
  })

  # Synthetic target: Remaining Useful Life in hours
  rul_target = pd.Series(
      1000 - (features["vibration_x_mean"] * 500) + np.random.normal(0, 10, num_samples)
  )

  # Pipeline execution
  pipeline = PredictiveMaintenanceModel(n_estimators=50, learning_rate=0.05)

  # Step 1: Run Anomaly Detection
  anomalies = pipeline.train_anomaly_detector(features)

  # Step 2: Train/Test Split for RUL Regression
  X_tr, X_te, y_tr, y_te = train_test_split(
      features, rul_target, test_size=0.2, random_state=42
  )

  # Step 3: Train & Evaluate RUL Regressor
  pipeline.train_rul_regressor(X_tr, y_tr)
  eval_results = pipeline.evaluate_model(X_te, y_te)

  # Step 4: Save Model Artifact
  pipeline.save_model_artifacts("week3_rul_model.joblib")
      
