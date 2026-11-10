"""Machine Learning Model Evaluation and Validation Framework.

Task: Week 4 Task - Model Evaluation and Validation Techniques
Track: Machine Learning Engineer (Yuva Internship Program)
"""

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit


class ModelEvaluator:
  """Evaluation and Diagnostic Suite for Classification & Regression ML Models."""

  def __init__(self, random_state: int = 42):
    self.random_state = random_state

  def evaluate_regression_metrics(
      self, y_true: np.ndarray, y_pred: np.ndarray
  ) -> Dict[str, float]:
    """Computes continuous metrics for Remaining Useful Life (RUL) regression."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2_Score": float(r2),
    }

    print("\n=== REGRESSION EVALUATION METRICS ===")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"Mean Absolute Error (MAE)      : {mae:.4f}")
    print(f"Coefficient of Determination R²: {r2:.4f}")
    return metrics

  def evaluate_classification_metrics(
      self, y_true: np.ndarray, y_pred: np.ndarray
  ) -> Dict[str, float]:
    """Computes binary classification metrics for anomaly detection."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
    }

    print("\n=== CLASSIFICATION EVALUATION METRICS ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    return metrics

  def perform_time_series_cv(
      self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
  ) -> Tuple[list, list]:
    """Executes forward-chaining TimeSeriesSplit cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    r2_scores = []

    print(
        f"\n=== EXECUTING TIME-SERIES CROSS-VALIDATION ({n_splits} SPLITS) ==="
    )

    fold = 1
    for train_index, test_index in tscv.split(X):
      X_tr, X_val = X.iloc[train_index], X.iloc[test_index]
      y_tr, y_val = y.iloc[train_index], y.iloc[test_index]

      model = RandomForestRegressor(
          n_estimators=50, random_state=self.random_state
      )
      model.fit(X_tr, y_tr)
      preds = model.predict(X_val)

      fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
      fold_r2 = r2_score(y_val, preds)

      rmse_scores.append(fold_rmse)
      r2_scores.append(fold_r2)

      print(
          f"Fold {fold}: Train Size = {len(X_tr)} | Test Size = {len(X_val)} |"
          f" Fold RMSE = {fold_rmse:.4f} | Fold R² = {fold_r2:.4f}"
      )
      fold += 1

    print(
        f"\nMean CV RMSE: {np.mean(rmse_scores):.4f} (± {np.std(rmse_scores):.4f})"
    )
    print(f"Mean CV R²  : {np.mean(r2_scores):.4f} (± {np.std(r2_scores):.4f})")
    return rmse_scores, r2_scores

  def plot_residual_diagnostics(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      filename: str = "residual_diagnostics.png",
  ) -> None:
    """Generates residual error diagnostic charts for model error analysis."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted vs Residuals plot
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0], alpha=0.7, color="navy")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residuals vs. Predicted Values")
    axes[0].set_xlabel("Predicted Remaining Useful Life (Hours)")
    axes[0].set_ylabel("Residuals (True - Predicted)")

    # Residual Distribution Histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color="teal", bins=20)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residual Distribution (Error Normality)")
    axes[1].set_xlabel("Residual Error")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[DIAGNOSTIC] Residual plot saved successfully to '{filename}'.")
    plt.close()

  def plot_confusion_matrix(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      filename: str = "confusion_matrix.png",
  ) -> None:
    """Generates a confusion matrix heatmap for classification evaluation."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title("Anomaly Detection Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(
        f"[DIAGNOSTIC] Confusion matrix saved successfully to '{filename}'."
    )
    plt.close()


if __name__ == "__main__":
  print("Initializing Machine Learning Model Evaluation & Validation Suite...\n")

  # Generate synthetic time-series data for validation tests
  np.random.seed(42)
  n_samples = 300

  feature_matrix = pd.DataFrame({
      "vibration_rolling_mean": np.random.normal(0.5, 0.1, n_samples),
      "temperature_rolling_mean": np.random.normal(70.0, 3.0, n_samples),
      "pressure_rolling_mean": np.random.normal(120.0, 5.0, n_samples),
  })

  # Target RUL (Remaining Useful Life) in hours
  true_rul = pd.Series(
      1000
      - (feature_matrix["vibration_rolling_mean"] * 600)
      + np.random.normal(0, 15, n_samples)
  )

  # Target Anomaly Class (0: Normal, 1: Anomaly)
  true_anomalies = np.where(feature_matrix["vibration_rolling_mean"] > 0.65, 1, 0)

  evaluator = ModelEvaluator(random_state=42)

  # 1. Execute Time-Series Cross-Validation
  evaluator.perform_time_series_cv(feature_matrix, true_rul, n_splits=5)

  # 2. Simulate Regressor Predictions & Evaluate Metrics
  mock_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
  mock_regressor.fit(feature_matrix[:200], true_rul[:200])
  predicted_rul = mock_regressor.predict(feature_matrix[200:])

  evaluator.evaluate_regression_metrics(
      true_rul[200:].to_numpy(), predicted_rul
  )
  evaluator.plot_residual_diagnostics(
      true_rul[200:].to_numpy(),
      predicted_rul,
      filename="week4_residuals.png",
  )

  # 3. Simulate Classification Anomaly Predictions & Evaluate Metrics
  iso_forest = IsolationForest(
      n_estimators=50, contamination=0.1, random_state=42
  )
  iso_forest.fit(feature_matrix[:200])
  raw_preds = iso_forest.predict(feature_matrix[200:])
  predicted_anomalies = np.where(raw_preds == -1, 1, 0)

  evaluator.evaluate_classification_metrics(
      true_anomalies[200:], predicted_anomalies
  )
  evaluator.plot_confusion_matrix(
      true_anomalies[200:],
      predicted_anomalies,
      filename="week4_confusion_matrix.png",
    )
    
