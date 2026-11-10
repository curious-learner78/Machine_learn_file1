"""Data Preprocessing and Feature Engineering Pipeline Module.

Task: Week 2 Task - Data Preprocessing and Feature Engineering Strategy
Internship: Machine Learning Engineer (Yuva Internship Program)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
  """Handles duplicate records, missing values, and time-series imputation."""
  df = df.copy()

  # Drop exact duplicates
  initial_rows = len(df)
  df.drop_duplicates(inplace=True)
  print(
      f'[INFO] Removed {initial_rows - len(df)} duplicate records from dataset.'
  )

  # Forward fill short sensor dropouts, then fill remaining with rolling median
  df.ffill(inplace=True)
  df.bfill(inplace=True)
  print('[INFO] Missing value imputation completed successfully.')
  return df


def remove_outliers_iqr(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
  """Removes extreme outliers using the Interquartile Range (IQR) method."""
  df = df.copy()
  for col in target_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = df[
        (df[col] < lower_bound) | (df[col] > upper_bound)
    ].shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(
        f'[INFO] Filtered {outliers_count} outliers from column "{col}" using'
        ' IQR.'
    )
  return df


def engineer_rolling_features(
    df: pd.DataFrame, target_col: str, window_size: int = 5
) -> pd.DataFrame:
  """Extracts rolling window statistical features for time-series telemetry."""
  df = df.copy()
  rolling_window = df[target_col].rolling(window=window_size)
  df[f'{target_col}_rolling_mean'] = rolling_window.mean()
  df[f'{target_col}_rolling_std'] = rolling_window.std()
  df.fillna(0, inplace=True)
  print(f'[INFO] Engineered rolling window features for: {target_col}')
  return df


def scale_features(
    df: pd.DataFrame, feature_cols: list
) -> tuple[pd.DataFrame, StandardScaler]:
  """Scales continuous features using StandardScaler."""
  df = df.copy()
  scaler = StandardScaler()
  df[feature_cols] = scaler.fit_transform(df[feature_cols])
  print('[INFO] Feature scaling completed using StandardScaler.')
  return df, scaler


if __name__ == '__main__':
  print('Running Preprocessing & Feature Engineering Pipeline Execution...\n')

  # Create mock time-series sensor data
  np.random.seed(42)
  data = {
      'timestamp': pd.date_range(
          start='2026-10-01', periods=100, freq='min'
      ),
      'vibration_x': np.random.normal(0.5, 0.1, 100),
      'temperature_c': np.random.normal(70.0, 2.5, 100),
  }
  sample_df = pd.DataFrame(data)

  # Introduce a deliberate duplicate and outlier for testing cleaning functions
  sample_df.loc[10] = sample_df.loc[9]
  sample_df.loc[5, 'vibration_x'] = 15.0  # Outlier

  # Execute pipeline steps
  cleaned_df = clean_and_impute_data(sample_df)
  filtered_df = remove_outliers_iqr(
      cleaned_df, target_cols=['vibration_x', 'temperature_c']
  )
  featured_df = engineer_rolling_features(filtered_df, target_col='vibration_x')
  scaled_df, feature_scaler = scale_features(
      featured_df, feature_cols=['vibration_x', 'temperature_c']
  )

  print('\nProcessed Data Preview:')
  print(scaled_df.head())
