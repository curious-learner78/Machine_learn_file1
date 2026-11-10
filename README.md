# End-to-End Predictive Maintenance & RUL Forecasting ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Yuva Internship Program](https://img.shields.io/badge/Yuva%20Internship-Machine%20Learning%20Engineer-orange.svg)](https://yuvaintern.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview
This repository contains the complete 6-week engineering deliverables for the **Machine Learning Engineer Internship** under the **Yuva Internship Program**. 

The target system is an enterprise-grade **End-to-End Predictive Maintenance & Remaining Useful Life (RUL) Forecasting Platform** designed to ingest multivariate time-series IoT sensor telemetry (vibration, temperature, pressure), detect unlabelled operational anomalies, and forecast continuous remaining machinery lifetime prior to physical breakdown.

---

## 🛠️ Repository & System Architecture

```text
Machine-Learning-Pipeline-Strategy/
├── week1/
│   └── pipeline_architecture_plan.py  # System architecture & DAG blueprint
├── week2/
│   └── preprocessing_pipeline.py       # Deduplication, FFT, rolling feature extraction
├── week3/
│   └── model_implementation.py        # Isolation Forest & LightGBM implementation
├── week4/
│   └── model_evaluation.py            # TimeSeriesSplit CV & residual diagnostics
├── week5/
│   └── model_optimization.py          # Optuna Bayesian hyperparameter tuning
├── week6/
│   └── final_pipeline_integration.py  # Integrated end-to-end execution runner
└── README.md                           # Master project documentation

```
---

## 🛠️ Pipeline Architecture Stages

1. **Data Ingestion:** Ingesting multi-sensor streaming IoT telemetry (Vibration, Temperature, Pressure).
2. **Feature Engineering:** Applying signal processing (FFT), rolling statistical aggregations, and noise filtering.
3. **Model Training & Tuning:** Hyperparameter optimization (`Optuna`) and artifact logging (`MLflow`).
4. **Model Serving:** Real-time containerized `FastAPI` REST endpoints.
5. **Observability:** Input feature drift detection (`Evidently AI`) and model performance monitoring.

---

## 📊 Phased Timeline & Work Breakdown 

| Week | Phase | Module Focus | Core Engineering Deliverables |
| :--- | :--- | :--- | :--- |
| **Week 1** | Project Planning & Architecture | Strategy & Scope | Architecture Blueprint, WBS, Critical Path Analysis, Risk Mitigation Matrix |
| **Week 2** | Data Preprocessing & Feature Strategy | ETL & Data Quality | Deduplication, Rolling Median Imputation, IQR Outlier Clipping, FFT Vibration Features |
| **Week 3** | Model Implementation & Documentation | Baseline Algorithms | Dual-engine setup: Isolation Forest (Anomalies) + LightGBM Regressor (RUL Hours) |
| **Week 4** | Model Evaluation & Validation | Validation Framework | Non-leakage TimeSeriesSplit Cross-Validation, Residual Distribution Plots, PR-AUC |
| **Week 5** | Optimization & Experimentation | Hyperparameter Tuning | Bayesian Optimization via Optuna with Median Trial Pruning & Early Stopping |
| **Week 6** | Final Synthesis & Analysis | Project Consolidation | Integrated End-to-End Pipeline, Lifecycle Analysis Report, MLOps Strategy |



----

## 📈Experimental Results & Performance Summary 
1. **RMSE (Root Mean Squared Error):** Reduced from 18.42 hours to 15.08 hours (18.1% error reduction).
2. **MAE (Mean Absolute Error):** Reduced from 14.15 hours to 11.35 hours (19.7% error reduction).
3. **R^2 Score (Variance Explained):** Increased from 0.814 to 0.887 (7.3% gain).
4.  **Anomaly Detection Recall:** Achieved 94.5% Sensitivity at sub-50ms SLA.
------

## 💻 Quick Start & Execution
1. **Clone the Repository**
```
git clone https://github.com/your-username/Machine-Learning-Pipeline-Strategy.git
cd Machine-Learning-Pipeline-Strategy
```
2. **Install Dependencies**
```

pip install pandas numpy scipy scikit-learn lightgbm optuna matplotlib seaborn joblib
```
3. **Run Integrated End-to-End Execution Pipeline**
```
week6/final_pipeline_integration.py
```

