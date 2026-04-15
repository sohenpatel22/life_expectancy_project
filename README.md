# Machine Learning Pipeline for Classification (Life Expectancy Dataset)

This project implements a complete end-to-end machine learning pipeline for classification, including data preprocessing, hyperparameter tuning, model evaluation, and explainability.

---

## Project Overview

The system evaluates multiple machine learning models and selects the best-performing one based on cross-validation and test metrics.

It includes:
- Automated preprocessing using pipelines
- Hyperparameter tuning using RandomizedSearchCV
- Stratified cross-validation for robustness
- Comprehensive evaluation metrics
- Model explainability using feature importance and SHAP
- Full experiment artifact saving

---

## Models Implemented

The following models are compared:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Extra Trees Classifier

---

## Pipeline Design

Each model is trained using a reproducible pipeline:

- Missing value imputation (Median strategy)
- Standardization (for distance-based models)
- Model training
- Hyperparameter tuning using RandomizedSearchCV
- StratifiedKFold (5 splits, shuffled, seeded)

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score (weighted)
- ROC-AUC score
- Optimal classification threshold (F1-based)

Additionally, visual diagnostics are generated:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Classification Report

---

## Explainability

The project includes interpretability tools:

- Feature Importance (tree-based models)
- SHAP analysis for global and local explanations
- Error analysis for model failure inspection

---

## Output Structure

All results are saved in the `results/` directory:

results/
│── best_model.pkl
│── model_results.csv
│── evaluation_summary.json
│── confusion_matrix.png
│── roc_curve.png
│── precision_recall_curve.png
│── classification_report.txt
│── feature_importance.png
│── shap_summary.png


---

## Model Selection Strategy

The best model is selected based on:

- Cross-validation F1-score (primary metric)
- Stability across folds
- Test set performance

---