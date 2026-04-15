import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_split_data
from preprocess import preprocess_data

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def tune_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_cv_score = grid.best_score_

    return (
    grid.best_estimator_,
    grid.best_score_,
    grid.best_params_,
    grid.cv_results_["std_test_score"][grid.best_index_]
    )

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data(
        url="https://raw.githubusercontent.com/Sabaae/Dataset/main/LifeExpectancy.csv",
        random_state=0
    )

    # Preprocess
    X_train_scaled, X_test_scaled, imputer, scaler = preprocess_data(X_train, X_test)

    models = {
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": list(range(1, 50))}
    ),

    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1, 10]}
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [3, 5, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=0),
        {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=0
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(random_state=0),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10]
        }
    )
    }
    
    results = []

    for name, (model, params) in models.items():
        best_model, best_cv, best_params, std_score = tune_model(model, params, X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        results.append([
        name,
        best_params,
        best_cv,
        std_score,
        test_acc,
        test_f1
        ])

    results_df = pd.DataFrame(results, columns=[
    "Model",
    "Best Params",
    "CV Accuracy",
    "CV Std",
    "Test Accuracy",
    "Test F1"
    ])

    # Save CSV results
    results_path = os.path.join(RESULTS_DIR, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\nModel Comparison:\n")
    print(results_df)
        
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()