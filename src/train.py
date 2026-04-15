import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_split_data

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint, uniform

from evaluation.shap_analysis import run_shap_analysis
from evaluation.feature_importance import plot_feature_importance
from evaluation.error_analysis import analyze_errors
from evaluation.model_metrics import evaluate_model
from utils.io import save_model

np.random.seed(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def tune_model(model, param_dist, X_train, y_train, use_scaler=True):

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    pipe = Pipeline(steps)

    # Prefix params
    param_dist = {f"model__{k}": v for k, v in param_dist.items()}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=25,  # key control (not full grid)
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=0
    )

    random_search.fit(X_train, y_train)

    return (
        random_search.best_estimator_,
        random_search.best_score_,
        random_search.best_params_,
        random_search.cv_results_["std_test_score"][random_search.best_index_]
    )

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data(
        url="https://raw.githubusercontent.com/Sabaae/Dataset/main/LifeExpectancy.csv",
        random_state=0
    )

    models = {

        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": list(range(3, 51, 2)),  # odd numbers (avoid ties)
                "weights": ["uniform", "distance"],
                "p": [1, 2]  # Manhattan vs Euclidean
            }
        ),

        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=0),
            {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs"]
            }
        ),

        "Decision Tree": (
            DecisionTreeClassifier(random_state=0),
            {
                "max_depth": [3, 5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "criterion": ["gini", "entropy"]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=0),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            }
        ),

        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=0,
                verbosity=0,
                n_jobs=-1
            ),
            {
                "n_estimators": randint(100, 400),
                "max_depth": randint(3, 7),
                "learning_rate": uniform(0.01, 0.1),
                "subsample": uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.7, 0.3)
            }
        ),

        "Extra Trees": (
            ExtraTreesClassifier(random_state=0),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            }
        )
    }
    
    results = []

    best_model_overall = None
    best_score_overall = 0
    best_model_name = None

    for name, (model, params) in models.items():

        use_scaler = name in ["KNN", "Logistic Regression"]

        best_model, best_cv, best_params, std_score = tune_model(
            model,
            params,
            X_train,
            y_train,
            use_scaler=use_scaler
        )

        # Evaluate on test set
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

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

        # Track best model
        if best_cv > best_score_overall:
            best_score_overall = best_cv
            best_model_overall = best_model
            best_model_name = name

    print(f"\nBest Model: {best_model_name}")

    save_model(
        best_model_overall,
        os.path.join(RESULTS_DIR, "best_model.pkl")
    )

    y_pred_best = best_model_overall.predict(X_test)

    error_metrics = analyze_errors(y_test, y_pred_best)
    print("\nError Analysis:", error_metrics)

    final_estimator = best_model_overall.named_steps["model"]

    if hasattr(final_estimator, "feature_importances_"):
        plot_feature_importance(
    final_estimator,
    X_train.columns,
    save_path=os.path.join(RESULTS_DIR, "feature_importance.png")
    )
        
    else:
        print("\nFeature importance not available for this model.")

    #Results datafame
    results_df = pd.DataFrame(results, columns=[
    "Model",
    "Best Params",
    "CV F1 (Weighted)",
    "CV Std",
    "Test Accuracy",
    "Test F1"
    ])

    results_df = results_df.sort_values(by="Test Accuracy", ascending=False)

    # Save CSV results
    results_path = os.path.join(RESULTS_DIR, "model_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\nModel Comparison:\n")
    print(results_df)
        
    print(f"\nResults saved to: {results_path}")

    run_shap_analysis(
        best_model_overall,
        X_train,
        X_test,
        feature_names=X_train.columns,
        results_dir=RESULTS_DIR
    )

    evaluate_model(
    best_model_overall,
    X_test,
    y_test,
    RESULTS_DIR
    )

if __name__ == "__main__":
    main()