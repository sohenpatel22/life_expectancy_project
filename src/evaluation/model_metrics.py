import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(model, X_test, y_test, results_dir):

    # 1. Probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model does not support predict_proba")

    # 2. Optimal Threshold (F1-based)
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = (
        2 * precision_vals[:-1] * recall_vals[:-1]
        / (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    y_pred = (y_proba >= best_threshold).astype(int)

    # 3. Core Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure()
    disp.plot()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # 5. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # 6. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    pr_path = os.path.join(results_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()

    # 7. Classification Report
    report = classification_report(y_test, y_pred)

    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # 8. Save summary JSON
    summary = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_score,
        "optimal_threshold": float(best_threshold)
    }

    summary_path = os.path.join(results_dir, "evaluation_summary.json")

    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # 9. Logging
    print("\nEvaluation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved:")
    print(cm_path)
    print(roc_path)
    print(pr_path)
    print(report_path)
    print(summary_path)

    return summary