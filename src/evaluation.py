"""
evaluation.py
--------------
Final held-out TEST SET evaluation for each tuned model:
accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
plus training time bookkeeping used to build the comparison table
(Section 10).
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)

from . import config


def evaluate_model(name: str, estimator, X_train_feats, y_train,
                    X_test_feats, y_test):
    """
    Refits `estimator` (already the best pipeline from GridSearchCV) on
    the full training feature set, times that fit, then scores it on
    the untouched test set.
    """
    t0 = time.time()
    estimator.fit(X_train_feats, y_train)
    train_time = time.time() - t0

    y_pred = estimator.predict(X_test_feats)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test_feats)[:, 1]
    else:
        y_score = estimator.decision_function(X_test_feats)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_score),
        "Training Time (s)": train_time,
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    report = classification_report(y_test, y_pred,
                                    target_names=["Benign", "Malignant"])

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "y_pred": y_pred,
        "y_score": y_score,
        "classification_report": report,
        "fitted_estimator": estimator,
    }


def evaluate_all(best_estimators: dict, X_train_feats, y_train,
                  X_test_feats, y_test):
    """Evaluate every model in best_estimators; returns (summary_df, detail_dict)."""
    detail = {}
    rows = []
    for name, estimator in best_estimators.items():
        result = evaluate_model(name, estimator, X_train_feats, y_train,
                                 X_test_feats, y_test)
        detail[name] = result
        rows.append(result["metrics"])
        print(f"[evaluation] {name}: "
              f"acc={result['metrics']['Accuracy']:.4f}  "
              f"prec={result['metrics']['Precision']:.4f}  "
              f"rec={result['metrics']['Recall']:.4f}  "
              f"f1={result['metrics']['F1']:.4f}  "
              f"auc={result['metrics']['ROC-AUC']:.4f}")

    summary_df = pd.DataFrame(rows)

    # Split model name into base model + PCA flag for the comparison table
    summary_df["PCA"] = summary_df["Model"].apply(
        lambda n: "Yes" if "(PCA)" in n else "No")
    summary_df["Base Model"] = summary_df["Model"].apply(
        lambda n: n.split(" (")[0])
    ordered_cols = ["Base Model", "PCA", "Accuracy", "Precision", "Recall",
                     "F1", "ROC-AUC", "Training Time (s)"]
    summary_df = summary_df[ordered_cols].sort_values(
        ["Base Model", "PCA"]).reset_index(drop=True)

    return summary_df, detail
