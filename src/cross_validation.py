"""
cross_validation.py
--------------------
Stratified K-Fold cross-validation on the CNN-extracted training
features, run separately for every classifier / PCA combination.
Reports mean +/- std for accuracy, precision, recall, F1 and ROC-AUC.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def run_cross_validation(model_specs: dict, X_train_feats, y_train,
                          n_splits: int = config.CV_FOLDS) -> pd.DataFrame:
    """
    model_specs: output of hybrid_models.get_all_model_specs()
    X_train_feats: CNN-extracted (not yet scaled/PCA'd) TRAIN features.
                   Scaling + PCA happen inside the pipeline, refit fresh
                   on each fold's training portion -> no leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=config.RANDOM_SEED)
    rows = []

    for name, (builder, _grid, use_pca) in model_specs.items():
        pipeline = builder(use_pca=use_pca)
        cv_results = cross_validate(
            pipeline, X_train_feats, y_train,
            cv=skf, scoring=SCORING, n_jobs=-1, error_score="raise",
        )
        row = {"Model": name}
        for metric in SCORING:
            scores = cv_results[f"test_{metric}"]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows.append(row)
        print(f"[cross_validation] {name}: "
              f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
              f"f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
              f"roc_auc={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}")

    return pd.DataFrame(rows)
