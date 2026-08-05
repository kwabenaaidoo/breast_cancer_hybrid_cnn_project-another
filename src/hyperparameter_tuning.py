"""
hyperparameter_tuning.py
--------------------------
GridSearchCV over each classifier's pipeline (StandardScaler -> [PCA]
-> Classifier). GridSearchCV internally performs its own Stratified
K-Fold CV and refits the WHOLE pipeline (scaler + PCA + classifier)
on each candidate/fold combination, so PCA never sees data outside
the fold it is being fit on.
"""

import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config


def tune_all_models(model_specs: dict, X_train_feats, y_train,
                     n_splits: int = config.CV_FOLDS, scoring: str = "f1"):
    """
    Runs GridSearchCV for every entry in model_specs.
    Returns:
        best_estimators: dict[name] -> fitted best Pipeline (refit=True)
        tuning_summary:  DataFrame with best params / best CV score / time
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=config.RANDOM_SEED)
    best_estimators = {}
    summary_rows = []

    for name, (builder, param_grid, use_pca) in model_specs.items():
        pipeline = builder(use_pca=use_pca)
        search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=skf,
            scoring=scoring, n_jobs=-1, refit=True,
        )
        t0 = time.time()
        search.fit(X_train_feats, y_train)
        elapsed = time.time() - t0

        best_estimators[name] = search.best_estimator_
        summary_rows.append({
            "Model": name,
            "Best Params": search.best_params_,
            f"Best CV {scoring}": search.best_score_,
            "Search Time (s)": elapsed,
        })
        print(f"[hyperparameter_tuning] {name}: best_{scoring}="
              f"{search.best_score_:.4f}  params={search.best_params_}  "
              f"({elapsed:.1f}s)")

    tuning_summary = pd.DataFrame(summary_rows)
    return best_estimators, tuning_summary
