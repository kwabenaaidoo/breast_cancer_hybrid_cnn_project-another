"""
main.py
========
End-to-end pipeline for "Hybrid CNN-Based Models for Breast Cancer
Prediction" (CNN-SVM, CNN-LR, CNN-KNN, with and without PCA).

Run from the project root:

    python main.py                # full run (as specified in the brief)
    python main.py --quick         # fast smoke-test (fewer CNN epochs/grid)
    python main.py --folds 10      # use 10-fold CV instead of the default 5

Outputs:
    outputs/figures/   -> all PNG plots (Section 9)
    outputs/results/   -> CSV tables: CV results, tuning summary,
                           final comparison table (Section 10)
    outputs/models/    -> saved CNN model (.keras)
"""

import argparse
import os
import sys
import time
import warnings

# Harmless version-specific FutureWarnings (e.g. sklearn's LogisticRegression
# 'penalty' argument naming) are silenced here so console output stays
# readable; nothing about the computed results is affected.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src import data_preprocessing as dp
from src import cnn_feature_extractor as cfe
from src import hybrid_models as hm
from src import cross_validation as cv
from src import hyperparameter_tuning as ht
from src import evaluation as ev
from src import visualization as viz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                    help="Fast smoke-test run: fewer CNN epochs, smaller "
                         "tuning grid, 3-fold CV. Use this first to confirm "
                         "everything runs before the full experiment.")
    p.add_argument("--folds", type=int, default=config.CV_FOLDS,
                    help=f"Number of Stratified K-Folds (default {config.CV_FOLDS}).")
    p.add_argument("--data", type=str, default=config.DATA_PATH,
                    help="Path to the WBCD/WDBC CSV file.")
    return p.parse_args()


def main():
    args = parse_args()
    config.set_global_seed()
    t_start = time.time()

    # -----------------------------------------------------------------
    # 1. Load, clean, split, scale the raw tabular data
    # -----------------------------------------------------------------
    print("\n=== [1/8] Loading & preprocessing data ===")
    data = dp.prepare_dataset(path=args.data)
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # -----------------------------------------------------------------
    # 2. Train (and lightly tune) the 1D CNN feature extractor
    # -----------------------------------------------------------------
    print("\n=== [2/8] Training / tuning CNN feature extractor ===")
    if args.quick:
        grid = [{"epochs": 15, "batch_size": 16, "lr": 1e-3}]
    else:
        grid = config.CNN_TUNING_GRID

    best_cnn, best_history, best_cfg, cnn_tuning_df = cfe.tune_cnn(
        X_train, y_train, X_val, y_val, grid=grid, verbose=0)
    print(f"[main] Best CNN config: {best_cfg}")
    cnn_tuning_df.to_csv(
        os.path.join(config.RESULTS_DIR, "cnn_tuning_results.csv"), index=False)

    best_cnn.save(os.path.join(config.MODELS_DIR, "cnn_feature_extractor.npz"))
    viz.plot_cnn_training_curves(best_history)

    # -----------------------------------------------------------------
    # 3. Extract CNN features for train/val/test
    #    (train+val are combined into one training feature pool for the
    #     classical-ML stage; the test set stays completely held out)
    # -----------------------------------------------------------------
    print("\n=== [3/8] Extracting CNN feature vectors ===")
    extractor = cfe.build_feature_extractor(best_cnn)

    feats_train = cfe.extract_features(extractor, X_train)
    feats_val = cfe.extract_features(extractor, X_val)
    feats_test = cfe.extract_features(extractor, X_test)

    import numpy as np
    X_train_feats = np.vstack([feats_train, feats_val])
    y_train_full = np.concatenate([y_train, y_val])
    X_test_feats = feats_test

    print(f"[main] CNN feature matrix -> train: {X_train_feats.shape}, "
          f"test: {X_test_feats.shape}")

    # -----------------------------------------------------------------
    # 4. PCA explained-variance diagnostic plot
    # -----------------------------------------------------------------
    print("\n=== [4/8] PCA explained variance analysis ===")
    _, n_components_95 = viz.plot_pca_explained_variance(X_train_feats)
    print(f"[main] ~{n_components_95} components needed to retain "
          f"{int(config.PCA_VARIANCE_RETAINED*100)}% variance.")

    # -----------------------------------------------------------------
    # 5. Build all six hybrid pipelines
    # -----------------------------------------------------------------
    model_specs = hm.get_all_model_specs()

    # -----------------------------------------------------------------
    # 6. Stratified K-Fold cross-validation (Section 6)
    # -----------------------------------------------------------------
    print(f"\n=== [5/8] {args.folds}-Fold Stratified Cross-Validation ===")
    n_splits = 3 if args.quick else args.folds
    cv_results_df = cv.run_cross_validation(model_specs, X_train_feats,
                                             y_train_full, n_splits=n_splits)
    cv_results_df.to_csv(
        os.path.join(config.RESULTS_DIR, "cross_validation_results.csv"),
        index=False)

    # -----------------------------------------------------------------
    # 7. Hyperparameter tuning via GridSearchCV (Section 7)
    # -----------------------------------------------------------------
    print("\n=== [6/8] Hyperparameter tuning (GridSearchCV) ===")
    if args.quick:
        # Shrink grids drastically for the smoke test so it finishes fast.
        quick_grids = {
            "clf__kernel": ["rbf"], "clf__C": [1], "clf__gamma": ["scale"],
        }
        quick_lr_grid = {"clf__penalty": ["l2"], "clf__C": [1], "clf__solver": ["liblinear"]}
        quick_knn_grid = {"clf__n_neighbors": [5], "clf__metric": ["euclidean"]}
        for name, (builder, _grid, use_pca) in list(model_specs.items()):
            if "SVM" in name:
                model_specs[name] = (builder, quick_grids, use_pca)
            elif "LR" in name:
                model_specs[name] = (builder, quick_lr_grid, use_pca)
            elif "KNN" in name:
                model_specs[name] = (builder, quick_knn_grid, use_pca)
    best_estimators, tuning_summary_df = ht.tune_all_models(
        model_specs, X_train_feats, y_train_full, n_splits=n_splits)
    tuning_summary_df.to_csv(
        os.path.join(config.RESULTS_DIR, "hyperparameter_tuning_summary.csv"),
        index=False)

    # -----------------------------------------------------------------
    # 8. Final held-out test-set evaluation + comparison table
    # -----------------------------------------------------------------
    print("\n=== [7/8] Final evaluation on held-out test set ===")
    summary_df, detail = ev.evaluate_all(
        best_estimators, X_train_feats, y_train_full, X_test_feats, y_test)
    summary_df.to_csv(
        os.path.join(config.RESULTS_DIR, "final_comparison_table.csv"),
        index=False)

    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE (Section 10)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    print("\n=== [8/8] Generating remaining visualizations ===")
    viz.plot_roc_curves(detail)
    viz.plot_confusion_matrices(detail)
    viz.plot_comparison_bars(summary_df)

    # Save classification reports as text
    with open(os.path.join(config.RESULTS_DIR, "classification_reports.txt"), "w") as f:
        for name, result in detail.items():
            f.write(f"\n{'='*70}\n{name}\n{'='*70}\n")
            f.write(result["classification_report"])
            f.write("\n")

    elapsed = time.time() - t_start
    print(f"\n[main] Full pipeline finished in {elapsed/60:.1f} minutes.")
    print(f"[main] Figures  -> {config.FIGURES_DIR}")
    print(f"[main] Results  -> {config.RESULTS_DIR}")
    print(f"[main] Model    -> {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
