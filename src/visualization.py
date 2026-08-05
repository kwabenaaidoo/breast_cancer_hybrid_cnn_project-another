"""
visualization.py
------------------
All plots requested in Section 9:
  - CNN training/validation accuracy & loss curves
  - ROC curves for all models (overlaid)
  - Confusion matrix heatmaps
  - PCA explained-variance curve
Every function saves a PNG into outputs/figures/ AND returns the
matplotlib Figure (so it can also be displayed in a notebook).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless / VS Code "Run" execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config

sns.set_theme(style="whitegrid")


def _savefig(fig, filename):
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[visualization] saved {path}")
    return path


def plot_cnn_training_curves(history, filename="cnn_training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy")
    axes[1].legend()

    fig.suptitle("CNN Feature Extractor: Training History")
    fig.tight_layout()
    _savefig(fig, filename)
    return fig


def plot_roc_curves(detail: dict, filename="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in detail.items():
        fpr, tpr = result["roc_curve"]
        auc = result["metrics"]["ROC-AUC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Hybrid Models")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _savefig(fig, filename)
    return fig


def plot_confusion_matrices(detail: dict, filename="confusion_matrices.png"):
    n = len(detail)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, result) in zip(axes, detail.items()):
        cm = result["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(detail):]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices — All Hybrid Models")
    fig.tight_layout()
    _savefig(fig, filename)
    return fig


def plot_pca_explained_variance(X_train_feats: np.ndarray,
                                 filename="pca_explained_variance.png"):
    """
    Fit PCA on the (scaled) CNN-extracted TRAIN features purely to
    visualize how many components are needed to reach ~95% variance.
    This plot is diagnostic only -- it is NOT the PCA used inside the
    cross-validated pipelines (those are refit per-fold).
    """
    X_scaled = StandardScaler().fit_transform(X_train_feats)
    pca_full = PCA(random_state=config.RANDOM_SEED).fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    n_95 = int(np.argmax(cum_var >= config.PCA_VARIANCE_RETAINED) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=3)
    ax.axhline(config.PCA_VARIANCE_RETAINED, color="red", linestyle="--",
               label=f"{int(config.PCA_VARIANCE_RETAINED*100)}% variance")
    ax.axvline(n_95, color="green", linestyle="--",
               label=f"{n_95} components")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance — CNN-Extracted Features")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, filename)
    return fig, n_95


def plot_comparison_bars(summary_df, filename="model_comparison_bars.png"):
    """Grouped bar chart: Accuracy / F1 / ROC-AUC for every model."""
    metrics = ["Accuracy", "F1", "ROC-AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        pivot = summary_df.pivot(index="Base Model", columns="PCA", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=30)
        ax.set_title(metric)
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel(metric)
    fig.suptitle("Model Comparison: With vs. Without PCA")
    fig.tight_layout()
    _savefig(fig, filename)
    return fig
