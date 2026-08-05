"""
config.py
---------
Global configuration: paths, random seeds, and experiment constants.
Editing this file is the single place to change behaviour for the whole
project (dataset location, split ratios, CNN settings, CV folds, etc.)
"""

import os
import random
import numpy as np

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix every relevant RNG so results are reproducible run-to-run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

for _d in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------
# Data split ratios (train / validation / test == 70 / 15 / 15)
# ---------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE_OF_REMAINDER = 0.15 / (1 - TEST_SIZE)  # ~0.1765, applied after test split

# ---------------------------------------------------------------------
# CNN feature extractor settings
# ---------------------------------------------------------------------
CNN_FEATURE_DIM = 32          # size of the penultimate (feature) layer
CNN_DEFAULT_EPOCHS = 100
CNN_DEFAULT_BATCH_SIZE = 16
CNN_DEFAULT_LR = 1e-3
CNN_EARLY_STOP_PATIENCE = 12

# Small grid used for the CNN's own hyperparameter search (Section 7)
CNN_TUNING_GRID = [
    {"epochs": 60,  "batch_size": 16, "lr": 1e-3},
    {"epochs": 100, "batch_size": 16, "lr": 1e-3},
    {"epochs": 100, "batch_size": 32, "lr": 5e-4},
    {"epochs": 150, "batch_size": 32, "lr": 1e-3},
]

# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------
CV_FOLDS = 5                  # set to 10 for 10-fold CV
PCA_VARIANCE_RETAINED = 0.95  # n_components=0.95 -> keep ~95% variance

# ---------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------
POSITIVE_CLASS = "M"  # Malignant -> 1
NEGATIVE_CLASS = "B"  # Benign    -> 0
