"""
hybrid_models.py
-----------------
Builds the six hybrid pipelines requested in the brief:
    CNN-SVM, CNN-LR, CNN-KNN   x   {without PCA, with PCA}

Every pipeline is: StandardScaler -> [PCA] -> Classifier
Because this is an sklearn Pipeline, StandardScaler and PCA are
refit from scratch on the training fold ONLY inside every single
cross-validation split and every GridSearchCV candidate -- so no
information from a validation/test fold ever leaks into fitting the
scaler or the PCA components. The CNN feature extraction itself is
done once beforehand (see main.py) using only the CNN's own
train/validation split, which never includes the held-out test set.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from . import config


def _base_steps(use_pca: bool):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=config.PCA_VARIANCE_RETAINED,
                                  random_state=config.RANDOM_SEED)))
    return steps


def make_svm_pipeline(use_pca: bool = False) -> Pipeline:
    steps = _base_steps(use_pca)
    steps.append(("clf", SVC(probability=True, random_state=config.RANDOM_SEED)))
    return Pipeline(steps)


def make_lr_pipeline(use_pca: bool = False) -> Pipeline:
    steps = _base_steps(use_pca)
    steps.append(("clf", LogisticRegression(
        max_iter=5000, random_state=config.RANDOM_SEED)))
    return Pipeline(steps)


def make_knn_pipeline(use_pca: bool = False) -> Pipeline:
    steps = _base_steps(use_pca)
    steps.append(("clf", KNeighborsClassifier()))
    return Pipeline(steps)


# ---------------------------------------------------------------------
# Hyperparameter search spaces (Section 7)
# ---------------------------------------------------------------------
SVM_PARAM_GRID = {
    "clf__kernel": ["linear", "rbf"],
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto"],
}

LR_PARAM_GRID = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__solver": ["liblinear"],   # supports both l1 and l2
}

KNN_PARAM_GRID = {
    "clf__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "clf__metric": ["euclidean", "manhattan", "minkowski"],
}


def get_all_model_specs():
    """
    Returns a dict describing every (model, pca) combination:
    name -> (pipeline_builder_fn, param_grid, use_pca)
    """
    specs = {}
    for use_pca, tag in [(False, "No PCA"), (True, "PCA")]:
        specs[f"CNN-SVM ({tag})"] = (make_svm_pipeline, SVM_PARAM_GRID, use_pca)
        specs[f"CNN-LR ({tag})"] = (make_lr_pipeline, LR_PARAM_GRID, use_pca)
        specs[f"CNN-KNN ({tag})"] = (make_knn_pipeline, KNN_PARAM_GRID, use_pca)
    return specs
