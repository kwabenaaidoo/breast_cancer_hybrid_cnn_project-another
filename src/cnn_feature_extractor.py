"""
cnn_feature_extractor.py
-------------------------
A 1D CNN trained as a binary classifier on the (already standardized)
tabular features, then used as a feature extractor: its penultimate
Dense layer becomes the learned feature representation fed into the
downstream classical classifiers (SVM / LR / KNN).

This is implemented entirely on top of `numpy_nn.py` -- there is NO
TensorFlow/PyTorch dependency anywhere in this project. That is a
deliberate design choice: numpy has universal prebuilt wheels for every
Python version/OS/architecture, so `pip install -r requirements.txt`
cannot fail the way large compiled deep-learning frameworks sometimes do
on constrained networks or newer/less-common Python builds.

Architecture (unchanged from the brief's Section 3):
    Conv1D(1->32, k=3) -> ReLU -> MaxPool1D(2)
    -> Conv1D(32->64, k=3) -> ReLU -> MaxPool1D(2)
    -> Flatten -> Dense(64) -> ReLU -> Dropout(0.3)
    -> Dense(feature_dim) -> ReLU   [<- extracted feature vector]
    -> Dropout(0.3) -> Dense(1) -> Sigmoid
"""

import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config
from . import numpy_nn as nn


class History:
    """Mimics the small subset of keras.callbacks.History used by visualization.py."""
    def __init__(self):
        self.history = {"loss": [], "val_loss": [],
                         "accuracy": [], "val_accuracy": [],
                         "auc": [], "val_auc": []}

    def append(self, loss, val_loss, acc, val_acc, auc, val_auc):
        self.history["loss"].append(loss)
        self.history["val_loss"].append(val_loss)
        self.history["accuracy"].append(acc)
        self.history["val_accuracy"].append(val_acc)
        self.history["auc"].append(auc)
        self.history["val_auc"].append(val_auc)


class CNN1D:
    """The hand-rolled 1D CNN classifier / feature extractor."""

    def __init__(self, input_dim, feature_dim=config.CNN_FEATURE_DIM,
                 seed=config.RANDOM_SEED):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv1D(1, 32, 3, rng)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1D(2)

        self.conv2 = nn.Conv1D(32, 64, 3, rng)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1D(2)

        self.flatten = nn.Flatten()

        L1 = input_dim
        L2 = math.ceil(L1 / 2)
        L3 = math.ceil(L2 / 2)
        flat_dim = L3 * 64

        self.dense1 = nn.Dense(flat_dim, 64, rng)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3, rng)

        self.dense_feat = nn.Dense(64, feature_dim, rng)
        self.relu_feat = nn.ReLU()
        self.drop2 = nn.Dropout(0.3, rng)

        self.dense_out = nn.Dense(feature_dim, 1, rng)

    # -------------------------------------------------------------
    def forward(self, x, training=True):
        """x: (N, input_dim) raw standardized tabular features."""
        h = x.reshape(x.shape[0], x.shape[1], 1)          # (N, L, 1)
        h = self.conv1.forward(h, training)
        h = self.relu1.forward(h, training)
        h = self.pool1.forward(h, training)

        h = self.conv2.forward(h, training)
        h = self.relu2.forward(h, training)
        h = self.pool2.forward(h, training)

        h = self.flatten.forward(h, training)
        h = self.dense1.forward(h, training)
        h = self.relu3.forward(h, training)
        h = self.drop1.forward(h, training)

        h = self.dense_feat.forward(h, training)
        feat = self.relu_feat.forward(h, training)         # <- feature vector

        h = self.drop2.forward(feat, training)
        logits = self.dense_out.forward(h, training).ravel()
        return logits, feat

    def backward(self, dlogits):
        dh = dlogits.reshape(-1, 1)
        dh = self.dense_out.backward(dh)
        dh = self.drop2.backward(dh)
        dfeat = self.relu_feat.backward(dh)
        dh = self.dense_feat.backward(dfeat)

        dh = self.drop1.backward(dh)
        dh = self.relu3.backward(dh)
        dh = self.dense1.backward(dh)
        dh = self.flatten.backward(dh)

        dh = self.pool2.backward(dh)
        dh = self.relu2.backward(dh)
        dh = self.conv2.backward(dh)

        dh = self.pool1.backward(dh)
        dh = self.relu1.backward(dh)
        self.conv1.backward(dh)

    def params_and_grads(self):
        d = {}
        d.update(self.conv1.params_and_grads("conv1"))
        d.update(self.conv2.params_and_grads("conv2"))
        d.update(self.dense1.params_and_grads("dense1"))
        d.update(self.dense_feat.params_and_grads("dense_feat"))
        d.update(self.dense_out.params_and_grads("dense_out"))
        return d

    def predict_proba(self, x):
        logits, _ = self.forward(x, training=False)
        return nn.sigmoid(logits)

    def extract_features(self, x):
        _, feat = self.forward(x, training=False)
        return feat

    def save(self, path):
        arrays = {name: param for name, (param, _grad) in self.params_and_grads().items()}
        arrays["input_dim"] = np.array(self.input_dim)
        arrays["feature_dim"] = np.array(self.feature_dim)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["input_dim"]), int(data["feature_dim"]))
        for name, (param, _grad) in model.params_and_grads().items():
            param[...] = data[name]
        return model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def _binary_cross_entropy(y_true, y_prob, eps=1e-7):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def train_cnn(X_train, y_train, X_val, y_val,
              epochs: int = config.CNN_DEFAULT_EPOCHS,
              batch_size: int = config.CNN_DEFAULT_BATCH_SIZE,
              learning_rate: float = config.CNN_DEFAULT_LR,
              verbose: int = 0):
    """Train the CNN with mini-batch Adam + early stopping on validation loss."""
    config.set_global_seed()
    rng = np.random.default_rng(config.RANDOM_SEED)

    model = CNN1D(input_dim=X_train.shape[1], seed=config.RANDOM_SEED)
    optimizer = nn.Adam(lr=learning_rate)
    history = History()

    n = X_train.shape[0]
    best_val_loss = np.inf
    best_params = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        # -- shuffle & mini-batch training --
        perm = rng.permutation(n)
        X_shuf, y_shuf = X_train[perm], y_train[perm]
        epoch_losses = []

        for start in range(0, n, batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size].astype(float)
            if len(xb) == 0:
                continue

            logits, _ = model.forward(xb, training=True)
            y_prob = nn.sigmoid(logits)
            loss = _binary_cross_entropy(yb, y_prob)
            epoch_losses.append(loss)

            # Combined sigmoid + BCE gradient wrt logits: (p - y) / N
            dlogits = (y_prob - yb) / len(xb)
            model.backward(dlogits)
            optimizer.step(model.params_and_grads())

        # -- epoch-end metrics --
        train_logits, _ = model.forward(X_train, training=False)
        train_prob = nn.sigmoid(train_logits)
        train_loss = _binary_cross_entropy(y_train.astype(float), train_prob)
        train_acc = ((train_prob >= 0.5).astype(int) == y_train).mean()
        train_auc = roc_auc_score(y_train, train_prob)

        val_logits, _ = model.forward(X_val, training=False)
        val_prob = nn.sigmoid(val_logits)
        val_loss = _binary_cross_entropy(y_val.astype(float), val_prob)
        val_acc = ((val_prob >= 0.5).astype(int) == y_val).mean()
        val_auc = roc_auc_score(y_val, val_prob)

        history.append(train_loss, val_loss, train_acc, val_acc, train_auc, val_auc)

        if verbose:
            print(f"  epoch {epoch+1}/{epochs} - loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        # -- early stopping (restore-best-weights style) --
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_params = {name: param.copy()
                            for name, (param, _g) in model.params_and_grads().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.CNN_EARLY_STOP_PATIENCE:
                break

    if best_params is not None:
        for name, (param, _grad) in model.params_and_grads().items():
            param[...] = best_params[name]

    train_time = time.time() - t0
    return model, history, train_time


def tune_cnn(X_train, y_train, X_val, y_val, grid=None, verbose: int = 0):
    """
    Manual grid search over the CNN's own hyperparameters (epochs,
    batch_size, learning_rate), as requested in Section 7. Selection
    criterion: best validation AUC (early-stopped).
    """
    grid = grid or config.CNN_TUNING_GRID
    results = []
    best = {"val_auc": -np.inf}

    for cfg in grid:
        model, history, train_time = train_cnn(
            X_train, y_train, X_val, y_val,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            learning_rate=cfg["lr"], verbose=verbose,
        )
        val_auc = max(history.history["val_auc"])
        val_acc = max(history.history["val_accuracy"])
        results.append({**cfg, "val_auc": val_auc, "val_accuracy": val_acc,
                         "train_time_s": train_time,
                         "epochs_ran": len(history.history["loss"])})
        print(f"[cnn_tuning] cfg={cfg} -> val_auc={val_auc:.4f}, "
              f"val_acc={val_acc:.4f}, epochs_ran={len(history.history['loss'])}, "
              f"time={train_time:.1f}s")

        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "model": model, "history": history,
                     "config": cfg}

    import pandas as pd
    results_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    return best["model"], best["history"], best["config"], results_df


# ---------------------------------------------------------------------
# Feature extraction (kept as separate functions to mirror the original
# TF-based API used by main.py)
# ---------------------------------------------------------------------
def build_feature_extractor(trained_model: CNN1D) -> CNN1D:
    """No separate sub-model is needed in the numpy implementation --
    the trained model already exposes `.extract_features()` directly."""
    return trained_model


def extract_features(extractor: CNN1D, X: np.ndarray) -> np.ndarray:
    return extractor.extract_features(X)
