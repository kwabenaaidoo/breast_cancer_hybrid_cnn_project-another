"""
data_preprocessing.py
----------------------
Loading, cleaning, encoding, and splitting the Wisconsin Breast Cancer
Dataset (WBCD / WDBC). All scaling is fit on the TRAINING split only and
then applied to validation/test to avoid data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


def load_raw_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load the WBCD CSV exactly as exported from the UCI/Kaggle source."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier / empty columns, encode the target, and report any
    missing values (the canonical WDBC file has none, but we check anyway
    so the pipeline is safe on other exports of the same dataset).
    """
    df = df.copy()

    # Drop the sample id (not predictive) and any fully-empty trailing
    # column some CSV exports include (typically "Unnamed: 32").
    drop_cols = [c for c in df.columns if c.lower() == "id"]
    drop_cols += [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Report + handle missing values (median-impute numeric columns)
    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        print(f"[data_preprocessing] Found {n_missing} missing values -> "
              f"median-imputing numeric columns.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        print("[data_preprocessing] No missing values found.")

    # Encode diagnosis: Malignant = 1, Benign = 0
    # (robust to pandas' "object" vs newer "string"/pyarrow-backed dtypes)
    if not pd.api.types.is_numeric_dtype(df["diagnosis"]):
        df["diagnosis"] = df["diagnosis"].astype(str).str.strip().map(
            {config.POSITIVE_CLASS: 1, config.NEGATIVE_CLASS: 0}
        )
    df["diagnosis"] = df["diagnosis"].astype(int)

    # Drop exact duplicate rows if any slipped in
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"[data_preprocessing] Dropping {n_dupes} duplicate rows.")
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def get_features_and_target(df: pd.DataFrame):
    y = df["diagnosis"].values
    X = df.drop(columns=["diagnosis"]).values
    feature_names = df.drop(columns=["diagnosis"]).columns.tolist()
    return X, y, feature_names


def split_data(X: np.ndarray, y: np.ndarray, seed: int = config.RANDOM_SEED):
    """
    Stratified 70 / 15 / 15 train / validation / test split.
    The test set is held out completely until final evaluation.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VAL_SIZE_OF_REMAINDER,
        stratify=y_train_full,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Fit StandardScaler on TRAIN ONLY, then transform val/test.
    This scaler prepares raw tabular features for the CNN input.
    (A second, independent scaler is fit later, inside each CV fold,
    on the CNN-*extracted* features -- see hybrid_models.py.)
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def prepare_dataset(path: str = config.DATA_PATH, seed: int = config.RANDOM_SEED):
    """End-to-end convenience wrapper used by main.py."""
    df = load_raw_data(path)
    df = clean_data(df)
    X, y, feature_names = get_features_and_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    print(f"[data_preprocessing] Class balance -> "
          f"Malignant: {int(y.sum())} ({y.mean()*100:.1f}%), "
          f"Benign: {int((1 - y).sum())} ({(1 - y.mean())*100:.1f}%)")
    print(f"[data_preprocessing] Split sizes -> "
          f"train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    return {
        "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names, "scaler": scaler,
    }
