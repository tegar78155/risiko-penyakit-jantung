from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

TARGET_COLUMN = "target"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.shape != (303, 14):
        raise ValueError(f"Unexpected dataset shape: {df.shape}")
    if df.duplicated().sum() != 0:
        raise ValueError("Dataset contains duplicate rows")
    if set(df["num"].unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("Unexpected target values in num column")
    missing = df.isna().sum()
    if int(missing.sum()) != 6:
        raise ValueError(f"Unexpected missing value count: {int(missing.sum())}")
    if missing.get("ca", 0) != 4 or missing.get("thal", 0) != 2:
        raise ValueError("Missing values are not in expected columns")
    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    if "num" in X.columns or TARGET_COLUMN in X.columns:
        raise ValueError("Target or num leaked into features")
    return X, y
