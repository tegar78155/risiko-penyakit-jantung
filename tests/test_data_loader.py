from pathlib import Path

from src.data_loader import load_dataset, get_features_and_target
from src.preprocessing import build_pipeline


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed.cleveland.data"


def test_load_dataset_shape_and_target():
    df = load_dataset(DATA_PATH)
    assert df.shape == (303, 15)
    assert df["target"].value_counts().to_dict() == {0: 164, 1: 139}


def test_features_and_target_no_leakage():
    df = load_dataset(DATA_PATH)
    X, y = get_features_and_target(df)
    assert list(X.columns) == [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    assert "num" not in X.columns
    assert "target" not in X.columns
    assert len(X) == len(y) == 303


def test_pipeline_build():
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
