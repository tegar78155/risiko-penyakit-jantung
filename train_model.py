import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split

from src.preprocessing import build_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.evaluation import evaluate_model
from src.data_loader import FEATURE_COLUMNS, get_features_and_target, load_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed.cleveland.data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def save_confusion_matrix(matrix, path: Path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc_curve(fpr, tpr, auc, path: Path):
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_target_distribution(y, path: Path):
    plt.figure(figsize=(5, 4))
    sns.countplot(x=y)
    plt.xlabel("Target")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    df = load_dataset(DATA_PATH)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_pipeline = build_pipeline()
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipeline, X_test, y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {"model__C": [0.01, 0.1, 1.0, 10.0], "model__solver": ["liblinear", "lbfgs"]}
    grid = GridSearchCV(
        build_pipeline(),
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    tuned_pipeline = grid.best_estimator_
    tuned_metrics = evaluate_model(tuned_pipeline, X_test, y_test)

    cv_scores = cross_val_score(tuned_pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    oof_proba = cross_val_predict(tuned_pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold_analysis = {}
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        preds = (oof_proba >= threshold).astype(int)
        threshold_analysis[str(threshold)] = {
            "accuracy": float(accuracy_score(y_train, preds)),
            "precision": float(precision_score(y_train, preds, zero_division=0)),
            "recall": float(recall_score(y_train, preds, zero_division=0)),
            "f1_score": float(f1_score(y_train, preds, zero_division=0)),
        }

    metrics = tuned_metrics

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(tuned_pipeline, ARTIFACTS_DIR / "heart_disease_pipeline.joblib")

    metadata = {
        "dataset": "processed.cleveland.data",
        "n_samples": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {"0": "Tidak ada indikasi", "1": "Ada indikasi"},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_size_ratio": 0.2,
        "random_state": 42,
        "baseline_metrics": {
            "confusion_matrix": baseline_metrics["confusion_matrix"],
            "accuracy": baseline_metrics["accuracy"],
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "f1_score": baseline_metrics["f1_score"],
            "roc_auc": baseline_metrics["roc_auc"],
        },
        "tuned_hyperparameters": grid.best_params_,
        "metrics": {
            "confusion_matrix": metrics["confusion_matrix"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "classification_report": metrics["classification_report"],
        },
        "threshold_analysis_out_of_fold": threshold_analysis,
        "cross_validation": {
            "folds": 5,
            "scoring": "accuracy",
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
            "scores": [float(x) for x in cv_scores],
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    save_confusion_matrix(metrics["confusion_matrix"], REPORTS_DIR / "confusion_matrix.png")
    save_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], REPORTS_DIR / "roc_curve.png")
    save_target_distribution(y, REPORTS_DIR / "target_distribution.png")

    print("Training complete")
    print(json.dumps(metadata["metrics"], indent=2, ensure_ascii=False))
    print("Cross-validation accuracy:", cv_scores.mean(), "+/-", cv_scores.std())


if __name__ == "__main__":
    main()
