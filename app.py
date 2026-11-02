import json
import math
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_pipeline.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"

NUMERIC_FIELDS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FIELDS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

VALIDATION_RULES = {
    "age": (18, 100),
    "trestbps": (80, 250),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 6.2),
    "sex": ("0", "1"),
    "cp": ("1", "2", "3", "4"),
    "fbs": ("0", "1"),
    "restecg": ("0", "1", "2"),
    "exang": ("0", "1"),
    "slope": ("1", "2", "3"),
    "ca": ("0", "1", "2", "3"),
    "thal": ("3", "6", "7"),
}


def create_app():
    app = Flask(__name__, static_folder=str(BASE_DIR / "public"), static_url_path="/static")
    app.secret_key = "heart-disease-academic-prototype"

    try:
        model = joblib.load(MODEL_PATH)
        metadata = None
        if METADATA_PATH.exists():
            import json
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        model = None
        metadata = None
        startup_error = str(exc)
    else:
        startup_error = None

    @app.route("/")
    def index():
        return render_template("index.html", metadata=metadata, startup_error=startup_error)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        errors = {}
        values = {}
        result = None

        if request.method == "POST":
            for field in FEATURE_COLUMNS:
                raw = request.form.get(field, "").strip()
                values[field] = raw
                if raw == "":
                    errors[field] = "Wajib diisi."
                    continue
                if field in VALIDATION_RULES and field in NUMERIC_FIELDS:
                    try:
                        value = float(raw)
                    except ValueError:
                        errors[field] = "Harus berupa angka."
                        continue
                    if not math.isfinite(value):
                        errors[field] = "Nilai tidak valid."
                        continue
                    low, high = VALIDATION_RULES[field]
                    if value < low or value > high:
                        errors[field] = f"Nilai harus antara {low} dan {high}."
                elif field in VALIDATION_RULES:
                    if raw not in VALIDATION_RULES[field]:
                        errors[field] = "Pilihan tidak valid. Pilih salah satu opsi yang tersedia."

            if not errors and model is not None:
                try:
                    sample = {field: float(values[field]) if field in NUMERIC_FIELDS else int(values[field]) for field in FEATURE_COLUMNS}
                    import pandas as pd
                    frame = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
                    prediction = int(model.predict(frame)[0])
                    probabilities = model.predict_proba(frame)[0]
                    result = {
                        "label": "Ada indikasi menurut model" if prediction == 1 else "Tidak ada indikasi menurut model",
                        "class_0": float(probabilities[0]) * 100,
                        "class_1": float(probabilities[1]) * 100,
                    }
                except Exception as exc:
                    errors["__form__"] = f"Prediksi gagal: {exc}"

        return render_template(
            "predict.html",
            errors=errors,
            values=values,
            result=result,
            startup_error=startup_error,
            metadata=metadata,
        )

    @app.route("/about")
    def about():
        return render_template("about.html", metadata=metadata, startup_error=startup_error)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": model is not None})

    @app.route("/evaluasi")
    def evaluation():
        return render_template("evaluation.html", metadata=metadata, startup_error=startup_error)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
