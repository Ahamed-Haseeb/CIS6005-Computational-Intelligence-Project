from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bank_marketing_random_forest.pkl"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=logging.INFO)


CATEGORIES = {
    "job": [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ],
    "poutcome": ["failure", "other", "success", "unknown"],
}

NUMERIC_RULES = {
    "age": (18, 100, True),
    "balance": (-1_000_000, 1_000_000, False),
    "day": (1, 31, True),
    "duration": (0, 20_000, True),
    "campaign": (1, 1_000, True),
    "pdays": (-1, 10_000, True),
    "previous": (0, 10_000, True),
}

DEFAULT_INPUT = {
    "age": 39,
    "job": "management",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 634,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 17,
    "month": "may",
    "duration": 133,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


class PredictionService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._bundle: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def _load(self) -> dict[str, Any]:
        if self._bundle is None:
            with self._lock:
                if self._bundle is None:
                    if not self.model_path.exists():
                        raise FileNotFoundError(
                            f"Model file not found at {self.model_path}"
                        )
                    app.logger.info("Loading model bundle from %s", self.model_path)
                    self._bundle = joblib.load(self.model_path, mmap_mode="r")
                    self._verify_bundle(self._bundle)
                    app.logger.info("Model bundle loaded")
        return self._bundle

    @staticmethod
    def _verify_bundle(bundle: dict[str, Any]) -> None:
        required = {
            "model",
            "scaler",
            "label_encoder",
            "feature_columns",
            "raw_feature_columns",
            "categorical_levels",
        }
        missing = required.difference(bundle)
        if missing:
            raise ValueError(f"Model bundle is missing: {', '.join(sorted(missing))}")

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        bundle = self._load()
        data = pd.DataFrame([values])
        data["balance_per_campaign"] = data["balance"] / (data["campaign"] + 1)

        raw_columns = bundle["raw_feature_columns"]
        missing = [column for column in raw_columns if column not in data.columns]
        if missing:
            raise ValueError(f"Model inputs are missing: {', '.join(missing)}")
        data = data[raw_columns]

        for column, levels in bundle["categorical_levels"].items():
            data[column] = pd.Categorical(
                data[column].astype(str), categories=levels
            )

        encoded = pd.get_dummies(data, drop_first=True).reindex(
            columns=bundle["feature_columns"], fill_value=0
        )
        scaled = bundle["scaler"].transform(encoded)
        model = bundle["model"]
        encoded_prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]

        class_positions = {str(value): index for index, value in enumerate(model.classes_)}
        positive_index = class_positions.get("1", len(probabilities) - 1)
        probability = float(probabilities[positive_index])
        decoded = bundle["label_encoder"].inverse_transform(
            np.asarray([encoded_prediction], dtype=int)
        )[0]
        prediction = int(decoded) if isinstance(decoded, (int, np.integer)) else str(decoded)

        if probability >= 0.65:
            segment = "High potential"
        elif probability >= 0.35:
            segment = "Worth nurturing"
        else:
            segment = "Low likelihood"

        return {
            "prediction": prediction,
            "probability": round(probability, 6),
            "confidence": round(max(probability, 1 - probability), 6),
            "segment": segment,
            "model": bundle.get("model_name", type(model).__name__),
        }


prediction_service = PredictionService(MODEL_PATH)


def validate_payload(payload: Any) -> tuple[dict[str, Any] | None, dict[str, str]]:
    if not isinstance(payload, dict):
        return None, {"request": "Send a JSON object containing customer details."}

    clean: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for field, (minimum, maximum, integer_only) in NUMERIC_RULES.items():
        raw_value = payload.get(field)
        try:
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError
            if integer_only and not value.is_integer():
                errors[field] = "Enter a whole number."
                continue
            if not minimum <= value <= maximum:
                errors[field] = f"Enter a value from {minimum:,} to {maximum:,}."
                continue
            clean[field] = int(value) if integer_only else value
        except (TypeError, ValueError):
            errors[field] = "Enter a valid number."

    for field, choices in CATEGORIES.items():
        value = str(payload.get(field, "")).strip().lower()
        if value not in choices:
            errors[field] = "Choose a valid option."
        else:
            clean[field] = value

    return (clean if not errors else None), errors


@app.get("/")
def index():
    model_size_gb = MODEL_PATH.stat().st_size / (1024**3) if MODEL_PATH.exists() else 0
    return render_template(
        "index.html",
        categories=CATEGORIES,
        defaults=DEFAULT_INPUT,
        model_exists=MODEL_PATH.exists(),
        model_size_gb=f"{model_size_gb:.2f}",
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ready" if MODEL_PATH.exists() else "model_missing",
            "model_exists": MODEL_PATH.exists(),
            "model_loaded": prediction_service.loaded,
            "model_file": MODEL_PATH.name,
        }
    )


@app.post("/api/predict")
def predict():
    values, errors = validate_payload(request.get_json(silent=True))
    if errors:
        return jsonify({"error": "Please correct the highlighted fields.", "fields": errors}), 400

    try:
        result = prediction_service.predict(values or {})
    except FileNotFoundError as exc:
        app.logger.error("Prediction model is unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(
            {"error": "The model could not complete this prediction. Check the server log."}
        ), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
