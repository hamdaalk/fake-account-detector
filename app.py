from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.joblib")

FIELD_METADATA = [
    {
        "name": "profile pic",
        "label": "Has Profile Picture",
        "hint": "Choose yes if the account has a visible profile image.",
        "kind": "boolean",
    },
    {
        "name": "nums/length username",
        "label": "Username Number Ratio",
        "hint": "Share of numeric characters inside the username, usually between 0 and 1.",
        "kind": "number",
        "step": "0.01",
        "min": "0",
        "placeholder": "e.g. 0.20",
    },
    {
        "name": "fullname words",
        "label": "Full Name Word Count",
        "hint": "How many words appear in the full display name.",
        "kind": "number",
        "step": "1",
        "min": "0",
        "placeholder": "e.g. 2",
    },
    {
        "name": "nums/length fullname",
        "label": "Full Name Number Ratio",
        "hint": "Share of numeric characters in the full name, usually between 0 and 1.",
        "kind": "number",
        "step": "0.01",
        "min": "0",
        "placeholder": "e.g. 0.00",
    },
    {
        "name": "name==username",
        "label": "Name Matches Username",
        "hint": "Choose yes if the visible name is the same as the username.",
        "kind": "boolean",
    },
    {
        "name": "description length",
        "label": "Bio Length",
        "hint": "Total number of characters in the account description or bio.",
        "kind": "number",
        "step": "1",
        "min": "0",
        "placeholder": "e.g. 40",
    },
    {
        "name": "external URL",
        "label": "Has External URL",
        "hint": "Choose yes if the profile includes a website or external link.",
        "kind": "boolean",
    },
    {
        "name": "private",
        "label": "Private Account",
        "hint": "Choose yes if the account is private.",
        "kind": "boolean",
    },
    {
        "name": "#posts",
        "label": "Posts Count",
        "hint": "Number of posts visible on the account.",
        "kind": "number",
        "step": "1",
        "min": "0",
        "placeholder": "e.g. 120",
    },
    {
        "name": "#followers",
        "label": "Followers Count",
        "hint": "Number of followers on the account.",
        "kind": "number",
        "step": "1",
        "min": "0",
        "placeholder": "e.g. 850",
    },
    {
        "name": "#follows",
        "label": "Following Count",
        "hint": "Number of other accounts this profile follows.",
        "kind": "number",
        "step": "1",
        "min": "0",
        "placeholder": "e.g. 300",
    },
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
BORDERLINE_MIN = 0.40
BORDERLINE_MAX = 0.60


def get_expected_columns() -> list[str]:
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        raise RuntimeError("Loaded model does not expose fitted input feature names.")
    return list(preprocessor.feature_names_in_)


EXPECTED_COLUMNS = get_expected_columns()


def normalize_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("Payload must be a non-empty JSON object or a list of JSON objects.")

    frame = pd.DataFrame(payload)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    # Ignore extra columns but preserve the model's training column order.
    return frame[EXPECTED_COLUMNS]


def coerce_form_values(form_data: dict[str, str]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for field in FIELD_METADATA:
        raw_value = form_data.get(field["name"], "").strip()
        if raw_value == "":
            raise ValueError(f"'{field['label']}' is required.")
        cleaned[field["name"]] = float(raw_value)
    return cleaned


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def classify_prediction(prediction: int, probability: float | None) -> dict[str, Any]:
    fake_probability = float(probability) if probability is not None else None
    is_fake = prediction == 1

    if fake_probability is not None and BORDERLINE_MIN <= fake_probability <= BORDERLINE_MAX:
        confidence = max(fake_probability, 1 - fake_probability)
        return {
            "status": "borderline",
            "label": "Borderline",
            "confidence": confidence,
            "probability_display": format_percentage(fake_probability),
        }

    confidence = 1.0
    if fake_probability is not None:
        confidence = fake_probability if is_fake else 1 - fake_probability

    return {
        "status": "fake" if is_fake else "not-fake",
        "label": "Fake" if is_fake else "Not Fake",
        "confidence": confidence,
        "probability_display": format_percentage(confidence),
    }


def run_prediction(payload: dict[str, float]) -> dict[str, Any]:
    frame = normalize_payload(payload)
    prediction = int(model.predict(frame)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0, 1])

    classification = classify_prediction(prediction, probability)
    status = classification["status"]

    if status == "fake":
        tone = "risky"
        badge = "High-risk profile"
        title = "Likely Fake Account"
        message = "The model detected patterns commonly associated with suspicious or automated accounts."
        probability_label = "Probability"
    elif status == "borderline":
        tone = "borderline"
        badge = "Needs manual review"
        title = "Borderline Account"
        message = "This profile sits close to the decision boundary, so it should be reviewed manually."
        probability_label = "Fake Probability"
    else:
        tone = "safe"
        badge = "Low-risk profile"
        title = "Likely Genuine Account"
        message = "The profile signals are closer to the pattern seen in genuine accounts."
        probability_label = "Probability"

    return {
        "prediction_value": prediction,
        "probability": probability,
        "result": {
            "tone": tone,
            "badge": badge,
            "title": title,
            "message": message,
            "prediction": classification["label"],
            "probability_label": probability_label,
            "probability": classification["probability_display"],
        },
    }


def blank_form_values() -> dict[str, str]:
    return {field["name"]: "" for field in FIELD_METADATA}


@app.get("/")
def root() -> Any:
    return render_template(
        "index.html",
        fields=FIELD_METADATA,
        values=blank_form_values(),
        result=None,
        error=None,
    )


@app.post("/")
def predict_form() -> Any:
    values = {field["name"]: request.form.get(field["name"], "") for field in FIELD_METADATA}
    try:
        payload = coerce_form_values(values)
        prediction_data = run_prediction(payload)
        return render_template(
            "index.html",
            fields=FIELD_METADATA,
            values=values,
            result=prediction_data["result"],
            error=None,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            fields=FIELD_METADATA,
            values=values,
            result=None,
            error=str(exc),
        ), 400


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "model_path": MODEL_PATH})


@app.get("/schema")
def schema() -> Any:
    return jsonify({"required_features": EXPECTED_COLUMNS})


@app.post("/predict")
def predict() -> Any:
    try:
        payload = request.get_json(force=True, silent=False)
        if isinstance(payload, dict):
            prediction_data = run_prediction(payload)
            response: dict[str, Any] = {
                "predictions": [prediction_data["prediction_value"]],
                "result": prediction_data["result"],
            }
            if prediction_data["probability"] is not None:
                response["probabilities"] = [prediction_data["probability"]]
            return jsonify(response)

        frame = normalize_payload(payload)
        predictions = model.predict(frame).tolist()
        response = {"predictions": predictions}
        if hasattr(model, "predict_proba"):
            response["probabilities"] = model.predict_proba(frame)[:, 1].tolist()
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
