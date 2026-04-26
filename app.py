from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.joblib")
OUTPUT_DIR = Path("generated_results")

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = None
MODEL_LOAD_ERROR = None
EXPECTED_COLUMNS: list[str] = []


def load_model():
    global model, MODEL_LOAD_ERROR, EXPECTED_COLUMNS
    try:
        model = joblib.load(MODEL_PATH)
        EXPECTED_COLUMNS = get_expected_columns()
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        model = None
        EXPECTED_COLUMNS = []
        MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def get_expected_columns() -> list[str]:
    if model is None:
        raise RuntimeError("Model is not loaded.")

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)

    raise RuntimeError("Loaded model does not expose fitted input feature names.")


load_model()


def ensure_model_loaded():
    if model is None:
        raise RuntimeError(f"Model failed to load: {MODEL_LOAD_ERROR}")


def normalize_payload(payload: Any) -> pd.DataFrame:
    ensure_model_loaded()

    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list) or not payload:
        raise ValueError("Payload must be a non-empty JSON object or a list of JSON objects.")

    frame = pd.DataFrame(payload)
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    return frame[EXPECTED_COLUMNS]


def coerce_form_values(form_data: dict[str, str]) -> dict[str, float]:
    cleaned: dict[str, float] = {}

    for field in FIELD_METADATA:
        raw_value = form_data.get(field["name"], "").strip()
        if raw_value == "":
            raise ValueError(f"'{field['label']}' is required.")
        cleaned[field["name"]] = float(raw_value)

    return cleaned


def build_result(prediction: int, probability: float | None) -> dict[str, Any]:
    is_fake = prediction == 1
    confidence = probability if probability is not None else float(prediction)

    return {
        "tone": "risky" if is_fake else "safe",
        "badge": "High-risk profile" if is_fake else "Low-risk profile",
        "title": "Likely Fake Account" if is_fake else "Likely Genuine Account",
        "message": (
            "The model found patterns often linked to suspicious or automated accounts."
            if is_fake
            else "The profile signals are closer to the pattern seen in genuine accounts."
        ),
        "prediction": "Fake" if is_fake else "Not Fake",
        "confidence": f"{confidence * 100:.1f}%",
    }


def run_prediction(payload: dict[str, float]) -> dict[str, Any]:
    ensure_model_loaded()

    frame = normalize_payload(payload)
    prediction = int(model.predict(frame)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0, 1])

    return {
        "prediction_value": prediction,
        "probability": probability,
        "result": build_result(prediction, probability),
    }


def blank_form_values() -> dict[str, str]:
    return {field["name"]: "" for field in FIELD_METADATA}


def load_uploaded_dataframe(file_storage) -> pd.DataFrame:
    filename = (file_storage.filename or "").lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file_storage)
    if filename.endswith(".xlsx"):
        return pd.read_excel(file_storage)
    if filename.endswith(".json"):
        return pd.read_json(file_storage)

    raise ValueError("Unsupported file type. Please upload a CSV, XLSX, or JSON file.")


@app.get("/")
def home() -> Any:
    return render_template("home.html")


@app.get("/manual")
def manual_page() -> Any:
    return render_template(
        "manual.html",
        fields=FIELD_METADATA,
        values=blank_form_values(),
        result=None,
        error=None,
    )


@app.post("/manual")
def predict_form() -> Any:
    values = {field["name"]: request.form.get(field["name"], "") for field in FIELD_METADATA}

    try:
        payload = coerce_form_values(values)
        prediction_data = run_prediction(payload)

        return render_template(
            "manual.html",
            fields=FIELD_METADATA,
            values=values,
            result=prediction_data["result"],
            error=None,
        )
    except Exception as exc:
        return render_template(
            "manual.html",
            fields=FIELD_METADATA,
            values=values,
            result=None,
            error=str(exc),
        ), 400


@app.get("/upload")
def upload_page() -> Any:
    return render_template("upload.html")


@app.get("/health")
def health() -> Any:
    if model is None:
        return jsonify(
            {
                "status": "error",
                "model_path": MODEL_PATH,
                "model_loaded": False,
                "error": MODEL_LOAD_ERROR,
            }
        ), 500

    return jsonify(
        {
            "status": "ok",
            "model_path": MODEL_PATH,
            "model_loaded": True,
        }
    )


@app.get("/schema")
def schema() -> Any:
    ensure_model_loaded()
    return jsonify({"required_features": EXPECTED_COLUMNS})


@app.get("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.post("/predict")
def predict() -> Any:
    try:
        ensure_model_loaded()
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


@app.post("/predict-file")
def predict_file() -> Any:
    try:
        ensure_model_loaded()

        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file uploaded."}), 400

        df = load_uploaded_dataframe(file)

        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required feature columns: {missing_columns}"}), 400

        model_input = df[EXPECTED_COLUMNS].copy()
        predictions = model.predict(model_input)
        probabilities = model.predict_proba(model_input)[:, 1] if hasattr(model, "predict_proba") else None

        result_df = df.copy()
        result_df["prediction"] = predictions
        result_df["prediction_label"] = ["Fake" if pred == 1 else "Not Fake" for pred in predictions]

        if probabilities is not None:
            result_df["probability_fake"] = [round(float(p), 4) for p in probabilities]

        fake_count = int((result_df["prediction"] == 1).sum())
        real_count = int((result_df["prediction"] == 0).sum())

        output_name = f"bulk_predictions_{uuid.uuid4().hex[:10]}.csv"
        output_path = OUTPUT_DIR / output_name
        result_df.to_csv(output_path, index=False)

        return jsonify(
            {
                "total_rows": int(len(result_df)),
                "fake_count": fake_count,
                "real_count": real_count,
                "download_url": f"/download/{output_name}",
                "preview": result_df.head(10).to_dict(orient="records"),
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
