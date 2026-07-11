"""
OptiCrop - Flask Web Application (Epic 5)
This script runs the Flask web server to serve the OptiCrop crop recommendation system.
It loads the pre-trained machine learning model and provides routes for input and prediction.
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "opticrop_secret_key_for_flash_messages"

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"

# Global model variable
model = None

# Load the trained model bundle on startup
try:
    if not MODEL_PATH.exists():
        print(f"CRITICAL WARNING: Model file not found at {MODEL_PATH.absolute()}")
    else:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        print("Model bundle loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. Details: {e}")


def validate_inputs(form_data: dict) -> tuple[dict, str | None]:
    """
    Validates input parameters from the HTML form.
    Returns a dictionary of parsed float values and an error message if validation fails.
    """
    required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    parsed_values = {}

    # Check for missing fields
    for field in required_fields:
        if field not in form_data or form_data[field].strip() == "":
            return {}, f"Field '{field}' is required and cannot be empty."

    try:
        # Convert and check ranges
        parsed_values["N"] = float(form_data["N"])
        parsed_values["P"] = float(form_data["P"])
        parsed_values["K"] = float(form_data["K"])
        parsed_values["temperature"] = float(form_data["temperature"])
        parsed_values["humidity"] = float(form_data["humidity"])
        parsed_values["ph"] = float(form_data["ph"])
        parsed_values["rainfall"] = float(form_data["rainfall"])
    except ValueError:
        return {}, "All inputs must be valid numerical values."

    # Specific range checks for agricultural plausibility
    if parsed_values["N"] < 0 or parsed_values["N"] > 200:
        return {}, "Nitrogen (N) must be between 0 and 200 mg/kg."
    
    if parsed_values["P"] < 0 or parsed_values["P"] > 200:
        return {}, "Phosphorous (P) must be between 0 and 200 mg/kg."
        
    if parsed_values["K"] < 0 or parsed_values["K"] > 300:
        return {}, "Potassium (K) must be between 0 and 300 mg/kg."

    if parsed_values["temperature"] < -10 or parsed_values["temperature"] > 60:
        return {}, "Temperature must be between -10°C and 60°C."

    if parsed_values["humidity"] < 0 or parsed_values["humidity"] > 100:
        return {}, "Humidity must be between 0% and 100%."

    if parsed_values["ph"] < 0 or parsed_values["ph"] > 14:
        return {}, "pH must be between 0 and 14."

    if parsed_values["rainfall"] < 0 or parsed_values["rainfall"] > 500:
        return {}, "Rainfall must be between 0 and 500 mm."

    return parsed_values, None


@app.route("/", methods=["GET"])
def index():
    """Renders the Home page with the input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives form data, validates it, makes a prediction using the
    loaded crop recommendation model, and displays the result.
    """
    global model

    # 1. Check if model is loaded
    if model is None:
        # Try to reload if it wasn't loaded on startup
        try:
            if MODEL_PATH.exists():
                bundle = joblib.load(MODEL_PATH)
                model = bundle["model"]
            else:
                return render_template(
                    "index.html",
                    error="Prediction system is offline: Model file is missing. Please contact administration.",
                )
        except Exception:
            return render_template(
                "index.html",
                error="Prediction system is offline: Model load failure. Please contact administration.",
            )

    # 2. Validate form inputs
    parsed_data, error_message = validate_inputs(request.form)
    if error_message:
        return render_template("index.html", error=error_message, form_data=request.form)

    # 3. Perform prediction
    try:
        # Build dataframe in the exact order and names expected by the model
        feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        sample_df = pd.DataFrame([parsed_data], columns=feature_cols)

        # Make prediction
        prediction = model.predict(sample_df)[0]
        
        # Get confidence score if available
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(sample_df)[0]
            confidence = float(np.max(probs)) * 100

        # Format prediction name (e.g. capitalize)
        recommended_crop = str(prediction).strip().capitalize()

        return render_template(
            "result.html",
            crop=recommended_crop,
            confidence=confidence,
            inputs=parsed_data,
        )

    except Exception as e:
        # Log the internal error and show a user-friendly error page
        print(f"Prediction Error: {e}", file=sys.stderr)
        return render_template(
            "index.html",
            error="An error occurred during prediction. Please verify your inputs and try again.",
            form_data=request.form,
        )


if __name__ == "__main__":
    # Run the Flask app in debug mode on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
