"""
predict.py
----------
Load the saved best model and preprocessing objects,
then run a prediction on a sample student input.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib

from preprocess import (
    load_preprocessing_objects,
    prepare_features,
    build_sample_input,
    MODELS_DIR,
)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH       = os.path.join(MODELS_DIR, "best_model_meta.pkl")


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_best_model():
    """Load the best model (sklearn or Keras) from disk."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{BEST_MODEL_PATH}'.\n"
            "Please run `python src/train.py` first."
        )
    payload = joblib.load(BEST_MODEL_PATH)

    return "sklearn", payload["model"]


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(student_data=None) -> float:
    """
    Run a prediction for a given student input dict or DataFrame.
    Falls back to the built-in sample student if no data is provided.

    Returns the predicted G3 score (0–20).
    """
    # Load artifacts
    encoders, scaler, feature_columns = load_preprocessing_objects()
    model_type, model = load_best_model()

    # Prepare input
    if student_data is None:
        df_input = build_sample_input()
        print("[predict] Using built-in sample student input")
    else:
        import pandas as pd
        df_input = pd.DataFrame([student_data]) if isinstance(student_data, dict) else student_data

    X, _, _ = prepare_features(
        df_input,
        encoders,
        scaler=scaler,
        fit_scaler=False,
        feature_columns=feature_columns,
    )

    raw = model.predict(X)[0]

    # Clamp to valid grade range 0–20
    score = float(np.clip(round(raw, 2), 0, 20))
    return score


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  EduAnalytics — Student Score Predictor")
    print("=" * 50)

    # Print model metadata if available
    if os.path.exists(META_PATH):
        meta = joblib.load(META_PATH)
        print(f"  Active model : {meta['name']}")
        print(f"  Train RMSE   : {meta['rmse']}")
        print(f"  Train R²     : {meta['r2']}")
    print("=" * 50)

    score = predict()

    print(f"\n  Sample Student → Predicted G3 Score: {score} / 20")
    print(f"  Grade band: ", end="")

    if score >= 16:
        print("Excellent (A)")
    elif score >= 14:
        print("Good (B)")
    elif score >= 12:
        print("Average (C)")
    elif score >= 10:
        print("Passing (D)")
    else:
        print("At Risk (F)")

    print("\n[predict] Done.")
