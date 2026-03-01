"""
preprocess.py
-------------
Reusable preprocessing functions for EduAnalytics.
Handles encoding, feature engineering, and data preparation.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_COLUMN = "G3"

CATEGORICAL_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NUMERIC_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout", "Dalc",
    "Walc", "health", "absences", "G1", "G2",
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load the student dataset from a semicolon-separated CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please place student-mat.csv in the data/ directory."
        )
    df = pd.read_csv(path, sep=";")
    print(f"[preprocess] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─── Encoding ─────────────────────────────────────────────────────────────────

def fit_label_encoders(df: pd.DataFrame) -> dict:
    """
    Fit a LabelEncoder for every categorical column present in the DataFrame.
    Returns a dict mapping column name → fitted LabelEncoder.
    """
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            encoders[col] = le
    print(f"[preprocess] Fitted encoders for {len(encoders)} categorical columns")
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply pre-fitted LabelEncoders to categorical columns.
    Unseen values are mapped to the most frequent class to avoid errors.
    """
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen labels gracefully
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df


# ─── Feature Preparation ──────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    encoders: dict,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
    feature_columns: list | None = None,
) -> tuple[pd.DataFrame, StandardScaler, list]:
    """
    Full pipeline:
      1. Encode categoricals
      2. Select feature columns (all except target)
      3. Optionally fit or apply StandardScaler

    Returns (X_df, scaler, feature_columns)
    """
    df_enc = apply_label_encoders(df, encoders)

    # Determine feature columns on first call (training)
    if feature_columns is None:
        feature_columns = [c for c in df_enc.columns if c != TARGET_COLUMN]

    X = df_enc[feature_columns].copy()

    # Handle any remaining non-numeric columns defensively
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.factorize(X[col])[0]

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    X_df = pd.DataFrame(X_scaled, columns=feature_columns)
    return X_df, scaler, feature_columns


def get_target(df: pd.DataFrame) -> pd.Series:
    """Extract the target variable G3."""
    return df[TARGET_COLUMN].copy()


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_preprocessing_objects(encoders: dict, scaler: StandardScaler, feature_columns: list):
    """Persist encoders, scaler, and feature column list to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_PATH)
    print(f"[preprocess] Saved encoders  → {ENCODER_PATH}")
    print(f"[preprocess] Saved scaler    → {SCALER_PATH}")
    print(f"[preprocess] Saved features  → {FEATURE_PATH}")


def load_preprocessing_objects() -> tuple[dict, StandardScaler, list]:
    """Load encoders, scaler, and feature columns from disk."""
    for path in [ENCODER_PATH, SCALER_PATH, FEATURE_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preprocessing object not found: {path}\n"
                "Run train.py first to generate all required artifacts."
            )
    encoders        = joblib.load(ENCODER_PATH)
    scaler          = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    print("[preprocess] Loaded preprocessing objects from disk")
    return encoders, scaler, feature_columns


# ─── Sample Input Builder ─────────────────────────────────────────────────────

def build_sample_input() -> pd.DataFrame:
    """
    Returns a single-row DataFrame representing a sample student.
    Used by predict.py and the API for demonstration / testing.
    """
    sample = {
        "school": "GP", "sex": "M", "age": 17,
        "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 3, "Fedu": 2,
        "Mjob": "teacher", "Fjob": "other",
        "reason": "course", "guardian": "mother",
        "traveltime": 2, "studytime": 3, "failures": 0,
        "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 2,
        "Dalc": 1, "Walc": 2, "health": 4,
        "absences": 2, "G1": 12, "G2": 13,
    }
    return pd.DataFrame([sample])
