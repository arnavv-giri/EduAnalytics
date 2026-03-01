"""
api/main.py
-----------
EduAnalytics FastAPI prediction service — fully corrected.

ROOT CAUSE OF STATIC PREDICTIONS (now fixed):
  The model was trained on 32 features. G1 and G2 (first/second period grades)
  are by far the strongest predictors (r ≈ 0.9 with G3). The old schema had
  G1=12 and G2=13 as hardcoded Pydantic Field() defaults, so EVERY request
  sent identical G1/G2 regardless of user input → prediction always ≈ 12.9.

FIX:
  The schema now accepts all 32 training features with realistic dataset
  medians/modes as defaults. The frontend sends all 32 fields: the six most
  impactful ones (age, G1, G2, studytime, failures, absences) are shown as
  user inputs; the remaining 26 travel as hidden fields with sensible defaults.
  Changing G1 from 10 → 18 now shifts the prediction dramatically, as expected.

Endpoints:
  GET  /            → health check
  GET  /model-info  → active model metadata
  POST /predict     → predict G3 from all 32 student features
"""

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Allow imports from src/ ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocess import load_preprocessing_objects, prepare_features, MODELS_DIR

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eduanalytics")

# ── Paths ──────────────────────────────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH       = os.path.join(MODELS_DIR, "best_model_meta.pkl")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduAnalytics API",
    description="Predict student final exam scores (G3, 0–20) using ML.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  Request schema — ALL 32 training features
#
#  Defaults are dataset medians (numeric) and modes (categorical) from the
#  UCI student-mat.csv (n = 395 Portuguese students).  They represent a
#  realistic "average student" prior for fields the frontend doesn't expose.
#
#  THE SIX FIELDS BELOW ARE THE ONES THE FRONTEND ACTUALLY COLLECTS:
#    age, G1, G2, studytime, failures, absences
#  All other fields default silently but can be overridden.
# ══════════════════════════════════════════════════════════════════════════════
class StudentInput(BaseModel):
    # ── User-facing inputs (frontend exposes these) ────────────────────────
    age:       int   = Field(17,  ge=10, le=22,  description="Student age")
    G1:        int   = Field(10,  ge=0,  le=20,  description="First period grade (0–20)")
    G2:        int   = Field(10,  ge=0,  le=20,  description="Second period grade (0–20)")
    studytime: int   = Field(2,   ge=1,  le=4,   description="Weekly study hours (1=<2h … 4=>10h)")
    failures:  int   = Field(0,   ge=0,  le=4,   description="Number of past class failures")
    absences:  int   = Field(4,   ge=0,  le=93,  description="Number of school absences")

    # ── Background fields (sent by frontend as hidden defaults) ───────────
    school:     str  = Field("GP",      description="School (GP | MS)")
    sex:        str  = Field("M",       description="Sex (M | F)")
    address:    str  = Field("U",       description="Address type (U | R)")
    famsize:    str  = Field("GT3",     description="Family size (GT3 | LE3)")
    Pstatus:    str  = Field("T",       description="Parents' cohabitation (T | A)")
    Medu:       int  = Field(2,  ge=0,  le=4,   description="Mother's education (0–4)")
    Fedu:       int  = Field(2,  ge=0,  le=4,   description="Father's education (0–4)")
    Mjob:       str  = Field("other",   description="Mother's job")
    Fjob:       str  = Field("other",   description="Father's job")
    reason:     str  = Field("course",  description="Reason to choose school")
    guardian:   str  = Field("mother",  description="Guardian")
    traveltime: int  = Field(1,  ge=1,  le=4,   description="Travel time to school (1–4)")
    schoolsup:  str  = Field("no",      description="Extra school support (yes | no)")
    famsup:     str  = Field("yes",     description="Family support (yes | no)")
    paid:       str  = Field("no",      description="Paid extra classes (yes | no)")
    activities: str  = Field("no",      description="Extracurricular activities (yes | no)")
    nursery:    str  = Field("yes",     description="Attended nursery (yes | no)")
    higher:     str  = Field("yes",     description="Wants higher education (yes | no)")
    internet:   str  = Field("yes",     description="Internet at home (yes | no)")
    romantic:   str  = Field("no",      description="In a relationship (yes | no)")
    famrel:     int  = Field(4,  ge=1,  le=5,   description="Family relationship quality (1–5)")
    freetime:   int  = Field(3,  ge=1,  le=5,   description="Free time after school (1–5)")
    goout:      int  = Field(3,  ge=1,  le=5,   description="Going out with friends (1–5)")
    Dalc:       int  = Field(1,  ge=1,  le=5,   description="Workday alcohol consumption (1–5)")
    Walc:       int  = Field(1,  ge=1,  le=5,   description="Weekend alcohol consumption (1–5)")
    health:     int  = Field(3,  ge=1,  le=5,   description="Current health status (1–5)")


class PredictionResponse(BaseModel):
    predicted_score:   float
    performance_level: str
    grade_band:        str   # alias kept for frontend compatibility
    model_used:        str
    status:            str = "success"


class HealthResponse(BaseModel):
    status:  str
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    rmse:       float
    r2:         float


# ── Helpers ────────────────────────────────────────────────────────────────────
def score_to_level(score: float) -> str:
    """Map a 0–20 score to a human-readable performance level."""
    if score >= 16: return "Excellent"
    if score >= 13: return "Good"
    if score >= 10: return "Average"
    if score >= 7:  return "Poor"
    return "Very Poor"


def score_to_band(score: float) -> str:
    """Map a 0–20 score to a letter-grade band (kept for frontend compat)."""
    if score >= 16: return "Excellent (A)"
    if score >= 14: return "Good (B)"
    if score >= 12: return "Average (C)"
    if score >= 10: return "Passing (D)"
    return "At Risk (F)"


def load_model():
    """Load the best saved sklearn model from disk."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. "
            "Run `python src/train.py` first."
        )
    payload = joblib.load(BEST_MODEL_PATH)
    return payload["model"]


def get_model_name() -> str:
    if os.path.exists(META_PATH):
        return joblib.load(META_PATH).get("name", "Unknown")
    return "Unknown"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    return HealthResponse(
        status="running",
        service="EduAnalytics Prediction API",
        version="2.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if not os.path.exists(META_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python src/train.py` first.",
        )
    meta = joblib.load(META_PATH)
    return ModelInfoResponse(
        model_name=meta["name"],
        rmse=float(meta["rmse"]),
        r2=float(meta["r2"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_score(student: StudentInput):
    """
    Predict a student's final G3 exam score (0–20).

    All 32 training features must be present in the request body.
    The frontend sends the 6 user-entered fields plus 26 pre-filled defaults.
    """
    try:
        # ── 1. Log raw input ──────────────────────────────────────────────
        student_dict = student.model_dump()
        print("\n" + "=" * 55)
        print("[predict] Received input:")
        for k, v in student_dict.items():
            print(f"  {k:12s} = {v}")
        print("=" * 55)
        log.info("Received prediction request: age=%s G1=%s G2=%s studytime=%s failures=%s absences=%s",
                 student.age, student.G1, student.G2,
                 student.studytime, student.failures, student.absences)

        # ── 2. Load preprocessing objects ────────────────────────────────
        encoders, scaler, feature_columns = load_preprocessing_objects()

        # ── 3. Build DataFrame with columns in the exact training order ───
        #       feature_columns was saved during training and is the ground truth.
        df_input = pd.DataFrame([student_dict])

        # Reorder columns to match training order exactly
        df_input = df_input[feature_columns]

        print(f"[predict] DataFrame shape:   {df_input.shape}")
        print(f"[predict] DataFrame columns: {list(df_input.columns)}")
        print(f"[predict] Key values → age={df_input['age'].iloc[0]}, "
              f"G1={df_input['G1'].iloc[0]}, G2={df_input['G2'].iloc[0]}, "
              f"studytime={df_input['studytime'].iloc[0]}, "
              f"failures={df_input['failures'].iloc[0]}, "
              f"absences={df_input['absences'].iloc[0]}")

        # ── 4. Apply the same preprocessing pipeline used during training ─
        X, _, _ = prepare_features(
            df_input,
            encoders,
            scaler=scaler,
            fit_scaler=False,
            feature_columns=feature_columns,
        )

        # ── 5. Load model and predict ─────────────────────────────────────
        model      = load_model()
        raw        = model.predict(X)[0]
        prediction = float(np.clip(round(raw, 2), 0, 20))

        print(f"[predict] Raw prediction:    {raw:.4f}")
        print(f"[predict] Clamped score:     {prediction}")
        log.info("Prediction: %.2f → %s", prediction, score_to_level(prediction))

        # ── 6. Return response ────────────────────────────────────────────
        return PredictionResponse(
            predicted_score=prediction,
            performance_level=score_to_level(prediction),
            grade_band=score_to_band(prediction),
            model_used=get_model_name(),
            status="success",
        )

    except FileNotFoundError as exc:
        log.error("Artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except KeyError as exc:
        log.error("Feature mismatch: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Feature mismatch between request and training data: {exc}",
        )
    except Exception as exc:
        log.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
