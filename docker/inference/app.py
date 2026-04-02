"""
app.py — Vertex AI custom prediction container for the UCI Diabetes XGBoost model.

Endpoints (Vertex AI AIP-1.0 custom container interface)
---------
GET  /ping    → {"status": "ok"}
POST /predict → {"predictions": [{"probability": float, "flagged": bool}, ...]}

Input (AIP-1.0 format)
-----------------------
{
  "instances": [
    {
      "race": "Caucasian",
      "gender": "Male",
      "age_band": "[30-40)",
      "time_in_hospital": 3,
      ...  (raw encounter features — same schema as the Snowflake mart)
    }
  ]
}

Identifier, leakage, and near-zero-variance columns are silently dropped if
present, matching the behaviour of ml/sklearn/preprocess.py.

Environment variables
---------------------
MODEL_DIR  : directory containing xgb_model.joblib + preprocessor.joblib
             Default: /app/model
TAU        : decision threshold for "flagged" field.  Default: 0.12
PORT       : server port, set by the Dockerfile.     Default: 8080
"""

import logging
import os
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", "/app/model"))
TAU = float(os.environ.get("TAU", "0.12"))

# Columns to drop before preprocessing — mirrors preprocess.py
_DROP_COLS: set[str] = {
    # Identifier / leakage
    "encounter_id", "patient_nbr", "readmitted_raw", "readmitted_any",
    "split_group", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "readmitted_30day",
    # Near-zero variance
    "glimepiride_pioglitazone", "metformin_pioglitazone",
    "metformin_rosiglitazone", "glipizide_metformin", "tolbutamide",
    "miglitol", "tolazamide", "chlorpropamide", "acarbose",
    "glyburide_metformin", "nateglinide",
}


def _sanitize(name: str) -> str:
    """XGBoost forbids '[', ']', '<' in feature names."""
    return name.replace("[", "(").replace("]", ")").replace("<", "lt")


# ---------------------------------------------------------------------------
# Artifact loading (once at startup)
# ---------------------------------------------------------------------------
_model = None
_preprocessor = None


def _load_artifacts() -> None:
    global _model, _preprocessor
    model_path = MODEL_DIR / "xgb_model.joblib"
    prep_path  = MODEL_DIR / "preprocessor.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {prep_path}")
    _model        = joblib.load(model_path)
    _preprocessor = joblib.load(prep_path)
    logger.info("Artifacts loaded from %s  |  TAU=%.2f", MODEL_DIR, TAU)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="UCI Diabetes Readmission Predictor", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    _load_artifacts()


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]]


@app.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame(request.instances)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid instances: {exc}")

    # Drop ID / leakage / NZV columns if present
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # Preprocess
    try:
        X_arr = _preprocessor.transform(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}")

    # Build DataFrame with sanitized feature names (required by XGBoost)
    feature_names = [_sanitize(c) for c in _preprocessor.get_feature_names_out()]
    X = pd.DataFrame(X_arr, columns=feature_names)

    probs = _model.predict_proba(X)[:, 1]

    return {
        "predictions": [
            {"probability": round(float(p), 6), "flagged": bool(p >= TAU)}
            for p in probs
        ]
    }
