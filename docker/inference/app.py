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
AIP_STORAGE_URI : GCS URI to model artifacts — set automatically by Vertex AI
                  (e.g. gs://bucket/artifacts/v1).  When set, artifacts are
                  downloaded from GCS at startup to /tmp/model/.
MODEL_DIR       : fallback local directory when AIP_STORAGE_URI is not set.
                  Default: /app/model  (used for local docker-compose testing)
TAU             : decision threshold for "flagged" field.  Default: 0.12
PORT            : server port, set by the Dockerfile.     Default: 8080
"""

import logging
import os
import pathlib
import shutil
import tempfile
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
TAU = float(os.environ.get("TAU", "0.12"))

# Vertex AI sets AIP_STORAGE_URI; fall back to MODEL_DIR for local use.
_AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "").rstrip("/")
_LOCAL_MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", "/app/model"))

_ARTIFACTS = ["xgb_model.joblib", "preprocessor.joblib"]

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


def _download_from_gcs(gcs_uri: str, dest_dir: pathlib.Path) -> None:
    """Download required artifacts from a GCS URI to a local directory."""
    from google.cloud import storage  # deferred — not needed for local runs
    dest_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket_name, _, prefix = gcs_uri.replace("gs://", "").partition("/")
    bucket = client.bucket(bucket_name)
    for filename in _ARTIFACTS:
        blob_name = f"{prefix}/{filename}" if prefix else filename
        dest_path = dest_dir / filename
        logger.info("Downloading gs://%s/%s → %s", bucket_name, blob_name, dest_path)
        bucket.blob(blob_name).download_to_filename(str(dest_path))


def _load_artifacts() -> None:
    global _model, _preprocessor
    if _AIP_STORAGE_URI:
        # Running on Vertex AI — download artifacts from GCS
        local_dir = pathlib.Path(tempfile.mkdtemp(prefix="model_"))
        logger.info("AIP_STORAGE_URI=%s  Downloading artifacts…", _AIP_STORAGE_URI)
        _download_from_gcs(_AIP_STORAGE_URI, local_dir)
    else:
        # Local docker-compose / manual testing — load from mounted volume
        local_dir = _LOCAL_MODEL_DIR
        logger.info("AIP_STORAGE_URI not set — loading from MODEL_DIR=%s", local_dir)

    model_path = local_dir / "xgb_model.joblib"
    prep_path  = local_dir / "preprocessor.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {prep_path}")
    _model        = joblib.load(model_path)
    _preprocessor = joblib.load(prep_path)
    logger.info("Artifacts loaded from %s  |  TAU=%.2f", local_dir, TAU)


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
