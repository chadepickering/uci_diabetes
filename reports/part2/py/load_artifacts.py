"""
load_artifacts.py
Shared loader for Part 2 Quarto reports.

Sourced at the top of each .qmd chunk via:
    exec(open(here("reports/part2/py/load_artifacts.py")).read())

Loads all model artifacts and data splits from ml/data/ — no Snowflake
connection needed; everything is pre-processed local parquet / JSON / CSV.
"""

import pathlib
import json
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

DATA_DIR  = REPO_ROOT / "ml" / "data"
ML_DIR    = REPO_ROOT / "ml"

TARGET  = "readmitted_30day"
TAU     = 0.12   # deployment threshold — τ=0.12 flags ~16% of patients, consistent with
                 # Part 1 power analysis target (~15.7%). τ=0.10 metrics cached in JSON files.
TAU_10  = 0.10   # retained for reference; τ=0.10 metrics cached in *_metrics.json


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
    return df.drop(columns=[TARGET]), df[TARGET]

X_train, y_train = _load_split("train")
X_val,   y_val   = _load_split("validation")
X_holdout, y_holdout = _load_split("holdout")

feature_names = (DATA_DIR / "feature_names.txt").read_text().splitlines()


# ---------------------------------------------------------------------------
# Metrics JSON
# ---------------------------------------------------------------------------
def _load_metrics(path: pathlib.Path) -> dict:
    records = json.loads(path.read_text())
    return {r["split"]: r for r in records}

lr_metrics   = _load_metrics(DATA_DIR / "lr_metrics.json")
xgb_metrics  = _load_metrics(DATA_DIR / "xgb_metrics.json")
lgbm_metrics = _load_metrics(DATA_DIR / "lgbm_metrics.json")
tf_metrics   = _load_metrics(DATA_DIR / "tf_metrics.json")


# ---------------------------------------------------------------------------
# Feature importance CSVs
# ---------------------------------------------------------------------------
lr_coef_df   = pd.read_csv(DATA_DIR / "lr_coefficients.csv")
xgb_shap_df  = pd.read_csv(DATA_DIR / "xgb_shap_importance.csv")
lgbm_shap_df = pd.read_csv(DATA_DIR / "lgbm_shap_importance.csv")
tf_ig_df     = pd.read_csv(DATA_DIR / "tf_ig_importance.csv")


# ---------------------------------------------------------------------------
# Model loading helpers (lazy — call only when needed to avoid TF import cost)
# ---------------------------------------------------------------------------
def load_sklearn_model(name: str):
    """Load a joblib-serialized sklearn model (lr, xgb, lgbm)."""
    return joblib.load(DATA_DIR / f"{name}_model.joblib")


def load_tf_model():
    """Load TF/Keras model + isotonic calibrator."""
    import tensorflow as tf  # deferred import
    model      = tf.keras.models.load_model(DATA_DIR / "tf_model.keras")
    calibrator = joblib.load(DATA_DIR / "tf_calibrator.joblib")
    return model, calibrator


def tf_predict_proba(model, calibrator, X: np.ndarray) -> np.ndarray:
    """Run TF model + isotonic calibrator → calibrated probabilities."""
    raw = model.predict(X, verbose=0).ravel()
    return np.clip(calibrator.predict(raw), 0, 1)


# ---------------------------------------------------------------------------
# Preprocessor (for re-transforming raw features if needed)
# ---------------------------------------------------------------------------
def load_preprocessor():
    return joblib.load(DATA_DIR / "preprocessor.joblib")


# ---------------------------------------------------------------------------
# Convenience: summary metrics table across all splits/models
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "auc_roc", "auc_pr", "brier",
    "flagged_rate", "precision_tau", "recall_tau", "f1_tau",
]

def metrics_table(split: str = "validation") -> pd.DataFrame:
    """Return a wide comparison table for a given split."""
    rows = []
    for label, m in [
        ("LR",       lr_metrics),
        ("XGBoost",  xgb_metrics),
        ("LightGBM", lgbm_metrics),
        ("TF/Keras", tf_metrics),
    ]:
        row = {"model": label}
        row.update({k: m[split][k] for k in METRIC_COLS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
