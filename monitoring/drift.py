"""
drift.py — Temporal drift monitoring on holdout batches.

Splits the holdout set into N temporal batches (sorted by encounter_id) to
simulate quarterly production scoring.  For each batch computes:

  PSI  (Population Stability Index) on top numeric features vs. training
  KS   (Kolmogorov-Smirnov) on predicted probability distribution vs. batch 1
  AUC-ROC   (labels are available in holdout)
  Flag rate at τ=0.12

PSI interpretation
------------------
  < 0.10  — no significant drift          (OK)
  0.10–0.20 — moderate drift              (monitor)
  > 0.20  — significant drift             (consider retraining)

Usage:
    python monitoring/drift.py [--n-batches 4] [--output monitoring/drift_report.json]
"""

import argparse
import json
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "ml" / "data"

TAU = 0.12

# Numeric features tracked for PSI — top contributors from SHAP analysis
PSI_FEATURES = [
    "number_inpatient",
    "num_medications",
    "time_in_hospital",
    "number_diagnoses",
    "num_lab_procedures",
    "number_emergency",
    "age_midpoint",
]

# Columns to drop before sending to the preprocessor
_DROP_COLS: set[str] = {
    "encounter_id", "patient_nbr", "readmitted_raw", "readmitted_any",
    "split_group", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "readmitted_30day",
    "glimepiride_pioglitazone", "metformin_pioglitazone",
    "metformin_rosiglitazone", "glipizide_metformin", "tolbutamide",
    "miglitol", "tolazamide", "chlorpropamide", "acarbose",
    "glyburide_metformin", "nateglinide",
}


def _sanitize(name: str) -> str:
    return name.replace("[", "(").replace("]", ")").replace("<", "lt")


def _predict(model, preprocessor, df_raw: pd.DataFrame) -> np.ndarray:
    X_in  = df_raw.drop(columns=[c for c in _DROP_COLS if c in df_raw.columns])
    X_arr = preprocessor.transform(X_in)
    names = [_sanitize(c) for c in preprocessor.get_feature_names_out()]
    X     = pd.DataFrame(X_arr, columns=names)
    return model.predict_proba(X)[:, 1]


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for one numeric feature."""
    breakpoints = np.unique(np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1)))
    if len(breakpoints) < 3:
        return 0.0  # degenerate — skip

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual,   bins=breakpoints)

    exp_pct = (exp_counts / exp_counts.sum()).clip(1e-4)
    act_pct = (act_counts / act_counts.sum()).clip(1e-4)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_drift(n_batches: int = 4) -> list[dict]:
    model        = joblib.load(DATA_DIR / "xgb_model.joblib")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.joblib")

    raw     = pd.read_parquet(DATA_DIR / "raw_features.parquet")
    train   = raw[raw["split_group"] == "train"]
    holdout = (
        raw[raw["split_group"] == "holdout"]
        .reset_index(drop=True)  # row order from Snowflake is already temporal
    )

    batch_size = len(holdout) // n_batches
    batches = [
        holdout.iloc[
            i * batch_size : (i + 1) * batch_size if i < n_batches - 1 else len(holdout)
        ].reset_index(drop=True)
        for i in range(n_batches)
    ]

    print(f"Holdout: {len(holdout):,} rows → {n_batches} batches of ~{batch_size:,}")
    print(f"Training reference: {len(train):,} rows  |  τ = {TAU}\n")

    # Batch 1 probabilities serve as the distribution baseline for KS
    batch1_probs = _predict(model, preprocessor, batches[0])

    results = []
    for i, batch in enumerate(batches):
        probs  = batch1_probs if i == 0 else _predict(model, preprocessor, batch)
        labels = batch["readmitted_30day"].values

        auc       = float(roc_auc_score(labels, probs)) if labels.sum() > 0 else float("nan")
        flag_rate = float((probs >= TAU).mean())
        pos_rate  = float(labels.mean())

        ks_stat = ks_pval = 0.0
        if i > 0:
            ks_stat, ks_pval = stats.ks_2samp(batch1_probs, probs)

        # PSI: each feature vs. training distribution
        psi_scores: dict[str, float] = {}
        for feat in PSI_FEATURES:
            if feat in batch.columns and feat in train.columns:
                psi_scores[feat] = _psi(
                    train[feat].dropna().values,
                    batch[feat].dropna().values,
                )
        mean_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0

        drift_alert = mean_psi > 0.20 or (i > 0 and ks_stat > 0.10)

        record: dict = {
            "batch":        i + 1,
            "n_rows":       len(batch),
            "auc_roc":      round(auc,       4),
            "flag_rate":    round(flag_rate,  4),
            "pos_rate":     round(pos_rate,   4),
            "ks_stat":      round(float(ks_stat), 4),
            "ks_pval":      round(float(ks_pval), 4),
            "mean_psi":     round(mean_psi,   4),
            "psi_by_feature": {k: round(v, 4) for k, v in psi_scores.items()},
            "drift_alert":  drift_alert,
        }
        results.append(record)

        status = "ALERT" if drift_alert else "OK"
        print(f"Batch {i+1}/{n_batches}  [{status}]")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(f"  Flag rate : {flag_rate:.3f}  (pos rate: {pos_rate:.3f})")
        print(f"  KS stat   : {ks_stat:.4f}  (p={ks_pval:.3f})")
        print(f"  Mean PSI  : {mean_psi:.4f}")
        if drift_alert:
            high_psi = {k: v for k, v in psi_scores.items() if v > 0.10}
            if high_psi:
                print(f"  PSI flags : {high_psi}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--output",    default="monitoring/drift_report.json")
    args = parser.parse_args()

    results = compute_drift(n_batches=args.n_batches)

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDrift report → {out_path.relative_to(REPO_ROOT)}")
