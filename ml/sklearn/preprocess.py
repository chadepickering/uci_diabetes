"""
preprocess.py — Step 1 of the UCI Diabetes Part 2 ML pipeline.

Loads DIABETES_FEATURES from Snowflake, applies full preprocessing via
sklearn Pipeline objects, and writes train/validation/holdout splits as
Parquet files under ml/data/.

Usage:
    python ml/sklearn/preprocess.py [--dry-run]

    --dry-run  Skip Snowflake; load from ml/data/raw_features.parquet if present.
"""

import argparse
import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "ml" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_CACHE = DATA_DIR / "raw_features.parquet"
PIPELINE_PATH = DATA_DIR / "preprocessor.joblib"

# ---------------------------------------------------------------------------
# Column definitions (from Part 1 EDA)
# ---------------------------------------------------------------------------

# Identifier / leakage columns — drop before modelling
ID_COLS = [
    "encounter_id",
    "patient_nbr",
    "readmitted_raw",
    "readmitted_any",
    "split_group",
    # raw IDs are informative only; kept separately if needed
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# Near-zero variance columns documented in EDA — drop
NZV_COLS = [
    "glimepiride_pioglitazone",
    "metformin_pioglitazone",
    "metformin_rosiglitazone",
    "glipizide_metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
    "chlorpropamide",
    "acarbose",
    "glyburide_metformin",
    "nateglinide",
]

TARGET = "readmitted_30day"

# Medication columns: 4-level ordered (No < Steady < Down < Up)
# Treating as ordinal preserves the dose-change direction signal.
MED_COLS = [
    "metformin",
    "repaglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
]
MED_ORDER = [["No", "Steady", "Down", "Up"]]  # shared order for all med cols

# Ordinal categoricals with defined order
# max_glu_serum / a1cresult: "None" is clinically meaningful (test not done)
A1C_ORDER = [["None", "Norm", ">7", ">8"]]
GLU_ORDER = [["None", "Norm", ">200", ">300"]]

# Nominal categoricals (one-hot encode)
NOMINAL_COLS = [
    "race",
    "gender",
    "age_band",
    "age_group",
    "admission_type_group",
    "specialty_group",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "change",
    "diabetes_med",
]

# Continuous / numeric columns (standard-scale)
NUMERIC_COLS = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_prior_encounters",
    "num_med_changes",
    "num_meds_active",
]


# ---------------------------------------------------------------------------
# Snowflake loader
# ---------------------------------------------------------------------------

def load_from_snowflake() -> pd.DataFrame:
    """Pull DIABETES_FEATURES mart from Snowflake and return as DataFrame."""
    import snowflake.connector  # imported lazily so dry-run skips it

    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account="ar29154.us-central1.gcp",
        warehouse="COMPUTE_WH",
        database="UCI_DIABETES",
        schema="MARTS",
        role="TRANSFORMER",
    )
    query = "SELECT * FROM UCI_DIABETES.MARTS.DIABETES_FEATURES"
    print(f"Querying Snowflake: {query}")
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.lower()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from Snowflake.")
    return df


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by the split_group column produced in dbt."""
    train = df[df["split_group"] == "train"].copy()
    val   = df[df["split_group"] == "validation"].copy()
    hold  = df[df["split_group"] == "holdout"].copy()
    return train, val, hold


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier and NZV columns (keep TARGET and split_group)."""
    to_drop = [c for c in (ID_COLS + NZV_COLS) if c in df.columns]
    return df.drop(columns=to_drop)


def build_preprocessor(df_train: pd.DataFrame) -> ColumnTransformer:
    """
    Fit a ColumnTransformer on the training set.

    Transformers:
      - numeric   : StandardScaler
      - med_ord   : OrdinalEncoder (No < Steady < Down < Up)
      - a1c_ord   : OrdinalEncoder (None < Norm < >7 < >8)
      - glu_ord   : OrdinalEncoder (None < Norm < >200 < >300)
      - nominal   : OneHotEncoder  (drop='first', handle_unknown='ignore')

    Columns present in the schema but absent from the DataFrame are silently
    skipped so the pipeline degrades gracefully if the mart ever changes.
    """
    present = set(df_train.columns)

    num_cols    = [c for c in NUMERIC_COLS if c in present]
    med_cols    = [c for c in MED_COLS     if c in present]
    nom_cols    = [c for c in NOMINAL_COLS if c in present]
    a1c_cols    = ["a1cresult"]    if "a1cresult"    in present else []
    glu_cols    = ["max_glu_serum"] if "max_glu_serum" in present else []

    print("\nColumn groups sent to ColumnTransformer:")
    print(f"  numeric  ({len(num_cols)}): {num_cols}")
    print(f"  med_ord  ({len(med_cols)}): {med_cols}")
    print(f"  a1c_ord  ({len(a1c_cols)}): {a1c_cols}")
    print(f"  glu_ord  ({len(glu_cols)}): {glu_cols}")
    print(f"  nominal  ({len(nom_cols)}): {nom_cols}")

    transformers = []

    if num_cols:
        transformers.append((
            "numeric",
            StandardScaler(),
            num_cols,
        ))

    if med_cols:
        transformers.append((
            "med_ord",
            OrdinalEncoder(
                categories=MED_ORDER * len(med_cols),
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            med_cols,
        ))

    if a1c_cols:
        transformers.append((
            "a1c_ord",
            OrdinalEncoder(
                categories=A1C_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            a1c_cols,
        ))

    if glu_cols:
        transformers.append((
            "glu_ord",
            OrdinalEncoder(
                categories=GLU_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            glu_cols,
        ))

    if nom_cols:
        transformers.append((
            "nominal",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ),
            nom_cols,
        ))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # anything not listed is excluded
        verbose_feature_names_out=True,
    )

    return ct


def get_feature_names(ct: ColumnTransformer) -> list[str]:
    """Return feature names after the ColumnTransformer is fitted."""
    return list(ct.get_feature_names_out())


def transform_split(
    ct: ColumnTransformer,
    df: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the fitted transformer; return (X, y) for a split."""
    y = df[TARGET]
    X_arr = ct.transform(df.drop(columns=[TARGET]))
    feature_names = get_feature_names(ct)
    X = pd.DataFrame(X_arr, columns=feature_names, index=df.index)
    print(f"  {split_name:12s}: X={X.shape},  y pos-rate={y.mean():.3f}")
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    # 1. Load raw data --------------------------------------------------
    if dry_run and RAW_CACHE.exists():
        print(f"[dry-run] Loading from cache: {RAW_CACHE}")
        df_raw = pd.read_parquet(RAW_CACHE)
    else:
        df_raw = load_from_snowflake()
        df_raw.to_parquet(RAW_CACHE, index=False)
        print(f"Cached raw data → {RAW_CACHE}")

    print(f"\nRaw shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}\n")

    # 2. Split by split_group before any transformation ----------------
    train_raw, val_raw, hold_raw = split_data(df_raw)
    print(f"Split sizes — train: {len(train_raw):,}  val: {len(val_raw):,}  holdout: {len(hold_raw):,}")

    # 3. Drop ID / NZV columns ----------------------------------------
    train_clean = drop_columns(train_raw)
    val_clean   = drop_columns(val_raw)
    hold_clean  = drop_columns(hold_raw)

    # 4. Fit preprocessor on training set only -------------------------
    print("\nFitting preprocessor on training set…")
    ct = build_preprocessor(train_clean.drop(columns=[TARGET]))
    ct.fit(train_clean.drop(columns=[TARGET]))
    print(f"Preprocessor fitted. Output features: {len(get_feature_names(ct))}")

    # 5. Transform all splits -----------------------------------------
    print("\nTransforming splits:")
    X_train, y_train = transform_split(ct, train_clean, "train")
    X_val,   y_val   = transform_split(ct, val_clean,   "validation")
    X_hold,  y_hold  = transform_split(ct, hold_clean,  "holdout")

    # 6. Class imbalance summary (informational) ----------------------
    print(f"\nClass imbalance (train): {y_train.value_counts().to_dict()}")
    print(f"  Positive rate: {y_train.mean():.3f}")
    print(
        "  → Models should use class_weight='balanced' or scale_pos_weight; "
        "evaluate SMOTE separately during model selection."
    )

    # 7. Save processed splits to Parquet -----------------------------
    for split_name, X, y in [
        ("train",      X_train, y_train),
        ("validation", X_val,   y_val),
        ("holdout",    X_hold,  y_hold),
    ]:
        out = DATA_DIR / f"{split_name}.parquet"
        pd.concat([X, y.rename(TARGET)], axis=1).to_parquet(out, index=False)
        print(f"Saved {split_name:12s} → {out.relative_to(REPO_ROOT)}")

    # 8. Save fitted preprocessor (needed for inference) --------------
    joblib.dump(ct, PIPELINE_PATH)
    print(f"Saved preprocessor  → {PIPELINE_PATH.relative_to(REPO_ROOT)}")

    # 9. Feature name manifest ----------------------------------------
    feat_manifest = DATA_DIR / "feature_names.txt"
    feat_manifest.write_text("\n".join(get_feature_names(ct)) + "\n")
    print(f"Saved feature names → {feat_manifest.relative_to(REPO_ROOT)}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Snowflake; load from cached parquet if available.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
