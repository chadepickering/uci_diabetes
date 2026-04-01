# ml/data/ — Generated Artifacts

All files in this directory are generated and excluded from version control
(`.parquet` and `.joblib` are gitignored). Run the steps below to reproduce them.

---

## Prerequisites

Activate the ML virtual environment and ensure `.env` is present at the repo root
with `SNOWFLAKE_USER` and `SNOWFLAKE_PASSWORD` set.

```bash
source venv_ml/bin/activate
```

---

## Step 1 — Preprocessing

```bash
python ml/sklearn/preprocess.py
```

Connects to Snowflake (`UCI_DIABETES.MARTS.DIABETES_FEATURES`), applies the full
sklearn `ColumnTransformer` pipeline, and writes:

| File | Description |
|---|---|
| `raw_features.parquet` | Raw 69,581-row pull from Snowflake (cache for `--dry-run`) |
| `train.parquet` | 48,706 rows — 70% temporal split |
| `validation.parquet` | 6,958 rows — 10% temporal split |
| `holdout.parquet` | 13,917 rows — 20% temporal split (held until final evaluation) |
| `preprocessor.joblib` | Fitted `ColumnTransformer` (training-set statistics only) |
| `feature_names.txt` | Ordered list of 74 output feature names |

Use `--dry-run` to skip Snowflake and load from `raw_features.parquet` if cached.

---

## Step 2 — Model Training

Each script runs an Optuna hyperparameter search, fits a final calibrated model,
and writes a `.joblib`, a `_metrics.json`, and a feature importance file.
See the corresponding README in `ml/sklearn/` for full model details.

### Step 2a — Logistic Regression Baseline

```bash
python ml/sklearn/train_lr.py [--n-trials N]
```

| File | Description |
|---|---|
| `lr_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `LogisticRegression` |
| `lr_metrics.json` | Train + validation metrics |
| `lr_coefficients.csv` | All 74 features ranked by `abs(coefficient)` |

→ See [ml/sklearn/README_lr.md](../sklearn/README_lr.md)

### Step 2b — XGBoost

```bash
python ml/sklearn/train_xgb.py [--n-trials N]
```

| File | Description |
|---|---|
| `xgb_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `XGBClassifier` |
| `xgb_metrics.json` | Train + validation metrics |
| `xgb_shap_importance.csv` | Mean \|SHAP\| per feature on validation set, ranked |

→ See [ml/sklearn/README_xgb.md](../sklearn/README_xgb.md)

### Step 2c — LightGBM

```bash
python ml/sklearn/train_lgbm.py [--n-trials N]
```

| File | Description |
|---|---|
| `lgbm_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `LGBMClassifier` |
| `lgbm_metrics.json` | Train + validation metrics |
| `lgbm_shap_importance.csv` | Mean \|SHAP\| per feature on validation set, ranked |

→ See [ml/sklearn/README_lgbm.md](../sklearn/README_lgbm.md)

---

## Model Comparison (validation set, τ=0.10)

| Metric | LR | XGBoost | LightGBM |
|---|---|---|---|
| AUC-ROC | 0.676 | **0.687** | 0.685 |
| AUC-PR | **0.128** | 0.113 | 0.125 |
| Brier | 0.052 | 0.052 | 0.052 |
| Recall@τ | 0.366 | **0.512** | 0.501 |
| Precision@τ | 0.107 | **0.111** | 0.106 |
| Train-val gap | **0.023** | 0.099 | 0.141 |

XGBoost leads on AUC-ROC and recall at τ=0.10. Holdout evaluation is the tiebreaker.
