# ml/data/ — Generated Artifacts

All files in this directory are generated and excluded from version control
(`.parquet` and `.joblib` are gitignored). Run the steps below to reproduce them.

---

## Reproduction Steps

### Prerequisites
Activate the ML virtual environment and ensure `.env` is present at the repo root
with `SNOWFLAKE_USER` and `SNOWFLAKE_PASSWORD` set.

```bash
source venv_ml/bin/activate
```

### Step 1 — Preprocessing

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

Use `--dry-run` to skip Snowflake and load from `raw_features.parquet` if already cached.

### Step 2a — Logistic Regression Baseline

```bash
python ml/sklearn/train_lr.py [--n-trials N]
```

Runs an Optuna hyperparameter search (default 50 trials) over `C` and `l1_ratio`
using 5-fold stratified CV on the training set (scoring: AUC-ROC), then fits the
final model on the full training set. Writes:

| File | Description |
|---|---|
| `lr_model.joblib` | Fitted `LogisticRegression` (ElasticNet, best params) |
| `lr_metrics.json` | Train + validation metrics including best hyperparameters |
| `lr_coefficients.csv` | All 74 features ranked by `abs(coefficient)` |

---

## Logistic Regression — Model Insights

### Configuration
- Penalty: ElasticNet (`solver='saga'`, `l1_ratio` passed directly — sklearn ≥ 1.8 compatible)
- Class weighting: `balanced` (~13:1 imbalance corrected), followed by Platt scaling calibration
- Calibration: `CalibratedClassifierCV(method='sigmoid', cv=5)` fit on training set only
- Hyperparameter search: Optuna TPE, 50 trials, 5-fold stratified CV (scoring: AUC-ROC)
- SAGA convergence: `max_iter=5000`, `tol=1e-4`

### Best Hyperparameters
| Parameter | Value | Interpretation |
|---|---|---|
| `C` | 0.017534 | Strong regularization |
| `l1_ratio` | 0.6508 | Substantial L1 component — variable selection active |
| CV AUC-ROC | 0.6932 | Honest cross-validated estimate |

### Performance (calibrated probabilities)
| Split | AUC-ROC | AUC-PR | Brier | Flagged | Flagged pos rate | Precision@τ | Recall@τ | F1@τ |
|---|---|---|---|---|---|---|---|---|
| Train | 0.699 | 0.148 | 0.064 | 16.7% | 14.9% | 0.149 | 0.350 | 0.209 |
| Validation | 0.676 | 0.128 | 0.052 | 19.4% | 10.7% | 0.107 | 0.366 | 0.165 |

Train → validation AUC gap is modest (0.023). Brier scores are low and well-scaled
after Platt calibration — flagged rates are realistic (~15-19%) and close to the
Part 1 experimental design expectation of ~15.7% at τ=0.10.

### Feature Selection
`l1_ratio=0.651` zeroed out 26 of 74 features, leaving **48 non-zero coefficients**.
Zeroed features include redundant utilization composites (`number_outpatient`,
several `number_emergency` variants), medication columns collinear with
`diabetes_med_Yes`, and low-signal diagnosis subcategories. The 48 retained
features are the recommended starting set for XGBoost and the neural network.

### Top Features by |Coefficient| (causal consistency check)
| Feature | Coefficient | Direction consistent with Part 1? |
|---|---|---|
| `numeric__number_inpatient` | +0.314 | **Yes — AIPW RR ~2.0 in Part 1** |
| `numeric__time_in_hospital` | +0.302 | Yes — longer stays → sicker patients |
| `nominal__diag_1_group_Respiratory` | −0.295 | Yes — acute respiratory events are discrete, not chronic high-utilizer presentations |
| `nominal__diabetes_med_Yes` | +0.236 | Expected — signals disease severity |
| `numeric__number_diagnoses` | +0.177 | Yes — comorbidity burden |
| `a1c_ord__a1cresult` | −0.029 | Yes — HbA1c testing is protective (IPTW OR ~0.918 in Part 1) |
| `med_ord__insulin` | +0.023 | Yes — positive association (PSM OR ~1.17 in Part 1) |
| `med_ord__metformin` | −0.080 | Plausible — active metformin use signals better-managed diabetes |
