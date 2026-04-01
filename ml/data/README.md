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

**Splits:** Temporal split via encounter_id percentile (no explicit date column).
Outcome: `readmitted_30day` (binary). Positive rates: 7.1% train / 5.6% val / 5.6%
holdout. Temporal drift in outcome rate is documented and expected.

**Features (74 total):** 12 numeric (StandardScaler), 8 medication ordinal
(No < Steady < Down < Up), 2 lab ordinal (a1cresult, max_glu_serum), 11 nominal
one-hot encoded. 11 near-zero-variance columns and identifier/leakage columns
dropped in preprocessing. See `feature_names.txt` for full list.

---

## Step 2 — Model Training

**Class imbalance (~13:1):** All models apply imbalance correction:
- LR, LightGBM, TF: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=13.04`

**Calibration:** All models apply post-hoc probability calibration so τ=0.10
threshold metrics are meaningful. LR/XGBoost/LightGBM use Platt scaling
(`CalibratedClassifierCV(method='sigmoid', cv=5)`). TF uses isotonic regression
on 5-fold OOF training probabilities.

**Clinical threshold:** τ=0.10 per Part 1 experimental design (~15.7% of patients
flagged, 3pp ARR MDE, group sequential O'Brien-Fleming, 4 analyses).

---

### Step 2a — Logistic Regression Baseline

```bash
python ml/sklearn/train_lr.py [--n-trials N]
```

ElasticNet (`solver='saga'`, sklearn ≥ 1.8 compatible). Optuna TPE, 50 trials,
5-fold stratified CV on training set. `max_iter=5000`, `tol=1e-4`.

| File | Description |
|---|---|
| `lr_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `LogisticRegression` |
| `lr_metrics.json` | Train + validation metrics |
| `lr_coefficients.csv` | All 74 features ranked by `abs(coefficient)` |

**Best params:** C=0.017534, l1_ratio=0.651 (substantial L1 — 26/74 features zeroed),
CV AUC=0.693. 48 non-zero coefficients provide principled feature selection output.

→ See [ml/sklearn/README_lr.md](../sklearn/README_lr.md)

---

### Step 2b — XGBoost

```bash
python ml/sklearn/train_xgb.py [--n-trials N]
```

Optuna TPE, 100 trials, validation set + early stopping (30 rounds).
`n_estimators` range [150, 600]. No monotone constraints (tested, degraded
validation AUC from 0.674 → 0.644 due to depth compensation with sparse positive class).

| File | Description |
|---|---|
| `xgb_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `XGBClassifier` |
| `xgb_metrics.json` | Train + validation metrics |
| `xgb_shap_importance.csv` | Mean \|SHAP\| per feature on validation set, ranked |

**Best params:** n_estimators=217, max_depth=5, lr=0.041, subsample=0.618,
colsample_bytree=0.614, min_child_weight=4, gamma=0.123, reg_alpha=5.045
(strong L1 leaf regularization — primary anti-overfit lever), reg_lambda=0.028.

**Feature names sanitized:** `[` → `(`, `]` → `)`, `<` → `lt` (XGBoost restriction).

→ See [ml/sklearn/README_xgb.md](../sklearn/README_xgb.md)

---

### Step 2c — LightGBM

```bash
python ml/sklearn/train_lgbm.py [--n-trials N]
```

Leaf-wise growth. Optuna TPE, 100 trials, validation set + early stopping (30 rounds).
`num_leaves` range [15, 60] (initial range 20–150 caused severe overfitting:
train AUC 0.965, val AUC 0.673, gap 0.292 — capped to fix).

| File | Description |
|---|---|
| `lgbm_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `LGBMClassifier` |
| `lgbm_metrics.json` | Train + validation metrics |
| `lgbm_shap_importance.csv` | Mean \|SHAP\| per feature on validation set, ranked |

**Best params:** n_estimators=201, num_leaves=56, lr=0.033, subsample=0.997,
colsample_bytree=0.586, min_child_samples=132 (absolute leaf floor — more
interpretable than XGBoost's hessian-sum for sparse classes), reg_alpha=3.190.

→ See [ml/sklearn/README_lgbm.md](../sklearn/README_lgbm.md)

---

### Step 2d — TensorFlow/Keras Neural Network

```bash
python ml/tensorflow/train_tf.py [--n-trials N]
```

MLP: 128→64→32, ReLU, BatchNorm, Dropout(0.3/0.3/0.2). Adam + cosine decay
(`decay_steps=50×steps_per_epoch`, `alpha=0.01` floor). Optuna TPE, 25 trials,
tuning lr [1e-4, 1e-3] and batch_size {256, 512, 1024}. Isotonic regression
calibration (more flexible than Platt for heavy-tailed distributions).
Feature attribution via Integrated Gradients (`tf.GradientTape`, zero baseline,
50 Riemann steps).

**Note on save format:** Keras model state cannot be pickled. Saved as two files:
```python
model      = tf.keras.models.load_model("ml/data/tf_model.keras")
calibrator = joblib.load("ml/data/tf_calibrator.joblib")
probs      = np.clip(calibrator.predict(model.predict(X).ravel()), 0, 1)
```

| File | Description |
|---|---|
| `tf_model.keras` | Trained Keras model (native format, TF Serving compatible) |
| `tf_calibrator.joblib` | Fitted isotonic regression calibrator |
| `tf_metrics.json` | Train + validation metrics |
| `tf_ig_importance.csv` | Mean \|IG attribution\| per feature on validation set, ranked |

**Best params:** lr=0.000982, batch_size=512. Best epoch: 6, Optuna val AUC=0.6879.
Cosine decay critical: using `MAX_EPOCHS×steps` caused near-flat LR in early-stopping
window (val AUC 0.675); fixing to `50×steps` recovered to 0.689.

→ See [ml/tensorflow/README_tf.md](../tensorflow/README_tf.md)

---

## Model Comparison (validation set, τ=0.10)

### Discrimination and Calibration

| Metric | LR | XGBoost | LightGBM | TF/Keras |
|---|---|---|---|---|
| AUC-ROC | 0.676 | 0.687 | 0.685 | **0.689** |
| AUC-PR | **0.128** | 0.113 | 0.125 | **0.128** |
| Brier score | 0.052 | 0.052 | 0.052 | 0.052 |
| Train AUC | 0.699 | 0.785 | 0.826 | 0.721 |
| Train-val gap | **0.023** | 0.099 | 0.141 | 0.032 |

### Clinical Operating Point (τ=0.10)

| Metric | LR | XGBoost | LightGBM | TF/Keras |
|---|---|---|---|---|
| Flagged | 19.4% | 26.1% | 26.7% | 31.0% |
| Flagged pos rate | 10.7% | 11.1% | 10.6% | 10.4% |
| Precision@τ | **0.107** | **0.111** | 0.106 | 0.104 |
| Recall@τ | 0.366 | 0.512 | 0.501 | **0.570** |
| F1@τ | 0.165 | **0.182** | 0.175 | 0.176 |

### Interpretation

- **TF/Keras** leads on AUC-ROC (0.689), AUC-PR (ties LR at 0.128), and recall
  at τ=0.10 (0.570). Higher flagged rate (31%) reflects a more liberal threshold
  calibration — clinically appropriate for recall-oriented screening.
- **XGBoost** leads on precision (0.111) and F1 (0.182), flags fewest patients
  among tree models (26.1%), and has the most conservative operating profile.
- **LR** has the smallest train-val gap (0.023) and ties TF on AUC-PR — the
  dominant signal in this dataset is largely linear; tree/NN models provide
  modest incremental lift.
- **LightGBM** is competitive but does not lead on any metric; leaf-wise growth
  offers no clear advantage over XGBoost on this dataset size and class sparsity.

### Top Features by Attribution Method (validation set)

| Rank | LR (|coef|) | XGBoost (|SHAP|) | LightGBM (|SHAP|) | TF/Keras (|IG|) |
|---|---|---|---|---|
| 1 | number_inpatient (+) | time_in_hospital | time_in_hospital | time_in_hospital |
| 2 | time_in_hospital (+) | number_inpatient | number_inpatient | insulin (+) |
| 3 | diag_1_Respiratory (−) | total_prior_encounters | num_lab_procedures | race_Caucasian |
| 4 | diabetes_med_Yes (+) | num_lab_procedures | number_diagnoses | number_diagnoses |
| 5 | number_diagnoses (+) | number_diagnoses | total_prior_encounters | num_lab_procedures |

**Cross-model consistency:** `time_in_hospital`, `number_inpatient`, and
`number_diagnoses` appear in the top 5 across all four attribution methods.
`total_prior_encounters` ranks low in LR (#45) but high in tree/NN models,
indicating non-linear utilization signal. `insulin` rises in TF/Keras consistent
with Part 1 causal estimate (PSM OR ~1.17). `diag_1_group_Respiratory` is
protective across all models, consistent with the clinical interpretation that
acute respiratory admissions are discrete events rather than chronic high-utilizer
presentations.

**Holdout evaluation is the next step and tiebreaker between models.**
