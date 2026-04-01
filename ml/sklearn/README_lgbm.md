# LightGBM Model — Step 2c

## Reproduction

```bash
source venv_ml/bin/activate
python ml/sklearn/train_lgbm.py [--n-trials N]
```

Default: 100 Optuna trials. Requires preprocessed splits in `ml/data/` (run `preprocess.py` first).

## Outputs

| File | Description |
|---|---|
| `lgbm_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `LGBMClassifier` |
| `lgbm_metrics.json` | Train + validation metrics including best hyperparameters |
| `lgbm_shap_importance.csv` | Mean \|SHAP\| per feature on validation set, ranked |

## Design Decisions

### Class Imbalance
`class_weight='balanced'` — LightGBM's native interface, equivalent to XGBoost's
`scale_pos_weight`. Followed by Platt scaling calibration consistent with LR and
XGBoost approaches.

### Leaf-Wise Growth vs Level-Wise (XGBoost)
LightGBM grows trees leaf-wise rather than level-wise: at each step it expands the
leaf with the largest loss reduction. This finds better fits faster but overfits more
aggressively on sparse positive classes. `num_leaves` is the primary complexity
control — it directly caps the number of leaves regardless of depth.

### `num_leaves` Range
Initial search over 20–150 resulted in `num_leaves=114` and severe overfitting
(train AUC 0.965, val AUC 0.673, gap 0.292). Range was capped to **15–60** on the
second run, which reduced the gap to 0.141 and improved val AUC to 0.685.

### `min_child_samples`
Absolute sample count per leaf (range 20–200). More interpretable than XGBoost's
hessian-sum `min_child_weight` for a sparse 7% positive class. Best value of 132
enforces a meaningful minimum of real samples per leaf.

### `subsample_freq=1`
Required to activate LightGBM's row subsampling (bagging). Without it, the
`subsample` parameter has no effect.

### Hyperparameter Search
Optuna TPE (100 trials) optimizing validation AUC-ROC with LightGBM early stopping
(`early_stopping_rounds=30`). Validation set used directly as eval set. Holdout
never touched.

### Calibration
`CalibratedClassifierCV(method='sigmoid', cv=5)` fit on training set only, consistent
with LR and XGBoost.

### Feature Name Sanitization
LightGBM also forbids `[`, `]`, `<` in feature names. Same sanitization as XGBoost:
`[` → `(`, `]` → `)`, `<` → `lt`. Cosmetic only.

## Best Hyperparameters (final run, num_leaves 15–60)

| Parameter | Value | Notes |
|---|---|---|
| `n_estimators` | 201 | |
| `num_leaves` | 56 | Near ceiling — optimizer pushed to upper bound |
| `learning_rate` | 0.033 | Low — consistent with XGBoost's best config |
| `subsample` | 0.997 | Near full row sampling |
| `colsample_bytree` | 0.586 | ~59% of features per tree |
| `min_child_samples` | 132 | High — strong leaf sample floor |
| `reg_alpha` | 3.190 | Moderate-strong L1 |
| `reg_lambda` | 0.005 | Near-zero L2 |
| Validation AUC | 0.6893 | |

## Performance (calibrated probabilities)

| Split | AUC-ROC | AUC-PR | Brier | Flagged | Flagged pos rate | Precision@τ | Recall@τ | F1@τ |
|---|---|---|---|---|---|---|---|---|
| Train | 0.826 | 0.263 | 0.061 | 23.8% | 20.4% | 0.204 | 0.682 | 0.314 |
| Validation | 0.685 | 0.125 | 0.052 | 26.7% | 10.6% | 0.106 | 0.501 | 0.175 |

### Train–Validation Gap
AUC-ROC gap of 0.141 — reduced from 0.292 in the first run by capping `num_leaves`.
Leaf-wise growth still overfits more than XGBoost's level-wise approach on this
dataset, but the gap is now in a reasonable range.

### Tuning History

| Run | `num_leaves` range | Val AUC-ROC | Train AUC | Gap |
|---|---|---|---|---|
| v1 | 20–150 | 0.673 | 0.965 | 0.292 |
| v2 | 15–60 | 0.685 | 0.826 | 0.141 |

### Three-Model Comparison (validation set)

| Metric | LR | XGBoost | LightGBM |
|---|---|---|---|
| AUC-ROC | 0.676 | **0.687** | 0.685 |
| AUC-PR | **0.128** | 0.113 | 0.125 |
| Brier | 0.052 | 0.052 | 0.052 |
| Recall@τ | 0.366 | **0.512** | 0.501 |
| Precision@τ | **0.107** | 0.111 | 0.106 |
| Train-val gap | **0.023** | 0.099 | 0.141 |

XGBoost leads on AUC-ROC and recall at τ=0.10. LightGBM is close on both and
outperforms XGBoost on AUC-PR (0.125 vs 0.113), nearly matching LR (0.128). LR
remains the most calibrated model by train-val gap. Holdout evaluation is the
tiebreaker.

## SHAP Feature Importance (Top 15, validation set)

| Feature | Mean \|SHAP\| | XGBoost rank | LR rank | Notes |
|---|---|---|---|---|
| `numeric__time_in_hospital` | 0.350 | #1 | #2 | Consistent across all models |
| `numeric__number_inpatient` | 0.174 | #2 | #1 | Consistent — AIPW RR ~2.0 in Part 1 |
| `numeric__num_lab_procedures` | 0.137 | #4 | #11 | Rises in both tree models |
| `numeric__number_diagnoses` | 0.137 | #5 | #5 | Consistent |
| `numeric__total_prior_encounters` | 0.135 | #3 | #45 | Non-linear utilization signal |
| `nominal__diag_1_group_Respiratory` | 0.091 | #8 | #3 | Consistent direction (protective) |
| `numeric__age_midpoint` | 0.090 | #6 | #18 | Rises in tree models |
| `numeric__num_medications` | 0.087 | #7 | #32 | Rises in tree models |
| `med_ord__insulin` | 0.058 | #9 | #39 | Consistent direction (positive) |
| `numeric__num_meds_active` | 0.053 | #10 | #41 | Near-zero in LR; rises in trees |
| `a1c_ord__a1cresult` | — | #13 | #37 | Present but lower in LightGBM top 15 |

Top feature ordering is highly consistent between XGBoost and LightGBM, reinforcing
that the dominant signal sources are robust across tree architectures.
