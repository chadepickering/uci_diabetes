# XGBoost Model — Step 2b

## Reproduction

```bash
source venv_ml/bin/activate
python ml/sklearn/train_xgb.py [--n-trials N]
```

Default: 100 Optuna trials. Requires preprocessed splits in `ml/data/` (run `preprocess.py` first).

## Outputs

| File | Description |
|---|---|
| `xgb_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `XGBClassifier` |
| `xgb_metrics.json` | Train + validation metrics including best hyperparameters |
| `xgb_shap_importance.csv` | Mean \|SHAP\| per feature on validation set, ranked |

## Design Decisions

### Class Imbalance
`scale_pos_weight = 45236 / 3470 ≈ 13.04` — hardcoded from training set counts.
This is a dataset property, not a tunable hyperparameter. Native to XGBoost; no
post-hoc reweighting needed during search.

### Hyperparameter Search
Optuna TPE (100 trials) optimizing validation AUC-ROC with early stopping
(`early_stopping_rounds=30`). Validation set is used directly as the eval set —
appropriate for XGBoost's early stopping mechanism. Holdout is never touched.

### Calibration
`CalibratedClassifierCV(method='sigmoid', cv=5)` fit on training set only, consistent
with the LR baseline. XGBoost with `scale_pos_weight` is less severely miscalibrated
than balanced LR, but calibration is applied for clinical consistency and to ensure
τ=0.10 threshold metrics are meaningful.

### Feature Names
XGBoost forbids `[`, `]`, `<` in feature names. Age band OHE features (e.g.
`nominal__age_band_[80-90)`) are sanitized on load: `[` → `(`, `]` → `)`, `<` → `lt`.
This is cosmetic only — no data is altered.

### All 74 Features Used
Unlike ElasticNet LR (which zeroed 26 features via L1), XGBoost has four independent
regularization mechanisms (`gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`)
that handle irrelevant features through tree structure. Features with weak linear
signal may still participate in non-linear interaction splits.

## Best Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_estimators` | 246 | Early stopping found this sufficient |
| `max_depth` | 4 | Shallow — prevents overfitting on sparse positive class |
| `learning_rate` | 0.124 | Moderate |
| `subsample` | 0.623 | Stochastic — ~62% of rows per tree |
| `colsample_bytree` | 0.972 | Near full feature set per tree |
| `min_child_weight` | 9 | High — guards against splits on tiny positive-class subgroups |
| `gamma` | 3.83 | Strong minimum split-loss threshold — conservative split policy |
| `reg_alpha` | 0.274 | Moderate L1 on leaf weights |
| `reg_lambda` | 0.275 | Moderate L2 on leaf weights |
| Validation AUC | 0.6901 | Best trial (trial 52 of 100) |

`min_child_weight=9` and `gamma=3.83` are notably high, reflecting the optimizer's
preference for conservative splits — consistent with a 7% positive rate where
aggressive splitting risks memorizing noise.

## Performance (calibrated probabilities)

| Split | AUC-ROC | AUC-PR | Brier | Flagged | Flagged pos rate | Precision@τ | Recall@τ | F1@τ |
|---|---|---|---|---|---|---|---|---|
| Train | 0.826 | 0.280 | 0.061 | 21.8% | 21.1% | 0.211 | 0.645 | 0.318 |
| Validation | 0.674 | 0.106 | 0.053 | 23.2% | 10.6% | 0.106 | 0.433 | 0.170 |

### Train–Validation Gap
AUC-ROC gap of 0.152 (0.826 train vs 0.674 val) indicates moderate overfitting
despite regularization. This is expected — tree models memorize training patterns
more aggressively than linear models, and the positive class is sparse enough that
the model has room to over-index on training-set minority examples.

### Comparison with LR Baseline

| Metric | LR (val) | XGBoost (val) | Δ |
|---|---|---|---|
| AUC-ROC | 0.676 | 0.674 | −0.002 |
| AUC-PR | 0.128 | 0.106 | −0.022 |
| Brier | 0.052 | 0.053 | +0.001 |
| Recall@τ | 0.366 | 0.433 | +0.067 |
| Precision@τ | 0.107 | 0.106 | −0.001 |

XGBoost marginally trails LR on AUC-ROC and AUC-PR on the validation set despite
a much larger train AUC. This suggests the non-linear signal available in this dataset
is limited — the dominant predictors (`time_in_hospital`, `number_inpatient`,
`num_lab_procedures`) are strong individual signals that a linear model captures well.
XGBoost's recall gain at τ=0.10 (+6.7pp) may reflect better probability calibration
in the tails. Neither model has a decisive advantage at this stage; holdout evaluation
will be the tiebreaker.

## SHAP Feature Importance (Top 15, validation set)

| Feature | Mean \|SHAP\| | LR coefficient rank | Notes |
|---|---|---|---|
| `numeric__time_in_hospital` | 0.370 | #2 | Consistent across both models |
| `numeric__number_inpatient` | 0.190 | #1 | Consistent — AIPW RR ~2.0 in Part 1 |
| `numeric__num_lab_procedures` | 0.168 | #11 | Rises in importance with non-linear model |
| `numeric__total_prior_encounters` | 0.159 | #45 | Large jump — weak linear but strong non-linear signal |
| `numeric__number_diagnoses` | 0.113 | #5 | Consistent |
| `numeric__age_midpoint` | 0.106 | #18 | Rises — non-linear age effects captured |
| `numeric__num_medications` | 0.105 | #32 | Rises substantially |
| `nominal__diag_1_group_Respiratory` | 0.095 | #3 | Consistent direction (protective) |
| `numeric__num_meds_active` | 0.086 | #41 | Notable rise — LR zeroed it nearly out |
| `med_ord__insulin` | 0.053 | #39 | Consistent direction (positive) |

**Key divergence from LR:** `total_prior_encounters` jumps from near-zero LR
coefficient (#45) to 4th by SHAP — its signal is non-linear (high utilizers have
disproportionate risk beyond what a linear term captures). Similarly `num_medications`
and `num_meds_active` rise, suggesting interaction effects with other utilization
variables. This validates keeping all 74 features for XGBoost rather than using
the LR-selected 48.
