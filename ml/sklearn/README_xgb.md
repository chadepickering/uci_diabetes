# XGBoost Model

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
| `n_estimators` | 217 | |
| `max_depth` | 5 | Moderate depth |
| `learning_rate` | 0.041 | Low — slow learning, better generalization |
| `subsample` | 0.618 | Stochastic — ~62% of rows per tree |
| `colsample_bytree` | 0.614 | ~61% of features per tree |
| `min_child_weight` | 4 | Moderate — prevents splits on tiny subgroups |
| `gamma` | 0.123 | Low — optimizer found L1 leaf regularization more useful |
| `reg_alpha` | 5.045 | Strong L1 on leaf weights — key to generalization this run |
| `reg_lambda` | 0.028 | Low L2 |
| Validation AUC | 0.6907 | |

`reg_alpha=5.05` is notably high — the optimizer converged on strong L1 leaf
regularization as the primary anti-overfit mechanism, combined with a low learning
rate. This reduced the train-val AUC gap from 0.152 to 0.099 vs the previous run.

## Performance (calibrated probabilities)

| Split | AUC-ROC | AUC-PR | Brier | Flagged | Flagged pos rate | Precision@τ | Recall@τ | F1@τ |
|---|---|---|---|---|---|---|---|---|
| Train | 0.785 | 0.241 | 0.061 | 23.7% | 18.4% | 0.184 | 0.612 | 0.283 |
| Validation | 0.687 | 0.113 | 0.052 | 26.1% | 11.1% | 0.111 | 0.512 | 0.182 |

### Train–Validation Gap
AUC-ROC gap of 0.099 (0.785 train vs 0.687 val) — substantially reduced from the
previous run (0.152). Strong `reg_alpha` combined with low `learning_rate` was the
effective combination. Some overfitting remains, which is expected for a tree model
on a sparse positive class.

### Comparison with LR Baseline

| Metric | LR (val) | XGBoost (val) | Δ |
|---|---|---|---|
| AUC-ROC | 0.676 | **0.687** | **+0.011** |
| AUC-PR | 0.128 | 0.113 | −0.015 |
| Brier | 0.052 | 0.052 | 0.000 |
| Recall@τ | 0.366 | **0.512** | **+0.146** |
| Precision@τ | 0.107 | 0.111 | +0.004 |

XGBoost now clearly leads on AUC-ROC (+1.1pp) and recall at τ=0.10 (+14.6pp) —
the clinically most important metric for flagging high-risk patients. LR retains an
edge on AUC-PR, indicating it is more precise across all thresholds. At the specific
operating point of τ=0.10, XGBoost is the stronger model. Holdout evaluation will
serve as the tiebreaker.

### Note on Monotone Constraints
Monotone constraints derived from Part 1 causal estimates were tested
(`number_inpatient` +1, `time_in_hospital` +1, `total_prior_encounters` +1,
`insulin` +1, `a1cresult` −1) but worsened generalization: validation AUC dropped
to 0.644 and the train-val gap widened to 0.341. With a 7% positive rate, the model
compensated for constrained splits by growing deeper trees. Constraints were removed.

## SHAP Feature Importance (Top 15, validation set)

| Feature | Mean \|SHAP\| | LR coefficient rank | Notes |
|---|---|---|---|
| `numeric__time_in_hospital` | 0.330 | #2 | Consistent across both models |
| `numeric__number_inpatient` | 0.182 | #1 | Consistent — AIPW RR ~2.0 in Part 1 |
| `numeric__total_prior_encounters` | 0.146 | #45 | Large jump — non-linear utilization signal |
| `numeric__num_lab_procedures` | 0.127 | #11 | Rises with non-linear model |
| `numeric__number_diagnoses` | 0.106 | #5 | Consistent |
| `numeric__age_midpoint` | 0.082 | #18 | Rises — non-linear age effects |
| `numeric__num_medications` | 0.072 | #32 | Rises substantially |
| `nominal__diag_1_group_Respiratory` | 0.069 | #3 | Consistent direction (protective) |
| `med_ord__insulin` | 0.046 | #39 | Consistent direction (positive) |
| `numeric__num_meds_active` | 0.043 | #41 | Notable rise — near-zero in LR |

**Key divergence from LR:** `total_prior_encounters` jumps from near-zero LR
coefficient (#45) to 3rd by SHAP — its signal is non-linear (high utilizers have
disproportionate risk beyond what a linear term captures). `num_medications` and
`num_meds_active` also rise, suggesting interaction effects with other utilization
variables. This validates keeping all 74 features for XGBoost rather than the
LR-selected 48.
