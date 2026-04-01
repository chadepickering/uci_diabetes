# Logistic Regression Baseline

## Reproduction

```bash
source venv_ml/bin/activate
python ml/sklearn/train_lr.py [--n-trials N]
```

Default: 50 Optuna trials. Requires preprocessed splits in `ml/data/` (run `preprocess.py` first).

## Outputs

| File | Description |
|---|---|
| `lr_model.joblib` | Fitted `CalibratedClassifierCV` wrapping `LogisticRegression` (ElasticNet, best params) |
| `lr_metrics.json` | Train + validation metrics including best hyperparameters |
| `lr_coefficients.csv` | All 74 features ranked by `abs(coefficient)` |

## Configuration
- Penalty: ElasticNet (`solver='saga'`, `l1_ratio` passed directly — sklearn ≥ 1.8 compatible)
- Class weighting: `balanced` (~13:1 imbalance corrected), followed by Platt scaling calibration
- Calibration: `CalibratedClassifierCV(method='sigmoid', cv=5)` fit on training set only
- Hyperparameter search: Optuna TPE, 50 trials, 5-fold stratified CV (scoring: AUC-ROC)
- SAGA convergence: `max_iter=5000`, `tol=1e-4`

## Best Hyperparameters

| Parameter | Value | Interpretation |
|---|---|---|
| `C` | 0.017534 | Strong regularization |
| `l1_ratio` | 0.6508 | Substantial L1 component — variable selection active |
| CV AUC-ROC | 0.6932 | Honest cross-validated estimate |

## Performance (calibrated probabilities)

| Split | AUC-ROC | AUC-PR | Brier | Flagged | Flagged pos rate | Precision@τ | Recall@τ | F1@τ |
|---|---|---|---|---|---|---|---|---|
| Train | 0.699 | 0.148 | 0.064 | 16.7% | 14.9% | 0.149 | 0.350 | 0.209 |
| Validation | 0.676 | 0.128 | 0.052 | 19.4% | 10.7% | 0.107 | 0.366 | 0.165 |

Train → validation AUC gap is modest (0.023). Brier scores are well-scaled after
Platt calibration — flagged rates are realistic (~15–19%) and close to the Part 1
experimental design expectation of ~15.7% at τ=0.10.

## Feature Selection
`l1_ratio=0.651` zeroed out 26 of 74 features, leaving **48 non-zero coefficients**.
Zeroed features include redundant utilization composites (`number_outpatient`,
several `number_emergency` variants), medication columns collinear with
`diabetes_med_Yes`, and low-signal diagnosis subcategories. The 48 retained
features are a principled starting set for tree-based models, though all 74 are
used for XGBoost and LightGBM given their native regularization mechanisms.

## Top Features by |Coefficient| (causal consistency check)

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

## sklearn Compatibility
`penalty='elasticnet'` was removed in sklearn ≥ 1.8. `train_lr.py` passes
`l1_ratio` directly with `solver='saga'`, which implies ElasticNet behavior
without the deprecated parameter.
