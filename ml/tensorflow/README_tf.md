# TensorFlow/Keras Neural Network

## Reproduction

```bash
source venv_ml/bin/activate
python ml/tensorflow/train_tf.py [--n-trials N]
```

Default: 25 Optuna trials. Requires preprocessed splits in `ml/data/` (run `preprocess.py` first).

## Outputs

| File | Description |
|---|---|
| `tf_model.keras` | Trained Keras model (native format, TF Serving compatible) |
| `tf_calibrator.joblib` | Fitted isotonic regression calibrator |
| `tf_metrics.json` | Train + validation metrics |
| `tf_ig_importance.csv` | Mean \|IG attribution\| per feature on validation set, ranked |

Note: `tf_model.keras` and `tf_calibrator.joblib` are separate by design — Keras
model state cannot be pickled by `joblib`. Load both for inference:
```python
import numpy as np, joblib, tensorflow as tf
model      = tf.keras.models.load_model("ml/data/tf_model.keras")
calibrator = joblib.load("ml/data/tf_calibrator.joblib")
probs      = np.clip(calibrator.predict(model.predict(X).ravel()), 0, 1)
```

## Architecture

```
Input(74)
  → Dense(128, ReLU) → BatchNormalization → Dropout(0.3)
  → Dense(64,  ReLU) → BatchNormalization → Dropout(0.3)
  → Dense(32,  ReLU) → BatchNormalization → Dropout(0.2)
  → Dense(1, sigmoid)
```

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | Adam + cosine decay | LR decays over 50-epoch window, floor `alpha=0.01` |
| Loss | Binary crossentropy | |
| Max epochs | 200 | |
| Early stopping | patience=10, monitor=`val_auc` | `restore_best_weights=True` |
| Class imbalance | `class_weight='balanced'` | ~13:1 ratio |
| Random seed | 26904 | |

### Tuned via Optuna (25 trials)

| Parameter | Search range | Best value |
|---|---|---|
| `learning_rate` | [1e-4, 1e-3] log | 0.000982 |
| `batch_size` | {256, 512, 1024} | 512 |
| Optuna val AUC | — | 0.6879 |

Early stopping triggered at **epoch 6** (final model val AUC: 0.6888). Fast
convergence is consistent with the high initial LR (≈0.001) decaying quickly
under the cosine schedule — the model finds the optimum early and stabilises.

### Cosine Decay Design
`decay_steps = 50 × steps_per_epoch` — calibrated to the expected early-stopping
window (~20 epochs with patience=10) so the LR meaningfully traverses the decay
curve rather than remaining flat. `alpha=0.01` prevents decay to zero if training
runs longer. Using `MAX_EPOCHS × steps_per_epoch` caused near-flat LR within the
early-stopping window, degrading performance (val AUC 0.675 → 0.689 after fix).

## Calibration

Isotonic regression fit on 5-fold OOF training probabilities — more flexible than
Platt (sigmoid) scaling for non-monotonic or heavy-tailed probability distributions.
Implemented manually to avoid `joblib` serialization issues with Keras model state:

1. 5-fold stratified CV generates out-of-fold sigmoid probabilities on the training set
2. `IsotonicRegression(out_of_bounds='clip')` fits a non-parametric calibration map
3. The calibrator is saved separately as `tf_calibrator.joblib`

## Feature Attribution — Integrated Gradients

`tf.GradientTape` computes gradients along a straight-line path from a zero
baseline to each input (Riemann sum, 50 steps). Zero baseline is appropriate for
standardized features (mean=0 after `StandardScaler`) — it represents the
"average patient".

Output: mean absolute attribution per feature across the validation set, directly
comparable to mean |SHAP| from tree models.

**Note on IG vs SHAP magnitudes:** IG attributions reflect gradient × input, not
marginal probability contributions. Relative *rankings* are comparable; absolute
values are not.

## Performance (calibrated probabilities)

| Split | AUC-ROC | AUC-PR | Brier | Flagged | Flagged pos rate | Precision@τ | Recall@τ | F1@τ |
|---|---|---|---|---|---|---|---|---|
| Train | 0.721 | 0.158 | 0.063 | 27.3% | 14.4% | 0.144 | 0.551 | 0.228 |
| Validation | 0.689 | 0.128 | 0.052 | 31.0% | 10.4% | 0.104 | 0.570 | 0.176 |

Train-val AUC gap: **0.032** — second only to LR (0.023), confirming BatchNorm
and Dropout generalise effectively on this dataset.

### Four-Model Comparison (validation set)

| Metric | LR | XGBoost | LightGBM | TF/Keras |
|---|---|---|---|---|
| AUC-ROC | 0.676 | 0.687 | 0.685 | **0.689** |
| AUC-PR | **0.128** | 0.113 | 0.125 | **0.128** |
| Brier | 0.052 | 0.052 | 0.052 | **0.052** |
| Recall@τ | 0.366 | 0.512 | 0.501 | **0.570** |
| Precision@τ | **0.107** | **0.111** | 0.106 | 0.104 |
| Train-val gap | **0.023** | 0.099 | 0.141 | 0.032 |
| Flagged | 19.4% | 26.1% | 26.7% | 31.0% |

The TF model leads on AUC-ROC (0.689), ties LR on AUC-PR (0.128), and has the
best recall at τ=0.10 (0.570). The higher flagged rate (31.0%) reflects the NN
being more liberal at this threshold — clinically appropriate for a recall-oriented
screening tool where missing true positives is costly.

XGBoost leads on precision (0.111) and flags fewer patients (26.1%), making it the
more conservative option. Holdout evaluation is the tiebreaker.

### Tuning History

| Version | Config | Val AUC-ROC | Train-val gap | Notes |
|---|---|---|---|---|
| v1 | Fixed lr=0.001, batch=512, Platt | 0.686 | 0.069 | Baseline |
| v2 | Optuna, Platt, broken cosine decay | 0.675 | 0.048 | Decay too slow; Optuna/final gap |
| v3 | Optuna, isotonic, fixed cosine decay | **0.689** | **0.032** | Final |

## Feature Attribution (Top 15, validation set)

| Feature | Mean \|IG\| | XGBoost rank | LR rank | Notes |
|---|---|---|---|---|
| `numeric__time_in_hospital` | 0.0302 | #1 | #2 | Consistent across all models |
| `med_ord__insulin` | 0.0277 | #9 | #39 | Rises to #2 — strongest IG signal after time |
| `nominal__race_Caucasian` | 0.0192 | — | #26 | Persistently high in NN; proxy for demographics |
| `numeric__number_diagnoses` | 0.0188 | #5 | #5 | Consistent |
| `numeric__num_procedures` | 0.0169 | #12 | #40 | Rises in NN — non-linear procedure interactions |
| `nominal__diabetes_med_Yes` | 0.0168 | #14 | #4 | Lower than v1 IG but still elevated vs trees |
| `numeric__num_lab_procedures` | 0.0157 | #4 | #11 | Consistent rise in non-linear models |
| `numeric__number_inpatient` | 0.0133 | #2 | #1 | Lower IG rank — likely sigmoid saturation |
| `a1c_ord__a1cresult` | 0.0162 | #13 | #37 | Rises in NN — protective signal captured |

**Key divergence:** `number_inpatient` drops to #9 by IG despite being #1–2 in
tree models. Sigmoid saturation is the likely cause — for high-utilizer patients
the output is already near 1.0, producing near-zero gradients. IG understates
importance for saturated features; this is a known limitation.

`insulin` rises to #2 — consistent with the Part 1 causal estimate (PSM OR ~1.17)
and may reflect the NN capturing non-linear dose-change interactions that tree
splits encode differently.
