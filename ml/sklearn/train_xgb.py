"""
train_xgb.py — Step 2b: XGBoost primary classifier.

Tunes hyperparameters via Optuna using the validation set as the eval
metric with early stopping. This is intentional: XGBoost's early stopping
requires a held-out eval set, and the holdout parquet remains untouched
throughout. The best hyperparameters are used to fit a final model on the
full training set, calibrated via Platt scaling (same pattern as train_lr.py),
then evaluated on validation.

SHAP values are computed via TreeExplainer on the base (uncalibrated)
XGBoost estimator and saved alongside LR coefficients for direct comparison.

Outputs saved to ml/data/:
  xgb_model.joblib       — fitted CalibratedClassifierCV wrapping XGBClassifier
  xgb_metrics.json       — train + validation metrics
  xgb_shap_importance.csv — mean |SHAP| per feature, ranked

Usage:
    python ml/sklearn/train_xgb.py [--n-trials N]
"""

import argparse
import pathlib
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from xgboost import XGBClassifier

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from evaluate import evaluate, save_metrics

from sklearn.calibration import CalibratedClassifierCV

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT     = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR      = REPO_ROOT / "ml" / "data"
MODEL_PATH    = DATA_DIR / "xgb_model.joblib"
METRICS_PATH  = DATA_DIR / "xgb_metrics.json"
SHAP_PATH     = DATA_DIR / "xgb_shap_importance.csv"

TARGET = "readmitted_30day"

# Class imbalance ratio (n_negative / n_positive, computed from training set)
# Fixed rather than tuned — this is a dataset property, not a hyperparameter.
SCALE_POS_WEIGHT = 45236 / 3470  # ≈ 13.03


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]


# ---------------------------------------------------------------------------
# Optuna objective
# Tunes against validation AUC-ROC with XGBoost early stopping.
# ---------------------------------------------------------------------------
def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=26904,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, val_probs)

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_trials: int = 100) -> None:
    print("Loading preprocessed splits…")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("validation")
    print(f"  Train:      X={X_train.shape}, pos-rate={y_train.mean():.3f}")
    print(f"  Validation: X={X_val.shape}, pos-rate={y_val.mean():.3f}")
    print(f"  scale_pos_weight = {SCALE_POS_WEIGHT:.2f}")

    # XGBoost forbids '[', ']', '<' in feature names — sanitize in place.
    def sanitize(name: str) -> str:
        return name.replace("[", "(").replace("]", ")").replace("<", "lt")

    feature_names = [sanitize(c) for c in X_train.columns]
    X_train.columns = feature_names
    X_val.columns   = feature_names

    # -------------------------------------------------------------------
    # Hyperparameter search via Optuna (validation set + early stopping)
    # -------------------------------------------------------------------
    print(f"\nOptuna search: {n_trials} trials, AUC-ROC on validation set…")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=26904),
    )
    study.optimize(
        make_objective(
            X_train.values, y_train.values,
            X_val.values,   y_val.values,
            feature_names,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nBest params:")
    for k, v in best.items():
        print(f"  {k:<22} = {v}")
    print(f"  Validation AUC    = {study.best_value:.4f}")

    # -------------------------------------------------------------------
    # Fit final model on full training set with best params
    # No early stopping here — use the best n_estimators from search.
    # -------------------------------------------------------------------
    print("\nFitting final model on full training set…")
    base_model = XGBClassifier(
        **best,
        scale_pos_weight=SCALE_POS_WEIGHT,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=26904,
        n_jobs=-1,
        verbosity=0,
        feature_names=feature_names,
    )
    base_model.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # Platt calibration (sigmoid, 5-fold CV on training set)
    # Consistent with LR approach; corrects score distribution for τ=0.10.
    # XGBoost with scale_pos_weight is less severely miscalibrated than
    # balanced LR, but calibration is applied for clinical consistency.
    # -------------------------------------------------------------------
    print("Calibrating probabilities (Platt scaling, 5-fold CV)…")
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # SHAP feature importance (TreeExplainer on base model)
    # -------------------------------------------------------------------
    print("\nComputing SHAP values on validation set…")
    explainer   = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_val)

    shap_df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\nTop 15 features by mean |SHAP|:")
    print(shap_df.head(15).to_string(index=False))

    # -------------------------------------------------------------------
    # Evaluate (calibrated probabilities)
    # -------------------------------------------------------------------
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs   = model.predict_proba(X_val)[:, 1]

    metrics = [
        evaluate(y_train.values, train_probs, "train"),
        evaluate(y_val.values,   val_probs,   "validation"),
    ]

    for m in metrics:
        m["best_params"]   = {k: round(v, 6) if isinstance(v, float) else v
                              for k, v in best.items()}
        m["calibration"]   = "platt_sigmoid_cv5"
        m["scale_pos_weight"] = round(SCALE_POS_WEIGHT, 4)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved           → {MODEL_PATH.relative_to(REPO_ROOT)}")

    save_metrics(metrics, METRICS_PATH)

    shap_df.to_csv(SHAP_PATH, index=False)
    print(f"SHAP importance saved → {SHAP_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=75)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
