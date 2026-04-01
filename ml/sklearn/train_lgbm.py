"""
train_lgbm.py — Step 2c: LightGBM classifier.

LightGBM uses leaf-wise tree growth rather than XGBoost's level-wise growth.
The primary regularization levers are `num_leaves` (controls tree complexity
directly) and `min_child_samples` (absolute sample count per leaf — more
interpretable than XGBoost's hessian-sum `min_child_weight` for sparse classes).

Class imbalance is handled via `class_weight='balanced'`, followed by Platt
scaling calibration — consistent with LR and XGBoost.

Hyperparameters are tuned via Optuna (100 trials) using the validation set
with LightGBM early stopping. Holdout is never touched.

SHAP values are computed via TreeExplainer on the base (uncalibrated) estimator.

Outputs saved to ml/data/:
  lgbm_model.joblib       — fitted CalibratedClassifierCV wrapping LGBMClassifier
  lgbm_metrics.json       — train + validation metrics
  lgbm_shap_importance.csv — mean |SHAP| per feature on validation set, ranked

Usage:
    python ml/sklearn/train_lgbm.py [--n-trials N]
"""

import argparse
import pathlib
import sys

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from evaluate import evaluate, save_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR     = REPO_ROOT / "ml" / "data"
MODEL_PATH   = DATA_DIR / "lgbm_model.joblib"
METRICS_PATH = DATA_DIR / "lgbm_metrics.json"
SHAP_PATH    = DATA_DIR / "lgbm_shap_importance.csv"

TARGET = "readmitted_30day"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]


# ---------------------------------------------------------------------------
# Feature name sanitization
# LightGBM also forbids '[', ']', '<' in feature names.
# ---------------------------------------------------------------------------
def sanitize(name: str) -> str:
    return name.replace("[", "(").replace("]", ")").replace("<", "lt")


# ---------------------------------------------------------------------------
# Optuna objective
# Tunes against validation AUC-ROC with LightGBM early stopping.
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
            "n_estimators":      trial.suggest_int("n_estimators", 150, 600),
            # num_leaves is the primary complexity control in LightGBM.
            # Rule of thumb: num_leaves < 2^max_depth. Range covers shallow
            # (20) to moderately complex (150) trees.
            "num_leaves":        trial.suggest_int("num_leaves", 15, 60),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # subsample / bagging — requires bagging_freq > 0 to activate.
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # min_child_samples: absolute sample count per leaf.
            # With ~3,470 positives in training, a floor of 20-200 prevents
            # leaves formed from just a handful of minority-class examples.
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        model = LGBMClassifier(
            **params,
            class_weight="balanced",
            objective="binary",
            metric="auc",
            random_state=26904,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
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
    # Fit final model on full training set with best params.
    # No early stopping — use n_estimators from search directly.
    # -------------------------------------------------------------------
    print("\nFitting final model on full training set…")
    base_model = LGBMClassifier(
        **best,
        class_weight="balanced",
        objective="binary",
        metric="auc",
        random_state=26904,
        n_jobs=-1,
        verbosity=-1,
        feature_name=feature_names,
    )
    base_model.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # Platt calibration (sigmoid, 5-fold CV on training set)
    # Consistent with LR and XGBoost approaches.
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

    # LightGBM TreeExplainer may return a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

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
        m["best_params"]  = {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in best.items()}
        m["calibration"]  = "platt_sigmoid_cv5"

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
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
