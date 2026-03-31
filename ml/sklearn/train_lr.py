"""
train_lr.py — ElasticNet logistic regression baseline

Tunes C (inverse regularization strength) and l1_ratio via Optuna using
stratified 5-fold cross-validation on the training set. The L1 component
provides variable selection; the L2 component handles multicollinearity.
The best model is re-fit on the full training set and evaluated on the
held-out validation set.

Non-zero coefficients after fitting give a principled feature selection
output to inform XGBoost and the neural network.

Outputs saved to ml/data/:
  lr_model.joblib      — fitted LogisticRegression object (best params)
  lr_metrics.json      — train + validation metrics
  lr_coefficients.csv  — feature name, coefficient, abs(coefficient)

Usage:
    python ml/sklearn/train_lr.py [--n-trials N]
"""

import argparse
import pathlib
import sys

import joblib
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from evaluate import evaluate, save_metrics

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR     = REPO_ROOT / "ml" / "data"
MODEL_PATH   = DATA_DIR / "lr_model.joblib"
METRICS_PATH = DATA_DIR / "lr_metrics.json"
COEF_PATH    = DATA_DIR / "lr_coefficients.csv"

TARGET = "readmitted_30day"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(X_train: np.ndarray, y_train: np.ndarray, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        C        = trial.suggest_float("C",        1e-4, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0,  1.0)

        model = LogisticRegression(
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=26904,
        )
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return scores.mean()

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_trials: int = 50) -> None:
    print("Loading preprocessed splits…")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("validation")
    print(f"  Train:      X={X_train.shape}, pos-rate={y_train.mean():.3f}")
    print(f"  Validation: X={X_val.shape}, pos-rate={y_val.mean():.3f}")

    feature_names = list(X_train.columns)

    # -------------------------------------------------------------------
    # Hyperparameter search via Optuna (CV on training set)
    # -------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=26904)

    print(f"\nOptuna search: {n_trials} trials, 5-fold CV AUC-ROC on training set…")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=26904),
    )
    study.optimize(
        make_objective(X_train.values, y_train.values, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nBest params:")
    print(f"  C        = {best['C']:.6f}")
    print(f"  l1_ratio = {best['l1_ratio']:.4f}  (0=ridge, 1=lasso)")
    print(f"  CV AUC   = {study.best_value:.4f}")

    # -------------------------------------------------------------------
    # Fit final model on full training set with best params
    # -------------------------------------------------------------------
    print("\nFitting final model on full training set…")
    base_model = LogisticRegression(
        solver="saga",
        C=best["C"],
        l1_ratio=best["l1_ratio"],
        class_weight="balanced",
        max_iter=5000,
            tol=1e-4,
        random_state=26904,
    )
    base_model.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # Platt calibration (sigmoid, 5-fold CV on training set)
    # Corrects inflated probabilities from class_weight='balanced'.
    # CalibratedClassifierCV with cv=5 fits on out-of-fold predictions
    # internally — validation set is never touched.
    # -------------------------------------------------------------------
    print("Calibrating probabilities (Platt scaling, 5-fold CV)…")
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # Feature coefficients (from the underlying base estimator)
    # -------------------------------------------------------------------
    coef = base_model.coef_[0]
    coef_df = pd.DataFrame({
        "feature":    feature_names,
        "coefficient": coef,
        "abs_coef":   np.abs(coef),
    }).sort_values("abs_coef", ascending=False)

    n_nonzero = (coef_df["coefficient"] != 0).sum()
    print(f"\nNon-zero coefficients: {n_nonzero} / {len(coef_df)}")
    print("\nTop 15 features by |coefficient|:")
    print(coef_df.head(15).to_string(index=False))

    # -------------------------------------------------------------------
    # Evaluate (calibrated probabilities)
    # -------------------------------------------------------------------
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs   = model.predict_proba(X_val)[:, 1]

    metrics = [
        evaluate(y_train.values, train_probs, "train"),
        evaluate(y_val.values,   val_probs,   "validation"),
    ]

    # Attach best params and calibration note to metrics for reference
    for m in metrics:
        m["best_C"]          = round(best["C"], 6)
        m["best_l1_ratio"]   = round(best["l1_ratio"], 6)
        m["calibration"]     = "platt_sigmoid_cv5"

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved      → {MODEL_PATH.relative_to(REPO_ROOT)}")

    save_metrics(metrics, METRICS_PATH)

    coef_df.to_csv(COEF_PATH, index=False)
    print(f"Coefficients saved → {COEF_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
