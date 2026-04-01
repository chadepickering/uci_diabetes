"""
train_tf.py — TensorFlow/Keras feedforward neural network

Architecture:
  Input(74) → Dense(128, ReLU) → BN → Dropout(0.3)
             → Dense(64,  ReLU) → BN → Dropout(0.3)
             → Dense(32,  ReLU) → BN → Dropout(0.2)
             → Dense(1, sigmoid)

Training:
  - Adam with cosine decay schedule, binary crossentropy loss
  - class_weight to handle ~13:1 imbalance
  - Batch size and learning rate tuned via Optuna (25 trials)
  - Max 200 epochs, early stopping patience=10 on val AUC

Calibration:
  Isotonic regression fit on out-of-fold training probabilities.
  Saved separately as tf_calibrator.joblib to avoid sklearn/joblib
  serialization issues with Keras model state.

Feature attribution:
  Integrated Gradients via tf.GradientTape (zero baseline).
  Mean absolute attribution per feature saved to tf_ig_importance.csv,
  directly comparable to mean |SHAP| from tree models.

Outputs saved to ml/data/:
  tf_model.keras           — trained Keras model (native format)
  tf_calibrator.joblib     — fitted isotonic regression calibrator
  tf_metrics.json          — train + validation metrics
  tf_ig_importance.csv     — mean |IG attribution| per feature, ranked

Usage:
    python ml/tensorflow/train_tf.py [--n-trials N]
"""

import argparse
import pathlib
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sklearn"))
from evaluate import evaluate, save_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT       = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR        = REPO_ROOT / "ml" / "data"
MODEL_PATH      = DATA_DIR / "tf_model.keras"
CALIBRATOR_PATH = DATA_DIR / "tf_calibrator.joblib"
METRICS_PATH    = DATA_DIR / "tf_metrics.json"
IG_PATH         = DATA_DIR / "tf_ig_importance.csv"

TARGET = "readmitted_30day"

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------
LAYER_WIDTHS  = [128, 64, 32]
DROPOUT_RATES = [0.3, 0.3, 0.2]
MAX_EPOCHS    = 200
ES_PATIENCE   = 10
IG_STEPS      = 50
RANDOM_SEED   = 26904

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_features,), name="features")
    x = inputs
    for width, dropout in zip(LAYER_WIDTHS, DROPOUT_RATES):
        x = tf.keras.layers.Dense(width, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="diabetes_readmission_mlp")


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------
def integrated_gradients(
    model: tf.keras.Model,
    X: np.ndarray,
    steps: int = IG_STEPS,
) -> np.ndarray:
    baseline  = np.zeros_like(X)
    alphas    = np.linspace(0, 1, steps + 1)
    grads_sum = np.zeros_like(X, dtype=np.float32)

    for alpha in alphas:
        interpolated = tf.constant(baseline + alpha * (X - baseline),
                                   dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = model(interpolated, training=False)
        grads_sum += tape.gradient(preds, interpolated).numpy()

    # Trapezoidal approximation
    avg_grads = (grads_sum - 0.5 * (grads_sum[0] + grads_sum[-1])) / steps
    return (X - baseline) * avg_grads


# ---------------------------------------------------------------------------
# Isotonic calibration
# Fits IsotonicRegression on out-of-fold training probabilities.
# Non-parametric — more flexible than Platt (sigmoid) scaling, better suited
# when the raw probability distribution is non-monotonic or heavy-tailed.
# ---------------------------------------------------------------------------
def fit_isotonic_calibrator(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    n_splits: int = 5,
) -> IsotonicRegression:
    cv        = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=RANDOM_SEED)
    oof_probs = np.zeros(len(y_train))

    for _, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        oof_probs[val_idx] = model.predict(
            X_train[val_idx], batch_size=batch_size, verbose=0
        ).ravel()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, y_train)
    return calibrator


def calibrated_proba(
    model: tf.keras.Model,
    calibrator: IsotonicRegression,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    raw = model.predict(X, batch_size=batch_size, verbose=0).ravel()
    return np.clip(calibrator.predict(raw), 0, 1)


# ---------------------------------------------------------------------------
# Optuna objective
# Tunes learning rate and batch size; all other config is fixed.
# ---------------------------------------------------------------------------
def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: dict,
    n_features: int,
):
    def objective(trial: optuna.Trial) -> float:
        lr         = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        # Cosine decay over a realistic training window (50 epochs).
        # Using MAX_EPOCHS causes the schedule to be nearly flat within the
        # early-stopping window (~20 epochs). alpha=0.01 sets an LR floor.
        steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
        decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=50 * steps_per_epoch,
            alpha=0.01,
        )

        model = build_model(n_features)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=decay),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc", curve="ROC")],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=ES_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        return max(history.history["val_auc"])

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_trials: int = 25) -> None:
    print("Loading preprocessed splits…")
    X_train_df, y_train = load_split("train")
    X_val_df,   y_val   = load_split("validation")
    print(f"  Train:      X={X_train_df.shape}, pos-rate={y_train.mean():.3f}")
    print(f"  Validation: X={X_val_df.shape}, pos-rate={y_val.mean():.3f}")

    feature_names = list(X_train_df.columns)
    X_train     = X_train_df.values.astype(np.float32)
    X_val       = X_val_df.values.astype(np.float32)
    y_train_arr = y_train.values
    y_val_arr   = y_val.values
    n_features  = X_train.shape[1]

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train_arr)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  class_weight: {class_weight}")

    # -------------------------------------------------------------------
    # Optuna: tune learning rate and batch size
    # -------------------------------------------------------------------
    print(f"\nOptuna search: {n_trials} trials (lr, batch_size)…")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        make_objective(X_train, y_train_arr, X_val, y_val_arr,
                       class_weight, n_features),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_lr         = best["learning_rate"]
    best_batch_size = best["batch_size"]
    print(f"\nBest params:")
    print(f"  learning_rate = {best_lr:.6f}")
    print(f"  batch_size    = {best_batch_size}")
    print(f"  Optuna val AUC = {study.best_value:.4f}")

    # -------------------------------------------------------------------
    # Final model: fit on full training set with best params
    # -------------------------------------------------------------------
    print("\nFitting final model on full training set…")
    steps_per_epoch = int(np.ceil(len(X_train) / best_batch_size))
    decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=best_lr,
        decay_steps=50 * steps_per_epoch,
        alpha=0.01,
    )

    model = build_model(n_features)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=decay),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train_arr,
        validation_data=(X_val, y_val_arr),
        epochs=MAX_EPOCHS,
        batch_size=best_batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch   = int(np.argmax(history.history["val_auc"]) + 1)
    best_val_auc = float(max(history.history["val_auc"]))
    print(f"\nBest epoch: {best_epoch}  |  Best val AUC: {best_val_auc:.4f}")

    # -------------------------------------------------------------------
    # Isotonic calibration
    # -------------------------------------------------------------------
    print("\nFitting isotonic calibration (5-fold OOF on training set)…")
    calibrator = fit_isotonic_calibrator(
        model, X_train, y_train_arr, best_batch_size
    )

    # -------------------------------------------------------------------
    # Integrated Gradients
    # -------------------------------------------------------------------
    print(f"\nComputing Integrated Gradients on validation set ({IG_STEPS} steps)…")
    ig_vals = integrated_gradients(model, X_val)
    ig_df = pd.DataFrame({
        "feature":     feature_names,
        "mean_abs_ig": np.abs(ig_vals).mean(axis=0),
    }).sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)

    print("\nTop 15 features by mean |IG|:")
    print(ig_df.head(15).to_string(index=False))

    # -------------------------------------------------------------------
    # Evaluate (calibrated probabilities)
    # -------------------------------------------------------------------
    train_probs = calibrated_proba(model, calibrator, X_train, best_batch_size)
    val_probs   = calibrated_proba(model, calibrator, X_val,   best_batch_size)

    metrics = [
        evaluate(y_train_arr, train_probs, "train"),
        evaluate(y_val_arr,   val_probs,   "validation"),
    ]

    for m in metrics:
        m["best_epoch"]    = best_epoch
        m["best_val_auc"]  = round(best_val_auc, 6)
        m["learning_rate"] = round(best_lr, 6)
        m["batch_size"]    = best_batch_size
        m["calibration"]   = "isotonic_oof_cv5"
        m["lr_schedule"]   = "cosine_decay"
        m["architecture"]  = f"MLP {LAYER_WIDTHS} relu+bn+dropout"

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    model.save(MODEL_PATH)
    print(f"\nKeras model saved    → {MODEL_PATH.relative_to(REPO_ROOT)}")

    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"Calibrator saved     → {CALIBRATOR_PATH.relative_to(REPO_ROOT)}")

    save_metrics(metrics, METRICS_PATH)

    ig_df.to_csv(IG_PATH, index=False)
    print(f"IG importance saved  → {IG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
