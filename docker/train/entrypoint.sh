#!/bin/bash
# docker/train/entrypoint.sh
# Routes to the appropriate training script.
#
# Usage:
#   docker run uci-diabetes-train preprocess [--dry-run]
#   docker run uci-diabetes-train train-xgb  [--n-trials N]
#   docker run uci-diabetes-train train-lr
#   docker run uci-diabetes-train train-lgbm
#   docker run uci-diabetes-train train-tf
#   docker run uci-diabetes-train all        [--dry-run]   <- default
set -e

COMMAND="${1:-all}"
shift || true   # allow empty remaining args

case "$COMMAND" in
  preprocess)
    exec python ml/sklearn/preprocess.py "$@"
    ;;
  train-xgb)
    exec python ml/sklearn/train_xgb.py "$@"
    ;;
  train-lr)
    exec python ml/sklearn/train_lr.py "$@"
    ;;
  train-lgbm)
    exec python ml/sklearn/train_lgbm.py "$@"
    ;;
  train-tf)
    exec python ml/tensorflow/train_tf.py "$@"
    ;;
  all)
    # Full pipeline: preprocess → XGBoost (selected production model)
    echo "=== [1/2] Preprocessing ==="
    python ml/sklearn/preprocess.py "$@"
    echo "=== [2/2] XGBoost training ==="
    python ml/sklearn/train_xgb.py
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Valid: preprocess, train-xgb, train-lr, train-lgbm, train-tf, all"
    exit 1
    ;;
esac
