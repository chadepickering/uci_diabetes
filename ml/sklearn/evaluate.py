"""
evaluate.py — Shared evaluation utilities for all sklearn models.

Metrics reported:
  - AUC-ROC
  - AUC-PR (average precision)
  - Brier score
  - At threshold τ: precision, recall, F1, flagged rate, flagged positive rate
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TAU = 0.10  # clinical risk threshold from Part 1 experimental design


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    tau: float = TAU,
) -> dict:
    """
    Compute and print evaluation metrics. Returns a flat dict for serialization.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_prob : predicted probabilities for the positive class
    split_name : label for printed output (e.g. "train", "validation")
    tau : classification threshold for threshold-dependent metrics
    """
    y_pred = (y_prob >= tau).astype(int)
    flagged = y_pred.sum()
    n = len(y_true)

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)

    # Threshold-dependent (guard against all-negative predictions)
    if flagged == 0:
        prec = rec = f1 = 0.0
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

    flagged_rate     = flagged / n
    flagged_pos_rate = float(y_true[y_pred == 1].mean()) if flagged > 0 else 0.0

    print(f"\n{'─' * 52}")
    print(f"  {split_name.upper()}  (n={n:,}, τ={tau})")
    print(f"{'─' * 52}")
    print(f"  AUC-ROC          : {auc_roc:.4f}")
    print(f"  AUC-PR           : {auc_pr:.4f}")
    print(f"  Brier score      : {brier:.4f}")
    print(f"  Flagged          : {flagged:,} / {n:,}  ({flagged_rate:.1%})")
    print(f"  Flagged pos rate : {flagged_pos_rate:.3f}  (baseline: {y_true.mean():.3f})")
    print(f"  Precision @ τ    : {prec:.4f}")
    print(f"  Recall @ τ       : {rec:.4f}")
    print(f"  F1 @ τ           : {f1:.4f}")

    return {
        "split":            split_name,
        "n":                n,
        "tau":              tau,
        "auc_roc":          round(auc_roc, 6),
        "auc_pr":           round(auc_pr, 6),
        "brier":            round(brier, 6),
        "flagged_n":        int(flagged),
        "flagged_rate":     round(flagged_rate, 6),
        "flagged_pos_rate": round(flagged_pos_rate, 6),
        "precision_tau":    round(prec, 6),
        "recall_tau":       round(rec, 6),
        "f1_tau":           round(f1, 6),
    }


def save_metrics(metrics_list: list[dict], path: pathlib.Path) -> None:
    """Write a list of per-split metric dicts to JSON."""
    path.write_text(json.dumps(metrics_list, indent=2))
    print(f"\nMetrics saved → {path}")
