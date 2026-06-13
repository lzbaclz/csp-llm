"""Data-mining evaluation metrics for the ICDM framing.

The systems framing cared about TPOT/WCET; the DM framing cares about ranking
quality on an *imbalanced* stream and whether the predicted probabilities are
*trustworthy*. This module adds the metrics ICDM reviewers expect and the
existing `eval.py` lacks: average precision (AUPRC), expected calibration error
+ reliability curve, precision@k, and the Brier score.

All functions are NumPy-only and CPU-runnable; pair them with `eval.roc_auc`.
"""
from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision–recall curve (AUPRC), the right headline for a
    ~10%-positive stream. AP = sum_k (R_k - R_{k-1}) * P_k over the ranked list.
    Ties are broken arbitrarily (stable sort); returns NaN if no positives.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = float(y_true.sum())
    if n_pos == 0 or y_true.shape[0] == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> float:
    """Fraction of the predicted top-k that is truly positive (k = k_frac·N)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * n)))
    idx = np.argpartition(-y_score, kth=min(k - 1, n - 1))[:k]
    return float(np.asarray(y_true)[idx].sum()) / k


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> float:
    """Fraction of true positives captured in the predicted top-k."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n = y_true.shape[0]
    n_pos = float((y_true > 0.5).sum())
    if n == 0 or n_pos == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * n)))
    idx = np.argpartition(-y_score, kth=min(k - 1, n - 1))[:k]
    return float(np.asarray(y_true)[idx].sum()) / n_pos


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions (lower = better)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.shape[0] == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Per-bin (confidence, accuracy, count) for a reliability diagram."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if b > 0 else (y_prob >= lo) & (y_prob <= hi)
        c = int(mask.sum())
        cnt.append(c)
        conf.append(float(y_prob[mask].mean()) if c else float("nan"))
        acc.append(float(y_true[mask].mean()) if c else float("nan"))
    return dict(bin_edges=edges.tolist(), confidence=conf, accuracy=acc, count=cnt)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) |conf_b - acc_b|. 0 = perfectly calibrated.

    A proper-scoring-rule fit (the closed-form logistic regression) should be
    well-calibrated out of the box — a genuine advantage over threshold
    heuristics that the DM framing rewards and the systems framing ignored.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    rc = reliability_curve(y_true, y_prob, n_bins=n_bins)
    ece = 0.0
    for conf, acc, c in zip(rc["confidence"], rc["accuracy"], rc["count"]):
        if c == 0:
            continue
        ece += (c / n) * abs(conf - acc)
    return float(ece)
