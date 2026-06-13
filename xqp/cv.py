"""Cross-validation harness for predictor hyperparameters.

Addresses 100-round R05 (Princeton): L2 strength should be picked by CV,
not hardcoded.
"""
from __future__ import annotations

import numpy as np

from .eval import roc_auc
from .predictor import ClosedFormXQP


def cv_select_l2(F: np.ndarray, y: np.ndarray,
                 candidates: tuple = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
                 k_folds: int = 5, seed: int = 0) -> tuple[float, dict]:
    """K-fold CV to pick L2; returns (best_l2, full_scores_dict)."""
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    N = F.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    fold_size = N // k_folds
    scores = {}
    for l2 in candidates:
        aucs = []
        for k in range(k_folds):
            val_idx = perm[k * fold_size:(k + 1) * fold_size]
            train_idx = np.concatenate([perm[:k * fold_size],
                                         perm[(k + 1) * fold_size:]])
            pred = ClosedFormXQP.from_fit(F[train_idx], y[train_idx], l2=l2)
            aucs.append(roc_auc(y[val_idx], pred.score(F[val_idx])))
        scores[l2] = float(np.mean(aucs))
    best_l2 = max(scores, key=scores.get)
    return best_l2, scores


def bootstrap_auc_ci(F: np.ndarray, y: np.ndarray, predictor: ClosedFormXQP,
                     n_bootstrap: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """Returns AUC + (1-alpha) confidence interval via bootstrap.

    Addresses 100-round R10 (NVIDIA KVPress): claims need CI."""
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, F.shape[0], size=F.shape[0])
        if y[idx].sum() == 0 or y[idx].sum() == y[idx].shape[0]:
            continue
        aucs.append(roc_auc(y[idx], predictor.score(F[idx])))
    aucs = np.asarray(aucs, dtype=np.float32)
    return dict(
        auc_mean=float(aucs.mean()),
        auc_std=float(aucs.std()),
        ci_lo=float(np.percentile(aucs, 100 * alpha / 2)),
        ci_hi=float(np.percentile(aucs, 100 * (1 - alpha / 2))),
        n=int(aucs.shape[0]),
    )
