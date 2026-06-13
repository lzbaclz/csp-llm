"""Coverage-driven KV budgeter — the inverse of every prior method.

Prior KV-cache methods consume a budget (a retention ratio) and report the
resulting quality. This module solves the inverse: given a target missed-saliency
rate alpha, it derives a per-layer threshold via split-conformal calibration of a
calibrated scorer, so the retained hot set {b : p_b >= tau_layer} keeps at least
(1 - alpha) of the truly-salient blocks. The BUDGET (retention ratio) then falls
out of the guarantee, and adapts per layer automatically — recovering a
PyramidKV-style allocation from the risk target rather than hand-setting it.

Split-conformal validity. With a held-out calibration set, set tau as the
finite-sample alpha-quantile of the calibration salient blocks' scores; on
exchangeable test data the realized miss-rate is <= alpha in expectation. Decode
drift is mild (measured), so split conformal is well-posed; an adaptive variant
(xqp.conformal) absorbs residual drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gated_predictor import _gather4


def _conformal_tau(scores_salient: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal threshold: the largest tau such that at most
    an alpha fraction of salient calibration scores fall below it (so retaining
    {p >= tau} misses <= alpha of salient blocks). Uses the (n+1) correction."""
    s = np.sort(np.asarray(scores_salient, np.float64))
    n = s.shape[0]
    if n == 0:
        return 0.0
    k = int(np.floor(alpha * (n + 1)))      # index of the conformal quantile
    k = min(max(k, 0), n - 1)
    return float(s[k])


@dataclass
class CoverageDrivenBudgeter:
    """Turn a target miss-rate alpha into per-layer thresholds (hence budgets)."""
    scorer: object                  # exposes .score(F4) -> prob in [0,1]
    cols: tuple = (0, 1)            # within+cross (the calibrated minimal model)
    alpha: float = 0.10
    tau_by_layer: dict = field(default_factory=dict)
    tau_global: float = 0.5

    @classmethod
    def calibrate(cls, scorer, F, y, layer_ids, *, cols=(0, 1), alpha=0.10,
                  min_per_layer=64) -> "CoverageDrivenBudgeter":
        F = np.asarray(F, np.float32); y = np.asarray(y).reshape(-1)
        layer_ids = np.asarray(layer_ids).reshape(-1)
        p = np.asarray(scorer.score(_gather4(F, cols)))
        sal = y > 0.5
        tau = {}
        for l in np.unique(layer_ids):
            m = sal & (layer_ids == l)
            if m.sum() >= min_per_layer:
                tau[int(l)] = _conformal_tau(p[m], alpha)
        tau_g = _conformal_tau(p[sal], alpha)
        return cls(scorer=scorer, cols=tuple(cols), alpha=float(alpha),
                   tau_by_layer=tau, tau_global=float(tau_g))

    def _thresholds(self, layer_ids):
        return np.array([self.tau_by_layer.get(int(l), self.tau_global)
                         for l in layer_ids], np.float32)

    def norm_curve(self):
        """Per-layer thresholds keyed by NORMALIZED layer position in [0,1], so the
        guarantee transfers to a model with a different layer count (the deployment
        sim normalizes layer id the same way). Returns sorted [(pos, tau), ...]."""
        if not self.tau_by_layer:
            return [(0.0, self.tau_global), (1.0, self.tau_global)]
        lid = sorted(self.tau_by_layer)
        denom = max(1, lid[-1])
        return [(l / denom, float(self.tau_by_layer[l])) for l in lid]

    def save(self, path):
        """Persist the risk target + per-layer thresholds (raw + normalized) so the
        SEER GuardKV policy can reload the guarantee. JSON, numpy-free."""
        import json
        from pathlib import Path
        obj = dict(alpha=float(self.alpha), cols=list(self.cols),
                   tau_global=float(self.tau_global),
                   tau_by_layer={str(k): float(v) for k, v in self.tau_by_layer.items()},
                   norm_curve=[[float(p), float(t)] for p, t in self.norm_curve()])
        Path(path).write_text(json.dumps(obj, indent=2))
        return path

    def keep_mask(self, F, layer_ids, per_layer=True):
        p = np.asarray(self.scorer.score(_gather4(np.asarray(F, np.float32), self.cols)))
        thr = self._thresholds(layer_ids) if per_layer else self.tau_global
        return p >= thr, p

    def evaluate(self, F, y, layer_ids, per_layer=True) -> dict:
        """Realized miss-rate + emergent budget, overall and per layer."""
        y = np.asarray(y).reshape(-1); layer_ids = np.asarray(layer_ids).reshape(-1)
        keep, _ = self.keep_mask(F, layer_ids, per_layer=per_layer)
        sal = y > 0.5
        miss = float((sal & ~keep).sum() / max(1, sal.sum()))
        budget = float(keep.mean())
        per = {}
        for l in np.unique(layer_ids):
            m = layer_ids == l
            ms = sal & m
            per[int(l)] = dict(
                miss=float((ms & ~keep).sum() / max(1, ms.sum())),
                budget=float(keep[m].mean()))
        miss_spread = float(np.std([v["miss"] for v in per.values()]))
        return dict(target_alpha=self.alpha, realized_miss=miss, emergent_budget=budget,
                    per_layer_miss_std=miss_spread, per_layer=per)


def fixed_ratio_select(scores: np.ndarray, budget: float) -> np.ndarray:
    """Baseline: keep the top-`budget` fraction by score (global), the way Ada-KV /
    SnapKV consume a ratio. Used to contrast against coverage-driven allocation."""
    n = scores.shape[0]
    k = max(1, int(round(budget * n)))
    m = np.zeros(n, bool)
    m[np.argpartition(-scores, kth=min(k - 1, n - 1))[:k]] = True
    return m
