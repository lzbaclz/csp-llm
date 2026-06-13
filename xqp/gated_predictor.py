"""Regime-gated selective saliency predictor — the design that makes every view
earn its place instead of leaning only on attention-magnitude history.

Motivation (measured). The within- and cross-layer attention-magnitude views are
*lagging* indicators: strong for blocks with history, blind for cold-start /
long-horizon blocks (no realized attention to average). The query--key view is a
*leading* indicator that should help exactly where magnitude is blind. Computing
query for every block is wasteful (and, mean-pooled, near-useless on average);
we instead compute it SELECTIVELY for the blocks the cheap magnitude model is
unsure about, under a per-step compute budget.

    Stage 1 (cheap, always):     p1 = sigma(w . [within, cross] + b)        # no query
    Defer:                       route the least-confident `budget` fraction to stage 2
    Stage 2 (expensive, gated):  p2 = sigma(w'. [within, cross, query] + b')
    Final:                       p  = p2 on deferred blocks, p1 elsewhere

Cost = base (free; magnitude is computed for attention anyway) + budget * query_cost.
Composes with :class:`xqp.conformal.AdaptiveConformalSaliency` for a
distribution-free coverage guarantee on the missed-saliency rate.

Each parameter earns its place: within/cross own the confident regime, query owns
the deferred regime — quantified per-regime in experiments/run_gated_eval.py.
The query column index is configurable, so the SAME class consumes the cheap
mean-pooled query (column 2 of the current traces) or a faithful per-token /
per-head-max Quest signal (an extra column) with no code change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .predictor import ClosedFormXQP


def _gather4(F: np.ndarray, cols) -> np.ndarray:
    """Gather up to 4 source columns of F into the fixed (N,4) layout that
    ClosedFormXQP expects; unused slots are zero (→ ~0 weight). Slot order
    follows `cols`, so e.g. cols=(0,1,4) puts within, cross, faithful-query into
    slots 0,1,2."""
    F = np.asarray(F, np.float32)
    out = np.zeros((F.shape[0], 4), np.float32)
    for slot, c in enumerate(list(cols)[:4]):
        out[:, slot] = F[:, c]
    return out


@dataclass
class SelectiveCascadeXQP:
    """Two-stage selective cascade with budget-controlled deferral.

    base:        ClosedFormXQP on the cheap (magnitude) views.
    expert:      ClosedFormXQP on the cheap views + the (expensive) query view.
    base_cols:   source columns the base uses (default within=0, cross=1).
    expert_cols: source columns the expert uses (adds the query column).
    query_col:   the source column whose computation the budget meters.
    """
    base: ClosedFormXQP
    expert: ClosedFormXQP
    base_cols: tuple = (0, 1)
    expert_cols: tuple = (0, 1, 2)
    query_col: int = 2

    @classmethod
    def from_fit(cls, F, y, *, base_cols=(0, 1), expert_cols=(0, 1, 2),
                 query_col=2, l2: float = 1e-3) -> "SelectiveCascadeXQP":
        F = np.asarray(F, np.float32); y = np.asarray(y, np.float32).reshape(-1)
        base = ClosedFormXQP.from_fit(_gather4(F, base_cols), y, l2=l2)
        expert = ClosedFormXQP.from_fit(_gather4(F, expert_cols), y, l2=l2)
        return cls(base=base, expert=expert, base_cols=tuple(base_cols),
                   expert_cols=tuple(expert_cols), query_col=int(query_col))

    def _defer_mask(self, F, p1, budget, rule="confidence"):
        """Which blocks get the expensive stage-2 (query) evaluation."""
        n = F.shape[0]
        if budget >= 1.0:
            return np.ones(n, bool)
        if budget <= 0.0:
            return np.zeros(n, bool)
        k = max(1, int(np.ceil(budget * n)))
        if rule == "cold":
            # explicit regime: defer where the magnitude view is weakest (the
            # cold-start / magnitude-blind regime) — within-layer EMA is col 0.
            key = F[:, 0]
        else:                                     # confidence: defer least-sure
            key = np.abs(2.0 * p1 - 1.0)
        idx = np.argpartition(key, kth=min(k - 1, n - 1))[:k]
        m = np.zeros(n, bool); m[idx] = True
        return m

    def predict_with_cost(self, F, budget=0.2, rule="confidence"):
        """Return (scores, defer_mask, query_frac). The expensive query is only
        *computed* for deferred blocks, so query_frac (== budget up to rounding)
        is the metered per-step compute the design adds over the cheap base."""
        F = np.asarray(F, np.float32)
        p1 = self.base.score(_gather4(F, self.base_cols))
        defer = self._defer_mask(F, p1, budget, rule=rule)
        p = p1.copy()
        if defer.any():
            p[defer] = self.expert.score(_gather4(F[defer], self.expert_cols))
        return p, defer, float(defer.mean())

    def score(self, F, budget=0.2, rule="confidence"):
        return self.predict_with_cost(F, budget=budget, rule=rule)[0]
