"""GuardKV — a query-free, conformally-calibrated, cross-layer KV controller.

The design that the SOTA sweep + our measurements point to (see
experiments/DESIGN_v2_conformal_crosslayer.md). It composes the validated pieces
into one controller:

  P1  calibrated minimal scorer   p = sigma(w.[within, cross] + b)   (3 params)
  P2  coverage-driven budget      keep {b : p_b >= tau_layer};  tau set so the
                                  missed-saliency rate <= alpha (budget = output)
  P2' adaptive conformal          tau_layer <- clip(tau + gamma(alpha - err))     (drift)
  P3  cross-layer prefetch hint   the blocks layer L will likely want, from
                                  layer L-1's hot set (query-free => legal)

It is query-free by design (query is redundant + computing it would block
prefetch), never permanently evicts (returns a HOT set; the rest are demoted),
and turns a risk target alpha into the budget rather than consuming a ratio.

This is backend-agnostic (NumPy); it maps onto SEER's ``KVPolicy.select_to_keep``
(see SEER/seer/policy for the adapter). Calibrate offline with
``xqp.budgeter.CoverageDrivenBudgeter``; pass its per-layer thresholds here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gated_predictor import _gather4
from .features import topk_indicator


@dataclass
class GuardKV:
    scorer: object                       # calibrated within+cross scorer (.score(F4))
    tau_by_layer: dict                   # layer -> conformal threshold (from budgeter)
    cols: tuple = (0, 1)
    alpha: float = 0.10
    gamma: float = 0.0                   # >0 enables online adaptive-conformal
    sink: int = 4
    window: int = 4
    tau_global: float = 0.5
    _errs: list = field(default_factory=list)
    _prev_hot: set = field(default_factory=set)   # cross-layer state (prev layer hot set)

    # ---- scoring + budgeted, guaranteed selection -------------------------
    def score(self, F):
        return np.asarray(self.scorer.score(_gather4(np.asarray(F, np.float32), self.cols)))

    def select(self, F, layer, *, budget=None, block_ids=None):
        """Return (keep_mask, scores). keep = {p >= tau_layer}, plus a sink+window
        floor, optionally capped at `budget` blocks (HBM ceiling). Budget is an
        OUTPUT of alpha unless a hard cap is supplied."""
        p = self.score(F)
        n = p.shape[0]
        tau = self.tau_by_layer.get(int(layer), self.tau_global)
        keep = p >= tau
        # deadline-safety floor: attention sink (first) + sliding window (last)
        bids = np.arange(n) if block_ids is None else np.asarray(block_ids)
        order = np.argsort(bids)
        fidx = list(order[:self.sink])
        if self.window > 0:                              # note: order[-0:] would force ALL
            fidx += list(order[-self.window:])
        fidx = np.unique(fidx).astype(int)
        keep[fidx] = True
        if budget is not None and keep.sum() > budget:   # honor a hard HBM ceiling
            forced = np.zeros(n, bool); forced[fidx] = True
            if budget <= int(forced.sum()):              # floor alone exceeds the ceiling
                fb = fidx[np.argsort(-p[fidx])][:budget]
                keep = np.zeros(n, bool); keep[fb] = True
            else:
                cand = np.where(keep & ~forced)[0]
                cand = cand[np.argsort(-p[cand])]
                keep = forced.copy()
                keep[cand[: budget - int(forced.sum())]] = True
        return keep, p

    def observe_miss(self, keep, y, layer):
        """Reveal labels: record realized miss-rate and (if gamma>0) adapt tau."""
        y = np.asarray(y).reshape(-1)
        sal = y > 0.5
        err = 0.0 if sal.sum() == 0 else float((sal & ~keep).sum() / sal.sum())
        self._errs.append(err)
        if self.gamma > 0:
            # ACI update: missing too much (err>alpha) => LOWER tau => keep more.
            tau = self.tau_by_layer.get(int(layer), self.tau_global)
            self.tau_by_layer[int(layer)] = float(np.clip(tau + self.gamma * (self.alpha - err), 0, 1))
        return err

    def realized_miss(self):
        return float(np.mean(self._errs)) if self._errs else float("nan")

    # ---- P3: query-free cross-layer prefetch hint -------------------------
    def prefetch_hint(self, prev_within, r=0.10):
        """Blocks to prefetch for the NEXT layer, predicted from THIS layer's
        within-EMA top-r (87.5% next-layer recall, measured). Query-free, so it
        can run before the next layer's query exists. Returns block indices."""
        hot = topk_indicator(np.asarray(prev_within, np.float32), r) > 0.5
        idx = set(np.flatnonzero(hot).tolist())
        self._prev_hot = idx
        return idx

    def reset(self):
        self._errs.clear(); self._prev_hot = set()
