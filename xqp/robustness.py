"""Cross-model robustness analysis — addresses round-1 review M3.

Train predictor on model A's traces; evaluate AUC on models B, C with the
*frozen* weights. AUC degradation > 0.02 = cross-model claim is overstated.
"""
from __future__ import annotations

import numpy as np

from .eval import roc_auc
from .predictor import ClosedFormXQP
from .trace import load_trace


def cross_model_auc(train_trace: str, eval_traces: dict[str, str],
                    horizon: str = "y_h4", l2: float = 1e-3) -> dict:
    """Train on train_trace, evaluate on each eval_traces[name].

    Returns dict[name → AUC].
    """
    tr = load_trace(train_trace)
    F_tr = np.stack([tr["f_within"], tr["f_cross"], tr["f_query"], tr["f_pos"]], axis=1)
    y_tr = tr[horizon].astype(np.float32)
    pred = ClosedFormXQP.from_fit(F_tr, y_tr, l2=l2)

    results = {}
    for name, path in eval_traces.items():
        ev = load_trace(path)
        F_ev = np.stack([ev["f_within"], ev["f_cross"], ev["f_query"], ev["f_pos"]], axis=1)
        y_ev = ev[horizon].astype(np.float32)
        s = pred.score(F_ev)
        results[name] = float(roc_auc(y_ev, s))
    return results


def cross_model_degradation_report(*, baseline: dict, transfer: dict) -> dict:
    """Compare AUC trained-and-evaluated on same model vs. transferred."""
    return {
        m: {
            "in_model": baseline.get(m, float("nan")),
            "transfer": transfer.get(m, float("nan")),
            "degradation": baseline.get(m, float("nan")) - transfer.get(m, float("nan")),
        }
        for m in baseline.keys() | transfer.keys()
    }
