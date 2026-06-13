"""Competitive baselines for the ICDM evaluation.

An ICDM predictor paper is rejected without a real baseline suite. This module
provides, behind one uniform `.score(F) -> P(salient)` interface (so they flow
through `dm_metrics` / `eval` exactly like `ClosedFormXQP`):

  Learned, trained on the same features
    * HistGBDT       — histogram gradient boosting (the de-facto tabular winner;
                       the closed-form-vs-GBDT comparison is the make-or-break
                       experiment for the "a minimal calibrated model suffices"
                       thesis — ICDM_PIVOT.md §A).
    * SklearnMLP     — a *trained* MLP (the repo's TinyMLP was random-init only).
    * OnlineSGD      — online logistic regression (partial_fit) — the streaming /
                       concept-drift baseline.

  SOTA heuristics recast as single-signal predictors (no fit; raw monotone view)
    * recency (f_pos), attention-EMA / H2O-style (f_within),
      Quest (f_query), InfiniGen (f_cross).

All learned baselines handle the ~10% class imbalance where the estimator
supports it (`class_weight="balanced"`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Baseline:
    """Uniform wrapper: `.score(F)` returns P(positive); `.meta` holds bookkeeping."""
    name: str
    _scorer: object                      # callable F -> proba, or a fitted sklearn clf
    is_sklearn: bool = False
    meta: dict = field(default_factory=dict)

    def score(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float32)
        if self.is_sklearn:
            p = self._scorer.predict_proba(F)
            return p[:, 1].astype(np.float32)
        return np.asarray(self._scorer(F), dtype=np.float32)


# --------------------------- single-signal recasts -------------------------

#: column index in the FEATURE_NAMES order (s_within, s_cross, s_query, s_pos)
_SIGNAL_COL = {
    "recency": 3,
    "H2O/attn-EMA": 0,     # H2O/HALO within-layer attention magnitude
    "Quest": 2,            # query-key affinity
    "InfiniGen": 1,        # prev-layer hot-set
}


def single_signal_baselines() -> list[Baseline]:
    out = []
    for name, col in _SIGNAL_COL.items():
        out.append(Baseline(
            name=name,
            _scorer=(lambda c: (lambda F: F[:, c]))(col),
            is_sklearn=False,
            meta=dict(kind="heuristic", params=0, column=col),
        ))
    return out


# ------------------------------ learned baselines --------------------------

def fit_gbdt(F, y, *, max_depth=3, n_estimators=150, seed=0) -> Baseline:
    """Gradient-boosted trees — the de-facto tabular winner and the key
    comparison for the "a minimal calibrated model suffices" thesis.

    Prefers LightGBM / XGBoost when installed (the real-world GBDTs); otherwise
    falls back to sklearn's exact ``GradientBoostingClassifier``. We deliberately
    avoid ``HistGradientBoostingClassifier``: its OpenMP histogram binning
    segfaults alongside torch's libomp on macOS.
    """
    import importlib.util as u
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    t0 = time.perf_counter()
    if u.find_spec("lightgbm") is not None:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                 class_weight="balanced", verbosity=-1,
                                 random_state=seed).fit(F, y)
        name, backend = "GBDT(LightGBM)", "lightgbm"
    elif u.find_spec("xgboost") is not None:
        import xgboost as xgb
        spw = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))
        clf = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                scale_pos_weight=spw, verbosity=0,
                                eval_metric="logloss", random_state=seed).fit(F, y)
        name, backend = "GBDT(XGBoost)", "xgboost"
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y)
        clf = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                         learning_rate=0.1, random_state=seed).fit(F, y, sample_weight=sw)
        name, backend = "GBDT(sklearn)", "sklearn"
    dt = time.perf_counter() - t0
    return Baseline(name, clf, is_sklearn=True,
                    meta=dict(kind="gbdt", backend=backend, params=n_estimators,
                              max_depth=max_depth, fit_seconds=dt))


def fit_sklearn_mlp(F, y, *, hidden=(16,), seed=0, max_iter=400) -> Baseline:
    from sklearn.neural_network import MLPClassifier
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    t0 = time.perf_counter()
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        alpha=1e-3, max_iter=max_iter, random_state=seed).fit(F, y)
    dt = time.perf_counter() - t0
    n_params = sum(int(w.size) for w in clf.coefs_) + sum(int(b.size) for b in clf.intercepts_)
    return Baseline(f"sklearnMLP{hidden}", clf, is_sklearn=True,
                    meta=dict(kind="mlp", params=n_params, fit_seconds=dt))


def fit_online_sgd(F, y, *, seed=0, n_passes=1, batch=64) -> Baseline:
    """Online logistic regression via partial_fit — the streaming baseline."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.utils.class_weight import compute_class_weight
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    classes = np.array([0, 1])
    # 'balanced' is unsupported with partial_fit, so precompute the weights.
    cw = compute_class_weight("balanced", classes=classes, y=y)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4,
                        class_weight={0: float(cw[0]), 1: float(cw[1])},
                        random_state=seed)
    t0 = time.perf_counter()
    for _ in range(n_passes):
        for i in range(0, F.shape[0], batch):
            clf.partial_fit(F[i:i + batch], y[i:i + batch], classes=classes)
    dt = time.perf_counter() - t0
    return Baseline("OnlineSGD(logloss)", clf, is_sklearn=True,
                    meta=dict(kind="online", params=F.shape[1] + 1, fit_seconds=dt))


def all_learned_baselines(F, y, seed=0) -> list[Baseline]:
    return [
        fit_gbdt(F, y, seed=seed),
        fit_sklearn_mlp(F, y, seed=seed),
        fit_online_sgd(F, y, seed=seed),
    ]
