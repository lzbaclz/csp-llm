"""Information-theoretic checks — addresses 100-round R35 (HKUST), R38 (Osaka).

The Bayes-optimality argument in §3.2 assumes conditional independence
of the four features given block importance. We provide:
- mutual_information(F, y) per feature
- chi_square_independence(f_i, f_j | y) for pairs
- kl_joint_vs_product(F) to quantify overall dependency
"""
from __future__ import annotations

import numpy as np


def discretize(x: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Quantile-bin a continuous feature into integer bins."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(x, qs[1:-1])
    return bins.astype(np.int64)


def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 16,
                       miller_madow: bool = False) -> float:
    """Estimate I(x; y) for continuous x and binary y.

    Plug-in (maximum-likelihood) MI is positively biased, and the bias grows as
    cells sparsen. With ``miller_madow=True`` we apply the Miller--Madow
    correction, ``I_MM = I_plugin + (Kx + Ky - Kxy - 1) / (2N)`` where K* are the
    counts of non-empty bins; this shrinks the upward bias and is the bias-robust
    cross-check reported for the load-bearing cross-vs-query comparison.
    """
    x = discretize(x, n_bins=n_bins)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n = x.shape[0]
    c_xy = np.zeros((n_bins + 1, 2), dtype=np.float64)
    for i in range(n):
        c_xy[x[i], int(y[i])] += 1
    p_xy = c_xy / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = p_x * p_y
        log_term = np.where(p_xy > 0,
                             np.log((p_xy + 1e-12) / (denom + 1e-12)),
                             0.0)
    mi = float((p_xy * log_term).sum())
    if miller_madow:
        k_xy = int((c_xy > 0).sum())
        k_x = int((c_xy.sum(axis=1) > 0).sum())
        k_y = int((c_xy.sum(axis=0) > 0).sum())
        mi = max(0.0, mi + (k_x + k_y - k_xy - 1) / (2.0 * n))
    return mi


def chi_square_conditional(f_i: np.ndarray, f_j: np.ndarray, y: np.ndarray,
                            n_bins: int = 8) -> dict:
    """χ² test of independence between f_i and f_j given y.

    Returns {stat, dof, p_approx} per y class. Higher stat → more
    dependence (rejecting independence)."""
    f_i = discretize(f_i, n_bins=n_bins)
    f_j = discretize(f_j, n_bins=n_bins)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    results = {}
    for c in np.unique(y):
        mask = y == c
        if mask.sum() < 8:
            continue
        joint = np.zeros((n_bins + 1, n_bins + 1), dtype=np.float64)
        for a, b in zip(f_i[mask], f_j[mask]):
            joint[a, b] += 1
        row = joint.sum(axis=1, keepdims=True)
        col = joint.sum(axis=0, keepdims=True)
        total = joint.sum()
        if total < 1:
            continue
        expected = row @ col / total
        with np.errstate(invalid="ignore", divide="ignore"):
            stat = np.where(expected > 0,
                            (joint - expected) ** 2 / (expected + 1e-12),
                            0.0).sum()
        dof = (n_bins - 1) ** 2  # (rows-1)(cols-1) for the n_bins x n_bins table
        results[int(c)] = dict(stat=float(stat), dof=int(dof))
    return results


def kl_joint_vs_product(F: np.ndarray, n_bins: int = 8) -> float:
    """KL(p(F) || prod_i p(F_i)). 0 ⇒ features are independent."""
    F = np.asarray(F, dtype=np.float32)
    N, D = F.shape
    # For tractability we only do pairs; full joint blows up
    total = 0.0
    pairs = 0
    for i in range(D):
        for j in range(i + 1, D):
            a = discretize(F[:, i], n_bins=n_bins)
            b = discretize(F[:, j], n_bins=n_bins)
            p_ab = np.zeros((n_bins + 1, n_bins + 1), dtype=np.float64)
            for ai, bj in zip(a, b):
                p_ab[ai, bj] += 1
            p_ab = p_ab / N
            p_a = p_ab.sum(axis=1, keepdims=True)
            p_b = p_ab.sum(axis=0, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(p_ab > 0,
                             p_ab * np.log((p_ab + 1e-12) / (p_a * p_b + 1e-12)),
                             0.0)
            total += float(t.sum())
            pairs += 1
    return total / max(1, pairs)


# ---------------------------------------------------------------------------
# ICDM redundancy analysis: conditional MI + interaction information.
# This is the measurement that backs C2 (low-redundancy multi-view fusion) and
# gates the ICDM go/no-go — see ICDM_PIVOT.md §A. Run on REAL attention traces;
# on synthetic data it only reflects the generator's hard-coded correlations.
# ---------------------------------------------------------------------------

def _mi_binned(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    """MI between two already-integer-binned arrays via the joint histogram."""
    n = a.shape[0]
    if n == 0:
        return 0.0
    p = np.zeros((n_bins + 1, n_bins + 1), dtype=np.float64)
    np.add.at(p, (a, b), 1.0)
    p /= n
    pa = p.sum(axis=1, keepdims=True)
    pb = p.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(p > 0, p * np.log((p + 1e-12) / (pa * pb + 1e-12)), 0.0)
    return float(max(0.0, t.sum()))


def mutual_information_xx(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """I(X;Y) between two continuous features (both quantile-binned)."""
    return _mi_binned(discretize(x, n_bins=n_bins), discretize(y, n_bins=n_bins), n_bins)


def conditional_mutual_information(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                   n_bins: int = 8) -> float:
    """I(X;Y|Z) for continuous x, y and discrete class z (the top-r label).

    I(X;Y|Z) = sum_z p(z) * I(X;Y | Z=z), each term on the z-subset. >= 0.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(z, dtype=np.int64).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return 0.0
    total = 0.0
    for c in np.unique(z):
        mask = z == c
        if mask.sum() < 8:
            continue
        total += (mask.sum() / n) * mutual_information_xx(x[mask], y[mask], n_bins=n_bins)
    return float(total)


def interaction_information(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            n_bins: int = 8) -> float:
    """II = I(X;Y|Z) - I(X;Y).

      < 0  -> X, Y are REDUNDANT about Z (share information; e.g. within- and
              cross-layer attention magnitude).
      > 0  -> SYNERGISTIC (jointly informative beyond the sum).
      ~ 0  -> independent contributors — the regime where additive / log-linear
              fusion is near-Bayes-optimal.
    """
    return (conditional_mutual_information(x, y, z, n_bins=n_bins)
            - mutual_information_xx(x, y, n_bins=n_bins))


def redundancy_report(F: np.ndarray, y: np.ndarray, feature_names=None,
                      n_bins: int = 8) -> dict:
    """The §3 redundancy table.

    Returns per-feature I(Xi;Y) (relevance) and, for each pair, I(Xi;Xj),
    I(Xi;Xj|Y), and the interaction information with a redundant/synergistic/
    independent verdict. The fusion's near-optimality claim (C2) is supported
    exactly when most pairs are independent or redundant (never strongly
    synergistic).
    """
    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    _, D = F.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(D)]
    per_feature = {names[i]: mutual_information(F[:, i], y, n_bins=max(n_bins, 16))
                   for i in range(D)}
    pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            mi = mutual_information_xx(F[:, i], F[:, j], n_bins=n_bins)
            cmi = conditional_mutual_information(F[:, i], F[:, j], y, n_bins=n_bins)
            ii = cmi - mi
            verdict = ("redundant" if ii < -1e-3 else
                       "synergistic" if ii > 1e-3 else "independent")
            pairs.append(dict(pair=f"{names[i]}~{names[j]}", mi=mi, cond_mi=cmi,
                              interaction=ii, verdict=verdict))
    n_syn = sum(1 for p in pairs if p["verdict"] == "synergistic")
    return dict(per_feature_mi=per_feature, pairs=pairs,
                n_synergistic_pairs=n_syn,
                fusion_near_optimal_signal=bool(n_syn == 0))


def conditional_mi_view(F: np.ndarray, y: np.ndarray, i: int,
                        n_bins: int = 12, n_cond: int = 3) -> float:
    """``I(X_i; Y | X_{-i})`` — the predictive information view ``i`` adds *beyond
    all the other views jointly* (binned plug-in).

    This is the "unique information" test that backs the new design's claim that
    every retained parameter is useful: ``~0`` means view ``i`` is redundant
    given the others (a dead parameter); ``>0`` means it carries information no
    combination of the others does. The other views are coarsely binned
    (``n_cond`` bins each) and we condition on their joint cell.
    """
    F = np.asarray(F, np.float32)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    N, D = F.shape
    if N == 0:
        return 0.0
    xi = discretize(F[:, i], n_bins=n_bins)
    code = np.zeros(N, dtype=np.int64)
    base = 1
    for j in range(D):
        if j == i:
            continue
        oj = discretize(F[:, j], n_bins=n_cond)
        code += oj * base
        base *= (n_cond + 1)
    total = 0.0
    for c in np.unique(code):
        mask = code == c
        if mask.sum() < 16:
            continue
        total += (mask.sum() / N) * _mi_binned(xi[mask], y[mask], n_bins)
    return float(total)


def unique_information_report(F: np.ndarray, y: np.ndarray, feature_names=None,
                              n_bins: int = 12, n_cond: int = 3) -> dict:
    """Per-view relevance ``I(X_i;Y)`` and unique contribution ``I(X_i;Y|X_{-i})``.

    A view is "useful" (its parameter earns its place) iff its conditional MI is
    materially above zero. Used by the gated design to certify that within, cross
    and the (selectively-computed) query view each add unique information.
    """
    F = np.asarray(F, np.float32)
    y = np.asarray(y).reshape(-1)
    _, D = F.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(D)]
    out = {}
    for i in range(D):
        out[names[i]] = dict(
            relevance_mi=mutual_information(F[:, i], y, n_bins=max(n_bins, 16)),
            unique_cmi=conditional_mi_view(F, y, i, n_bins=n_bins, n_cond=n_cond),
        )
    return out

