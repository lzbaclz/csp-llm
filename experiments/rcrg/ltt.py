"""Learn-then-Test (Angelopoulos+ 2021): PAC(1-delta) threshold selection.

CRC controls only the EXPECTED risk E[R(lambda)] <= eps (marginal). LTT upgrades to a
HIGH-PROBABILITY guarantee: P( R(lambda_hat) <= eps ) >= 1 - delta. This is the project's
"marginal != PAC" lesson made concrete.

Method: for each candidate threshold lambda, test H0_lambda: R(lambda) > eps using a valid
p-value (Hoeffding-Bentkus) on the calibration losses. Because the one-sided loss
[g>=lambda]*max(0,o-c) is MONOTONE in lambda, we use FIXED-SEQUENCE testing (test from the
safest/high-lambda downward, stop at first non-reject) -> controls family-wise error at
delta with NO multiplicity penalty. Return the most aggressive (smallest-lambda) certified.
"""
import numpy as np
try:
    from scipy.stats import binom
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _hoeffding_p(rhat, eps, n):
    if rhat >= eps:
        return 1.0
    return float(np.exp(-2 * n * (eps - rhat) ** 2))


def _hb_p(rhat, eps, n):
    """Hoeffding-Bentkus p-value for H0: R >= eps, observing empirical mean rhat (loss in [0,1])."""
    if rhat >= eps:
        return 1.0
    # Hoeffding (relative-entropy) term
    a = max(min(rhat, 1 - 1e-12), 1e-12); b = eps
    h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    p_hoef = float(np.exp(-n * h1))
    if not _HAVE_SCIPY:
        return min(1.0, p_hoef)
    # Bentkus term: e * P(Bin(n, eps) <= ceil(n*rhat))
    p_bent = float(np.e * binom.cdf(np.ceil(n * rhat), n, eps))
    return min(1.0, p_hoef, p_bent)


def ltt_threshold(g, o, c, eps, delta=0.1, pval="hb"):
    """Return the most aggressive lambda with P(R(lambda)<=eps) >= 1-delta (fixed-sequence)."""
    n = len(g)
    loss_all = np.maximum(0, o - c)
    cand = np.sort(np.unique(g))                       # ascending = more aggressive (skip more) at low lambda
    pf = _hb_p if pval == "hb" else _hoeffding_p
    # fixed sequence: from safest (high lambda) down to aggressive (low lambda)
    certified = cand.max() + 1e-6                       # = retrieve everything (R=0, trivially safe)
    for lam in cand[::-1]:                              # descending lambda
        skip = (g >= lam)
        rhat = float(np.mean(skip * loss_all))
        if pf(rhat, eps, n) <= delta:                  # reject H0: R>eps  => certified
            certified = lam
        else:
            break                                      # first non-reject stops the sequence
    return certified


def ltt_threshold_weighted(g, o, c, eps, weights, delta=0.1):
    """Weighted LTT for covariate shift: weighted empirical risk + an effective-sample-size
    Hoeffding bound (n_eff = (sum w)^2 / sum w^2). Conservative but valid under reweighting."""
    n = len(g)
    w = weights / weights.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    loss_all = np.maximum(0, o - c)
    cand = np.sort(np.unique(g))
    certified = cand.max() + 1e-6
    for lam in cand[::-1]:
        skip = (g >= lam)
        rhat = float(np.sum(w * skip * loss_all))
        if _hoeffding_p(rhat, eps, n_eff) <= delta:
            certified = lam
        else:
            break
    return certified


if __name__ == "__main__":
    # sanity: synthetic, loss bounded; LTT should be conservative vs CRC
    rng = np.random.RandomState(0)
    g = rng.rand(300); o = (rng.rand(300) < 0.5).astype(float); c = (rng.rand(300) < 0.4).astype(float)
    print("LTT lambda @ eps=0.05 delta=0.1:", round(ltt_threshold(g, o, c, 0.05, 0.1), 3))
