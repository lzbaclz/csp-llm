"""RCRG: Risk-Controlled Retrieval Gating.

Per query i: closed_correct c_i, open_correct o_i, gate score g_i (self-consistency
agreement; HIGH => model is confident closed-book => SKIP retrieval).
Decision: skip retrieval (use closed-book) iff g_i >= lambda, else retrieve (open-book).

Accuracy-loss vs always-retrieve for a decision: L_i(lambda) = [g_i>=lambda]*(o_i - c_i)
(skipping loses o_i-c_i; negative when closed beats open => skipping HELPS).
E[L] is the accuracy we sacrifice vs always-retrieve; we certify E[L] <= eps.

L(lambda) is monotone non-increasing in lambda (higher lambda => skip fewer => less loss),
so Conformal Risk Control (Angelopoulos+ 2023) applies: pick the SMALLEST lambda whose
conformal-corrected calibration risk <= eps -> finite-sample E[L_test] <= eps under
exchangeability. Under SHIFT we use weighted (non-exchangeable) CRC.

Baselines: always-retrieve, never-retrieve, TARG (naive calib threshold, NO conformal
correction), oracle (skip iff c_i>=o_i). Win = RCRG hits the eps target with coverage,
esp. under shift where TARG's certificate collapses; and gating beats always-retrieve
on retrieve-hurts data.
"""
import json, sys, numpy as np
from collections import defaultdict

GVALS = None  # sorted candidate lambdas


def load(paths):
    R = []
    for p in paths:
        for line in open(p):
            R.append(json.loads(line))
    return R


def risk_at(lam, g, o, c):
    """empirical one-sided accuracy-DROP risk = mean over i of [g>=lam]*max(0,o-c).
    One-sided (skipping when closed>=open costs nothing) so it is MONOTONE non-increasing
    in lam (CRC requirement) and conservatively upper-bounds the true accuracy loss."""
    skip = g >= lam
    return float(np.mean(skip * np.maximum(0, o - c)))


def retr_freq(lam, g):
    return float(np.mean(g < lam))   # fraction that RETRIEVE


def crc_threshold(g, o, c, eps, B=1.0, weights=None):
    """Conformal Risk Control: smallest lambda s.t. (n*Rhat + B)/(n+1) <= eps.
    Higher lambda => lower risk; we want the smallest (most skipping) that certifies.
    weights: per-calibration-point weights for non-exchangeable CRC (normalized)."""
    cand = np.unique(g)
    cand = np.concatenate([cand, [cand.max() + 1e-6]])   # +inf => never skip (risk<=0)
    n = len(g)
    best = cand.max()
    for lam in sorted(cand):                              # monotone risk -> first feasible = smallest lam (most skipping)
        skip = (g >= lam).astype(float)
        loss = skip * np.maximum(0, o - c)
        if weights is None:
            rhat = loss.mean()
            corrected = (n * rhat + B) / (n + 1)
        else:
            w = weights / weights.sum()
            rhat = float(np.sum(w * loss))
            # non-exchangeable CRC inflation: add B * (max weight) as the finite-sample slack
            corrected = rhat + B * float(weights.max() / weights.sum())
        if corrected <= eps:
            best = lam
            break
    return best


def targ_threshold(g, o, c, eps):
    """Naive (TARG-style): smallest lambda whose POINT calibration risk <= eps. No
    conformal correction -> no finite-sample guarantee."""
    cand = np.sort(np.unique(g))
    for lam in cand:
        if risk_at(lam, g, o, c) <= eps:
            return lam
    return cand.max() + 1e-6


def evaluate(lam, g, o, c):
    skip = g >= lam
    acc = float(np.mean(np.where(skip, c, o)))
    return {"acc": acc, "retr_freq": retr_freq(lam, g), "loss_vs_always": float(np.mean(o)) - acc}


def main():
    R = load(sys.argv[1:] or ["results/g0.jsonl", "results/g1.jsonl"])
    g = np.array([r["gate_agree"] for r in R]); o = np.array([r["open_correct"] for r in R]); c = np.array([r["closed_correct"] for r in R])
    ds = np.array([r["ds"] for r in R])
    print(f"=== RCRG  N={len(R)} ; always-retrieve acc={o.mean():.3f}  never(closed) acc={c.mean():.3f}  oracle acc={np.mean(np.maximum(o,c)):.3f} ===")
    print("per-dataset (closed / open / retrieve-helps?):")
    for d in sorted(set(ds)):
        m = ds == d
        print(f"  {d:>16}: closed={c[m].mean():.2f} open={o[m].mean():.2f} delta(open-closed)={o[m].mean()-c[m].mean():+.2f}  n={m.sum()}")

    EPS = 0.02   # allow <=2% accuracy loss vs always-retrieve (meaningful: never-retrieve loses ~6%)
    rng = np.random.RandomState(0)
    n = len(R)

    # ---------- (1) SMALL-CALIBRATION regime: where the finite-sample guarantee earns its keep ----------
    # CRC controls the EXPECTED test risk E[loss]<=eps (marginal, not PAC). The naive threshold
    # (TARG) has NO finite-sample control: at small calib n it overfits to an over-aggressive
    # threshold and its MEAN test loss exceeds eps. Report mean test loss + violation rate.
    print(f"\n--- (1) CALIBRATION-SIZE sweep @ eps={EPS} (1000 splits each; deployment uses SMALL calib) ---")
    print(f"{'n_cal':>6} | {'TARG mean_loss':>14} {'TARG viol%':>10} {'TARG retr%':>10} | {'RCRG mean_loss':>14} {'RCRG viol%':>10} {'RCRG retr%':>10}")
    for ncal in [15, 25, 50, 100, 300, 575]:
        agg = {m: {"loss": [], "viol": [], "rf": []} for m in ["TARG", "RCRG"]}
        for _ in range(1000):
            idx = rng.permutation(n); cal, te = idx[:ncal], idx[ncal:]
            for name, thr in [("TARG", targ_threshold(g[cal], o[cal], c[cal], EPS)),
                              ("RCRG", crc_threshold(g[cal], o[cal], c[cal], EPS))]:
                ev = evaluate(thr, g[te], o[te], c[te])
                agg[name]["loss"].append(ev["loss_vs_always"]); agg[name]["viol"].append(ev["loss_vs_always"] > EPS)
                agg[name]["rf"].append(ev["retr_freq"])
        t, r = agg["TARG"], agg["RCRG"]
        print(f"{ncal:>6} | {np.mean(t['loss']):>+14.4f} {100*np.mean(t['viol']):>9.0f}% {100*np.mean(t['rf']):>9.0f}% |"
              f" {np.mean(r['loss']):>+14.4f} {100*np.mean(r['viol']):>9.0f}% {100*np.mean(r['rf']):>9.0f}%")
    print("  (win = at small n_cal, TARG mean_loss EXCEEDS eps (violates) while RCRG stays <= eps; RCRG pays via higher retr%)")

    # ---------- (2) UNDER SHIFT (small calib): calibrate on retrieve-helps-LEAST ds, deploy on retrieve-helps-MOST ----------
    by = {d: (ds == d) for d in set(ds)}
    helps = sorted(set(ds), key=lambda d: o[by[d]].mean() - c[by[d]].mean())
    easy = helps[:len(helps)//2]; hard = helps[len(helps)//2:]
    cal_pool = np.where(np.isin(ds, easy))[0]; te_m = np.isin(ds, hard)
    print(f"\n--- (2) UNDER SHIFT @ eps={EPS}: calib on {[str(x) for x in easy]} -> deploy on {[str(x) for x in hard]} (mean over 500 calib draws of n=60) ---")
    res = {m: {"loss": [], "viol": [], "rf": []} for m in ["TARG", "RCRG(plain)", "RCRG(weighted)"]}
    hte, edges = np.histogram(g[te_m], bins=10, range=(0, 1), density=True)
    for _ in range(500):
        cal = rng.choice(cal_pool, size=min(60, len(cal_pool)), replace=False)
        hcal, _ = np.histogram(g[cal], bins=10, range=(0, 1), density=True)
        bidx = np.clip(np.digitize(g[cal], edges[1:-1]), 0, 9)
        w = (hte[bidx] + 1e-2) / (hcal[bidx] + 1e-2)            # importance weight toward deploy gate-dist
        for name, thr in [("TARG", targ_threshold(g[cal], o[cal], c[cal], EPS)),
                          ("RCRG(plain)", crc_threshold(g[cal], o[cal], c[cal], EPS)),
                          ("RCRG(weighted)", crc_threshold(g[cal], o[cal], c[cal], EPS, weights=w))]:
            ev = evaluate(thr, g[te_m], o[te_m], c[te_m])
            res[name]["loss"].append(ev["loss_vs_always"]); res[name]["viol"].append(ev["loss_vs_always"] > EPS); res[name]["rf"].append(ev["retr_freq"])
    print(f"  {'method':>16} {'mean_loss':>10} {'viol%':>7} {'retr%':>7}  (target {EPS})")
    for name in ["TARG", "RCRG(plain)", "RCRG(weighted)"]:
        m = res[name]
        print(f"  {name:>16} {np.mean(m['loss']):>+10.4f} {100*np.mean(m['viol']):>6.0f}% {100*np.mean(m['rf']):>6.0f}%")

    # (2b) shift eps-sweep: weighted CRC trades guarantee-tightness for gating (not degenerate)
    print(f"\n  shift eps-sweep (weighted CRC vs TARG): eps -> retr% / viol%")
    for eps in [0.02, 0.05, 0.10]:
        tw = {"t_rf": [], "t_v": [], "w_rf": [], "w_v": []}
        for _ in range(300):
            cal = rng.choice(cal_pool, size=min(60, len(cal_pool)), replace=False)
            hcal, _ = np.histogram(g[cal], bins=10, range=(0, 1), density=True)
            bidx = np.clip(np.digitize(g[cal], edges[1:-1]), 0, 9)
            w = (hte[bidx] + 1e-2) / (hcal[bidx] + 1e-2)
            et = evaluate(targ_threshold(g[cal], o[cal], c[cal], eps), g[te_m], o[te_m], c[te_m])
            ew = evaluate(crc_threshold(g[cal], o[cal], c[cal], eps, weights=w), g[te_m], o[te_m], c[te_m])
            tw["t_rf"].append(et["retr_freq"]); tw["t_v"].append(et["loss_vs_always"] > eps)
            tw["w_rf"].append(ew["retr_freq"]); tw["w_v"].append(ew["loss_vs_always"] > eps)
        print(f"    eps={eps:>4}: TARG retr%={100*np.mean(tw['t_rf']):>3.0f} viol%={100*np.mean(tw['t_v']):>3.0f}  |  wCRC retr%={100*np.mean(tw['w_rf']):>3.0f} viol%={100*np.mean(tw['w_v']):>3.0f}")

    # ---------- (3) Certified accuracy-vs-cost Pareto (sweep eps) ----------
    print(f"\n--- (3) Certified Pareto (in-dist, mean over 100 splits): eps -> (acc, retr%, realized loss, coverage) ---")
    for eps in [0.0, 0.02, 0.05, 0.10, 0.15]:
        a_, r_, l_, cov_ = [], [], [], []
        for _ in range(100):
            idx = rng.permutation(n); cal, te = idx[:n//2], idx[n//2:]
            thr = crc_threshold(g[cal], o[cal], c[cal], eps)
            ev = evaluate(thr, g[te], o[te], c[te])
            a_.append(ev["acc"]); r_.append(ev["retr_freq"]); l_.append(ev["loss_vs_always"]); cov_.append(ev["loss_vs_always"] <= eps)
        print(f"  eps={eps:>4}: acc={np.mean(a_):.3f} retr%={100*np.mean(r_):>3.0f} realized_loss={np.mean(l_):+.3f} coverage={100*np.mean(cov_):.0f}%")


if __name__ == "__main__":
    main()
