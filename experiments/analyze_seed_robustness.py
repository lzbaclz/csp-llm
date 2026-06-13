"""Sample-robustness of the TOST equivalence over a GENUINELY DISJOINT prompt subset.

Reviewer-driven fix: a literal ``--seed`` is a no-op for prompt selection here
(greedy decode + deterministic LongBench file order => the first N prompts are drawn
regardless of seed, so per-seed F1 was byte-identical -- the old "3 independent
seeds" premise was false). The meaningful robustness question is whether the
+/-0.02 F1 equivalence verdict survives a *different* prompt sample, which we obtain
with ``LONGBENCH_OFFSET`` (see run_disjoint_subset.sh):

  window A (main)     = prompts [0:64]   -> experiments/results/expand/llama31_8b
  window B (disjoint) = prompts [64:128] -> experiments/results/expand_disjoint

Llama-3.1-8B, 7 LongBench QA datasets, matched 20% budget, selectors vs H2O. We
report a BCa (bias-corrected, accelerated) bootstrap CI in addition to the percentile
CI so knife-edge ties are not an artifact of one resampling scheme.
"""
import json, os, argparse
import numpy as np
from scipy import stats

WINDOW_DIRS = {
    "A_main_[0:64]":     "experiments/results/expand/llama31_8b",
    "B_disjoint_[64:128]": "experiments/results/expand_disjoint",
}
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]


def load_f1(d, ds, sel):
    f = f"{d}/{ds}_{sel}.json"
    if not os.path.exists(f):
        return {}
    o = json.load(open(f))
    return {r["id"]: float(r["f1"]) for r in o.get("results", []) if "f1" in r}


def paired(d, selA, selB):
    A, B = [], []
    for ds in DS:
        a, b = load_f1(d, ds, selA), load_f1(d, ds, selB)
        for i in sorted(set(a) & set(b)):
            A.append(a[i]); B.append(b[i])
    return np.array(A), np.array(B)


def _bca_ci(D, stat, boot, alpha, rng):
    """BCa CI for the mean of D (paired diff)."""
    n = len(D); theta = stat(D)
    bs = np.array([stat(D[rng.integers(0, n, n)]) for _ in range(boot)])
    # bias-correction z0
    p0 = np.mean(bs < theta)
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(p0)
    # acceleration via jackknife
    jk = np.array([stat(np.delete(D, i)) for i in range(n)]) if n <= 1500 else None
    if jk is not None:
        jbar = jk.mean(); num = ((jbar - jk) ** 3).sum(); den = 6.0 * (((jbar - jk) ** 2).sum() ** 1.5 + 1e-12)
        a = num / den
    else:
        a = 0.0
    def adj(q):
        z = stats.norm.ppf(q)
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.percentile(bs, 100 * adj(alpha)); hi = np.percentile(bs, 100 * adj(1 - alpha))
    return float(lo), float(hi)


def tost_row(A, B, margin=0.02, boot=10000, seed=1234):
    D = A - B; n = len(D); md = D.mean(); se = D.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    bs = np.array([D[rng.integers(0, n, n)].mean() for _ in range(boot)])
    lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
    bca_lo, bca_hi = _bca_ci(D, np.mean, boot, 0.05, np.random.default_rng(seed + 1))
    equiv = (lo > -margin) and (hi < margin)
    bca_equiv = (bca_lo > -margin) and (bca_hi < margin)
    p_low = 1 - stats.t.cdf((md + margin) / se, n - 1)
    p_up = stats.t.cdf((md - margin) / se, n - 1)
    return dict(n=n, diff=float(md), ci90=[float(lo), float(hi)], bca90=[bca_lo, bca_hi],
                equiv=bool(equiv), bca_equiv=bool(bca_equiv), tost_p=float(max(p_low, p_up)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/tost/seed_robustness.json")
    g = ap.parse_args()
    contrasts = [("xqp", "h2o"), ("pyramidkv", "h2o"), ("adakv", "h2o")]
    res = {}
    for a, b in contrasts:
        print(f"\n=== {a} vs {b} (Llama, 7 datasets, matched 20%) ===")
        print(f"{'window':>22} {'n':>4} {'diff':>9} {'90% CI (pctl)':>22} {'BCa 90%':>22} {'equiv ±0.02':>12} {'TOST p':>9}")
        res[f"{a}_vs_{b}"] = {}
        for w, d in WINDOW_DIRS.items():
            A, B = paired(d, a, b)
            if len(A) == 0:
                print(f"{w:>22}  (no data yet)"); continue
            r = tost_row(A, B)
            res[f"{a}_vs_{b}"][w] = r
            print(f"{w:>22} {r['n']:>4} {r['diff']:>+9.4f} "
                  f"[{r['ci90'][0]:>+8.4f},{r['ci90'][1]:>+8.4f}] "
                  f"[{r['bca90'][0]:>+8.4f},{r['bca90'][1]:>+8.4f}] "
                  f"{str(r['equiv'])+'/'+str(r['bca_equiv']):>12} {r['tost_p']:>9.4f}")
    os.makedirs(os.path.dirname(g.out), exist_ok=True)
    json.dump(res, open(g.out, "w"), indent=2)
    print("\nWROTE", g.out)
