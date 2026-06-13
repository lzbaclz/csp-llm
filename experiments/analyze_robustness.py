"""Sample-robustness of the TOST equivalence (honest answer to 'multi-seed').

Literal --seed is a no-op (greedy decode + deterministic LongBench order => identical
F1; verified: seeds 1-2 reproduce seed 0 bit-for-bit). The meaningful question is
whether the +/-0.02 equivalence verdict survives a DIFFERENT prompt sample. We compare
the canonical prompts [0:64] (main sweep) against the DISJOINT subset [64:128]
(LONGBENCH_OFFSET=64), Llama-3.1-8B, 7 LongBench QA datasets, matched 20% budget.
"""
import json, os, numpy as np
from scipy import stats

SUBSETS = {
    "main [0:64]":     "experiments/results/expand/llama31_8b",
    "disjoint [64:128]": "experiments/results/expand_disjoint",
}
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]


def load_f1(d, ds, sel):
    f = f"{d}/{ds}_{sel}.json"
    if not os.path.exists(f):
        return {}
    return {r["id"]: float(r["f1"]) for r in json.load(open(f)).get("results", []) if "f1" in r}


def paired(d, a, b):
    A, B = [], []
    for ds in DS:
        x, y = load_f1(d, ds, a), load_f1(d, ds, b)
        for i in sorted(set(x) & set(y)):
            A.append(x[i]); B.append(y[i])
    return np.array(A), np.array(B)


def tost(A, B, margin=0.02, boot=10000, seed=1234):
    D = A - B; n = len(D); md = D.mean(); se = D.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    bs = np.array([D[rng.integers(0, n, n)].mean() for _ in range(boot)])
    lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
    p = max(1 - stats.t.cdf((md + margin) / se, n - 1), stats.t.cdf((md - margin) / se, n - 1))
    return dict(n=n, diff=float(md), ci90=[float(lo), float(hi)],
                equiv=bool(lo > -margin and hi < margin), tost_p=float(p))


if __name__ == "__main__":
    res = {}
    for a, b in [("xqp", "h2o"), ("pyramidkv", "h2o"), ("adakv", "h2o")]:
        print(f"\n=== {a} vs {b} (Llama, 7 datasets, matched 20%) ===")
        print(f"{'subset':>20} {'n':>4} {'diff':>9} {'90% CI':>22} {'equiv ±0.02':>12} {'TOST p':>8}")
        res[f"{a}_vs_{b}"] = {}
        for name, d in SUBSETS.items():
            A, B = paired(d, a, b)
            if len(A) == 0:
                print(f"{name:>20}  (no data yet)"); continue
            r = tost(A, B); res[f"{a}_vs_{b}"][name] = r
            print(f"{name:>20} {r['n']:>4} {r['diff']:>+9.4f} "
                  f"[{r['ci90'][0]:>+8.4f},{r['ci90'][1]:>+8.4f}] {str(r['equiv']):>12} {r['tost_p']:>8.4f}")
    os.makedirs("experiments/results/tost", exist_ok=True)
    json.dump(res, open("experiments/results/tost/robustness.json", "w"), indent=2)
    print("\nWROTE experiments/results/tost/robustness.json")
