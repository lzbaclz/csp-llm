"""Analyze the panel-strengthening sweeps: 3-seed robustness + Mistral transfer.

(c2) TRUE-seed: headline (xqp vs h2o) and per-layer (pyramidkv/adakv vs h2o) TOST at
     each of 3 genuinely-random prompt subsets (LONGBENCH_SEED 1/2/3), Llama, 7 datasets.
(a)  3rd PAIR: Llama-trained scorer frozen in Mistral's loop vs Mistral's own H2O.
"""
import json, os, numpy as np
from scipy import stats
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]


def f1map(p):
    if not os.path.exists(p): return {}
    return {r["id"]: float(r["f1"]) for r in json.load(open(p)).get("results", []) if "f1" in r}


def paired(d, a, b):
    A, B = [], []
    for ds in DS:
        x, y = f1map(f"{d}/{ds}_{a}.json"), f1map(f"{d}/{ds}_{b}.json")
        for i in sorted(set(x) & set(y)):
            A.append(x[i]); B.append(y[i])
    return np.array(A), np.array(B)


def tost(A, B, m=0.02, boot=10000, seed=1234):
    if len(A) == 0: return None
    D = A - B; n = len(D); md = D.mean(); se = D.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed); bs = np.array([D[rng.integers(0, n, n)].mean() for _ in range(boot)])
    lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
    p = max(1 - stats.t.cdf((md + m) / se, n - 1), stats.t.cdf((md - m) / se, n - 1))
    return dict(n=n, diff=float(md), ci90=[float(lo), float(hi)], equiv=bool(lo > -m and hi < m), p=float(p))


res = {}
print("=== (c2) TRUE-seed robustness: TOST vs H2O at 3 random prompt subsets (Llama) ===")
for a in ["xqp", "pyramidkv", "adakv"]:
    print(f"\n  {a} vs h2o:")
    print(f"    {'seed':>6} {'n':>4} {'diff':>9} {'90% CI':>22} {'equiv ±0.02':>12} {'TOST p':>9}")
    diffs = []
    res[f"{a}_vs_h2o"] = {}
    for s in [1, 2, 3]:
        r = tost(*paired(f"experiments/results/trueseed/s{s}", a, "h2o"))
        if r is None: print(f"    s{s}: (no data)"); continue
        diffs.append(r["diff"]); res[f"{a}_vs_h2o"][f"s{s}"] = r
        print(f"    {('s'+str(s)):>6} {r['n']:>4} {r['diff']:>+9.4f} [{r['ci90'][0]:>+8.4f},{r['ci90'][1]:>+8.4f}] "
              f"{str(r['equiv']):>12} {r['p']:>9.4f}")
    if diffs:
        print(f"    -> across-seed mean diff {np.mean(diffs):+.4f}  spread [{min(diffs):+.4f},{max(diffs):+.4f}]")

print("\n=== (a) 3rd transfer pair: Llama-scorer frozen in MISTRAL loop vs Mistral H2O ===")
r = tost(*paired("experiments/results/transfer_mistral", "xqp", "h2o"))
if r:
    print(f"  xqp(Llama-trained) on Mistral vs Mistral-H2O: n={r['n']} diff={r['diff']:+.4f} "
          f"CI=[{r['ci90'][0]:+.4f},{r['ci90'][1]:+.4f}] equiv±0.02={r['equiv']} TOST_p={r['p']:.4f}")
    res["mistral_transfer"] = r
os.makedirs("experiments/results/tost", exist_ok=True)
json.dump(res, open("experiments/results/tost/panel_strengthen.json", "w"), indent=2)
print("\nWROTE experiments/results/tost/panel_strengthen.json")
