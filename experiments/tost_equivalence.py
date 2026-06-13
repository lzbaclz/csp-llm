"""Paired TOST equivalence test: learned 2-view (xqp) vs H2O on LongBench F1.

Replaces the underpowered 'statistically tied' claim (reviewer C1) with a real
equivalence test. Per-item F1 recomputed from stored pred/ref with SEER's exact
metric (verified == stored f1_mean), paired by request id, pooled across datasets.
Reports paired diff, 90% CI (the TOST CI), TOST p at several margins, and power.

Usage: python experiments/tost_equivalence.py [--dir DIR] [--a xqp --b h2o]
"""
import json, glob, os, sys, argparse, numpy as np
from scipy import stats
from scipy.stats import norm
sys.path.insert(0, "/home/lzq/codes/SEER")
from seer.eval.metrics import f1_score as seer_f1

def load_items(path):
    o = json.load(open(path)); out = {}
    for r in o.get("results", []):
        ref = r["ref"]; out[r["id"]] = (r["pred"], ref if isinstance(ref, list) else [ref])
    return out

def item_f1(pred, refs):
    return max(seer_f1(pred, rf) for rf in refs)

def collect(dirs, sel):
    """pooled per-item F1 keyed (dataset,id) across all result dirs for selector sel."""
    out = {}
    for d in dirs:
        for f in glob.glob(f"{d}/*_{sel}.json"):
            ds = os.path.basename(f)[:-(len(sel)+6)]
            for i, (p, r) in load_items(f).items():
                out[(ds, i)] = item_f1(p, r)
    return out

def tost(a, b, label, margins=(0.01,0.02,0.03,0.05)):
    keys = sorted(set(a) & set(b))
    A = np.array([a[k] for k in keys]); B = np.array([b[k] for k in keys])
    D = A - B; n = len(D); md = D.mean(); sd = D.std(ddof=1); se = sd/np.sqrt(n)
    ci90 = (md-1.645*se, md+1.645*se)
    t, p = stats.ttest_rel(A, B)
    res = {"label": label, "n": n, "mean_a": float(A.mean()), "mean_b": float(B.mean()),
           "mean_paired_diff": float(md), "sd": float(sd), "se": float(se),
           "ci90": [float(ci90[0]), float(ci90[1])], "ttest_p": float(p), "tost": {}, "power": {}}
    print(f"\n=== {label} | n={n} ===")
    print(f"mean_a={A.mean():.4f} mean_b={B.mean():.4f} diff={md:.4f} se={se:.4f} "
          f"90%CI=[{ci90[0]:.4f},{ci90[1]:.4f}] ttest_p={p:.3f}")
    for delta in margins:
        equiv = (ci90[0] > -delta) and (ci90[1] < delta)
        p_low = 1-stats.t.cdf((md+delta)/se, n-1); p_up = stats.t.cdf((md-delta)/se, n-1)
        tp = max(p_low, p_up)
        res["tost"][f"{delta}"] = {"equivalent": bool(equiv), "p": float(tp)}
        z = delta/se; power = norm.cdf(z-1.96)+norm.cdf(-z-1.96)
        res["power"][f"{delta}"] = float(power)
        print(f"  margin +/-{delta:.2f}: equivalent={equiv}  TOST_p={tp:.4f}  power(detect {delta})={power:.3f}")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["experiments/results/e2e_confirm"])
    ap.add_argument("--a", default="xqp"); ap.add_argument("--b", default="h2o")
    ap.add_argument("--out", default="experiments/results/tost/tost_xqp_vs_h2o.json")
    g = ap.parse_args()
    A = collect(g.dirs, g.a); B = collect(g.dirs, g.b)
    r = tost(A, B, f"{g.a} vs {g.b}")
    json.dump(r, open(g.out, "w"), indent=2); print("\nWROTE", g.out)
