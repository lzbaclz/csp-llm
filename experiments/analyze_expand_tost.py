"""Powered TOST + matched-budget comparison on the expansion sweep.

7 LongBench QA-F1 datasets x 2 architectures (Llama-3.1-8B, Qwen2.5-7B) x
{full,h2o,xqp,pyramidkv,adakv} at matched 20% budget, n=64/cell. Per-item F1 is the
runner's own SQuAD metric (output["results"][i]["f1"]), paired by (arch,dataset,id).

Addresses reviewer must-fixes:
  C1/C8  xqp vs h2o equivalence (TOST, pre-registered margin +/-0.02), pooled + per-arch.
  C3/C5  pyramidkv/adakv at matched budget vs h2o and vs xqp (the real per-layer comparators).
  C4     cross-architecture task-quality transfer: Llama-trained xqp applied to Qwen serving.
"""
import json, glob, os, argparse, numpy as np
from scipy import stats
from scipy.stats import norm

ROOT = "experiments/results/expand"
ARCHS = ["llama31_8b", "qwen25_7b"]
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]


def load_f1(arch, ds, sel):
    f = f"{ROOT}/{arch}/{ds}_{sel}.json"
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))
    return {r["id"]: float(r["f1"]) for r in d.get("results", []) if "f1" in r}


def paired(selA, selB, archs):
    A, B = [], []
    for arch in archs:
        for ds in DS:
            a, b = load_f1(arch, ds, selA), load_f1(arch, ds, selB)
            for i in sorted(set(a) & set(b)):
                A.append(a[i]); B.append(b[i])
    return np.array(A), np.array(B)


def tost(A, B, label, margins=(0.01, 0.02, 0.03), boot=10000, seed=1234):
    D = A - B
    n = len(D); md = D.mean(); sd = D.std(ddof=1); se = sd / np.sqrt(n)
    # bootstrap CI on the paired mean diff (robust, non-normal F1)
    rng = np.random.default_rng(seed)
    bs = np.array([D[rng.integers(0, n, n)].mean() for _ in range(boot)])
    ci90 = (np.percentile(bs, 5), np.percentile(bs, 95))
    ci95 = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    t, p = stats.ttest_rel(A, B)
    out = {"label": label, "n": n, "mean_A": float(A.mean()), "mean_B": float(B.mean()),
           "diff": float(md), "se": float(se), "ci90": [float(ci90[0]), float(ci90[1])],
           "ci95": [float(ci95[0]), float(ci95[1])], "ttest_p": float(p), "tost": {}, "power": {}}
    print(f"\n{label} | n={n}  mean_A={A.mean():.4f} mean_B={B.mean():.4f} "
          f"diff={md:+.4f}  90%CI=[{ci90[0]:+.4f},{ci90[1]:+.4f}]  t-p={p:.3f}")
    for dlt in margins:
        equiv = (ci90[0] > -dlt) and (ci90[1] < dlt)
        p_low = 1 - stats.t.cdf((md + dlt) / se, n - 1)
        p_up = stats.t.cdf((md - dlt) / se, n - 1)
        tp = max(p_low, p_up)
        z = dlt / se; power = norm.cdf(z - 1.96) + norm.cdf(-z - 1.96)
        out["tost"][f"{dlt}"] = {"equivalent": bool(equiv), "p": float(tp)}
        out["power"][f"{dlt}"] = float(power)
        print(f"    margin +/-{dlt:.2f}: equivalent={str(equiv):5s} TOST_p={tp:.4f} power={power:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/tost/expand_tost.json")
    g = ap.parse_args()
    results = {}

    print("=" * 70 + "\nHEADLINE EQUIVALENCE  (learned 2-view xqp  vs  H2O)\n" + "=" * 70)
    for tag, archs in [("pooled(2arch)", ARCHS), ("llama", ["llama31_8b"]), ("qwen", ["qwen25_7b"])]:
        A, B = paired("xqp", "h2o", archs)
        results[f"xqp_vs_h2o::{tag}"] = tost(A, B, f"xqp vs h2o [{tag}]")

    print("\n" + "=" * 70 + "\nPER-LAYER BUDGET BASELINES at matched budget (C3/C5)\n" + "=" * 70)
    for a, b in [("pyramidkv", "h2o"), ("adakv", "h2o"), ("xqp", "pyramidkv"), ("xqp", "adakv")]:
        A, B = paired(a, b, ARCHS)
        results[f"{a}_vs_{b}::pooled"] = tost(A, B, f"{a} vs {b} [pooled 2arch]")

    print("\n" + "=" * 70 + "\nC4  CROSS-ARCH TASK-QUALITY TRANSFER (Llama-trained xqp on Qwen)\n" + "=" * 70)
    # xqp on qwen vs h2o on qwen: does the transferred scorer still tie the in-arch baseline?
    A, B = paired("xqp", "h2o", ["qwen25_7b"])
    results["C4_xqp_vs_h2o_on_qwen"] = tost(A, B, "xqp(Llama-trained) vs h2o, on Qwen serving")

    print("\n" + "=" * 70 + "\nCOMPRESSION COST (full vs xqp), context only\n" + "=" * 70)
    A, B = paired("full", "xqp", ARCHS)
    results["full_vs_xqp::pooled"] = tost(A, B, "full vs xqp [pooled]")

    os.makedirs(os.path.dirname(g.out), exist_ok=True)
    json.dump(results, open(g.out, "w"), indent=2)
    print("\nWROTE", g.out)
