"""Budget-at-fixed-quality (applied-review must-fix #1): invert tab:perlayer.

Instead of F1 at a fixed 20% budget, build each selector's F1-vs-budget curve and
find the BUDGET at which it reaches a fixed task-quality target. If the learned
2-view reaches the target at a LOWER budget than H2O/Ada-KV -> a memory win at equal
quality (the adoption metric a serving team optimizes). If equal -> honest parity.

Budgets: 0.10/0.30/0.50 from results/budget_sweep/b*, 0.20 from results/expand
(llama), full from results/expand (the quality ceiling = target reference).
"""
import json, os, glob, numpy as np
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]
SELECTORS = ["h2o", "xqp", "adakv"]
EXPAND = "experiments/results/expand/llama31_8b"
SWEEP = "experiments/results/budget_sweep"


def f1_mean(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d.get("f1_mean")


def curve(sel):
    """pooled mean F1 across datasets at each budget (datasets present everywhere)."""
    pts = {}
    for b, base in [(0.10, f"{SWEEP}/b0.10"), (0.20, EXPAND), (0.30, f"{SWEEP}/b0.30"),
                    (0.50, f"{SWEEP}/b0.50")]:
        vals = [f1_mean(f"{base}/{ds}_{sel}.json") for ds in DS]
        vals = [v for v in vals if v is not None]
        if len(vals) == len(DS):
            pts[b] = float(np.mean(vals))
    return pts


def budget_to_target(pts, target):
    """smallest budget whose F1 >= target (linear interp between measured points)."""
    bs = sorted(pts)
    for i, b in enumerate(bs):
        if pts[b] >= target:
            if i == 0:
                return b
            b0, f0 = bs[i - 1], pts[bs[i - 1]]
            f1 = pts[b]
            if f1 == f0:
                return b
            return b0 + (target - f0) * (b - b0) / (f1 - f0)
    return None  # never reaches target within measured range


if __name__ == "__main__":
    full = [f1_mean(f"{EXPAND}/{ds}_full.json") for ds in DS]
    full = float(np.mean([v for v in full if v is not None]))
    curves = {s: curve(s) for s in SELECTORS}
    print(f"Full-cache pooled F1 (target ceiling) = {full:.4f}\n")
    print("F1-vs-budget (pooled over 7 Llama datasets):")
    print(f"{'sel':>6} " + " ".join(f"b={b:<5}" for b in [0.10, 0.20, 0.30, 0.50]))
    for s in SELECTORS:
        print(f"{s:>6} " + " ".join(f"{curves[s].get(b, float('nan')):<7.4f}" for b in [0.10, 0.20, 0.30, 0.50]))

    print("\nBudget to reach a fixed quality target (memory at equal quality):")
    print(f"{'target':>22} " + " ".join(f"{s:>8}" for s in SELECTORS) + "   verdict")
    out = {"full": full, "curves": curves, "budget_at_target": {}}
    for label, target in [("full-0.02", full - 0.02), ("0.90*full", 0.90 * full),
                          ("0.95*full", 0.95 * full)]:
        bts = {s: budget_to_target(curves[s], target) for s in SELECTORS}
        out["budget_at_target"][label] = {s: bts[s] for s in SELECTORS}
        def fmt(x): return f"{x:.3f}" if x is not None else "  >0.5"
        # win if learned (xqp) budget < h2o budget by a margin
        bx, bh = bts["xqp"], bts["h2o"]
        if bx is not None and bh is not None:
            d = bh - bx
            verdict = f"xqp {'-' if d>0 else '+'}{abs(d):.3f} vs H2O ({'WIN' if d>0.02 else 'parity' if abs(d)<=0.02 else 'worse'})"
        else:
            verdict = "n/a (target out of range)"
        print(f"{label+' (F1='+format(target,'.3f')+')':>22} " +
              " ".join(f"{fmt(bts[s]):>8}" for s in SELECTORS) + f"   {verdict}")
    os.makedirs("experiments/results/tost", exist_ok=True)
    json.dump(out, open("experiments/results/tost/budget_at_quality.json", "w"), indent=2)
    print("\nWROTE experiments/results/tost/budget_at_quality.json")
