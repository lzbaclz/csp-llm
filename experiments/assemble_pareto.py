"""Quality-throughput Pareto: real LongBench F1 (quality) vs real offloaded-KV TPOT
(throughput), at matched quality, vs FlexGen fetch-all. Answers Q3/Q4 honestly."""
import json, os, glob
import numpy as np

QB = "experiments/results/quality_vs_budget"
EC = "experiments/results/e2e_confirm"
CTX = 32768   # representative offload context

# TPOT (ms) at the chosen ctx for each scheme, by hot fraction h (measured)
tpot = {}
for h, f in [(0.25, "offload_decode_real.json"), (0.50, "offload_decode_h50.json"),
             (0.70, "offload_decode_h70.json")]:
    d = json.load(open(f"experiments/results/{f}"))["by_ctx"][str(CTX)]
    tpot[h] = d
fetch_all = tpot[0.25]["fetch_all"]          # full-quality FlexGen baseline (h-independent)


def f1_at(ds, budget):
    for base, pat in [(QB, f"{ds}_xqp_b{budget:.2f}.json"), (EC, f"{ds}_xqp.json")]:
        p = f"{base}/{pat}"
        if os.path.exists(p):
            return json.load(open(p))["f1_mean"]
    return None


def tpot_at(h, scheme):
    # nearest measured h
    hk = min(tpot, key=lambda k: abs(k - h))
    return tpot[hk][scheme]


for ds in ["narrativeqa", "qasper"]:
    full = json.load(open(f"{EC}/{ds}_full.json"))["f1_mean"]
    print(f"\n=== {ds} (ctx {CTX//1024}K) — full F1={full:.3f}, FlexGen fetch-all TPOT={fetch_all:.0f}ms ===")
    print(f"{'budget(h)':>9} {'F1':>6} {'%full':>6} {'TPOT_prefetch':>14} {'vs fetchall':>12} {'vs reactive':>12}")
    for b in [0.20, 0.30, 0.40, 0.50, 0.70]:
        f1 = f1_at(ds, b)
        if f1 is None:
            continue
        tp = tpot_at(b, "prefetch_hot"); tr = tpot_at(b, "reactive_hot")
        print(f"{b:>9.2f} {f1:>6.3f} {100*f1/full:>5.0f}% {tp:>12.0f}ms "
              f"{-100*(1-tp/fetch_all):>11.0f}% {-100*(1-tp/tr):>11.0f}%")
    # matched-quality point: smallest h with F1 >= 0.9*full
    best = None
    for b in [0.20, 0.30, 0.40, 0.50, 0.70]:
        f1 = f1_at(ds, b)
        if f1 and f1 >= 0.9 * full:
            best = (b, f1, tpot_at(b, "prefetch_hot")); break
    if best:
        b, f1, tp = best
        print(f"  -> matched-quality (>=90% full) at h={b}: F1={f1:.3f}, prefetch TPOT={tp:.0f}ms "
              f"= {-100*(1-tp/fetch_all):.0f}% vs FlexGen fetch-all ({fetch_all:.0f}ms)")
