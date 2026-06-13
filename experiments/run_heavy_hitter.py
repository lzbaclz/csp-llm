"""Heavy-hitter structure + salient-set stability of the attention stream (the
quantitative backing for paper section 3.2). Computes, per model:

  - Gini concentration of the per-block within-layer attention at each step
    (high Gini => a few blocks carry the mass; the heavy-hitter premise).
  - Salient-set Jaccard vs. horizon: within each (request, layer, step) group,
    Jaccard( {b: y_h1=1}, {b: y_h=1} ) for h in {1,4,16,64} — how fast the
    top-r set shifts as the target recedes (temporally stable but drifting).

    python experiments/run_heavy_hitter.py --traces experiments/traces \
        --out experiments/results/heavy_hitter.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import numpy as np

from run_icdm_full import load_model_trace, HORIZONS


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, np.float64))
    n = x.shape[0]
    if n == 0 or x.sum() <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def analyze(d: dict, max_groups: int = 40000, seed: int = 0) -> dict:
    rid, layer, step, F = d["rid"], d["layer"], d["step"], d["F"]
    y = {h: d["y"][h] for h in HORIZONS}
    # group key = (rid, layer, step); encode as a single int
    key = (rid.astype(np.int64) * 100000 + layer.astype(np.int64) * 1000 + step.astype(np.int64))
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    fw_s = F[order, 0]
    ys = {h: y[h][order] for h in HORIZONS}
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(key_s)]])
    rng = np.random.default_rng(seed)
    gi = list(range(len(starts)))
    if len(gi) > max_groups:
        gi = rng.choice(len(gi), max_groups, replace=False)
    ginis = []
    jacc = {h: [] for h in HORIZONS}
    for g in gi:
        s, e = starts[g], ends[g]
        if e - s < 4:
            continue
        ginis.append(gini(fw_s[s:e]))
        base = ys["h1"][s:e] > 0.5
        if base.sum() == 0:
            continue
        for h in HORIZONS:
            cur = ys[h][s:e] > 0.5
            union = (base | cur).sum()
            jacc[h].append(float((base & cur).sum()) / union if union else 1.0)
    return dict(
        n_groups=len(ginis),
        gini_mean=float(np.nanmean(ginis)), gini_median=float(np.nanmedian(ginis)),
        jaccard_vs_h1={h: float(np.mean(jacc[h])) for h in HORIZONS},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--out", default="experiments/results/heavy_hitter.json")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model stems (default: all)")
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl")))
             if ".smoke." not in f]
    out = {}
    for f in files:
        nm = os.path.basename(f)[:-len(".jsonl")]
        if a.models and nm not in a.models:
            continue
        print(f"[load] {nm}", flush=True)
        d = load_model_trace(f)
        out[nm] = analyze(d)
        print(f"  gini_mean={out[nm]['gini_mean']:.3f} "
              f"jaccard(h1,h4)={out[nm]['jaccard_vs_h1']['h4']:.3f} "
              f"jaccard(h1,h64)={out[nm]['jaccard_vs_h1']['h64']:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
