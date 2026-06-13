"""Stratified marginal value of the query view — the experiment that decides the
gated design.

The pooled finding is "query is near-useless". The gated design bets that this is
a *regime average*: query is a LEADING indicator that should help exactly where
the attention-magnitude views are BLIND — cold blocks (low within-EMA), at long
horizons. This driver measures the marginal AUC the query view adds over
within+cross, stratified by:
  * within-EMA tercile   (low = magnitude-blind / cold regime)
  * recency tercile      (f_pos; low = stale)
  * horizon              (h1 < h4 < h16 < h64)

Run on the current traces (mean-pooled query, field ``f_query``) now; rerun with
``--query-field f_query_dotmax`` once the faithful per-token/per-head-max signal
is extracted (extractor ``query_variants=True``). If the marginal concentrates in
the low-within / long-horizon strata, the gated design is warranted.

    python experiments/run_quest_stratified.py --traces experiments/traces \
        --query-field f_query --out experiments/results/quest_stratified.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import rankdata

from xqp.predictor import ClosedFormXQP

HORIZONS = ("h1", "h4", "h16", "h64")


def auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load(path, query_field):
    """Compact loader: within, cross, query(field), recency, step, layer, labels;
    request id recovered from (step==0,layer==0,block==0) boundaries."""
    layer, step, blk = [], [], []
    fw, fc, fq, fp = [], [], [], []
    ys = {h: [] for h in HORIZONS}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            layer.append(r["layer"]); step.append(r["step"]); blk.append(r["block_idx"])
            fw.append(r["f_within"]); fc.append(r["f_cross"])
            fq.append(r.get(query_field, r["f_query"])); fp.append(r["f_pos"])
            for h in HORIZONS:
                ys[h].append(r[f"y_{h}"])
    layer = np.asarray(layer, np.int16); step = np.asarray(step, np.int16)
    blk = np.asarray(blk, np.int32)
    rid = (np.cumsum((step == 0) & (layer == 0) & (blk == 0)) - 1).astype(np.int32)
    F = np.stack([np.asarray(fw, np.float32), np.asarray(fc, np.float32),
                  np.asarray(fq, np.float32), np.asarray(fp, np.float32)], 1)
    y = {h: np.asarray(ys[h], np.int8) for h in HORIZONS}
    finite = np.isfinite(F).all(1)
    return dict(F=F[finite], rid=rid[finite], step=step[finite],
                y={h: v[finite] for h, v in y.items()})


def req_split(rid, frac=0.25, seed=0):
    u = np.unique(rid); rng = np.random.default_rng(seed)
    te = set(rng.permutation(u)[: max(1, int(frac * len(u)))].tolist())
    m = np.isin(rid, list(te))
    return ~m, m


def marginal(Ftr, ytr, Fte, yte):
    """AUC(within+cross+query) - AUC(within+cross) on a test (sub)set."""
    m2 = np.array([1, 1, 0, 0], np.float32); m3 = np.array([1, 1, 1, 0], np.float32)
    s2 = ClosedFormXQP.from_fit(Ftr * m2, ytr).score(Fte * m2)
    s3 = ClosedFormXQP.from_fit(Ftr * m3, ytr).score(Fte * m3)
    return auc(yte, s2), auc(yte, s3)


def terciles(x):
    q = np.quantile(x, [1 / 3, 2 / 3])
    return np.digitize(x, q)   # 0=low,1=mid,2=high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--query-field", default="f_query")
    ap.add_argument("--cap", type=int, default=3_000_000, help="rows loaded per file cap")
    ap.add_argument("--out", default="experiments/results/quest_stratified.json")
    a = ap.parse_args()

    Fs, ys, rids, steps = [], {h: [] for h in HORIZONS}, [], []
    off = 0
    for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))):
        if ".smoke." in f:
            continue
        print(f"[load] {os.path.basename(f)}", flush=True)
        d = load(f, a.query_field)
        # cap per file for memory/time
        if d["F"].shape[0] > a.cap:
            idx = np.random.default_rng(0).choice(d["F"].shape[0], a.cap, replace=False)
            idx.sort()
            d = dict(F=d["F"][idx], rid=d["rid"][idx], step=d["step"][idx],
                     y={h: d["y"][h][idx] for h in HORIZONS})
        Fs.append(d["F"]); rids.append(d["rid"] + off); steps.append(d["step"])
        off += int(d["rid"].max() + 1)
        for h in HORIZONS:
            ys[h].append(d["y"][h])
    F = np.concatenate(Fs); rid = np.concatenate(rids); step = np.concatenate(steps)
    y = {h: np.concatenate(ys[h]) for h in HORIZONS}
    print(f"pooled rows={F.shape[0]:,} requests={int(rid.max()+1)} query_field={a.query_field}", flush=True)

    tr, te = req_split(rid)
    out = {"query_field": a.query_field, "n_rows": int(F.shape[0])}

    # ---- pooled marginal per horizon ----
    out["by_horizon"] = {}
    for h in HORIZONS:
        b, g = marginal(F[tr], y[h][tr].astype(np.float32), F[te], y[h][te].astype(np.float32))
        out["by_horizon"][h] = dict(within_cross=b, plus_query=g, delta=g - b)
        print(f"  horizon {h:4s}: within+cross={b:.4f} +query={g:.4f}  delta={g-b:+.4f}", flush=True)

    # ---- marginal by within-EMA tercile (the magnitude-blind regime), h4 ----
    h = "h4"
    ytr = y[h][tr].astype(np.float32)
    m2 = np.array([1, 1, 0, 0], np.float32); m3 = np.array([1, 1, 1, 0], np.float32)
    mdl2 = ClosedFormXQP.from_fit(F[tr] * m2, ytr)
    mdl3 = ClosedFormXQP.from_fit(F[tr] * m3, ytr)
    s2 = mdl2.score(F[te] * m2); s3 = mdl3.score(F[te] * m3); yte = y[h][te].astype(np.float32)
    for axis, vals in [("within_tercile", terciles(F[te, 0])),
                       ("recency_tercile", terciles(F[te, 3]))]:
        out[axis] = {}
        for t in (0, 1, 2):
            mk = vals == t
            if mk.sum() < 100 or 0 == yte[mk].sum() or yte[mk].sum() == mk.sum():
                continue
            b, g = auc(yte[mk], s2[mk]), auc(yte[mk], s3[mk])
            out[axis][["low", "mid", "high"][t]] = dict(
                n=int(mk.sum()), pos_rate=float(yte[mk].mean()),
                within_cross=b, plus_query=g, delta=g - b)
        lab = "within-EMA" if axis == "within_tercile" else "recency"
        print(f"  [{lab} tercile, h4] " + "  ".join(
            f"{k}:Δ{v['delta']:+.4f}" for k, v in out[axis].items()), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)
    print("\nDecision: if Δ(query) is materially >0 in the LOW within-EMA tercile "
          "and/or grows with horizon, the gated design is warranted.")


if __name__ == "__main__":
    main()
