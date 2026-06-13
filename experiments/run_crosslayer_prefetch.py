"""Direction 2 — cross-layer speculative prefetch (InfiniGen-done-right), a
SYSTEMS-value test that does not contradict "query is AUC-redundant within a layer".

Setup. While computing layer L-1 we already have its attention and can compute the
upcoming query q_L (from L-1's output). If we can predict layer L's hot set now, we
can prefetch those KV blocks from the slow tier before layer L needs them — saving
latency, not AUC. The question is whether the query view, redundant for *within*-
layer ranking, is COMPLEMENTARY for *cross*-layer prediction: does it catch the
blocks that pure prev-layer prefetch (InfiniGen's 60-80%-overlap heuristic) misses?

Metrics (pooled over requests x steps x consecutive layer pairs), all on existing
traces (CPU, no new extraction):
  1. prev-layer prefetch recall: recall of layer L's top-r hot set captured by
     layer L-1's top-r set at equal budget (the InfiniGen premise number).
  2. newly-hot detection: among blocks NOT in L-1's top-r, AUC of the upcoming
     query q_L (vs the prev-layer residual within) at identifying which become hot
     in layer L. >0.5 means query carries cross-layer prefetch signal.
  3. combined recall: prefetch (prev-top-r plus query-picked extras) vs prev-top-r
     alone at the SAME budget. A lift means query buys prefetch coverage.

    python experiments/run_crosslayer_prefetch.py --trace experiments/traces/Llama-3.1-8B-Instruct.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import rankdata


def auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def topr(x, r):
    n = x.shape[0]; k = max(1, int(np.ceil(r * n)))
    m = np.zeros(n, bool); m[np.argpartition(-x, kth=min(k - 1, n - 1))[:k]] = True
    return m


def load(path, step_stride, max_rows):
    """Read (rid, layer, step, block, within, query) keeping block_idx; subsample
    by step (every `step_stride`-th step) to bound memory."""
    rid = -1
    L, S, B, W, Q = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["layer"] == 0 and r["step"] == 0 and r["block_idx"] == 0:
                rid += 1
            if (r["step"] % step_stride) != 0:
                continue
            w = r["f_within"]
            if w != w:                      # drop fp16 NaN within
                continue
            L.append(r["layer"]); S.append(rid * 100000 + r["step"]); B.append(r["block_idx"])
            W.append(w); Q.append(r["f_query"])
            if len(L) >= max_rows:
                break
    return (np.asarray(L, np.int16), np.asarray(S, np.int64), np.asarray(B, np.int32),
            np.asarray(W, np.float32), np.asarray(Q, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="experiments/traces/Llama-3.1-8B-Instruct.jsonl")
    ap.add_argument("--r", type=float, default=0.10)
    ap.add_argument("--step-stride", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--out", default="experiments/results/crosslayer_prefetch.json")
    a = ap.parse_args()

    print(f"[load] {a.trace} (every {a.step_stride}th step) ...", flush=True)
    L, S, B, W, Q = load(a.trace, a.step_stride, a.max_rows)
    print(f"  rows={L.shape[0]:,}", flush=True)

    order = np.lexsort((B, L, S))
    L, S, B, W, Q = L[order], S[order], B[order], W[order], Q[order]
    bounds = np.flatnonzero(np.diff(S)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [S.shape[0]]])

    prev_recall, newly_q_auc, newly_w_auc = [], [], []
    base_recall, aug_recall = [], []
    n_groups = 0
    for s0, e0 in zip(starts, ends):
        ll, bb, ww, qq = L[s0:e0], B[s0:e0], W[s0:e0], Q[s0:e0]
        layers = np.unique(ll)
        if layers.size < 2:
            continue
        nb = int(bb.max()) + 1
        wmat = np.full((int(layers.max()) + 1, nb), np.nan, np.float32)
        qmat = np.full_like(wmat, np.nan)
        wmat[ll, bb] = ww; qmat[ll, bb] = qq
        n_groups += 1
        for li in range(1, int(layers.max()) + 1):
            wp, wc, qc = wmat[li - 1], wmat[li], qmat[li]
            ok = np.isfinite(wp) & np.isfinite(wc) & np.isfinite(qc)
            if ok.sum() < 8:
                continue
            wp, wc, qc = wp[ok], wc[ok], qc[ok]
            hot_prev = topr(wp, a.r); hot_cur = topr(wc, a.r)
            # 1. prev-layer prefetch recall
            if hot_cur.sum():
                prev_recall.append((hot_cur & hot_prev).sum() / hot_cur.sum())
            # 2. newly-hot detection among NOT-prev-hot candidates
            cand = ~hot_prev
            if cand.sum() >= 8 and 0 < hot_cur[cand].sum() < cand.sum():
                newly = hot_cur[cand].astype(np.float32)
                newly_q_auc.append(auc(newly, qc[cand]))     # upcoming query
                newly_w_auc.append(auc(newly, wp[cand]))     # prev-layer residual
                # 3. spend a 50%-extra prefetch budget on query-picks vs on more
                #    prev-within picks; keep ALL prev-hot. Which extra helps more?
                K = int(hot_prev.sum())
                extra = max(1, int(0.5 * K))
                ci = np.flatnonzero(cand)
                q_extra = ci[np.argsort(-qc[cand])][:extra]          # extras by query
                w_extra = ci[np.argsort(-wp[cand])][:extra]          # extras by prev-within
                if hot_cur.sum():
                    aug_q = hot_prev.copy(); aug_q[q_extra] = True
                    aug_w = hot_prev.copy(); aug_w[w_extra] = True
                    base_recall.append((hot_cur & aug_w).sum() / hot_cur.sum())   # extra=prev-within
                    aug_recall.append((hot_cur & aug_q).sum() / hot_cur.sum())    # extra=query

    def mean(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    out = dict(trace=os.path.basename(a.trace), r=a.r, n_groups=n_groups,
               n_layer_pairs=len(prev_recall),
               prev_layer_prefetch_recall=mean(prev_recall),
               newly_hot_auc_query=mean(newly_q_auc),
               newly_hot_auc_prev_within=mean(newly_w_auc),
               combined_recall_base=mean(base_recall),
               combined_recall_query_aug=mean(aug_recall),
               combined_recall_lift=mean(aug_recall) - mean(base_recall))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nReading: (1) prev-layer recall = the InfiniGen premise; "
          "(2) newly_hot_auc_query > 0.5 and > prev_within => query carries "
          "cross-layer signal the prev layer lacks; (3) combined lift > 0 => "
          "query buys real prefetch coverage at fixed budget.")


if __name__ == "__main__":
    main()
