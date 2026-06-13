"""Observability de-risk for the cross-layer query-free prefetch module.

The module predicts layer l's hot blocks from layer l-1's attention. But a real
selective-fetch system only OBSERVES attention over the blocks it fetched -> errors
compound across layers (miss a block -> don't fetch -> don't observe -> miss again).
The 87.5% recall was measured under FULL attention; this asks whether it survives
the closed loop.

Three regimes (per (request, decode step), over the 32-layer stack):
  open_loop      : predict from the FULL prev-layer attention (ceiling ~0.875).
  closed_noref   : observe only the fetched set; explore via a recency window; NO refresh.
  closed_refK    : as closed, but every K layers fetch-all (observe full attention) to refresh.

Recall_l = | fetch_set_l  ∩  true_hot_l | / |true_hot_l|, true_hot = top-r of FULL attn.
fetch budget = top-(h) ; hot = top-(r) ; recency window = w (always fetched).
"""
import argparse, json
import numpy as np


def topk_set(arr, k):
    k = min(max(1, k), len(arr))
    return set(np.argpartition(-arr, k - 1)[:k].tolist())


def run_group(A, r=0.10, h=0.25, w_frac=0.10, refreshK=None):
    """A: [L, N] attention. Returns per-layer recall list (layers 1..L-1)."""
    L, N = A.shape
    kr = max(1, int(r * N)); kh = max(1, int(h * N)); w = max(1, int(w_frac * N))
    recent = set(range(N - w, N))                      # recency window (last positions)
    hot = [topk_set(A[l], kr) for l in range(L)]
    fetch = set(range(N))                              # layer 0: bootstrap full
    recs = []
    for l in range(1, L):
        full_obs = (refreshK == "open") or (isinstance(refreshK, int) and l % refreshK == 0)
        if full_obs:
            obs = A[l - 1].copy()                      # full prev attention (open / refresh layer)
        else:
            obs = np.full(N, -1e30)                    # observe only what we fetched at l-1
            idx = list(fetch); obs[idx] = A[l - 1][idx]
        pred = topk_set(obs, kh - w)                   # predicted-hot (from observed prev attn)
        fetch = pred | recent                          # + recency exploration
        recs.append(len(fetch & hot[l]) / len(hot[l]))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="experiments/traces/Llama-3.1-8B-Instruct.jsonl")
    ap.add_argument("--max_req", type=int, default=24)
    ap.add_argument("--min_blocks", type=int, default=50)
    ap.add_argument("--out", default="experiments/results/observability_derisk.json")
    a = ap.parse_args()

    regimes = {"open_loop": "open", "closed_noref": None,
               "closed_ref8": 8, "closed_ref4": 4, "closed_ref2": 2}
    agg = {k: [] for k in regimes}
    n_groups = 0
    cur_req, rows = None, []

    def process(rows):
        nonlocal n_groups
        # rows -> {step: {layer: {blk: fwithin}}}
        steps = {}
        Lmax = 0
        for rr in rows:
            steps.setdefault(rr["step"], {}).setdefault(rr["layer"], {})[rr["block_idx"]] = rr["f_within"]
            Lmax = max(Lmax, rr["layer"])
        L = Lmax + 1
        for st, layers in steps.items():
            if len(layers) < L:
                continue
            blocks = sorted({b for lay in layers.values() for b in lay})
            if len(blocks) < a.min_blocks:
                continue
            bi = {b: i for i, b in enumerate(blocks)}
            A = np.zeros((L, len(blocks)), np.float32)
            for lyr, d in layers.items():
                for b, v in d.items():
                    A[lyr, bi[b]] = v
            for name, K in regimes.items():
                agg[name].extend(run_group(A, refreshK=K))
            n_groups += 1

    with open(a.trace) as f:
        for line in f:
            rr = json.loads(line)
            if rr["request_id"] != cur_req:
                if rows:
                    process(rows)
                rows = []
                cur_req = rr["request_id"]
                if n_groups >= a.max_req * 50:           # enough groups collected
                    break
            rows.append(rr)
    if rows and n_groups < a.max_req * 50:
        process(rows)

    out = {"n_groups": n_groups, "params": {"r": 0.10, "h": 0.25, "w": 0.10}, "regimes": {}}
    print(f"groups={n_groups}\n{'regime':14s} mean_recall  recall@last_layer  (lower=worse)")
    for name in regimes:
        arr = np.array(agg[name], np.float32)
        out["regimes"][name] = {"mean_recall": round(float(arr.mean()), 4),
                                "n": len(arr)}
        print(f"{name:14s} {arr.mean():.3f}")
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
