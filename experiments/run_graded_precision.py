"""B2 (slim) — does the offline->deployment gap reproduce on the QUANTIZATION axis?

Companion to the eviction transfer-gap (TRANSFER_GAP_RESULTS.md). Instead of
keep/evict, allocate graded PRECISION (bits) to KV blocks at a fixed average-bit
budget, graded by a block-importance score, and measure a deployment quality
PROXY: the fraction of the model's REAL served attention mass that lands in the
high-precision tier (preserving attention output == preserving quality; the
standard KV-compression fidelity proxy). A good allocator puts the high-precision
bits where the model actually attends NOW.

We compare, per (prompt, layer) decision, top-f-by-score -> high precision:
  * XQP        — the learned within+cross scorer probability
  * H2O        — accumulated within-attention (within_accum)
  * uniform    — random allocation (lower bound)
  * oracle     — top-f by served attention mass itself (upper bound / ceiling)

If XQP <= H2O (both far below oracle), the gap reproduces for graded precision:
the learned scorer does not allocate bits better than a trivial accumulator,
because (B1) it is uninformative (AUC 0.643) about the served-now target.

Input: a serving calib log with cols including attn_now, within_accum, xqp_score
(from `runner --policy guardkv --log-calib`, sim.py logs policy.last_scores).

    python experiments/run_graded_precision.py \
        --calib experiments/results/serving_calib/calib_graded.json
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
import numpy as np

# two-tier precision: top-f blocks at HI bits, rest at LO bits.
HI_BITS, LO_BITS = 8, 2
def avg_bits(f): return LO_BITS + (HI_BITS - LO_BITS) * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="experiments/results/serving_calib/calib_graded.json")
    ap.add_argument("--fracs", default="0.10,0.20,0.30")
    ap.add_argument("--out", default="experiments/results/graded_precision.json")
    a = ap.parse_args()

    d = json.load(open(a.calib))
    cols = d["cols"]; idx = {c: i for i, c in enumerate(cols)}
    need = ["prompt_id", "layer_pos", "attn_now", "within_accum", "xqp_score"]
    miss = [c for c in need if c not in idx]
    if miss:
        raise SystemExit(f"calib log missing cols {miss}; cols={cols}. "
                         f"Re-run `runner --policy guardkv --log-calib` with the XQP-score column.")
    rows = d["rows"]
    # group blocks by (prompt, layer) decision context
    groups = defaultdict(list)
    for r in rows:
        groups[(r[idx["prompt_id"]], r[idx["layer_pos"]])].append(r)

    scorers = {
        "XQP":     lambda g: np.array([x[idx["xqp_score"]] for x in g], float),
        "H2O":     lambda g: np.array([x[idx["within_accum"]] for x in g], float),
        "oracle":  lambda g: np.array([x[idx["attn_now"]] for x in g], float),
    }
    rng = np.random.default_rng(0)
    fracs = [float(x) for x in a.fracs.split(",")]

    out = {"hi_bits": HI_BITS, "lo_bits": LO_BITS, "n_groups": len(groups),
           "n_blocks": len(rows), "by_frac": {}}
    print(f"groups={len(groups)} blocks={len(rows)}")
    print(f"{'avg_bits':>9}{'frac_hi':>8}{'XQP':>9}{'H2O':>9}{'uniform':>9}{'oracle':>9}"
          f"{'XQP-H2O':>9}")
    for f in fracs:
        # served-attention-mass preserved at HIGH precision, averaged over decisions
        acc = defaultdict(list)
        for g in groups.values():
            a_now = np.array([x[idx["attn_now"]] for x in g], float)
            tot = a_now.sum()
            if tot <= 0 or len(g) < 3:
                continue
            k = max(1, int(round(f * len(g))))
            for name, sc in scorers.items():
                top = np.argsort(-sc(g))[:k]
                acc[name].append(a_now[top].sum() / tot)
            ru = rng.permutation(len(g))[:k]
            acc["uniform"].append(a_now[ru].sum() / tot)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        out["by_frac"][f"{f:.2f}"] = dict(avg_bits=avg_bits(f), **m)
        print(f"{avg_bits(f):>9.1f}{f:>8.2f}{m['XQP']:>9.3f}{m['H2O']:>9.3f}"
              f"{m['uniform']:>9.3f}{m['oracle']:>9.3f}{m['XQP']-m['H2O']:>+9.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)
    # verdict
    deltas = [out["by_frac"][k]["XQP"] - out["by_frac"][k]["H2O"] for k in out["by_frac"]]
    print("\nVERDICT: XQP-graded vs H2O-tiered (served-attention preserved at hi-precision):",
          f"mean delta = {np.mean(deltas):+.3f}",
          "-> gap reproduces (XQP no better than H2O)" if np.mean(deltas) <= 0.01
          else "-> XQP allocates better here")


if __name__ == "__main__":
    main()
