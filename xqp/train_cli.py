"""CLI: xqp-train — fit ClosedFormXQP from a JSONL trace file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .predictor import ClosedFormXQP
from .trace import load_trace
from .eval import roc_auc, topk_recall


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="JSONL trace from xqp.trace")
    p.add_argument("--horizon", choices=["h1", "h4", "h16", "h64"], default="h4")
    p.add_argument("--per-layer", action="store_true")
    p.add_argument("--out", required=True, help="Path to write predictor JSON")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rows = load_trace(args.trace)
    if not rows:
        print(f"Empty trace: {args.trace}", file=sys.stderr)
        return 1
    F = np.stack([rows["f_within"], rows["f_cross"], rows["f_query"], rows["f_pos"]], axis=1)
    y = rows[f"y_{args.horizon}"].astype(np.float32)
    layer_ids = rows["layer"].astype(np.int64)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(F.shape[0])
    n_val = int(args.val_frac * F.shape[0])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    predictor = ClosedFormXQP.from_fit(
        F[train_idx], y[train_idx], l2=1e-3,
        layer_ids=layer_ids[train_idx] if args.per_layer else None,
        per_layer=args.per_layer,
    )

    # Validate
    if args.per_layer:
        # BUGFIX (audit): old code did `F[val_idx][mask]` — mask was on
        # layer_ids[val_idx], but the second indexing applied it to the
        # already-sliced array. The shapes happened to match (both
        # len(val_idx)) so it ran, but composing val_idx[mask] is the
        # safer expression.
        s = np.zeros(val_idx.shape[0], dtype=np.float32)
        layer_ids_val = layer_ids[val_idx]
        for l in np.unique(layer_ids_val):
            mask = layer_ids_val == l
            s[mask] = predictor.score(F[val_idx[mask]], layer=int(l))
    else:
        s = predictor.score(F[val_idx])

    auc = roc_auc(y[val_idx], s)
    recall = topk_recall(y[val_idx], s, k_frac=0.10)
    print(json.dumps({
        "horizon": args.horizon,
        "per_layer": args.per_layer,
        "n_train": int(train_idx.shape[0]),
        "n_val": int(val_idx.shape[0]),
        "auc": auc,
        "top10_recall": recall,
        "weights": predictor.weights.tolist(),
        "bias": predictor.bias.tolist() if predictor.bias.ndim else float(predictor.bias),
    }, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    predictor.save(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
