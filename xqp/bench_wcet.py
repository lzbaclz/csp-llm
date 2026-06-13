"""xqp.bench_wcet — CPU WCET sanity over trained predictors.

`scripts/run_benchmarks.sh` calls this for the `e2` step. The real P99.9 WCET
must come from the TensorRT-10 + CUDA-Graph path on ga100; this CPU benchmark
is a sanity check (it confirms the closed form does far less work than the
TinyMLP), not the reported envelope. See experiments/e3_wcet_envelope.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from .predictor import ClosedFormXQP
from .eval import measure_wcet_cpu, synthetic_dataset


def main(argv=None):
    p = argparse.ArgumentParser(prog="xqp-bench-wcet")
    p.add_argument("--predictors", required=True, help="dir of predictor JSONs")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--replays", type=int, default=3000)
    a = p.parse_args(argv)

    # A representative feature batch — WCET is shape-bound, so synthetic is fine.
    F, _ = synthetic_dataset(n_blocks=max(a.batch, 64), n_steps=2, seed=0)
    Fb = F[:a.batch].astype(np.float32)

    results = {
        "batch": a.batch,
        "note": "CPU sanity only; real WCET from ga100 TRT+CUDA-Graph",
    }
    for pf in sorted(glob.glob(os.path.join(a.predictors, "*.json"))):
        name = Path(pf).stem
        try:
            pred = ClosedFormXQP.load(pf)
        except Exception as e:  # malformed predictor file
            results[name] = {"error": str(e)}
            continue
        if getattr(pred, "per_layer", False):
            results[name] = {"skipped": "per-layer; deploy uses the shared variant"}
            continue
        results[name] = measure_wcet_cpu(pred, Fb, n_replays=a.replays)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
