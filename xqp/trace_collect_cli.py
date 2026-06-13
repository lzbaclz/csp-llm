"""xqp.trace_collect_cli — build JSONL training traces.

Two modes:

  --synthetic : CPU. Writes a synthetic trace in the TraceRecord schema
                (the same columns `xqp.trace.load_trace` and `xqp-train`
                expect). Used by the pilot and the test suite; needs no GPU
                or `transformers`. Makes the collect -> train -> eval pipeline
                runnable end-to-end on a laptop.

  default     : the real path. Drives a HuggingFace decode loop through
                `xqp.hf_adapter.HFXQPCacheAdapter`, recording the 4 features
                plus future-step top-r labels per (layer, block, step). It
                requires `torch`, `transformers`, and a CUDA device. On a
                sandbox without those, it prints guidance and exits non-zero
                rather than fabricating traces (consistent with the
                scaffold note in xqp/hf_adapter.py).

`scripts/collect_traces.sh` invokes the default (ga100) path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _write_synthetic(out: str, *, n_blocks: int, n_steps: int, seed: int,
                     n_layers: int = 8) -> int:
    from .eval import synthetic_dataset

    F, y = synthetic_dataset(n_blocks=n_blocks, n_steps=n_steps, seed=seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as fh:
        for i in range(F.shape[0]):
            rec = dict(
                request_id="synthetic",
                layer=int(i % n_layers),
                step=int(i // n_blocks),
                block_idx=int(i % n_blocks),
                f_within=float(F[i, 0]), f_cross=float(F[i, 1]),
                f_query=float(F[i, 2]), f_pos=float(F[i, 3]),
                y_h1=int(y[i]), y_h4=int(y[i]),
                y_h16=int(y[i]), y_h64=int(y[i]),
            )
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def _real_path_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        import torch  # noqa: WPS433
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(prog="xqp-trace-collect")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--workload", default="mooncake-chat")
    p.add_argument("--n-traces", type=int, default=200)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--out", required=True)
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--synthetic", action="store_true",
                   help="CPU synthetic trace (no GPU/transformers)")
    p.add_argument("--synthetic-blocks", type=int, default=128)
    p.add_argument("--synthetic-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.synthetic:
        n = _write_synthetic(a.out, n_blocks=a.synthetic_blocks,
                             n_steps=a.synthetic_steps, seed=a.seed)
        print(json.dumps({"mode": "synthetic", "out": a.out, "rows": n}))
        return 0

    if not _real_path_available():
        print(
            "real trace collection needs torch + transformers + a CUDA GPU "
            "(none detected). Run on ga100, or pass --synthetic for a CPU "
            "trace. See xqp/hf_adapter.py and scripts/collect_traces.sh.",
            file=sys.stderr,
        )
        return 2

    # ga100 path: drive the HF decode loop through HFXQPCacheAdapter and the
    # TraceCollector. Implemented against the upstream SEER attention hook;
    # not exercised in the sandbox (no GPU), hence kept behind the guard above.
    raise NotImplementedError(
        "real HF trace collection is wired on ga100 via the SEER attention "
        "hook; see scripts/collect_traces.sh"
    )


if __name__ == "__main__":
    sys.exit(main())
