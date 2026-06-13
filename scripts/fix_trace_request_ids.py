"""Rewrite request_id in collected JSONL traces.

The trace harness historically wrote request_id="p0" for every prompt (each
prompt was extracted in its own call where the local index was always 0). The
emission order is deterministic (step-major, layer-minor, block-minor), so each
prompt's first row is uniquely (step==0, layer==0, block_idx==0). We assign
request_id = "p{cumulative boundary count}", making the on-disk files
self-describing. Idempotent. Streams line-by-line (low memory).

    python scripts/fix_trace_request_ids.py experiments/traces/*.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fix(path: str) -> tuple[int, int]:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    n_rows = 0
    rid = -1
    with open(p) as fin, open(tmp, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["step"] == 0 and r["layer"] == 0 and r["block_idx"] == 0:
                rid += 1
            r["request_id"] = f"p{rid}"
            fout.write(json.dumps(r) + "\n")
            n_rows += 1
    tmp.replace(p)
    return n_rows, rid + 1


if __name__ == "__main__":
    for path in sys.argv[1:]:
        n, nreq = fix(path)
        print(f"{path}: {n:,} rows, {nreq} requests")
