"""Verify request_id integrity, completeness, and leakage-safety of XQP traces.

A request-level (group) train/test split is leakage-safe only if request_id
correctly partitions rows into per-prompt groups. The historical bug wrote
request_id="p0" for EVERY row (one degenerate group) -> a row-random split that
leaks. A crash mid-collection leaves a TRUNCATED file (fewer prompts, or a
half-written last prompt) that is internally consistent but incomplete.

Per file this checks:
  1. no rows missing request_id / step / layer / block_idx        (malformed)
  2. NOT the all-p0 collapse  (fail iff boundaries>1 AND distinct==1)
  3. distinct request_ids == #prompt boundaries                    (1 boundary
     = row with step==0 & layer==0 & block_idx==0, one per prompt)
  4. each request_id is a single contiguous run                    (no interleave)
  5. request_ids are exactly p0..p{n-1} in order                   (well-formed)
  6. completeness: every request has exactly 1 boundary, at least one step>0
     row (real decode, not just prefill), and max(step) equal to the file's
     global max step (a truncated last prompt has a short max step)
  7. [--expect N] distinct request_ids == N                        (full corpus)

Single-prompt files (distinct==boundaries==1) legitimately PASS.

    python scripts/verify_trace_splits.py experiments/traces/*.jsonl
    python scripts/verify_trace_splits.py --expect 256 experiments/traces/foo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def verify(path: str, expect: int | None) -> tuple[bool, str]:
    n_rows = 0
    boundaries = 0
    order: list[str] = []          # request_ids in first-seen order
    seen: set[str] = set()
    runs: dict[str, int] = {}      # request_id -> contiguous run count
    rows_per: dict[str, int] = {}  # request_id -> row count
    maxstep: dict[str, int] = {}   # request_id -> max step seen
    bnds_per: dict[str, int] = {}  # request_id -> boundary rows
    prev: str | None = None
    malformed = 0
    null_rid = 0

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r.get("request_id")
            step, layer, blk = r.get("step"), r.get("layer"), r.get("block_idx")
            if rid is None:
                null_rid += 1
            if not isinstance(step, int) or not isinstance(layer, int) or not isinstance(blk, int):
                malformed += 1
                n_rows += 1
                continue
            if step == 0 and layer == 0 and blk == 0:
                boundaries += 1
                bnds_per[rid] = bnds_per.get(rid, 0) + 1
            if rid != prev:
                runs[rid] = runs.get(rid, 0) + 1
                if rid not in seen:
                    seen.add(rid)
                    order.append(rid)
                prev = rid
            rows_per[rid] = rows_per.get(rid, 0) + 1
            maxstep[rid] = max(maxstep.get(rid, 0), step)
            n_rows += 1

    distinct = len(seen)
    interleaved = [rid for rid, c in runs.items() if c > 1]
    expected_ids = [f"p{i}" for i in range(distinct)]
    wellformed = order == expected_ids
    global_max_step = max(maxstep.values()) if maxstep else 0
    truncated = [rid for rid, ms in maxstep.items() if ms < global_max_step]
    no_decode = [rid for rid, ms in maxstep.items() if ms == 0]
    multi_boundary = [rid for rid, c in bnds_per.items() if c != 1]

    checks = {
        "no malformed rows": malformed == 0 and null_rid == 0,
        "not all-p0 collapse": not (boundaries > 1 and distinct == 1),
        "distinct==boundaries": distinct == boundaries,
        "no interleaved request_ids": len(interleaved) == 0,
        "request_ids p0..p{n-1} in order": wellformed,
        "one boundary per request": len(multi_boundary) == 0,
        "every request has decode (step>0)": len(no_decode) == 0,
        "no truncated request (short max-step)": len(truncated) == 0,
    }
    if expect is not None:
        checks[f"distinct==expected({expect})"] = distinct == expect

    ok = all(checks.values())
    detail = (
        f"rows={n_rows:>12,}  requests={distinct:>5}  boundaries={boundaries:>5}  "
        f"max_step={global_max_step:>4}  groupable={'YES' if ok else 'NO '}"
    )
    if not ok:
        fails = [k for k, v in checks.items() if not v]
        detail += "\n      FAIL: " + "; ".join(fails)
        if malformed or null_rid:
            detail += f"\n      malformed_rows={malformed} null_request_id={null_rid}"
        if interleaved:
            detail += f"\n      interleaved rids (first 5): {interleaved[:5]}"
        if truncated:
            detail += f"\n      truncated rids (max_step<{global_max_step}, first 5): {truncated[:5]}"
        if no_decode:
            detail += f"\n      no-decode rids (first 5): {no_decode[:5]}"
        if multi_boundary:
            detail += f"\n      bad-boundary rids (first 5): {multi_boundary[:5]}"
        if not wellformed and order:
            detail += f"\n      rid order head/tail: {order[:3]} ... {order[-3:]}"
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect", type=int, default=None,
                    help="required #distinct request_ids per file (fail if short)")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    if not args.files:
        ap.print_help()
        return 2
    all_ok = True
    for p in args.files:
        if not os.path.isfile(p):
            print(f"{os.path.basename(p):45s} MISSING")
            all_ok = False
            continue
        ok, detail = verify(p, args.expect)
        all_ok &= ok
        print(f"{os.path.basename(p):45s} {detail}")
    print()
    print("ALL CLEAN — request-level splits are leakage-safe and complete."
          if all_ok else
          "SOME FILES FAILED — recollect (FORCE=1) or fix with scripts/fix_trace_request_ids.py.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
