#!/usr/bin/env python3
"""Collect real XQP JSONL traces on GPU via attn_trace_extract.

Uses models under /public/model_zoo (no HF download). Supports prompt-range
slicing and --worker-id for dual-GPU parallel runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEER_ROOT = ROOT.parent / "SEER"
if SEER_ROOT.is_dir():
    sys.path.insert(0, str(SEER_ROOT))

MODEL_ZOO = Path("/public/model_zoo")

# Writer-side disk self-defense: abort rather than fill the volume holding the
# output (the home/root partition has ~47G; a real run needs hundreds of GB).
# Floor in GiB; override via XQP_MIN_FREE_GB. This guards the WRITER itself so a
# wrong --out-dir / TRACEDIR can never silently fill a disk mid-run.
_MIN_FREE_GB = float(os.environ.get("XQP_MIN_FREE_GB", "15"))


def _free_gb(path) -> float:
    return shutil.disk_usage(str(path)).free / 2**30

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("Llama-3.1-8B-Instruct", str(MODEL_ZOO / "Llama-3.1-8B-Instruct")),
    ("Qwen2.5-7B-Instruct", str(MODEL_ZOO / "Qwen2.5-7B-Instruct")),
    ("Qwen3-8B", str(MODEL_ZOO / "Qwen3-8B")),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status_path(worker_id: str | None) -> Path:
    logdir = ROOT / "experiments" / "logs"
    if worker_id:
        return logdir / f"status.{worker_id}.json"
    return logdir / "status.json"


def _write_status(payload: dict, worker_id: str | None) -> None:
    p = _status_path(worker_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")


def _load_prompts(workload: str, n: int, max_context: int) -> list[str]:
    try:
        from seer.trace.datasets import load_prompts

        return load_prompts(workload, [max_context], n, tokenizer=None)
    except Exception as e:
        print(f"[warn] SEER prompts failed ({e}); using ruler synthetic", file=sys.stderr)
        filler = "The quick brown fox jumps over the lazy dog. "
        prompts = []
        repeats = max(1, max_context // 10)
        haystack = filler * repeats
        for i in range(n):
            needle = f"\nThe secret password is {7919 * (i + 1)}.\n"
            mid = len(haystack) // 2
            prompts.append(
                haystack[:mid] + needle + haystack[mid:]
                + "\n\nQ: What is the secret password? Reply with only the number.\nA:"
            )
        return prompts


def _resolve_models(raw: list[str] | None) -> list[tuple[str, str]]:
    if not raw:
        return DEFAULT_MODELS
    out: list[tuple[str, str]] = []
    for item in raw:
        p = Path(item)
        if p.is_dir():
            out.append((p.name, str(p)))
        elif ":" in item:
            stem, path = item.split(":", 1)
            out.append((stem, path))
        else:
            guess = MODEL_ZOO / item.split("/")[-1]
            out.append((item.split("/")[-1], str(guess)))
    return out


def _collect_model(
    stem: str,
    model_path: str,
    trimmed: list[str],
    *,
    prompt_offset: int,
    out_path: Path,
    device: str,
    block_size: int,
    max_new_tokens: int,
    worker_id: str | None,
    status: dict,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xqp.attn_trace_extract import extract_attention_traces

    model_status = status["models"].setdefault(stem, {})
    model_status.update({
        "path": model_path,
        "out": str(out_path),
        "device": device,
        "worker_id": worker_id or "main",
        "status": "running",
        "done_prompts": 0,
        "total_prompts": len(trimmed),
        "prompt_offset": prompt_offset,
        "rows": 0,
        "started_at": _now(),
    })
    status["updated_at"] = _now()
    _write_status(status, worker_id)

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f"[collect] {stem} on {device} worker={worker_id} "
          f"prompts {prompt_offset}+{len(trimmed)} -> {out_path}", flush=True)

    t0 = time.time()
    total_rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Load model once per worker (avoid re-loading shards every prompt).
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        output_attentions=True,
        local_files_only=True,
    ).to(device).eval()

    id_list = []
    for p in trimmed:
        ids = tok.encode(p)
        ids_t = tok(p, return_tensors="pt").input_ids.to(device)
        id_list.append(ids_t)

    for pi, ids_t in enumerate(id_list):
        if _free_gb(out_path.parent) < _MIN_FREE_GB:
            raise SystemExit(
                f"[disk] free space on {out_path.parent} dropped below "
                f"{_MIN_FREE_GB} GiB at prompt {pi}/{len(id_list)}; aborting to "
                f"protect the volume (partial file left; rerun to resume).")
        tmp = out_path.parent / f".{stem}.{worker_id or 'main'}.p{pi}.jsonl"
        try:
            n_rows = extract_attention_traces(
                model_path,
                None,
                str(tmp),
                model=model,
                tokenizer=tok,
                input_ids=[ids_t],
                device=device,
                block_size=block_size,
                max_new_tokens=max_new_tokens,
                request_id_start=prompt_offset + pi,
            )
            with open(out_path, "a") as out_f, open(tmp) as in_f:
                out_f.write(in_f.read())
            total_rows += n_rows
        finally:
            tmp.unlink(missing_ok=True)

        model_status["done_prompts"] = pi + 1
        model_status["rows"] = total_rows
        model_status["elapsed_s"] = round(time.time() - t0, 1)
        done = pi + 1
        model_status["eta_s"] = round(
            (time.time() - t0) / done * (len(trimmed) - done), 1
        ) if done < len(trimmed) else 0
        status["updated_at"] = _now()
        _write_status(status, worker_id)

        if done % 5 == 0 or done == len(trimmed):
            print(json.dumps({
                "model": stem, "worker": worker_id, "device": device,
                "prompt": prompt_offset + done, "slice_done": done,
                "slice_total": len(trimmed), "rows": total_rows,
                "elapsed_s": round(time.time() - t0, 1),
            }), flush=True)

    elapsed = time.time() - t0
    model_status["status"] = "done"
    model_status["seconds"] = round(elapsed, 1)
    model_status["eta_s"] = 0
    status["updated_at"] = _now()
    _write_status(status, worker_id)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "rows": total_rows,
        "seconds": round(elapsed, 1),
        "out": str(out_path),
        "path": model_path,
        "device": device,
        "worker_id": worker_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traces", type=int, default=200)
    ap.add_argument("--prompt-start", type=int, default=0,
                    help="first prompt index (inclusive)")
    ap.add_argument("--prompt-end", type=int, default=-1,
                    help="last prompt index (exclusive); -1 = n_traces")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--workload", default="mooncake")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker-id", default=None,
                    help="tag for status file, e.g. gpu0 / gpu1")
    ap.add_argument("--out-suffix", default=None,
                    help="write to <stem>.<suffix>.jsonl instead of <stem>.jsonl")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <repo>/experiments/traces). "
                         "Use a large volume for big collections.")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    prompt_end = args.n_traces if args.prompt_end < 0 else args.prompt_end
    if not (0 <= args.prompt_start < prompt_end <= args.n_traces):
        raise SystemExit(
            f"invalid prompt range [{args.prompt_start}, {prompt_end}) "
            f"for n_traces={args.n_traces}"
        )

    models = _resolve_models(args.models)
    for _, path in models:
        if not Path(path).is_dir():
            raise SystemExit(f"model path missing: {path}")

    outdir = Path(args.out_dir) if args.out_dir else (ROOT / "experiments" / "traces")
    outdir.mkdir(parents=True, exist_ok=True)
    free = _free_gb(outdir)
    if free < _MIN_FREE_GB:
        raise SystemExit(
            f"[disk] only {free:.1f} GiB free on the volume holding {outdir} "
            f"(< {_MIN_FREE_GB} GiB floor); refusing to start. Point --out-dir at a "
            f"larger volume (e.g. /public) or free space.")
    print(f"[disk] {free:.1f} GiB free on {outdir} (floor {_MIN_FREE_GB})", flush=True)

    all_prompts = _load_prompts(args.workload, args.n_traces, args.max_context)
    prompts = all_prompts[args.prompt_start:prompt_end]

    from transformers import AutoTokenizer

    status = {
        "phase": "collect",
        "updated_at": _now(),
        "n_traces": args.n_traces,
        "prompt_range": [args.prompt_start, prompt_end],
        "max_context": args.max_context,
        "workload": args.workload,
        "device": args.device,
        "worker_id": args.worker_id,
        "models": {},
    }
    _write_status(status, args.worker_id)

    summary = {}
    for stem, model_path in models:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        trimmed = []
        for p in prompts:
            ids = tok.encode(p)
            if len(ids) > args.max_context:
                ids = ids[: args.max_context]
            trimmed.append(tok.decode(ids, skip_special_tokens=True))

        suffix = f".{args.out_suffix}" if args.out_suffix else ""
        out_path = outdir / f"{stem}{suffix}.jsonl"

        summary[stem] = _collect_model(
            stem, model_path, trimmed,
            prompt_offset=args.prompt_start,
            out_path=out_path,
            device=args.device,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            worker_id=args.worker_id,
            status=status,
        )
        print(json.dumps({"model": stem, "done": True, **summary[stem]}), flush=True)

    status["phase"] = "collect_done"
    status["updated_at"] = _now()
    status["summary"] = summary
    _write_status(status, args.worker_id)
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
