"""Decisive test for the paper's "query--key is near-irrelevant" claim.

The main traces use a *weak* Quest proxy: cosine of the head-mean query against
the token-mean, head-mean key of a block (xqp/features.py, attn_trace_extract.py).
That discards exactly what Quest relies on: the per-token, per-head MAX over a
page. If the claim "query--key is near-useless at block granularity" is to be
trusted, it must survive a FAITHFUL Quest signal.

This script re-extracts, on a small subset, three query signals per block and
compares their single-view AUC + their marginal AUC over within+cross:
  * f_query          cosine(mean-head q, mean-token mean-head K)   [current/weak]
  * f_query_dotmean  mean over (heads,tokens) of raw q.k            [dot, mean-pool]
  * f_query_dotmax   MAX  over (heads,tokens) of raw q.k            [faithful Quest]
Reading: if dotmax is still ~chance, the claim holds and is now bulletproof; if
it jumps, the claim was a proxy artifact and the wording must be revised.

    python experiments/run_quest_baseline.py --device cuda:0 --n 8 \
        --out experiments/results/quest_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "SEER"))

import numpy as np
from scipy.stats import rankdata

from xqp.attn_trace_extract import extract_attention_traces
from xqp.predictor import ClosedFormXQP

ZOO = "/public/model_zoo"
DEFAULT_MODELS = [("Llama-3.1-8B-Instruct", f"{ZOO}/Llama-3.1-8B-Instruct"),
                  ("Qwen2.5-7B-Instruct", f"{ZOO}/Qwen2.5-7B-Instruct")]
COLS = ["f_within", "f_cross", "f_query", "f_query_dotmean", "f_query_dotmax", "y_h4", "request_id"]


def auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_rows(path):
    cols = {c: [] for c in COLS}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for c in COLS:
                cols[c].append(r.get(c, 0.0))
    out = {c: np.asarray(cols[c], np.float32) for c in COLS if c != "request_id"}
    # request id -> integer code
    rid_map = {}
    out["rid"] = np.asarray([rid_map.setdefault(v, len(rid_map)) for v in cols["request_id"]], np.int32)
    return out


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def marginal(d, extra_col, seed=0):
    """Request-level split; AUC of closed-form on within+cross with vs without
    an extra (z-scored) query column."""
    uniq = np.unique(d["rid"]); rng = np.random.default_rng(seed)
    te = set(rng.permutation(uniq)[: max(1, len(uniq) // 4)].tolist())
    is_te = np.isin(d["rid"], list(te))
    tr, ev = ~is_te, is_te
    y = d["y_h4"].astype(np.float32)
    z = np.zeros_like(d["f_within"])
    # ClosedFormXQP is hardcoded to 4 features; pad unused slots with zeros
    # (they receive ~0 weight). base = within+cross; aug = +query variant.
    base = np.stack([d["f_within"], d["f_cross"], z, z], 1)
    base_auc = auc(y[ev], ClosedFormXQP.from_fit(base[tr], y[tr]).score(base[ev]))
    aug = np.stack([d["f_within"], d["f_cross"], zscore(d[extra_col]), z], 1)
    aug_auc = auc(y[ev], ClosedFormXQP.from_fit(aug[tr], y[tr]).score(aug[ev]))
    return base_auc, aug_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=8, help="prompts per model")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tmpdir", default="/public/xqp_traces/quest_probe")
    ap.add_argument("--block-size", type=int, default=32,
                    help="KV block size; 1 = token granularity (the regime where "
                         "per-token query might be complementary)")
    ap.add_argument("--out", default="experiments/results/quest_baseline.json")
    a = ap.parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.makedirs(a.tmpdir, exist_ok=True)

    from seer.trace.datasets import load_prompts
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    models = DEFAULT_MODELS if not a.models else [(os.path.basename(m), m) for m in a.models]
    prompts = load_prompts("mooncake", [a.max_context], a.n, tokenizer=None)
    results = {}
    for stem, mp in models:
        print(f"[quest] {stem}: extracting {a.n} prompts w/ query variants ...", flush=True)
        tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
        ids = []
        for p in prompts:
            t = tok(p, return_tensors="pt").input_ids[:, : a.max_context].to(a.device)
            ids.append(t)
        model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.float16, attn_implementation="eager",
            output_attentions=True, local_files_only=True).to(a.device).eval()
        outp = os.path.join(a.tmpdir, f"{stem}.quest.jsonl")
        extract_attention_traces(mp, None, outp, model=model, tokenizer=tok, input_ids=ids,
                                 device=a.device, block_size=a.block_size,
                                 max_new_tokens=a.max_new_tokens, query_variants=True)
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
        d = load_rows(outp)
        y = d["y_h4"]
        sv = {c: auc(y, d[c]) for c in ["f_within", "f_cross", "f_query",
                                        "f_query_dotmean", "f_query_dotmax"]}
        marg = {}
        for col in ["f_query", "f_query_dotmean", "f_query_dotmax"]:
            b, g = marginal(d, col)
            marg[col] = dict(within_cross=b, plus_query=g, delta=g - b)
        results[stem] = dict(n_rows=int(y.size), n_requests=int(d["rid"].max() + 1),
                             pos_rate=float(y.mean()), single_view_auc=sv, marginal=marg)
        print(f"  single-view AUC: within={sv['f_within']:.3f} cross={sv['f_cross']:.3f} | "
              f"query(cos)={sv['f_query']:.3f} dotmean={sv['f_query_dotmean']:.3f} "
              f"dotmax(faithful Quest)={sv['f_query_dotmax']:.3f}", flush=True)
        print(f"  marginal over within+cross: +cos {marg['f_query']['delta']:+.4f}  "
              f"+dotmean {marg['f_query_dotmean']['delta']:+.4f}  "
              f"+dotmax {marg['f_query_dotmax']['delta']:+.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(config=dict(n=a.n, max_new_tokens=a.max_new_tokens), per_model=results),
              open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
