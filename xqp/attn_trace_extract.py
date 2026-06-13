"""Extract real attention-dynamics traces on a GPU (the ICDM data source).

This turns the A100 box into a data generator for the ICDM paper. For each
prompt it runs an incremental decode loop on a HuggingFace causal LM, captures
per-layer attention over the KV cache at every step, aggregates keys into
blocks, and writes the JSONL schema that `xqp.trace.load_trace`, `xqp-train`,
and the ICDM drivers consume.

The pure-NumPy aggregation core (`blockify`, `update_ema`, `labels_for_horizons`)
is unit-tested on CPU. The full driver is **also CPU-validatable**: pass a
preloaded tiny model (built with `attn_implementation="eager"`) plus `input_ids`
and `device="cpu"` to exercise the entire tensor path offline (see tests), so
the A100 run is de-risked. Production: pass `model_id` + `prompts`, `device="cuda"`.

Assumes a Llama/Qwen/Mistral-family module layout
(`model.model.layers[i].self_attn.q_proj`).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


# ----------------------------- testable core -------------------------------

def blockify(x: np.ndarray, block_size: int) -> np.ndarray:
    """Segment-mean over the key axis.

    x: (kv,) per-key attention weights -> (n_blocks,);
       (kv, d) per-key vectors          -> (n_blocks, d).
    The last (ragged) block is averaged over its actual members.
    """
    x = np.asarray(x, dtype=np.float32)
    kv = x.shape[0]
    if kv == 0:
        return x.reshape((0,) + x.shape[1:])
    n_blocks = int(np.ceil(kv / block_size))
    return np.stack([x[b * block_size:(b + 1) * block_size].mean(axis=0)
                     for b in range(n_blocks)], axis=0).astype(np.float32)


def blockify_max(x: np.ndarray, block_size: int) -> np.ndarray:
    """Like ``blockify`` but segment-MAX over the key axis (for the faithful
    Quest per-block signal: a block is hot if *any* token in it scores high)."""
    x = np.asarray(x, dtype=np.float32)
    kv = x.shape[0]
    if kv == 0:
        return x.reshape((0,) + x.shape[1:])
    n_blocks = int(np.ceil(kv / block_size))
    return np.stack([x[b * block_size:(b + 1) * block_size].max(axis=0)
                     for b in range(n_blocks)], axis=0).astype(np.float32)


def update_ema(prev, cur, decay: float = 0.9) -> np.ndarray:
    """EMA tolerant of a growing block count (new blocks seed at their value)."""
    cur = np.asarray(cur, dtype=np.float32)
    if prev is None:
        return cur.copy()
    if prev.shape[0] == cur.shape[0]:
        return (decay * prev + (1 - decay) * cur).astype(np.float32)
    out = cur.copy()
    m = min(prev.shape[0], cur.shape[0])
    out[:m] = decay * prev[:m] + (1 - decay) * cur[:m]
    return out.astype(np.float32)


def labels_for_horizons(block_attn_seq: list, t: int, n_blocks_t: int,
                        r_label: float, horizons=(1, 4, 16, 64)) -> dict:
    """Top-r labels for the blocks present at step t, read off future steps.

    block_attn_seq[s] is the (n_blocks_s,) per-block attention at step s; the
    blocks present at t are the first n_blocks_t entries of any later step.
    """
    from .features import topk_indicator
    out = {}
    T = len(block_attn_seq)
    for h in horizons:
        s = min(t + h, T - 1)
        fut = np.asarray(block_attn_seq[s], dtype=np.float32)[:n_blocks_t]
        out[h] = topk_indicator(fut, r_label).astype(np.int64)
    return out


# ----------------------------- GPU/driver ----------------------------------

def _gpu_available() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _layer_key(past, l):
    """Key tensor (batch, n_kv_heads, kv, head_dim) for layer l, across the
    transformers cache API versions (5.x DynamicCache.layers[l].keys; 4.x
    .key_cache[l]; legacy tuple-of-(k,v))."""
    layers = getattr(past, "layers", None)
    if layers is not None:
        return layers[l].keys
    kc = getattr(past, "key_cache", None)
    if kc is not None:
        return kc[l]
    return past[l][0]


def extract_attention_traces(model_id=None, prompts=None, out_path="traces.jsonl", *,
                             model=None, tokenizer=None, input_ids=None,
                             device="cuda", block_size: int = 32, r_label: float = 0.10,
                             horizons=(1, 4, 16, 64), max_new_tokens: int = 64,
                             dtype: str = "float16", ema_decay: float = 0.9,
                             request_id_start: int = 0,
                             query_variants: bool = False) -> int:
    """Write a JSONL trace; return the row count.

    Production (A100): ``extract_attention_traces(model_id, prompts, out, device="cuda")``.
    Offline validation: pass a preloaded ``model`` (+ ``input_ids`` or
    ``tokenizer``) with ``device="cpu"``.
    """
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        raise RuntimeError("extract_attention_traces needs torch + transformers")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' but no CUDA visible. Run on the A100 box, or pass "
            "device='cpu' with a small model for validation."
        )
    if len(horizons) != 4:
        raise ValueError("horizons must have exactly 4 entries (y_h1..y_h64 schema)")

    from .features import extract_features

    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype),
            attn_implementation="eager", output_attentions=True,
        )
    model = model.to(device).eval()

    if input_ids is not None:
        id_list = [t.to(device) for t in input_ids]
    else:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        id_list = [tokenizer(p, return_tensors="pt").input_ids.to(device) for p in prompts]

    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)

    rows_out: list = []
    for pi, ids in enumerate(id_list):
        q_capture: dict = {}
        hooks = []
        for li, layer in enumerate(model.model.layers):
            def _mk(idx):
                def _h(_m, _inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    q_capture[idx] = o[:, -1, :].detach().float().cpu().numpy()
                return _h
            hooks.append(layer.self_attn.q_proj.register_forward_hook(_mk(li)))

        # Prefill: request use_cache only. We do NOT ask for prefill attentions
        # -- materializing the O(L^2) prompt attention would OOM at long context
        # and we never use it (we only need the KV cache here; per-step decode
        # attention below is O(kv), cheap).
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values
        n_layers = getattr(model.config, "num_hidden_layers", None) or len(past)

        ema = [None] * n_layers
        block_attn_hist = [[] for _ in range(n_layers)]
        feat_rows = []
        last_used_cache: dict = {}

        next_id = ids[:, -1:]
        for t in range(max_new_tokens):
            with torch.no_grad():
                out = model(next_id, past_key_values=past, use_cache=True,
                            output_attentions=True)
            past = out.past_key_values
            for l in range(n_layers):
                attn_l = out.attentions[l][0].mean(0).squeeze(0).float().cpu().numpy()  # (kv,)
                block_attn = blockify(attn_l, block_size)
                block_attn_hist[l].append(block_attn)
                ema[l] = update_ema(ema[l], block_attn, ema_decay)
                k = _layer_key(past, l)[0].mean(0).float().cpu().numpy()                 # (kv, hd)
                K_block = blockify(k, block_size)
                q_prev = q_capture.get(l, np.zeros(n_heads * head_dim, dtype=np.float32))
                q_prev = q_prev.reshape(-1, head_dim).mean(0)
                nb = block_attn.shape[0]
                last_used = last_used_cache.get(l, np.zeros(0, dtype=np.float32))
                if last_used.shape[0] < nb:
                    last_used = np.concatenate(
                        [last_used, np.full(nb - last_used.shape[0], float(t), np.float32)])
                last_used_cache[l] = last_used
                F = extract_features(
                    ema_within=ema[l], ema_prev_layer=(ema[l - 1] if l > 0 else None),
                    K_layer=K_block, q_prev=q_prev, step=t, last_used=last_used,
                    r_cross=r_label,
                )
                qv = None
                if query_variants:
                    # Faithful Quest signal: raw dot product q.k per (query head,
                    # token), then per-block MAX over tokens & heads — the
                    # per-token/per-head tail that the mean-pooled cosine f_query
                    # discards. Also a raw-dot MEAN variant to isolate the
                    # cosine-vs-dot axis from the mean-vs-max axis.
                    K_full = _layer_key(past, l)[0].float().cpu().numpy()      # (n_kv, kv, hd)
                    n_kv = K_full.shape[0]
                    qh = q_capture.get(l, np.zeros(n_heads * head_dim, np.float32)).reshape(-1, head_dim)
                    grp = max(1, qh.shape[0] // max(1, n_kv))
                    kv_of_head = np.minimum(np.arange(qh.shape[0]) // grp, n_kv - 1)
                    dots = np.einsum("hkd,hd->hk", K_full[kv_of_head], qh)     # (n_heads, kv)
                    qmax = blockify_max(dots.max(0), block_size)              # per-token/head MAX
                    qmean = blockify(dots.mean(0), block_size)               # raw-dot MEAN
                    qv = (qmax, qmean)
                feat_rows.append((l, t, F, nb, qv))
            next_id = out.logits[:, -1:].argmax(-1)   # greedy

        for (l, t, F, nb, qv) in feat_rows:
            labels = labels_for_horizons(block_attn_hist[l], t, nb, r_label, horizons)
            for b in range(F.shape[0]):
                row = dict(
                    request_id=f"p{request_id_start + pi}", layer=int(l), step=int(t), block_idx=int(b),
                    f_within=float(F[b, 0]), f_cross=float(F[b, 1]),
                    f_query=float(F[b, 2]), f_pos=float(F[b, 3]),
                    y_h1=int(labels[horizons[0]][b]), y_h4=int(labels[horizons[1]][b]),
                    y_h16=int(labels[horizons[2]][b]), y_h64=int(labels[horizons[3]][b]),
                )
                if qv is not None:
                    row["f_query_dotmax"] = float(qv[0][b])
                    row["f_query_dotmean"] = float(qv[1][b])
                rows_out.append(row)
        for hook in hooks:
            hook.remove()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    return len(rows_out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="xqp-attn-trace")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.prompts:
        prompts = [ln for ln in Path(a.prompts).read_text().splitlines() if ln.strip()]
    else:
        prompts = ["Summarize the history of long-context language models."]
    n = extract_attention_traces(a.model, prompts, a.out, device=a.device,
                                 block_size=a.block_size, max_new_tokens=a.max_new_tokens)
    print(json.dumps({"out": a.out, "rows": n}))
