"""Multi-turn attention-trace extraction with a PERSISTENT KV cache (E5).

The single-turn extractor (`attn_trace_extract.extract_attention_traces`) prefills
one prompt and greedily decodes a short continuation — the mild-drift regime where
the adaptive-conformal layer looks idle. This module collects the regime KV
management actually targets: a **multi-turn conversation** whose KV cache persists
across turns, with a **continuous decode-step counter** over the whole conversation,
and a topic switch at every turn boundary. At a switch the newly-salient blocks (the
new user tokens + first generated tokens) have no within-EMA history, so a
within-dominated scorer underscores them ⇒ a missed-saliency spike a *fixed*
threshold cannot absorb but an adaptive one can. See `experiments/E5_DESIGN_*.md`.

Schema is a superset of the single-turn rows (so the existing loaders still work):
the usual `request_id/layer/step/block_idx/f_*/y_*` plus a `turn` field. One
conversation = one continuous `step` stream = one request under the existing
`(step==0,layer==0,block==0)` boundary recovery, so no loader change is needed.

The per-(layer,block,step) feature/label core is imported unchanged from
`attn_trace_extract`; only the decode driver differs. CPU-validatable on a tiny
eager Llama exactly like the single-turn extractor (see tests).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from .attn_trace_extract import (
    blockify, blockify_max, update_ema, labels_for_horizons, _layer_key,
)


def _strip_leading_bos(ids, bos_id):
    """Drop a leading BOS so an appended turn does not inject a mid-sequence BOS."""
    if bos_id is not None and ids.shape[1] > 0 and int(ids[0, 0]) == int(bos_id):
        return ids[:, 1:]
    return ids


def _render_turn_ids(tokenizer, user_text, turn_idx, device):
    """Token ids to *append* for a new user turn + assistant generation prompt.

    Turn 0 uses the full template (BOS + system wrapper). Later turns render the
    user turn standalone and strip the leading BOS, then append it to the persistent
    cache. We deliberately do NOT re-render the conversation history (which would
    require re-tokenising decoded assistant text and risk a prefix mismatch with the
    cache); appending a fresh user header after the previous assistant tokens is a
    faithful-enough multi-turn cache for attention-trace collection — the topic
    switch, not the exact inter-turn glue token, is what the experiment needs.
    """
    import torch
    msgs = [{"role": "user", "content": user_text}]
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):       # transformers 5.x returns a BatchEncoding
        ids = ids.input_ids
    ids = ids.to(device)
    if turn_idx > 0:
        ids = _strip_leading_bos(ids, getattr(tokenizer, "bos_token_id", None))
    return ids


def extract_multiturn_traces(out_path, *, conversations, model=None, tokenizer=None,
                             model_id=None, topics=None, device="cuda",
                             block_size: int = 32, r_label: float = 0.10,
                             horizons=(1, 4, 16, 64), max_new_tokens_per_turn: int = 48,
                             ema_decay: float = 0.9, dtype: str = "float16",
                             request_id_start: int = 0, max_total_tokens: int = 12000,
                             query_variants: bool = False) -> int:
    """Write a multi-turn JSONL trace; return the row count.

    conversations: list of conversations, each a list of user-message strings (turns).
    topics:        optional parallel list of per-turn topic tags (for the `topic` field).
    """
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        raise RuntimeError("extract_multiturn_traces needs torch + transformers")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' but no CUDA visible; run on the A100 box "
                           "or pass device='cpu' with a tiny model for validation.")
    if len(horizons) != 4:
        raise ValueError("horizons must have exactly 4 entries (y_h1..y_h64 schema)")

    from .features import extract_features

    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype),
            attn_implementation="eager", output_attentions=True, local_files_only=True)
    model = model.to(device).eval()
    if tokenizer is None and model_id is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer is None:
        raise ValueError("need a tokenizer (pass tokenizer= or model_id=)")

    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    rows_out: list = []
    for ci, conv in enumerate(conversations):
        conv_topics = (topics[ci] if topics is not None else None)
        # one persistent cache + EMA/history state for the whole conversation
        q_capture: dict = {}
        hooks = []
        for li, layer in enumerate(model.model.layers):
            def _mk(idx):
                def _h(_m, _inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    q_capture[idx] = o[:, -1, :].detach().float().cpu().numpy()
                return _h
            hooks.append(layer.self_attn.q_proj.register_forward_hook(_mk(li)))

        past = None
        n_layers = getattr(model.config, "num_hidden_layers", None)
        ema = None
        block_attn_hist = None
        last_used_cache: dict = {}
        feat_rows = []        # (layer, t, F, nb, turn_idx, qv)
        t = 0                 # continuous decode step across all turns

        for turn_idx, user_text in enumerate(conv):
            new_ids = _render_turn_ids(tokenizer, user_text, turn_idx, device)
            cur_len = 0 if past is None else _cache_len(past)
            if max_total_tokens and cur_len + new_ids.shape[1] >= max_total_tokens:
                break  # would overflow the budget; stop adding turns to this conv

            # ---- prefill the new user turn into the persistent cache (no rows) ----
            with torch.no_grad():
                out = model(new_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if n_layers is None:
                n_layers = len(past)
            if ema is None:
                ema = [None] * n_layers
                block_attn_hist = [[] for _ in range(n_layers)]
            next_id = out.logits[:, -1:].argmax(-1)

            # ---- greedy decode this turn, continuous step counter ----
            for j in range(max_new_tokens_per_turn):
                if max_total_tokens and _cache_len(past) >= max_total_tokens:
                    break
                with torch.no_grad():
                    out = model(next_id, past_key_values=past, use_cache=True,
                                output_attentions=True)
                past = out.past_key_values
                for l in range(n_layers):
                    attn_l = out.attentions[l][0].mean(0).squeeze(0).float().cpu().numpy()  # (kv,)
                    block_attn = blockify(attn_l, block_size)
                    block_attn_hist[l].append(block_attn)
                    ema[l] = update_ema(ema[l], block_attn, ema_decay)
                    k = _layer_key(past, l)[0].mean(0).float().cpu().numpy()                # (kv, hd)
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
                        r_cross=r_label)
                    qv = None
                    if query_variants:
                        K_full = _layer_key(past, l)[0].float().cpu().numpy()
                        n_kv = K_full.shape[0]
                        qh = q_capture.get(l, np.zeros(n_heads * head_dim, np.float32)).reshape(-1, head_dim)
                        grp = max(1, qh.shape[0] // max(1, n_kv))
                        kv_of_head = np.minimum(np.arange(qh.shape[0]) // grp, n_kv - 1)
                        dots = np.einsum("hkd,hd->hk", K_full[kv_of_head], qh)
                        qv = (blockify_max(dots.max(0), block_size), blockify(dots.mean(0), block_size))
                    feat_rows.append((l, t, F, nb, turn_idx, qv))
                tok = int(next_id)
                next_id = out.logits[:, -1:].argmax(-1)
                t += 1
                # natural turn end: stop early on EOS once the turn has some content
                if eos_id is not None and tok == int(eos_id) and j >= 4:
                    break
            # close the assistant turn in the running chat history is implicit:
            # the next user turn is appended directly after the generated tokens.

        # ---- emit rows with horizon labels (labels read the continuous history) ----
        for (l, t_row, F, nb, turn_idx, qv) in feat_rows:
            labels = labels_for_horizons(block_attn_hist[l], t_row, nb, r_label, horizons)
            topic = (conv_topics[turn_idx] if conv_topics is not None
                     and turn_idx < len(conv_topics) else "")
            for b in range(F.shape[0]):
                row = dict(
                    request_id=f"c{request_id_start + ci}", layer=int(l), step=int(t_row),
                    turn=int(turn_idx), topic=topic, block_idx=int(b),
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


def _cache_len(past) -> int:
    """Sequence length held in a transformers KV cache, across API versions."""
    try:
        k = _layer_key(past, 0)
        return int(k.shape[-2])
    except Exception:
        try:
            return int(past.get_seq_length())
        except Exception:
            return 0
