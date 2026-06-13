#!/usr/bin/env python3
"""What actually predicts per-block KV *quantization* sensitivity?

Novel measurement (the importance-aware-quant field never did this unified study):
for each KV block, the GROUND-TRUTH quantization sensitivity = KL divergence of the
next-token logits when ONLY that block's K,V are fake-quantized to n-bit (all layers),
vs full precision. Then we test which cheap signal predicts it:
  - attn      : last-query attention mass to the block (ZipCache's saliency signal)
  - vnorm     : mean L2 norm of the block's value vectors (VATP's "value matters")
  - knorm     : mean L2 norm of the block's key vectors
  - attn_vnorm: attn * vnorm (VATP's product)
  - posn      : position (recency)
Output: one JSONL row per (prompt, block) with features + sensitivity, for the
relevance/redundancy analysis and the adaptive-vs-uniform head-to-head.
"""
import json, argparse, os, math
import torch

def fake_quant(x, n_bits):
    """Asymmetric per-token (over last dim) min-max quantize->dequantize to n_bits."""
    if n_bits >= 16:
        return x
    qmax = (1 << n_bits) - 1
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / qmax
    q = torch.round((x - mn) / scale).clamp_(0, qmax)
    return q * scale + mn

def kl(p_logits, q_logits):
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    return torch.sum(p.exp() * (p - q)).item()

def load_prompts(n, ctx, tok):
    import json as _j
    src = "/public/data_zoo/longbench/data/gov_report.jsonl"
    out = []
    for line in open(src, encoding="utf-8"):
        try:
            c = _j.loads(line).get("context", "")
        except Exception:
            continue
        if len(c) < 4000:
            continue
        ids = tok(c, return_tensors="pt").input_ids[:, :ctx]
        if ids.shape[1] >= ctx:
            out.append(ids)
        if len(out) >= n:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/public/model_zoo/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--nbits", type=int, default=2)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="cuda", attn_implementation="eager")
    model.eval()
    prompts = load_prompts(a.n, a.ctx, tok)
    print(f"loaded {len(prompts)} prompts ctx={a.ctx} nbits={a.nbits}")
    rows = []
    for pi, ids in enumerate(prompts):
        ids = ids.to("cuda")
        with torch.no_grad():
            pre = model(input_ids=ids[:, :-1], use_cache=True)
            cache = pre.past_key_values
            L = int(cache.get_seq_length())          # prefill length
            seq = L
            nb = (seq + a.block - 1) // a.block

            def kv():
                if hasattr(cache, "layers"):
                    return ([l.keys for l in cache.layers], [l.values for l in cache.layers])
                if hasattr(cache, "key_cache"):
                    return (cache.key_cache, cache.value_cache)
                return ([k for k, _ in cache], [v for _, v in cache])

            # (a) attention probe — SEPARATE forward; output_attentions changes the kernel,
            #     so keep it OUT of the reference/per-block logit path to avoid an offset.
            probe = model(input_ids=ids[:, -1:], past_key_values=cache,
                          use_cache=False, output_attentions=True)
            attn_blk = torch.zeros(nb)
            for at in probe.attentions:               # [1, heads, 1, L+1]
                aa = at[0].mean(0).squeeze(0).float().cpu()
                for b in range(nb):
                    attn_blk[b] += aa[b*a.block:(b+1)*a.block].sum()
            attn_blk /= len(probe.attentions)
            cache.crop(L)
            # (b) reference logits — IDENTICAL conditions to the per-block forwards.
            L_ref = model(input_ids=ids[:, -1:], past_key_values=cache,
                          use_cache=False).logits[0, -1].detach()
            cache.crop(L)
            # per-block key/value norms (mean over layers, heads, tokens)
            kc, vc = kv()
            knorm = torch.zeros(nb); vnorm = torch.zeros(nb)
            for l in range(len(kc)):
                kn = kc[l][0].norm(dim=-1).float().mean(0).cpu()
                vn = vc[l][0].norm(dim=-1).float().mean(0).cpu()
                for b in range(nb):
                    knorm[b] += kn[b*a.block:(b+1)*a.block].mean()
                    vnorm[b] += vn[b*a.block:(b+1)*a.block].mean()
            knorm /= len(kc); vnorm /= len(vc)
            # ground-truth sensitivity: quantize block b in ALL layers -> KL on last-token logits
            for b in range(nb):
                cache.crop(L); kc, vc = kv()          # fresh prefill-only cache each block
                s, e = b*a.block, min((b+1)*a.block, seq)
                saved = [(kc[l][:, :, s:e, :].clone(), vc[l][:, :, s:e, :].clone())
                         for l in range(len(kc))]
                for l in range(len(kc)):
                    kc[l][:, :, s:e, :] = fake_quant(kc[l][:, :, s:e, :], a.nbits)
                    vc[l][:, :, s:e, :] = fake_quant(vc[l][:, :, s:e, :], a.nbits)
                o = model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=False)
                sens = kl(L_ref, o.logits[0, -1])
                cache.crop(L); kc, vc = kv()          # drop appended token, then undo quant
                for l in range(len(kc)):
                    kc[l][:, :, s:e, :] = saved[l][0]
                    vc[l][:, :, s:e, :] = saved[l][1]
                rows.append({"p": pi, "b": b, "sens": sens,
                             "attn": float(attn_blk[b]), "vnorm": float(vnorm[b]),
                             "knorm": float(knorm[b]), "posn": b / max(1, nb-1)})
        print(f"[{pi+1}/{len(prompts)}] nb={nb} mean_sens={sum(r['sens'] for r in rows[-nb:])/nb:.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")

if __name__ == "__main__":
    main()
