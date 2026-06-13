"""REAL offloaded-KV decode microbenchmark — actual TPOT (not a model).

KV lives in CPU PINNED memory (token-major [ctx, NKV, HD] so a block fetch is a
contiguous DMA). Each decode layer fetches its KV to GPU over PCIe; prefetch of layer
l+1 runs on a 2nd CUDA stream, overlapping layer l's compute (event-based double
buffer). Real GPU ops at Llama-3.1-8B shapes (QKV/O proj, attention over the fetched
blocks, MLP). Schemes:

  gpu_resident : KV already on GPU (throughput ceiling, no fetch).
  fetch_all    : prefetch layer l+1's FULL KV during layer l compute (FlexGen, full quality).
  prefetch_hot : prefetch only the hot fraction h (query-free cross-layer prediction,
                 recall r); the (1-r) mispredictions fetched on-demand (stall). == module.
  reactive_hot : fetch the hot fraction h on-demand at layer l, blocking (no overlap).

TPOT physics only; the quality axis is the real-model LongBench F1 (hot fraction==budget).
    python experiments/offload_decode_bench.py --ctx 16384 32768 65536
"""
import argparse, json, time
import torch

D, NH, NKV, HD, NL, FF = 4096, 32, 8, 128, 32, 14336
BLK = 32
DEV = "cuda:0"


def alloc_weights():
    g = lambda *s: (torch.randn(*s, device=DEV, dtype=torch.float16) * 0.02)
    return dict(Wq=g(D, NH*HD), Wk=g(D, NKV*HD), Wv=g(D, NKV*HD), Wo=g(NH*HD, D),
                Wg=g(D, FF), Wu=g(D, FF), Wd=g(FF, D))


def layer_compute(x, W, Kg, Vg):
    """x:[1,D]; Kg,Vg:[T,NKV,HD] on GPU. Real attention over the fetched KV + MLP."""
    q = (x @ W["Wq"]).view(NH, HD)
    rep = NH // NKV
    Kf = Kg.permute(1, 0, 2).repeat_interleave(rep, 0)   # [NH,T,HD]
    Vf = Vg.permute(1, 0, 2).repeat_interleave(rep, 0)
    att = torch.softmax((q.unsqueeze(1) @ Kf.transpose(1, 2)).squeeze(1) / HD**0.5, -1)
    o = (att.unsqueeze(1) @ Vf).reshape(1, NH*HD) @ W["Wo"]
    x = x + o
    mlp = (torch.nn.functional.silu(x @ W["Wg"]) * (x @ W["Wu"])) @ W["Wd"]
    return x + mlp


def run(ctx, scheme, W, kv_cpu, h_frac, recall, steps=12):
    N = ctx // BLK
    n_hot = max(1, int(h_frac * N))
    nb_tok = (N if scheme == "fetch_all" else n_hot) * BLK
    n_miss_tok = max(0, int((1 - recall) * n_hot)) * BLK
    s_fetch = torch.cuda.Stream()
    comp = torch.cuda.current_stream()

    def prefetch(layer):                      # async on s_fetch -> (K,V,event)
        with torch.cuda.stream(s_fetch):
            Kg = kv_cpu[layer][0][:nb_tok].to(DEV, non_blocking=True)
            Vg = kv_cpu[layer][1][:nb_tok].to(DEV, non_blocking=True)
            ev = s_fetch.record_event()
        return Kg, Vg, ev

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(steps):
        x = torch.randn(1, D, device=DEV, dtype=torch.float16)
        if scheme in ("fetch_all", "prefetch_hot"):
            cur = prefetch(0)
            for l in range(NL):
                nxt = prefetch(l + 1) if l + 1 < NL else None   # overlaps with compute below
                comp.wait_event(cur[2])                          # layer l KV ready
                if scheme == "prefetch_hot" and n_miss_tok:      # on-demand miss fetch (stall)
                    _ = kv_cpu[l][0][:n_miss_tok].to(DEV, non_blocking=False)
                x = layer_compute(x, W, cur[0], cur[1])
                cur = nxt
        else:  # reactive_hot: blocking on-demand, no overlap
            for l in range(NL):
                K = kv_cpu[l][0][:nb_tok].to(DEV, non_blocking=False)
                V = kv_cpu[l][1][:nb_tok].to(DEV, non_blocking=False)
                x = layer_compute(x, W, K, V)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--h", type=float, default=0.25)
    ap.add_argument("--recall", type=float, default=0.875)
    ap.add_argument("--out", default="experiments/results/offload_decode_real.json")
    a = ap.parse_args()
    W = alloc_weights()
    out = {"h": a.h, "recall": a.recall, "by_ctx": {}}
    print(f"REAL offloaded-KV decode TPOT (h={a.h}, recall={a.recall}), Llama-3.1-8B shapes\n")
    hdr = f"{'ctx':>7} | {'gpu_res':>9} {'fetch_all':>10} {'reactive':>10} {'prefetch':>10} | vs all / vs reactive"
    print(hdr)
    for ctx in a.ctx:
        kv_cpu = [(torch.randn(ctx, NKV, HD, dtype=torch.float16).pin_memory(),
                   torch.randn(ctx, NKV, HD, dtype=torch.float16).pin_memory()) for _ in range(NL)]
        kv_gpu = [(kv_cpu[l][0].to(DEV), kv_cpu[l][1].to(DEV)) for l in range(NL)]
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(12):
            x = torch.randn(1, D, device=DEV, dtype=torch.float16)
            for l in range(NL):
                x = layer_compute(x, W, kv_gpu[l][0], kv_gpu[l][1])
            torch.cuda.synchronize()
        res = {"gpu_resident": (time.perf_counter() - t) / 12 * 1e3}
        del kv_gpu; torch.cuda.empty_cache()
        for s in ["fetch_all", "reactive_hot", "prefetch_hot"]:
            res[s] = run(ctx, s, W, kv_cpu, a.h, a.recall)
        out["by_ctx"][ctx] = {k: round(v, 2) for k, v in res.items()}
        va = 100 * (1 - res["prefetch_hot"] / res["fetch_all"])
        vr = 100 * (1 - res["prefetch_hot"] / res["reactive_hot"])
        print(f"{ctx:>7} | {res['gpu_resident']:>7.1f}ms {res['fetch_all']:>8.1f}ms "
              f"{res['reactive_hot']:>8.1f}ms {res['prefetch_hot']:>8.1f}ms | -{va:.0f}% / -{vr:.0f}%")
        del kv_cpu; torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nWROTE", a.out)


if __name__ == "__main__":
    main()
