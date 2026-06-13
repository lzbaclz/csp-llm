"""Throughput model for cross-layer query-free prefetch in an offloaded-KV serving
regime (FlexGen-style: KV in CPU, fetched per layer over PCIe). All constants are
MEASURED: per-block KV DMA over PCIe (offload_consts.json) and per-layer decode
compute (compute_consts.json). The realizable query-free recall r=0.875 is the
measured prev-layer-attention -> this-layer-hot-set recall (run_crosslayer_prefetch).

Schemes (per layer, pipelined; TPOT = n_layers * (compute + stall)):
  gpu_resident      : KV in HBM, no fetch (throughput ceiling).
  flexgen_fetch_all : fetch all N blocks/layer (FULL quality), overlapped with compute.
  reactive_fetch_hot: fetch only the hot fraction h on demand at layer l (NOT overlapped,
                      because reactive needs layer l's query to know what is hot) -> stall.
  prefetch_hot      : predict the hot set from layer l-1's attention (query-free, recall r),
                      PREFETCH it during layer l-1 compute (overlapped, hidden up to compute);
                      only the (1-r) mispredictions stall. == the module.

reactive_fetch_hot and prefetch_hot fetch the SAME blocks (same hot fraction -> SAME quality);
the module's only difference is hiding the fetch a layer ahead. flexgen_fetch_all is full
quality but fetches everything.
"""
import json
import numpy as np

C = json.load(open("experiments/results/offload_consts.json"))
K = json.load(open("experiments/results/compute_consts.json"))
c_dma = C["c_dma_us_per_block"]                 # us per block, measured PCIe
nL = K["n_layers"]
tc = {int(k): v for k, v in K["per_layer_us_by_ctx"].items()}
BLK = 32                                         # tokens/block
h = 0.25                                         # hot fraction (B_t / N), measured ~25%
r = 0.875                                        # realizable query-free recall, measured

def tpot_ms(ctx, scheme):
    N = ctx // BLK
    t_c = tc[ctx]
    if scheme == "gpu_resident":
        stall = 0.0
    elif scheme == "flexgen_fetch_all":
        stall = max(0.0, N * c_dma - t_c)                       # overlap fetch with compute
    elif scheme == "reactive_fetch_hot":
        stall = h * N * c_dma                                   # on-demand, not overlapped
    elif scheme == "prefetch_hot":
        hidden = max(0.0, r * h * N * c_dma - t_c)              # prefetched, overlap w/ prev layer
        stall = (1 - r) * h * N * c_dma + hidden               # misprediction stalls + overflow
    return nL * (t_c + stall) / 1e3                            # ms

ctxs = sorted(tc)
print(f"measured: c_dma={c_dma:.2f}us/block, per-layer compute={tc}, h={h}, r={r}\n")
schemes = ["gpu_resident", "flexgen_fetch_all", "reactive_fetch_hot", "prefetch_hot"]
print(f"{'ctx':>7} | " + " ".join(f"{s:>18}" for s in schemes) + " | module vs flexgen / reactive")
rows = {}
for ctx in ctxs:
    t = {s: tpot_ms(ctx, s) for s in schemes}
    rows[ctx] = t
    vs_flex = 100 * (1 - t["prefetch_hot"] / t["flexgen_fetch_all"])
    vs_react = 100 * (1 - t["prefetch_hot"] / t["reactive_fetch_hot"])
    print(f"{ctx:>7} | " + " ".join(f"{t[s]:>16.1f}ms" for s in schemes) +
          f" | -{vs_flex:.0f}% / -{vs_react:.0f}%")

print("\nThroughput (tok/s) at 64K:")
for s in schemes:
    print(f"  {s:>18}: {1000/rows[65536][s]:.1f} tok/s")
json.dump({"constants": {"c_dma_us": c_dma, "h": h, "r": r, "n_layers": nL},
           "tpot_ms": rows}, open("experiments/results/offload_prefetch_model.json", "w"), indent=2)
print("\nWROTE experiments/results/offload_prefetch_model.json")
