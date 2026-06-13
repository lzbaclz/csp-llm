"""Real A100 predictor-latency microbenchmark (the deployment §6 number).

TensorRT is not installed in this env, so instead of the TRT path we measure the
predictor on the A100 directly with PyTorch + CUDA-Graph capture + CUDA-event
timing — the honest, reproducible latency of the three scorers on the hot path.
We report per-replay P50/P99/P99.9 over many graph replays at batch B blocks,
for the closed form (4 weights), pairwise (15), and tiny MLP (148), in fp16 and
fp32, with and without CUDA-Graph capture.

    python experiments/bench_wcet_gpu.py --batch 4096 --replays 20000 \
        --out experiments/results/wcet_gpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def _percentiles(us: np.ndarray) -> dict:
    return dict(p50=float(np.percentile(us, 50)), p90=float(np.percentile(us, 90)),
                p99=float(np.percentile(us, 99)), p999=float(np.percentile(us, 99.9)),
                mean=float(us.mean()), std=float(us.std()), n=int(us.shape[0]))


def make_scorers(dtype, device):
    """Return {name: (callable(F)->scores, n_params)} as pure-tensor ops."""
    g = torch.Generator(device="cpu").manual_seed(0)
    w = torch.randn(4, generator=g).to(device=device, dtype=dtype)
    b = torch.randn(1, generator=g).to(device=device, dtype=dtype)
    M = torch.randn(4, 4, generator=g).to(device=device, dtype=dtype)
    W1 = torch.randn(4, 16, generator=g).to(device=device, dtype=dtype)
    b1 = torch.randn(16, generator=g).to(device=device, dtype=dtype)
    W2 = torch.randn(16, 4, generator=g).to(device=device, dtype=dtype)
    b2 = torch.randn(4, generator=g).to(device=device, dtype=dtype)

    def closed(F):
        return torch.sigmoid(F @ w + b)

    def pairwise(F):
        quad = ((F @ M) * F).sum(-1)
        return torch.sigmoid(F @ w + quad + b)

    def tinymlp(F):
        h = torch.nn.functional.gelu(F @ W1 + b1)
        return torch.sigmoid((h @ W2 + b2)[:, 0])

    return {"closed": (closed, 4), "pairwise": (pairwise, 15), "tinymlp": (tinymlp, 148)}


def time_graph(fn, F, replays, warmup=200):
    """CUDA-Graph captured per-replay timing (µs)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(F)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    static_out = None
    with torch.cuda.graph(g):
        static_out = fn(F)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for i in range(replays):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    us = np.array([starts[i].elapsed_time(ends[i]) * 1e3 for i in range(replays)])
    return _percentiles(us)


def time_eager(fn, F, replays, warmup=200):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(F)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for i in range(replays):
        starts[i].record()
        fn(F)
        ends[i].record()
    torch.cuda.synchronize()
    us = np.array([starts[i].elapsed_time(ends[i]) * 1e3 for i in range(replays)])
    return _percentiles(us)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--replays", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="experiments/results/wcet_gpu.json")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr); return 1
    dev = torch.device(a.device)
    name = torch.cuda.get_device_name(dev)
    out = dict(device=name, batch=a.batch, replays=a.replays,
               note="PyTorch CUDA-Graph capture + CUDA-event per-replay timing; "
                    "predictor scoring only (feature extraction excluded)")
    res = {}
    for prec, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
        rng = np.random.default_rng(0)
        F = torch.tensor(rng.random((a.batch, 4)), device=dev, dtype=dtype)
        scorers = make_scorers(dtype, dev)
        res[prec] = {}
        for sname, (fn, nparams) in scorers.items():
            graph = time_graph(fn, F, a.replays)
            eager = time_eager(fn, F, min(a.replays, 5000))
            res[prec][sname] = dict(params=nparams, cuda_graph_us=graph, eager_us=eager)
            print(f"{prec:4s} {sname:9s} (p={nparams:3d})  graph P50={graph['p50']:.2f} "
                  f"P99={graph['p99']:.2f} P99.9={graph['p999']:.2f} µs", flush=True)
    out["results"] = res
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
