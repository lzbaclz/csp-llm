"""WCET breakdown — addresses round-1 review M5.

Breaks the per-call cost into:
  (a) launch overhead     (cudaGraphLaunch entry)
  (b) matmul              (W·F)
  (c) bias add + sigmoid
  (d) output copy

On GPU+TRT we measure via NVTX ranges + nsight. On CPU we approximate via
per-stage Python timers — useful for catching gross asymmetries but not for
the actual reported number, which must come from ga100.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .predictor import ClosedFormXQP


def cpu_stage_breakdown(predictor: ClosedFormXQP, F: np.ndarray, n: int = 5000) -> dict:
    """Decompose CPU latency into matmul / bias+sigmoid / output."""
    F = np.ascontiguousarray(F.astype(np.float32))
    w = predictor.weights.astype(np.float32)
    b = float(predictor.bias) if predictor.bias.ndim == 0 else float(predictor.bias[0])

    matmul_us = np.zeros(n, dtype=np.float64)
    bias_sig_us = np.zeros(n, dtype=np.float64)
    output_us = np.zeros(n, dtype=np.float64)

    # warmup
    for _ in range(50):
        z = F @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        out = p.copy()

    for i in range(n):
        t0 = time.perf_counter_ns()
        z = F @ w
        t1 = time.perf_counter_ns()
        z += b
        p = 1.0 / (1.0 + np.exp(-z))
        t2 = time.perf_counter_ns()
        out = p.copy()
        t3 = time.perf_counter_ns()
        matmul_us[i] = (t1 - t0) / 1e3
        bias_sig_us[i] = (t2 - t1) / 1e3
        output_us[i] = (t3 - t2) / 1e3

    def pct(x):
        return {
            "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)),
            "p999": float(np.percentile(x, 99.9)),
        }
    return {
        "matmul":   pct(matmul_us),
        "bias_sig": pct(bias_sig_us),
        "output":   pct(output_us),
        "n":        n,
        "B":        int(F.shape[0]),
    }
