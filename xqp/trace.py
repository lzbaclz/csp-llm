"""Trace collection — runs alongside an HF model decode loop, recording the 4
input features and the ground-truth top-r label per block per layer per step.

This is the *off-line* training data builder; it is not used in the hot path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import numpy as np

from .features import extract_features, topk_indicator


@dataclass
class TraceRecord:
    """One (layer, step) trace record, flat for parquet/jsonl export."""
    request_id: str
    layer: int
    step: int
    block_idx: int
    f_within: float
    f_cross: float
    f_query: float
    f_pos: float
    y_h1: int   # top-r at step t+1
    y_h4: int
    y_h16: int
    y_h64: int


@dataclass
class TraceCollector:
    """Buffers TraceRecord rows; writes JSONL at flush().

    Usage:
        tc = TraceCollector("traces.jsonl")
        for step, ... in decode_loop:
            for layer, ... :
                tc.observe(request_id, layer, step, ema_within, ..., future_ema_window)
        tc.flush()
    """
    out_path: str | Path
    r_label: float = 0.10
    horizons: tuple = (1, 4, 16, 64)
    _buf: List[TraceRecord] = field(default_factory=list)

    def observe(
        self,
        *,
        request_id: str,
        layer: int,
        step: int,
        ema_within: np.ndarray,
        ema_prev_layer: np.ndarray | None,
        K_layer: np.ndarray,
        q_prev: np.ndarray,
        last_used: np.ndarray,
        future_ema: dict,
    ) -> int:
        """Record one (layer, step) batch of block stats and their horizon labels.

        future_ema: dict[horizon -> (B,) attention EMA at step t+horizon]
        """
        F = extract_features(
            ema_within=ema_within,
            ema_prev_layer=ema_prev_layer,
            K_layer=K_layer,
            q_prev=q_prev,
            step=step,
            last_used=last_used,
        )
        labels = {h: topk_indicator(np.asarray(future_ema[h], dtype=np.float32),
                                    self.r_label).astype(np.int64)
                  for h in self.horizons}
        B = F.shape[0]
        for b in range(B):
            self._buf.append(TraceRecord(
                request_id=request_id, layer=layer, step=step, block_idx=b,
                f_within=float(F[b, 0]), f_cross=float(F[b, 1]),
                f_query=float(F[b, 2]), f_pos=float(F[b, 3]),
                y_h1=int(labels[self.horizons[0]][b]),
                y_h4=int(labels[self.horizons[1]][b]),
                y_h16=int(labels[self.horizons[2]][b]),
                y_h64=int(labels[self.horizons[3]][b]),
            ))
        return B

    def flush(self) -> int:
        n = len(self._buf)
        with open(self.out_path, "w") as fh:
            for rec in self._buf:
                fh.write(json.dumps(asdict(rec)))
                fh.write("\n")
        self._buf.clear()
        return n


def load_trace(path: str | Path) -> dict:
    """Load JSONL trace into a dict of np arrays."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    keys = list(rows[0].keys())
    out = {k: np.asarray([r[k] for r in rows]) for k in keys}
    return out
