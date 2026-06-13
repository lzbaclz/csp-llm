"""Thread-safe per-layer history — addresses 100-round R29 (KTH).

The default `XQPPolicy._per_layer_history` is a plain dict, racy under
concurrent serving with multiple workers. This module provides a
drop-in lock-guarded variant.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class ThreadSafeLayerHistory:
    """Lock-guarded {layer_id -> deque-like list}."""
    max_per_layer: int = 100
    _data: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, layer: int, value: float) -> int:
        with self._lock:
            buf = self._data.setdefault(layer, [])
            buf.append(float(value))
            if len(buf) > self.max_per_layer:
                buf.pop(0)
            return len(buf)

    def get(self, layer: int) -> list:
        with self._lock:
            return list(self._data.get(layer, []))

    def clear(self, layer: int | None = None) -> None:
        with self._lock:
            if layer is None:
                self._data.clear()
            else:
                self._data.pop(layer, None)
