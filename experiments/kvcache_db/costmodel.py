"""Cost model for the multi-tier LLM KV cache (RECC).

The defining KV twist: a cache MISS has three recovery modes, not one fetch:
  KEEP    — resident in a tier (no recovery cost; costs that tier's capacity)
  RELOAD  — present in a lower tier, copy up: cost = bytes / bandwidth(tier)
  RECOMPUTE — absent everywhere, re-prefill: cost = prefix_len * t_prefill_per_token
So a prefix's *value of being cached* is LENGTH-DEPENDENT (long prefix => recompute
dear => keep; short prefix => recompute cheap => fine to drop). LRU/LFU/GDSF/Marconi
do not use the recompute alternative; RECC and the Belady-with-recompute oracle do.

Defaults are realistic Llama-3.1-8B / A100 numbers; experiments sweep them.
"""
from dataclasses import dataclass

# Llama-3.1-8B: 2(K,V) * 32 layers * 8 kv-heads * 128 head_dim * 2 bytes(fp16)
BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2            # = 131072 B = 128 KiB / token
# Prefill is SUPER-LINEAR: linear MLP term + QUADRATIC attention term (O(L^2) d).
# So recompute cost per cached byte GROWS with prefix length -> long reused prefixes
# are worth far more per byte than short ones (the key property LRU/LFU ignore).
T_PREFILL_LIN_S = 0.40e-3                          # linear term, s/token (MLP-dominated)
T_PREFILL_QUAD_S = 2.0e-8                          # quadratic term, s/token^2 (attention)

# Tier bandwidths (bytes/s) for RELOAD up to GPU. GPU is resident (no reload).
BW = {
    "GPU":  float("inf"),
    "DRAM": 20e9,    # PCIe/NVLink host<->device, ~20 GB/s effective
    "SSD":  4e9,     # local NVMe, ~4 GB/s
    "REMOTE": 12e9,  # RDMA/network, ~12 GB/s
}
TIERS_DEFAULT = ["GPU", "DRAM", "SSD"]            # ordered fast->slow; below SSD = dropped


@dataclass
class CostModel:
    bytes_per_token: float = BYTES_PER_TOKEN
    t_lin_s: float = T_PREFILL_LIN_S
    t_quad_s: float = T_PREFILL_QUAD_S
    bw: dict = None
    tiers: tuple = tuple(TIERS_DEFAULT)
    # per-tier capacity in BYTES (GPU small, DRAM larger, SSD huge). Set by experiment.
    capacity: dict = None

    def __post_init__(self):
        if self.bw is None:
            self.bw = dict(BW)

    def size_bytes(self, length):
        return length * self.bytes_per_token

    def recompute_cost_s(self, length):
        """Re-prefill from scratch: linear MLP + quadratic attention (super-linear)."""
        return length * self.t_lin_s + length * length * self.t_quad_s

    def reload_cost_s(self, length, tier):
        """Copy the prefix's KV from `tier` up to GPU."""
        bw = self.bw.get(tier, float("inf"))
        if bw == float("inf"):
            return 0.0
        return self.size_bytes(length) / bw

    def recovery_cost_s(self, length, tier_or_none):
        """Cost to serve a request whose prefix currently sits at `tier_or_none`.
        None => dropped => must recompute. 'GPU' => resident => free.
        Otherwise => min(reload-from-that-tier, recompute) (the rational server picks
        the cheaper recovery — the heart of the recompute twist)."""
        if tier_or_none == "GPU":
            return 0.0
        if tier_or_none is None:
            return self.recompute_cost_s(length)
        return min(self.reload_cost_s(length, tier_or_none), self.recompute_cost_s(length))
