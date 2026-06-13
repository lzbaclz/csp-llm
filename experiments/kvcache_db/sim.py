"""Multi-tier KV-cache simulator + policies + Belady-with-recompute oracle (RECC).

Each request references a cacheable prefix object (key, length). Reuse = same key
again. An object lives in exactly one tier (GPU/DRAM/SSD) or is dropped. On a request
we pay the recovery cost for where the prefix currently is (0 if GPU; reload or
recompute, whichever cheaper, otherwise), then promote it to GPU; over-capacity tiers
evict by the policy's priority, cascading demotions GPU->DRAM->SSD->drop.

Policies differ ONLY in the eviction priority (lower priority = evict first):
  LRU, LFU            : recency / frequency (cost- and recompute-blind, the status quo)
  GDSF                : aging + freq * recompute_cost / size (cost-aware, classic web cache)
  Marconi             : recency * (recompute_cost / size)   (FLOP-efficiency x recency)
  RECC (ours)         : P_reuse * recovery_cost_if_evicted / size  (recompute-AWARE + predicted reuse)
  OPT-RC (oracle)     : offline Belady-with-recompute (true future reuse + recovery cost)
"""
import math, random
from collections import defaultdict

SAMPLE_K = 64          # Redis-style sampled eviction: evict min over a random K-sample
_RNG = random.Random(12345)

def _sample_min(items, keyfn):
    """min over a random sample of dict .values() (O(K), realistic approximate eviction)."""
    if len(items) <= SAMPLE_K:
        return min(items, key=keyfn)
    return min(_RNG.sample(items, SAMPLE_K), key=keyfn)


class Obj:
    __slots__ = ("key", "length", "last", "count", "ins")
    def __init__(self, key, length, t):
        self.key = key; self.length = length; self.last = t; self.count = 1; self.ins = t


class ReusePredictor:
    """Cheap online reuse SCORE per key: LRFU-style recency x frequency. A key seen
    `count` times and last seen `last` scores count * decay(now-last) -- subsumes LRU
    (recency) and LFU (frequency), and discounts one-shots (count=1). This is the reuse
    signal RECC multiplies by the recompute-cost-per-byte (the cost the cache avoids)."""
    def __init__(self, half_life=3000.0):
        self.count = defaultdict(int)
        self.half_life = half_life

    def observe_access(self, key):
        self.count[key] += 1

    def observe_evict_or_end(self, key):
        pass

    def p_reuse(self, key, now, last):
        c = self.count.get(key, 1)
        decay = self.half_life / (self.half_life + (now - last))   # cheap recency in (0,1], no exp
        return c * (0.2 + 0.8 * decay)        # LRFU score (recency-boosted frequency)


# ---------------- priority functions (lower => evict first) ----------------
def prio_lru(o, cm, pred, now):    return o.last
def prio_lfu(o, cm, pred, now):    return o.count
def prio_gdsf(o, cm, pred, now, L=0.0):
    return L + o.count * cm.recompute_cost_s(o.length) / cm.size_bytes(o.length)
def prio_marconi(o, cm, pred, now):
    recency = 1.0 / (1.0 + (now - o.last))
    return recency * (cm.recompute_cost_s(o.length) / cm.size_bytes(o.length))
def prio_recc(o, cm, pred, now, demote_tier=None):
    # RECC value = reuse-score x recompute-cost-PER-BYTE (the super-linear cost the cache
    # avoids on reuse). Cost-per-byte grows with prefix length (O(L^2) prefill), so RECC
    # protects long, reused prefixes -- the thing LRU/LFU (cost-blind) miss.
    return pred.p_reuse(o.key, now, o.last) * cm.recompute_cost_s(o.length) / cm.size_bytes(o.length)

def prio_lrfu(o, cm, pred, now):   # RECC ablation: reuse score WITHOUT recompute-cost term
    return pred.p_reuse(o.key, now, o.last)

POLICIES = {"lru": prio_lru, "lfu": prio_lfu, "gdsf": prio_gdsf,
            "marconi": prio_marconi, "lrfu": prio_lrfu, "recc": prio_recc}


class Cache:
    def __init__(self, cm, policy="lru"):
        self.cm = cm; self.policy = policy
        self.tiers = list(cm.tiers)                  # ["GPU","DRAM","SSD"]
        self.store = {t: {} for t in self.tiers}     # tier -> {key: Obj}
        self.bytes = {t: 0.0 for t in self.tiers}
        self.where = {}                              # key -> tier
        self.pred = ReusePredictor()
        self._gdsf_L = 0.0

    def _prio(self, o, now, tier):
        p = self.policy
        if p == "recc":
            ti = self.tiers.index(tier)
            demote = self.tiers[ti + 1] if ti + 1 < len(self.tiers) else None
            return prio_recc(o, self.cm, self.pred, now, demote)
        if p == "gdsf":
            return prio_gdsf(o, self.cm, self.pred, now, self._gdsf_L)
        return POLICIES[p](o, self.cm, self.pred, now)

    def _evict_to_fit(self, tier, now):
        ti = self.tiers.index(tier)
        cap = self.cm.capacity[tier]
        st = self.store[tier]
        while self.bytes[tier] > cap and st:
            victim = _sample_min(list(st.values()), lambda o: self._prio(o, now, tier))
            del st[victim.key]; self.bytes[tier] -= self.cm.size_bytes(victim.length)
            if self.policy == "gdsf":
                self._gdsf_L = self._prio(victim, now, tier)   # aging
            nxt = self.tiers[ti + 1] if ti + 1 < len(self.tiers) else None
            if nxt is None:
                # below last tier => drop. (Recompute-aware: only worth storing if a
                # future reload would beat recompute; recovery_cost handles that at use.)
                self.where.pop(victim.key, None); self.pred.observe_evict_or_end(victim.key)
            else:
                self.store[nxt][victim.key] = victim
                self.bytes[nxt] += self.cm.size_bytes(victim.length); self.where[victim.key] = nxt
                self._evict_to_fit(nxt, now)

    def access(self, key, length, now):
        tier = self.where.get(key)
        cost = self.cm.recovery_cost_s(length, tier)     # 0 if GPU; reload/recompute else
        self.pred.observe_access(key)
        # remove from current tier
        if tier is not None:
            o = self.store[tier].pop(key); self.bytes[tier] -= self.cm.size_bytes(length)
            o.last = now; o.count += 1; o.length = length
        else:
            o = Obj(key, length, now)
        # promote to GPU
        gpu = self.tiers[0]
        self.store[gpu][key] = o; self.bytes[gpu] += self.cm.size_bytes(length); self.where[key] = gpu
        self._evict_to_fit(gpu, now)
        return cost, (tier == gpu)     # (recovery cost seconds, was_gpu_hit)


def run_policy(trace, cm, policy):
    """trace: list of (now, key, length). Returns metrics dict."""
    c = Cache(cm, policy)
    total, recompute_like, n_gpu_hit, n_recompute = 0.0, 0.0, 0, 0
    costs = []
    for (now, key, length) in trace:
        cost, gpu_hit = c.access(key, length, now)
        total += cost; costs.append(cost)
        if gpu_hit:
            n_gpu_hit += 1
        # classify recovery: recompute if cost == recompute_cost (within eps)
        if cost > 0 and abs(cost - cm.recompute_cost_s(length)) < 1e-12:
            n_recompute += 1
    costs.sort()
    n = len(trace)
    return {
        "policy": policy, "n": n,
        "total_cost_s": total,
        "mean_cost_ms": 1e3 * total / n,
        "p99_cost_ms": 1e3 * costs[min(n - 1, int(0.99 * n))],
        "gpu_hit_rate": n_gpu_hit / n,
        "recompute_frac": n_recompute / n,
    }


# ---------------- Belady-with-recompute offline oracle (OPT-RC) ----------------
def run_oracle(trace, cm):
    """Offline policy with true future knowledge. On each over-capacity eviction, evict
    the resident object with the LOWEST future value:
        value = (reused_again ? 1 : 0) * recovery_cost_if_it_falls_one_tier / size,
    tie-broken by FURTHEST next use (classic Belady). Objects never reused again are
    evicted first (value 0). This is a strong recompute-aware Belady-style lower-cost
    baseline (not provably OPT for variable size, but uses the true future)."""
    # precompute, for each position, the next-use index of its key
    nxt = [math.inf] * len(trace)
    seen = {}
    for i in range(len(trace) - 1, -1, -1):
        k = trace[i][1]
        nxt[i] = seen.get(k, math.inf)
        seen[k] = i

    tiers = list(cm.tiers)
    store = {t: {} for t in tiers}        # tier -> {key: (length, next_use)}
    bytes_ = {t: 0.0 for t in tiers}
    where = {}
    total = 0.0; n_gpu = 0; n_rec = 0; costs = []

    def evict_fit(tier, i):
        ti = tiers.index(tier); cap = cm.capacity[tier]; st = store[tier]
        while bytes_[tier] > cap and st:
            ti2 = tiers.index(tier)
            demote = tiers[ti2 + 1] if ti2 + 1 < len(tiers) else None
            def val(item):
                k, (ln, nu) = item
                reused = 0.0 if nu == math.inf else 1.0
                rc = cm.recovery_cost_s(ln, demote)
                return (reused * rc / cm.size_bytes(ln), -nu)  # low value + furthest next-use evicted first
            items = list(st.items())
            if len(items) > SAMPLE_K:
                items = _RNG.sample(items, SAMPLE_K)
            vk, (vln, vnu) = min(items, key=lambda kv: val(kv))
            del st[vk]; bytes_[tier] -= cm.size_bytes(vln)
            if demote is None:
                where.pop(vk, None)
            else:
                store[demote][vk] = (vln, vnu); bytes_[demote] += cm.size_bytes(vln); where[vk] = demote
                evict_fit(demote, i)

    for i, (now, key, length) in enumerate(trace):
        tier = where.get(key)
        cost = cm.recovery_cost_s(length, tier)
        total += cost; costs.append(cost)
        if tier == tiers[0]:
            n_gpu += 1
        if cost > 0 and abs(cost - cm.recompute_cost_s(length)) < 1e-12:
            n_rec += 1
        if tier is not None:
            del store[tier][key]; bytes_[tier] -= cm.size_bytes(length)
        store[tiers[0]][key] = (length, nxt[i]); bytes_[tiers[0]] += cm.size_bytes(length); where[key] = tiers[0]
        evict_fit(tiers[0], i)
    costs.sort(); n = len(trace)
    return {"policy": "OPT-RC", "n": n, "total_cost_s": total,
            "mean_cost_ms": 1e3 * total / n,
            "p99_cost_ms": 1e3 * costs[min(n - 1, int(0.99 * n))],
            "gpu_hit_rate": n_gpu / n, "recompute_frac": n_rec / n}
