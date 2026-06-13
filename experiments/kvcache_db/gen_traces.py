"""Synthetic cross-request KV-reuse traces (key-level cache abstraction).

A request = (timestamp, key, length_tokens). Reuse = same key recurs. Four workload
processes, mixed, reproducing the documented reuse structure:
  - system   : shared system prompts, Zipf popularity, SHORT-ish, very high reuse
  - rag      : retrieved doc chunks, Zipf popularity, MEDIUM length, high reuse
  - conv     : multi-turn conversations, LONG prefix, bursty reuse then dies
               (exponential inter-turn gap -- Azure finding), recompute-expensive
  - oneshot  : unique prefixes, accessed once -> cache pollution

The length spread across workloads is what makes the recompute-vs-reload twist bite:
long conv/RAG prefixes are recompute-expensive (worth keeping/reloading), short system
prompts are cheap to recompute (fine to drop). LRU/LFU are blind to this.
"""
import random, math


def zipf_weights(n, s=1.1):
    w = [1.0 / (i + 1) ** s for i in range(n)]
    z = sum(w)
    return [x / z for x in w]


def gen_trace(n_requests=200_000, seed=0,
              mix=(("system", 0.30), ("rag", 0.30), ("conv", 0.25), ("oneshot", 0.15)),
              n_system=200, n_docs=4000, n_conv=8000,
              t_per_request=0.05):
    """Returns list of (timestamp_s, key, length_tokens), time-ordered."""
    rng = random.Random(seed)
    mixd = dict(mix)
    # static object pools
    sys_ids = [f"sys{i}" for i in range(n_system)]
    sys_len = {k: rng.randint(200, 1000) for k in sys_ids}        # SHORT: cheap recompute
    sys_w = zipf_weights(n_system, 1.2)
    doc_ids = [f"doc{i}" for i in range(n_docs)]
    doc_len = {k: rng.randint(500, 8000) for k in doc_ids}        # MEDIUM-LONG
    doc_w = zipf_weights(n_docs, 1.0)

    # conversations: each is a burst of T accesses to ONE long key, exponential gaps
    conv_state = []   # active conversations: (key, length, turns_left, next_time)
    conv_counter = [0]

    def new_conv(now):
        cid = f"conv{conv_counter[0]}"; conv_counter[0] += 1
        T = 1 + int(rng.expovariate(1 / 3.0))            # ~3-4 turns
        L = rng.randint(4000, 32000)                      # LONG prefix: expensive (O(L^2)) recompute
        return [cid, L, T, now]

    events = []   # (time, key, length)
    now = 0.0
    oneshot_counter = [0]
    kinds = [k for k, _ in mix]; probs = [mixd[k] for k in kinds]
    for _ in range(n_requests):
        now += rng.expovariate(1 / t_per_request)
        kind = rng.choices(kinds, probs)[0]
        if kind == "system":
            k = rng.choices(sys_ids, sys_w)[0]; events.append((now, k, sys_len[k]))
        elif kind == "rag":
            k = rng.choices(doc_ids, doc_w)[0]; events.append((now, k, doc_len[k]))
        elif kind == "oneshot":
            k = f"one{oneshot_counter[0]}"; oneshot_counter[0] += 1
            events.append((now, k, rng.randint(100, 8000)))
        else:  # conv
            if conv_state and rng.random() < 0.7:
                c = rng.choice(conv_state)
            else:
                c = new_conv(now); conv_state.append(c)
            events.append((now, c[0], c[1]))
            c[2] -= 1; c[3] = now
            if c[2] <= 0:
                conv_state.remove(c)
            if len(conv_state) > 2000:                    # cap active set
                conv_state.pop(0)
    events.sort(key=lambda e: e[0])
    return events


def trace_stats(trace):
    from collections import Counter
    keys = Counter(k for _, k, _ in trace)
    reused = sum(1 for k, c in keys.items() if c > 1)
    n = len(trace)
    reuse_hits = sum(c - 1 for c in keys.values())
    lengths = [l for _, _, l in trace]
    return {
        "requests": n, "unique_keys": len(keys),
        "reuse_rate": reuse_hits / n,            # fraction of requests that are repeats
        "reused_keys_frac": reused / len(keys),
        "mean_len": sum(lengths) / n, "max_len": max(lengths),
    }


if __name__ == "__main__":
    t = gen_trace(50_000, seed=0)
    print(trace_stats(t))
