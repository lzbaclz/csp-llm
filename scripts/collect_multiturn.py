#!/usr/bin/env python3
"""Collect persistent-cache MULTI-TURN attention traces (E5 / Idea 1).

Each conversation cycles its turns through *different domains* (code / Chinese-QA /
summarization / math / open chat), so a hard topic switch happens at every turn
boundary while the KV cache persists across turns — the regime in which a fixed
threshold loses coverage and the adaptive-conformal layer should earn its place.

Self-contained: the conversation pool is built in (no dataset fetch), so the run is
reproducible on the A100 box against /public/model_zoo. Dual-GPU via --worker-id /
--device / --conv-start/--conv-end, mirroring scripts/collect_traces_attn.py.

    python scripts/collect_multiturn.py --n-convs 48 --turns 6 \
        --device cuda:0 --worker-id gpu0 --out-dir /public/xqp_traces_mt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ZOO = Path("/public/model_zoo")
DEFAULT_MODELS = [
    ("Llama-3.1-8B-Instruct", str(MODEL_ZOO / "Llama-3.1-8B-Instruct")),
    ("Qwen2.5-7B-Instruct", str(MODEL_ZOO / "Qwen2.5-7B-Instruct")),
    ("Qwen3-8B", str(MODEL_ZOO / "Qwen3-8B")),
    ("Mistral-7B-Instruct-v0.3", str(MODEL_ZOO / "Mistral-7B-Instruct-v0.3")),
]

# ---- built-in topic pool: distinct attention statistics per domain ------------
_CODE = (
    "Here is a Python module. Read it carefully, then refactor it for clarity and "
    "explain every change you make.\n\n"
    "def parse_config(path):\n"
    "    cfg = {}\n"
    "    for line in open(path):\n"
    "        line = line.strip()\n"
    "        if not line or line.startswith('#'): continue\n"
    "        k, _, v = line.partition('=')\n"
    "        cfg[k.strip()] = v.strip()\n"
    "    return cfg\n\n"
    "class Cache:\n"
    "    def __init__(self, cap): self.cap = cap; self.d = {}; self.order = []\n"
    "    def get(self, k):\n"
    "        if k in self.d: self.order.remove(k); self.order.append(k); return self.d[k]\n"
    "        return None\n"
    "    def put(self, k, v):\n"
    "        if k in self.d: self.order.remove(k)\n"
    "        elif len(self.d) >= self.cap: ev = self.order.pop(0); del self.d[ev]\n"
    "        self.d[k] = v; self.order.append(k)\n")
_ZH = (
    "请阅读下面这段关于大型语言模型推理的中文材料，并回答最后的问题。\n\n"
    "在自回归解码过程中，模型每生成一个词元都要访问此前所有词元的键值缓存。"
    "随着上下文不断增长，键值缓存占用的显存也线性增长，成为长上下文推理的主要瓶颈。"
    "研究者发现，注意力在任一时刻只集中在少数关键位置上，因此可以只在快速显存中"
    "保留这些显著的键值块，而把其余的块迁移到较慢的存储层。问题在于，如何在每一步"
    "的算力预算之内，准确预测下一步哪些块仍然显著。\n\n"
    "问题：根据上文，长上下文推理的主要瓶颈是什么？为什么只保留显著块是可行的？")
_SUMM = (
    "Summarize the following passage in three sentences, preserving the key claims.\n\n"
    "Tiered memory systems for language-model serving keep a small, fast pool of "
    "high-bandwidth memory resident with the model and demote colder state to host "
    "memory or NVMe. The promise is to serve far longer contexts than would fit in "
    "device memory, at the cost of recall latency whenever a demoted block is needed "
    "again. Whether this trade is favorable depends on how predictable the working "
    "set is: if the set of useful past positions shifts slowly, a controller can "
    "prefetch the about-to-be-hot blocks and hide the recall behind compute; if it "
    "shifts abruptly, the controller pays a stall. Measuring that predictability, "
    "and bounding the probability of dropping a position that turns out to matter, "
    "is the central design question.")
_MATH = (
    "Solve this step by step and state the final answer clearly.\n\n"
    "A serving cluster has 8 GPUs. Each GPU holds a KV cache budget of 40 GB. A "
    "request at 32k context uses 5 GB of KV per 8k tokens. If 60% of a GPU's budget "
    "must stay free for bursts, how many simultaneous 32k-context requests can one "
    "GPU hold, and how many can the cluster hold? Then, if an eviction policy frees "
    "30% of each request's KV with no quality loss, recompute both numbers.")
_CHAT = (
    "Let's just chat. Tell me about a hobby you think more people should try, why it "
    "is rewarding, what a beginner needs to get started, the most common mistake new "
    "people make, and one small first project that would take less than an afternoon. "
    "Keep it warm and concrete, like you are talking to a friend over coffee.")

DOMAIN_POOL = [("code", _CODE), ("zh-qa", _ZH), ("summ", _SUMM),
               ("math", _MATH), ("chat", _CHAT)]


_SHAREGPT_PATH = ("/public/data_zoo/huggingface/hub/"
                  "datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/"
                  "ShareGPT_V3_unfiltered_cleaned_split.json")


def build_conversations_sharegpt(n_convs: int, turns: int, seed: int = 0,
                                 max_chars_per_turn: int = 1600):
    """Real multi-turn ShareGPT conversations (natural topic drift) — external
    validity vs the forced topic-switch pool. Returns the first `turns` human-turn
    texts of conversations that have at least `turns` human turns."""
    import json
    import random
    d = json.load(open(_SHAREGPT_PATH))
    rng = random.Random(seed)
    rng.shuffle(d)
    convs, topics = [], []
    for row in d:
        cv = row.get("conversations") or []
        humans = [c.get("value", "") for c in cv if c.get("from") in ("human", "user")]
        humans = [h for h in humans if len(h.strip()) >= 16]
        if len(humans) < turns:
            continue
        convs.append([h[:max_chars_per_turn] for h in humans[:turns]])
        topics.append([f"t{k}" for k in range(turns)])
        if len(convs) >= n_convs:
            break
    if len(convs) < n_convs:
        raise SystemExit(f"ShareGPT yielded only {len(convs)}/{n_convs} convs with >={turns} turns")
    return convs, topics


def build_conversations(n_convs: int, turns: int, seed: int = 0):
    """Each conversation cycles domains in a per-conversation rotated order, so the
    turn-boundary topic switch is guaranteed and the domain at a given turn varies
    across conversations (de-correlates boundary from absolute step)."""
    import random
    rng = random.Random(seed)
    convs, topics = [], []
    for ci in range(n_convs):
        order = list(range(len(DOMAIN_POOL)))
        rng.shuffle(order)
        seq = [order[(ci + k) % len(order)] for k in range(turns)]
        convs.append([DOMAIN_POOL[i][1] for i in seq])
        topics.append([DOMAIN_POOL[i][0] for i in seq])
    return convs, topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convs", type=int, default=48)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--conv-start", type=int, default=0)
    ap.add_argument("--conv-end", type=int, default=-1)
    ap.add_argument("--max-new-tokens-per-turn", type=int, default=48)
    ap.add_argument("--max-total-tokens", type=int, default=12000)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-suffix", default="multiturn")
    ap.add_argument("--source", choices=["topicswitch", "sharegpt"], default="topicswitch")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xqp.multiturn_trace_extract import extract_multiturn_traces

    models = DEFAULT_MODELS
    if a.models:
        models = [(Path(m).name, m) if Path(m).is_dir()
                  else (m.split("/")[-1], str(MODEL_ZOO / m.split("/")[-1])) for m in a.models]
    for _, p in models:
        if not Path(p).is_dir():
            raise SystemExit(f"model path missing: {p}")

    outdir = Path(a.out_dir) if a.out_dir else (ROOT / "experiments" / "traces")
    outdir.mkdir(parents=True, exist_ok=True)

    if a.source == "sharegpt":
        convs_all, topics_all = build_conversations_sharegpt(a.n_convs, a.turns, a.seed)
    else:
        convs_all, topics_all = build_conversations(a.n_convs, a.turns, a.seed)
    end = a.n_convs if a.conv_end < 0 else a.conv_end
    convs, topics = convs_all[a.conv_start:end], topics_all[a.conv_start:end]
    print(f"[mt] {len(convs)} conversations x {a.turns} turns, convs[{a.conv_start}:{end}] "
          f"on {a.device} worker={a.worker_id}", flush=True)

    summary = {}
    for stem, mpath in models:
        tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            mpath, torch_dtype=torch.float16, attn_implementation="eager",
            output_attentions=True, local_files_only=True).to(a.device).eval()
        out_path = outdir / f"{stem}.{a.out_suffix}.jsonl"
        t0 = time.time()
        n = extract_multiturn_traces(
            str(out_path), conversations=convs, topics=topics, model=model, tokenizer=tok,
            device=a.device, block_size=a.block_size,
            max_new_tokens_per_turn=a.max_new_tokens_per_turn,
            max_total_tokens=a.max_total_tokens, request_id_start=a.conv_start)
        dt = round(time.time() - t0, 1)
        summary[stem] = dict(rows=n, seconds=dt, out=str(out_path))
        print(json.dumps({"model": stem, "rows": n, "seconds": dt, "out": str(out_path)}), flush=True)
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
