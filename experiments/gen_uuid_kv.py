#!/usr/bin/env python3
"""UUID key-value retrieval (RULER kv_retrieval style) in a DIVERSE real-text haystack.

The fairest retrieval test: K random hex key->value pairs scattered in real LongBench
text; query one key by its (non-vocab) hex string; answer is the random hex VALUE the
model must actually retrieve (no vocab/world-knowledge prior, unlike city names). This
is the standard the critics demanded to kill the lexical-memorization shortcut.
"""
import json, argparse, random, os

SRCS = ["/public/data_zoo/longbench/data/gov_report.jsonl",
        "/public/data_zoo/longbench/data/multi_news.jsonl",
        "/public/data_zoo/longbench/data/qasper.jsonl"]


def real_text(approx):
    blob, tot = [], 0
    for src in SRCS:
        if not os.path.exists(src):
            continue
        for line in open(src, encoding="utf-8"):
            try:
                t = json.loads(line).get("context", "")
            except Exception:
                continue
            if t:
                blob.append(t); tot += len(t)
            if tot > approx * 4:
                return "\n".join(blob)
    return "\n".join(blob)


def hexid(rng):
    return "".join(rng.choice("0123456789abcdef") for _ in range(8))


def make(rng, big, k, approx):
    start = rng.randint(0, max(1, len(big) - approx - 1))
    chunks = [c + ". " for c in big[start:start + approx].split(". ") if c.strip()]
    pairs = [(hexid(rng), hexid(rng)) for _ in range(k)]
    needles = [f"\nKey {kk} maps to value {vv}.\n" for kk, vv in pairs]
    n = len(chunks)
    for i, ndl in enumerate(needles):
        pos = min(n, max(0, int((i + 1) / (k + 1) * n) + rng.randint(-2, 2)))
        chunks.insert(pos, ndl)
    ctx = "".join(chunks)
    tk, tv = rng.choice(pairs)
    q = f"What value does key {tk} map to? Reply with only the 8-character value."
    return {"context": ctx, "input": q, "answers": [tv], "_k": tk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chars", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    big = real_text(a.chars * (a.n + 4))
    assert len(big) > a.chars * 2
    with open(a.out, "w") as f:
        for _ in range(a.n):
            f.write(json.dumps(make(rng, big, a.k, a.chars)) + "\n")
    print(f"wrote {a.n} UUID-kv retrieval (k={a.k}, {a.chars}c) -> {a.out}")


if __name__ == "__main__":
    main()
