#!/usr/bin/env python3
"""Multi-value NIAH with a REAL, diverse-text haystack (not repeated filler).

The repeated-filler version is the most-favorable-possible setup for H2O to fail
(cumulative attention concentrates on the one repeated sentence). The fair, standard
NIAH practice uses diverse real text as the haystack. Here the filler is drawn from
real LongBench documents (gov_report / multi_news / qasper contexts), so H2O's
cumulative attention is spread over genuine content -- a much harder test for the
"query-aware wins" claim. K labeled needles inserted at spread depths; query one.
"""
import json, argparse, random, os

CITIES = ["Lisbon", "Berlin", "Cairo", "Tokyo", "Oslo", "Lima", "Quebec", "Nairobi",
          "Madrid", "Dublin", "Vienna", "Bogota", "Helsinki", "Manila", "Accra", "Riga"]
SRCS = ["/public/data_zoo/longbench/data/gov_report.jsonl",
        "/public/data_zoo/longbench/data/multi_news.jsonl",
        "/public/data_zoo/longbench/data/qasper.jsonl"]


def real_text_pool(approx_chars_needed):
    """Concatenate real LongBench contexts into one big diverse text blob."""
    blob = []
    total = 0
    for src in SRCS:
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = r.get("context", "")
                if t:
                    blob.append(t)
                    total += len(t)
                if total > approx_chars_needed * 4:
                    return "\n".join(blob)
    return "\n".join(blob)


def make_prompt(rng, big_text, k_needles, approx_chars):
    # take a random diverse window of real text as the haystack
    start = rng.randint(0, max(1, len(big_text) - approx_chars - 1))
    haystack = big_text[start:start + approx_chars]
    # split into sentence-ish chunks to insert needles between
    chunks = haystack.split(". ")
    chunks = [c + ". " for c in chunks if c.strip()]
    cities = rng.sample(CITIES, k_needles)
    codes = {c: rng.randint(10000, 99999) for c in cities}
    needles = [f"\nThe secret {c} access code is {codes[c]}.\n" for c in cities]
    n = len(chunks)
    for i, ndl in enumerate(needles):
        frac = (i + 1) / (k_needles + 1)
        pos = min(n, max(0, int(frac * n) + rng.randint(-2, 2)))
        chunks.insert(pos, ndl)
    context = "".join(chunks)
    target = rng.choice(cities)
    q = f"What is the secret {target} access code? Reply with only the 5-digit number."
    return {"context": context, "input": q, "answers": [str(codes[target])],
            "_target": target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chars", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    big = real_text_pool(a.chars * (a.n + 4))
    assert len(big) > a.chars * 2, f"not enough real text ({len(big)} chars)"
    with open(a.out, "w") as f:
        for _ in range(a.n):
            f.write(json.dumps(make_prompt(rng, big, a.k, a.chars)) + "\n")
    print(f"wrote {a.n} REAL-haystack multi-value NIAH (k={a.k}, {a.chars}c) -> {a.out}")


if __name__ == "__main__":
    main()
