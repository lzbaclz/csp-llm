#!/usr/bin/env python3
"""Multi-value needle-in-a-haystack (RULER multi-key/multi-value) as LongBench JSONL.

H2O's documented failure mode that the single-needle test missed: plant K distinct
*labeled* facts, then ask for ONE by its label. All needles have similar attention
magnitude (all are "secret codes"), so a magnitude/cumulative selector (H2O) keeps a
blend and at a tight budget may evict the queried one; a *query-aware* selector that
uses the question's label can keep the right needle. This is exactly where Quest /
SnapKV / question-aware selection is supposed to beat H2O.

Output: {context, input, answers} JSONL, readable by SEER's longbench loader
(LONGBENCH_PATH). Each prompt picks a random target label among K needles.
"""
import json, argparse, random

CITIES = ["Lisbon", "Berlin", "Cairo", "Tokyo", "Oslo", "Lima", "Quebec", "Nairobi",
          "Madrid", "Dublin", "Vienna", "Bogota", "Helsinki", "Manila", "Accra", "Riga"]

FILLER = ("The grass was green and the sky was blue, and the quiet afternoon stretched "
          "on without any particular event worth recording in this long document. ")


def make_prompt(rng, k_needles, approx_chars):
    cities = rng.sample(CITIES, k_needles)
    # distinct 5-digit codes
    codes = {c: rng.randint(10000, 99999) for c in cities}
    needles = [f"\nThe secret {c} access code is {codes[c]}.\n" for c in cities]
    # build haystack of filler, insert needles at evenly spaced depths (jittered)
    reps = max(1, approx_chars // len(FILLER))
    chunks = [FILLER] * reps
    n = len(chunks)
    # insertion points spread 10%..90%
    for i, ndl in enumerate(needles):
        frac = (i + 1) / (k_needles + 1)
        pos = min(n, int(frac * n) + rng.randint(-2, 2))
        chunks.insert(max(0, pos), ndl)
    context = "".join(chunks)
    target = rng.choice(cities)
    q = f"What is the secret {target} access code? Reply with only the 5-digit number."
    return {"context": context, "input": q, "answers": [str(codes[target])],
            "_target": target, "_all": codes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=8, help="needles per prompt")
    ap.add_argument("--chars", type=int, default=60000, help="approx haystack chars (~16K tok)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.out, "w") as f:
        for _ in range(a.n):
            r = make_prompt(rng, a.k, a.chars)
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.n} multi-value NIAH prompts (k={a.k}) -> {a.out}")


if __name__ == "__main__":
    main()
