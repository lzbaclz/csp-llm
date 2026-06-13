"""Cheap retrieval-QUALITY features per query (CPU only, no GPU), aligned to the existing
llama_bm25.jsonl / qwen_bm25.jsonl row order. These give the gate a signal for whether
RETRIEVAL will help (predict o), complementing self-consistency (which predicts c).

Features (all available BEFORE the expensive open-book generation):
  bm25_top1, bm25_top3_mean, bm25_gap (top1-top3), q_len, overlap (query-passage lexical),
  ret_len. NOTE: we do NOT use answer-containment (that leaks the label).
"""
import json, numpy as np
from retriever import build_dataset_index, tokenize

LB = "/public/data_zoo/longbench/data"
DATASETS = ["triviaqa", "hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
N = 200


def load_qa(path, n, ctx_chars=200000):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not (r.get("answers") or []):
            continue
        rows.append({"context": (r.get("context") or "")[:ctx_chars], "q": r.get("input") or ""})
        if len(rows) >= n:
            break
    return rows


def main():
    fout = open("results/features.jsonl", "w")
    for ds in DATASETS:
        rows = load_qa(f"{LB}/{ds}.jsonl", N)
        bm = build_dataset_index(rows)
        for r in rows:
            q = r["q"]
            passages, scores = bm.retrieve_scored(q, k=3)
            qt = set(tokenize(q)); rt = set(tokenize(" ".join(passages)))
            overlap = len(qt & rt) / max(1, len(qt))
            top1 = scores[0] if scores else 0.0
            top3m = float(np.mean(scores)) if scores else 0.0
            gap = (scores[0] - scores[-1]) if len(scores) > 1 else 0.0
            fout.write(json.dumps({"ds": ds, "bm25_top1": top1, "bm25_top3_mean": top3m,
                                   "bm25_gap": gap, "q_len": len(qt), "overlap": overlap,
                                   "ret_len": len(tokenize(" ".join(passages)))}) + "\n")
        print(f"[{ds}] features for {len(rows)}")
    fout.close()
    print("WROTE results/features.jsonl")


if __name__ == "__main__":
    main()
