"""Dependency-free Okapi BM25 retriever with an inverted index.

External validity: instead of feeding the gold context (an oracle retriever that always
contains the answer), we pool every query's document in a dataset into ONE corpus,
chunk to ~100-word passages, and return the BM25 top-k. Retrieval is realistic and
IMPERFECT -- top-k can miss the evidence (esp. multi-hop), so the retrieve/skip decision
is non-trivial. This is the realistic version of the "retrieve" action.
"""
import re, math
from collections import defaultdict, Counter

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return _TOK.findall(s.lower())


def chunk_text(text, words_per_chunk=100):
    w = text.split()
    return [" ".join(w[i:i + words_per_chunk]) for i in range(0, len(w), words_per_chunk)] or [""]


class BM25:
    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.toks = [tokenize(p) for p in passages]
        self.len = [len(t) for t in self.toks]
        self.avgdl = (sum(self.len) / len(self.len)) if self.len else 0.0
        self.N = len(passages)
        # inverted index: term -> list of (doc_id, tf)
        self.inv = defaultdict(list)
        df = Counter()
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            for term, f in tf.items():
                self.inv[term].append((i, f))
                df[term] += 1
        self.idf = {term: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for term, n in df.items()}

    def _score(self, query):
        q = tokenize(query)
        scores = defaultdict(float)
        for term in q:
            if term not in self.inv:
                continue
            idf = self.idf[term]
            for (i, f) in self.inv[term]:
                dl = self.len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        return scores

    def retrieve(self, query, k=3):
        top = sorted(self._score(query).items(), key=lambda kv: -kv[1])[:k]
        return [self.passages[i] for i, _ in top]

    def retrieve_scored(self, query, k=3):
        """Returns (passages, scores) sorted desc -- scores are a cheap retrieval-QUALITY signal."""
        top = sorted(self._score(query).items(), key=lambda kv: -kv[1])[:k]
        return [self.passages[i] for i, _ in top], [s for _, s in top]


def build_dataset_index(rows, words_per_chunk=100):
    """rows: list of dicts with 'context'. Pool ALL contexts -> chunks -> one BM25 index."""
    passages = []
    for r in rows:
        passages.extend(chunk_text(r.get("context", ""), words_per_chunk))
    passages = [p for p in passages if p.strip()]
    return BM25(passages)


if __name__ == "__main__":
    import json
    LB = "/public/data_zoo/longbench/data"
    rows = [json.loads(l) for l in open(f"{LB}/hotpotqa.jsonl")][:20]
    bm = build_dataset_index(rows)
    print("corpus passages:", bm.N)
    q = rows[0]["input"]
    print("Q:", q[:100])
    for p in bm.retrieve(q, k=2):
        print(" ->", p[:120])
