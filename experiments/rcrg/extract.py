"""RCRG data extraction: per query, is it answerable closed-book vs open-book, and a
cheap retrieval-free GATE signal.

"retrieve"  = include the dataset's context (the retrieved passage)   -> open-book
"skip"      = question only                                            -> closed-book
For each query we record:
  closed_correct : closed-book answer correct? (substring/F1 vs gold)
  open_correct   : open-book (with context) answer correct?
  gate_score     : retrieval-free confidence that the model KNOWS it closed-book
                   = self-consistency agreement over K closed-book samples (high => skip)
  + closed logprob-margin as a 2nd cheap signal.
The CRC layer (rcrg.py) then calibrates a gate threshold with a certified bound on
accuracy-loss vs always-retrieve. Output: one JSONL row per (dataset, query).
"""
import json, os, argparse, collections, math
import torch


def f1(pred, gold):
    pt = pred.lower().split(); gt = gold.lower().split()
    if not pt or not gt:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p = ns / len(pt); r = ns / len(gt)
    return 2 * p * r / (p + r)


def correct(pred, golds):
    pl = pred.lower()
    return 1 if any(g.lower() in pl or f1(pred, g) >= 0.5 for g in golds if g) else 0


def load_qa(path, n, ctx_chars=200000):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ans = r.get("answers") or []
        if not ans:
            continue
        rows.append({"context": (r.get("context") or "")[:ctx_chars],
                     "q": r.get("input") or "", "answers": ans})
        if len(rows) >= n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/public/model_zoo/Llama-3.1-8B-Instruct")
    ap.add_argument("--datasets", nargs="+",
                    default=["triviaqa", "hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=5, help="closed-book self-consistency samples for gate")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "gold"])
    ap.add_argument("--topk", type=int, default=3, help="BM25 passages retrieved")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from retriever import build_dataset_index
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cuda")
    model.eval()
    LB = "/public/data_zoo/longbench/data"

    def gen(prompt, n=1, temp=0.0, max_new=32):
        msg = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):                      # transformers 5.x may return a dict
            ids = ids["input_ids"]
        ids = ids.to("cuda"); L = ids.shape[1]
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=(temp > 0),
                                 temperature=(temp or 1.0), num_return_sequences=n,
                                 pad_token_id=tok.eos_token_id)
        return [tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w")
    for ds in a.datasets:
        path = f"{LB}/{ds}.jsonl"
        if not os.path.exists(path):
            print("skip missing", ds); continue
        rows = load_qa(path, a.n)
        bm = build_dataset_index(rows) if a.retriever == "bm25" else None
        retr_hit = 0
        nc = 0
        for r in rows:
            q, golds = r["q"], r["answers"]
            if bm is not None:
                ctx = "\n".join(bm.retrieve(q, k=a.topk))      # realistic imperfect retrieval
                retr_hit += 1 if any(g.lower() in ctx.lower() for g in golds if g) else 0
            else:
                ctx = r["context"][:6000]                       # gold-context oracle (old setup)
            # closed-book greedy answer + correctness
            cb = gen(f"Answer concisely.\nQuestion: {q}\nAnswer:", n=1, temp=0.0)[0]
            cb_corr = correct(cb, golds)
            # gate score = self-consistency agreement over K closed-book samples
            samples = gen(f"Answer concisely.\nQuestion: {q}\nAnswer:", n=a.k, temp=0.7)
            norm = [s.lower().strip().strip(".") for s in samples]
            top = collections.Counter(norm).most_common(1)[0][1] if norm else 0
            agree = top / max(1, len(norm))            # in [1/k, 1]: high => confident => skip
            # open-book (with RETRIEVED context) answer + correctness
            ob = gen(f"Use the context to answer concisely.\nContext: {ctx}\nQuestion: {q}\nAnswer:",
                     n=1, temp=0.0)[0]
            ob_corr = correct(ob, golds)
            fout.write(json.dumps({"ds": ds, "closed_correct": cb_corr, "open_correct": ob_corr,
                                   "gate_agree": agree}) + "\n")
            nc += 1
        fout.flush()
        rh = retr_hit / max(1, nc)
        print(f"[{ds}] n={nc} retr_hit_rate={rh:.2f}")
    fout.close()
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
