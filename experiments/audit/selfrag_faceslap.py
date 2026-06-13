"""Face-slap: run Self-RAG's OWN learned retrieve-gate (the [Retrieval]-token probability)
on real QA + a real BM25 retriever, then test it against the matched-budget random control.

Gate signal per query = softmax P([Retrieval]) over {[Retrieval],[No Retrieval]} at the first
response step (Self-RAG's exact decision variable). We then ask: at Self-RAG's own realized
retrieval rate, does its gate select better WHICH queries to retrieve than random? (oracle anchor)
Caveat: we use BM25 over the dataset corpus, not the paper's Contriever-Wikipedia, so absolute
accuracy differs from the headline; the gate-vs-random comparison is internally valid and
corroborated by Self-Routing RAG / TARG.
"""
import json, os, argparse, collections
import torch
from retriever import build_dataset_index

LB = "/public/data_zoo/longbench/data"


def f1(pred, gold):
    pt, gt = pred.lower().split(), gold.lower().split()
    if not pt or not gt: return 0.0
    c = collections.Counter(pt) & collections.Counter(gt); ns = sum(c.values())
    if ns == 0: return 0.0
    p, r = ns / len(pt), ns / len(gt); return 2 * p * r / (p + r)


def correct(pred, golds):
    pl = pred.lower()
    return int(any(g.lower() in pl or f1(pred, g) >= 0.5 for g in golds if g))


def strip_reflect(t):
    for tk in ["[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]", "[Continue to Use Evidence]",
               "[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[Utility:1]",
               "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "<paragraph>", "</paragraph>"]:
        t = t.replace(tk, " ")
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/public/model_zoo/selfrag_llama2_7b")
    ap.add_argument("--datasets", nargs="+",
                    default=["triviaqa", "hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="results/selfrag_gate.jsonl")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda").eval()
    ret_id = tok.convert_tokens_to_ids("[Retrieval]"); noret_id = tok.convert_tokens_to_ids("[No Retrieval]")
    print("ret_id", ret_id, "noret_id", noret_id)

    def load_qa(path, n):
        rows = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); ans = r.get("answers") or []
            if not ans: continue
            q = r.get("input") or ""
            if "Question:" in q: q = q.split("Question:")[-1].split("Answer:")[0].strip()
            rows.append({"q": q[:500], "answers": ans, "context": (r.get("context") or "")})
            if len(rows) >= n: break
        return rows

    def gen(prompt, max_new=64):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=False)

    def gate_prob(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            lg = model(ids).logits[0, -1]
        pr, pn = lg[ret_id].item(), lg[noret_id].item()
        import math
        m = max(pr, pn); er, en = math.exp(pr - m), math.exp(pn - m)
        return er / (er + en)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w")
    for ds in a.datasets:
        path = f"{LB}/{ds}.jsonl"
        if not os.path.exists(path): continue
        rows = load_qa(path, a.n)
        bm = build_dataset_index(rows)
        for r in rows:
            q = r["q"]
            base = f"### Instruction:\n{q}\n\n### Response:\n"
            g = gate_prob(base)                                      # Self-RAG's gate signal
            cb = strip_reflect(gen(base, 64))                        # closed-book
            ctx = "\n".join(bm.retrieve(q, k=3))
            ob = strip_reflect(gen(base + f"[Retrieval]<paragraph>{ctx[:2000]}</paragraph>", 64))
            fout.write(json.dumps({"ds": ds, "gate": g,
                                   "closed_correct": correct(cb, r["answers"]),
                                   "open_correct": correct(ob, r["answers"])}) + "\n")
        fout.flush(); print(f"[{ds}] n={len(rows)}", flush=True)
    fout.close()
    print("DONE", a.out)


if __name__ == "__main__":
    main()
