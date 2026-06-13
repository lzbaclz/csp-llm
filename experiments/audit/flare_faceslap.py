"""Face-slap on FLARE (Jiang et al., EMNLP 2023, training-free): its gate retrieves when the
model's drafted answer has a LOW-confidence token (min token probability < theta). We compute
FLARE's exact signal (min token prob of the closed-book draft) on real QA + BM25, then test it
against the matched-budget random control. Fully offline on open models (no checkpoint needed).
"""
import json, os, argparse, collections, math
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
    pl = pred.lower(); return int(any(g.lower() in pl or f1(pred, g) >= 0.5 for g in golds if g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/public/model_zoo/Llama-3.1-8B-Instruct")
    ap.add_argument("--datasets", nargs="+",
                    default=["triviaqa", "hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="results/flare_gate.jsonl")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda").eval()

    def load_qa(path, n):
        rows = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); ans = r.get("answers") or []
            if not ans: continue
            q = r.get("input") or ""
            if "Question:" in q: q = q.split("Question:")[-1].split("Answer:")[0].strip()
            rows.append({"q": q[:500], "answers": ans, "context": r.get("context") or ""})
            if len(rows) >= n: break
        return rows

    def gen_with_conf(prompt, max_new=40):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids): ids = ids["input_ids"]
        ids = ids.to("cuda"); L = ids.shape[1]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        seq = out.sequences[0][L:]
        txt = tok.decode(seq, skip_special_tokens=True)
        # FLARE signal: minimum token probability across the drafted answer
        min_p = 1.0
        for t, sc in enumerate(out.scores):
            if t >= len(seq): break
            p = torch.softmax(sc[0].float(), -1)[seq[t]].item()
            min_p = min(min_p, p)
        return txt, min_p

    def gen(prompt, max_new=40):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids): ids = ids["input_ids"]
        ids = ids.to("cuda"); L = ids.shape[1]
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][L:], skip_special_tokens=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w")
    for ds in a.datasets:
        path = f"{LB}/{ds}.jsonl"
        if not os.path.exists(path): continue
        rows = load_qa(path, a.n); bm = build_dataset_index(rows)
        for r in rows:
            q = r["q"]
            cb, min_p = gen_with_conf(f"Answer concisely.\nQuestion: {q}\nAnswer:")
            ctx = "\n".join(bm.retrieve(q, k=3))
            ob = gen(f"Use the context to answer concisely.\nContext: {ctx[:3000]}\nQuestion: {q}\nAnswer:")
            fout.write(json.dumps({"ds": ds, "flare_minprob": min_p,
                                   "closed_correct": correct(cb, r["answers"]),
                                   "open_correct": correct(ob, r["answers"])}) + "\n")
        fout.flush(); print(f"[{ds}] n={len(rows)}", flush=True)
    fout.close()
    print("DONE", a.out)


if __name__ == "__main__":
    main()
