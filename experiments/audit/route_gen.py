"""P3 routing data: per query, a model's greedy correctness (+ optional self-consistency
confidence). Run small (Qwen2.5-7B) and big (Qwen2.5-32B); route by small-model confidence.
"""
import json, os, re, argparse
import torch
from sample_gen import extract_gsm8k, extract_boxed, norm

LETTERS = ["A", "B", "C", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500", "mmlu"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--conf_k", type=int, default=0, help="self-consistency samples for confidence (0=skip)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cuda").eval()
    rows = [json.loads(l) for l in open(f"data/{a.dataset}.jsonl")][:a.n]

    def prompt(r):
        if a.dataset == "mmlu":
            ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
            return f"Answer with the single letter (A/B/C/D).\n\nQuestion: {r['q']}\n{ch}\nAnswer:"
        return f"Solve the problem. Think step by step, then give the final answer after 'Answer:'.\n\nProblem: {r['q']}"

    def gen(p, n=1, temp=0.0, mx=320):
        msg = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids): ids = ids["input_ids"]
        ids = ids.to("cuda"); L = ids.shape[1]
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=mx, do_sample=(temp > 0), temperature=(temp or 1.0),
                                 top_p=0.95, num_return_sequences=n, pad_token_id=tok.eos_token_id)
        return [tok.decode(o[L:], skip_special_tokens=True) for o in out]

    def parse(txt, r):
        if a.dataset == "mmlu":
            m = re.search(r"\b([ABCD])\b", txt.split("Answer:")[-1][:20])
            return m.group(1) if m else ""
        seg = txt.split("Answer:")[-1] if "Answer:" in txt else txt
        return (extract_gsm8k if a.dataset == "gsm8k" else extract_boxed)(seg)

    def goldnorm(r):
        return LETTERS[r["ans"]] if a.dataset == "mmlu" else norm(r["ans"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w")
    for qi, r in enumerate(rows):
        g = gen(prompt(r), n=1, temp=0.0)[0]
        ans = parse(g, r); gold = goldnorm(r)
        correct = int((ans if a.dataset == "mmlu" else norm(ans)) == gold and ans != "")
        conf = None
        if a.conf_k > 0:
            import collections
            ss = gen(prompt(r), n=a.conf_k, temp=0.8)
            aa = [(parse(x, r) if a.dataset == "mmlu" else norm(parse(x, r))) for x in ss]
            top = collections.Counter(aa).most_common(1)[0][1]
            conf = top / len(aa)
        fout.write(json.dumps({"ds": a.dataset, "qi": qi, "correct": correct, "conf": conf}) + "\n")
        if (qi + 1) % 50 == 0:
            fout.flush(); print(f"[{a.dataset}] {qi+1}/{len(rows)}", flush=True)
    fout.close()
    print(f"DONE {a.out}")


if __name__ == "__main__":
    main()
