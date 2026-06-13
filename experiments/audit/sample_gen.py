"""P2 adaptive-sampling data: K CoT samples per query -> per-sample (answer, correct).
Offline we replay ASC/ESC stopping and audit vs permuted-random + oracle at matched #samples.
"""
import json, os, re, argparse, collections
import torch


def extract_gsm8k(text):
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].replace("$", "").rstrip(".") if nums else ""


def extract_boxed(text):
    i = text.rfind("\\boxed")
    if i >= 0:
        j = text.find("{", i)
        if j >= 0:
            d = 0
            for k in range(j, len(text)):
                if text[k] == "{": d += 1
                elif text[k] == "}":
                    d -= 1
                    if d == 0:
                        return text[j + 1:k].strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else ""


def norm(s):
    return re.sub(r"\s+", "", str(s).lower().replace("$", "").replace("\\!", "").rstrip("."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500", "triviaqa"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--bs", type=int, default=8, help="queries per batch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cuda").eval()
    rows = [json.loads(l) for l in open(f"data/{a.dataset}.jsonl")][:a.n]
    is_qa = a.dataset == "triviaqa"
    ext = extract_gsm8k if a.dataset == "gsm8k" else extract_boxed
    instr = ("Answer the question concisely. Give only the short answer after 'Answer:'." if is_qa
             else "Solve the problem. Think step by step, then give the final answer after 'Answer:'.")
    field = "Question" if is_qa else "Problem"
    mxnew = 32 if is_qa else 288

    def qa_correct(pred, golds):
        p = norm(pred)
        return int(p != "" and any(norm(g) in p or p in norm(g) for g in golds))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w")
    done = 0
    for b0 in range(0, len(rows), a.bs):
        batch = rows[b0:b0 + a.bs]
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{instr}\n\n{field}: {r['q']}"}],
                                           add_generation_prompt=True, tokenize=False) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=mxnew, do_sample=True, temperature=0.8, top_p=0.95,
                                 num_return_sequences=a.K, pad_token_id=tok.eos_token_id)
        for bi, r in enumerate(batch):
            samples = []
            for o in out[bi * a.K:(bi + 1) * a.K]:
                txt = tok.decode(o[L:], skip_special_tokens=True)
                seg = txt.split("Answer:")[-1] if "Answer:" in txt else txt
                if is_qa:
                    ans = seg.strip().split("\n")[0][:60]
                    samples.append({"a": norm(ans)[:40], "c": qa_correct(ans, r["ans"])})
                else:
                    ans = ext(seg)
                    samples.append({"a": norm(ans), "c": int(norm(ans) == norm(r["ans"]) and ans != "")})
            gold = "list" if is_qa else norm(r["ans"])
            fout.write(json.dumps({"ds": a.dataset, "qi": b0 + bi, "gold": gold, "samples": samples}) + "\n")
        done += len(batch); fout.flush()
        if (b0 // a.bs) % 5 == 0: print(f"[{a.dataset}] {done}/{len(rows)}", flush=True)
    fout.close()
    print(f"DONE {a.out}")


if __name__ == "__main__":
    main()
