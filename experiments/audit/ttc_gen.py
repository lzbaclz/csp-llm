"""P4 test-time-compute data: reasoning-token budget allocation.

Per query: one long reasoning trace; then BUDGET-FORCING at caps b in a grid -> outcome_i(b)
(truncate thinking at b tokens, force an answer). Signals: prompt-only self-rated difficulty
(cheap, pre-generation) and mid-gen early-answer consistency (costs probe tokens -> charged).
Model: a thinking model (Qwen3-8B) if it emits long reasoning, else Qwen2.5-7B CoT.
"""
import json, os, re, argparse
import torch
from sample_gen import extract_gsm8k, extract_boxed, norm

CAPS = [64, 128, 256, 512, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/public/model_zoo/Qwen3-8B")
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cuda").eval()
    rows = [json.loads(l) for l in open(f"data/{a.dataset}.jsonl")][:a.n]
    ext = extract_gsm8k if a.dataset == "gsm8k" else extract_boxed

    def chat(content, think=True):
        try:
            return tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=think)
        except Exception:
            return tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)

    def batch_gen(prompts, max_new, temp=0.0):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=(temp > 0), temperature=(temp or 1.0),
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
        return [o[L:] for o in out]                        # token ids of the continuation

    instr = "Solve the problem step by step, then give the final answer after 'Answer:'."
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w")
    for b0 in range(0, len(rows), a.bs):
        batch = rows[b0:b0 + a.bs]
        # 1) long reasoning trace per query
        long_prompts = [chat(f"{instr}\n\nProblem: {r['q']}", think=bool(a.think)) for r in batch]
        traces = batch_gen(long_prompts, max_new=max(CAPS), temp=0.7)
        trace_txt = [tok.decode(t, skip_special_tokens=True) for t in traces]
        # 2) prompt-only difficulty self-rating (cheap)
        diff_prompts = [chat(f"Rate the difficulty of this problem from 1 (trivial) to 5 (very hard). Reply with just the number.\n\nProblem: {r['q']}", think=False) for r in batch]
        diffs = batch_gen(diff_prompts, max_new=8, temp=0.0)
        # 3) budget-forced answers at each cap -- CLOSE the thinking block (</think>) before
        #    asking for the answer (canonical budget-forcing, S1/Muennighoff), not re-opening it.
        cap_answers = {b: [] for b in CAPS}
        for b in CAPS:
            fp = []
            for bi, r in enumerate(batch):
                cut = tok.decode(traces[bi][:b], skip_special_tokens=False)
                base = chat(f"{instr}\n\nProblem: {r['q']}", think=bool(a.think))
                think_so_far = cut.split("</think>")[0]            # reasoning before any close
                fp.append(base + think_so_far + "\n</think>\n\nThe final answer is:")
            outs = batch_gen(fp, max_new=24, temp=0.0)
            for bi in range(len(batch)):
                cap_answers[b].append(norm(ext(tok.decode(outs[bi], skip_special_tokens=True))))
        for bi, r in enumerate(batch):
            gold = norm(r["ans"])
            outcome = {b: int(cap_answers[b][bi] == gold and cap_answers[b][bi] != "") for b in CAPS}
            dm = re.search(r"[1-5]", tok.decode(diffs[bi], skip_special_tokens=True))
            fout.write(json.dumps({"ds": a.dataset, "qi": b0 + bi, "gold": gold,
                                   "outcome": outcome, "diff_prompt": int(dm.group()) if dm else 3,
                                   "early": [cap_answers[CAPS[0]][bi], cap_answers[CAPS[1]][bi]],
                                   "trace_len": len(traces[bi])}) + "\n")
        fout.flush()
        if (b0 // a.bs) % 3 == 0: print(f"[{a.dataset}] {b0+len(batch)}/{len(rows)}", flush=True)
    fout.close()
    print(f"DONE {a.out}")


if __name__ == "__main__":
    main()
