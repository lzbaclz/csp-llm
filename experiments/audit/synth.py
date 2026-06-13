"""Synthesis: collect every operating point across the 3 VALID areas (RAG, sampling,
routing; TTC dropped as harness-limited), apply Benjamini-Hochberg multiple-testing
correction over the whole family, and test the base-accuracy confound r(floor_acc, captured).
"""
import json, numpy as np
from protocol import audit_point, binary_outcome_table
import samp_audit, rag_audit, route_audit

ROWS = []   # (label, area, floor_acc, m_rand, p, captured)


def add(label, area, floor, r):
    ROWS.append((label, area, floor, r["method_minus_random"], r["p"], r["frac_captured"]))


# ---- RAG ----
for path, tag in [("../rcrg/results/llama_bm25.jsonl", "RAG-Llama"), ("../rcrg/results/qwen_bm25.jsonl", "RAG-Qwen")]:
    g, o, c = rag_audit.load(path); n = len(g); outcome, budgets = binary_outcome_table(c, o, 1.0)
    order = np.argsort(g)
    for p in [0.2, 0.3, 0.5, 0.7]:
        alloc = np.zeros(n); alloc[order[:int(p*n)]] = 1.0
        add(f"{tag} retr{p}", "RAG", float(c.mean()), audit_point(outcome, budgets, alloc))

# ---- Sampling ----
for path, tag in [("results/samp_qwen_gsm8k.jsonl", "Samp-Qwen-GSM8K"), ("results/samp_qwen_math.jsonl", "Samp-Qwen-MATH"),
                  ("results/samp_llama_gsm8k.jsonl", "Samp-Llama-GSM8K"), ("results/samp_llama_math.jsonl", "Samp-Llama-MATH")]:
    try:
        outcome, budgets, answers = samp_audit.load(path)
    except FileNotFoundError:
        continue
    floor = float(outcome[:, 0].mean())
    for name, alloc in [("ESCw3", samp_audit.esc_alloc(answers, 3)), ("ESCw5", samp_audit.esc_alloc(answers, 5)),
                        ("ASC.7", samp_audit.asc_alloc(answers, 0.7)), ("ASC.8", samp_audit.asc_alloc(answers, 0.8)), ("ASC.9", samp_audit.asc_alloc(answers, 0.9))]:
        add(f"{tag} {name}", "Sampling", floor, audit_point(outcome, budgets, alloc))

# ---- Routing ----
for sp, bp, tag in [("results/samp_qwen_gsm8k.jsonl", "results/route_big_gsm8k.jsonl", "Route-GSM8K"),
                    ("results/route_small_mmlu.jsonl", "results/route_big_mmlu.jsonl", "Route-MMLU")]:
    try:
        sc, conf = route_audit.load_small(sp); bc = route_audit.load_big(bp)
    except FileNotFoundError:
        continue
    n = min(len(sc), len(bc)); sc, conf, bc = sc[:n], conf[:n], bc[:n]
    outcome, budgets = binary_outcome_table(sc, bc, 1.0); order = np.argsort(conf)
    for p in [0.1, 0.2, 0.3, 0.5]:
        alloc = np.zeros(n); alloc[order[:int(p*n)]] = 1.0
        add(f"{tag} big{p}", "Routing", float(sc.mean()), audit_point(outcome, budgets, alloc))

# ---- Test-time compute (fixed </think> harness) ----
import ttc_audit
for path, tag in [("results/ttc_gsm8k.jsonl", "TTC-GSM8K"), ("results/ttc_math.jsonl", "TTC-MATH")]:
    try:
        outcome, budgets, diff, midhard = ttc_audit.load(path)
    except FileNotFoundError:
        continue
    floor = float(outcome[:, 0].mean())
    mapA = {1: 64, 2: 128, 3: 256, 4: 512, 5: 1024}; mapB = {1: 128, 2: 256, 3: 512, 4: 1024, 5: 1024}
    add(f"{tag} promptA", "TTC", floor, audit_point(outcome, budgets, np.array([mapA[int(d)] for d in diff], float) + 8))
    add(f"{tag} promptB", "TTC", floor, audit_point(outcome, budgets, np.array([mapB[int(d)] for d in diff], float) + 8))
    add(f"{tag} midgen", "TTC", floor, audit_point(outcome, budgets, np.where(midhard, 1024.0, 128.0)))

# ---- Benjamini-Hochberg over the whole family ----
m = len(ROWS)
ps = sorted((r[4], i) for i, r in enumerate(ROWS))
q = 0.05; kmax = 0
for rank, (p, i) in enumerate(ps, 1):
    if p <= rank / m * q:
        kmax = rank
bh_thresh = ps[kmax - 1][0] if kmax > 0 else -1
reject = {i for rank, (p, i) in enumerate(ps, 1) if rank <= kmax}
bonf = 0.05 / m

print(f"=== Matched-budget audit synthesis: {m} operating points across 3 areas ===")
print(f"BH q=0.05 -> {kmax} significant ; Bonferroni alpha={bonf:.4f}")
print(f"{'cell':>22} {'area':>9} {'floor':>6} {'m-rand':>8} {'p':>7} {'capt%':>6} {'BH':>4} {'Bonf':>5}")
for i, (label, area, floor, mr, p, capt) in enumerate(ROWS):
    bh = "sig" if i in reject else "-"
    bf = "sig" if p < bonf else "-"
    print(f"{label:>22} {area:>9} {floor:>6.2f} {mr:>+8.3f} {p:>7.3f} {100*capt:>5.0f}% {bh:>4} {bf:>5}")

# confound: r(floor_acc, captured) over binary-budget comparable cells (sampling+routing)
sr = [(f, c) for (l, a, f, mr, p, c) in ROWS if a in ("Sampling", "Routing") and np.isfinite(c)]
fa = np.array([x[0] for x in sr]); ca = np.array([x[1] for x in sr])
r = float(np.corrcoef(fa, ca)[0, 1])
print(f"\nConfound: r(floor_accuracy, captured%) over {len(sr)} sampling+routing cells = {r:+.2f}")
print(f"BH-significant cells: {sorted(ROWS[i][0] for i in reject)}")
