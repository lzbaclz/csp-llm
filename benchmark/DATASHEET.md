# KVSalienceBench — dataset card

A benchmark for **streaming KV-cache saliency prediction**: predict, per KV
*block*, the probability it will be among the top-10% most-attended blocks a few
decode steps later, from cheap per-block features — the prediction problem a
serving system solves to decide which blocks stay resident under a memory budget.

## Motivation
LLM KV caches grow unboundedly; only a small fraction of past blocks receive most
attention at any step. Prior eviction heuristics (H2O, SnapKV, Quest, InfiniGen)
each bet on one signal. We recast the underlying *prediction* problem as a
first-class streaming, class-imbalanced, drift-prone learning task and release a
corpus + frozen protocol + calibrated reference baseline so methods are compared
apples-to-apples, and so the *structure* of the task (which signals matter, how
hard it is per workload) is documented rather than folklore.

## Composition
- **Source:** real per-step attention traces from greedy decoding of HuggingFace
  causal LMs on locked A100 silicon (no synthetic attention).
- **Models (4 attention families):** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct,
  Qwen3-8B (QK-norm), Mistral-7B-Instruct-v0.3.
- **Workloads (8):** `mooncake` (conversation/KV-trace), `sharegpt` (chat, short
  context) + 6 LongBench tasks — `narrativeqa`, `hotpotqa` (multi-doc QA),
  `gov_report`, `multi_news` (summarization), `lcc` (code), `multifieldqa_zh`
  (Chinese QA). Spans QA / multi-doc QA / summarization / code / Chinese and
  short↔long context.
- **Scale:** 8 workloads × 4 models × 128 prompts = **4096 request groups**
  (≈1536 (model,workload) cells × ... ; 128 prompts per cell), context ≤4096
  tokens, 128 decode steps, 32-token blocks. ~545M+ labeled rows.
- **Per-row schema:** `request_id, layer, step, block_idx, f_within, f_cross,
  f_query, f_pos, y_h1, y_h4, y_h16, y_h64`. Features: within-layer attention EMA
  (`s_within`), previous-layer top-r indicator (`s_cross`), Quest cosine proxy
  (`s_query`), recency (`s_pos`). Labels: top-10% membership at horizons {1,4,16,64}.
- **Integrity:** every cell passed a request-level leakage-safe + completeness
  check (`scripts/verify_trace_splits.py --expect 128`); zero synthetic-fallback
  contamination (strict-workload collection + quarantine); fp16-NaN rows dropped.

## Task & protocol (frozen — see `benchmark/protocol.py`)
- Predict `y_h4` (top-10% membership 4 steps ahead) from the 4 block features.
- **Request-level (group) hold-out split**, 25% test (whole prompts held out).
- **Metrics:** AUC, AUPRC (headline, imbalanced), P@10 / R@10 (selection),
  ECE + Brier (calibration — first-class). 95% CIs by request-clustered bootstrap.

## Reference baseline & difficulty atlas
The reference baseline (`benchmark/reference_model.json`) is a **3-parameter
calibrated logistic model on within+cross** (exact weights/metrics in the JSON;
regenerate with `benchmark/fit_reference.py`). Headline findings (`experiments/results/atlas.md`):

| | short-context (sharegpt) | long-context (7 workloads) |
|---|---|---|
| 2-view vs GBDT (ΔAUC) | −0.016 (GBDT wins) | −0.001 … −0.005 (near-parity) |
| calibration edge (GBDT_ECE / 2view_ECE) | 2.1× | 18–33× |

Universal across all 8 workloads: **query and recency are near-useless**
(single-view AUC ~0.55), **model complexity barely helps** (pairwise ≈ 2-view),
**concept drift is mild and online adaptation HURTS**. The 2-view law transfers
near-losslessly across **architectures** (0.0017 AUC drop) AND **workloads**
(mean 0.005 drop; `experiments/results/cross_workload_transfer.json`).

**Difficulty characterization (the contribution):** the "tiny calibrated model ≈
GBDT + large calibration edge" result is a **long-context property**, not
universal — short-context chat retains a GBDT edge and a small calibration gap.

## Recommended uses & the open challenge
Beat the 3-parameter reference on **AUPRC and calibration (ECE) jointly**,
especially on **short-context** workloads where GBDT keeps an edge. Submit a
`score(F)` (see `benchmark/submit_template.py`); score with
`benchmark/run_leaderboard.py`.

## Limitations / scope (honest)
- 128 prompts/cell, context ≤4096, greedy decode, 4 ~7–8B models — not a claim
  about >8B models, very-long context (>4k), or sampling-based decode.
- `s_query` is a mean-pooled cosine proxy. A faithful per-head-max Quest signal
  is a *stronger standalone* signal on some models but still carries ≈0 unique
  information beyond within+cross (`experiments/QUEST_VERDICT.md`); the
  "query-useless-at-block-granularity" finding survives the faithful signal.
- Adaptive conformal beats a *naive* fixed threshold for coverage, but an
  offline-calibrated fixed threshold matches it because intra-generation drift is
  mild — conformal's value is for *unknown/large* drift not stressed here.
- Synthetic workloads (RULER, pile→RULER) are deliberately excluded from the
  headline corpus (they conflict with the "real attention" premise); usable only
  as robustness ablations.
- Expanding to more model scales, >4k context, and multi-turn drift streams is the
  explicit open invitation; LongBench-v2 / InfiniteBench would need download + a
  loader wired in.
