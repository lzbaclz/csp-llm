"""Headline-scale faithful-Quest refutation with clustered CIs (A10/A9).

Answers the strongest systems-reviewer attack: "is the query--key redundancy a
BLOCK-GRANULARITY artifact? At token granularity Quest's per-token, per-head
max q.k should matter." We re-extract, for ALL FOUR architectures, the faithful
Quest signal (f_query_dotmax = max over heads,tokens of raw q.k -- exactly what
Quest pages exploit) at block sizes bs in {32, 16 (Quest-native page), 1 (token
granularity)} and measure, with REQUEST-CLUSTERED bootstrap CIs:
  * single-view AUC of each query variant vs within/cross
  * marginal AUC of each query variant OVER within+cross (the decisive number)
  * within-EMA-tercile marginal of faithful dotmax (does query help where
    attention magnitude is BLIND, i.e. low within-EMA?)
  * unique-information certificate uCMI = I(X_i;Y|X_{-i}) for [within,cross,dotmax]

Crash-safe: each (model, bs) cell writes a .done sentinel; re-running skips done
cells. Extraction (GPU) and analysis (CPU) are separable via --no-extract /
--no-analyze so CIs can be recomputed without a GPU.

    python experiments/run_quest_headline.py --device cuda:0 \
        --out experiments/results/quest_headline.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "SEER"))

import numpy as np
from scipy.stats import rankdata

from xqp.predictor import ClosedFormXQP
from xqp.info_theory import unique_information_report

ZOO = "/public/model_zoo"
MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-8B",
    "Mistral-7B-Instruct-v0.3",
]
FEATS = ["f_within", "f_cross", "f_query", "f_query_dotmean", "f_query_dotmax"]
QUERY_VARIANTS = ["f_query", "f_query_dotmean", "f_query_dotmax"]

# (block_size, n_prompts, max_new_tokens, max_context).  bs=1 is the token-
# granularity extreme; we shrink n/steps/ctx there to keep the trace tractable
# while still giving >=16 request groups for a clustered CI.
CELLS = [
    dict(bs=32, n=32, mnt=64, ctx=4096),
    dict(bs=16, n=32, mnt=64, ctx=4096),
    dict(bs=1,  n=16, mnt=32, ctx=2048),
]

MAX_ANALYZE_ROWS = 12_000_000   # subsample (group-preserving) above this for AUC speed
N_BOOT = 1000
SEED = 0


def auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


# --------------------------------------------------------------------------- #
# extraction (GPU)
# --------------------------------------------------------------------------- #
def extract_all(device, tmpdir, models):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.makedirs(tmpdir, exist_ok=True)
    from seer.trace.datasets import load_prompts
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from xqp.attn_trace_extract import extract_attention_traces

    max_n = max(c["n"] for c in CELLS)
    max_ctx = max(c["ctx"] for c in CELLS)
    for stem in models:
        # skip the model entirely if all its cells are done
        if all(os.path.exists(cell_path(tmpdir, stem, c["bs"]) + ".done") for c in CELLS):
            print(f"[skip] {stem}: all cells done", flush=True)
            continue
        mp = os.path.join(ZOO, stem)
        print(f"[load] {stem} from {mp}", flush=True)
        tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
        prompts = load_prompts("mooncake", [max_ctx], max_n, tokenizer=None)
        model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.float16, attn_implementation="eager",
            output_attentions=True, local_files_only=True).to(device).eval()
        for c in CELLS:
            outp = cell_path(tmpdir, stem, c["bs"])
            if os.path.exists(outp + ".done"):
                print(f"  [skip] bs={c['bs']} done", flush=True)
                continue
            ids = []
            for p in prompts[: c["n"]]:
                t = tok(p, return_tensors="pt").input_ids[:, : c["ctx"]].to(device)
                ids.append(t)
            print(f"  [extract] bs={c['bs']} n={c['n']} mnt={c['mnt']} ctx={c['ctx']} -> {outp}",
                  flush=True)
            try:
                n_rows = extract_attention_traces(
                    mp, None, outp, model=model, tokenizer=tok, input_ids=ids,
                    device=device, block_size=c["bs"], max_new_tokens=c["mnt"],
                    query_variants=True)
                open(outp + ".done", "w").write(str(n_rows))
                print(f"    wrote {n_rows} rows", flush=True)
            except Exception as e:  # keep going to the next cell/model
                print(f"    !! FAILED bs={c['bs']} {stem}: {e}", flush=True)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def cell_path(tmpdir, stem, bs):
    return os.path.join(tmpdir, f"{stem}.bs{bs}.quest.jsonl")


# --------------------------------------------------------------------------- #
# analysis (CPU)
# --------------------------------------------------------------------------- #
def load_cell(path, seed=SEED):
    cols = {c: [] for c in FEATS + ["y_h4", "request_id"]}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for c in cols:
                cols[c].append(r.get(c, 0.0))
    d = {c: np.asarray(cols[c], np.float32) for c in FEATS + ["y_h4"]}
    rid_raw = np.asarray(cols["request_id"])
    finite = np.isfinite(np.stack([d[c] for c in FEATS], 1)).all(1)
    drop = int((~finite).sum())
    for c in FEATS + ["y_h4"]:
        d[c] = d[c][finite]
    rid_raw = rid_raw[finite]
    rid_map = {}
    d["rid"] = np.asarray([rid_map.setdefault(v, len(rid_map)) for v in rid_raw], np.int32)
    d["_dropped"] = drop
    d["_n_full"] = int(finite.sum())
    # group-preserving subsample for tractable AUC if the cell is huge
    n = d["rid"].size
    if n > MAX_ANALYZE_ROWS:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=MAX_ANALYZE_ROWS, replace=False)
        keep.sort()
        for c in FEATS + ["y_h4", "rid"]:
            d[c] = d[c][keep]
    d["_n"] = int(d["rid"].size)
    return d


def split(rid, seed=SEED):
    uniq = np.unique(rid); rng = np.random.default_rng(seed)
    te = set(rng.permutation(uniq)[: max(1, len(uniq) // 4)].tolist())
    is_te = np.isin(rid, list(te))
    return ~is_te, is_te


def clustered_ci_delta(y, sa, sb, groups, n_boot=N_BOOT, seed=SEED):
    """Clustered-bootstrap CI of auc(sa)-auc(sb), resampling whole requests."""
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb); groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    obs = auc(y, sa) - auc(y, sb)
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        yy = y[rows]
        if yy.sum() == 0 or yy.sum() == yy.size:
            continue
        da = auc(yy, sa[rows]); db = auc(yy, sb[rows])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    deltas = np.asarray(deltas, np.float64)
    if deltas.size == 0:
        return dict(delta=float(obs), lo=float("nan"), hi=float("nan"), n_groups=int(len(uniq)))
    return dict(delta=float(obs), lo=float(np.percentile(deltas, 2.5)),
                hi=float(np.percentile(deltas, 97.5)),
                p_gt0=float(np.mean(deltas > 0)), n_groups=int(len(uniq)))


def marginal_with_ci(d, col, seed=SEED):
    tr, ev = split(d["rid"], seed)
    y = d["y_h4"].astype(np.float32)
    z = np.zeros_like(d["f_within"])
    base = np.stack([d["f_within"], d["f_cross"], z, z], 1)
    aug = np.stack([d["f_within"], d["f_cross"], zscore(d[col]), z], 1)
    bm = ClosedFormXQP.from_fit(base[tr], y[tr])
    am = ClosedFormXQP.from_fit(aug[tr], y[tr])
    sb = bm.score(base[ev]); sa = am.score(aug[ev])
    ci = clustered_ci_delta(y[ev], sa, sb, d["rid"][ev], seed=seed)
    return dict(base=auc(y[ev], sb), aug=auc(y[ev], sa), **ci)


def analyze_cell(d, seed=SEED):
    y_all = d["y_h4"]
    sv = {c: auc(y_all, d[c]) for c in FEATS}
    marg = {col: marginal_with_ci(d, col, seed) for col in QUERY_VARIANTS}
    # within-EMA tercile marginal of faithful dotmax
    tr, ev = split(d["rid"], seed)
    y = d["y_h4"].astype(np.float32); z = np.zeros_like(d["f_within"])
    base = np.stack([d["f_within"], d["f_cross"], z, z], 1)
    aug = np.stack([d["f_within"], d["f_cross"], zscore(d["f_query_dotmax"]), z], 1)
    bm = ClosedFormXQP.from_fit(base[tr], y[tr]); am = ClosedFormXQP.from_fit(aug[tr], y[tr])
    w_ev = d["f_within"][ev]; yev = y[ev]; bs_ = bm.score(base[ev]); gs = am.score(aug[ev])
    q1, q2 = np.quantile(w_ev, [1 / 3, 2 / 3])
    terc = {}
    for nm, m in [("low", w_ev <= q1), ("mid", (w_ev > q1) & (w_ev <= q2)), ("high", w_ev > q2)]:
        terc[nm] = dict(base=auc(yev[m], bs_[m]), aug=auc(yev[m], gs[m]),
                        delta=auc(yev[m], gs[m]) - auc(yev[m], bs_[m]))
    F3 = np.stack([d["f_within"], d["f_cross"], d["f_query_dotmax"]], 1)
    ui = unique_information_report(F3, y_all, feature_names=["within", "cross", "query_dotmax"])
    return dict(n=d["_n"], n_full=d["_n_full"], dropped_nonfinite=d["_dropped"],
                n_requests=int(d["rid"].max() + 1), pos_rate=float(y_all.mean()),
                single_view_auc=sv, marginal=marg, within_tercile_dotmax=terc, unique_info=ui)


def analyze_all(tmpdir, models, out_path, seed=SEED):
    per_cell = {}
    pooled_by_bs = {}
    for c in CELLS:
        bs = c["bs"]
        loaded = {}
        for stem in models:
            p = cell_path(tmpdir, stem, bs)
            if not os.path.exists(p):
                print(f"[analyze] MISSING {p}", flush=True); continue
            print(f"[analyze] bs={bs} {stem} ...", flush=True)
            d = load_cell(p, seed)
            loaded[stem] = d
            per_cell.setdefault(stem, {})[f"bs{bs}"] = analyze_cell(d, seed)
        # pool models within this bs (namespace rids)
        if loaded:
            off = 0; pool = {cc: [] for cc in FEATS + ["y_h4"]}; rids = []
            for stem, d in loaded.items():
                for cc in FEATS + ["y_h4"]:
                    pool[cc].append(d[cc])
                rids.append(d["rid"] + off); off += int(d["rid"].max() + 1)
            pd = {cc: np.concatenate(pool[cc]) for cc in FEATS + ["y_h4"]}
            pd["rid"] = np.concatenate(rids)
            pd["_n"] = int(pd["y_h4"].size); pd["_n_full"] = pd["_n"]; pd["_dropped"] = 0
            pooled_by_bs[f"bs{bs}"] = analyze_cell(pd, seed)

    out = dict(cells=CELLS, models=models, n_boot=N_BOOT, seed=seed,
               per_model=per_cell, pooled_by_bs=pooled_by_bs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)

    print("\n=== FAITHFUL-QUERY HEADLINE (4 archs x bs, clustered CIs) ===")
    print(f"{'cell':<10}{'within':>7}{'cross':>7}{'dotmax':>7}"
          f"{'marg+dotmax [95% CI]':>30}{'  lowW Δ':>9}{'  uCMI(q)':>10}")
    for bs in [f"bs{c['bs']}" for c in CELLS]:
        if bs not in pooled_by_bs:
            continue
        r = pooled_by_bs[bs]
        sv = r["single_view_auc"]; mg = r["marginal"]["f_query_dotmax"]
        lowd = r["within_tercile_dotmax"]["low"]["delta"]
        uq = r["unique_info"]["query_dotmax"]["unique_cmi"]
        ci = f"{mg['delta']:+.4f} [{mg['lo']:+.4f},{mg['hi']:+.4f}]"
        print(f"POOL.{bs:<5}{sv['f_within']:>7.3f}{sv['f_cross']:>7.3f}"
              f"{sv['f_query_dotmax']:>7.3f}{ci:>30}{lowd:>+9.4f}{uq:>10.4f}", flush=True)
    print(f"\nWROTE {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tmpdir", default="/public/xqp_traces/quest_headline")
    ap.add_argument("--out", default="experiments/results/quest_headline.json")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--block-sizes", nargs="*", type=int, default=None,
                    help="restrict CELLS to these block sizes (e.g. 32 for big models)")
    ap.add_argument("--n", type=int, default=None,
                    help="override prompts/cell (smaller for big models)")
    ap.add_argument("--ctx", type=int, default=None,
                    help="override context length (shorter so eager attn fits big models)")
    a = ap.parse_args()
    if a.block_sizes or a.n or a.ctx:
        global CELLS
        CELLS = [dict(c, n=(a.n or c["n"]), ctx=(a.ctx or c["ctx"])) for c in CELLS
                 if (not a.block_sizes or c["bs"] in a.block_sizes)]
    if not a.no_extract:
        extract_all(a.device, a.tmpdir, a.models)
    if not a.no_analyze:
        analyze_all(a.tmpdir, a.models, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
