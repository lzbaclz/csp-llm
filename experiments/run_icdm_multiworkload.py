"""Multi-workload ICDM analysis (external validity + tighter CIs).

Addresses two reviewer points at once:
  * #5 statistical base: the headline corpus is one workload (mooncake) × 32
    prompts. This driver pools the EXPANDED corpus (`<model>.<workload>.jsonl`,
    several workloads × N prompts × 4 models) and reports
    (a) a by-model headline with a proper GROUP-LEVEL bootstrap (resample whole
        held-out requests, never row-subsample-then-cluster), and
    (b) a per-workload table showing the within+cross-dominant / query-irrelevant
        pattern holds on every workload (external validity).
  * #4 IO scalability: the cells are 1--15 GB JSONL. We never load a file; a
    STREAMING SAMPLER does one pass with a cheap substring boundary check and
    reservoir-samples only the <1% of rows it keeps (JSON-parsing only those),
    so memory is O(caps) regardless of file size.

    python experiments/run_icdm_multiworkload.py --traces /public/xqp_traces \
        --out experiments/results/icdm_multiworkload.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import rankdata

from xqp.dm_metrics import average_precision, precision_at_k, recall_at_k, expected_calibration_error
from xqp.predictor import ClosedFormXQP
from xqp.baselines import all_learned_baselines

KNOWN_MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-8B",
                "Mistral-7B-Instruct-v0.3", "Llama-3.2-3B-Instruct"]
BMARK = '"layer": 0, "step": 0, "block_idx": 0,'
TRAIN_CAP = 80_000          # reservoir size per cell (train rows)
TEST_PER_REQ_CAP = 2_000    # test rows kept per held-out request
TEST_FRAC_MOD = 4           # rid % 4 == 3 -> held out (25%)
N_BOOT = 300
SEED = 0


def parse_cell(path: str):
    stem = os.path.basename(path)[: -len(".jsonl")]
    for m in KNOWN_MODELS:
        if stem == m or stem.startswith(m + "."):
            return m, (stem[len(m) + 1:] or "all")
    return (stem.rsplit(".", 1) + ["all"])[:2]


def stream_sample_cell(path: str, seed=SEED):
    """One streaming pass: request-level split by rid%4, reservoir-sample train,
    cap test per request. JSON-parses only kept rows. Uses stdlib ``random`` for
    the per-line reservoir decision (numpy scalar RNG is ~20x slower per call,
    which dominates on 70M-line cells)."""
    import random as _random
    rnd = _random.Random(seed)
    rid = -1
    tr_F = np.empty((TRAIN_CAP, 4), np.float32); tr_y = np.empty(TRAIN_CAP, np.float32)
    tr_n = 0; tr_seen = 0
    te_F, te_y, te_g = [], [], []
    te_count: dict[int, int] = {}
    isfinite = np.isfinite

    def _feat(r):
        return (r["f_within"], r["f_cross"], r["f_query"], r["f_pos"])

    with open(path) as fh:
        for line in fh:
            if BMARK in line:
                rid += 1
            if rid % TEST_FRAC_MOD == 3:                      # held out (test)
                c = te_count.get(rid, 0)
                if c >= TEST_PER_REQ_CAP:
                    continue
                r = json.loads(line); f = _feat(r)
                if not all(map(isfinite, f)):
                    continue
                te_F.append(f); te_y.append(r["y_h4"]); te_g.append(rid)
                te_count[rid] = c + 1
            else:                                             # train (reservoir)
                tr_seen += 1
                if tr_n < TRAIN_CAP:
                    r = json.loads(line); f = _feat(r)
                    if not all(map(isfinite, f)):
                        continue
                    tr_F[tr_n] = f; tr_y[tr_n] = r["y_h4"]; tr_n += 1
                elif rnd.random() * tr_seen < TRAIN_CAP:       # replace w/ prob CAP/seen
                    r = json.loads(line); f = _feat(r)
                    if all(map(isfinite, f)):
                        j = rnd.randrange(TRAIN_CAP); tr_F[j] = f; tr_y[j] = r["y_h4"]
    return dict(Ftr=tr_F[:tr_n], ytr=tr_y[:tr_n],
                Fte=np.asarray(te_F, np.float32), yte=np.asarray(te_y, np.float32),
                gte=np.asarray(te_g, np.int32), n_requests=rid + 1)


def auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def group_bootstrap_ci(y, s, groups, n_boot=N_BOOT, seed=SEED):
    """Resample whole groups (held-out requests) with replacement."""
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        v = auc(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return dict(mean=float(np.mean(vals)), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), n_groups=int(len(uniq)))


def fit_closed(F, y, cols):
    mask = np.zeros(4, np.float32); mask[cols] = 1.0
    return ClosedFormXQP.from_fit(F * mask, y), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/public/xqp_traces")
    ap.add_argument("--out", default="experiments/results/icdm_multiworkload.json")
    ap.add_argument("--require-done", action="store_true", default=True)
    ap.add_argument("--models", nargs="*", default=None, help="filter models (validation)")
    ap.add_argument("--workloads", nargs="*", default=None, help="filter workloads (validation)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.traces, "*.jsonl")))
    cells = {}   # (model, workload) -> sample dict
    for f in files:
        if a.require_done and not os.path.exists(f + ".done"):
            continue
        model, wl = parse_cell(f)
        if a.models and model not in a.models:
            continue
        if a.workloads and wl not in a.workloads:
            continue
        print(f"[stream] {model} / {wl} ...", flush=True)
        cells[(model, wl)] = stream_sample_cell(f)
    if not cells:
        print("no usable cells", file=sys.stderr); return 1
    models = sorted({m for m, _ in cells})
    workloads = sorted({w for _, w in cells})
    print(f"models={models}\nworkloads={workloads}", flush=True)

    # ---------- by-MODEL pooled (workloads pooled within a model) ----------
    def pool(keys):
        Ftr = np.concatenate([cells[k]["Ftr"] for k in keys])
        ytr = np.concatenate([cells[k]["ytr"] for k in keys])
        Fte = np.concatenate([cells[k]["Fte"] for k in keys])
        yte = np.concatenate([cells[k]["yte"] for k in keys])
        # group id = global per (cell, rid)
        g = np.concatenate([cells[k]["gte"] + 100000 * i for i, k in enumerate(keys)])
        return Ftr, ytr, Fte, yte, g

    all_keys = list(cells.keys())
    Ftr, ytr, Fte, yte, gte = pool(all_keys)
    print(f"[pooled] train={Ftr.shape[0]:,} test={Fte.shape[0]:,} groups={len(np.unique(gte))}", flush=True)

    # subsample train for learned baselines (fair: shared)
    rng = np.random.default_rng(SEED)
    tr_idx = rng.choice(Ftr.shape[0], min(120_000, Ftr.shape[0]), replace=False)
    Ftr_s, ytr_s = Ftr[tr_idx], ytr[tr_idx]

    cf2, m2 = fit_closed(Ftr_s, ytr_s, [0, 1])      # within+cross (the minimal model)
    cf4, m4 = fit_closed(Ftr_s, ytr_s, [0, 1, 2, 3])
    gbdt = all_learned_baselines(Ftr_s, ytr_s, seed=SEED)[0]   # LightGBM
    methods = {
        "within+cross(2)": cf2.score(Fte * m2),
        "XQP-closed(4)": cf4.score(Fte * m4),
        "GBDT": gbdt.score(Fte),
        "H2O(within)": Fte[:, 0],
        "Quest(query)": Fte[:, 2],
    }
    headline = {}
    for name, s in methods.items():
        ci = group_bootstrap_ci(yte, s, gte)
        headline[name] = dict(auc=auc(yte, s), auc_lo=ci["lo"], auc_hi=ci["hi"],
                              auprc=average_precision(yte, s),
                              p_at_10=precision_at_k(yte, s, 0.10),
                              ece=expected_calibration_error(yte, s))
        print(f"  {name:18s} AUC={headline[name]['auc']:.4f} "
              f"[{ci['lo']:.4f},{ci['hi']:.4f}] AUPRC={headline[name]['auprc']:.3f} "
              f"P@10={headline[name]['p_at_10']:.3f} ECE={headline[name]['ece']:.4f}", flush=True)

    # ---------- per-WORKLOAD external-validity table ----------
    per_workload = {}
    for wl in workloads:
        keys = [k for k in cells if k[1] == wl]
        Ftr_w, ytr_w, Fte_w, yte_w, _ = pool(keys)
        tri = rng.choice(Ftr_w.shape[0], min(80_000, Ftr_w.shape[0]), replace=False)
        cf2w, m2w = fit_closed(Ftr_w[tri], ytr_w[tri], [0, 1])
        per_workload[wl] = dict(
            n_models=len({k[0] for k in keys}), n_test=int(Fte_w.shape[0]),
            pos_rate=float(yte_w.mean()),
            auc_within=auc(yte_w, Fte_w[:, 0]), auc_cross=auc(yte_w, Fte_w[:, 1]),
            auc_query=auc(yte_w, Fte_w[:, 2]), auc_recency=auc(yte_w, Fte_w[:, 3]),
            auc_2view=auc(yte_w, cf2w.score(Fte_w * m2w)))
        v = per_workload[wl]
        print(f"  [wl] {wl:26s} within={v['auc_within']:.3f} cross={v['auc_cross']:.3f} "
              f"query={v['auc_query']:.3f} 2view={v['auc_2view']:.3f} (n={v['n_test']:,})", flush=True)

    # ---------- transfer by MODEL (pool workloads) ----------
    per_model_fit = {}
    for mdl in models:
        keys = [k for k in cells if k[0] == mdl]
        Ftr_m, ytr_m, _, _, _ = pool(keys)
        tri = rng.choice(Ftr_m.shape[0], min(80_000, Ftr_m.shape[0]), replace=False)
        mu = Ftr_m[tri].mean(0); sd = Ftr_m[tri].std(0) + 1e-6
        cf, mask = fit_closed((Ftr_m[tri] - mu) / sd, ytr_m[tri], [0, 1, 2, 3])
        per_model_fit[mdl] = (cf, mask, mu, sd)
    transfer = {}
    for a_m in models:
        transfer[a_m] = {}
        cf, mask, mu, sd = per_model_fit[a_m]
        for b_m in models:
            keys = [k for k in cells if k[0] == b_m]
            _, _, Fte_b, yte_b, _ = pool(keys)
            transfer[a_m][b_m] = auc(yte_b, cf.score(((Fte_b - mu) / sd) * mask))
    diag = np.mean([transfer[m][m] for m in models])
    off = np.mean([transfer[a][b] for a in models for b in models if a != b])

    out = dict(models=models, workloads=workloads,
               n_cells=len(cells),
               headline_by_model_pooled=headline,
               per_workload=per_workload,
               transfer=dict(matrix=transfer, mean_within=float(diag),
                             mean_cross=float(off), drop=float(diag - off)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\ntransfer: within={diag:.4f} cross={off:.4f} drop={diag-off:.4f}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
