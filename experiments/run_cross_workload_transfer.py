"""Cross-WORKLOAD transfer of the 2-view (within+cross) saliency model.

The paper shows the 2-view model transfers across ARCHITECTURES almost
losslessly. This asks the complementary, benchmark-level question: does the law
transfer across WORKLOADS? Fit the 3-parameter within+cross logistic on each
workload's held-out-request TRAIN split, evaluate it on every workload's TEST
split -> an 8x8 AUC matrix. Diagonal = in-domain; off-diagonal = transfer.
Small diagonal-minus-offdiagonal gap == the law is workload-portable.

Each workload pools its 4 architectures. Rows are capped per file (the 2-view
fit + AUC need only ~1e6 rows, not the full corpus) so this is fast and
memory-light. Non-finite rows are dropped (fp16 hygiene).

    python experiments/run_cross_workload_transfer.py --traces /public/xqp_traces
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xqp.predictor import ClosedFormXQP


def roc_auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def split_mw(nm):
    m, dot, w = nm.rpartition(".")
    if dot and re.match(r"^[a-z][a-z0-9_-]*$", w):
        return m, w
    return nm, "default"


def load_capped(path, cap):
    fw, fc, y, rid = [], [], [], []
    rmap = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            w_, c_ = r.get("f_within", 0.0), r.get("f_cross", 0.0)
            if not (np.isfinite(w_) and np.isfinite(c_)):
                continue
            fw.append(w_); fc.append(c_); y.append(r["y_h4"])
            rid.append(rmap.setdefault(r["request_id"], len(rmap)))
            if len(fw) >= cap:
                break
    return (np.asarray(fw, np.float32), np.asarray(fc, np.float32),
            np.asarray(y, np.int8), np.asarray(rid, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/public/xqp_traces")
    ap.add_argument("--cap", type=int, default=1_500_000, help="rows/file cap")
    ap.add_argument("--out", default="experiments/results/cross_workload_transfer.json")
    a = ap.parse_args()

    # group files by workload, pool the 4 architectures (namespace rid)
    by_wl = {}
    for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))):
        if ".smoke." in f or "quest" in f:
            continue
        _, w = split_mw(os.path.basename(f)[:-len(".jsonl")])
        by_wl.setdefault(w, []).append(f)

    data = {}
    for w, files in by_wl.items():
        FW, FC, Y, RID = [], [], [], []; off = 0
        for f in files:
            fw, fc, y, rid = load_capped(f, a.cap)
            FW.append(fw); FC.append(fc); Y.append(y); RID.append(rid + off)
            off += int(rid.max() + 1) if rid.size else 0
        fw = np.concatenate(FW); fc = np.concatenate(FC)
        y = np.concatenate(Y); rid = np.concatenate(RID)
        # request-level split
        uniq = np.unique(rid); rng = np.random.default_rng(0)
        te = set(rng.permutation(uniq)[: max(1, len(uniq) // 4)].tolist())
        is_te = np.isin(rid, list(te))
        z = np.zeros_like(fw)
        X = np.stack([fw, fc, z, z], 1)
        data[w] = dict(Xtr=X[~is_te], ytr=y[~is_te].astype(np.float32),
                       Xte=X[is_te], yte=y[is_te].astype(np.float32))
        print(f"[load] {w}: {fw.size:,} rows, {len(uniq)} requests", flush=True)

    wls = sorted(data)
    models = {w: ClosedFormXQP.from_fit(data[w]["Xtr"], data[w]["ytr"]) for w in wls}
    mat = {}
    for a_w in wls:
        mat[a_w] = {}
        for b_w in wls:
            mat[a_w][b_w] = roc_auc(data[b_w]["yte"], models[a_w].score(data[b_w]["Xte"]))

    # summary: per test-workload, in-domain AUC vs best foreign vs mean foreign
    diag = {w: mat[w][w] for w in wls}
    drops = []
    for b_w in wls:
        foreign = [mat[a_w][b_w] for a_w in wls if a_w != b_w]
        drops.append(diag[b_w] - float(np.mean(foreign)))
    summary = dict(mean_indomain=float(np.mean([diag[w] for w in wls])),
                   mean_transfer_drop=float(np.mean(drops)),
                   max_transfer_drop=float(np.max(drops)))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(workloads=wls, matrix=mat, diagonal=diag, summary=summary),
              open(a.out, "w"), indent=2)

    print("\n=== CROSS-WORKLOAD 2-VIEW TRANSFER (rows=test AUC; trained on col) ===")
    print("train\\test  " + "".join(f"{w[:9]:>10}" for w in wls))
    for a_w in wls:
        print(f"{a_w[:11]:<11}" + "".join(f"{mat[a_w][b_w]:>10.3f}" for b_w in wls))
    print(f"\nmean in-domain AUC = {summary['mean_indomain']:.4f}  "
          f"mean transfer drop = {summary['mean_transfer_drop']:+.4f}  "
          f"max drop = {summary['max_transfer_drop']:+.4f}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
