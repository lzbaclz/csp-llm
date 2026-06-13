"""B1 — does the coverage guarantee HOLD in deployment when calibrated against the
SERVING oracle (vs the offline-trace tau, which E4 showed fails)?

Protocol (split-conformal, prompt-level): from the runner's --log-calib dump of
per-block (prompt_id, layer_pos, scorer prob, serving-oracle label), hold out half
the prompts. Calibrate a per-layer threshold tau_l on the CALIB prompts so that the
salient blocks' miss rate is <= alpha; evaluate the realized miss (FN) + emergent
budget on the disjoint TEST prompts. Contrast with the OFFLINE-trace tau (the E4
setting). If serving-calibrated FN tracks alpha while offline-tau FN does not, the
guarantee transfers to deployment once recalibrated against the serving oracle.

    python experiments/analyze_serving_calib.py \
        --calib experiments/results/serving_calib/calib_llama.json \
        --offline-budgeter experiments/predictors/guardkv_budgeter_a10_h4.json
"""
from __future__ import annotations
import argparse, json
import numpy as np


def conformal_tau(salient_scores, alpha):
    """Finite-sample split-conformal threshold: keep {s >= tau} misses <= alpha of
    salient blocks. tau = the floor(alpha*(n+1))-th smallest salient score."""
    s = np.sort(np.asarray(salient_scores, np.float64))
    n = s.shape[0]
    if n == 0:
        return 0.0
    k = min(max(int(np.floor(alpha * (n + 1))), 0), n - 1)
    return float(s[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--offline-budgeter", default=None)
    ap.add_argument("--alphas", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--out", default="experiments/results/serving_calib/SUMMARY.json")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    d = json.load(open(a.calib))
    rows = np.array(d["rows"], dtype=np.float64)   # [prompt_id, layer_pos, score, label]
    pid, lpos, score, label = rows[:, 0].astype(int), rows[:, 1], rows[:, 2], rows[:, 3]
    layers = np.round(lpos, 4)
    uniq_p = np.unique(pid)
    half = len(uniq_p) // 2
    calib_p, test_p = set(uniq_p[:half].tolist()), set(uniq_p[half:].tolist())
    cmask = np.isin(pid, list(calib_p)); tmask = np.isin(pid, list(test_p))
    print(f"rows={len(rows):,} prompts={len(uniq_p)} (calib {len(calib_p)}/test {len(test_p)}) "
          f"layers={len(np.unique(layers))} pos_rate={label.mean():.3f}", flush=True)

    # offline-trace tau curve (the E4 setting), interpolated by layer position
    off = None
    if a.offline_budgeter:
        bg = json.load(open(a.offline_budgeter))
        nc = sorted(bg.get("norm_curve", []))
        if nc:
            off = (np.array([p for p, _ in nc]), np.array([t for _, t in nc]))

    out = {"n_rows": len(rows), "n_prompts": len(uniq_p), "by_alpha": []}
    for al in alphas:
        # --- serving-side per-layer conformal tau on CALIB prompts ---
        tau_by_layer = {}
        for L in np.unique(layers):
            m = cmask & (layers == L) & (label > 0.5)
            if m.sum() >= 32:
                tau_by_layer[float(L)] = conformal_tau(score[m], al)
        tau_g = conformal_tau(score[cmask & (label > 0.5)], al)
        thr_test = np.array([tau_by_layer.get(float(L), tau_g) for L in layers[tmask]])
        sc_t, lab_t = score[tmask], label[tmask]
        keep = sc_t >= thr_test
        sal = lab_t > 0.5
        fn_serving = float((sal & ~keep).sum() / max(1, sal.sum()))
        budget_serving = float(keep.mean())
        # per-layer FN spread
        per_fn = []
        for L in np.unique(layers[tmask]):
            mm = (layers[tmask] == L) & sal
            if mm.sum() >= 8:
                per_fn.append(float((mm & ~keep).sum() / mm.sum()))
        # --- offline-trace tau on the SAME test (reproduces E4) ---
        fn_offline = None
        if off is not None:
            thr_off = np.interp(layers[tmask], off[0], off[1], left=off[1][0], right=off[1][-1])
            keep_off = sc_t >= thr_off
            fn_offline = float((sal & ~keep_off).sum() / max(1, sal.sum()))
        rec = dict(target_alpha=al,
                   serving_calibrated_FN=round(fn_serving, 4),
                   emergent_budget=round(budget_serving, 4),
                   per_layer_FN_std=round(float(np.std(per_fn)) if per_fn else 0.0, 4),
                   offline_tau_FN=round(fn_offline, 4) if fn_offline is not None else None)
        out["by_alpha"].append(rec)
        print(f"  alpha={al:.2f}: serving-calib FN={fn_serving:.3f} (target {al}) "
              f"budget={budget_serving:.3f} per-layer FN std={rec['per_layer_FN_std']:.3f}"
              + (f" | offline-tau FN={fn_offline:.3f} (E4 failure)" if fn_offline is not None else ""),
              flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
