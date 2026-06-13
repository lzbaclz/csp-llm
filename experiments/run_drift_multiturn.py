"""E5 analysis — does adaptive-conformal earn its place under multi-turn drift?

Evaluates three thresholding policies over **per-conversation, temporally ordered**
streams (NOT pooled across requests at a step index, which is what E3 did and what
hides per-request drift), at a target miss rate alpha:

  fixed_global       — one offline split-conformal tau (static H2O/SnapKV-style cut)
  coverage_per_layer — GuardKV's static per-layer split-conformal budgeter (E1)
  adaptive_conformal — ACI; one controller per conversation, warm-started at tau_g,
                       updates tau per decode step (absorbs drift)

Reports, pooled over conversations: per-step miss TAIL (mean/p90/p99/max, frac>2a);
the BOUNDARY-ALIGNED miss & budget trajectory (mean at offset from each turn switch,
the money figure); adaptive RECOVERY time after a switch; average budget; and the
cross-boundary salient-set Jaccard (drift magnitude, H1).

Modes:
  --mode multiturn   real persistent-cache multi-turn traces (the headline)
  --mode mechanism   CPU sanity check: inject a controlled tau*-drift into a real
                     single-turn stream to confirm (a) the comparator is correct,
                     (b) adaptive tracks a moving tau* while fixed blows out, and
                     (c) with NO injected drift adaptive ~= fixed (reproduces E3).

    python experiments/run_drift_multiturn.py --mode multiturn \
        --traces /public/xqp_traces_mt --out experiments/results/drift_multiturn.json
    python experiments/run_drift_multiturn.py --mode mechanism \
        --traces experiments/traces --out experiments/results/drift_mechanism.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xqp.predictor import ClosedFormXQP
from xqp.gated_predictor import _gather4
from xqp.budgeter import CoverageDrivenBudgeter, _conformal_tau
from xqp.conformal import AdaptiveConformalSaliency

H = "h4"
COLS = (0, 1)            # within + cross (the calibrated minimal scorer)
ALPHA = 0.10
GAMMA = 0.10
PRE, POST = 8, 20        # boundary-aligned window [-PRE, +POST]
REC_KEEP = 0.5           # recency-floor: unconditionally keep blocks with f_pos>=this
                         # (cold-start fix: protect just-created blocks until EMA warms)


# --------------------------------------------------------------------------- #
# loading (multi-turn schema: adds `turn`; request_id is distinct per conv)
# --------------------------------------------------------------------------- #
def load_mt_trace(path: str, namespace: str) -> dict | None:
    rid, layer, step, turn = [], [], [], []
    fw, fc, fq, fp = [], [], [], []
    yh = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rid.append(f"{namespace}:{r['request_id']}")
            layer.append(r["layer"]); step.append(r["step"]); turn.append(r.get("turn", 0))
            fw.append(r["f_within"]); fc.append(r["f_cross"])
            fq.append(r["f_query"]); fp.append(r["f_pos"])
            yh.append(r[f"y_{H}"])
    if not rid:
        return None
    F = np.stack([np.asarray(fw, np.float32), np.asarray(fc, np.float32),
                  np.asarray(fq, np.float32), np.asarray(fp, np.float32)], axis=1)
    finite = np.isfinite(F).all(axis=1)
    rid = np.asarray(rid)[finite]
    return dict(
        rid=rid,
        layer=np.asarray(layer, np.int32)[finite],
        step=np.asarray(step, np.int32)[finite],
        turn=np.asarray(turn, np.int32)[finite],
        F=F[finite],
        y=np.asarray(yh, np.int8)[finite],
    )


def pool_mt(traces: dict) -> dict:
    keys = ["rid", "layer", "step", "turn", "y"]
    out = {k: np.concatenate([t[k] for t in traces.values()]) for k in keys}
    out["F"] = np.concatenate([t["F"] for t in traces.values()])
    return out


def load_pooled(traces_dir, only=None, suffix="multiturn"):
    """Pool all matching traces into int-coded arrays, with an npz cache so repeated
    analyses skip the ~10-min JSONL reload. rid is factorized to int codes."""
    cache = os.path.join(traces_dir, f".cache_{only or 'all'}_{suffix}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"  [cache] {cache}: {len(z['rid']):,} rows, {len(np.unique(z['rid']))} convs", flush=True)
        return {k: z[k] for k in ("rid", "layer", "step", "turn", "F", "y")}
    files = sorted(glob.glob(os.path.join(traces_dir, f"*.{suffix}.jsonl")))
    if only:
        files = [f for f in files if only.lower() in os.path.basename(f).lower()]
    traces = {}
    for f in files:
        ns = os.path.basename(f).split(".")[0]
        t = load_mt_trace(f, ns)
        if t is not None:
            traces[ns] = t
            print(f"  loaded {ns}: rows={len(t['rid']):,}", flush=True)
    if not traces:
        raise SystemExit(f"no '*.{suffix}.jsonl' traces under {traces_dir}")
    d = pool_mt(traces)
    _, codes = np.unique(d["rid"], return_inverse=True)
    d["rid"] = codes.astype(np.int32)
    try:
        np.savez(cache, **{k: d[k] for k in ("rid", "layer", "step", "turn", "F", "y")})
        print(f"  [cache] wrote {cache}", flush=True)
    except Exception as e:
        print(f"  [cache] skip ({e})", flush=True)
    return d


def conv_split(rid, seed=0):
    """conversation-level 50/25/25 split: fit-scorer / calibrate-tau / test."""
    u = np.unique(rid); rng = np.random.default_rng(seed); rng.shuffle(u)
    a, b = int(0.5 * len(u)), int(0.75 * len(u))
    sel = lambda S: np.where(np.isin(rid, list(S)))[0]
    return sel(set(u[:a])), sel(set(u[a:b])), sel(set(u[b:]))


# --------------------------------------------------------------------------- #
# per-conversation ordered-stream evaluation
# --------------------------------------------------------------------------- #
def _tail(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return dict(mean=float("nan"), p90=float("nan"), p99=float("nan"),
                    max=float("nan"), frac_blowout=float("nan"), n=0)
    return dict(mean=round(float(x.mean()), 4), p90=round(float(np.percentile(x, 90)), 4),
                p99=round(float(np.percentile(x, 99)), 4), max=round(float(x.max()), 4),
                frac_blowout=round(float((x > 2 * ALPHA).mean()), 4), n=int(x.size))


def evaluate_policies(d, idx, scorer, tau_g, budgeter, *, min_sal=4):
    """Per-conversation ordered streams. Returns per-policy per-step miss arrays,
    boundary-aligned trajectories, budgets, recovery, drift Jaccard."""
    rid, layer, step, turn = d["rid"][idx], d["layer"][idx], d["step"][idx], d["turn"][idx]
    F, y = d["F"][idx], d["y"][idx].astype(np.float32)
    fpos = F[:, 3]                                       # recency feature
    p = np.asarray(scorer.score(_gather4(F, COLS)))
    tau_layer = budgeter._thresholds(layer)            # per-layer static thresholds

    miss = {k: [] for k in ("fixed_global", "coverage_per_layer",
                            "adaptive_conformal", "recency_floor", "fixed_matched")}
    budg = {k: [] for k in miss}
    # boundary-aligned accumulators: offset -> list of miss
    aligned = {k: {o: [] for o in range(-PRE, POST + 1)} for k in miss}
    recov, jacc_boundary, jacc_within = [], [], []

    for c in np.unique(rid):
        cm = rid == c
        csteps = np.unique(step[cm])
        cturn = turn[cm]
        # boundary steps: first step of each turn > 0
        bsteps = sorted({int(step[cm][cturn == tt].min()) for tt in np.unique(cturn) if tt > 0})
        # one adaptive controller per conversation, warm-started at tau_g
        aci_tau = float(tau_g)
        per_step = {k: {} for k in miss}
        for s in csteps:
            sm = cm & (step == s)
            ys = y[sm]; ps = p[sm]; ls = layer[sm]; fp = fpos[sm]
            nsal = int((ys > 0.5).sum())
            if nsal < min_sal:
                continue
            sal = ys > 0.5
            # fixed_global
            keep = ps >= tau_g
            per_step["fixed_global"][int(s)] = float((sal & ~keep).sum() / nsal)
            budg["fixed_global"].append(float(keep.mean()))
            # coverage_per_layer (static per-layer tau)
            keep_cl = ps >= tau_layer[sm]
            per_step["coverage_per_layer"][int(s)] = float((sal & ~keep_cl).sum() / nsal)
            budg["coverage_per_layer"].append(float(keep_cl.mean()))
            # adaptive_conformal (update tau on realized aggregate miss)
            keep_a = ps >= aci_tau
            m_a = float((sal & ~keep_a).sum() / nsal)
            per_step["adaptive_conformal"][int(s)] = m_a
            budg["adaptive_conformal"].append(float(keep_a.mean()))
            aci_tau = float(np.clip(aci_tau + GAMMA * (ALPHA - m_a), 0.0, 1.0))
            # recency_floor: static tau_g UNION a recency floor that protects just-
            # created (cold-EMA) blocks until their within-EMA warms up — the
            # cold-start fix for the boundary miss spike (StreamingLLM-style window).
            keep_rf = (ps >= tau_g) | (fp >= REC_KEEP)
            per_step["recency_floor"][int(s)] = float((sal & ~keep_rf).sum() / nsal)
            budg["recency_floor"].append(float(keep_rf.mean()))
            # CONTROL: fixed at recency_floor's SAME per-step budget, but selected by
            # pure score (top-k). Isolates "keep the RIGHT (cold) blocks" from "keep
            # MORE blocks". If recency_floor << fixed_matched, recency is genuinely
            # complementary at boundaries; if ~equal, it was only buying budget.
            k_rf = int(keep_rf.sum()); nb = len(ps)
            if k_rf >= nb:
                keep_fm = np.ones(nb, bool)
            elif k_rf <= 0:
                keep_fm = np.zeros(nb, bool)
            else:
                keep_fm = np.zeros(nb, bool)
                keep_fm[np.argpartition(-ps, k_rf - 1)[:k_rf]] = True
            per_step["fixed_matched"][int(s)] = float((sal & ~keep_fm).sum() / nsal)
            budg["fixed_matched"].append(float(keep_fm.mean()))

        for k in miss:
            for s, mv in per_step[k].items():
                miss[k].append(mv)

        # boundary-aligned + recovery
        steps_sorted = sorted(per_step["fixed_global"].keys())
        for b in bsteps:
            for k in miss:
                for o in range(-PRE, POST + 1):
                    sv = b + o
                    if sv in per_step[k]:
                        aligned[k][o].append(per_step[k][sv])
            # adaptive recovery time: first offset>=0 with miss<alpha
            rec = None
            for o in range(0, POST + 1):
                sv = b + o
                if sv in per_step["adaptive_conformal"]:
                    if per_step["adaptive_conformal"][sv] < ALPHA:
                        rec = o; break
            if rec is not None:
                recov.append(rec)

        # drift magnitude: salient-set Jaccard across each boundary vs within-turn
        for b in bsteps:
            pre, post = b - 1, b
            sp = _salient_blocks(rid, step, layer, y, c, pre)
            so = _salient_blocks(rid, step, layer, y, c, post)
            if sp is not None and so is not None:
                jacc_boundary.append(_jaccard(sp, so))
        # within-turn baseline Jaccard (consecutive non-boundary steps)
        for s in steps_sorted[:-1]:
            if (s + 1) in per_step["fixed_global"] and (s + 1) not in bsteps:
                sa = _salient_blocks(rid, step, layer, y, c, s)
                sb = _salient_blocks(rid, step, layer, y, c, s + 1)
                if sa is not None and sb is not None:
                    jacc_within.append(_jaccard(sa, sb))

    out = {"policies": {}, "boundary_aligned": {}, "budget": {}}
    for k in miss:
        out["policies"][k] = _tail(np.asarray(miss[k]))
        out["budget"][k] = round(float(np.mean(budg[k])), 4) if budg[k] else float("nan")
        out["boundary_aligned"][k] = {str(o): (round(float(np.mean(v)), 4) if v else None)
                                       for o, v in aligned[k].items()}
    out["adaptive_recovery_steps_mean"] = round(float(np.mean(recov)), 2) if recov else None
    out["drift_jaccard_boundary_mean"] = round(float(np.mean(jacc_boundary)), 4) if jacc_boundary else None
    out["drift_jaccard_within_turn_mean"] = round(float(np.mean(jacc_within)), 4) if jacc_within else None
    return out


def _salient_blocks(rid, step, layer, y, c, s):
    """Set of (layer, block-rank) salient identifiers at one (conv, step) — uses the
    position within the step's row order as a stable block id (rows are emitted
    layer-major, block-minor, so position encodes (layer, block))."""
    m = (rid == c) & (step == s)
    if not m.any():
        return None
    ys = y[m]; ls = layer[m]
    if (ys > 0.5).sum() == 0:
        return None
    # identify salient by (layer, within-layer order) — reconstruct block index by
    # counting position within each layer
    out = set()
    for l in np.unique(ls):
        lm = ls == l
        yl = ys[lm]
        for bi in np.where(yl > 0.5)[0]:
            out.add((int(l), int(bi)))
    return out


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# --------------------------------------------------------------------------- #
# mode: multiturn (headline)
# --------------------------------------------------------------------------- #
def run_multiturn(a):
    d = load_pooled(a.traces, getattr(a, "only", None), suffix=getattr(a, "suffix", "multiturn"))
    fit_i, cal_i, te_i = conv_split(d["rid"], seed=0)
    print(f"pooled rows={len(d['rid']):,} | fit={len(fit_i):,} cal={len(cal_i):,} "
          f"test={len(te_i):,} | h4 pos={d['y'].mean():.3f}", flush=True)

    scorer = ClosedFormXQP.from_fit(_gather4(d["F"][fit_i], COLS), d["y"][fit_i].astype(np.float32))
    # calibrate thresholds on the disjoint calibration conversations
    p_cal = np.asarray(scorer.score(_gather4(d["F"][cal_i], COLS)))
    tau_g = _conformal_tau(p_cal[d["y"][cal_i] > 0.5], ALPHA)
    budgeter = CoverageDrivenBudgeter.calibrate(scorer, d["F"][cal_i], d["y"][cal_i],
                                                d["layer"][cal_i], cols=COLS, alpha=ALPHA)
    res = evaluate_policies(d, te_i, scorer, tau_g, budgeter)
    res["meta"] = dict(mode="multiturn", alpha=ALPHA, gamma=GAMMA, tau_g=round(float(tau_g), 4),
                       suffix=getattr(a, "suffix", "multiturn"),
                       n_test_convs=int(len(np.unique(d["rid"][te_i]))))
    _report(res)
    _write(res, a.out)


# --------------------------------------------------------------------------- #
# mode: mechanism (CPU sanity check on a real single-turn trace)
# --------------------------------------------------------------------------- #
def run_mechanism(a):
    """Inject a controlled tau*-drift into a real stream: in the 2nd half, depress
    the within-EMA of salient blocks by `depress` (simulating fresh post-switch
    cold blocks). At depress=1.0 there is no drift (expect adaptive ~= fixed); as
    depress falls, the optimal tau moves and fixed should blow out while adaptive
    tracks."""
    from experiments.run_icdm_full import load_model_trace
    f = sorted(glob.glob(os.path.join(a.traces, "*.jsonl")))[0]
    m = load_model_trace(f)
    # take a subsample of requests, build per-request 2-phase streams
    rng = np.random.default_rng(0)
    reqs = np.unique(m["rid"]); rng.shuffle(reqs); reqs = reqs[:40]
    y = m["y"][H].astype(np.float32)
    # fit scorer + tau on a disjoint half of requests (no injection there)
    fit_reqs, te_reqs = reqs[:20], reqs[20:]
    fit = np.where(np.isin(m["rid"], list(fit_reqs)))[0]
    scorer = ClosedFormXQP.from_fit(_gather4(m["F"][fit], COLS), y[fit])
    p_fit = np.asarray(scorer.score(_gather4(m["F"][fit], COLS)))
    tau_g = _conformal_tau(p_fit[y[fit] > 0.5], ALPHA)

    out = {"meta": dict(mode="mechanism", trace=os.path.basename(f), alpha=ALPHA,
                        gamma=GAMMA, tau_g=round(float(tau_g), 4)), "by_depress": []}
    for depress in [float(x) for x in a.depress.split(",")]:
        miss_fixed, miss_adapt = [], []
        for c in te_reqs:
            cm = np.where(m["rid"] == c)[0]
            csteps = np.unique(m["step"][cm])
            mid = csteps[len(csteps) // 2]
            aci = float(tau_g)
            for s in csteps:
                sm = cm[m["step"][cm] == s]
                if (y[sm] > 0.5).sum() < 4:
                    continue
                F = m["F"][sm].copy()
                if s >= mid:                         # phase 2: depress salient within-EMA
                    sal = y[sm] > 0.5
                    F[sal, 0] = F[sal, 0] * depress
                ps = np.asarray(scorer.score(_gather4(F, COLS)))
                sal = y[sm] > 0.5; nsal = int(sal.sum())
                mf = float((sal & (ps < tau_g)).sum() / nsal)
                ma = float((sal & (ps < aci)).sum() / nsal)
                miss_fixed.append(mf); miss_adapt.append(ma)
                aci = float(np.clip(aci + GAMMA * (ALPHA - ma), 0.0, 1.0))
        rec = dict(depress=depress, fixed=_tail(np.asarray(miss_fixed)),
                   adaptive=_tail(np.asarray(miss_adapt)))
        out["by_depress"].append(rec)
        print(f"  depress={depress:.2f}: fixed p99={rec['fixed']['p99']:.3f} "
              f"max={rec['fixed']['max']:.3f} blow={rec['fixed']['frac_blowout']:.3f} | "
              f"adaptive p99={rec['adaptive']['p99']:.3f} max={rec['adaptive']['max']:.3f} "
              f"blow={rec['adaptive']['frac_blowout']:.3f}", flush=True)
    _write(out, a.out)


def run_pareto(a):
    """Decisive complementarity test: a budget-matched coverage frontier. Sweep the
    recency floor (vary rho) and pure-score fixed thresholds (vary tau) over the SAME
    test stream; at each budget compare blowout. If the recency-floor frontier sits
    below the pure-score frontier across budgets, recency is genuinely complementary
    at boundaries; if the frontiers overlap, the boundary fix was only budget."""
    d = load_pooled(a.traces, getattr(a, "only", None), suffix=getattr(a, "suffix", "multiturn"))
    fit_i, cal_i, te_i = conv_split(d["rid"], seed=0)
    scorer = ClosedFormXQP.from_fit(_gather4(d["F"][fit_i], COLS), d["y"][fit_i].astype(np.float32))
    p = np.asarray(scorer.score(_gather4(d["F"][te_i], COLS)))
    fpos = d["F"][te_i][:, 3]
    sal = d["y"][te_i] > 0.5
    # per-(conv,step) group ids
    rid_te, step_te = d["rid"][te_i], d["step"][te_i]
    _, rinv = np.unique(rid_te, return_inverse=True)
    gid = rinv.astype(np.int64) * (int(step_te.max()) + 1) + step_te.astype(np.int64)
    _, ginv = np.unique(gid, return_inverse=True)
    nsal_g = np.bincount(ginv, weights=sal.astype(np.float64))
    valid = nsal_g >= 4

    def stats(keep):
        miss_g = np.bincount(ginv, weights=(sal & ~keep).astype(np.float64)) / np.maximum(nsal_g, 1)
        m = miss_g[valid]
        return dict(budget=round(float(keep.mean()), 4), mean=round(float(m.mean()), 4),
                    p99=round(float(np.percentile(m, 99)), 4),
                    blowout=round(float((m > 2 * ALPHA).mean()), 4))

    out = {"meta": dict(mode="pareto", alpha=ALPHA, suffix=getattr(a, "suffix", "multiturn"),
                        n_test_convs=int(len(np.unique(d["rid"][te_i])))),
           "recency_floor": [], "fixed_score": []}
    print("\n  recency-floor frontier (vary rho):", flush=True)
    tau_g = _conformal_tau(p[sal], ALPHA)
    for rho in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        s = stats((p >= tau_g) | (fpos >= rho)); s["rho"] = rho
        out["recency_floor"].append(s)
        print(f"   rho={rho:.2f}: budget={s['budget']:.3f} mean={s['mean']:.3f} "
              f"p99={s['p99']:.3f} blowout={s['blowout']:.3f}", flush=True)
    print("\n  pure-score frontier (vary budget):", flush=True)
    for b in [0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72]:
        tau = float(np.quantile(p, 1.0 - b))
        s = stats(p >= tau); s["tau"] = round(tau, 4)
        out["fixed_score"].append(s)
        print(f"   budget~{b:.2f}: budget={s['budget']:.3f} mean={s['mean']:.3f} "
              f"p99={s['p99']:.3f} blowout={s['blowout']:.3f}", flush=True)
    _write(out, a.out)


def run_boundary(a):
    """Boundary-triggered budget boost: the server KNOWS when a turn arrives, so use a
    low base budget mid-turn and a higher budget only within a window around each
    boundary. Tests whether concentrating budget where the drift hits achieves target
    coverage at lower AVERAGE budget than a uniform (pure-score) threshold."""
    d = load_pooled(a.traces, getattr(a, "only", None), suffix=getattr(a, "suffix", "multiturn"))
    fit_i, cal_i, te_i = conv_split(d["rid"], seed=0)
    scorer = ClosedFormXQP.from_fit(_gather4(d["F"][fit_i], COLS), d["y"][fit_i].astype(np.float32))
    p = np.asarray(scorer.score(_gather4(d["F"][te_i], COLS)))
    sal = d["y"][te_i] > 0.5
    rid_te, step_te, turn_te = d["rid"][te_i], d["step"][te_i], d["turn"][te_i]
    _, ginv = np.unique(rid_te.astype(np.int64) * 100000 + step_te.astype(np.int64), return_inverse=True)
    nsal_g = np.bincount(ginv, weights=sal.astype(np.float64))
    valid = nsal_g >= 4

    # signed offset from each row's step to the nearest topic-switch boundary in its conv
    delta = np.full(len(p), 10**6, np.int64)
    for c in np.unique(rid_te):
        cm = np.where(rid_te == c)[0]
        ct, cs = turn_te[cm], step_te[cm]
        bs = np.array(sorted({int(cs[ct == tt].min()) for tt in np.unique(ct) if tt > 0}))
        if bs.size == 0:
            continue
        dd = cs[:, None].astype(np.int64) - bs[None, :]
        delta[cm] = dd[np.arange(len(cm)), np.argmin(np.abs(dd), axis=1)]

    def stats(keep):
        miss_g = np.bincount(ginv, weights=(sal & ~keep).astype(np.float64)) / np.maximum(nsal_g, 1)
        m = miss_g[valid]
        return dict(budget=round(float(keep.mean()), 4), mean=round(float(m.mean()), 4),
                    p99=round(float(np.percentile(m, 99)), 4),
                    blowout=round(float((m > 2 * ALPHA).mean()), 4))

    out = {"meta": dict(mode="boundary", alpha=ALPHA), "pure_score": [], "boundary_boost": []}
    print("\n  pure-score uniform frontier (reference):", flush=True)
    for b in [0.20, 0.25, 0.30, 0.36, 0.42, 0.48, 0.54]:
        tau = float(np.quantile(p, 1.0 - b))
        s = stats(p >= tau); out["pure_score"].append(s)
        print(f"   budget={s['budget']:.3f} blowout={s['blowout']:.3f} p99={s['p99']:.3f}", flush=True)

    print("\n  boundary-boost frontier (window / base_b / boost_b -> avg_budget, blowout):", flush=True)
    for (wpre, wpost) in [(4, 8), (2, 6), (4, 12)]:
        near = (delta >= -wpre) & (delta <= wpost)
        for base_b in [0.12, 0.16, 0.20, 0.25]:
            tau_base = float(np.quantile(p[~near], 1.0 - base_b))
            for boost_b in [0.55, 0.70]:
                tau_boost = float(np.quantile(p[near], 1.0 - boost_b))
                keep = p >= np.where(near, tau_boost, tau_base)
                s = stats(keep)
                s.update(window=[wpre, wpost], base_b=base_b, boost_b=boost_b,
                         near_frac=round(float(near.mean()), 3))
                out["boundary_boost"].append(s)
                print(f"   W[{wpre},{wpost}] base={base_b:.2f} boost={boost_b:.2f}: "
                      f"avg_budget={s['budget']:.3f} blowout={s['blowout']:.3f} p99={s['p99']:.3f}", flush=True)

    # verdict: for each boost point, the pure-score blowout at the SAME avg budget (interp)
    ps = sorted(out["pure_score"], key=lambda r: r["budget"])
    bx = [r["budget"] for r in ps]; by = [r["blowout"] for r in ps]
    wins = []
    for s in out["boundary_boost"]:
        ref = float(np.interp(s["budget"], bx, by))
        s["pure_score_blowout_at_same_budget"] = round(ref, 4)
        if s["blowout"] < ref - 1e-9:
            wins.append((s["budget"], s["blowout"], ref))
    out["n_boundary_boost_points_beating_pure_score"] = len(wins)
    print(f"\n  boundary-boost points beating pure-score at equal avg budget: "
          f"{len(wins)}/{len(out['boundary_boost'])}", flush=True)
    for b, bl, ref in sorted(wins)[:6]:
        print(f"    budget={b:.3f}: boost blowout={bl:.3f} < pure-score {ref:.3f}", flush=True)
    _write(out, a.out)


def _report(res):
    print("\n  policy              mean   p90    p99    max   frac>2a  budget", flush=True)
    for k, v in res["policies"].items():
        b = res["budget"][k]
        print(f"  {k:18s} {v['mean']:.3f}  {v['p90']:.3f}  {v['p99']:.3f}  {v['max']:.3f}  "
              f"{v['frac_blowout']:.3f}   {b:.3f}", flush=True)
    print(f"\n  adaptive recovery steps (mean): {res['adaptive_recovery_steps_mean']}", flush=True)
    print(f"  drift Jaccard  boundary={res['drift_jaccard_boundary_mean']} "
          f"within-turn={res['drift_jaccard_within_turn_mean']}", flush=True)
    print("\n  boundary-aligned p-step miss (offset: fixed / cov-layer / adaptive / recency-floor):", flush=True)
    ba = res["boundary_aligned"]
    for o in range(-PRE, POST + 1):
        so = str(o)
        f_, c_, a_, r_ = (ba["fixed_global"][so], ba["coverage_per_layer"][so],
                          ba["adaptive_conformal"][so], ba.get("recency_floor", {}).get(so))
        bar = "<<" if o == 0 else "  "
        fmt = lambda v: ("%.3f" % v) if v is not None else " -- "
        print(f"   {bar}{o:+3d}: {fmt(f_)} / {fmt(c_)} / {fmt(a_)} / {fmt(r_)}", flush=True)


def _write(res, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("WROTE", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multiturn", "mechanism", "pareto", "boundary"], default="multiturn")
    ap.add_argument("--suffix", default="multiturn", help="trace filename suffix (multiturn | mt_sharegpt)")
    ap.add_argument("--traces", default="/public/xqp_traces_mt")
    ap.add_argument("--out", default="experiments/results/drift_multiturn.json")
    ap.add_argument("--depress", default="1.0,0.7,0.5,0.3",
                    help="mechanism mode: phase-2 within-EMA depression factors")
    ap.add_argument("--only", default=None,
                    help="substring filter on trace filenames (e.g. 'Llama') for fast iteration")
    a = ap.parse_args()
    if a.mode == "multiturn":
        run_multiturn(a)
    elif a.mode == "pareto":
        run_pareto(a)
    elif a.mode == "boundary":
        run_boundary(a)
    else:
        run_mechanism(a)


if __name__ == "__main__":
    main()
