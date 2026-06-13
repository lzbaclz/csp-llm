"""RCRG full experiments: PAC (Learn-then-Test) + cross-model drift + dataset shift.

Loads both models' BM25 data (aligned by dataset+index), and reports, for each method,
coverage = P(realized test risk <= eps) over many calibration draws (the PAC quantity),
mean realized loss, and retrieval%. Methods: TARG (naive), CRC (expected-risk), LTT
(PAC 1-delta), and their weighted (non-exchangeable) variants for shift.
"""
import json, sys, numpy as np
from rcrg import crc_threshold, targ_threshold, evaluate
from ltt import ltt_threshold, ltt_threshold_weighted

EPS, DELTA = 0.05, 0.10


def load(path):
    R = [json.loads(l) for l in open(path)]
    return (np.array([r["gate_agree"] for r in R]), np.array([r["open_correct"] for r in R]),
            np.array([r["closed_correct"] for r in R]), np.array([r["ds"] for r in R]))


def weights_for(cal_g, te_g):
    hte, edges = np.histogram(te_g, bins=10, range=(0, 1), density=True)
    hcal, _ = np.histogram(cal_g, bins=10, range=(0, 1), density=True)
    bidx = np.clip(np.digitize(cal_g, edges[1:-1]), 0, 9)
    return (hte[bidx] + 1e-2) / (hcal[bidx] + 1e-2)


def thr(method, g, o, c, eps, delta, w=None):
    if method == "TARG":   return targ_threshold(g, o, c, eps)
    if method == "CRC":    return crc_threshold(g, o, c, eps)
    if method == "wCRC":   return crc_threshold(g, o, c, eps, weights=w)
    if method == "LTT":    return ltt_threshold(g, o, c, eps, delta)
    if method == "wLTT":   return ltt_threshold_weighted(g, o, c, eps, w, delta)
    raise ValueError(method)


def coverage_run(cal_pool, te, methods, eps, delta, ncal, n_draws=400, shift=False, seed=0):
    """cal_pool/te = (g,o,c) tuples. Draw ncal calib points, evaluate on te. Coverage =
    fraction of draws with realized test loss <= eps."""
    rng = np.random.RandomState(seed)
    cg, co, cc = cal_pool; tg, to, tc = te
    out = {m: {"cov": [], "loss": [], "rf": []} for m in methods}
    for _ in range(n_draws):
        idx = rng.choice(len(cg), size=min(ncal, len(cg)), replace=False)
        gg, oo, ccc = cg[idx], co[idx], cc[idx]
        w = weights_for(gg, tg) if shift else None
        for m in methods:
            t = thr(m, gg, oo, ccc, eps, delta, w)
            ev = evaluate(t, tg, to, tc)
            out[m]["cov"].append(ev["loss_vs_always"] <= eps); out[m]["loss"].append(ev["loss_vs_always"]); out[m]["rf"].append(ev["retr_freq"])
    return {m: {"cov": 100*np.mean(v["cov"]), "loss": np.mean(v["loss"]), "rf": 100*np.mean(v["rf"])} for m, v in out.items()}


def main():
    lp = sys.argv[1] if len(sys.argv) > 1 else "results/llama_bm25.jsonl"
    qp = sys.argv[2] if len(sys.argv) > 2 else "results/qwen_bm25.jsonl"
    L = load(lp); Q = load(qp)
    for name, D in [("Llama-3.1-8B", L), ("Qwen2.5-7B", Q)]:
        g, o, c, ds = D
        print(f"\n##### {name}: N={len(g)} always={o.mean():.3f} never={c.mean():.3f} oracle={np.mean(np.maximum(o,c)):.3f} #####")
        for d in sorted(set(ds)):
            m = ds == d
            print(f"   {d:>16}: closed={c[m].mean():.2f} open={o[m].mean():.2f} d={o[m].mean()-c[m].mean():+.2f}")

    # ---------- (A) PAC coverage in-dist: TARG vs CRC vs LTT at small calib ----------
    print(f"\n=== (A) IN-DIST coverage (target eps={EPS}, delta={DELTA}); 50/50 calib/test disjoint ===")
    for name, D in [("Llama", L), ("Qwen", Q)]:
        g, o, c, ds = D; n = len(g)
        print(f"  -- {name} --   {'n_cal':>6} | " + " | ".join(f"{m+' cov/loss/rf':>22}" for m in ["TARG", "CRC", "LTT"]))
        for ncal in [25, 50, 100, 300]:
            rng = np.random.RandomState(1)
            agg = {m: {"cov": [], "loss": [], "rf": []} for m in ["TARG", "CRC", "LTT"]}
            for _ in range(400):
                idx = rng.permutation(n); cal, te = idx[:ncal], idx[ncal:]
                for m in ["TARG", "CRC", "LTT"]:
                    t = thr(m, g[cal], o[cal], c[cal], EPS, DELTA)
                    ev = evaluate(t, g[te], o[te], c[te])
                    agg[m]["cov"].append(ev["loss_vs_always"] <= EPS); agg[m]["loss"].append(ev["loss_vs_always"]); agg[m]["rf"].append(ev["retr_freq"])
            row = " | ".join(f"{100*np.mean(agg[m]['cov']):>5.0f}% {np.mean(agg[m]['loss']):>+.3f} {100*np.mean(agg[m]['rf']):>3.0f}%" for m in ["TARG", "CRC", "LTT"])
            print(f"  {'':>9}   {ncal:>6} | {row}")
        print(f"     (PAC win: LTT cov >= {100*(1-DELTA):.0f}%; CRC ~expected; TARG under-covers at small n)")

    # ---------- (B) CROSS-MODEL drift: calibrate on A, deploy on B ----------
    print(f"\n=== (B) CROSS-MODEL drift (calib model -> deploy model), n_cal=80, eps={EPS} ===")
    print(f"  {'direction':>20} {'method':>6} {'cover':>7} {'mean_loss':>10} {'retr%':>6}")
    for cal_name, cal_D, te_name, te_D in [("Llama", L, "Qwen", Q), ("Qwen", Q, "Llama", L)]:
        cal_pool = cal_D[:3]; te = te_D[:3]
        res = coverage_run(cal_pool, te, ["TARG", "CRC", "wCRC", "wLTT"], EPS, DELTA, ncal=80, shift=True)
        for m in ["TARG", "CRC", "wCRC", "wLTT"]:
            r = res[m]
            print(f"  {cal_name+'->'+te_name:>20} {m:>6} {r['cov']:>6.0f}% {r['loss']:>+10.3f} {r['rf']:>5.0f}%")

    # ---------- (C) DATASET shift within model (calib retrieve-helps-least -> deploy most) ----------
    print(f"\n=== (C) DATASET shift within model, n_cal=60, eps={EPS} ===")
    print(f"  {'model':>8} {'method':>6} {'cover':>7} {'mean_loss':>10} {'retr%':>6}")
    for name, D in [("Llama", L), ("Qwen", Q)]:
        g, o, c, ds = D
        helps = sorted(set(ds), key=lambda d: o[ds==d].mean() - c[ds==d].mean())
        easy, hard = helps[:3], helps[3:]
        cal_pool = (g[np.isin(ds, easy)], o[np.isin(ds, easy)], c[np.isin(ds, easy)])
        te = (g[np.isin(ds, hard)], o[np.isin(ds, hard)], c[np.isin(ds, hard)])
        res = coverage_run(cal_pool, te, ["TARG", "CRC", "wCRC", "wLTT"], EPS, DELTA, ncal=60, shift=True)
        for m in ["TARG", "CRC", "wCRC", "wLTT"]:
            r = res[m]
            print(f"  {name:>8} {m:>6} {r['cov']:>6.0f}% {r['loss']:>+10.3f} {r['rf']:>5.0f}%")


if __name__ == "__main__":
    main()
