"""Calibrate + serialize a CoverageDrivenBudgeter ckpt for the SEER GuardKV
policy (the budgeter half; the scorer half is experiments/predictors/
xqp_closed_2view_h4.json). Produces a JSON the runner loads via --budgeter-ckpt.

    python experiments/fit_guardkv_budgeter.py --alpha 0.10
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from run_icdm_full import load_model_trace, pool_models, request_split, subsample
from xqp.predictor import ClosedFormXQP
from xqp.budgeter import CoverageDrivenBudgeter

COLS = (0, 1)  # within + cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")  # deploy workload = mooncake
    ap.add_argument("--scorer", default="experiments/predictors/xqp_closed_2view_h4.json")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--cap", type=int, default=600_000)
    ap.add_argument("--out", default="experiments/predictors/guardkv_budgeter_a10.json")
    a = ap.parse_args()

    sc = json.load(open(a.scorer))
    scorer = ClosedFormXQP(weights=np.asarray(sc["weights"], np.float32),
                           bias=np.float32(sc["bias"]))
    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))) if ".smoke." not in f]
    d = pool_models({k: v for k, v in
                     {os.path.basename(f)[:-6]: load_model_trace(f) for f in files}.items() if v})
    # calibrate on a held-out-request FIT split (no leakage into any later eval)
    tr, _ = request_split(d["rid"])
    tr = subsample(tr, a.cap)
    y = d["y"]["h4"].astype(np.float32)
    bg = CoverageDrivenBudgeter.calibrate(scorer, d["F"][tr], y[tr], d["layer"][tr],
                                          cols=COLS, alpha=a.alpha)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bg.save(a.out)
    obj = json.load(open(a.out))
    print(f"alpha={obj['alpha']} tau_global={obj['tau_global']:.4f} "
          f"n_layers={len(obj['tau_by_layer'])} norm_curve_pts={len(obj['norm_curve'])}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
