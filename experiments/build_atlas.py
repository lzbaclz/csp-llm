"""Build the KVSalienceBench difficulty atlas from per-workload analysis JSONs.

Ingests experiments/results/icdm_full_<workload>.json (each produced by
run_icdm_full.py on one workload of the expanded corpus) and emits a single
per-workload table along the context-length axis. This is the core evidence
for the characterization contribution: how the headline quantities
(2-view-vs-GBDT gap, calibration edge, drift, signal usefulness) vary by
workload / context length.

    python experiments/build_atlas.py                      # all icdm_full_*.json
    python experiments/build_atlas.py experiments/results/icdm_full_mooncake.json ...
    python experiments/build_atlas.py --out experiments/results/atlas.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _row(method_table, name):
    for r in method_table:
        if r["method"] == name:
            return r
    return None


def extract(path: str) -> dict | None:
    d = json.load(open(path))
    p = d.get("pooled")
    if not p:
        return None
    wl = os.path.basename(path)[len("icdm_full_"):-len(".json")]
    s = p["summary"]
    t = p["headline"]["table"]
    tv = _row(t, "within+cross(2)")
    gb = _row(t, "GBDT(LightGBM)")
    pw = _row(t, "XQP-pairwise")
    pv = p["per_view"]["h4"]
    dr = p["drift"]
    rows_per_req = s["n_rows"] / max(s["n_requests"], 1)
    return dict(
        workload=wl,
        n_req=s["n_requests"],
        rows_per_req=rows_per_req,          # context-length proxy
        pos_rate=s["pos_rate"]["h4"],
        tv_auc=tv["auc"], gb_auc=gb["auc"],
        d_auc=tv["auc"] - gb["auc"],
        tv_ece=tv["ece"], gb_ece=gb["ece"],
        calib_x=gb["ece"] / max(tv["ece"], 1e-9),
        pw_gain=pw["auc"] - tv["auc"],       # does complexity help?
        query_auc=pv["s_query"]["auc"],
        recency_auc=pv["s_pos"]["auc"],
        within_auc=pv["s_within"]["auc"],
        cross_auc=pv["s_cross"]["auc"],
        online_gain=dr["online_gain"],       # <0 => online adaptation hurts
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", default=None, help="also write markdown here")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob("experiments/results/icdm_full_*.json"))
    rows = []
    for f in files:
        wl = os.path.basename(f)
        if "all12" in wl or "smoke" in wl:
            continue
        try:
            r = extract(f)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[skip] {f}: {e}")
    if not rows:
        print("no per-workload JSONs found")
        return 1
    rows.sort(key=lambda r: r["rows_per_req"])   # short context -> long

    hdr = (f"{'workload':<24}{'ctx(rows/req)':>13}{'pos':>6}"
           f"{'2view_AUC':>10}{'GBDT_AUC':>9}{'ΔAUC':>8}"
           f"{'calib×':>8}{'pw_gain':>8}{'query':>7}{'recency':>8}{'online_gain':>12}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['workload']:<24}{r['rows_per_req']:>13,.0f}{r['pos_rate']:>6.2f}"
            f"{r['tv_auc']:>10.4f}{r['gb_auc']:>9.4f}{r['d_auc']:>+8.4f}"
            f"{r['calib_x']:>8.1f}{r['pw_gain']:>+8.4f}{r['query_auc']:>7.3f}"
            f"{r['recency_auc']:>8.3f}{r['online_gain']:>+12.4f}")
    out = "\n".join(lines)
    print(out)
    print("\nLEGEND: ctx=rows/request (context-length proxy, sorted short->long); "
          "ΔAUC=2view-GBDT (neg=GBDT ahead); calib×=GBDT_ECE/2view_ECE (higher=2view better calibrated); "
          "pw_gain=pairwise(15)-2view AUC (does complexity help?); query/recency=single-view AUC (~0.5=useless); "
          "online_gain=online-static drift AUC (neg=online adaptation HURTS).")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("# KVSalienceBench difficulty atlas\n\n```\n" + out + "\n```\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
