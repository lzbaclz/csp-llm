"""Aggregate the Idea 2 IO-bound sweep: prefetch on vs off, per (context, tier).

For each pair it reports mean base_us, io_us(off), io_us(on), TPOT P50/P99 on/off,
the prefetch saving = TPOT(off)-TPOT(on) (clean, since base is identical), and the
IO fraction of TPOT — the ceiling on what prefetch can buy.

    python experiments/agg_idea2.py --dir experiments/results/idea2
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def load(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not d.get("results"):
        return None
    r = d["results"][0]
    io = np.asarray(r.get("per_step_io_us", []), float)
    base = np.asarray(r.get("per_step_base_us", []), float)
    if io.size == 0:
        return None
    return dict(p50=d.get("tpot_p50_us", float("nan")), p99=d.get("tpot_p99_us", float("nan")),
                io=float(io.mean()), base=float(base.mean()))


def pair(d, tag_on, tag_off):
    on = load(os.path.join(d, tag_on + ".json"))
    off = load(os.path.join(d, tag_off + ".json"))
    if not on or not off:
        return None
    save_us = off["p50"] - on["p50"]
    save_pct = 100.0 * save_us / off["p50"] if off["p50"] else float("nan")
    io_frac_off = 100.0 * off["io"] / (off["base"] + off["io"]) if (off["base"] + off["io"]) else float("nan")
    return dict(base=on["base"], io_off=off["io"], io_on=on["io"],
                tpot_off=off["p50"], tpot_on=on["p50"], p99_off=off["p99"], p99_on=on["p99"],
                save_us=save_us, save_pct=save_pct, io_frac_off=io_frac_off)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="experiments/results/idea2")
    a = ap.parse_args()
    out = {"context_sweep": [], "tier_sweep": []}

    print("\n== A) context sweep @ NVMe tier (ell_bar=3000us), budget 0.2, prefetch on vs off ==")
    print(f"  {'ctx':>7} {'base_ms':>8} {'io_off':>7} {'io_on':>7} {'TPOToff':>8} {'TPOTon':>8} "
          f"{'save_ms':>8} {'save%':>6} {'io%TPOT':>7}")
    for ctx in [4096, 16384, 32768, 65536]:
        p = pair(a.dir, f"ctx{ctx}_nvme_on", f"ctx{ctx}_nvme_off")
        if not p:
            print(f"  {ctx:>7}  (missing)"); continue
        p["context"] = ctx; out["context_sweep"].append(p)
        print(f"  {ctx:>7} {p['base']/1e3:>8.1f} {p['io_off']/1e3:>7.1f} {p['io_on']/1e3:>7.1f} "
              f"{p['tpot_off']/1e3:>8.1f} {p['tpot_on']/1e3:>8.1f} {p['save_us']/1e3:>8.1f} "
              f"{p['save_pct']:>5.1f}% {p['io_frac_off']:>6.1f}%")

    print("\n== B) tier sweep @ 32K, budget 0.2, prefetch on vs off ==")
    print(f"  {'ell_us':>7} {'io_off':>7} {'io_on':>7} {'TPOToff':>8} {'TPOTon':>8} "
          f"{'save_ms':>8} {'save%':>6} {'io%TPOT':>7}")
    for ell in [200, 1000, 3000, 10000]:
        p = pair(a.dir, f"tier{ell}_on", f"tier{ell}_off")
        if not p:
            print(f"  {ell:>7}  (missing)"); continue
        p["ell_bar_us"] = ell; out["tier_sweep"].append(p)
        print(f"  {ell:>7} {p['io_off']/1e3:>7.1f} {p['io_on']/1e3:>7.1f} {p['tpot_off']/1e3:>8.1f} "
              f"{p['tpot_on']/1e3:>8.1f} {p['save_us']/1e3:>8.1f} {p['save_pct']:>5.1f}% {p['io_frac_off']:>6.1f}%")

    json.dump(out, open(os.path.join(a.dir, "idea2_summary.json"), "w"), indent=2)
    print(f"\nWROTE {os.path.join(a.dir, 'idea2_summary.json')}")


if __name__ == "__main__":
    main()
