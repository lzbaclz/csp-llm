"""Parse the native-vs-H2O live-loop deployment: served-oracle miss (eps) per cell."""
import json, os
import numpy as np

OUT = "experiments/results/native_deploy"


def eps(tag):
    f = f"{OUT}/{tag}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); e = []
    for r in d["results"]:
        e += r.get("per_step_eps_measured", [])
    return float(np.mean(e)) if e else float("nan")


print("LLAMA (native trained here) — served-oracle miss (lower=better):")
print(f"{'budget':8s} {'H2O':8s} {'native':8s} {'Δ(native-H2O)':14s}")
for b in ["0.10", "0.20", "0.30"]:
    h, n = eps(f"llama_h2o_b{b}"), eps(f"llama_native_b{b}")
    if h is not None and n is not None:
        print(f"{b:8s} {h:<8.3f} {n:<8.3f} {n-h:+.3f}  {'WIN' if n < h else 'lose'}")

print("\nTRANSFER (native trained on Llama) @ b0.20:")
for m in ["mistral", "qwen"]:
    h, n = eps(f"{m}_h2o_b0.20"), eps(f"{m}_native_b0.20")
    if h is not None and n is not None:
        print(f"{m:8s} H2O={h:.3f} native={n:.3f} Δ={n-h:+.3f}  {'WIN' if n < h else 'lose'}")
