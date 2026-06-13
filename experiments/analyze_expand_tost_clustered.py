"""Cluster-honest TOST on the expansion sweep (addresses Reviewer W1: the i.i.d.
n=896 TOST overstates n and power).

The design is 2 architectures x 7 LongBench-QA datasets x 64 prompts. F1 is
strongly dataset-driven, so the 64 prompts inside a (arch,dataset) cell are
correlated. Treating all 896 as i.i.d. understates the SE and inflates both the
TOST p-value and the post-hoc power. Here we cluster at the (arch,dataset) level
(14 clusters) and report:
  - a t-based TOST on the 14 cluster-mean diffs (df = 13), and
  - a cluster bootstrap 90% CI (resample whole cells with replacement),
alongside the naive i.i.d. SE for comparison.

Run: python experiments/analyze_expand_tost_clustered.py
"""
import json, os, numpy as np
from scipy import stats

ROOT = "experiments/results/expand"
ARCHS = ["llama31_8b", "qwen25_7b"]
DS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]


def load_f1(arch, ds, sel):
    f = f"{ROOT}/{arch}/{ds}_{sel}.json"
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))
    return {r["id"]: float(r["f1"]) for r in d.get("results", []) if "f1" in r}


def cells(selA, selB, archs):
    out = []
    for arch in archs:
        for ds in DS:
            a, b = load_f1(arch, ds, selA), load_f1(arch, ds, selB)
            ids = sorted(set(a) & set(b))
            if ids:
                out.append((f"{arch}/{ds}", np.array([a[i] - b[i] for i in ids])))
    return out


def clustered_tost(selA, selB, archs, margins=(0.02, 0.03), boot=10000, seed=1234):
    cl = cells(selA, selB, archs)
    k = len(cl)
    cell_means = np.array([d.mean() for _, d in cl])
    alld = np.concatenate([d for _, d in cl])
    n = len(alld)
    md = cell_means.mean()                          # cluster grand mean (equal cell sizes)
    se_cluster = cell_means.std(ddof=1) / np.sqrt(k)
    se_iid = alld.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    bs = np.empty(boot)
    for j in range(boot):
        idx = rng.integers(0, k, k)
        bs[j] = np.concatenate([cl[i][1] for i in idx]).mean()
    ci90 = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
    res = {"label": f"{selA}_vs_{selB}", "archs": archs, "k_clusters": k, "n_items": n,
           "diff": float(md), "se_cluster": float(se_cluster), "se_iid": float(se_iid),
           "se_inflation": float(se_cluster / se_iid), "ci90_clusterboot": list(ci90), "tost": {}}
    print(f"\n{selA} vs {selB}  ({'+'.join(archs)})  k={k} clusters, n={n} items")
    print(f"  diff={md:+.4f}  SE_cluster={se_cluster:.4f}  SE_iid={se_iid:.4f}  "
          f"(SE inflation {se_cluster/se_iid:.2f}x)")
    print(f"  cluster-boot 90% CI = [{ci90[0]:+.4f}, {ci90[1]:+.4f}]")
    for dlt in margins:
        # t-based TOST on cluster means, df = k-1
        p_low = 1 - stats.t.cdf((md + dlt) / se_cluster, k - 1)
        p_up = stats.t.cdf((md - dlt) / se_cluster, k - 1)
        tp = max(p_low, p_up)
        equiv_ci = (ci90[0] > -dlt) and (ci90[1] < dlt)
        res["tost"][f"{dlt}"] = {"tost_p_clustered": float(tp),
                                 "equiv_by_clusterboot_ci": bool(equiv_ci)}
        print(f"    margin +/-{dlt:.2f}: clustered TOST_p={tp:.4f}  "
              f"equiv(CI)={equiv_ci}")
    return res


if __name__ == "__main__":
    out = {}
    out["xqp_vs_h2o::pooled(2arch)"] = clustered_tost("xqp", "h2o", ARCHS)
    out["xqp_vs_h2o::llama"] = clustered_tost("xqp", "h2o", ["llama31_8b"])
    out["xqp_vs_h2o::qwen"] = clustered_tost("xqp", "h2o", ["qwen25_7b"])
    out["pyramidkv_vs_h2o::pooled"] = clustered_tost("pyramidkv", "h2o", ARCHS)
    out["adakv_vs_h2o::pooled"] = clustered_tost("adakv", "h2o", ARCHS)
    os.makedirs("experiments/results/tost", exist_ok=True)
    json.dump(out, open("experiments/results/tost/expand_tost_clustered.json", "w"), indent=2)
    print("\nwrote experiments/results/tost/expand_tost_clustered.json")
