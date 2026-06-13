"""Shared matched-budget signal-isolation protocol (used by all audit areas).

Per-instance data: outcome_NG[i,g] = is instance i correct when given budget level
budgets_G[g] (monotone grid, e.g. #samples, #reasoning-tokens, or {skip,act}).
A method assigns each instance a budget alloc_N[i] from its signal. We compare:
  - METHOD:   mean outcome at alloc_N                          (acc, avg-budget B)
  - RANDOM:   the SAME budget multiset alloc_N, PERMUTED across instances (matched B)
  - ORACLE:   greedy marginal-gain allocation at total budget N*B (upper bound)
  - UNIFORM:  every instance at the grid level nearest B
Headline metric: fraction of the (oracle - random) accuracy gap captured by the signal,
at matched budget, with bootstrap/permutation CIs. Optionally CHARGE a per-instance
signal-estimation cost (added to alloc before evaluation) to expose "free oracle" signals.
"""
import numpy as np


def step_outcome(outcome_NG, budgets_G, alloc_N):
    """outcome at the largest grid budget <= alloc_i (step function)."""
    idx = np.searchsorted(budgets_G, alloc_N, side="right") - 1
    idx = np.clip(idx, 0, len(budgets_G) - 1)
    return outcome_NG[np.arange(len(alloc_N)), idx]


def permuted_random(outcome_NG, budgets_G, alloc_N, nperm=300, seed=0):
    rng = np.random.RandomState(seed)
    accs = [step_outcome(outcome_NG, budgets_G, rng.permutation(alloc_N)).mean() for _ in range(nperm)]
    return float(np.mean(accs)), np.percentile(accs, [2.5, 97.5])


def oracle_at_budget(outcome_NG, budgets_G, target_avg):
    """Best possible accuracy at matched avg budget, robust to NON-MONOTONE outcome curves
    (e.g. SC majority can flip with more samples). Each instance has a min-cost-to-be-correct
    m_i = smallest grid budget at which it is correct (inf if never). With total budget
    N*target_avg and a floor budgets_G[0] each, buy correctness for the cheapest-to-win
    instances first (knapsack maximizing #correct)."""
    N, G = outcome_NG.shape
    base = budgets_G[0]
    mcost = np.full(N, np.inf)
    for i in range(N):
        w = np.where(outcome_NG[i] == 1)[0]
        if len(w):
            mcost[i] = budgets_G[w[0]]                   # budgets sorted asc -> first win = cheapest
    free_correct = int((mcost <= base).sum())            # correct already at the floor budget
    extra_budget = target_avg * N - base * N
    extra_costs = sorted(float(m - base) for m in mcost if base < m < np.inf)
    bought = 0; spent = 0.0
    for cst in extra_costs:
        if spent + cst <= extra_budget:
            spent += cst; bought += 1
        else:
            break
    acc = (free_correct + bought) / N
    avgb = base + spent / N
    return float(acc), float(avgb)


def uniform_at_budget(outcome_NG, budgets_G, target_avg):
    g = int(np.clip(np.searchsorted(budgets_G, target_avg, side="right") - 1, 0, len(budgets_G) - 1))
    return float(outcome_NG[:, g].mean()), float(budgets_G[g])


def audit_point(outcome_NG, budgets_G, alloc_N, est_cost_N=None, seed=0):
    """One method operating point -> the full comparison at its matched budget."""
    alloc = alloc_N.astype(float).copy()
    if est_cost_N is not None:
        alloc = alloc + est_cost_N                      # charge the signal's own cost
    acc_m = float(step_outcome(outcome_NG, budgets_G, alloc).mean())
    B = float(alloc.mean())
    acc_r, ci_r = permuted_random(outcome_NG, budgets_G, alloc, seed=seed)
    acc_o, _ = oracle_at_budget(outcome_NG, budgets_G, B)
    acc_u, _ = uniform_at_budget(outcome_NG, budgets_G, B)
    gap = acc_o - acc_r
    captured = (acc_m - acc_r) / gap if gap > 1e-9 else float("nan")
    # paired bootstrap CI on (method - random) accuracy at matched budget
    rng = np.random.RandomState(seed + 1); N = len(alloc); d = []
    for _ in range(1000):
        idx = rng.choice(N, N, replace=True)
        am = step_outcome(outcome_NG, budgets_G, alloc[idx]).mean() if False else None
        # method on bootstrap sample
        am = outcome_NG[idx][np.arange(N), np.clip(np.searchsorted(budgets_G, alloc[idx], side="right") - 1, 0, len(budgets_G) - 1)].mean()
        ar = outcome_NG[idx][np.arange(N), np.clip(np.searchsorted(budgets_G, rng.permutation(alloc[idx]), side="right") - 1, 0, len(budgets_G) - 1)].mean()
        d.append(am - ar)
    ci_d = np.percentile(d, [2.5, 97.5])
    p_onesided = float(np.mean(np.array(d) <= 0))         # H0: method <= random
    return {"B": B, "acc_method": acc_m, "acc_random": acc_r, "rand_ci": ci_r,
            "acc_oracle": acc_o, "acc_uniform": acc_u, "frac_captured": captured,
            "oracle_B": None, "method_minus_random": acc_m - acc_r,
            "mr_ci": [float(ci_d[0]), float(ci_d[1])], "p": p_onesided}


def binary_outcome_table(out0, out1, cost1=1.0):
    """Convenience for retrieve/route (2 levels): budgets [0, cost1]."""
    N = len(out0)
    outcome = np.zeros((N, 2)); outcome[:, 0] = out0; outcome[:, 1] = out1
    return outcome, np.array([0.0, cost1])
