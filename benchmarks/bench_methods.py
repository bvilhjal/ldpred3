"""Benchmark LDpred3 variants on realistic (coalescent) LD across architectures.

Metric: genetic R2 = squared correlation between the PRS and the true genetic
value under population LD, (b^T R beta)^2 / [(b^T R b)(beta^T R beta)], summed
over the block-diagonal LD. Averaged over replicates. h2=0.5, N=50000.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ldpred3_by_blocks, ldpred3_auto_annot_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)          # (100, 500, 500)
NB, K = 100, 500
M = NB * K
H2 = 0.5
REPS = 5
N_VALUES = [10000, 50000]

# Build the genome: distinct realistic LD blocks (cycle the library), + cholesky.
blocks, chols, idxs = [], [], []
for b in range(NB):
    R = libR[b % libR.shape[0]].copy()
    blocks.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(np.linalg.cholesky(R + 1e-4 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))


def make_beta(model, rng, func):
    beta = np.zeros(M)
    if model == "infinitesimal":
        beta = rng.normal(0, np.sqrt(H2 / M), M)
    elif model == "sparse":                       # p = 0.01
        c = rng.random(M) < 0.01; beta[c] = rng.normal(0, 1, c.sum())
    elif model == "polygenic":                    # p = 0.2
        c = rng.random(M) < 0.2; beta[c] = rng.normal(0, 1, c.sum())
    elif model == "major_locus":                  # few huge + sparse bg
        c = rng.random(M) < 0.02; beta[c] = rng.normal(0, 1, c.sum()) * 0.3
        maj = rng.choice(M, 3, replace=False); beta[maj] = rng.choice([-1, 1], 3) * 4
    elif model == "annot_enriched":               # causals 10x in functional 20%
        base = np.where(func > 0, 10.0, 1.0)
        c = rng.random(M) < np.clip(base / base.sum() * (0.02 * M), 0, 1)
        beta[c] = rng.normal(0, 1, c.sum())
    # scale to total genetic variance H2 (beta^T R beta)
    gv = sum(beta[ix] @ (blocks[b][0].astype(float) @ beta[ix])
             for b, ix in enumerate(idxs))
    if gv > 0:
        beta *= np.sqrt(H2 / gv)
    return beta


def sumstats(beta, rng):
    bhat = np.empty(M)
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        bhat[ix] = R @ beta[ix] + (chols[b] @ rng.standard_normal(K)) / np.sqrt(N)
    return bhat


def genetic_r2(b_est, beta):
    num = den1 = den2 = 0.0
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        Rb = R @ beta[ix]
        num += b_est[ix] @ Rb
        den1 += b_est[ix] @ (R @ b_est[ix])
        den2 += beta[ix] @ Rb
    return float(num * num / (den1 * den2)) if den1 > 0 and den2 > 0 else 0.0


def fit(method, bhat, n, true_p, A):
    if method == "marginal":
        return bhat
    if method == "inf":
        return ldpred3_by_blocks(blocks, bhat, n, method="inf", h2=H2)
    if method == "grid":                          # oracle h2, p
        return ldpred3_by_blocks(blocks, bhat, n, method="grid", h2=H2,
                                 p=max(true_p, 1e-3), burn_in=80, num_iter=200)
    if method == "auto":
        return ldpred3_by_blocks(blocks, bhat, n, method="auto", burn_in=80,
                                 num_iter=200, seed=1)
    if method == "annot":
        return ldpred3_auto_annot_blocks(blocks, bhat, n, A, burn_in=80,
                                         num_iter=200, seed=1).beta_est


MODELS = ["infinitesimal", "sparse", "polygenic", "major_locus", "annot_enriched"]
TRUE_P = {"infinitesimal": 1.0, "sparse": 0.01, "polygenic": 0.2,
          "major_locus": 0.02, "annot_enriched": 0.02}
METHODS = ["marginal", "inf", "grid", "auto", "annot"]

rng0 = np.random.default_rng(0)
func = (rng0.random(M) < 0.2).astype(float)       # the functional annotation
A = func[:, None]

import csv
t0 = time.time()
allrows = []
for N in N_VALUES:
    n = np.full(M, float(N))
    results = {m: {meth: [] for meth in METHODS} for m in MODELS}
    for model in MODELS:
        for rep in range(REPS):
            rng = np.random.default_rng(1000 + rep)
            beta = make_beta(model, rng, func)
            bhat = sumstats(beta, rng)
            for meth in METHODS:
                be = fit(meth, bhat, n, TRUE_P[model], A)
                results[model][meth].append(genetic_r2(be, beta))
    print(f"\nGenetic R2 (PRS vs true genetic value), realistic coalescent LD, "
          f"m={M}, N={N}, h2={H2}, {REPS} reps\n")
    hdr = f"{'model':>14} | " + " ".join(f"{m:>9}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    for model in MODELS:
        means = [float(np.mean(results[model][meth])) for meth in METHODS]
        print(f"{model:>14} | " + " ".join(f"{x:>9.3f}" for x in means))
        allrows.append([N, model] + [round(x, 4) for x in means])

with open("methods_arch.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(["N", "model"] + METHODS)
    w.writerows(allrows)
print(f"\n({time.time()-t0:.0f}s)  wrote methods_arch.csv")
