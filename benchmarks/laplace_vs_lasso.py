"""Bayesian-lasso (Laplace-prior) posterior mean vs the lasso MAP, by architecture.

The lasso (``lassosum2``) is the posterior *mode* under a Laplace prior;
``method="laplace"`` samples the posterior *mean* of that same prior. This checks
the two land close (as theory says) and places both against the infinitesimal
(Gaussian) and spike-and-slab (``grid``/``auto``) models across genetic
architectures. Metric: genetic R² = (β̂ᵀRβ)² / [(β̂ᵀRβ̂)(βᵀRβ)] under population
LD, averaged over replicates. Self-contained (needs msprime); writes
``laplace_vs_lasso.csv``.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3 import (ldpred3_by_blocks, lassosum2,          # noqa: E402
                     ldpred3_laplace)
from ldpred3.simulate import simulate_genotypes_coalescent  # noqa: E402

K = 500                       # SNPs per LD block
NB = 20                       # blocks -> M = 10,000 SNPs
M = NB * K
N_REF = 4000
H2 = 0.5
N_GWAS = 20000
REPS = 5
ARCHS = {"infinitesimal": 1.0, "polygenic (p=0.1)": 0.1, "sparse (p=0.01)": 0.01}

print(f"simulating {N_REF} x {M} (coalescent) ...", flush=True)
t0 = time.time()
G, _ = simulate_genotypes_coalescent(N_REF, M, K, seed=11)
Gs = (G - G.mean(0)) / G.std(0)
blocks, chols, idxs = [], [], []
for b in range(NB):
    ix = np.arange(b * K, (b + 1) * K)
    R = (Gs[:, ix].T @ Gs[:, ix]) / N_REF
    np.fill_diagonal(R, 1.0)
    blocks.append((R.astype(np.float32), ix))
    chols.append(np.linalg.cholesky(R + 1e-4 * np.eye(K)))
    idxs.append(ix)
print(f"  ({time.time() - t0:.0f}s)", flush=True)


def make_beta(p, rng):
    beta = np.zeros(M)
    causal = rng.random(M) < p
    beta[causal] = rng.normal(0, 1, causal.sum())
    gv = sum(beta[ix] @ (blocks[b][0].astype(float) @ beta[ix])
             for b, ix in enumerate(idxs))
    if gv > 0:
        beta *= np.sqrt(H2 / gv)
    return beta


def sumstats(beta, rng):
    bhat = np.empty(M)
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        bhat[ix] = R @ beta[ix] + (chols[b] @ rng.standard_normal(K)) / np.sqrt(N_GWAS)
    return bhat


def genetic_r2(be, beta):
    num = d1 = d2 = 0.0
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        Rb = R @ beta[ix]
        num += be[ix] @ Rb
        d1 += be[ix] @ (R @ be[ix])
        d2 += beta[ix] @ Rb
    return float(num * num / (d1 * d2)) if d1 > 0 and d2 > 0 else 0.0


def fit(method, bhat, n, true_p):
    if method == "inf":
        return ldpred3_by_blocks(blocks, bhat, n, method="inf", h2=H2)
    if method == "grid":                                  # oracle (h2, p)
        return ldpred3_by_blocks(blocks, bhat, n, method="grid", h2=H2,
                                 p=max(true_p, 1e-3), burn_in=80, num_iter=200)
    if method == "auto":
        return ldpred3_by_blocks(blocks, bhat, n, method="auto", burn_in=80,
                                 num_iter=200, seed=1)
    if method == "laplace":
        return ldpred3_by_blocks(blocks, bhat, n, method="laplace", h2=H2,
                                 burn_in=80, num_iter=200, seed=1)
    if method == "lassosum2":
        return lassosum2(blocks, bhat).beta_est


METHODS = ["inf", "grid", "auto", "lassosum2", "laplace"]
n = np.full(M, float(N_GWAS))
# warm the JITs (not timed / scored)
ldpred3_laplace(blocks[0][0], np.zeros(K), n[:K], burn_in=2, num_iter=2)

rows = []
print(f"\nGenetic R2, realistic LD, m={M}, N={N_GWAS}, h2={H2}, {REPS} reps\n", flush=True)
hdr = f"{'architecture':>18} | " + " ".join(f"{x:>10}" for x in METHODS)
print(hdr)
print("-" * len(hdr))
for name, p in ARCHS.items():
    acc = {mth: [] for mth in METHODS}
    for rep in range(REPS):
        rng = np.random.default_rng(300 + rep)
        beta = make_beta(p, rng)
        bhat = sumstats(beta, rng)
        for mth in METHODS:
            acc[mth].append(genetic_r2(fit(mth, bhat, n, p), beta))
    means = [float(np.mean(acc[mth])) for mth in METHODS]
    print(f"{name:>18} | " + " ".join(f"{x:>10.3f}" for x in means))
    rows.append([name, p] + [round(x, 4) for x in means])

with open("laplace_vs_lasso.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["architecture", "p_causal"] + METHODS)
    w.writerows(rows)
print(f"\n({time.time() - t0:.0f}s)  wrote laplace_vs_lasso.csv", flush=True)
