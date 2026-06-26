"""Compare SNP-heritability estimates: LD Score regression vs LDpred2-auto.

Both estimate h2 from the *same* GWAS summary statistics (no individual data),
on realistic coalescent LD. We report each against the known true h2, for two
heritabilities and two architectures. Needs ``ld_library.npz`` (coalescent LD
blocks) in the working directory, as the other benchmarks do.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ld_scores, ldsc_h2, ldpred2_auto_infer

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
N = 50000
REPS = 5

blocks, chols, idxs = [], [], []
for b in range(NB):
    R = libR[b % libR.shape[0]].copy()
    blocks.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(np.linalg.cholesky(R + 1e-4 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))

ell = ld_scores(blocks)                         # population LD scores (no bias adj)
dense = np.zeros((M, M), dtype=np.float32)      # block-diagonal LD for the infer step
for R, idx in blocks:
    dense[np.ix_(idx, idx)] = R


def make_beta(model, h2, rng):
    beta = np.zeros(M)
    if model == "infinitesimal":
        beta = rng.normal(0, 1, M)
    else:                                       # sparse p=0.01
        c = rng.random(M) < 0.01
        beta[c] = rng.normal(0, 1, c.sum())
    gv = sum(beta[ix] @ (blocks[b][0].astype(float) @ beta[ix])
             for b, ix in enumerate(idxs))
    return beta * np.sqrt(h2 / gv) if gv > 0 else beta


def sumstats(beta, rng):
    bhat = np.empty(M)
    for b, ix in enumerate(idxs):
        bhat[ix] = blocks[b][0].astype(float) @ beta[ix] + \
            (chols[b] @ rng.standard_normal(K)) / np.sqrt(N)
    return bhat


t0 = time.time()
n = np.full(M, float(N))
print(f"h2 estimation on coalescent LD, m={M}, N={N}, {REPS} reps\n")
print(f"{'model':>14} {'h2_true':>8} | {'LDSC':>16} | {'LDpred2-auto':>16}")
print("-" * 64)
for model in ("infinitesimal", "sparse"):
    for h2_true in (0.2, 0.5):
        ldsc_e, infer_e = [], []
        for rep in range(REPS):
            rng = np.random.default_rng(2000 + rep)
            beta = make_beta(model, h2_true, rng)
            bhat = sumstats(beta, rng)
            ldsc_e.append(ldsc_h2(n * bhat ** 2, ell, n, n_blocks=100).h2)
            r = ldpred2_auto_infer(dense, bhat, n, n_chains=8,
                                   burn_in=120, num_iter=150, seed=rep)
            infer_e.append(r.h2_est)
        print(f"{model:>14} {h2_true:>8.2f} | "
              f"{np.mean(ldsc_e):>6.3f} ± {np.std(ldsc_e):>5.3f}   | "
              f"{np.mean(infer_e):>6.3f} ± {np.std(infer_e):>5.3f}")
print(f"\n({time.time()-t0:.0f}s)")
