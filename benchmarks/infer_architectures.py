"""Inference (h², polygenicity) across genetic architectures.

Does summary-statistic inference hold up as the genetic architecture changes?
For each of four architectures (infinitesimal, sparse, polygenic, major-locus) at
a fixed true h²=0.5, this simulates a GWAS on realistic coalescent LD, fits
heritability with **LDSC** and **LDpred3-auto** and polygenicity with LDpred3-auto,
and reports each estimate against the truth plus the empirical 95% CI coverage.
LD is estimated from a finite **reference panel** (the dominant real-world error),
not the population LD that generates the GWAS. Needs ``ld_library.npz``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/infer_architectures.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ld_scores, ldsc_h2, ldpred3_auto_infer

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
N = 50000
NREF = 2000
SHRINK = 0.05
H2 = 0.5
REPS = 10
ARCHS = ["infinitesimal", "sparse", "polygenic", "major_locus"]
P_TRUE = {"infinitesimal": 1.0, "sparse": 0.01, "polygenic": 0.2, "major_locus": 0.02}

# pop = true LD (generates the GWAS); ref = finite-reference-panel LD (used to fit).
rng0 = np.random.default_rng(0)
pop, chols, ref, idxs = [], [], [], []
for b in range(NB):
    R = libR[b % libR.shape[0]].copy()
    cp = np.linalg.cholesky(R + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T
    Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    ref.append((Rr.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(cp)
    idxs.append(np.arange(b * K, (b + 1) * K))

ell = ld_scores(ref, n_ref=NREF)
dense = np.zeros((M, M), dtype=np.float32)
for R, idx in ref:
    dense[np.ix_(idx, idx)] = R


def make_beta(model, rng):
    beta = np.zeros(M)
    if model == "infinitesimal":
        beta = rng.normal(0, 1, M)
    elif model == "sparse":
        c = rng.random(M) < 0.01; beta[c] = rng.normal(0, 1, c.sum())
    elif model == "polygenic":
        c = rng.random(M) < 0.2; beta[c] = rng.normal(0, 1, c.sum())
    else:                                       # major_locus: few huge + sparse bg
        c = rng.random(M) < 0.02; beta[c] = rng.normal(0, 1, c.sum()) * 0.3
        maj = rng.choice(M, 3, replace=False); beta[maj] = rng.choice([-1, 1], 3) * 4
    gv = sum(beta[ix] @ (pop[b][0].astype(float) @ beta[ix])
             for b, ix in enumerate(idxs))
    return beta * np.sqrt(H2 / gv) if gv > 0 else beta


def sumstats(beta, rng):
    bhat = np.empty(M)
    for b, ix in enumerate(idxs):
        bhat[ix] = pop[b][0].astype(float) @ beta[ix] + \
            (chols[b] @ rng.standard_normal(K)) / np.sqrt(N)
    return bhat


t0 = time.time()
n = np.full(M, float(N))
print(f"Inference across architectures, coalescent LD (ref panel Nref={NREF}), "
      f"m={M}, N={N}, h2={H2}, {REPS} reps\n")
print(f"{'architecture':>14} | {'LDSC h2':>13} | {'LDpred3 h2':>13} | {'h2 cov':>6} "
      f"| {'p_true':>6} | {'LDpred3 p':>15} | {'p cov':>5}")
print("-" * 86)
for model in ARCHS:
    ld_e, h2_e, p_e = [], [], []
    h2_hit = p_hit = 0
    pt = P_TRUE[model]
    for rep in range(REPS):
        rng = np.random.default_rng(3000 + rep)
        beta = make_beta(model, rng)
        bhat = sumstats(beta, rng)
        ld_e.append(ldsc_h2(n * bhat ** 2, ell, n, n_blocks=100).h2)
        r = ldpred3_auto_infer(dense, bhat, n, n_chains=8, burn_in=120,
                               num_iter=150, seed=rep)
        h2_e.append(r.h2_est); p_e.append(r.p_est)
        h2_hit += r.h2_ci[0] <= H2 <= r.h2_ci[1]
        p_hit += r.p_ci[0] <= pt <= r.p_ci[1]
    print(f"{model:>14} | {np.mean(ld_e):>6.3f} ± {np.std(ld_e):>4.3f} | "
          f"{np.mean(h2_e):>6.3f} ± {np.std(h2_e):>4.3f} | {h2_hit / REPS:>6.2f} | "
          f"{pt:>6.3f} | {np.mean(p_e):>7.4f} ± {np.std(p_e):>5.4f} | "
          f"{p_hit / REPS:>5.2f}")

print(f"\n(true h2={H2} every row; p_true is the causal fraction. {time.time()-t0:.0f}s)")
