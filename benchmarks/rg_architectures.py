"""Genetic-correlation (rg) estimation across genetic architectures.

For each of four architectures (infinitesimal, sparse, polygenic, major-locus)
and a sweep of true genetic correlations, two correlated traits are simulated
(shared causal variants with bivariate-normal effects of correlation rg), a GWAS
is run on realistic coalescent LD, and rg is estimated by **bivariate LDSC**
(`ldsc_rg`) and **bivariate LDpred3** (`ldpred3_auto_bivariate_blocks`) against
the truth. LD is fit from a finite **reference panel** (the realistic mismatch).
Needs ``ld_library.npz``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/rg_architectures.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ld_scores, ldsc_rg, ldpred3_auto_bivariate_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
NREF = 2000
N1, N2 = 50000, 20000
H2 = 0.5
SHRINK = 0.05
REPS = 4
ARCHS = ["infinitesimal", "sparse", "polygenic", "major_locus"]
RGS = [0.0, 0.3, 0.6, 0.9]

rng0 = np.random.default_rng(0)
pop, chol_pop, ref, idxs = [], [], [], []
for b in range(NB):
    Rp = libR[b % libR.shape[0]].copy()
    cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T
    Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    ref.append((Rr.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chol_pop.append(cp)
    idxs.append(np.arange(b * K, (b + 1) * K))

ell = ld_scores(ref, n_ref=NREF)


def gv(a, b):
    return sum(a[ix] @ (pop[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def sim(arch, rg, rng):
    """Two traits with shared causal variants correlated by ``rg``."""
    L = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    b1 = np.zeros(M); b2 = np.zeros(M)
    if arch == "infinitesimal":
        c = np.ones(M, bool)
    elif arch == "sparse":
        c = rng.random(M) < 0.01
    elif arch == "polygenic":
        c = rng.random(M) < 0.2
    else:                                       # major_locus: sparse bg + 3 huge
        c = rng.random(M) < 0.02
    raw = L @ rng.standard_normal((2, int(c.sum())))
    b1[c] = raw[0]; b2[c] = raw[1]
    if arch == "major_locus":
        maj = rng.choice(M, 3, replace=False)
        rawm = (L @ rng.standard_normal((2, 3))) * 4.0
        b1[maj] = rawm[0]; b2[maj] = rawm[1]
    b1 *= np.sqrt(H2 / gv(b1, b1)); b2 *= np.sqrt(H2 / gv(b2, b2))
    return b1, b2


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i][0].astype(float) @ beta[ix] + \
            (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


t0 = time.time()
print(f"Genetic correlation across architectures, ref-panel LD (Nref={NREF}), "
      f"coalescent LD, m={M}, N1={N1}, N2={N2}, {REPS} reps\n")
print(f"{'architecture':>14} | {'rg_true':>7} | {'bivariate LDSC':>17} | "
      f"{'bivariate LDpred3':>18}")
print("-" * 68)
for arch in ARCHS:
    for rg in RGS:
        ld, bp = [], []
        for rep in range(REPS):
            rng = np.random.default_rng(600 + rep)
            b1, b2 = sim(arch, rg, rng)
            bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
            ld.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=60).rg)
            bp.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2,
                                                    burn_in=120, num_iter=150,
                                                    seed=rep).rg)
        print(f"{arch:>14} | {rg:>7.1f} | {np.mean(ld):>7.3f} ± {np.std(ld):>5.3f}   "
              f"| {np.mean(bp):>7.3f} ± {np.std(bp):>5.3f}")
    print()
print(f"(true rg per row; estimates should track it across architectures. "
      f"{time.time()-t0:.0f}s)")
