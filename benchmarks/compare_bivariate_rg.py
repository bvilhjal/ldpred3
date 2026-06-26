"""Realistic rg estimation: fit with an LD matrix estimated from a finite
reference panel (not the true population LD used to generate the GWAS), on
coalescent LD. This is the dominant real-world error source that the earlier
'fit with the true LD' comparison omitted.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ld_scores, ldsc_rg, ldpred2_auto_bivariate_blocks, ldpred2_by_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K, NB = 500, 16
M = NB * K
NREF = 2000               # reference-panel size for LD estimation
N1, N2 = 50000, 20000
P = 0.01
H2 = 0.5
REPS = 5
SHRINK = 0.05             # shrink ref LD toward identity for sampler stability

rng0 = np.random.default_rng(0)
pop, chol_pop, ref, chol_ref, idxs = [], [], [], [], []
for b in range(NB):
    Rp = libR[b].copy()
    cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    # reference panel: NREF latent-Gaussian "genotypes" ~ N(0, Rp), sample corr.
    Z = rng0.standard_normal((NREF, K)) @ cp.T
    Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (Z.T @ Z) / NREF
    Rr = (1 - SHRINK) * Rr + SHRINK * np.eye(K)        # stabilise
    pop.append((Rp.astype(np.float32), np.arange(b*K, (b+1)*K)))
    ref.append((Rr.astype(np.float32), np.arange(b*K, (b+1)*K)))
    chol_pop.append(cp)
    idxs.append(np.arange(b*K, (b+1)*K))

ell = ld_scores(ref, n_ref=NREF)          # LD scores from the reference panel


def gv(a, b):
    return sum(a[ix] @ (pop[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def sim(rg, rng):
    c = rng.random(M) < P
    L = np.linalg.cholesky([[1, rg], [rg, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
    b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
    b1 *= np.sqrt(H2 / gv(b1, b1)); b2 *= np.sqrt(H2 / gv(b2, b2))
    return b1, b2


def sumstats(beta, n, rng):                # GWAS from the TRUE population LD
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i][0].astype(float) @ beta[ix] + (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


t0 = time.time()
print(f"REALISTIC rg: ref-panel LD (Nref={NREF}), coalescent LD, m={M}, "
      f"N1={N1}, N2={N2}, p={P}, {REPS} reps\n")
print(f"{'rg_true':>7} | {'bivariate LDSC':>18} | {'bivariate LDpred2':>18}")
print("-" * 52)
for rg in (0.0, 0.3, 0.6, 0.9):
    ld, bp = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(500 + rep)
        b1, b2 = sim(rg, rng)
        bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
        ld.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=80).rg)
        bp.append(ldpred2_auto_bivariate_blocks(ref, bh1, bh2, N1, N2,
                                                burn_in=120, num_iter=150, seed=rep).rg)
    print(f"{rg:>7.1f} | {np.mean(ld):>7.3f} ± {np.std(ld):>5.3f}    | "
          f"{np.mean(bp):>7.3f} ± {np.std(bp):>5.3f}")
print(f"\n({time.time()-t0:.0f}s)")
