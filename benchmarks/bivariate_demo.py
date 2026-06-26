"""Bivariate LDpred2-auto: a weak trait borrows strength from a correlated,
well-powered trait. Self-contained (heterogeneous AR(1) LD; no external data).

For a range of true genetic correlations we fit trait 2 (low N) on its own
(univariate auto) and jointly with a well-powered trait 1 (bivariate auto), and
report the genetic R2 of the trait-2 PRS plus the recovered rg.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ldpred2_auto_bivariate_blocks, ldpred2_by_blocks

K, NB = 200, 12
M = NB * K
N1, N2 = 100000, 3000           # trait 1 well powered, trait 2 weak
H2 = (0.5, 0.5)
P = 0.05
REPS = 5

rng0 = np.random.default_rng(2)
blocks, chols, idxs = [], [], []
for b in range(NB):
    rho = rng0.uniform(0.0, 0.8)
    d = np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    R = (rho ** d).astype(np.float64)
    blocks.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(np.linalg.cholesky(R + 1e-6 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))


def gv(a, b):
    return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def sim(rg, rng):
    causal = rng.random(M) < P
    L = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    raw = L @ rng.standard_normal((2, causal.sum()))
    b1 = np.zeros(M); b2 = np.zeros(M)
    b1[causal] = raw[0]; b2[causal] = raw[1]
    b1 *= np.sqrt(H2[0] / gv(b1, b1)); b2 *= np.sqrt(H2[1] / gv(b2, b2))
    return b1, b2


def sumstats(beta, n, rng):
    bhat = np.empty(M)
    for i, ix in enumerate(idxs):
        bhat[ix] = blocks[i][0].astype(float) @ beta[ix] + \
            (chols[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bhat


def genetic_r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


t0 = time.time()
print(f"trait2 genetic R2: univariate vs bivariate (N1={N1}, N2={N2}, "
      f"h2=0.5, p={P}, m={M}, {REPS} reps)\n")
print(f"{'rg_true':>7} | {'trait2 alone':>12} | {'trait2 joint':>12} | "
      f"{'gain':>6} | {'rg_est':>6}")
print("-" * 60)
for rg in (0.0, 0.3, 0.6, 0.9):
    solo, joint, rge = [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(70 + rep)
        b1, b2 = sim(rg, rng)
        bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
        res = ldpred2_auto_bivariate_blocks(blocks, bh1, bh2, N1, N2,
                                            burn_in=150, num_iter=200, seed=rep)
        s = ldpred2_by_blocks(blocks, bh2, np.full(M, float(N2)), method="auto",
                              burn_in=150, num_iter=200, seed=rep)
        joint.append(genetic_r2(res.beta2_est, b2))
        solo.append(genetic_r2(s, b2)); rge.append(res.rg)
    a, j = np.mean(solo), np.mean(joint)
    print(f"{rg:>7.1f} | {a:>12.3f} | {j:>12.3f} | {j-a:>+6.3f} | {np.mean(rge):>+6.2f}")
print(f"\n({time.time()-t0:.0f}s)")
