"""Compare genetic-correlation estimates: bivariate LD Score regression vs
bivariate LDpred2-auto, from the same two-trait summary statistics.

Both estimate rg with no individual-level data; we report each against the known
true rg over a range of values. Self-contained (heterogeneous AR(1) LD).
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ld_scores, ldsc_rg, ldpred2_auto_bivariate_blocks

K, NB, N1, N2 = 200, 60, 40000, 20000
M = NB * K
REPS = 5

rng0 = np.random.default_rng(1)
blocks, chols, idxs = [], [], []
for b in range(NB):
    rho = rng0.uniform(0.0, 0.9)
    d = np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    R = (rho ** d).astype(np.float64)
    blocks.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(np.linalg.cholesky(R + 1e-6 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))
ell = ld_scores(blocks)


def gv(a, b):
    return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = blocks[i][0].astype(float) @ beta[ix] + \
            (chols[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


def sim(rg, rng):
    c = rng.random(M) < 0.05
    L = np.linalg.cholesky([[1, rg], [rg, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
    b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
    b1 *= np.sqrt(0.5 / gv(b1, b1)); b2 *= np.sqrt(0.5 / gv(b2, b2))
    return b1, b2


t0 = time.time()
print(f"genetic correlation estimation, m={M}, N1={N1}, N2={N2}, {REPS} reps\n")
print(f"{'rg_true':>7} | {'bivariate LDSC':>18} | {'bivariate LDpred2':>18}")
print("-" * 52)
for rg in (0.0, 0.3, 0.6, 0.9):
    ld, bp = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(500 + rep)
        b1, b2 = sim(rg, rng)
        bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
        ld.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=60).rg)
        bp.append(ldpred2_auto_bivariate_blocks(blocks, bh1, bh2, N1, N2,
                                                burn_in=120, num_iter=150, seed=rep).rg)
    print(f"{rg:>7.1f} | {np.mean(ld):>7.3f} ± {np.std(ld):>5.3f}    | "
          f"{np.mean(bp):>7.3f} ± {np.std(bp):>5.3f}")
print(f"\n({time.time()-t0:.0f}s)")
