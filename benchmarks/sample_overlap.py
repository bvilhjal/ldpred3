"""Sample-overlap validation for the bivariate genetic-correlation estimators.

Overlapping GWAS samples induce a correlation in the two traits' sampling noise
(controlled here by ``rho_e``), which inflates a naive genetic-correlation
estimate even when the traits are genetically independent. This checks that the
corrections handle it:

  - bivariate LDSC: a free cross-trait *intercept* should absorb the overlap;
    constraining it to 0 should leave the bias.
  - bivariate LDpred3: passing ``cross_corr=rho_e`` should remove the bias;
    leaving ``cross_corr=0`` should not.

Needs ``ld_library.npz`` in the cwd.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ld_scores, ldsc_rg, ldpred3_auto_bivariate_blocks

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K, NB = 500, 16
M = NB * K
N1, N2 = 10000, 10000
NREF = 2000
SHRINK = 0.05
H2, P = 0.3, 0.01
RHO_E = 0.5            # cross-trait noise correlation from sample overlap
REPS = 5

idxs = [np.arange(b * K, (b + 1) * K) for b in range(NB)]
rng0 = np.random.default_rng(0)
pop, chol_pop, ref = [], [], []
for b in range(NB):
    Rp = libR[b].copy(); cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T; Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), idxs[b])); chol_pop.append(cp)
    ref.append((Rr.astype(np.float32), idxs[b]))
ell = ld_scores(ref, n_ref=NREF)


def gv(a, b):
    return sum(a[ix] @ (pop[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def sim(rg, rng):
    c = rng.random(M) < P
    L = np.linalg.cholesky([[1, rg], [rg, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
    b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
    return b1 * np.sqrt(H2 / gv(b1, b1)), b2 * np.sqrt(H2 / gv(b2, b2))


def sumstats_overlap(b1, b2, rng):
    """GWAS for both traits with rho_e-correlated sampling noise (overlap)."""
    bh1 = np.empty(M); bh2 = np.empty(M)
    for i, ix in enumerate(idxs):
        u1 = rng.standard_normal(K)
        u2 = RHO_E * u1 + np.sqrt(1 - RHO_E ** 2) * rng.standard_normal(K)
        e1 = (chol_pop[i] @ u1) / np.sqrt(N1)
        e2 = (chol_pop[i] @ u2) / np.sqrt(N2)
        Rb = pop[i][0].astype(float)
        bh1[ix] = Rb @ b1[ix] + e1
        bh2[ix] = Rb @ b2[ix] + e2
    return bh1, bh2


t0 = time.time()
print(f"Sample-overlap (rho_e={RHO_E}), m={M}, N1={N1}, N2={N2}, {REPS} reps\n")
print(f"{'rg_true':>7} | {'LDSC (no icpt)':>14} | {'LDSC (free icpt)':>16} | "
      f"{'biv (cc=0)':>11} | {'biv (cc=rho)':>12}")
print("-" * 76)
for rg in (0.0, 0.5):
    a, b, c, d = [], [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(500 + rep)
        b1, b2 = sim(rg, rng)
        bh1, bh2 = sumstats_overlap(b1, b2, rng)
        a.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=80,
                         constrain_intercept=0.0).rg)
        b.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=80).rg)
        c.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=120,
                                               num_iter=150, cross_corr=0.0, seed=rep).rg)
        d.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=120,
                                               num_iter=150, cross_corr=RHO_E, seed=rep).rg)
    print(f"{rg:>7.1f} | {np.mean(a):>14.3f} | {np.mean(b):>16.3f} | "
          f"{np.mean(c):>11.3f} | {np.mean(d):>12.3f}")
print(f"\n(true rg in each row; uncorrected columns should be biased upward)\n"
      f"({time.time()-t0:.0f}s)")
