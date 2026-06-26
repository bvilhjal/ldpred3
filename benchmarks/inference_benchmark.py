"""Inference accuracy AND running time for the summary-statistic estimators,
under realistic reference-panel LD (coalescent population LD; fit/score with an
LD estimated from a finite reference panel).

Heritability:        LDSC (ldsc_h2)  vs  LDpred2-auto (ldpred2_auto_infer)
Genetic correlation: bivariate LDSC (ldsc_rg)  vs  bivariate LDpred2
                     (ldpred2_auto_bivariate)

Reports the estimate (mean ± SD vs the known truth) and the wall-clock time per
run. Needs ``ld_library.npz`` in the cwd. Single core recommended
(OPENBLAS_NUM_THREADS=1).
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import (ld_scores, ldsc_h2, ldsc_rg, ldpred2_auto_infer,
                       ldpred2_auto_bivariate_blocks)

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
N1, N2 = 50000, 20000
NREF = 2000
SHRINK = 0.05
H2, RG = 0.5, 0.6
REPS = 5

rng0 = np.random.default_rng(0)
pop, chol_pop, ref, idxs = [], [], [], []
for b in range(NB):
    Rp = libR[b].copy(); cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T; Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    ref.append((Rr.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chol_pop.append(cp); idxs.append(np.arange(b * K, (b + 1) * K))
ell = ld_scores(ref, n_ref=NREF)
dense = np.zeros((M, M), dtype=np.float32)
for R, idx in ref:
    dense[np.ix_(idx, idx)] = R


def gv(a, b):
    return sum(a[i_] @ (pop[i][0].astype(float) @ b[i_]) for i, i_ in enumerate(idxs))


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i][0].astype(float) @ beta[ix] + (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


def sim_pair(rng):
    c = rng.random(M) < 0.01
    L = np.linalg.cholesky([[1, RG], [RG, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
    b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
    return b1 * np.sqrt(H2 / gv(b1, b1)), b2 * np.sqrt(H2 / gv(b2, b2))


def timed(fn):
    t = time.perf_counter(); v = fn(); return v, time.perf_counter() - t


# Warm up the Numba kernels (infer / bivariate) so timings are steady-state.
rng = np.random.default_rng(1)
b1, b2 = sim_pair(rng); bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
ldpred2_auto_infer(dense, bh1, N1, n_chains=4, burn_in=40, num_iter=40, seed=0)
ldpred2_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=40, num_iter=40,
                              h2_cap=(H2, H2), seed=0)

def marginal_h2(bh, n):
    """Naive no-LD heritability: assume each SNP independent (LD score = 1)."""
    return (np.mean(n * bh ** 2) - 1.0) * M / n


def marginal_rg(bh1, bh2, n1, n2):
    """Naive no-LD genetic correlation (LD score = 1)."""
    gcov = np.mean(bh1 * bh2) * M
    h1 = marginal_h2(bh1, n1); h2 = marginal_h2(bh2, n2)
    return gcov / np.sqrt(max(h1 * h2, 1e-12))


h2_marg, h2_ldsc, h2_inf = [], [], []
t_margh2, t_ldsch2, t_inf = [], [], []
rg_marg, rg_ldsc, rg_biv = [], [], []
t_margrg, t_ldscrg, t_biv = [], [], []
for rep in range(REPS):
    rng = np.random.default_rng(500 + rep)
    b1, b2 = sim_pair(rng)
    bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)

    (r, dt) = timed(lambda: marginal_h2(bh1, N1))
    h2_marg.append(r); t_margh2.append(dt)
    (r, dt) = timed(lambda: ldsc_h2(N1 * bh1 ** 2, ell, N1, n_blocks=80))
    h2_ldsc.append(r.h2); t_ldsch2.append(dt)
    (r, dt) = timed(lambda: ldpred2_auto_infer(dense, bh1, N1, n_chains=8,
                                               burn_in=120, num_iter=150, seed=rep))
    h2_inf.append(r.h2_est); t_inf.append(dt)

    (r, dt) = timed(lambda: marginal_rg(bh1, bh2, N1, N2))
    rg_marg.append(r); t_margrg.append(dt)
    (r, dt) = timed(lambda: ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=80))
    rg_ldsc.append(r.rg); t_ldscrg.append(dt)
    (r, dt) = timed(lambda: ldpred2_auto_bivariate_blocks(
        ref, bh1, bh2, N1, N2, burn_in=120, num_iter=150, seed=rep))
    rg_biv.append(r.rg); t_biv.append(dt)


def row(name, est, t, truth):
    print(f"{name:>20} | {np.mean(est):>6.3f} ± {np.std(est):>5.3f} "
          f"(true {truth:.2f}) | {np.mean(t):>9.4f} s")


print(f"\nInference accuracy & running time — realistic reference-panel LD "
      f"(Nref={NREF}), coalescent, m={M}, N1={N1}, N2={N2}, {REPS} reps\n")
print(f"{'method':>20} | {'estimate':>22} | {'time/run':>11}")
print("-" * 62)
print("heritability:")
row("marginal (no LD)", h2_marg, t_margh2, H2)
row("LDSC", h2_ldsc, t_ldsch2, H2)
row("LDpred2-auto", h2_inf, t_inf, H2)
print("genetic correlation:")
row("marginal (no LD)", rg_marg, t_margrg, RG)
row("bivariate LDSC", rg_ldsc, t_ldscrg, RG)
row("bivariate LDpred2", rg_biv, t_biv, RG)
