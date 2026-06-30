"""Does imputation help `ldpred3_auto_annot` under *random* missingness?

A stress test of the imputation layer that avoids the adversarial "drop exactly
the causal variants" setup of impute_annot.py. Here a random fraction of variants
is **untyped** (the realistic pattern -- a variant is missing because of the
genotyping array, not because it is causal). We fit `ldpred3_auto_annot` two ways
on the same data -- WITHOUT imputation (drop the untyped, fit on the typed subset)
and WITH imputation (impute the untyped marginals, fit on the union) -- and report
the PRS genetic R2 (population LD) for each, swept over:

  * missingness fraction
  * polygenicity p
  * heritability h2
  * number of SNPs

Realistic coalescent LD (ld_library), one informative functional annotation
(causals enriched), population LD used throughout so the only variable is the
missingness / imputation. Needs ``ld_library.npz``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/impute_missingness.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ldpred3_auto_annot_blocks, impute_sumstats_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K = 500
NB_MAX = 24
M_MAX = NB_MAX * K
N = 50000
FUNC_FRAC = 0.20
ENRICH = 12.0
BURN, ITER = 60, 120
REPS = 3

rng0 = np.random.default_rng(0)
R = [libR[b % libR.shape[0]].copy() for b in range(NB_MAX)]
chol = [np.linalg.cholesky(R[b] + 1e-4 * np.eye(K)) for b in range(NB_MAX)]
func_all = rng0.random(M_MAX) < FUNC_FRAC


def gv(nb, a, b):
    return sum(a[b_ * K:(b_ + 1) * K] @ (R[b_] @ b[b_ * K:(b_ + 1) * K])
               for b_ in range(nb))


def r2(nb, be, beta):
    num = gv(nb, be, beta); den = gv(nb, be, be) * gv(nb, beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def simulate(nb, p, h2, rng):
    m = nb * K
    func = func_all[:m]
    base = np.where(func, ENRICH, 1.0)
    pc = np.clip(base / base.sum() * (p * m), 0, 1)
    causal = rng.random(m) < pc
    beta = np.zeros(m); beta[causal] = rng.normal(0, 1, int(causal.sum()))
    g = gv(nb, beta, beta)
    if g > 0:
        beta *= np.sqrt(h2 / g)
    bhat = np.empty(m)
    for b in range(nb):
        ix = slice(b * K, (b + 1) * K)
        bhat[ix] = R[b] @ beta[ix] + (chol[b] @ rng.standard_normal(K)) / np.sqrt(N)
    return beta, bhat


def subset_blocks(blocks, keep):
    out, orig, off = [], [], 0
    for Rb, idx in blocks:
        loc = keep[idx]; k = int(loc.sum())
        if k:
            out.append((np.asarray(Rb)[np.ix_(loc, loc)], np.arange(off, off + k)))
            orig.append(np.asarray(idx)[loc]); off += k
    return out, np.concatenate(orig)


def cell(miss, p, h2, nb, seed0):
    m = nb * K
    blocks = [(R[b].astype(np.float32), np.arange(b * K, (b + 1) * K)) for b in range(nb)]
    A = func_all[:m].astype(float)[:, None]
    noimp, imp = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(seed0 + rep)
        beta, bhat = simulate(nb, p, h2, rng)
        typed = rng.random(m) >= miss                  # RANDOM missingness
        n_full = np.full(m, float(N))
        # without imputation: fit on the typed subset
        sub, orig = subset_blocks(blocks, typed)
        be0 = np.zeros(m)
        be0[orig] = ldpred3_auto_annot_blocks(sub, bhat[orig], n_full[orig],
                                              A[orig], burn_in=BURN, num_iter=ITER,
                                              seed=1).beta_est
        noimp.append(r2(nb, be0, beta))
        # with imputation: impute the untyped, fit on the union
        ir = impute_sumstats_blocks(bhat, blocks, typed, N)
        be1 = ldpred3_auto_annot_blocks(blocks, ir.beta_hat, ir.n_eff, A,
                                        burn_in=BURN, num_iter=ITER, seed=1).beta_est
        imp.append(r2(nb, be1, beta))
    return float(np.mean(noimp)), float(np.mean(imp))


def sweep(title, rows):
    print(f"\n{title}")
    print(f"{'value':>8} | {'no-impute':>9} | {'impute':>7} | {'delta':>7}")
    print("-" * 40)
    for label, (miss, p, h2, nb) in rows:
        a, b = cell(miss, p, h2, nb, seed0=1000)
        print(f"{label:>8} | {a:>9.3f} | {b:>7.3f} | {b - a:>+7.3f}")


t0 = time.time()
print(f"Imputation effect on auto_annot under RANDOM missingness, coalescent LD, "
      f"N={N}, functional={FUNC_FRAC:.0%}, {REPS} reps")
sweep("(A) missingness fraction (p=0.01, h2=0.5, m=6000):",
      [(f"{mi:.0%}", (mi, 0.01, 0.5, 12)) for mi in (0.1, 0.3, 0.5, 0.7)])
sweep("(B) polygenicity p (miss=30%, h2=0.5, m=6000):",
      [(f"{p:g}", (0.3, p, 0.5, 12)) for p in (0.001, 0.01, 0.1)])
sweep("(C) heritability h2 (miss=30%, p=0.01, m=6000):",
      [(f"{h:g}", (0.3, 0.01, h, 12)) for h in (0.2, 0.5, 0.8)])
sweep("(D) number of SNPs (miss=30%, p=0.01, h2=0.5):",
      [(f"{nb * K}", (0.3, 0.01, 0.5, nb)) for nb in (6, 12, 24)])
print(f"\n(genetic R2 vs the true genetic value under population LD; delta = "
      f"impute - no-impute. {time.time()-t0:.0f}s)")
