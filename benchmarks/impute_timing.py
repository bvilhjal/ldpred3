"""Running-time cost of LD imputation vs the fit it feeds.

Imputation is a one-time pre-processing step (per LD block: a small dense solve,
``O(k_t^3)`` in the typed block size), like LD construction -- not a per-sweep
cost -- so it amortises across however many fits / methods you then run. This
measures the wall-clock cost of ``impute_sumstats_blocks`` against one
``ldpred3_auto_annot`` fit, across genome sizes (and vs missingness), on realistic
coalescent LD (blocks of 500). Needs ``ld_library.npz``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/impute_timing.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ldpred3_auto_annot_blocks, impute_sumstats_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K = 500
N = 50000
BURN, ITER = 60, 120


def build(nb, miss, seed):
    rng = np.random.default_rng(seed)
    blocks = [(libR[b % libR.shape[0]].astype(np.float32),
               np.arange(b * K, (b + 1) * K)) for b in range(nb)]
    m = nb * K
    beta = np.zeros(m); c = rng.random(m) < 0.01
    beta[c] = rng.normal(0, 1, int(c.sum()))
    bhat = np.empty(m)
    for b in range(nb):
        R = libR[b % libR.shape[0]]
        ch = np.linalg.cholesky(R + 1e-4 * np.eye(K))
        ix = slice(b * K, (b + 1) * K)
        bhat[ix] = R @ beta[ix] + (ch @ rng.standard_normal(K)) / np.sqrt(N)
    typed = rng.random(m) >= miss
    A = (rng.random(m) < 0.2).astype(float)[:, None]
    return blocks, bhat, typed, A


# warm the JIT
_b, _bh, _t, _A = build(2, 0.3, 0)
ldpred3_auto_annot_blocks(_b, _bh, np.full(2 * K, float(N)), _A, burn_in=5, num_iter=5)

print("Imputation running time vs the auto_annot fit (blocks of 500, single core)\n")
print("(A) vs #SNPs (30% missing):")
print(f"{'#SNPs':>8} | {'impute s':>8} | {'fit s':>7} | {'impute/fit':>10} | {'impute us/SNP':>13}")
print("-" * 60)
for nb in (12, 24, 50, 100, 200):
    blocks, bhat, typed, A = build(nb, 0.3, seed=nb)
    n = np.full(nb * K, float(N))
    t0 = time.perf_counter()
    ir = impute_sumstats_blocks(bhat, blocks, typed, N)
    t_imp = time.perf_counter() - t0
    t0 = time.perf_counter()
    ldpred3_auto_annot_blocks(blocks, ir.beta_hat, ir.n_eff, A,
                              burn_in=BURN, num_iter=ITER, seed=1)
    t_fit = time.perf_counter() - t0
    print(f"{nb * K:>8} | {t_imp:>8.2f} | {t_fit:>7.2f} | {t_imp / t_fit:>9.0%} | "
          f"{t_imp / (nb * K) * 1e6:>13.1f}")

print("\n(B) vs missingness (m=50000):")
print(f"{'missing':>8} | {'impute s':>8} | {'impute us/SNP':>13}")
print("-" * 36)
for miss in (0.1, 0.3, 0.5, 0.7):
    blocks, bhat, typed, A = build(100, miss, seed=7)
    t0 = time.perf_counter()
    impute_sumstats_blocks(bhat, blocks, typed, N)
    t_imp = time.perf_counter() - t0
    print(f"{miss:>7.0%} | {t_imp:>8.2f} | {t_imp / 50000 * 1e6:>13.1f}")

print("\n(imputation is a one-time pre-step, O(k^3) per block in the typed block "
      "size; it amortises across every later fit / method.)")
