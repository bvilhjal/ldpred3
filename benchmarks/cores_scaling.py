"""Multi-core scaling of the LDpred2-auto Gibbs sampler (``--ncores``).

Self-contained. Times the *packed* auto sampler (the one ``--ncores>1`` uses,
whose per-sweep block loop is parallelised with Numba ``prange``) at increasing
thread counts and reports the speed-up and parallel efficiency. ncores=1 here is
the serial packed kernel, so this isolates parallel scaling of one kernel rather
than the streaming-vs-packed switch. Needs Numba for any parallelism.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/cores_scaling.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2._numba import HAVE_NUMBA
from pyldpred2.simulate import simulate_genotypes
from pyldpred2.ld import compute_ld_blocks
from pyldpred2.ldpred2 import _gibbs_blocks

NB, K = 40, 500            # m = 20000, large blocks -> heavy per-sweep matmuls
M = NB * K
N_REF = 2000
N_GWAS = 50000
H2, P = 0.5, 0.01
RHO = 0.8
BURN, ITER = 100, 200
CORES = [1, 2, 4]


def build(seed):
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, M)
    G, _ = simulate_genotypes(N_REF, [K] * NB, maf, RHO, rng)
    blocks = compute_ld_blocks(G, block_size=K)
    beta = np.zeros(M); c = rng.random(M) < P
    beta[c] = rng.standard_normal(int(c.sum()))
    beta_hat = np.empty(M)
    for R, ix in [(R.astype(float), idx) for R, idx in blocks]:
        ch = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        beta_hat[ix] = R @ beta[ix] + (ch @ rng.standard_normal(len(ix))) / np.sqrt(N_GWAS)
    return blocks, beta_hat


print(f"Cores scaling, auto fit, m={M} ({NB}x{K}), N_gwas={N_GWAS}, "
      f"burn_in={BURN}, num_iter={ITER}, {os.cpu_count()} CPUs\n")
if not HAVE_NUMBA:
    print("Numba not installed; --ncores has no effect (pure-Python single core).")
    print("Install numba (`pip install numba`) to measure parallel scaling.")
    sys.exit(0)

blocks, beta_hat = build(0)
n = np.full(M, float(N_GWAS))
common = dict(sparse=False, seed=0, estimate_hyper=True, h2_bounds=(1e-4, 1.0))


def fit(nc, burn, it):
    return _gibbs_blocks(blocks, beta_hat, n, 0.1, 0.1,
                         burn_in=burn, num_iter=it, ncores=nc, **common)


print(f"{'ncores':>6} | {'fit time (s)':>12} | {'speed-up':>8} | {'efficiency':>10}")
print("-" * 46)
t1 = None
for nc in CORES:
    fit(nc, 5, 5)                                  # warm this kernel + set threads
    t = time.time()
    fit(nc, BURN, ITER)
    dt = time.time() - t
    if t1 is None:
        t1 = dt
    print(f"{nc:>6} | {dt:>12.2f} | {t1/dt:>7.2f}x | {100*t1/dt/nc:>9.0f}%")
