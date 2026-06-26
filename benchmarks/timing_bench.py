"""Running-time benchmark (single core) on realistic coalescent LD, m=50000.

Part A: per-method fit time (inf / grid / auto / annot), same chain length.
Part B: annot fit time vs #annotations K and theta_every (validates the
        O(m*K^2) theta-update cost and the default theta_every=1 choice).
"""
import sys, time, os
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ldpred2_by_blocks, ldpred2_auto_annot_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K = 500; NB = 100; M = NB * K
BURN, ITER = 80, 200

blocks = []
for b in range(NB):
    R = libR[b % libR.shape[0]].astype(np.float32)
    blocks.append((R, np.arange(b * K, (b + 1) * K)))

rng = np.random.default_rng(0)
bhat = rng.normal(0, 0.01, M)
n = np.full(M, 50000.0)


def timeit(fn, reps=2):
    fn()                                  # warm-up (numba JIT compile)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


print(f"single core (BLAS threads={os.environ.get('OPENBLAS_NUM_THREADS')}), "
      f"m={M}, blocks={NB}x{K}, burn={BURN}/iter={ITER}\n")

print("=== Part A: per-method fit time ===")
A1 = (rng.random((M, 1)) < 0.2).astype(float)
methods = {
    "inf":   lambda: ldpred2_by_blocks(blocks, bhat, n, method="inf", h2=0.5),
    "grid":  lambda: ldpred2_by_blocks(blocks, bhat, n, method="grid", h2=0.5,
                                       p=0.01, burn_in=BURN, num_iter=ITER),
    "auto":  lambda: ldpred2_by_blocks(blocks, bhat, n, method="auto",
                                       burn_in=BURN, num_iter=ITER, seed=1),
    "annot (K=1)": lambda: ldpred2_auto_annot_blocks(blocks, bhat, n, A1,
                                       burn_in=BURN, num_iter=ITER, seed=1),
}
print(f"{'method':>14} | {'time (s)':>9}")
print("-" * 28)
for name, fn in methods.items():
    print(f"{name:>14} | {timeit(fn):>9.2f}")

print("\n=== Part B: annot fit time vs #annotations K and theta_every ===")
print(f"{'K':>4} | {'theta_every=1':>14} | {'theta_every=10':>15}")
print("-" * 40)
for Kann in [1, 5, 20, 50, 100]:
    Ak = (rng.random((M, Kann)) < 0.2).astype(float)
    t1 = timeit(lambda: ldpred2_auto_annot_blocks(blocks, bhat, n, Ak,
                burn_in=BURN, num_iter=ITER, theta_every=1, seed=1), reps=1)
    t10 = timeit(lambda: ldpred2_auto_annot_blocks(blocks, bhat, n, Ak,
                burn_in=BURN, num_iter=ITER, theta_every=10, seed=1), reps=1)
    print(f"{Kann:>4} | {t1:>14.2f} | {t10:>15.2f}")
