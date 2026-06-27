"""Running time of LDpred2-auto inference: dense vs streaming, as m grows.

The dense path assembles an m x m LD matrix (O(m^2) per sweep); the streaming
path runs one LD block at a time (O(m * block_size)). This times both on the
same data across increasing m, plus the streaming h2 to confirm agreement.
Needs ``ld_library.npz`` in the cwd. Single core recommended.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ldpred2_auto_infer

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K = 500
NREF = 2000
SHRINK = 0.05
N = 50000
H2, P = 0.5, 0.01
NCHAINS, BURN, ITER = 6, 100, 120
M_LIST = [2000, 4000, 8000, 16000]


def build(nb, seed=0):
    rng = np.random.default_rng(seed)
    pop, chol, ref, idxs = [], [], [], []
    for b in range(nb):
        Rp = libR[b % libR.shape[0]].copy()
        cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
        Z = rng.standard_normal((NREF, K)) @ cp.T; Z = (Z - Z.mean(0)) / Z.std(0)
        Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
        pop.append(Rp); chol.append(cp)
        ref.append((Rr.astype(np.float32), np.arange(b * K, (b + 1) * K)))
        idxs.append(np.arange(b * K, (b + 1) * K))
    return pop, chol, ref, idxs


def sumstats(pop, chol, idxs, m, rng):
    beta = np.zeros(m); c = rng.random(m) < P
    beta[c] = rng.standard_normal(c.sum())
    gv = sum(beta[ix] @ (pop[i] @ beta[ix]) for i, ix in enumerate(idxs))
    beta *= np.sqrt(H2 / gv)
    bh = np.empty(m)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i] @ beta[ix] + (chol[i] @ rng.standard_normal(K)) / np.sqrt(N)
    return bh


# warm up both Numba kernels at small size
pop, chol, ref, idxs = build(2)
bh = sumstats(pop, chol, idxs, 1000, np.random.default_rng(0))
dense0 = np.zeros((1000, 1000), np.float32)
for R, ix in ref:
    dense0[np.ix_(ix, ix)] = R
ldpred2_auto_infer(dense0, bh, N, n_chains=2, burn_in=20, num_iter=20, seed=0)
ldpred2_auto_infer(ref, bh, N, n_chains=2, burn_in=20, num_iter=20, seed=0)

print(f"LDpred2-auto inference time (s), single core, {NCHAINS} chains, "
      f"burn {BURN}/iter {ITER}, blocks of {K}\n")
print(f"{'m':>7} | {'dense (s)':>10} | {'stream (s)':>11} | {'speedup':>7} | "
      f"{'stream h2':>9}")
print("-" * 56)
for m in M_LIST:
    nb = m // K
    pop, chol, ref, idxs = build(nb)
    rng = np.random.default_rng(42)
    bh = sumstats(pop, chol, idxs, m, rng)

    dense = np.zeros((m, m), dtype=np.float32)
    for R, ix in ref:
        dense[np.ix_(ix, ix)] = R
    t = time.perf_counter()
    rd = ldpred2_auto_infer(dense, bh, N, n_chains=NCHAINS, burn_in=BURN,
                            num_iter=ITER, seed=1)
    t_dense = time.perf_counter() - t
    del dense

    t = time.perf_counter()
    rs = ldpred2_auto_infer(ref, bh, N, n_chains=NCHAINS, burn_in=BURN,
                            num_iter=ITER, seed=1)
    t_stream = time.perf_counter() - t
    print(f"{m:>7} | {t_dense:>10.2f} | {t_stream:>11.2f} | "
          f"{t_dense / t_stream:>6.1f}x | {rs.h2_est:>9.3f}")
