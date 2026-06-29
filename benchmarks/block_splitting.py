"""Optimal LD-block splitting (Prive 2022) vs fixed-size blocks.

Self-contained. Builds one region whose true LD consists of several independent
sub-blocks of *unequal* size (recombination valleys between them), then splits it
two ways at the same maximum block size:

  * fixed   -- cut every ``max_size`` SNPs, blind to structure (boundaries fall
               mid-block).
  * optimal -- ``optimal_ld_blocks`` puts boundaries in the low-LD valleys.

Reports, for each split: #blocks, the **discarded between-block LD^2** (the LD
thrown away when blocks are treated as independent -- lower is better), the dense
per-block storage (sum of k^2), and the genetic R2 of a block-diagonal auto PRS.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/block_splitting.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2.simulate import simulate_genotypes
from pyldpred2.ld import compute_ld_blocks
from pyldpred2 import ldpred2_by_blocks, optimal_ld_blocks

TRUE_SIZES = [137, 211, 89, 256, 170, 137]    # unequal true LD blocks, sum = 1000
M = sum(TRUE_SIZES)
MAX_SIZE = 250
MIN_SIZE = 30
N_REF = 4000
N_GWAS = 20000
H2, P = 0.5, 0.02
RHO = 0.9
REPS = 3


def build(seed):
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, M)
    # Independent true sub-blocks -> one region with valleys between them.
    G, _ = simulate_genotypes(N_REF, TRUE_SIZES, maf, RHO, rng)
    Gs = (G - G.mean(0)) / G.std(0)
    R = (Gs.T @ Gs) / N_REF                    # full dense region LD (m x m)

    causal = rng.random(M) < P
    beta = np.zeros(M); beta[causal] = rng.standard_normal(int(causal.sum()))
    beta *= np.sqrt(H2 / (beta @ (R @ beta)))
    chol = np.linalg.cholesky(R + 1e-6 * np.eye(M))
    beta_hat = R @ beta + (chol @ rng.standard_normal(M)) / np.sqrt(N_GWAS)
    return R, beta, beta_hat


def discarded_ld2(R, bounds):
    """Sum of r^2 over variant pairs that fall in different blocks (i < j)."""
    R2 = R * R
    total = np.triu(R2, 1).sum()
    within = sum(np.triu(R2[s:e, s:e], 1).sum() for s, e in bounds)
    return float(total - within)


def fixed_bounds(m, size):
    return [(s, min(s + size, m)) for s in range(0, m, size)]


def fit_r2(R, bounds, beta_hat, beta, n):
    blocks = [(R[s:e, s:e].astype(np.float32), np.arange(s, e)) for s, e in bounds]
    be = ldpred2_by_blocks(blocks, beta_hat, n, method="auto",
                           global_hyper=False, burn_in=60, num_iter=120, seed=0)
    num = be @ (R @ beta); den = (be @ (R @ be)) * (beta @ (R @ beta))
    return float(num * num / den) if den > 0 else 0.0


n = np.full(M, float(N_GWAS))
_R, _b, _bh = build(0)                          # warm up the JIT
ldpred2_by_blocks([(_R.astype(np.float32), np.arange(M))], _bh, n, method="auto",
                  global_hyper=False, burn_in=10, num_iter=10, seed=0)

t0 = time.time()
print(f"LD-block splitting, AR(1) sub-blocks {TRUE_SIZES} (m={M}), max_size="
      f"{MAX_SIZE}, Nref={N_REF}, N_gwas={N_GWAS}, h2={H2}, p={P}, {REPS} reps\n")
print(f"{'split':>9} | {'#blocks':>7} | {'discarded LD2':>13} | {'storage k^2':>11} | {'R2':>6}")
print("-" * 60)

agg = {"fixed": [], "optimal": []}
for rep in range(REPS):
    R, beta, beta_hat = build(100 + rep)
    fb = fixed_bounds(M, MAX_SIZE)
    ob, _cost = optimal_ld_blocks(R, max_size=MAX_SIZE, min_size=MIN_SIZE,
                                  window=MAX_SIZE)
    for name, bounds in (("fixed", fb), ("optimal", ob)):
        store = sum((e - s) ** 2 for s, e in bounds)
        agg[name].append((len(bounds), discarded_ld2(R, bounds), store,
                          fit_r2(R, bounds, beta_hat, beta, n)))

for name in ("fixed", "optimal"):
    nb, dl, st, r2 = (np.mean([a[i] for a in agg[name]]) for i in range(4))
    print(f"{name:>9} | {nb:>7.0f} | {dl:>13.1f} | {st:>11.0f} | {r2:>6.3f}")

print(f"\n({time.time()-t0:.0f}s)")
