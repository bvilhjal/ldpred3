"""Sparse / banded LD: storage and speed vs accuracy.

Self-contained (no external LD library). Real LD is banded, so most off-diagonal
entries are ~0; ``sparsify_ld`` thresholds and/or distance-bands a dense block
into a CSR ``SparseLD`` the sampler can update in O(bandwidth). This sweeps a few
thresholding / banding settings and reports, against the dense baseline:

  * density   -- stored entries as a fraction of the dense block (memory proxy)
  * fit time  -- auto fit over all blocks (single core)
  * genetic R2 of the resulting PRS

Banding can cost positive-definiteness, which destabilises the sampler; ``shrink``
< 1 multiplies the kept off-diagonals to restore diagonal dominance. The last row
shows a tight band with shrink applied.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/sparse_ld_tradeoff.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks
from ldpred3 import ldpred3_by_blocks, sparsify_ld

NB, K = 8, 500             # 8 blocks of 500 -> m = 4000 (large blocks: banding helps)
M = NB * K
N_REF = 5000
N_GWAS = 20000
H2, P = 0.5, 0.02
REPS = 3


def build(seed):
    rng = np.random.default_rng(seed)
    G, _ = simulate_genotypes_coalescent(N_REF, M, K, seed=seed)   # realistic LD
    blocks = compute_ld_blocks(G, block_size=K)
    Rfull = [(R.astype(float), idx) for R, idx in blocks]

    def gv(a, b):
        return sum(a[ix] @ (R @ b[ix]) for R, ix in Rfull)

    causal = rng.random(M) < P
    beta = np.zeros(M); beta[causal] = rng.standard_normal(int(causal.sum()))
    beta *= np.sqrt(H2 / gv(beta, beta))
    beta_hat = np.empty(M)
    for R, ix in Rfull:
        chol = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        beta_hat[ix] = R @ beta[ix] + (chol @ rng.standard_normal(len(ix))) / np.sqrt(N_GWAS)
    return blocks, gv, beta, beta_hat


def r2(be, gv, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


# (label, threshold, max_dist, shrink); threshold/max_dist=None -> dense.
CONFIGS = [
    ("dense",               None,  None, 1.0),
    ("threshold 1e-2",      1e-2,  None, 1.0),
    ("threshold 1e-3",      1e-3,  None, 1.0),
    ("band max_dist=50",    1e-4,    50, 1.0),
    ("band 25 + shrink .9", 1e-4,    25, 0.9),
]

n = np.full(M, float(N_GWAS))
# Sparse/banded LD is a per-block path (global_hyper=False); the streaming global
# path requires dense LD. Use per-block auto throughout for an apples-to-apples
# comparison. Warm up the JIT once so the first timed config is not penalised.
_b, _gv, _beta, _bh = build(0)
ldpred3_by_blocks(_b, _bh, n, method="auto", global_hyper=False,
                  burn_in=10, num_iter=10, seed=0)
_sp = [(sparsify_ld(R, threshold=1e-2), idx) for R, idx in _b]   # warm sparse kernel
ldpred3_by_blocks(_sp, _bh, n, method="auto", global_hyper=False,
                  burn_in=10, num_iter=10, seed=0)

t0 = time.time()
print(f"Sparse/banded LD tradeoff, coalescent LD, m={M} ({NB}x{K}), Nref={N_REF}, "
      f"N_gwas={N_GWAS}, h2={H2}, p={P}, {REPS} reps\n")
print(f"{'config':>20} | {'density':>8} | {'fit s':>7} | {'R2':>6}")
print("-" * 50)

for label, thr, md, shrink in CONFIGS:
    dens, times, r2s = [], [], []
    for rep in range(REPS):
        blocks, gv, beta, beta_hat = build(100 + rep)
        if thr is None:                                   # dense baseline
            fit_blocks = blocks
            dens.append(1.0)
        else:
            sp = [(sparsify_ld(R, threshold=thr, max_dist=md, shrink=shrink), idx)
                  for R, idx in blocks]
            dens.append(sum(b.nnz for b, _ in sp) / float(M * K))
            fit_blocks = sp
        t = time.time()
        be = ldpred3_by_blocks(fit_blocks, beta_hat, n, method="auto",
                               global_hyper=False, burn_in=60, num_iter=120, seed=0)
        times.append(time.time() - t)
        r2s.append(r2(be, gv, beta))
    print(f"{label:>20} | {np.mean(dens):>7.1%} | {np.mean(times):>7.2f} | "
          f"{np.mean(r2s):>6.3f}")

print(f"\n({time.time()-t0:.0f}s)")
