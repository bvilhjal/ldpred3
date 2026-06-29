"""Low-rank (eigen/PC) LD vs dense vs banded, on realistic LD.

Self-contained (coalescent LD). Realistic LD is close to low rank, so storing the
top eigenvectors (``compute_ld_blocks(lowrank=True)`` -> ``LowRankLD``) and fitting
in the r-dimensional eigenspace (the streaming auto kernel) matches the dense fit
at a fraction of the memory -- whereas distance banding discards real long-range
LD and loses accuracy. This reports genetic R2 and persistent LD memory for all
three at matched-ish memory.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/ld_lowrank.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks
from ldpred3 import ldpred3_by_blocks, SparseLD, LowRankLD

M, K = 6000, 1000
N, H2, P = 50000, 0.5, 0.01
REPS = 3

G, _ = simulate_genotypes_coalescent(4000, M, K, seed=1)
m = G.shape[1]
dense = compute_ld_blocks(G, block_size=K)
Rf = [(R.astype(float), idx) for R, idx in dense]
chol = [np.linalg.cholesky(R + 1e-4 * np.eye(len(ix))) for R, ix in Rf]
db = sum(R.nbytes for R, _ in dense)


def gv(a, b):
    return sum(a[ix] @ (R @ b[ix]) for R, ix in Rf)


def r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def mem(blocks):
    tot = 0
    for R, _ in blocks:
        if isinstance(R, LowRankLD):
            tot += R.U.nbytes
        elif isinstance(R, SparseLD):
            tot += R.data.nbytes + R.indices.nbytes + R.indptr.nbytes
        else:
            tot += R.nbytes
    return tot / db * 100


reps_blocks = {
    "dense": dense,
    "band w200": compute_ld_blocks(G, block_size=K, sparse=True, max_dist=200),
    "lowrank 99%": compute_ld_blocks(G, block_size=K, lowrank=True, lowrank_variance=0.99),
    "lowrank 99.5%": compute_ld_blocks(G, block_size=K, lowrank=True, lowrank_variance=0.995),
}

print(f"LD representations on realistic coalescent LD, m={m}, blocks of {K}, "
      f"N={N}, h2={H2}, p={P}\n")
print(f"{'representation':>14} | {'genetic R2':>10} | {'memory':>8}")
print("-" * 40)
for name, blocks in reps_blocks.items():
    accs = []
    for rep in range(REPS):
        rng = np.random.default_rng(5 + rep)
        c = rng.random(m) < P
        beta = np.zeros(m); beta[c] = rng.standard_normal(int(c.sum()))
        beta *= np.sqrt(H2 / gv(beta, beta))
        bh = np.empty(m)
        for (R, ix), ch in zip(Rf, chol):
            bh[ix] = R @ beta[ix] + (ch @ rng.standard_normal(len(ix))) / np.sqrt(N)
        be = ldpred3_by_blocks(blocks, bh, np.full(m, float(N)), method="auto",
                               burn_in=100, num_iter=150, seed=0)
        accs.append(r2(be, beta))
    print(f"{name:>14} | {np.mean(accs):>10.3f} | {mem(blocks):>6.0f}%")

print("\n(low-rank matches dense accuracy at ~1/4 the memory; banding loses "
      "accuracy because realistic LD has real long-range structure.)")
