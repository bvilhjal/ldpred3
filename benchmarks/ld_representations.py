"""Running time, memory and accuracy by LD representation (realistic LD).

Self-contained (coalescent LD). Compares the LD representations the sampler can
fit -- dense, banded ``SparseLD``, and low-rank ``LowRankLD`` -- on the same
realistic blocks, reporting persistent LD memory, LD **build** time (low-rank
pays an eigendecomposition), per-fit **time**, and genetic R2.

The headline is an honest trade-off: the compact representations cut memory
(low-rank to ~1/4, lossless on realistic LD) but cost more fit time, because the
dense sampler keeps the full residual vector (O(1) reads) whereas the eigenspace
fit recomputes ``(R beta)_j = U[j].s`` in O(rank) per SNP. They are the tool for
*scale* (LD that would not fit dense), not for speeding up a problem that already
fits in RAM.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/ld_representations.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks
from ldpred3 import ldpred3_by_blocks, SparseLD, LowRankLD

M, K = 10000, 2000        # large blocks -- the genome / sequencing-scale regime
N, H2, P = 50000, 0.5, 0.01
REPS = 3


G, _ = simulate_genotypes_coalescent(4000, M, K, seed=1)
m = G.shape[1]
dense = compute_ld_blocks(G, block_size=K)
Rf = [(R.astype(float), idx) for R, idx in dense]
chol = [np.linalg.cholesky(R + 1e-4 * np.eye(len(ix))) for R, ix in Rf]


def gv(a, b):
    return sum(a[ix] @ (R @ b[ix]) for R, ix in Rf)


def r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def mem_mb(blocks):
    tot = 0
    for R, _ in blocks:
        if isinstance(R, LowRankLD):
            tot += R.U.nbytes
        elif isinstance(R, SparseLD):
            tot += R.data.nbytes + R.indices.nbytes + R.indptr.nbytes
        else:
            tot += R.nbytes
    return tot / 1e6


def build(kind):
    if kind == "lowrank 99.5%":
        return compute_ld_blocks(G, block_size=K, lowrank=True, lowrank_variance=0.995)
    if kind == "band w200":
        return compute_ld_blocks(G, block_size=K, sparse=True, max_dist=200)
    return compute_ld_blocks(G, block_size=K)


# fixed truth + sumstats across representations
rng = np.random.default_rng(5)
beta = np.zeros(m); c = rng.random(m) < P
beta[c] = rng.standard_normal(int(c.sum())); beta *= np.sqrt(H2 / gv(beta, beta))
bhs = []
for rep in range(REPS):
    r = np.random.default_rng(10 + rep); bh = np.empty(m)
    for (R, ix), ch in zip(Rf, chol):
        bh[ix] = R @ beta[ix] + (ch @ r.standard_normal(len(ix))) / np.sqrt(N)
    bhs.append(bh)
nv = np.full(m, float(N))

for kind in ("dense", "band w200", "lowrank 99.5%"):     # warm up each JIT kernel
    ldpred3_by_blocks(build(kind), bhs[0], nv, method="auto", burn_in=3, num_iter=3, seed=0)

print(f"LD representations on realistic coalescent LD, m={m}, blocks of {K}, "
      f"N={N}, h2={H2}, p={P}\n")
print(f"{'representation':>14} | {'LD MB':>6} | {'build s':>7} | {'fit s':>6} | {'R2':>6}")
print("-" * 52)
for kind in ("dense", "band w200", "lowrank 99.5%"):
    t = time.time(); blocks = build(kind); bt = time.time() - t
    fits, accs = [], []
    for bh in bhs:
        t = time.time()
        be = ldpred3_by_blocks(blocks, bh, nv, method="auto", burn_in=100,
                               num_iter=150, seed=0)
        fits.append(time.time() - t); accs.append(r2(be, beta))
    print(f"{kind:>14} | {mem_mb(blocks):>6.0f} | {bt:>7.2f} | "
          f"{np.mean(fits):>6.2f} | {np.mean(accs):>6.3f}")

print("\n(compact reps trade fit time for memory; low-rank matches dense accuracy "
      "at ~1/4 memory and is the tool for LD that would not fit dense.)")
