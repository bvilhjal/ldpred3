"""Persistent LD memory: dense O(k²) vs banded SparseLD O(k·bandwidth).

Self-contained. At genome / sequencing scale (~10M SNPs, often thousands of SNPs
per LD block) *all* blocks are held in RAM, so persistent LD storage = Σ kᵦ² for
dense blocks — which blows up. ``compute_ld_blocks(sparse=True, max_dist=w)``
stores each block as a banded ``SparseLD`` (O(k·w)); the streaming auto sampler
fits these directly. This measures persistent storage per block size and
extrapolates to 10M SNPs.

Two caveats this prints make explicit:
  * sparse reduces *persistent* memory (the genome-wide Σ kᵦ² bottleneck), not the
    *transient* peak of densifying one block during construction (O(k²) either
    way — a windowed builder would remove that too);
  * holding all blocks in RAM is itself avoidable with on-disk streaming, which
    would cap resident memory at one block.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/ld_memory_scaling.py
"""
import sys
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks

N_REF = 3000
MAXDIST = [50, 100, 200]
BLOCK_SIZES = [1000, 2000, 4000]


def sparse_bytes(b):
    return b.data.nbytes + b.indices.nbytes + b.indptr.nbytes


print(f"Persistent LD storage per block, realistic (coalescent) LD, Nref={N_REF}\n")
print(f"{'k':>5} | {'dense':>8} | " +
      " | ".join(f"band w{w:<4}" for w in MAXDIST))
print("-" * 52)
frac = {}
for k in BLOCK_SIZES:
    G, _ = simulate_genotypes_coalescent(N_REF, k, k, seed=1)
    R = compute_ld_blocks(G, block_size=k)[0][0]
    dense = R.nbytes
    cells = []
    for w in MAXDIST:
        b = compute_ld_blocks(G, block_size=k, sparse=True, max_dist=w)[0][0]
        f = sparse_bytes(b) / dense
        frac[(k, w)] = f
        cells.append(f"{sparse_bytes(b)/1e6:>5.1f}MB ({f*100:>3.0f}%)")
    print(f"{k:>5} | {dense/1e6:>6.1f}MB | " + " | ".join(cells))

# genome-wide extrapolation at the largest tested block size
k = BLOCK_SIZES[-1]; m = 10_000_000
dense_tot = m * k * 4 / 1e9
print(f"\nGenome-wide, {m:,} SNPs in blocks of {k} (~{m//k} blocks):")
print(f"  dense:        {dense_tot:>6.0f} GB")
for w in MAXDIST:
    print(f"  band w{w:<4}:   {dense_tot*frac[(k, w)]:>6.0f} GB "
          f"({frac[(k, w)]*100:.0f}% of dense)")
print("  + on-disk streaming -> resident peak = O(one block), not the total.")
