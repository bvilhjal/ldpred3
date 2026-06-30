"""Generate ``ld_library.npz`` -- the realistic-LD block library the benchmarks need.

Simulates ``N_BLOCKS`` independent coalescent-with-recombination regions of
``K`` common SNPs each (msprime), and stores their ``K x K`` LD correlation
matrices as ``{"R": array of shape (N_BLOCKS, K, K)}``. This is the realistic
"population LD" the accuracy/inference scripts both simulate from and fit with
(see ``benchmarks/README.md``). Run once from the repo root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/make_ld_library.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3.simulate import simulate_genotypes_coalescent

N_BLOCKS = 100        # distinct LD blocks
K = 500               # SNPs per block
N_REF = 4000          # individuals defining the (population) LD
OUT = "ld_library.npz"

t0 = time.time()
R = np.empty((N_BLOCKS, K, K), dtype=np.float32)
for b in range(N_BLOCKS):
    G, _ = simulate_genotypes_coalescent(N_REF, K, K, seed=b)
    Gs = (G - G.mean(0)) / G.std(0)
    R[b] = ((Gs.T @ Gs) / N_REF).astype(np.float32)
    np.fill_diagonal(R[b], 1.0)
    if (b + 1) % 10 == 0:
        print(f"  {b + 1}/{N_BLOCKS} blocks  ({time.time() - t0:.0f}s)")

np.savez(OUT, R=R)
sz = os.path.getsize(OUT) / 1e6
print(f"wrote {OUT}: R{R.shape} {R.dtype}, {sz:.0f} MB  ({time.time() - t0:.0f}s)")
