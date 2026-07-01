"""Per-size worker for method_scaling.py (run in its own process for clean RSS).

Builds a realistic-LD genome of ``nsnps`` SNPs (cycled from ``ld_library.npz``)
with one sparse architecture, fits every LDpred3 method, and prints a JSON line
with each method's fit time and genetic R2 plus the process peak memory. Run as::

    python benchmarks/_method_worker.py <nsnps>
"""
import os
import sys
import json
import time
import resource

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmarks.bench_vs_bigsnpr as B          # build / pheno_r2 / constants
from ldpred3 import ldpred3_by_blocks, ldpred3_auto_annot_blocks, lassosum2


def main():
    nsnps = int(float(sys.argv[1]))
    rng = np.random.default_rng(42)
    lib_blocks, beta, bhat = B.build(nsnps, rng)   # bare (K,K) LD matrices
    K = B.K
    m = bhat.shape[0]
    n = np.full(m, float(B.N))
    # (R, idx) block list the sampler consumes (pheno_r2 keeps the bare list).
    # Force a *distinct* copy per block (the library cycles 100 blocks, and the
    # sampler de-duplicates identical dense blocks by identity) so peak memory
    # reflects a real genome's LD footprint, not the 100 unique library blocks.
    blk = [(np.array(lib_blocks[b], dtype=np.float32),
            np.arange(b * K, (b + 1) * K)) for b in range(len(lib_blocks))]
    # one uninformative binary annotation for `annot` (so it should match `auto`)
    A = (rng.random(m) < 0.2).astype(float)[:, None]

    ldpred3_by_blocks(blk, bhat, n, method="auto",             # warm the JIT
                      burn_in=5, num_iter=5)

    out = {"marginal": {"time": 0.0, "r2": B.pheno_r2(bhat, beta, lib_blocks)}}
    for meth in ("inf", "grid", "auto", "annot", "lassosum2", "laplace"):
        t0 = time.perf_counter()
        if meth == "inf":
            be = ldpred3_by_blocks(blk, bhat, n, method="inf", h2=B.H2)
        elif meth == "grid":                                   # oracle (h2, p)
            be = ldpred3_by_blocks(blk, bhat, n, method="grid", h2=B.H2,
                                   p=B.P, burn_in=B.BURN_IN, num_iter=B.NUM_ITER)
        elif meth == "auto":                                   # self-tuning (no oracle)
            be = ldpred3_by_blocks(blk, bhat, n, method="auto",
                                   burn_in=B.BURN_IN, num_iter=B.NUM_ITER, seed=1)
        elif meth == "annot":
            be = ldpred3_auto_annot_blocks(blk, bhat, n, A, burn_in=B.BURN_IN,
                                           num_iter=B.NUM_ITER, seed=1).beta_est
        elif meth == "laplace":                                # Bayesian lasso (self-tuned lambda)
            be = ldpred3_by_blocks(blk, bhat, n, method="laplace", h2=B.H2,
                                   burn_in=B.BURN_IN, num_iter=B.NUM_ITER, seed=1)
        else:                                                  # lassosum2 (L1, pseudo-val)
            be = lassosum2(blk, bhat).beta_est
        dt = time.perf_counter() - t0
        out[meth] = {"time": dt, "r2": B.pheno_r2(np.asarray(be), beta, lib_blocks)}

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem = rss / 1e9 if sys.platform == "darwin" else rss / 1e6   # bytes / KB -> GB
    print("RESULT " + json.dumps({"m": m, "methods": out, "mem_gb": mem}))


if __name__ == "__main__":
    main()
