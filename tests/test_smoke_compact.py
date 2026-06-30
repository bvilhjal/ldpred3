"""Fast smoke test for the compact-LD streaming sweeps.

The no-Numba CI leg skips the slow ``test_infer`` / ``test_pipeline`` files, which
would otherwise leave the **pure-Python** low-rank / banded sweeps
(``_gibbs_one_sweep_lowrank`` / ``_sparse`` via the streaming samplers)
unexercised without the JIT. These tiny runs (m=20, a few iterations) keep both
the scoring (``ldpred3_by_blocks``) and inference (``ldpred3_auto_infer``) paths
covered on that leg at negligible cost.
"""

import numpy as np

from ldpred3 import (ldpred3_by_blocks, ldpred3_auto_infer,
                     sparsify_ld, lowrank_ld)


def _blocks(kind):
    R = (0.4 ** np.abs(np.subtract.outer(np.arange(10), np.arange(10)))).astype(float)
    conv = {"dense": lambda M: M.astype(np.float32),
            "sparse": sparsify_ld, "lowrank": lowrank_ld}[kind]
    return [(conv(R), np.arange(10)), (conv(R), np.arange(10, 20))]


_BH = (np.random.default_rng(0).standard_normal(20) * 0.05)


def test_smoke_byblocks_compact():
    # Streaming global-hyper auto scoring on banded and low-rank blocks.
    for kind in ("dense", "sparse", "lowrank"):
        out = ldpred3_by_blocks(_blocks(kind), _BH, 5000, method="auto",
                                burn_in=5, num_iter=10, seed=0)
        assert out.shape == (20,) and np.all(np.isfinite(out))


def test_smoke_infer_compact():
    # Streaming h2/p/r2 inference on low-rank and banded blocks.
    for kind in ("lowrank", "sparse"):
        res = ldpred3_auto_infer(_blocks(kind), _BH, 5000, n_chains=2,
                                 burn_in=5, num_iter=10, seed=0)
        assert np.isfinite(res.h2_est) and np.isfinite(res.p_est)
