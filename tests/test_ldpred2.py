"""Synthetic-data tests for the basic LDpred2 implementation.

The tests simulate a single LD block with a known sparse genetic architecture,
generate GWAS marginal effects from the LDpred2 model
``beta_hat = R @ beta + N(0, R / N)`` and check that each method recovers the
true joint effects substantially better than the raw marginal effects.

Run with ``pytest`` or directly with ``python tests/test_ldpred2.py``.
"""

import os
import sys

import numpy as np


import pytest  # noqa: E402

import pyldpred2.ldpred2 as ldpred2
from pyldpred2.ldpred2 import (  # noqa: E402
    block_diagonal_ld,
    ldpred2_auto,
    ldpred2_by_blocks,
    ldpred2_grid,
    ldpred2_inf,
    optimal_ld_blocks,
    sparsify_ld,
    standardize_betas,
)


def _ar1_corr(m, rho):
    """AR(1) correlation matrix: R[i, j] = rho ** |i - j|."""
    idx = np.arange(m)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def simulate(m=200, n=20000, h2=0.5, p=0.05, rho=0.6, seed=0):
    """Simulate one LD block and matching GWAS summary statistics.

    Returns ``(corr, beta_hat_std, true_beta, n)``.
    """
    rng = np.random.default_rng(seed)
    corr = _ar1_corr(m, rho)

    # True sparse standardized effects.
    is_causal = rng.random(m) < p
    n_causal = max(int(is_causal.sum()), 1)
    true_beta = np.zeros(m)
    true_beta[is_causal] = rng.normal(0.0, np.sqrt(h2 / n_causal), size=int(is_causal.sum()))

    # Marginal effects: beta_hat = R beta + noise, noise ~ N(0, R / n).
    chol = np.linalg.cholesky(corr + 1e-8 * np.eye(m))
    noise = (chol @ rng.standard_normal(m)) / np.sqrt(n)
    beta_hat = corr @ true_beta + noise
    return corr, beta_hat, true_beta, n


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_standardize_roundtrip():
    beta = np.array([0.1, -0.2, 0.05])
    se = np.array([0.02, 0.03, 0.01])
    n = 10000
    beta_std, scale = standardize_betas(beta, se, n)
    assert np.allclose(beta_std * scale, beta)
    # Standardized effect should be close to z / sqrt(n).
    z = beta / se
    assert np.allclose(beta_std, z / np.sqrt(n), rtol=0.05)


def test_inf_beats_marginal():
    corr, beta_hat, true_beta, n = simulate(seed=1)
    beta_inf = ldpred2_inf(corr, beta_hat, n, h2=0.5)
    assert _corr(beta_inf, true_beta) > _corr(beta_hat, true_beta)


def test_grid_beats_marginal():
    corr, beta_hat, true_beta, n = simulate(seed=2)
    beta_grid = ldpred2_grid(corr, beta_hat, n, h2=0.5, p=0.05,
                             burn_in=100, num_iter=300, seed=42)
    assert _corr(beta_grid, true_beta) > _corr(beta_hat, true_beta)


def test_auto_recovers_hyperparams():
    corr, beta_hat, true_beta, n = simulate(m=300, h2=0.5, p=0.05, seed=3)
    res = ldpred2_auto(corr, beta_hat, n, h2_init=0.3, p_init=0.1,
                       burn_in=200, num_iter=300, seed=7)
    assert _corr(res.beta_est, true_beta) > _corr(beta_hat, true_beta)
    # Hyper-parameter estimates should land in a sensible ballpark.
    assert 0.2 < res.h2_est < 0.9
    assert 0.005 < res.p_est < 0.4


def test_numba_and_python_paths_agree():
    """The JIT-compiled and pure-Python grid samplers must be identical.

    The -grid path uses only ``random`` / ``standard_normal`` draws, whose
    streams match between numba and NumPy's legacy RNG, so results should be
    bit-for-bit identical. Skipped when numba is not installed.
    """
    if not ldpred2.HAVE_NUMBA:
        import pytest
        pytest.skip("numba not installed")

    corr, beta_hat, true_beta, n = simulate(m=200, seed=5)
    n_vec = np.full(corr.shape[0], float(n))
    init = np.zeros(corr.shape[0])
    kwargs = dict(burn_in=40, num_iter=120, sparse=False, estimate_hyper=False,
                  h2_min=1e-6, h2_max=1.0, seed=11, init_beta=init, tol=0.0,
                  check_every=50, allow_jump_sign=True,
                  prior_w=np.ones(corr.shape[0]))
    corr_c = np.ascontiguousarray(corr)
    py = ldpred2._gibbs_kernel(corr_c, beta_hat, n_vec, 0.5, 0.05, **kwargs)
    jit = ldpred2._gibbs_kernel_jit(corr_c, beta_hat, n_vec, 0.5, 0.05, **kwargs)
    assert np.allclose(py[0], jit[0], atol=1e-10)


def test_sparse_matches_dense_when_full():
    """A SparseLD that keeps every entry must reproduce the dense sampler exactly."""
    corr, beta_hat, true_beta, n = simulate(m=200, seed=6)
    n_vec = np.full(corr.shape[0], float(n))
    beta_std, _ = standardize_betas(beta_hat, np.full(corr.shape[0], 0.01), n_vec)
    dense = ldpred2_grid(corr, beta_std, n_vec, h2=0.5, p=0.05,
                         burn_in=40, num_iter=120, seed=3)
    full = sparsify_ld(corr, threshold=0.0)        # keep all entries
    sp = ldpred2_grid(full, beta_std, n_vec, h2=0.5, p=0.05,
                      burn_in=40, num_iter=120, seed=3)
    assert np.allclose(dense, sp, atol=1e-6)


def test_sparse_inf_matches_dense_when_full():
    """Sparse (CG) inf with all entries must match the dense direct solve."""
    corr, beta_hat, true_beta, n = simulate(m=150, seed=7)
    n_vec = np.full(corr.shape[0], float(n))
    dense = ldpred2_inf(corr, beta_hat, n_vec, h2=0.5)
    sp = ldpred2_inf(sparsify_ld(corr, threshold=0.0), beta_hat, n_vec, h2=0.5)
    assert np.allclose(dense, sp, atol=1e-5)


def test_banded_sparse_recovers_signal():
    """A thresholded/banded SparseLD still beats the raw marginal betas."""
    corr, beta_hat, true_beta, n = simulate(m=300, seed=8)
    n_vec = np.full(corr.shape[0], float(n))
    sp = sparsify_ld(corr, threshold=1e-2)
    beta_grid = ldpred2_grid(sp, beta_hat, n_vec, h2=0.5, p=0.05,
                             burn_in=60, num_iter=200, seed=1)
    assert _corr(beta_grid, true_beta) > _corr(beta_hat, true_beta)


def test_warm_start_recovers_signal():
    """Warm-starting from inf should still recover the signal (and converge)."""
    corr, beta_hat, true_beta, n = simulate(m=300, seed=9)
    n_vec = np.full(corr.shape[0], float(n))
    cold = ldpred2_grid(corr, beta_hat, n_vec, h2=0.5, p=0.05,
                        burn_in=100, num_iter=300, seed=1)
    warm = ldpred2_grid(corr, beta_hat, n_vec, h2=0.5, p=0.05,
                        burn_in=100, num_iter=300, warm_start=True, seed=1)
    base = _corr(beta_hat, true_beta)
    assert _corr(cold, true_beta) > base
    assert _corr(warm, true_beta) > base
    # Cold and warm start estimate the same posterior mean (well correlated).
    assert _corr(cold, warm) > 0.9


def test_adaptive_stopping_stops_early():
    """Adaptive stopping should use fewer iterations yet recover the signal."""
    corr, beta_hat, true_beta, n = simulate(m=300, seed=10)
    n_vec = np.full(corr.shape[0], float(n))
    res = ldpred2_auto(corr, beta_hat, n_vec, h2_init=0.3, p_init=0.1,
                       burn_in=50, num_iter=2000, warm_start=True,
                       tol=1e-2, check_every=50, seed=1)
    assert res.n_iter < 2000                       # stopped before the cap
    assert _corr(res.beta_est, true_beta) > _corr(beta_hat, true_beta)


def _make_blocks(nb=4, k=150, rho=0.5, p=0.05, N=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = nb * k
    blk = _ar1_corr(k, rho)
    chol = np.linalg.cholesky(blk + 1e-9 * np.eye(k))
    beta = np.zeros(m)
    causal = rng.random(m) < p
    beta[causal] = rng.normal(0, np.sqrt(0.5 / max(int(causal.sum()), 1)),
                              int(causal.sum()))
    bhat = np.empty(m)
    blocks = []
    for b in range(nb):
        s = slice(b * k, (b + 1) * k)
        bhat[s] = blk @ beta[s] + (chol @ rng.standard_normal(k)) / np.sqrt(N)
        blocks.append((blk, np.arange(b * k, (b + 1) * k)))
    n = np.full(m, float(N))
    return blocks, bhat, beta, n


def test_block_diagonal_inf_matches_per_block():
    """inf on the assembled block-diagonal matrix == per-block inf (independence).

    The infinitesimal ridge is ``m / (h2 * N)``: a single block of size k with
    ``h2 = H*k/m`` and the whole matrix (size m) with ``h2 = H`` give the same
    ridge, so the block-diagonal solve must reproduce the per-block solves.
    """
    blocks, bhat, true_beta, n = _make_blocks(seed=1)
    m = len(bhat)
    per_block = np.concatenate([ldpred2_inf(c, bhat[idx], n[idx], h2=0.5 * len(idx) / m)
                                for c, idx in blocks])
    glob = ldpred2_inf(block_diagonal_ld(blocks), bhat, n, h2=0.5)
    assert np.allclose(per_block, glob, atol=1e-4)


def test_global_auto_recovers_signal():
    """LDpred2-auto with global hyper-parameters recovers the signal."""
    blocks, bhat, true_beta, n = _make_blocks(nb=5, seed=2)
    beta = ldpred2_by_blocks(blocks, bhat, n, method="auto",
                             burn_in=50, num_iter=150, seed=1, global_hyper=True)
    assert _corr(beta, true_beta) > _corr(bhat, true_beta)


def test_parallel_auto_recovers_signal():
    """Multicore (ncores>1) global auto runs and recovers the signal."""
    if not ldpred2.HAVE_NUMBA:
        import pytest
        pytest.skip("numba not installed")
    blocks, bhat, true_beta, n = _make_blocks(nb=6, seed=4)
    beta = ldpred2_by_blocks(blocks, bhat, n, method="auto", burn_in=40,
                             num_iter=120, seed=1, ncores=2)
    assert _corr(beta, true_beta) > _corr(bhat, true_beta)


def _discarded_ld2(R, blocks, window):
    blk = np.empty(R.shape[0], int)
    for bi, (a, b) in enumerate(blocks):
        blk[a:b] = bi
    s = 0.0
    m = R.shape[0]
    for i in range(m):
        for j in range(i + 1, min(i + window + 1, m)):
            if blk[i] != blk[j]:
                s += R[i, j] ** 2
    return s


def test_optimal_ld_blocks():
    """Optimal LD splitting finds the natural boundary and beats fixed blocks."""
    k1, k2 = 100, 150
    R = np.zeros((k1 + k2, k1 + k2))
    R[:k1, :k1] = _ar1_corr(k1, 0.6)
    R[k1:, k1:] = _ar1_corr(k2, 0.6)
    np.fill_diagonal(R, 1.0)
    m = k1 + k2

    blocks, cost = optimal_ld_blocks(R, max_size=200, min_size=20, window=50)
    # boundary placed exactly at the LD gap -> ~zero discarded LD
    assert cost < 1e-8
    assert (0, k1) in blocks and (k1, m) in blocks
    # valid tiling within the size bounds
    assert blocks[0][0] == 0 and blocks[-1][1] == m
    assert all(20 <= b - a <= 200 for a, b in blocks)
    # optimal never discards more LD than fixed blocks of the same max size
    fixed = [(i, min(i + 120, m)) for i in range(0, m, 120)]
    assert cost <= _discarded_ld2(R, fixed, 50) + 1e-9


# --------------------------------------------------------------------------- #
# Input validation / robustness
# --------------------------------------------------------------------------- #
def test_standardize_betas_validation_and_zero_guard():
    # n_eff must be positive; beta_se non-negative
    with pytest.raises(ValueError):
        standardize_betas(np.array([0.1]), np.array([0.01]), np.array([0.0]))
    with pytest.raises(ValueError):
        standardize_betas(np.array([0.1]), np.array([-0.01]), np.array([1000.0]))
    # beta == 0 and beta_se == 0 -> 0, not NaN
    bstd, scale = standardize_betas(np.array([0.0, 0.1]),
                                    np.array([0.0, 0.02]),
                                    np.array([1000.0, 1000.0]))
    assert np.all(np.isfinite(bstd)) and bstd[0] == 0.0


def test_hyperparameter_validation():
    corr, beta_hat, _, n = simulate(m=50, seed=1)
    nv = np.full(50, float(n))
    with pytest.raises(ValueError):
        ldpred2_inf(corr, beta_hat, nv, h2=0.0)
    with pytest.raises(ValueError):
        ldpred2_grid(corr, beta_hat, nv, h2=-1.0, p=0.05)
    with pytest.raises(ValueError):
        ldpred2_grid(corr, beta_hat, nv, h2=0.5, p=1.5)
    with pytest.raises(ValueError):
        ldpred2_auto(corr, beta_hat, nv, p_init=0.0)
    with pytest.raises(ValueError):
        ldpred2_auto(corr, beta_hat, nv, h2_init=-0.1)
    with pytest.raises(ValueError):
        ldpred2_auto(corr, beta_hat, nv, h2_bounds=(0.5, 0.1))
    with pytest.raises(ValueError):       # non-positive n_eff
        ldpred2_inf(corr, beta_hat, np.zeros(50), h2=0.5)


def test_global_auto_rejects_bad_blocks():
    blocks, bhat, _, n = _make_blocks(nb=3, seed=1)
    # non-contiguous: drop the middle block -> indices no longer tile 0..m-1
    bad = [blocks[0], blocks[2]]
    with pytest.raises(ValueError):
        ldpred2_by_blocks(bad, bhat, n, method="auto", global_hyper=True,
                          burn_in=5, num_iter=5)
    # sparsify with global auto is not supported
    with pytest.raises(NotImplementedError):
        ldpred2_by_blocks(blocks, bhat, n, method="auto", global_hyper=True,
                          sparsify=True, burn_in=5, num_iter=5)


def test_inf_zero_beta_is_finite():
    corr, _, _, n = simulate(m=80, seed=2)
    nv = np.full(80, float(n))
    zero = np.zeros(80)
    dense = ldpred2_inf(corr, zero, nv, h2=0.5)
    sparse = ldpred2_inf(sparsify_ld(corr, threshold=0.0), zero, nv, h2=0.5)
    assert np.allclose(dense, 0.0) and np.allclose(sparse, 0.0)


def test_extreme_params_no_overflow():
    # huge effects + tiny p drive log_odds large; the stable sigmoid must not
    # produce NaN/inf in the posterior-mean estimate.
    corr, beta_hat, _, n = simulate(m=60, seed=3)
    nv = np.full(60, 1e7)
    out = ldpred2_grid(corr, beta_hat * 50, nv, h2=0.9, p=1e-6,
                       burn_in=10, num_iter=20, seed=1)
    assert np.all(np.isfinite(out))


if __name__ == "__main__":
    corr, beta_hat, true_beta, n = simulate(seed=123)
    base = _corr(beta_hat, true_beta)
    inf = _corr(ldpred2_inf(corr, beta_hat, n, h2=0.5), true_beta)
    grid = _corr(
        ldpred2_grid(corr, beta_hat, n, h2=0.5, p=0.05, seed=1), true_beta
    )
    res = ldpred2_auto(corr, beta_hat, n, seed=1)
    auto = _corr(res.beta_est, true_beta)

    print("Correlation with true effects (higher is better):")
    print(f"  marginal beta_hat : {base:.3f}")
    print(f"  LDpred2-inf       : {inf:.3f}")
    print(f"  LDpred2-grid      : {grid:.3f}")
    print(f"  LDpred2-auto      : {auto:.3f}")
    print(f"LDpred2-auto estimates: h2={res.h2_est:.3f}, p={res.p_est:.4f}")
