"""Bayesian-lasso (Laplace-prior) posterior-mean sampler."""

import numpy as np

from ldpred3 import ldpred3_laplace, ldpred3_by_blocks, lassosum2, sparsify_ld


def _sim(m=500, nblk=5, rho=0.6, p=0.05, h2=0.5, N=10000, seed=0):
    rng = np.random.default_rng(seed)
    k = m // nblk
    blocks, R_full = [], np.zeros((m, m))
    i = np.arange(k)
    Rb = rho ** np.abs(i[:, None] - i[None, :])
    for b in range(nblk):
        blocks.append((Rb.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        R_full[b * k:(b + 1) * k, b * k:(b + 1) * k] = Rb
    causal = rng.random(m) < p
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
    L = np.linalg.cholesky(R_full + 1e-6 * np.eye(m))
    beta_hat = R_full @ beta + (L @ rng.standard_normal(m)) / np.sqrt(N)
    return blocks, R_full, beta, beta_hat


def _genetic_corr(a, b, R):
    num = a @ (R @ b)
    da = a @ (R @ a)
    db = b @ (R @ b)
    return num / np.sqrt(da * db) if da > 0 and db > 0 else 0.0


def _fit_blocks(blocks, beta_hat, N, **kw):
    m = sum(len(idx) for _, idx in blocks)
    n = np.full(m, float(N))
    out = np.zeros(m)
    for R, idx in blocks:
        out[idx] = ldpred3_laplace(R, beta_hat[idx], n[idx],
                                   h2=0.5 * len(idx) / m, **kw)
    return out


def test_laplace_recovers_signal_and_is_dense():
    blocks, R, beta, beta_hat = _sim(seed=1)
    be = _fit_blocks(blocks, beta_hat, 10000, burn_in=100, num_iter=300, seed=1)
    gc = _genetic_corr(be, beta, R)
    assert gc > 0.5, f"laplace genetic corr with truth too low: {gc:.3f}"
    assert np.all(np.isfinite(be))
    # the posterior mean of a Laplace prior is dense (no point mass at zero)
    assert np.mean(np.abs(be) > 1e-10) > 0.9


def test_laplace_reproducible():
    blocks, R, beta, beta_hat = _sim(seed=2)
    a = _fit_blocks(blocks, beta_hat, 10000, burn_in=40, num_iter=120, seed=7)
    b = _fit_blocks(blocks, beta_hat, 10000, burn_in=40, num_iter=120, seed=7)
    np.testing.assert_array_equal(a, b)


def test_laplace_similar_to_lassosum2():
    # The lasso is the MAP under a Laplace prior; the Bayesian posterior mean
    # should predict similarly. They need not be identical, but should track.
    blocks, R, beta, beta_hat = _sim(seed=3)
    be_lap = _fit_blocks(blocks, beta_hat, 10000, burn_in=100, num_iter=300, seed=1)
    be_las = lassosum2(blocks, beta_hat).beta_est
    r_lap = _genetic_corr(be_lap, beta, R)
    r_las = _genetic_corr(be_las, beta, R)
    # both are good, and the two Laplace-prior estimators land close together
    assert r_lap > 0.5 and r_las > 0.5
    assert abs(r_lap - r_las) < 0.1, f"laplace {r_lap:.3f} vs lassosum2 {r_las:.3f}"
    assert _genetic_corr(be_lap, be_las, R) > 0.9    # predictions highly aligned


def test_laplace_by_blocks_and_guards():
    blocks, R, beta, beta_hat = _sim(seed=4)
    n = np.full(beta_hat.shape[0], 10000.0)
    be = ldpred3_by_blocks(blocks, beta_hat, n, method="laplace", h2=0.5,
                           burn_in=50, num_iter=150, seed=1)
    assert _genetic_corr(be, beta, R) > 0.5
    # dense-only guard
    R0, idx0 = blocks[0]
    sp = [(sparsify_ld(R0, threshold=0.01), idx0)]
    import pytest
    with pytest.raises(ValueError, match="dense LD"):
        ldpred3_by_blocks(sp, beta_hat[idx0], n[idx0], method="laplace")


def test_laplace_empty_block():
    out = ldpred3_laplace(np.zeros((0, 0)), np.zeros(0), 1000.0)
    assert out.shape == (0,)
