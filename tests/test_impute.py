"""Tests for LD-based summary-statistic imputation (ldpred3/impute.py)."""
import numpy as np

from ldpred3 import impute_sumstats_blocks, ImputeResult


def ar1(m, rho):
    return (rho ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))).astype(float)


def _marginal(R, beta, n, seed):
    chol = np.linalg.cholesky(R + 1e-9 * np.eye(R.shape[0]))
    rng = np.random.default_rng(seed)
    return R @ beta + (chol @ rng.standard_normal(R.shape[0])) / np.sqrt(n)


def test_imputes_held_out_marginal():
    m, N = 40, 50000
    R = ar1(m, 0.7)
    beta = np.zeros(m); beta[15] = 0.3
    bhat = _marginal(R, beta, N, 0)
    typed = np.ones(m, bool)
    untyped = np.array([14, 16, 25])
    typed[untyped] = False
    obs = bhat.copy(); obs[untyped] = 0.0
    res = impute_sumstats_blocks(obs, [(R.astype(np.float32), np.arange(m))], typed, N)
    assert isinstance(res, ImputeResult)
    # imputed marginal is close to the true held-out marginal
    np.testing.assert_allclose(res.beta_hat[untyped], bhat[untyped], atol=0.02)
    # typed entries and their N are untouched
    np.testing.assert_array_equal(res.beta_hat[typed], obs[typed])
    assert np.all(res.n_eff[typed] == N)


def test_imp_r2_bounds_and_effective_n():
    m, N = 30, 20000
    R = ar1(m, 0.6)
    typed = np.ones(m, bool); typed[[10, 20]] = False
    res = impute_sumstats_blocks(np.zeros(m), [(R.astype(np.float32), np.arange(m))],
                                 typed, N)
    assert np.all((res.imp_r2 >= 0) & (res.imp_r2 <= 1.0 + 1e-9))
    assert np.all(res.imp_r2[typed] == 1.0)
    # untyped effective N is N * imp_r2
    np.testing.assert_allclose(res.n_eff[~typed], N * res.imp_r2[~typed], rtol=1e-6)


def test_perfect_ld_imputes_exactly():
    # variant 1 is a perfect duplicate of variant 0 -> imp_r2 ~ 1, imputed = beta_0.
    R = np.array([[1.0, 1.0, 0.4],
                  [1.0, 1.0, 0.4],
                  [0.4, 0.4, 1.0]])
    bhat = np.array([0.12, 0.12, 0.05])
    typed = np.array([True, False, True])
    res = impute_sumstats_blocks(bhat, [(R.astype(np.float32), np.arange(3))],
                                 typed, 10000.0)
    assert res.imp_r2[1] > 0.99
    assert abs(res.beta_hat[1] - bhat[0]) < 1e-2


def test_no_typed_in_block_excluded():
    R = ar1(10, 0.5)
    typed = np.zeros(10, bool)            # nothing observed -> nothing to impute from
    res = impute_sumstats_blocks(np.zeros(10), [(R.astype(np.float32), np.arange(10))],
                                 typed, 1000.0)
    assert np.all(res.imp_r2 == 0.0)
    # excluded variants get a negligible positive n_eff (beta=0) so the result
    # feeds the sampler directly (which requires n_eff > 0)
    assert np.all(res.n_eff == 1.0)
    assert np.all(res.beta_hat == 0.0)


def test_min_imp_r2_excludes_poorly_tagged():
    m = 60
    R = ar1(m, 0.3)                       # weak LD -> far variants poorly tagged
    typed = np.ones(m, bool); typed[5:55:7] = False
    res = impute_sumstats_blocks(np.zeros(m), [(R.astype(np.float32), np.arange(m))],
                                 typed, 1000.0, min_imp_r2=0.5)
    poorly = (~typed) & (res.imp_r2 == 0.0)
    assert poorly.sum() > 0               # some untyped fall below the threshold
    # excluded -> beta 0 and a negligible n_eff=1 (positive, so it's samplable)
    assert np.all(res.n_eff[poorly] == 1.0)
    assert np.all(res.beta_hat[poorly] == 0.0)


def test_two_blocks_independent():
    R = ar1(20, 0.6)
    blocks = [(R.astype(np.float32), np.arange(20)),
              (R.astype(np.float32), np.arange(20, 40))]
    beta = np.zeros(40); beta[5] = 0.25; beta[27] = 0.25
    bhat = np.concatenate([_marginal(R, beta[:20], 50000, 1),
                           _marginal(R, beta[20:], 50000, 2)])
    typed = np.ones(40, bool); typed[[6, 28]] = False
    res = impute_sumstats_blocks(bhat, blocks, typed, 50000)
    assert res.n_imputed == 2
    assert res.imp_r2[6] > 0 and res.imp_r2[28] > 0
