"""Tests for LDpred3-auto inference of h2, polygenicity and predictive r2.

The headline check simulates *independent* training and test cohorts and
confirms that the r2 inferred from the training summary statistics alone
matches the PRS's actual R2 in the held-out test cohort -- the central claim
of Privé et al. (AJHG 2023).
"""

import os
import sys

import numpy as np


from ldpred3.infer import ldpred3_auto_infer       # noqa: E402


def _block_R(m, nblk, rng):
    k = m // nblk
    R = np.zeros((m, m))
    for b in range(nblk):
        i = np.arange(k)
        rho = 0.4 + 0.4 * rng.random()
        R[b * k:(b + 1) * k, b * k:(b + 1) * k] = rho ** np.abs(i[:, None] - i[None, :])
    return R


def _simulate_cohorts(m=600, nblk=6, h2=0.5, p=0.05, n_train=8000,
                      n_test=4000, seed=0):
    """Standardized genotypes ~ MVN(0, R) for two independent cohorts."""
    rng = np.random.default_rng(seed)
    R = _block_R(m, nblk, rng)
    L = np.linalg.cholesky(R + 1e-6 * np.eye(m))

    causal = rng.random(m) < p
    if not causal.any():
        causal[0] = True
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())

    def cohort(n):
        G = (L @ rng.standard_normal((m, n))).T        # (n, m), columns ~ unit var
        G /= G.std(0)
        g = G @ beta
        g *= np.sqrt(h2) / g.std()                     # set genetic variance = h2
        y = g + rng.normal(0, np.sqrt(1 - h2), n)
        return G, y

    Gtr, ytr = cohort(n_train)
    Gte, yte = cohort(n_test)
    beta_hat = (Gtr.T @ ytr) / n_train                 # standardized marginal
    return R, beta_hat, n_train, Gte, yte


def test_inferred_r2_matches_held_out_R2():
    R, beta_hat, n_train, Gte, yte = _simulate_cohorts(
        h2=0.5, p=0.05, n_train=8000, n_test=4000, seed=1)
    res = ldpred3_auto_infer(R, beta_hat, n_train, n_chains=8, burn_in=150,
                             num_iter=200, sample_every=5, seed=3)

    prs = Gte @ res.beta_est
    held_out_r2 = np.corrcoef(prs, yte)[0, 1] ** 2

    assert abs(res.r2_est - held_out_r2) < 0.07, (res.r2_est, held_out_r2)
    assert res.r2_ci[0] <= res.r2_est <= res.r2_ci[1]
    # Predictive r2 cannot exceed heritability.
    assert res.r2_est <= res.h2_est + 0.08
    assert res.n_chains_kept >= 2


def test_low_power_gives_low_r2():
    # Low h2 + highly polygenic + small N -> genuinely low power. The inferred
    # r2 should be small and still track the held-out R2.
    R, beta_hat, n_train, Gte, yte = _simulate_cohorts(
        h2=0.2, p=0.2, n_train=1000, n_test=4000, seed=2)
    res = ldpred3_auto_infer(R, beta_hat, n_train, n_chains=8, burn_in=150,
                             num_iter=200, seed=3)
    prs = Gte @ res.beta_est
    held_out_r2 = np.corrcoef(prs, yte)[0, 1] ** 2
    assert held_out_r2 < 0.2
    assert res.r2_est < 0.2
    assert abs(res.r2_est - held_out_r2) < 0.12


def test_infer_recovers_h2_and_p():
    rng = np.random.default_rng(5)
    m, nblk, h2, p, N = 600, 6, 0.5, 0.05, 30000
    R = _block_R(m, nblk, rng)
    causal = rng.random(m) < p
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
    true_h2 = float(beta @ (R @ beta))
    true_p = causal.mean()
    L = np.linalg.cholesky(R + 1e-4 * np.eye(m))
    beta_hat = R @ beta + (L @ rng.standard_normal(m)) / np.sqrt(N)

    res = ldpred3_auto_infer(R, beta_hat, N, n_chains=8, burn_in=150,
                             num_iter=200, seed=4)
    assert abs(res.h2_est - true_h2) < 0.12
    # Polygenicity is recovered to within ~50% and its 95% CI covers the truth.
    assert abs(res.p_est - true_p) < 0.5 * true_p + 0.01
    assert res.p_ci[0] <= true_p <= res.p_ci[1]
    for est, ci in ((res.h2_est, res.h2_ci), (res.p_est, res.p_ci)):
        assert ci[0] <= est <= ci[1]


def test_polygenicity_tracks_truth_across_scales():
    # p_est tracks the true causal fraction across a 4x range (p >= 0.01).
    for true_p in (0.05, 0.2):
        rng = np.random.default_rng(11)
        m, nblk, h2, N = 1000, 8, 0.5, 50000
        R = _block_R(m, nblk, rng)
        causal = rng.random(m) < true_p
        beta = np.zeros(m)
        beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
        L = np.linalg.cholesky(R + 1e-4 * np.eye(m))
        beta_hat = R @ beta + (L @ rng.standard_normal(m)) / np.sqrt(N)
        res = ldpred3_auto_infer(R, beta_hat, N, n_chains=10, burn_in=200,
                                 num_iter=250, seed=7)
        assert 0.6 * true_p < res.p_est < 1.6 * true_p, (true_p, res.p_est)


def test_parallel_chains_match_serial(tmp_path):
    # ncores>1 runs chains in parallel processes; results are deterministic
    # (seeded per chain) and must match the serial run.
    rng = np.random.default_rng(3)
    R = _block_R(400, 4, rng)
    beta = np.zeros(400); c = rng.random(400) < 0.05
    beta[c] = rng.normal(0, np.sqrt(0.5 / c.sum()), c.sum())
    L = np.linalg.cholesky(R + 1e-4 * np.eye(400))
    bhat = R @ beta + (L @ rng.standard_normal(400)) / np.sqrt(20000)
    a = ldpred3_auto_infer(R, bhat, 20000, n_chains=6, burn_in=80,
                           num_iter=100, seed=1, ncores=1)
    b = ldpred3_auto_infer(R, bhat, 20000, n_chains=6, burn_in=80,
                           num_iter=100, seed=1, ncores=2)
    assert abs(a.h2_est - b.h2_est) < 1e-9
    assert abs(a.r2_est - b.r2_est) < 1e-9


def test_allow_jump_sign_stabilises():
    # On near-singular LD with a fixed (over-large) h2, the sampler can diverge;
    # forbidding within-step sign flips keeps the effects bounded.
    from ldpred3 import ldpred3_grid
    rng = np.random.default_rng(0)
    m = 150
    # Strong, near-collinear LD block (poorly conditioned).
    R = 0.95 ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    beta = np.zeros(m); beta[::25] = 0.4
    bhat = R @ beta + rng.standard_normal(m) / np.sqrt(2000)

    free = ldpred3_grid(R, bhat, 2000, h2=0.9, p=0.05, burn_in=50, num_iter=150,
                        seed=1, allow_jump_sign=True)
    guarded = ldpred3_grid(R, bhat, 2000, h2=0.9, p=0.05, burn_in=50,
                           num_iter=150, seed=1, allow_jump_sign=False)
    # The guarded run stays finite and no larger than the unguarded one.
    assert np.all(np.isfinite(guarded))
    assert np.abs(guarded).max() <= np.abs(free).max() + 1e-6


def test_needs_two_chains():
    rng = np.random.default_rng(0)
    R = _block_R(200, 2, rng)
    try:
        ldpred3_auto_infer(R, np.zeros(200), 20000, n_chains=1)
    except ValueError as e:
        assert "chain" in str(e)
    else:
        raise AssertionError("expected ValueError for n_chains < 2")


def _split_blocks(R, nblk):
    """Split a block-diagonal dense R into a [(R_block, idx), ...] list."""
    m = R.shape[0]; k = m // nblk
    return [(R[b * k:(b + 1) * k, b * k:(b + 1) * k].astype(np.float32),
             np.arange(b * k, (b + 1) * k)) for b in range(nblk)]


def test_streaming_blocks_matches_dense():
    # Inference on a list of per-block (R, idx) (streamed, no dense genome-wide
    # LD) should agree with the dense path and recover h2.
    R, beta_hat, n_train, Gte, yte = _simulate_cohorts(
        h2=0.5, p=0.05, n_train=8000, seed=2)
    blocks = _split_blocks(R, 6)
    dense = ldpred3_auto_infer(R, beta_hat, n_train, n_chains=8, burn_in=120,
                               num_iter=160, seed=5)
    strm = ldpred3_auto_infer(blocks, beta_hat, n_train, n_chains=8, burn_in=120,
                              num_iter=160, seed=5)
    assert abs(dense.h2_est - strm.h2_est) < 0.07
    assert abs(dense.p_est - strm.p_est) < 0.05
    assert abs(dense.r2_est - strm.r2_est) < 0.07
    assert abs(strm.h2_est - 0.5) < 0.12              # recovers true h2

    # the inferred r2 still tracks the held-out R2 via the streaming path
    pred = Gte @ strm.beta_est
    r2_test = np.corrcoef(pred, yte)[0, 1] ** 2
    assert abs(strm.r2_est - r2_test) < 0.12
