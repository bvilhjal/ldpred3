"""Tests for LDpred2-auto inference of h2, polygenicity and predictive r2.

The headline check simulates *independent* training and test cohorts and
confirms that the r2 inferred from the training summary statistics alone
matches the PRS's actual R2 in the held-out test cohort -- the central claim
of Privé et al. (AJHG 2023).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from infer import ldpred2_auto_infer       # noqa: E402


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
    res = ldpred2_auto_infer(R, beta_hat, n_train, n_chains=8, burn_in=150,
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
    res = ldpred2_auto_infer(R, beta_hat, n_train, n_chains=8, burn_in=150,
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
    L = np.linalg.cholesky(R + 1e-4 * np.eye(m))
    beta_hat = R @ beta + (L @ rng.standard_normal(m)) / np.sqrt(N)

    res = ldpred2_auto_infer(R, beta_hat, N, n_chains=8, burn_in=150,
                             num_iter=200, seed=4)
    assert abs(res.h2_est - true_h2) < 0.12
    assert 0 < res.p_est < 0.5
    for est, ci in ((res.h2_est, res.h2_ci), (res.p_est, res.p_ci)):
        assert ci[0] <= est <= ci[1]


def test_needs_two_chains():
    rng = np.random.default_rng(0)
    R = _block_R(200, 2, rng)
    try:
        ldpred2_auto_infer(R, np.zeros(200), 20000, n_chains=1)
    except ValueError as e:
        assert "chain" in str(e)
    else:
        raise AssertionError("expected ValueError for n_chains < 2")
