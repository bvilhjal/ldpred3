"""Tests for per-variant priors (SBayesRC-style annotation-informed prior)."""


import numpy as np

from ldpred3 import ldpred3_grid, ldpred3_auto
from ldpred3.prs import standardize_dosage


def _ar1(m, rho):
    i = np.arange(m)
    return rho ** np.abs(i[:, None] - i[None, :])


def test_equal_weights_match_default():
    rng = np.random.default_rng(0)
    m = 200
    R = _ar1(m, 0.5)
    beta = np.zeros(m); beta[::15] = 0.3
    bhat = R @ beta + rng.standard_normal(m) / np.sqrt(5000)
    a = ldpred3_grid(R, bhat, 5000, h2=0.5, p=0.05, burn_in=40, num_iter=120,
                     seed=1)
    b = ldpred3_grid(R, bhat, 5000, h2=0.5, p=0.05, burn_in=40, num_iter=120,
                     seed=1, prior_weights=np.ones(m))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    # same for auto
    ra = ldpred3_auto(R, bhat, 5000, burn_in=40, num_iter=120, seed=1)
    rb = ldpred3_auto(R, bhat, 5000, burn_in=40, num_iter=120, seed=1,
                      prior_weights=np.ones(m))
    np.testing.assert_allclose(ra.beta_est, rb.beta_est, rtol=1e-10, atol=1e-12)


def test_prior_weights_validation():
    R = _ar1(50, 0.5); bhat = np.zeros(50)
    for bad in (np.ones(49), -np.ones(50)):
        try:
            ldpred3_grid(R, bhat, 5000, h2=0.5, p=0.05, prior_weights=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for bad prior_weights")


def _geno(n, m, rho, rng):
    def hap():
        z = np.zeros((n, m)); z[:, 0] = rng.standard_normal(n)
        s = np.sqrt(1 - rho ** 2)
        for j in range(1, m):
            z[:, j] = rho * z[:, j - 1] + s * rng.standard_normal(n)
        return z
    thr = rng.uniform(-1, 1, m)
    return (hap() > thr).astype(float) + (hap() > thr).astype(float)


def _one(seed, N=2000, m=400, h2=0.5, p=0.05, enrich=15.0):
    rng = np.random.default_rng(seed)
    GA, GB = _geno(N, m, 0.6, rng), _geno(3000, m, 0.6, rng)
    ZA, ZB = standardize_dosage(GA), standardize_dosage(GB)
    R = (ZA.T @ ZA) / N; np.fill_diagonal(R, 1.0)
    func = rng.random(m) < 0.2
    pr = np.clip(np.where(func, enrich, 1.0) / np.where(func, enrich, 1.0).sum()
                * (p * m), 0, 1)
    causal = rng.random(m) < pr
    if not causal.any():
        causal[rng.integers(m)] = True
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
    gA = ZA @ beta
    y = gA + rng.normal(0, np.sqrt(max(1e-6, 1 - gA.var())), N)
    bhat = (ZA.T @ y) / N
    yte = ZB @ beta + rng.normal(0, np.sqrt(1 - h2), 3000)
    w = np.where(func, enrich, 1.0)
    b_plain = ldpred3_grid(R, bhat, N, h2=h2, p=p, burn_in=80, num_iter=200,
                           seed=1)
    b_annot = ldpred3_grid(R, bhat, N, h2=h2, p=p, burn_in=80, num_iter=200,
                           seed=1, prior_weights=w)
    r2 = lambda b: np.corrcoef(ZB @ b, yte)[0, 1] ** 2
    return r2(b_plain), r2(b_annot)


def test_informative_priors_help_held_out():
    # Averaged over seeds, informative per-variant priors raise held-out R2.
    rp = ra = 0.0
    for s in range(4):
        a, b = _one(s)
        rp += a; ra += b
    assert ra / 4 > rp / 4, (rp / 4, ra / 4)
