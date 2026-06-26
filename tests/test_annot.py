"""Tests for learning the annotation->prior map inside the sampler (SBayesRC)."""

import numpy as np
import pytest

from pyldpred2 import ldpred2_auto_annot
from pyldpred2.prs import standardize_dosage


def _geno(n, m, rho, rng):
    def hap():
        z = np.zeros((n, m)); z[:, 0] = rng.standard_normal(n)
        s = np.sqrt(1 - rho ** 2)
        for j in range(1, m):
            z[:, j] = rho * z[:, j - 1] + s * rng.standard_normal(n)
        return z
    thr = rng.uniform(-1, 1, m)
    return (hap() > thr).astype(float) + (hap() > thr).astype(float)


def _data(seed, N=2500, m=400, h2=0.5, p=0.05, enrich=12.0):
    rng = np.random.default_rng(seed)
    GA, GB = _geno(N, m, 0.6, rng), _geno(3000, m, 0.6, rng)
    ZA, ZB = standardize_dosage(GA), standardize_dosage(GB)
    R = (ZA.T @ ZA) / N; np.fill_diagonal(R, 1.0)
    func = (rng.random(m) < 0.2).astype(float)
    noise = (rng.random(m) < 0.3).astype(float)            # irrelevant
    base = np.where(func > 0, enrich, 1.0)
    causal = rng.random(m) < np.clip(base / base.sum() * (p * m), 0, 1)
    if not causal.any():
        causal[rng.integers(m)] = True
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
    gA = ZA @ beta
    y = gA + rng.normal(0, np.sqrt(max(1e-6, 1 - gA.var())), N)
    bhat = (ZA.T @ y) / N
    yte = ZB @ beta + rng.normal(0, np.sqrt(1 - h2), 3000)
    A = np.column_stack([func, noise])
    return R, bhat, N, A, ZB, yte


@pytest.mark.parametrize("learn", ["eb", "probit"])
def test_learns_enrichment_and_ignores_noise(learn):
    # Averaged over seeds, the coefficient on the informative annotation is
    # clearly positive and well above the irrelevant one.
    tf = tn = 0.0
    for seed in range(4):
        R, bhat, N, A, _, _ = _data(seed)
        res = ldpred2_auto_annot(R, bhat, N, A, learn=learn, burn_in=80,
                                 num_iter=200, seed=1)
        assert res.theta.shape == (3,)            # intercept + 2 annotations
        tf += res.theta[1]; tn += res.theta[2]
    tf /= 4; tn /= 4
    assert tf > 0.4, (learn, tf)
    assert tf > tn + 0.3, (learn, tf, tn)


def test_annot_predicts_at_least_as_well_as_uniform():
    # Learned annotation prior should not hurt held-out prediction on average.
    from pyldpred2 import ldpred2_auto
    r_uni = r_ann = 0.0
    for seed in range(4):
        R, bhat, N, A, ZB, yte = _data(seed)
        b_uni = ldpred2_auto(R, bhat, N, burn_in=80, num_iter=200, seed=1).beta_est
        b_ann = ldpred2_auto_annot(R, bhat, N, A, learn="eb", burn_in=80,
                                   num_iter=200, seed=1).beta_est
        r_uni += np.corrcoef(ZB @ b_uni, yte)[0, 1] ** 2
        r_ann += np.corrcoef(ZB @ b_ann, yte)[0, 1] ** 2
    assert r_ann / 4 >= r_uni / 4 - 0.01          # no meaningful regression


def test_sparse_ld_rejected():
    from pyldpred2 import sparsify_ld
    R = 0.5 ** np.abs(np.subtract.outer(np.arange(40), np.arange(40)))
    ld = sparsify_ld(R)
    with pytest.raises(NotImplementedError):
        ldpred2_auto_annot(ld, np.zeros(40), 5000, np.ones((40, 1)))
