"""Binary-trait scale helpers and PRS standardization."""

import numpy as np
import pytest

from ldpred3 import n_eff_case_control, h2_liability, standardize_prs


def test_n_eff_balanced_equals_total():
    # balanced case/control -> effective N == total N
    assert n_eff_case_control(5000, 5000) == pytest.approx(10000)
    # extreme imbalance -> approaches 4 * n_case
    assert n_eff_case_control(1000, 10**9) == pytest.approx(4000, rel=1e-3)
    # arrays
    n = n_eff_case_control(np.array([5000, 1000]), np.array([5000, 9000]))
    assert np.allclose(n, [10000, 4 / (1 / 1000 + 1 / 9000)])


def test_n_eff_rejects_nonpositive():
    with pytest.raises(ValueError):
        n_eff_case_control(0, 100)


def test_h2_liability_rare_disease_inflates_observed():
    # For a rare disease the observed-scale h2 is much smaller than liability;
    # the conversion factor for a population sample is K(1-K)/z^2.
    from statistics import NormalDist
    K = 0.01
    nd = NormalDist()
    t = nd.inv_cdf(1 - K); z = nd.pdf(t)
    expected_factor = (K * (1 - K)) ** 2 / (z ** 2 * K * (1 - K))  # P=K
    assert h2_liability(0.05, K) == pytest.approx(0.05 * expected_factor)
    # ascertained 50/50 study, rare disease: liability << naive observed
    h2l = h2_liability(0.2, K, prop_cases=0.5)
    assert 0 < h2l < 0.2


def test_h2_liability_validates():
    with pytest.raises(ValueError):
        h2_liability(0.1, 0.0)
    with pytest.raises(ValueError):
        h2_liability(0.1, 0.5, prop_cases=1.0)


def test_standardize_prs_z_and_percentile():
    rng = np.random.default_rng(0)
    s = rng.normal(3.0, 2.0, 10000)
    z, pct = standardize_prs(s)
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1.0) < 1e-6
    assert 0.0 < pct.min() < 0.001 and 0.999 < pct.max() < 1.0
    # frozen reference scale: a new cohort placed on the same scale
    z2, _ = standardize_prs(s + 5.0, ref_mean=s.mean(), ref_sd=s.std())
    assert z2.mean() == pytest.approx(5.0 / s.std(), rel=1e-6)
