"""Tests for PRS scoring."""

import os
import sys

import numpy as np


import pytest                                                              # noqa: E402

from ldpred3.prs import (allele_frequency, standardize_dosage,             # noqa: E402
                         dosage_stats, prs_score)


def test_allele_frequency_ignores_missing():
    dosage = np.array([[2, 0], [0, -1], [-1, 0], [2, 2]], dtype=np.int8)
    af = allele_frequency(dosage)
    # col0 non-missing: 2,0,2 -> mean 4/3 -> af 2/3; col1: 0,0,2 -> af 1/3
    np.testing.assert_allclose(af, [2 / 3, 1 / 3])


def test_standardize_dosage_centers_and_scales():
    rng = np.random.default_rng(1)
    dosage = rng.integers(0, 3, size=(200, 5)).astype(np.int8)
    Z = standardize_dosage(dosage)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-10)


def test_monomorphic_column_is_zero():
    dosage = np.array([[2, 1], [2, 0], [2, 2]], dtype=np.int8)  # col0 constant
    Z = standardize_dosage(dosage)
    np.testing.assert_array_equal(Z[:, 0], 0.0)


def test_prs_score_matches_manual_raw():
    dosage = np.array([[2, 0, 1], [1, 1, 0]], dtype=np.int8)
    beta = np.array([0.5, -1.0, 2.0])
    scores = prs_score(dosage, beta, standardize=False)
    np.testing.assert_allclose(scores, [2 * 0.5 + 0 * -1 + 1 * 2,
                                        1 * 0.5 + 1 * -1 + 0 * 2])


def test_prs_score_missing_imputed_to_mean():
    # With one missing entry, raw scoring imputes the column mean.
    dosage = np.array([[2], [0], [-1]], dtype=np.int8)
    beta = np.array([1.0])
    scores = prs_score(dosage, beta, standardize=False)
    # col mean over non-missing (2,0) = 1.0 -> imputed sample gets 1.0
    np.testing.assert_allclose(scores, [2.0, 0.0, 1.0])


def test_all_missing_column_does_not_poison_scores():
    # An entirely-missing variant column must contribute nothing, not NaN.
    dosage = np.array([[2, -1], [0, -1], [1, -1]], dtype=np.int8)
    Z = standardize_dosage(dosage)
    assert np.all(np.isfinite(Z))
    np.testing.assert_array_equal(Z[:, 1], 0.0)
    beta = np.array([0.7, 3.0])                 # nonzero weight on the dead column
    for std in (True, False):
        scores = prs_score(dosage, beta, standardize=std)
        assert np.all(np.isfinite(scores)), f"NaN PRS with standardize={std}"


def test_prs_score_frozen_uses_supplied_mean_sd():
    rng = np.random.default_rng(4)
    dosage = rng.integers(0, 3, size=(40, 3)).astype(np.int8)
    beta = rng.standard_normal(3)
    mean = np.array([1.0, 0.5, 1.5]); sd = np.array([0.8, 0.7, 0.9])
    got = prs_score(dosage, beta, mean=mean, sd=sd)
    expected = ((dosage.astype(float) - mean) / sd) @ beta
    np.testing.assert_allclose(got, expected)
    # frozen differs from the cohort's own standardization
    assert not np.allclose(got, prs_score(dosage, beta, standardize=True))
    with pytest.raises(ValueError, match="both mean and sd"):
        prs_score(dosage, beta, mean=mean)            # sd missing


def test_dosage_stats_matches_standardize():
    rng = np.random.default_rng(5)
    dosage = rng.integers(0, 3, size=(60, 4)).astype(np.int8)
    mean, sd = dosage_stats(dosage)
    # applying the frozen stats to the same cohort == standardize_dosage
    np.testing.assert_allclose(prs_score(dosage, np.ones(4), mean=mean, sd=sd),
                               standardize_dosage(dosage) @ np.ones(4), atol=1e-9)


def test_prs_score_standardized_default():
    rng = np.random.default_rng(2)
    dosage = rng.integers(0, 3, size=(50, 4)).astype(np.int8)
    beta = rng.standard_normal(4)
    scores = prs_score(dosage, beta, standardize=True)
    expected = standardize_dosage(dosage) @ beta
    np.testing.assert_allclose(scores, expected)
