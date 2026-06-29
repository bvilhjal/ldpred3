"""Tests for PRS scoring."""

import os
import sys

import numpy as np


from ldpred3.prs import allele_frequency, standardize_dosage, prs_score   # noqa: E402


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


def test_prs_score_standardized_default():
    rng = np.random.default_rng(2)
    dosage = rng.integers(0, 3, size=(50, 4)).astype(np.int8)
    beta = rng.standard_normal(4)
    scores = prs_score(dosage, beta, standardize=True)
    expected = standardize_dosage(dosage) @ beta
    np.testing.assert_allclose(scores, expected)
