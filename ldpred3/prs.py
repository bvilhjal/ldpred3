"""
Polygenic-score (PRS) computation from genotypes and per-variant effects.

Given a dosage matrix ``G`` of shape ``(n_samples, n_variants)`` (counting the
effect allele) and a vector of per-variant weights ``beta``, the PRS of each
individual is the weighted allele count::

    score_i = sum_j  beta_j * g_ij

LDpred3 returns weights on the **standardized** (allele-correlation) scale, so
by default this module standardizes each genotype column (``z = (g - 2f) / sd``)
before applying the weights -- the convention under which those weights are
defined. Missing calls (``-1``) are mean-imputed (equivalently, set to ``0``
after centering), the standard PRS treatment.

The caller is responsible for having aligned ``beta`` to the matrix's counted
(A1) allele beforehand; see :mod:`harmonize`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["allele_frequency", "standardize_dosage", "prs_score"]


def _as_float_with_nan(dosage):
    """Copy dosage to float, turning the ``-1`` missing sentinel into NaN."""
    g = np.asarray(dosage, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("dosage must be a 2-D (n_samples, n_variants) array")
    g = g.copy()
    g[g < 0] = np.nan
    return g


def allele_frequency(dosage):
    """A1 allele frequency per variant, ignoring missing calls."""
    g = _as_float_with_nan(dosage)
    af = np.nanmean(g, axis=0) / 2.0
    return np.where(np.isnan(af), 0.0, af)


def standardize_dosage(dosage, *, eps=1e-12):
    """Mean-impute missing calls and z-score each variant column.

    Returns ``Z`` of shape ``(n_samples, n_variants)`` with per-column mean 0
    and unit variance. Monomorphic columns (zero variance) become all-zero.
    """
    g = _as_float_with_nan(dosage)
    mean = np.nanmean(g, axis=0)
    # Mean-impute missing entries.
    inds = np.where(np.isnan(g))
    g[inds] = np.take(mean, inds[1])
    sd = g.std(axis=0)
    Z = g - mean
    np.divide(Z, sd, out=Z, where=sd > eps)
    Z[:, sd <= eps] = 0.0
    return Z


def prs_score(dosage, beta, *, standardize=True):
    """Per-individual polygenic score.

    Parameters
    ----------
    dosage : array_like, shape (n_samples, n_variants)
        Effect-allele dosages (``-1`` = missing).
    beta : array_like, shape (n_variants,)
        Per-variant weights, aligned to the dosage's counted allele.
    standardize : bool, default True
        If True (the LDpred3 convention), z-score genotype columns before
        weighting. If False, weight the raw mean-imputed dosages directly.

    Returns
    -------
    scores : ndarray, shape (n_samples,)
    """
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim != 1 or beta.shape[0] != np.shape(dosage)[1]:
        raise ValueError("beta must have one weight per variant")
    if standardize:
        G = standardize_dosage(dosage)
    else:
        g = _as_float_with_nan(dosage)
        mean = np.nanmean(g, axis=0)
        inds = np.where(np.isnan(g))
        g[inds] = np.take(mean, inds[1])
        G = g
    return G @ beta
