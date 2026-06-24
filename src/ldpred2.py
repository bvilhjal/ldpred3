"""
A basic, self-contained Python implementation of LDpred2.

LDpred2 (Privé, Arbel & Vilhjálmsson, *Bioinformatics* 2020) is a Bayesian
polygenic-score method that re-weights GWAS marginal effect sizes using an LD
(linkage-disequilibrium) correlation matrix. The original reference
implementation ships with the R package ``bigsnpr``; this module ports the core
algorithms to NumPy so they can be used and inspected from Python.

Three models are implemented here:

* ``ldpred2_inf``  -- the infinitesimal model (closed-form solution).
* ``ldpred2_grid`` -- the point-normal / spike-and-slab model fitted with a
  Gibbs sampler for fixed hyper-parameters ``(h2, p)``.
* ``ldpred2_auto`` -- the same Gibbs sampler, but ``h2`` (SNP heritability) and
  ``p`` (proportion of causal variants) are estimated on the fly.

Notation
--------
All effects are on the *standardized* scale (genotypes and phenotype scaled to
unit variance). With that convention the marginal (GWAS) effects ``beta_hat``
relate to the true joint effects ``beta`` through the LD matrix ``R``::

    beta_hat = R @ beta + noise,   noise ~ N(0, R / N)

where ``N`` is the GWAS sample size and ``R`` is the SNP correlation matrix.

The functions operate on a single LD block (a dense correlation matrix). Real
analyses run genome-wide by applying the model to each (approximately
independent) LD block separately; ``ldpred2_by_blocks`` is a thin helper for
that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "standardize_betas",
    "ldpred2_inf",
    "ldpred2_grid",
    "ldpred2_auto",
    "ldpred2_by_blocks",
    "AutoResult",
]


def standardize_betas(beta, beta_se, n_eff):
    """Put marginal GWAS effects on the standardized (allele-correlation) scale.

    GWAS are reported on many different scales (per-allele, log-odds, ...).
    LDpred2 works internally with effects scaled so that ``beta_hat`` is the
    correlation between the (standardized) genotype and phenotype. The standard
    transformation used by LDpred2 is::

        beta_std = beta / sqrt(n_eff * beta_se**2 + beta**2)

    which is approximately ``z / sqrt(n_eff)`` (``z`` being the GWAS z-score).

    Parameters
    ----------
    beta : array_like
        Marginal effect-size estimates.
    beta_se : array_like
        Standard errors of ``beta``.
    n_eff : array_like or float
        (Effective) GWAS sample size, per variant or a single scalar.

    Returns
    -------
    beta_std : ndarray
        Standardized marginal effects.
    scale : ndarray
        The per-variant scaling factor, so that ``beta == beta_std * scale``.
        Keep it to map adjusted standardized effects back to the input scale.
    """
    beta = np.asarray(beta, dtype=float)
    beta_se = np.asarray(beta_se, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)
    scale = np.sqrt(n_eff * beta_se ** 2 + beta ** 2)
    return beta / scale, scale


def _as_n_vector(n_eff, m):
    """Coerce ``n_eff`` into a length-``m`` float vector."""
    n_eff = np.asarray(n_eff, dtype=float)
    if n_eff.ndim == 0:
        return np.full(m, float(n_eff))
    if n_eff.shape != (m,):
        raise ValueError(f"n_eff must be a scalar or length-{m} vector")
    return n_eff


def ldpred2_inf(corr, beta_hat, n_eff, h2):
    """LDpred2 infinitesimal model (closed form).

    Assumes every variant is causal with effects drawn from
    ``beta ~ N(0, h2 / m)``. The posterior mean then has the closed form::

        beta_inf = (R + (m / (N * h2)) I)^{-1} beta_hat

    Parameters
    ----------
    corr : ndarray, shape (m, m)
        LD correlation matrix for the block.
    beta_hat : array_like, shape (m,)
        Standardized marginal effects (see :func:`standardize_betas`).
    n_eff : array_like or float
        GWAS sample size. A scalar (or the median of a vector) is used for the
        ridge term.
    h2 : float
        SNP heritability attributed to this block's variants.

    Returns
    -------
    ndarray, shape (m,)
        Adjusted (posterior-mean) standardized effects.
    """
    corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    N = float(np.median(n))
    A = corr + np.eye(m) * (m / (h2 * N))
    return np.linalg.solve(A, beta_hat)


def _gibbs_sampler(corr, beta_hat, n, h2, p, *, burn_in, num_iter, sparse,
                   rng, estimate_hyper, h2_bounds, shrink_corr):
    """Core point-normal Gibbs sampler shared by grid and auto.

    Returns ``(avg_beta, h2_path, p_path)`` where the paths hold the per-
    iteration hyper-parameter estimates (only meaningful when
    ``estimate_hyper`` is True).
    """
    m = beta_hat.shape[0]
    curr_beta = np.zeros(m)
    avg_beta = np.zeros(m)

    # Optionally shrink off-diagonal LD to stabilise the sampler (LDpred2
    # exposes this; default 1.0 = no shrinkage).
    if shrink_corr != 1.0:
        corr = corr * shrink_corr
        np.fill_diagonal(corr, 1.0)

    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)

    h2_min, h2_max = h2_bounds
    n_iter_total = burn_in + num_iter

    for it in range(n_iter_total):
        # Variance of a causal effect under the slab.
        c1 = h2 / (m * p)
        nb_causal = 0

        for j in range(m):
            # Residualised marginal effect for SNP j: remove the contribution
            # of every *other* SNP currently in the model. R[j, j] == 1.
            dotprod = corr[j].dot(curr_beta) - curr_beta[j]
            res_beta_j = beta_hat[j] - dotprod

            nj = n[j]
            # Posterior of the effect given that the SNP is causal.
            post_var = c1 / (nj * c1 + 1.0)        # = 1 / (nj + 1/c1)
            post_mean = nj * post_var * res_beta_j

            # Posterior probability that the SNP is causal (inclusion prob).
            # log-odds(null vs causal) = log((1-p)/p) + 0.5*log(1 + nj*c1)
            #                            - post_mean**2 / (2*post_var)
            log_odds = (np.log1p(-p) - np.log(p)
                        + 0.5 * np.log1p(nj * c1)
                        - 0.5 * post_mean ** 2 / post_var)
            postp = 1.0 / (1.0 + np.exp(log_odds))

            if sparse and postp < 0.5:
                # Hard-threshold to exactly zero (sparse LDpred2 variant).
                curr_beta[j] = 0.0
                continue

            if rng.random() < postp:
                curr_beta[j] = post_mean + rng.standard_normal() * np.sqrt(post_var)
                nb_causal += 1
            else:
                curr_beta[j] = 0.0

        if estimate_hyper:
            # Sample p from its Beta full-conditional given the causal count.
            p = rng.beta(1 + nb_causal, 1 + m - nb_causal)
            # Estimate h2 as the genetic variance implied by current effects:
            # h2 = beta^T R beta.
            h2 = float(curr_beta @ (corr @ curr_beta))
            h2 = min(max(h2, h2_min), h2_max)

        if it >= burn_in:
            k = it - burn_in
            avg_beta += curr_beta
            h2_path[k] = h2
            p_path[k] = p

    avg_beta /= num_iter
    return avg_beta, h2_path, p_path


def ldpred2_grid(corr, beta_hat, n_eff, h2, p, *, burn_in=100, num_iter=400,
                 sparse=False, shrink_corr=1.0, seed=None):
    """LDpred2 grid model: point-normal prior, fixed hyper-parameters.

    The prior is spike-and-slab: with probability ``p`` a variant is causal
    with effect ``N(0, h2 / (m * p))``, otherwise its effect is exactly zero.
    Posterior-mean effects are obtained by averaging a Gibbs sampler.

    In practice LDpred2-grid is run over a grid of ``(h2, p, sparse)`` values
    and the best combination is chosen with a validation set; this function
    fits a single grid point.

    Parameters
    ----------
    corr : ndarray, shape (m, m)
        LD correlation matrix.
    beta_hat : array_like, shape (m,)
        Standardized marginal effects.
    n_eff : array_like or float
        GWAS sample size (per variant or scalar).
    h2 : float
        SNP heritability for the block.
    p : float
        Proportion of causal variants (0 < p <= 1).
    burn_in, num_iter : int
        Number of burn-in and sampling iterations.
    sparse : bool
        If True, use the sparse variant (effects with inclusion prob < 0.5 are
        set exactly to zero).
    shrink_corr : float
        Multiplicative shrinkage applied to off-diagonal LD entries (1.0 = off).
    seed : int or None
        Seed for the random number generator.

    Returns
    -------
    ndarray, shape (m,)
        Posterior-mean standardized effects.
    """
    corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    rng = np.random.default_rng(seed)
    avg_beta, _, _ = _gibbs_sampler(
        corr, beta_hat, n, h2, p,
        burn_in=burn_in, num_iter=num_iter, sparse=sparse, rng=rng,
        estimate_hyper=False, h2_bounds=(1e-6, 1.0), shrink_corr=shrink_corr,
    )
    return avg_beta


@dataclass
class AutoResult:
    """Result of :func:`ldpred2_auto`."""

    beta_est: np.ndarray
    h2_est: float
    p_est: float
    h2_path: np.ndarray = field(repr=False)
    p_path: np.ndarray = field(repr=False)


def ldpred2_auto(corr, beta_hat, n_eff, *, h2_init=0.1, p_init=0.1,
                 burn_in=200, num_iter=200, shrink_corr=1.0,
                 h2_bounds=(1e-4, 1.0), seed=None):
    """LDpred2-auto: fit the point-normal model and estimate ``h2`` and ``p``.

    Unlike :func:`ldpred2_grid`, no validation set is needed: the proportion of
    causal variants ``p`` and the SNP heritability ``h2`` are updated within the
    Gibbs sampler and their posterior means are returned alongside the effects.

    Parameters
    ----------
    corr : ndarray, shape (m, m)
        LD correlation matrix.
    beta_hat : array_like, shape (m,)
        Standardized marginal effects.
    n_eff : array_like or float
        GWAS sample size.
    h2_init, p_init : float
        Starting values for the hyper-parameters.
    burn_in, num_iter : int
        Burn-in and sampling iterations.
    shrink_corr : float
        Off-diagonal LD shrinkage (1.0 = off).
    h2_bounds : (float, float)
        Clamp the per-iteration ``h2`` estimate to this range for stability.
    seed : int or None
        RNG seed.

    Returns
    -------
    AutoResult
        With fields ``beta_est``, ``h2_est``, ``p_est`` and the full sampling
        paths ``h2_path`` / ``p_path``.
    """
    corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    rng = np.random.default_rng(seed)
    avg_beta, h2_path, p_path = _gibbs_sampler(
        corr, beta_hat, n, h2_init, p_init,
        burn_in=burn_in, num_iter=num_iter, sparse=False, rng=rng,
        estimate_hyper=True, h2_bounds=h2_bounds, shrink_corr=shrink_corr,
    )
    return AutoResult(
        beta_est=avg_beta,
        h2_est=float(np.mean(h2_path)),
        p_est=float(np.mean(p_path)),
        h2_path=h2_path,
        p_path=p_path,
    )


def ldpred2_by_blocks(blocks, beta_hat, n_eff, method="auto", **kwargs):
    """Apply an LDpred2 model independently to a list of LD blocks.

    Genome-wide LDpred2 treats (approximately) independent LD blocks
    separately. This helper stitches the per-block results back into one vector.

    Parameters
    ----------
    blocks : sequence of (corr, index_array)
        Each element is a tuple ``(corr_block, idx)`` where ``corr_block`` is
        the ``(k, k)`` LD matrix for the block and ``idx`` are the positions of
        those ``k`` variants within the full ``beta_hat`` vector.
    beta_hat : array_like, shape (m,)
        Standardized marginal effects for all variants.
    n_eff : array_like or float
        GWAS sample size (per variant or scalar).
    method : {"inf", "grid", "auto"}
        Which model to run per block.
    **kwargs
        Passed through to the chosen model function. For ``inf``/``grid`` the
        per-block ``h2`` is rescaled by the fraction of variants in the block.

    Returns
    -------
    ndarray, shape (m,)
        Adjusted standardized effects for all variants.
    """
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    out = np.zeros(m)

    funcs = {"inf": ldpred2_inf, "grid": ldpred2_grid, "auto": ldpred2_auto}
    if method not in funcs:
        raise ValueError(f"method must be one of {sorted(funcs)}")

    # Total h2 (if given) is split across blocks proportionally to block size.
    total_h2 = kwargs.pop("h2", None)

    for corr_block, idx in blocks:
        idx = np.asarray(idx)
        k = idx.shape[0]
        block_kwargs = dict(kwargs)
        if method in ("inf", "grid"):
            block_kwargs["h2"] = (total_h2 * k / m) if total_h2 is not None else 0.1
        res = funcs[method](corr_block, beta_hat[idx], n[idx], **block_kwargs)
        out[idx] = res.beta_est if isinstance(res, AutoResult) else res
    return out
