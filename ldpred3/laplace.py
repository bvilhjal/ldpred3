"""Bayesian-lasso (Laplace-prior) Gibbs sampler for LDpred3.

The lasso (``lassosum2``) is the *posterior mode* under a Laplace (double-
exponential) prior on the effects. This module samples the **posterior mean**
instead — the proper Bayesian shrinkage estimator, which for prediction is
generally better than the mode.

It uses the normal / exponential scale-mixture representation of the Laplace
(Andrews & Mallows 1974; the Bayesian lasso of Park & Casella 2008): with a
per-SNP latent scale ``τ_j²``,

    β_j | τ_j²  ~  N(0, τ_j²)
    τ_j²        ~  Exponential(λ²/2)          =>   β_j ~ Laplace(λ)

so the Gibbs sweep is the *same* Gaussian per-SNP conditional the point-normal
sampler already uses (prior variance ``τ_j²`` in place of the slab), plus an
Inverse-Gaussian draw for ``1/τ_j²`` and a conjugate Gamma draw for the global
shrinkage ``λ²`` (so ``λ`` self-tunes, no penalty grid — the "auto" analogue).

Unlike the spike-and-slab there is no point mass at zero: the posterior mean is
dense, with heavier-tailed (less uniform) shrinkage than the infinitesimal
Gaussian. Dense LD blocks only. NumPy-only, optional Numba.
"""

from __future__ import annotations

import numpy as np

from ._numba import _jit

__all__ = ["ldpred3_laplace"]


def _rand_invgauss(mu, lam):
    """One draw from the Inverse-Gaussian (mean ``mu``, shape ``lam``).

    Michael, Schucany & Haas (1976): a normal + uniform, so it is Numba-safe.
    """
    v = np.random.normal(0.0, 1.0)
    y = v * v
    mu2 = mu * mu
    x = mu + (mu2 * y) / (2.0 * lam) - (mu / (2.0 * lam)) * np.sqrt(
        4.0 * mu * lam * y + mu2 * y * y)
    if not (x > 0.0):                      # floating-point guard (mu huge)
        x = mu
    if np.random.random() <= mu / (mu + x):
        return x
    return mu2 / x


_rand_invgauss = _jit(_rand_invgauss)


def _laplace_sweep(R, r, beta, Rb, n, tau2, lam, post_sum, accumulate):
    """One Gibbs sweep over a dense block; returns ``Σ τ_j²`` (for the λ draw).

    Draws each ``β_j`` from its Gaussian full-conditional (prior variance the
    latent ``τ_j²``), keeps the running ``Rβ`` up to date, accumulates the
    Rao-Blackwellised posterior mean, then refreshes ``1/τ_j² ~ InvGauss``.
    """
    k = beta.shape[0]
    tau_total = 0.0
    for j in range(k):
        old = beta[j]
        res = r[j] - (Rb[j] - old)          # residualised marginal (R_jj = 1)
        nj = n[j]
        t2 = tau2[j]
        pv = t2 / (nj * t2 + 1.0)           # posterior variance
        mean = nj * pv * res                # posterior mean
        new = mean + np.random.normal(0.0, 1.0) * np.sqrt(pv)
        d = new - old
        if d != 0.0:
            cj = R[j]
            for i in range(k):
                Rb[i] += cj[i] * d
            beta[j] = new
        if accumulate:
            post_sum[j] += mean             # Rao-Blackwellised estimate
        # latent-scale update: 1/tau_j^2 ~ InvGauss(lam/|beta_j|, lam^2)
        ab = new if new >= 0.0 else -new
        if ab < 1e-8:
            ab = 1e-8
        mu = lam / ab
        if mu > 1e6:
            mu = 1e6
        w = _rand_invgauss(mu, lam * lam)
        if w < 1e-12:
            w = 1e-12
        nt2 = 1.0 / w
        if nt2 > 1e6:
            nt2 = 1e6
        tau2[j] = nt2
        tau_total += nt2
    return tau_total


_laplace_sweep = _jit(_laplace_sweep)


def _seed(s):
    np.random.seed(s)


_seed = _jit(_seed)


def ldpred3_laplace(corr, beta_hat, n_eff, *, h2=0.1, burn_in=100, num_iter=400,
                    seed=None, sample_lambda=True, lam=None):
    """Fit the Bayesian-lasso (Laplace-prior) model to one dense LD block.

    Parameters
    ----------
    corr : ndarray (k, k)
        Dense LD matrix for the block.
    beta_hat : array_like (k,)
        Standardized marginal effects.
    n_eff : array_like or float
        GWAS sample size (per variant or scalar).
    h2 : float, default 0.1
        Heritability *of this block*. Sets the scale of ``λ`` (via
        ``λ = √(2k/h2)``, the value that makes the Laplace prior's total variance
        equal ``h2``): the initial value, and — when ``sample_lambda`` — the seed
        the self-tuning refines.
    burn_in, num_iter : int
        Discarded warm-up sweeps, then averaged sweeps.
    seed : int, optional
        RNG seed (reproducible fit).
    sample_lambda : bool, default True
        Self-tune the global shrinkage each sweep by the marginal-maximisation
        (EM) update ``λ² = 2k / Σ τ_j²`` (Park & Casella 2008), which converges to
        the value matching the fitted total variance. If False, ``λ`` is held at
        ``lam`` (or its ``h2`` init). (A naïve Gamma-Gibbs update on ``λ`` is
        *not* used: with a scale mixture it drifts to the hyper-prior's mean
        independently of the data and systematically mis-shrinks.)
    lam : float, optional
        Fixed / initial Laplace rate. Defaults to ``√(2k/h2)``.

    Returns
    -------
    ndarray (k,)
        Posterior-mean effect estimate (the PRS weights for the block).
    """
    corr = np.ascontiguousarray(corr, dtype=np.float64)
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    k = beta_hat.shape[0]
    if k == 0:
        return np.zeros(0)
    n = np.asarray(n_eff, dtype=float)
    if n.ndim == 0:
        n = np.full(k, float(n))
    h2 = float(max(h2, 1e-6))

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    _seed(int(seed) % (2 ** 31 - 1))          # sampler RNG (Numba's or NumPy's)

    tau2 = np.full(k, h2 / k)
    if lam is None:
        lam = float(np.sqrt(2.0 * k / h2))
    lam = float(lam)

    beta = np.zeros(k)
    Rb = np.zeros(k)
    post_sum = np.zeros(k)
    n_acc = 0
    for it in range(int(burn_in) + int(num_iter)):
        accumulate = it >= int(burn_in)
        tau_total = _laplace_sweep(corr, beta_hat, beta, Rb, n, tau2, lam,
                                   post_sum, accumulate)
        if accumulate:
            n_acc += 1
        if sample_lambda and tau_total > 1e-12:
            # marginal-maximisation (EM) update: lambda^2 = 2k / sum(tau_j^2).
            lam = float(np.sqrt(2.0 * k / tau_total))
    return post_sum / max(n_acc, 1)
