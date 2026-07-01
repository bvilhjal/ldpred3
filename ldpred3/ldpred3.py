"""
A basic, self-contained Python implementation of LDpred3.

LDpred re-weights GWAS marginal effect sizes using an LD (linkage-disequilibrium)
correlation matrix to recover the *joint* effects that drive a polygenic score.
The Bayesian point-normal model is due to Vilhjálmsson *et al.* (*AJHG* 2015);
the faster, better-behaved grid / auto samplers used here are LDpred2 (Privé,
Arbel & Vilhjálmsson, *Bioinformatics* 2020), whose reference implementation
ships with the R package ``bigsnpr``. This module ports the core algorithms to
NumPy so they can be used and inspected from Python. See
``docs/algorithm.md`` (References) for the full bibliography.

Effect-size models implemented here:

* ``ldpred3_inf``  -- infinitesimal (all variants causal, Gaussian prior); the
  posterior mean is a closed-form ridge/BLUP solve (Vilhjálmsson 2015).
* ``ldpred3_grid`` -- point-normal / spike-and-slab prior fitted with a Gibbs
  sampler at fixed hyper-parameters ``(h2, p)`` (LDpred2, Privé 2020).
* ``ldpred3_auto`` -- the same sampler, but ``h2`` (SNP heritability) and ``p``
  (proportion of causal variants) are estimated on the fly, needing no
  validation set (LDpred2-auto; the robust multi-chain estimator and the
  disease-architecture inference are Privé *et al.* *AJHG* 2023).
* ``ldpred3_laplace`` -- the Bayesian lasso: the posterior *mean* of a Laplace
  prior (Park & Casella 2008), the Bayesian counterpart of ``lassosum2`` (in
  ``lassosum.py``), which is that prior's *mode* (Mak *et al.* 2017).

The prior can be further shaped per variant: ``prior_weights`` re-weights the
*inclusion* probability from functional annotations (SBayesR/RC), and ``alpha``
scales the slab *variance* by allele frequency (the ``use_MLE`` prior of
LDpred2-auto; Speed *et al.* 2017, Zeng *et al.* 2018).

Notation
--------
All effects are on the *standardized* scale (genotypes and phenotype scaled to
unit variance). With that convention the marginal (GWAS) effects ``beta_hat``
relate to the true joint effects ``beta`` through the LD matrix ``R``::

    beta_hat = R @ beta + noise,   noise ~ N(0, R / N)

where ``N`` is the GWAS sample size and ``R`` is the SNP correlation matrix.
This is the LDpred2 sampling model; the per-SNP Gibbs conditional is exact
under it (``R_jj = 1``), so the approximations are the choice of model, the
reference-panel estimate of ``R``, and the block-diagonal split below.

The functions operate on a single LD block (a dense correlation matrix). Real
analyses run genome-wide by applying the model to each (approximately
independent) LD block separately; ``ldpred3_by_blocks`` is a thin helper for
that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# JIT decorators (no-op without Numba) and the LD utilities live in their own
# modules; re-exported here so the historical ``from .ldpred3 import ...`` import
# surface (used by infer / annot / bivariate and the tests) keeps working.
from ._numba import HAVE_NUMBA, _jit, _jit_parallel, _set_threads, prange  # noqa: F401,E402
from .ld_utils import (SparseLD, sparsify_ld, block_diagonal_ld,  # noqa: F401,E402
                       optimal_ld_blocks, shrink_ld_blocks,
                       LowRankLD, lowrank_ld)
from .laplace import ldpred3_laplace  # noqa: E402


def _stable_postp(log_odds):
    """P(causal) = 1 / (1 + exp(log_odds)), computed without overflow.

    ``log_odds`` is the log-odds of null vs causal; for large positive values a
    naive ``exp(log_odds)`` overflows. This branch keeps the argument of ``exp``
    non-positive.
    """
    if log_odds >= 0.0:
        e = np.exp(-log_odds)
        return e / (1.0 + e)
    return 1.0 / (1.0 + np.exp(log_odds))


_stable_postp = _jit(_stable_postp)


__all__ = [
    "standardize_betas",
    "ldpred3_inf",
    "ldpred3_grid",
    "ldpred3_auto",
    "ldpred3_laplace",
    "ldpred3_by_blocks",
    "maf_slab_weights",
    "AutoResult",
    "SparseLD",
    "sparsify_ld",
    "block_diagonal_ld",
    "optimal_ld_blocks",
    "shrink_ld_blocks",
    "LowRankLD",
    "lowrank_ld",
]


def standardize_betas(beta, beta_se, n_eff):
    """Put marginal GWAS effects on the standardized (allele-correlation) scale.

    GWAS are reported on many different scales (per-allele, log-odds, ...).
    LDpred3 works internally with effects scaled so that ``beta_hat`` is the
    correlation between the (standardized) genotype and phenotype. The standard
    transformation used by LDpred3 is::

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
    if np.any(beta_se < 0):
        raise ValueError("beta_se must be non-negative")
    if np.any(n_eff <= 0):
        raise ValueError("n_eff must be positive")
    scale = np.sqrt(n_eff * beta_se ** 2 + beta ** 2)
    # Guard the degenerate beta == 0 and beta_se == 0 case (scale == 0) -> 0.
    beta_std = np.divide(beta, scale, out=np.zeros_like(beta, dtype=float),
                         where=scale > 0)
    return beta_std, scale


def maf_slab_weights(af, alpha):
    """Per-SNP slab-variance weights for a MAF-dependent effect architecture.

    Scales the causal-effect variance by ``[2f(1−f)]^(1+α)`` (the MAF-coupling /
    "S" / α parameter — Speed et al. 2017, Zeng et al. 2018, and the ``use_MLE``
    extension of LDpred2-auto, Privé et al. 2023), normalised to **mean 1** so the
    total ``h2`` budget is preserved. The default LDpred model is ``α = −1`` → ``[2f(1−f)]^0 = 1``
    → uniform weights (no MAF dependence); ``α < −1`` up-weights rarer variants
    (per-standardized-effect larger for rare alleles, the signature of negative
    selection). Pass the result as ``slab_weights`` to scale the prior.
    """
    af = np.asarray(af, dtype=float)
    het = 2.0 * af * (1.0 - af)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(het > 0, het, np.nan) ** (1.0 + alpha)
    mw = np.nanmean(w)
    if not np.isfinite(mw) or mw <= 0:
        return np.ones_like(af)
    return np.where(np.isfinite(w), w / mw, 1.0)


def _as_n_vector(n_eff, m):
    """Coerce ``n_eff`` into a length-``m`` positive float vector."""
    n_eff = np.asarray(n_eff, dtype=float)
    if n_eff.ndim == 0:
        n_eff = np.full(m, float(n_eff))
    elif n_eff.shape != (m,):
        raise ValueError(f"n_eff must be a scalar or length-{m} vector")
    if np.any(n_eff <= 0):
        raise ValueError("n_eff must be positive")
    return n_eff


def _check_h2_p(h2=None, p=None):
    """Validate heritability and causal-fraction hyper-parameters."""
    if h2 is not None and not h2 > 0:
        raise ValueError("h2 must be > 0")
    if p is not None and not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")


def ldpred3_inf(corr, beta_hat, n_eff, h2):
    """LDpred3 infinitesimal model (closed form; Vilhjálmsson et al. 2015).

    Assumes every variant is causal with effects drawn from
    ``beta ~ N(0, h2 / m)``. Under the ``beta_hat = R beta + N(0, R/N)`` model
    this Gaussian prior is conjugate, so the posterior mean (the ridge/BLUP
    solution) has the closed form::

        beta_inf = (R + (m / (N * h2)) I)^{-1} beta_hat

    Parameters
    ----------
    corr : ndarray (m, m) or SparseLD
        LD correlation matrix for the block. A dense matrix is solved directly;
        a :class:`SparseLD` is solved iteratively (conjugate gradient on the
        sparse system), avoiding the dense O(m^3) factorisation.
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
    _check_h2_p(h2=h2)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    N = float(np.median(n))
    ridge = m / (h2 * N)

    if isinstance(corr, SparseLD):
        # Solve (R + ridge*I) x = beta_hat by conjugate gradient; the system is
        # SPD (R is a correlation matrix and ridge>0) and only needs matvecs.
        return _cg_solve(corr, ridge, beta_hat)

    if isinstance(corr, LowRankLD):
        # (U U^T + ridge I)^-1 b via Woodbury: O(m r + r^3), no dense m x m.
        U = np.asarray(corr.U, dtype=float)
        r = U.shape[1]
        Utb = U.T @ beta_hat
        inner = ridge * np.eye(r) + U.T @ U
        return (beta_hat - U @ np.linalg.solve(inner, Utb)) / ridge

    corr = np.asarray(corr, dtype=float)
    A = corr + np.eye(m) * ridge
    return np.linalg.solve(A, beta_hat)


def _sparse_matvec(indptr, indices, data, x, out):
    """out = R @ x for a symmetric CSR matrix (out is overwritten)."""
    m = out.shape[0]
    for j in range(m):
        acc = 0.0
        for idx in range(indptr[j], indptr[j + 1]):
            acc += data[idx] * x[indices[idx]]
        out[j] = acc
    return out


_sparse_matvec_jit = _jit(_sparse_matvec)


def _cg_solve(ld, ridge, b, tol=1e-6, max_iter=1000):
    """Conjugate-gradient solve of (R + ridge*I) x = b for SparseLD R."""
    m = ld.m
    indptr, indices, data = ld.indptr, ld.indices, ld.data
    x = np.zeros(m)
    r = b.copy()                       # residual = b - A@0
    pvec = r.copy()
    rs = float(r @ r)
    if rs == 0.0:                      # b is all zeros -> x = 0
        return x
    Ap = np.empty(m)
    tol2 = tol * tol * float(b @ b)
    converged = False
    for _ in range(max_iter):
        _sparse_matvec_jit(indptr, indices, data, pvec, Ap)
        Ap += ridge * pvec
        alpha = rs / float(pvec @ Ap)
        x += alpha * pvec
        r -= alpha * Ap
        rs_new = float(r @ r)
        if rs_new <= tol2:
            converged = True
            break
        pvec = r + (rs_new / rs) * pvec
        rs = rs_new
    if not converged:
        import warnings
        warnings.warn(f"ldpred3_inf conjugate gradient did not converge in "
                      f"{max_iter} iterations (residual {rs_new ** 0.5:.2e})",
                      RuntimeWarning)
    return x


def _gibbs_kernel(corr, beta_hat, n, h2, p, burn_in, num_iter, sparse,
                  estimate_hyper, h2_min, h2_max, seed, init_beta, tol,
                  check_every, allow_jump_sign, prior_w, slab_w):
    """Numeric core of the point-normal Gibbs sampler (JIT-compiled if numba).

    Takes only plain numeric / array arguments so it compiles under
    ``numba.njit``. Uses the legacy global ``np.random`` (seeded here): its
    ``random`` / ``standard_normal`` streams are identical between the compiled
    and pure-Python paths; ``beta`` (used only for the -auto p-update) may differ
    slightly between the two but yields an equally valid sampler.

    ``init_beta`` warm-starts the chain (e.g. from LDpred3-inf). When ``tol`` > 0,
    the sampler stops early once the running posterior mean's relative RMS change
    over ``check_every`` sweeps falls below ``tol`` (adaptive stopping).

    Returns ``(avg_beta, h2_path, p_path, count)``; ``count`` is the number of
    post-burn-in sweeps actually used, and the paths are truncated to it.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]

    # Hoist the constant-N / uniform-prior posterior scalars out of the per-SNP
    # loop (as the batched/streaming kernels do): with a shared N and a uniform
    # slab the point-normal variance / normaliser are identical for every SNP, so
    # recomputing sqrt/log1p per SNP each sweep is pure overhead; a uniform
    # prior_w likewise collapses the log prior-odds to a scalar (no length-m
    # vector). Both fall back to the exact per-SNP path when N/slab/prior vary.
    n_const = m > 0 and n.min() == n.max()
    n0 = n[0] if m > 0 else 0.0
    slab0 = slab_w[0] if m > 0 else 1.0
    pw0 = prior_w[0] if m > 0 else 1.0
    slab_uniform = m > 0 and np.all(slab_w == slab0)
    prior_uniform = m > 0 and np.all(prior_w == pw0)
    fast = n_const and slab_uniform
    dummy = np.zeros(1)                           # placeholder when prior is uniform

    curr_beta = init_beta.copy()                 # warm start (zeros = cold start)
    avg_beta = np.zeros(m)
    # Per-sweep Rao-Blackwellized contribution E[beta_j | rest] = postp * post_mean.
    post_means = np.zeros(m)
    # Running product Rb = R @ curr_beta, maintained incrementally; initialised
    # from curr_beta by the resync at it == 0 below.
    Rb = np.zeros(m)
    prev_mean = np.zeros(m)                       # snapshot for convergence check

    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    n_iter_total = burn_in + num_iter
    count = 0

    for it in range(n_iter_total):
        # Base causal-effect slab variance (global p). ``slab_w`` scales it per
        # SNP for a MAF-dependent architecture (slab_w == 1 -> uniform); ``prior_w``
        # separately re-weights the *inclusion* probability.
        c1 = h2 / (m * p)
        # Per-SNP prior log-odds of null vs causal: p_j = p * prior_w[j]. A uniform
        # prior_w makes this a single scalar; only the non-uniform case builds the
        # length-m vector (two transcendentals per SNP).
        if prior_uniform:
            pj0 = min(max(p * pw0, 1e-9), 1.0 - 1e-9)
            lpo_scalar = np.log1p(-pj0) - np.log(pj0)
            log_prior_odds_v = dummy
        else:
            pj = np.minimum(np.maximum(p * prior_w, 1e-9), 1.0 - 1e-9)
            log_prior_odds_v = np.log1p(-pj) - np.log(pj)
            lpo_scalar = 0.0
        # Constant N + uniform slab -> precompute pv/sd/half-log/n·pv once a sweep.
        pv0, psd0, half0, npv0 = _pn_const_scalars(fast, n0, c1 * slab0)
        nb_causal = 0

        # Resync Rb from scratch at it == 0 (initialises it from the warm-start
        # curr_beta) and periodically thereafter to bound floating-point drift.
        if it % 100 == 0:
            Rb[:] = 0.0
            for k in range(m):
                bk = curr_beta[k]
                if bk != 0.0:
                    ck = corr[k]
                    for i in range(m):
                        Rb[i] += ck[i] * bk

        # Batch the random draws for the whole sweep (far cheaper than per-SNP
        # RNG calls in the Python loop).
        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)

        for j in range(m):
            old = curr_beta[j]
            # Residualised marginal effect: subtract every other SNP's
            # contribution. Rb[j] includes this SNP (corr[j, j] == 1), so add
            # back its own term.
            res_beta_j = beta_hat[j] - Rb[j] + old
            # Shared per-SNP point-normal update; the Rao-Blackwellised post_means
            # (accumulated below) drive the dense estimate, the sampled value the
            # chain. The ``fast`` branch feeds the per-sweep constant-N scalars
            # (bit-identical arithmetic); otherwise N/slab vary and _pn_step
            # recomputes per SNP. prior_w enters through lpo_j (scalar if uniform).
            lpo_j = lpo_scalar if prior_uniform else log_prior_odds_v[j]
            if fast:
                new, pm, dc, _ = _pn_step(res_beta_j, old, n[j], c1 * slab0, True,
                                       pv0, psd0, half0, npv0, lpo_j,
                                       unif[j], gauss[j], sparse, allow_jump_sign)
            else:
                new, pm, dc, _ = _pn_step(res_beta_j, old, n[j], c1 * slab_w[j], False,
                                       0.0, 0.0, 0.0, 0.0, lpo_j,
                                       unif[j], gauss[j], sparse, allow_jump_sign)
            post_means[j] = pm
            nb_causal += dc

            delta = new - old
            if delta != 0.0:
                # Rank-1 update of Rb = R @ curr_beta. Fused element loop (no
                # temporary array) over a single-precision row -- this is the
                # sampler's hot path and is memory-bandwidth bound, so the
                # float32 row halves the dominant traffic.
                cj = corr[j]
                for i in range(m):
                    Rb[i] += cj[i] * delta
                curr_beta[j] = new

        if estimate_hyper:
            # Sample p from its Beta full-conditional given the causal count.
            p = np.random.beta(1.0 + nb_causal, 1.0 + m - nb_causal)
            # h2 = beta^T R beta, reusing the maintained Rb (no extra matvec).
            h2 = 0.0
            for i in range(m):
                h2 += curr_beta[i] * Rb[i]
            if h2 < h2_min:
                h2 = h2_min
            elif h2 > h2_max:
                h2 = h2_max

        if it >= burn_in:
            # Rao-Blackwellized posterior mean for the dense estimator; for the
            # sparse variant accumulate the sampled (hard-thresholded) effects so
            # the result stays sparse.
            if sparse:
                avg_beta += curr_beta
            else:
                avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            count += 1

            # Adaptive stopping: relative RMS change of the running mean.
            if tol > 0.0 and count % check_every == 0:
                num = 0.0
                den = 0.0
                for i in range(m):
                    cm = avg_beta[i] / count
                    d = cm - prev_mean[i]
                    num += d * d
                    den += cm * cm
                    prev_mean[i] = cm
                if count > check_every and num <= tol * tol * den:
                    break

    if count == 0:
        count = 1
    avg_beta /= count
    return avg_beta, h2_path[:count], p_path[:count], count


# Compiled (or pass-through) version of the kernel.
_gibbs_kernel_jit = _jit(_gibbs_kernel)


def _gibbs_kernel_sample(corr, beta_hat, n, h2, p, burn_in, num_iter,
                         h2_min, h2_max, seed, sample_every, allow_jump_sign,
                         estimate_p, estimate_h2):
    """Auto Gibbs kernel that also retains thinned *sampled* effect vectors.

    A trimmed copy of :func:`_gibbs_kernel` (dense, ``estimate_hyper`` always on,
    no sparse / warm-start / adaptive-stop branches) that, in addition to the
    Rao-Blackwellized posterior mean and the ``h2``/``p`` paths, stores the
    sampled ``curr_beta`` every ``sample_every`` post-burn-in sweeps. Those
    samples drive the LDpred2-auto predictive-r2 estimator (Privé et al. 2023).

    Returns ``(avg_beta, h2_path, p_path, beta_samples, n_saved, pip)`` where
    ``pip`` is the per-SNP posterior inclusion probability (the post-burn-in mean
    of the Rao-Blackwellized inclusion probability ``postp``) -- the quantity
    fine-mapping needs.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]
    # Constant N (the usual scalar-N inference / fine-mapping case) lets the
    # posterior scalars be computed once a sweep instead of per SNP.
    n_const = m > 0 and n.min() == n.max()
    n0 = n[0] if m > 0 else 0.0
    curr_beta = np.zeros(m)
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
    pip = np.zeros(m)
    Rb = np.zeros(m)

    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    max_saved = num_iter // sample_every + 1
    beta_samples = np.zeros((max_saved, m))
    n_iter_total = burn_in + num_iter
    count = 0
    n_saved = 0

    for it in range(n_iter_total):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)
        nb_causal = 0
        acc = it >= burn_in            # accumulate PIPs only post burn-in

        if it % 100 == 0:
            Rb[:] = 0.0
            for k in range(m):
                bk = curr_beta[k]
                if bk != 0.0:
                    ck = corr[k]
                    for i in range(m):
                        Rb[i] += ck[i] * bk

        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)

        for j in range(m):
            old = curr_beta[j]
            res_beta_j = beta_hat[j] - Rb[j] + old
            new, pm, dc, postp = _pn_step(res_beta_j, old, n[j], c1, n_const,
                                          pv0, psd0, half0, npv0, log_prior_odds,
                                          unif[j], gauss[j], False, allow_jump_sign)
            post_means[j] = pm
            nb_causal += dc
            if acc:
                pip[j] += postp
            delta = new - old
            if delta != 0.0:
                cj = corr[j]
                for i in range(m):
                    Rb[i] += cj[i] * delta
                curr_beta[j] = new

        if estimate_p:
            p = np.random.beta(1.0 + nb_causal, 1.0 + m - nb_causal)
        if estimate_h2:
            h2 = 0.0
            for i in range(m):
                h2 += curr_beta[i] * Rb[i]
            if h2 < h2_min:
                h2 = h2_min
            elif h2 > h2_max:
                h2 = h2_max

        if it >= burn_in:
            avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            if count % sample_every == 0:
                for i in range(m):
                    beta_samples[n_saved, i] = curr_beta[i]
                n_saved += 1
            count += 1

    if count == 0:
        count = 1
    avg_beta /= count
    pip /= count
    return avg_beta, h2_path[:count], p_path[:count], beta_samples[:n_saved], n_saved, pip


_gibbs_kernel_sample_jit = _jit(_gibbs_kernel_sample)


def _gibbs_kernel_sparse(indptr, indices, data, beta_hat, n, h2, p, burn_in,
                         num_iter, sparse, estimate_hyper, h2_min, h2_max, seed,
                         init_beta, tol, check_every, allow_jump_sign, prior_w):
    """Sparse (CSR) counterpart of :func:`_gibbs_kernel`.

    Identical point-normal Gibbs / Rao-Blackwellized sampler (incl. warm start
    and adaptive stopping), but the LD matrix is stored as CSR
    (``indptr``/``indices``/``data``), so the rank-1 update and the resync touch
    only the non-zero neighbours of each SNP -- O(bandwidth) rather than O(m).
    The diagonal must be present in the CSR structure.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]

    # Same constant-N / uniform-prior hoisting as the dense kernel (there is no
    # per-SNP slab here, so the fast path is just constant N).
    n_const = m > 0 and n.min() == n.max()
    n0 = n[0] if m > 0 else 0.0
    pw0 = prior_w[0] if m > 0 else 1.0
    prior_uniform = m > 0 and np.all(prior_w == pw0)
    dummy = np.zeros(1)

    curr_beta = init_beta.copy()
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
    Rb = np.zeros(m)
    prev_mean = np.zeros(m)

    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    n_iter_total = burn_in + num_iter
    count = 0

    for it in range(n_iter_total):
        c1 = h2 / (m * p)
        if prior_uniform:
            pj0 = min(max(p * pw0, 1e-9), 1.0 - 1e-9)
            lpo_scalar = np.log1p(-pj0) - np.log(pj0)
            log_prior_odds_v = dummy
        else:
            pj = np.minimum(np.maximum(p * prior_w, 1e-9), 1.0 - 1e-9)
            log_prior_odds_v = np.log1p(-pj) - np.log(pj)
            lpo_scalar = 0.0
        pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)
        nb_causal = 0

        # Resync Rb at it == 0 (from warm start) and periodically (non-zeros only).
        if it % 100 == 0:
            Rb[:] = 0.0
            for k in range(m):
                bk = curr_beta[k]
                if bk != 0.0:
                    for idx in range(indptr[k], indptr[k + 1]):
                        Rb[indices[idx]] += data[idx] * bk

        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)

        for j in range(m):
            old = curr_beta[j]
            res_beta_j = beta_hat[j] - Rb[j] + old
            lpo_j = lpo_scalar if prior_uniform else log_prior_odds_v[j]
            new, pm, dc, _ = _pn_step(res_beta_j, old, n[j], c1, n_const,
                                   pv0, psd0, half0, npv0, lpo_j,
                                   unif[j], gauss[j], sparse, allow_jump_sign)
            post_means[j] = pm
            nb_causal += dc

            delta = new - old
            if delta != 0.0:
                # Rank-1 update over the non-zero neighbours only (O(bandwidth)).
                for idx in range(indptr[j], indptr[j + 1]):
                    Rb[indices[idx]] += data[idx] * delta
                curr_beta[j] = new

        if estimate_hyper:
            p = np.random.beta(1.0 + nb_causal, 1.0 + m - nb_causal)
            h2 = 0.0
            for i in range(m):
                h2 += curr_beta[i] * Rb[i]
            if h2 < h2_min:
                h2 = h2_min
            elif h2 > h2_max:
                h2 = h2_max

        if it >= burn_in:
            if sparse:
                avg_beta += curr_beta
            else:
                avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            count += 1

            if tol > 0.0 and count % check_every == 0:
                num = 0.0
                den = 0.0
                for i in range(m):
                    cm = avg_beta[i] / count
                    d = cm - prev_mean[i]
                    num += d * d
                    den += cm * cm
                    prev_mean[i] = cm
                if count > check_every and num <= tol * tol * den:
                    break

    if count == 0:
        count = 1
    avg_beta /= count
    return avg_beta, h2_path[:count], p_path[:count], count


_gibbs_kernel_sparse_jit = _jit(_gibbs_kernel_sparse)


def _gibbs_kernel_batched(block_data, block_offsets, snp_offsets, sizes,
                          beta_hat, n, h2, p, burn_in, num_iter, sparse,
                          estimate_hyper, h2_min, h2_max, seed, init_beta, tol,
                          check_every, n_const):
    """Process all (dense) LD blocks in a single compiled sweep loop.

    Blocks are packed contiguously (``block_data`` = each block's k*k matrix,
    row-major; ``block_offsets`` index into it; ``snp_offsets``/``sizes`` give
    each block's global SNP range). This keeps the fast *dense contiguous* rank-1
    update (no CSR index indirection) while estimating ``h2``/``p`` GLOBALLY --
    ``nb_causal`` and ``beta^T R beta`` are pooled across all blocks each sweep --
    and avoids one Python call per block. Same point-normal / Rao-Blackwellized
    sampler, warm start and adaptive stopping as the other kernels.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]
    nblk = sizes.shape[0]

    curr_beta = init_beta.copy()
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
    Rb = np.zeros(m)
    prev_mean = np.zeros(m)

    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    n_iter_total = burn_in + num_iter
    count = 0

    # When N is constant the per-SNP posterior constants collapse to scalars
    # (_pn_const_scalars), computed once per sweep -- a big saving at genome
    # scale; _pn_step recomputes per SNP only in the rare per-variant-N case.
    n0 = float(n[0])

    for it in range(n_iter_total):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)
        nb_causal = 0

        # Resync Rb at it == 0 (from warm start) and periodically; within-block.
        if it % 100 == 0:
            Rb[:] = 0.0
            for b in range(nblk):
                base = snp_offsets[b]
                k = sizes[b]
                boff = block_offsets[b]
                for lj in range(k):
                    bj = curr_beta[base + lj]
                    if bj != 0.0:
                        rowoff = boff + lj * k
                        for li in range(k):
                            Rb[base + li] += block_data[rowoff + li] * bj

        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)

        for b in range(nblk):
            base = snp_offsets[b]
            k = sizes[b]
            boff = block_offsets[b]
            for lj in range(k):
                gj = base + lj
                old = curr_beta[gj]
                res_beta_j = beta_hat[gj] - Rb[gj] + old
                new, pm, dc, _ = _pn_step(res_beta_j, old, n[gj], c1, n_const,
                                       pv0, psd0, half0, npv0, log_prior_odds,
                                       unif[gj], gauss[gj], sparse, True)
                post_means[gj] = pm
                nb_causal += dc
                delta = new - old
                if delta != 0.0:
                    rowoff = boff + lj * k       # contiguous float32 block row
                    for li in range(k):
                        Rb[base + li] += block_data[rowoff + li] * delta
                    curr_beta[gj] = new

        if estimate_hyper:
            # GLOBAL hyper-parameter updates pooled across all blocks.
            p = np.random.beta(1.0 + nb_causal, 1.0 + m - nb_causal)
            h2 = 0.0
            for i in range(m):
                h2 += curr_beta[i] * Rb[i]
            if h2 < h2_min:
                h2 = h2_min
            elif h2 > h2_max:
                h2 = h2_max

        if it >= burn_in:
            if sparse:
                avg_beta += curr_beta
            else:
                avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            count += 1
            if tol > 0.0 and count % check_every == 0:
                num = 0.0
                den = 0.0
                for i in range(m):
                    cm = avg_beta[i] / count
                    d = cm - prev_mean[i]
                    num += d * d
                    den += cm * cm
                    prev_mean[i] = cm
                if count > check_every and num <= tol * tol * den:
                    break

    if count == 0:
        count = 1
    avg_beta /= count
    return avg_beta, h2_path[:count], p_path[:count], count


_gibbs_kernel_batched_jit = _jit(_gibbs_kernel_batched)


def _gibbs_kernel_batched_par(block_data, block_offsets, snp_offsets, sizes,
                              beta_hat, n, h2, p, burn_in, num_iter, sparse,
                              estimate_hyper, h2_min, h2_max, seed, init_beta,
                              tol, check_every, n_const):
    """Multicore variant of the batched kernel: the per-sweep loop over blocks
    runs under ``numba.prange``. Blocks are independent (block-diagonal LD) and
    write only to their own disjoint slices of ``Rb``/``curr_beta``/``post_means``,
    so the parallel sweep is race-free; ``nb_causal`` is a prange reduction. RNG
    is drawn serially each sweep, so results are independent of the thread count.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]
    nblk = sizes.shape[0]

    curr_beta = init_beta.copy()
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
    Rb = np.zeros(m)
    prev_mean = np.zeros(m)
    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    n_iter_total = burn_in + num_iter
    count = 0
    n0 = float(n[0])

    for it in range(n_iter_total):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)

        if it % 100 == 0:
            Rb[:] = 0.0
            for b in prange(nblk):
                base = snp_offsets[b]
                k = sizes[b]
                boff = block_offsets[b]
                for lj in range(k):
                    bj = curr_beta[base + lj]
                    if bj != 0.0:
                        rowoff = boff + lj * k
                        for li in range(k):
                            Rb[base + li] += block_data[rowoff + li] * bj

        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)
        nb_causal = 0

        for b in prange(nblk):
            base = snp_offsets[b]
            k = sizes[b]
            boff = block_offsets[b]
            for lj in range(k):
                gj = base + lj
                old = curr_beta[gj]
                res_beta_j = beta_hat[gj] - Rb[gj] + old
                new, pm, dc, _ = _pn_step(res_beta_j, old, n[gj], c1, n_const,
                                       pv0, psd0, half0, npv0, log_prior_odds,
                                       unif[gj], gauss[gj], sparse, True)
                post_means[gj] = pm
                nb_causal += dc
                delta = new - old
                if delta != 0.0:
                    rowoff = boff + lj * k
                    for li in range(k):
                        Rb[base + li] += block_data[rowoff + li] * delta
                    curr_beta[gj] = new

        if estimate_hyper:
            p = np.random.beta(1.0 + nb_causal, 1.0 + m - nb_causal)
            h2 = 0.0
            for i in range(m):
                h2 += curr_beta[i] * Rb[i]
            if h2 < h2_min:
                h2 = h2_min
            elif h2 > h2_max:
                h2 = h2_max

        if it >= burn_in:
            if sparse:
                avg_beta += curr_beta
            else:
                avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            count += 1
            if tol > 0.0 and count % check_every == 0:
                num = 0.0
                den = 0.0
                for i in range(m):
                    cm = avg_beta[i] / count
                    d = cm - prev_mean[i]
                    num += d * d
                    den += cm * cm
                    prev_mean[i] = cm
                if count > check_every and num <= tol * tol * den:
                    break

    if count == 0:
        count = 1
    avg_beta /= count
    return avg_beta, h2_path[:count], p_path[:count], count


_gibbs_kernel_batched_par_jit = _jit_parallel(_gibbs_kernel_batched_par)


def _pack_blocks(blocks):
    """Pack dense LD blocks into contiguous arrays for the batched kernel.

    Blocks must tile ``0 .. m-1`` contiguously. Returns
    ``(block_data, block_offsets, snp_offsets, sizes, m)``.
    """
    blocks = sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0]))
    data_parts = []
    sizes = []
    for cb, idx in blocks:
        k = int(np.asarray(idx).shape[0])
        cb = np.ascontiguousarray(cb, dtype=np.float32)
        if cb.shape != (k, k):
            raise ValueError("each block must be a dense (k, k) matrix")
        data_parts.append(cb.ravel())
        sizes.append(k)
    sizes = np.asarray(sizes, dtype=np.int32)
    snp_offsets = np.zeros(sizes.shape[0], dtype=np.int32)
    if sizes.shape[0] > 1:
        np.cumsum(sizes[:-1], out=snp_offsets[1:])
    block_offsets = np.zeros(sizes.shape[0], dtype=np.int64)
    if sizes.shape[0] > 1:
        np.cumsum(sizes[:-1].astype(np.int64) ** 2, out=block_offsets[1:])
    block_data = (np.concatenate(data_parts) if data_parts
                  else np.empty(0, np.float32)).astype(np.float32)
    return block_data, block_offsets, snp_offsets, sizes, int(sizes.sum())


def _gibbs_blocks(blocks, beta_hat, n, h2, p, *, burn_in, num_iter, sparse,
                  seed, estimate_hyper, h2_bounds, warm_start=False, tol=0.0,
                  check_every=50, ncores=1):
    """Run the batched single-call sampler over a list of dense LD blocks.

    With ``ncores`` > 1 the per-sweep block loop is parallelised (``prange``);
    this packs the blocks into one contiguous array, trading the streaming
    sampler's low memory for multicore speed.
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    block_data, block_offsets, snp_offsets, sizes, m = _pack_blocks(blocks)
    h2_min, h2_max = h2_bounds
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]

    if warm_start:
        init_beta = np.zeros(m)
        for cb, idx in blocks:
            idx = np.asarray(idx)
            k = idx.shape[0]
            init_beta[idx] = ldpred3_inf(cb, beta_hat[idx], n[idx], h2 * k / m)
    else:
        init_beta = np.zeros(m)

    n_const = bool(n.size > 0 and n.min() == n.max())
    args = (block_data, block_offsets, snp_offsets, sizes, beta_hat, n,
            float(h2), float(p), int(burn_in), int(num_iter), bool(sparse),
            bool(estimate_hyper), float(h2_min), float(h2_max), int(seed),
            init_beta, float(tol), int(check_every), n_const)
    if ncores and ncores > 1 and HAVE_NUMBA:
        _set_threads(ncores)
        return _gibbs_kernel_batched_par_jit(*args)
    return _gibbs_kernel_batched_jit(*args)


def _pn_step(res, old, nj, c1, n_const, pv0, psd0, half0, npv0,
             lpo, u, g, sparse, allow_jump_sign):
    """The per-SNP point-normal update shared by every Gibbs sweep kernel.

    Given a SNP's residualised marginal estimate ``res`` and its current effect
    ``old``, returns ``(new, pm_rb, dc, postp)``: the resampled effect, the
    Rao-Blackwellised contribution ``P(causal)·E[beta|causal]``, the change in
    the causal count (0 or 1), and the inclusion probability ``postp`` (the
    fine-mapping PIP contribution). Keeping this in one place means the inclusion-
    probability math and the sign-flip divergence guard (Privé et al.) have a
    single definition for the dense / banded / low-rank / packed kernels.

    ``n_const`` lets a constant-N caller pass the per-sweep precomputed posterior
    scalars (``pv0`` etc.) instead of recomputing the ``sqrt``/``log1p`` per SNP;
    a per-variant-N caller passes ``n_const=False`` and the SNP's ``nj``. ``lpo``
    is the (possibly per-SNP) log prior-odds of null vs causal. With
    ``allow_jump_sign=False`` a proposal that would flip the effect's sign in one
    step is zeroed (it would otherwise diverge on ill-conditioned LD).
    """
    if n_const:
        pv = pv0
        psd = psd0
        half = half0
        post_mean = npv0 * res
    else:
        nc1 = nj * c1
        pv = c1 / (nc1 + 1.0)
        psd = np.sqrt(pv)
        half = 0.5 * np.log1p(nc1)
        post_mean = nj * pv * res
    log_odds = lpo + half - 0.5 * post_mean * post_mean / pv
    postp = _stable_postp(log_odds)
    pm_rb = postp * post_mean

    if sparse and postp < 0.5:
        new = 0.0
        dc = 0
    elif u < postp:
        new = post_mean + g * psd
        dc = 1
    else:
        new = 0.0
        dc = 0

    # Forbid a within-step sign flip (a major source of divergence on noisy /
    # ill-conditioned LD); a flipped proposal is zeroed (and uncounted).
    if (not allow_jump_sign and old != 0.0 and new != 0.0
            and (new > 0.0) != (old > 0.0)):
        new = 0.0
        dc = 0
    return new, pm_rb, dc, postp


_pn_step = _jit(_pn_step)


def _pn_const_scalars(n_const, n0, c1):
    """Precompute the constant-N posterior scalars (pv, sd, half-log, n·pv)."""
    if n_const:
        nc1 = n0 * c1
        pv0 = c1 / (nc1 + 1.0)
        return pv0, np.sqrt(pv0), 0.5 * np.log1p(nc1), n0 * pv0
    return 0.0, 0.0, 0.0, 0.0


_pn_const_scalars = _jit(_pn_const_scalars)


def _gibbs_one_sweep(corr, beta_hat, n, curr_beta, Rb, post_means, unif, gauss,
                     c1, log_prior_odds, sparse, allow_jump_sign,
                     n_const, n0, resync):
    """One point-normal Gibbs sweep over a single dense block (in place).

    Operates on length-k slices of the global state, so the caller can stream
    blocks without materialising a packed copy. Returns this block's
    ``(nb_causal, beta^T R beta)`` so the driver can pool h2/p globally.
    """
    k = beta_hat.shape[0]
    if resync:
        for li in range(k):
            Rb[li] = 0.0
        for lj in range(k):
            bj = curr_beta[lj]
            if bj != 0.0:
                cj = corr[lj]
                for li in range(k):
                    Rb[li] += cj[li] * bj

    pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)
    nb_causal = 0
    for lj in range(k):
        old = curr_beta[lj]
        res_beta_j = beta_hat[lj] - Rb[lj] + old
        new, pm, dc, _ = _pn_step(res_beta_j, old, n[lj], c1, n_const,
                               pv0, psd0, half0, npv0, log_prior_odds,
                               unif[lj], gauss[lj], sparse, allow_jump_sign)
        post_means[lj] = pm
        nb_causal += dc
        delta = new - old
        if delta != 0.0:
            cj = corr[lj]
            for li in range(k):
                Rb[li] += cj[li] * delta
            curr_beta[lj] = new

    gv = 0.0
    for i in range(k):
        gv += curr_beta[i] * Rb[i]
    return nb_causal, gv


_gibbs_one_sweep_jit = _jit(_gibbs_one_sweep)


def _gibbs_one_sweep_sparse(indptr, indices, data, beta_hat, n, curr_beta, Rb,
                            post_means, unif, gauss, c1, log_prior_odds, sparse,
                            allow_jump_sign, n_const, n0, resync):
    """CSR counterpart of :func:`_gibbs_one_sweep`: one sweep over a banded block.

    ``indptr``/``indices``/``data`` are the block's local CSR (indices in
    ``0..k-1``); the rank-1 residual update and the resync touch only each SNP's
    non-zero neighbours -- O(bandwidth) -- so a large block costs O(k·bandwidth)
    memory and time instead of O(k²). Returns ``(nb_causal, beta^T R beta)`` for
    global h2/p pooling.
    """
    k = beta_hat.shape[0]
    if resync:
        for li in range(k):
            Rb[li] = 0.0
        for lj in range(k):
            bj = curr_beta[lj]
            if bj != 0.0:
                for idx in range(indptr[lj], indptr[lj + 1]):
                    Rb[indices[idx]] += data[idx] * bj

    pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)
    nb_causal = 0
    for lj in range(k):
        old = curr_beta[lj]
        res_beta_j = beta_hat[lj] - Rb[lj] + old
        new, pm, dc, _ = _pn_step(res_beta_j, old, n[lj], c1, n_const,
                               pv0, psd0, half0, npv0, log_prior_odds,
                               unif[lj], gauss[lj], sparse, allow_jump_sign)
        post_means[lj] = pm
        nb_causal += dc
        delta = new - old
        if delta != 0.0:
            for idx in range(indptr[lj], indptr[lj + 1]):
                Rb[indices[idx]] += data[idx] * delta
            curr_beta[lj] = new

    gv = 0.0
    for i in range(k):
        gv += curr_beta[i] * Rb[i]
    return nb_causal, gv


_gibbs_one_sweep_sparse_jit = _jit(_gibbs_one_sweep_sparse)


def _gibbs_one_sweep_lowrank(U, beta_hat, n, curr_beta, s, post_means, unif,
                             gauss, c1, log_prior_odds, sparse, allow_jump_sign,
                             n_const, n0, resync):
    """Eigenspace counterpart of :func:`_gibbs_one_sweep` for a low-rank block.

    ``R ~= U U^T`` (unit diagonal). The block's residual is carried in the
    r-vector ``s = U^T beta`` instead of a length-k ``Rb``: ``(R beta)_j = U[j].s``
    and an effect change ``delta`` updates ``s += delta*U[j]`` -- O(r) per SNP,
    O(k*r) total, with no dense k×k. ``beta^T R beta = ||s||^2``.
    """
    k = beta_hat.shape[0]
    r = U.shape[1]
    if resync:
        for c in range(r):
            s[c] = 0.0
        for j in range(k):
            bj = curr_beta[j]
            if bj != 0.0:
                for c in range(r):
                    s[c] += U[j, c] * bj

    pv0, psd0, half0, npv0 = _pn_const_scalars(n_const, n0, c1)
    nb_causal = 0
    for j in range(k):
        old = curr_beta[j]
        rbj = 0.0
        for c in range(r):
            rbj += U[j, c] * s[c]
        res_beta_j = beta_hat[j] - rbj + old        # (U U^T)_jj = 1
        new, pm, dc, _ = _pn_step(res_beta_j, old, n[j], c1, n_const,
                               pv0, psd0, half0, npv0, log_prior_odds,
                               unif[j], gauss[j], sparse, allow_jump_sign)
        post_means[j] = pm
        nb_causal += dc
        delta = new - old
        if delta != 0.0:
            for c in range(r):
                s[c] += U[j, c] * delta
            curr_beta[j] = new

    gv = 0.0
    for c in range(r):
        gv += s[c] * s[c]
    return nb_causal, gv


_gibbs_one_sweep_lowrank_jit = _jit(_gibbs_one_sweep_lowrank)


def _gibbs_blocks_stream(blocks, beta_hat, n, h2, p, *, burn_in, num_iter,
                         sparse, seed, estimate_hyper, h2_bounds,
                         warm_start=False, tol=0.0, check_every=50,
                         allow_jump_sign=True):
    """Streaming global-hyper sampler: process blocks one at a time.

    Keeps the LD blocks in place (a float32 copy per distinct block; shared
    blocks are de-duplicated by identity) instead of packing them into one big
    contiguous array, so peak memory is the LD itself plus O(m) state -- not the
    ~2-3x packing peak. h2/p are pooled across all blocks each sweep (global
    hyper-parameters). Returns ``(avg_beta, h2_path, p_path, count)``.
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    m = beta_hat.shape[0]
    h2_min, h2_max = h2_bounds
    rng = np.random.default_rng(seed)
    n_const = bool(n.size > 0 and n.min() == n.max())
    n0 = float(n[0]) if n_const else 0.0

    # Per-block payloads, de-duplicated by object identity (shared -> 1 copy).
    # Dense blocks are stored as float32; SparseLD blocks keep their banded CSR
    # so a large block costs O(k*bandwidth) memory, not O(k^2).
    # kind: 0=dense (float32), 1=SparseLD (banded CSR), 2=LowRankLD (eigenspace).
    # Each carries O(k*bandwidth) / O(k*rank) memory instead of O(k^2). LowRank
    # blocks also get a persistent score buffer s = U^T beta (length r).
    cache = {}
    fblocks = []           # (kind, obj, start, k, s_or_None)
    for cb, idx in sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0])):
        idx = np.asarray(idx)
        start, k = int(idx[0]), int(idx.shape[0])
        if isinstance(cb, SparseLD):
            fblocks.append((1, cb, start, k, None))
        elif isinstance(cb, LowRankLD):
            fblocks.append((2, cb, start, k, np.zeros(cb.U.shape[1])))
        else:
            key = id(cb)
            if key not in cache:
                cache[key] = np.ascontiguousarray(cb, dtype=np.float32)
            fblocks.append((0, cache[key], start, k, None))

    curr_beta = np.zeros(m)
    if warm_start:
        for kind, obj, start, k, _s in fblocks:
            sl = slice(start, start + k)
            curr_beta[sl] = ldpred3_inf(obj, beta_hat[sl], n[sl], h2 * k / m)
    Rb = np.zeros(m)
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
    prev_mean = np.zeros(m)
    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    count = 0

    for it in range(burn_in + num_iter):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        unif = rng.random(m)
        gauss = rng.standard_normal(m)
        resync = (it % 100 == 0)
        nb_causal = 0
        gv = 0.0
        for kind, obj, start, k, s in fblocks:
            sl = slice(start, start + k)
            if kind == 1:
                nbc, gvb = _gibbs_one_sweep_sparse_jit(
                    obj.indptr, obj.indices, obj.data, beta_hat[sl], n[sl],
                    curr_beta[sl], Rb[sl], post_means[sl], unif[sl], gauss[sl],
                    c1, log_prior_odds, bool(sparse), bool(allow_jump_sign),
                    n_const, n0, resync)
            elif kind == 2:
                nbc, gvb = _gibbs_one_sweep_lowrank_jit(
                    obj.U, beta_hat[sl], n[sl], curr_beta[sl], s, post_means[sl],
                    unif[sl], gauss[sl], c1, log_prior_odds, bool(sparse),
                    bool(allow_jump_sign), n_const, n0, resync)
            else:
                nbc, gvb = _gibbs_one_sweep_jit(
                    obj, beta_hat[sl], n[sl], curr_beta[sl], Rb[sl], post_means[sl],
                    unif[sl], gauss[sl], c1, log_prior_odds, bool(sparse),
                    bool(allow_jump_sign), n_const, n0, resync)
            nb_causal += nbc
            gv += gvb

        if estimate_hyper:
            p = float(rng.beta(1.0 + nb_causal, 1.0 + m - nb_causal))
            h2 = min(max(gv, h2_min), h2_max)

        if it >= burn_in:
            if sparse:
                avg_beta += curr_beta
            else:
                avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            count += 1
            if tol > 0.0 and count % check_every == 0:
                cm = avg_beta / count
                num = float(np.sum((cm - prev_mean) ** 2))
                den = float(np.sum(cm * cm))
                prev_mean = cm
                if count > check_every and num <= tol * tol * den:
                    break

    if count == 0:
        count = 1
    avg_beta /= count
    return avg_beta, h2_path[:count], p_path[:count], count


def _gibbs_blocks_stream_sample(blocks, beta_hat, n, h2, p, *, burn_in, num_iter,
                                seed, h2_bounds, sample_every,
                                allow_jump_sign=True):
    """Streaming auto sampler that also retains thinned *sampled* effect vectors.

    The block-diagonal counterpart of :func:`_gibbs_kernel_sample`: it estimates
    ``h2``/``p`` each sweep (global hyper-parameters, pooled across blocks) and
    stores ``curr_beta`` every ``sample_every`` post-burn-in sweeps for the
    LDpred3-auto predictive-r2 estimator -- without ever materialising a
    genome-wide LD matrix. Like :func:`_gibbs_blocks_stream`, blocks may be dense
    (float32), banded :class:`SparseLD` (O(k·bandwidth)) or :class:`LowRankLD`
    (O(k·rank) eigenspace), so inference scales the same way scoring does.
    Returns ``(avg_beta, h2_path, p_path, samples)`` with ``samples`` an
    ``(n_saved, m)`` float32 array.
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    m = beta_hat.shape[0]
    h2_min, h2_max = h2_bounds
    rng = np.random.default_rng(seed)
    n_const = bool(n.size > 0 and n.min() == n.max())
    n0 = float(n[0]) if n_const else 0.0

    # Per-block payloads, de-duplicated by identity for dense blocks.
    # kind: 0=dense (float32), 1=SparseLD (banded CSR), 2=LowRankLD (eigenspace).
    # LowRank blocks carry a persistent score buffer s = U^T beta (length r).
    cache = {}
    fblocks = []           # (kind, obj, start, k, s_or_None)
    for cb, idx in sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0])):
        idx = np.asarray(idx)
        start, k = int(idx[0]), int(idx.shape[0])
        if isinstance(cb, SparseLD):
            fblocks.append((1, cb, start, k, None))
        elif isinstance(cb, LowRankLD):
            fblocks.append((2, cb, start, k, np.zeros(cb.U.shape[1])))
        else:
            key = id(cb)
            if key not in cache:
                cache[key] = np.ascontiguousarray(cb, dtype=np.float32)
            fblocks.append((0, cache[key], start, k, None))

    curr_beta = np.zeros(m)
    Rb = np.zeros(m)
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
    h2_path = np.empty(num_iter)
    p_path = np.empty(num_iter)
    max_saved = num_iter // sample_every + 1
    samples = np.zeros((max_saved, m), dtype=np.float32)
    count = 0
    n_saved = 0

    for it in range(burn_in + num_iter):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        unif = rng.random(m)
        gauss = rng.standard_normal(m)
        resync = (it % 100 == 0)
        nb_causal = 0
        gv = 0.0
        for kind, obj, start, k, s in fblocks:
            sl = slice(start, start + k)
            if kind == 1:
                nbc, gvb = _gibbs_one_sweep_sparse_jit(
                    obj.indptr, obj.indices, obj.data, beta_hat[sl], n[sl],
                    curr_beta[sl], Rb[sl], post_means[sl], unif[sl], gauss[sl],
                    c1, log_prior_odds, False, bool(allow_jump_sign),
                    n_const, n0, resync)
            elif kind == 2:
                nbc, gvb = _gibbs_one_sweep_lowrank_jit(
                    obj.U, beta_hat[sl], n[sl], curr_beta[sl], s, post_means[sl],
                    unif[sl], gauss[sl], c1, log_prior_odds, False,
                    bool(allow_jump_sign), n_const, n0, resync)
            else:
                nbc, gvb = _gibbs_one_sweep_jit(
                    obj, beta_hat[sl], n[sl], curr_beta[sl], Rb[sl],
                    post_means[sl], unif[sl], gauss[sl], c1, log_prior_odds,
                    False, bool(allow_jump_sign), n_const, n0, resync)
            nb_causal += nbc
            gv += gvb

        p = float(rng.beta(1.0 + nb_causal, 1.0 + m - nb_causal))
        h2 = min(max(gv, h2_min), h2_max)

        if it >= burn_in:
            avg_beta += post_means
            h2_path[count] = h2
            p_path[count] = p
            if count % sample_every == 0:
                samples[n_saved] = curr_beta
                n_saved += 1
            count += 1

    if count == 0:
        count = 1
    avg_beta /= count
    return avg_beta, h2_path[:count], p_path[:count], samples[:n_saved]


def _gibbs_sampler(corr, beta_hat, n, h2, p, *, burn_in, num_iter, sparse,
                   seed, estimate_hyper, h2_bounds, shrink_corr,
                   warm_start=False, tol=0.0, check_every=50,
                   allow_jump_sign=True, prior_w=None, slab_w=None):
    """Prepare arguments and dispatch to the (optionally JIT-compiled) kernel.

    ``corr`` may be a dense ndarray or a :class:`SparseLD`; the matching dense or
    sparse kernel is used. With ``warm_start`` the chain is initialised from the
    LDpred3-inf solution; with ``tol`` > 0 the sampler stops early once the
    running estimate converges. Returns ``(avg_beta, h2_path, p_path, count)``.
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    h2_min, h2_max = h2_bounds
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]

    m = beta_hat.shape[0]
    if prior_w is None:
        prior_w = np.ones(m)
    else:
        prior_w = np.ascontiguousarray(prior_w, dtype=np.float64)
        if prior_w.shape != (m,):
            raise ValueError("prior_weights must be a length-m vector")
        if np.any(prior_w < 0):
            raise ValueError("prior_weights must be non-negative")
    if slab_w is None:
        slab_w = np.ones(m)
    else:
        slab_w = np.ascontiguousarray(slab_w, dtype=np.float64)
        if slab_w.shape != (m,):
            raise ValueError("slab_weights must be a length-m vector")
        if np.any(slab_w <= 0):
            raise ValueError("slab_weights must be positive")

    # Warm start from the (cheap) infinitesimal solution, else cold start.
    if warm_start:
        init_beta = np.ascontiguousarray(
            ldpred3_inf(corr, beta_hat, n, h2), dtype=np.float64)
    else:
        init_beta = np.zeros(beta_hat.shape[0])

    if isinstance(corr, SparseLD):
        if shrink_corr != 1.0:
            raise ValueError("shrink_corr is only supported for dense LD")
        if np.any(slab_w != 1.0):
            raise ValueError("the MAF-dependent prior (alpha/slab_weights) needs "
                             "dense LD, not SparseLD")
        return _gibbs_kernel_sparse_jit(
            corr.indptr, corr.indices, corr.data, beta_hat, n, float(h2),
            float(p), int(burn_in), int(num_iter), bool(sparse),
            bool(estimate_hyper), float(h2_min), float(h2_max), int(seed),
            init_beta, float(tol), int(check_every), bool(allow_jump_sign),
            prior_w,
        )

    # Single-precision, contiguous LD matrix. ``corr`` is symmetric, so row j
    # (a contiguous slice) is also column j -- used for the rank-1 update. float32
    # halves the memory traffic of that (bandwidth-bound) hot loop with no
    # meaningful accuracy cost; it also matches bigsnpr, which stores LD in single
    # precision. The accumulator Rb and effects stay in float64.
    corr = np.ascontiguousarray(corr, dtype=np.float32)

    # Optionally shrink off-diagonal LD to stabilise the sampler (LDpred3
    # exposes this; default 1.0 = no shrinkage). Copy so the caller's matrix is
    # untouched.
    if shrink_corr != 1.0:
        corr = corr * np.float32(shrink_corr)
        np.fill_diagonal(corr, np.float32(1.0))

    return _gibbs_kernel_jit(
        corr, beta_hat, n, float(h2), float(p), int(burn_in), int(num_iter),
        bool(sparse), bool(estimate_hyper), float(h2_min), float(h2_max),
        int(seed), init_beta, float(tol), int(check_every),
        bool(allow_jump_sign), prior_w, slab_w,
    )


def ldpred3_grid(corr, beta_hat, n_eff, h2, p, *, burn_in=100, num_iter=400,
                 sparse=False, shrink_corr=1.0, warm_start=False, tol=0.0,
                 check_every=50, allow_jump_sign=True, prior_weights=None,
                 af=None, alpha=-1.0, seed=None):
    """LDpred3 grid model: point-normal prior, fixed hyper-parameters.

    The point-normal (spike-and-slab) prior of LDpred (Vilhjálmsson et al.,
    *AJHG* 2015), sampled with the LDpred2 Gibbs sampler (Privé et al.,
    *Bioinformatics* 2020): with probability ``p`` a variant is causal with
    effect ``N(0, h2 / (m * p))``, otherwise its effect is exactly zero. The
    per-SNP full conditional is the standard point-normal update (a Gaussian
    posterior gated by its Bayes factor); the returned effects are the
    Rao-Blackwellised posterior mean averaged over the sampler.

    In practice LDpred3-grid is run over a grid of ``(h2, p, sparse)`` values
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
    prior_weights : array_like, shape (m,), optional
        Per-variant relative causal propensity (SBayesRC-style annotation-
        informed prior). Each SNP's causal probability becomes ``p_j = p *
        prior_weights[j]`` (clamped to ``(0, 1)``); pass weights with mean ~1 to
        keep the expected causal count and ``h2`` coherent. ``None`` (default)
        gives the uniform-``p`` point-normal model. Informative weights raise
        accuracy; misleading ones lower it, so supply trustworthy annotations.
        (The SBayesR/RC idea; see :func:`ldpred3_auto_annot` to *learn* the map.)
    af : array_like, shape (m,), optional
        Per-variant allele frequency, enabling the MAF-dependent slab-variance
        prior together with ``alpha`` (the ``use_MLE`` prior of LDpred2-auto,
        Privé et al. 2023; the ``S``/``α`` model of Speed et al. 2017, Zeng et
        al. 2018). ``None`` (default) leaves the slab variance uniform.
    alpha : float, default -1.0
        Exponent of the MAF-dependent prior: the **standardized-effect** slab
        variance is scaled by ``[2f(1-f)]^(1+alpha)`` (mean-normalised, so ``h2``
        is preserved). ``-1`` is the flat prior (uniform standardized effects) and
        reproduces the uniform model exactly; ``alpha < -1`` up-weights **rarer**
        variants (larger standardized effect for rare alleles — the negative-
        selection signature), ``alpha > -1`` up-weights common variants. Needs
        ``af``.
    seed : int or None
        Seed for the random number generator.

    Returns
    -------
    ndarray, shape (m,)
        Posterior-mean standardized effects.
    """
    _check_h2_p(h2=h2, p=p)
    if not isinstance(corr, SparseLD):
        corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    slab_w = maf_slab_weights(af, alpha) if af is not None and alpha != -1.0 else None
    avg_beta, _, _, _ = _gibbs_sampler(
        corr, beta_hat, n, h2, p,
        burn_in=burn_in, num_iter=num_iter, sparse=sparse, seed=seed,
        estimate_hyper=False, h2_bounds=(1e-6, 1.0), shrink_corr=shrink_corr,
        warm_start=warm_start, tol=tol, check_every=check_every,
        allow_jump_sign=allow_jump_sign, prior_w=prior_weights, slab_w=slab_w,
    )
    return avg_beta


@dataclass
class AutoResult:
    """Result of :func:`ldpred3_auto`.

    ``beta_est`` are the posterior-mean effects; ``h2_est`` / ``p_est`` the
    estimated heritability and causal fraction.
    """

    beta_est: np.ndarray = field(repr=False)
    h2_est: float = 0.0
    p_est: float = 0.0
    n_iter: int = 0
    h2_path: np.ndarray = field(default=None, repr=False)
    p_path: np.ndarray = field(default=None, repr=False)

    def __repr__(self):
        return (f"AutoResult(h2_est={self.h2_est:.3f}, p_est={self.p_est:.4g}, "
                f"n_iter={self.n_iter}, n_variants={len(self.beta_est)})")


def ldpred3_auto(corr, beta_hat, n_eff, *, h2_init=0.1, p_init=0.1,
                 burn_in=200, num_iter=200, shrink_corr=1.0,
                 h2_bounds=(1e-4, 1.0), warm_start=False, tol=0.0,
                 check_every=50, allow_jump_sign=True, prior_weights=None,
                 af=None, alpha=-1.0, seed=None):
    """LDpred3-auto: fit the point-normal model and estimate ``h2`` and ``p``.

    LDpred2-auto (Privé, Arbel & Vilhjálmsson, *Bioinformatics* 2020). Unlike
    :func:`ldpred3_grid`, no validation set is needed: the proportion of causal
    variants ``p`` and the SNP heritability ``h2`` are updated within the Gibbs
    sampler (each from its conjugate conditional) and their posterior means are
    returned alongside the effects. For the robust multi-chain estimator and the
    predictive-``r2`` / architecture inference built on the same chains
    (Privé et al., *AJHG* 2023), see :func:`ldpred3_auto_infer`.

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
    prior_weights : array_like, shape (m,), optional
        Per-variant inclusion-probability weights (see :func:`ldpred3_grid`).
    af, alpha :
        MAF-dependent slab-variance prior (see :func:`ldpred3_grid`).
    seed : int or None
        RNG seed.

    Returns
    -------
    AutoResult
        With fields ``beta_est``, ``h2_est``, ``p_est`` and the full sampling
        paths ``h2_path`` / ``p_path``.
    """
    _check_h2_p(h2=h2_init, p=p_init)
    lo, hi = h2_bounds
    if not (0 < lo <= hi):
        raise ValueError("h2_bounds must satisfy 0 < lower <= upper")
    if not isinstance(corr, SparseLD):
        corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    # `auto` estimates p from its Beta(1+#causal, 1+m-#causal) full conditional,
    # which is only conjugate when every SNP shares the inclusion probability p.
    # Non-uniform prior_weights make p_j = p*w_j, so that Beta draw is wrong. Use
    # ldpred3_grid (fixed p) or ldpred3_auto_annot (a learned, proper p-map) for
    # per-SNP inclusion priors. Uniform weights are fine (p_j = p).
    if prior_weights is not None:
        pw = np.asarray(prior_weights, dtype=float)
        if pw.size and not np.allclose(pw, pw.flat[0]):
            raise ValueError(
                "non-uniform prior_weights are not supported with ldpred3_auto "
                "(its p-update assumes a shared inclusion probability); use "
                "ldpred3_grid with fixed p, or ldpred3_auto_annot for a learned "
                "per-SNP inclusion prior")
    slab_w = maf_slab_weights(af, alpha) if af is not None and alpha != -1.0 else None
    avg_beta, h2_path, p_path, count = _gibbs_sampler(
        corr, beta_hat, n, h2_init, p_init,
        burn_in=burn_in, num_iter=num_iter, sparse=False, seed=seed,
        estimate_hyper=True, h2_bounds=h2_bounds, shrink_corr=shrink_corr,
        warm_start=warm_start, tol=tol, check_every=check_every,
        allow_jump_sign=allow_jump_sign, prior_w=prior_weights, slab_w=slab_w,
    )
    return AutoResult(
        beta_est=avg_beta,
        h2_est=float(np.mean(h2_path)),
        p_est=float(np.mean(p_path)),
        n_iter=int(count),
        h2_path=h2_path,
        p_path=p_path,
    )


def ldpred3_by_blocks(blocks, beta_hat, n_eff, method="auto",
                      sparsify=False, ld_threshold=1e-3, ld_max_dist=None,
                      global_hyper=True, ncores=1, **kwargs):
    """Apply an LDpred3 model independently to a list of LD blocks.

    Genome-wide LDpred3 treats (approximately) independent LD blocks
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
    method : {"inf", "grid", "auto", "laplace"}
        Which model to run per block. ``"laplace"`` is the Bayesian-lasso
        (Laplace-prior) posterior-mean sampler (dense blocks only).
    sparsify : bool
        If True, convert each dense block to a :class:`SparseLD` (via
        :func:`sparsify_ld` with ``ld_threshold`` / ``ld_max_dist``) before
        fitting, so the sampler/solver only touch non-zero LD entries. Blocks
        that are already ``SparseLD`` are used as-is.
    ld_threshold, ld_max_dist :
        Passed to :func:`sparsify_ld` when ``sparsify`` is True.
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

    funcs = {"inf": ldpred3_inf, "grid": ldpred3_grid, "auto": ldpred3_auto,
             "laplace": ldpred3_laplace}
    if method not in funcs:
        raise ValueError(f"method must be one of {sorted(funcs)}")

    # The MAF-dependent prior (af + alpha != -1) is per-block dense only; it needs
    # the per-block path (the streaming global-hyper kernel has no slab weights).
    af = kwargs.pop("af", None)
    alpha = kwargs.pop("alpha", -1.0)
    if af is not None:
        af = np.asarray(af, dtype=float)
        if alpha != -1.0 and method == "auto" and global_hyper:
            raise ValueError("the MAF-dependent prior (alpha != -1) requires "
                             "global_hyper=False (per-block) — the streaming "
                             "global-hyper sampler has no slab weights")

    # LDpred3-auto with GLOBAL hyper-parameters: assemble one block-diagonal
    # matrix and run a single auto fit, so h2 and p are estimated jointly across
    # all variants (matching bigsnpr) rather than noisily per block. Falls back
    # to per-block when global_hyper is off.
    if method == "auto" and global_hyper:
        if sparsify:
            raise NotImplementedError(
                "sparsify=True is not supported with global_hyper=True; "
                "use global_hyper=False or pass already-sparse blocks")
        kwargs.pop("h2", None)                       # auto estimates h2 itself
        blk = [(cb, np.asarray(idx)) for cb, idx in blocks]
        # The streaming/packed global path assumes blocks tile 0..m-1 contiguously.
        expected = 0
        for _, idx in sorted(blk, key=lambda bi: int(bi[1][0]) if bi[1].size else 0):
            if not np.array_equal(idx, np.arange(expected, expected + idx.shape[0])):
                raise ValueError("global_hyper=True requires contiguous blocks "
                                 "tiling 0..m-1; use global_hyper=False otherwise")
            expected += idx.shape[0]
        if expected != m:
            raise ValueError("blocks do not tile beta_hat (global_hyper=True)")
        common = dict(
            burn_in=kwargs.pop("burn_in", 200),
            num_iter=kwargs.pop("num_iter", 200), sparse=False,
            seed=kwargs.pop("seed", None), estimate_hyper=True,
            h2_bounds=kwargs.pop("h2_bounds", (1e-4, 1.0)),
            warm_start=kwargs.pop("warm_start", False),
            tol=kwargs.pop("tol", 0.0), check_every=kwargs.pop("check_every", 50))
        h2_init = kwargs.pop("h2_init", 0.1)
        p_init = kwargs.pop("p_init", 0.1)
        allow_jump_sign = kwargs.pop("allow_jump_sign", True)
        if kwargs:
            raise TypeError(
                f"unsupported keyword(s) for global_hyper auto: {sorted(kwargs)}. "
                "prior_weights / shrink_corr are only honoured with "
                "global_hyper=False (per-block); they would otherwise be "
                "silently ignored here.")
        has_special = any(isinstance(cb, (SparseLD, LowRankLD)) for cb, _ in blk)
        # The packed multicore kernel does not implement the sign-flip guard, so a
        # non-default allow_jump_sign uses the (single-core) streaming sampler.
        if ncores and ncores > 1 and not has_special and allow_jump_sign:
            # Multicore: packed blocks + prange (more memory, parallel sweeps).
            avg_beta, _, _, _ = _gibbs_blocks(blk, beta_hat, n, h2_init, p_init,
                                              ncores=ncores, **common)
        else:
            # Single core, or banded/low-rank blocks (the packed multicore kernel
            # is dense-only): streaming sampler -- O(k·bandwidth)/O(k·rank) memory.
            avg_beta, _, _, _ = _gibbs_blocks_stream(
                blk, beta_hat, n, h2_init, p_init,
                allow_jump_sign=allow_jump_sign, **common)
        return avg_beta

    # Total h2 (if given) is split across blocks proportionally to block size.
    total_h2 = kwargs.pop("h2", None)

    # laplace sets its shrinkage lambda from h2 (plug-in); a good genome-wide h2
    # matters, and per-block guesses are far too noisy. When the caller gives no
    # h2, estimate it once by LD Score regression (robust across power regimes)
    # rather than falling back to a fixed per-block default.
    if method == "laplace" and total_h2 is None:
        from .ldsc import ld_scores, ldsc_h2
        try:
            ell = ld_scores(blocks)
            total_h2 = float(ldsc_h2(n * beta_hat ** 2, ell, n, m_snps=m).h2)
        except Exception:
            total_h2 = None                          # fall back to the default
        if total_h2 is not None:
            total_h2 = min(max(total_h2, 1e-4), 1.0)

    for corr_block, idx in blocks:
        idx = np.asarray(idx)
        k = idx.shape[0]
        if sparsify and not isinstance(corr_block, SparseLD):
            corr_block = sparsify_ld(corr_block, threshold=ld_threshold,
                                     max_dist=ld_max_dist)
        block_kwargs = dict(kwargs)
        if method in ("inf", "grid", "laplace"):
            block_kwargs["h2"] = (total_h2 * k / m) if total_h2 is not None else 0.1
        if method == "laplace" and isinstance(corr_block, (SparseLD, LowRankLD)):
            raise ValueError("method='laplace' needs dense LD blocks "
                             "(not SparseLD / LowRankLD)")
        if af is not None and method in ("auto", "grid"):
            block_kwargs["af"] = af[idx]            # MAF-dependent slab, per block
            block_kwargs["alpha"] = alpha
        res = funcs[method](corr_block, beta_hat[idx], n[idx], **block_kwargs)
        out[idx] = res.beta_est if isinstance(res, AutoResult) else res
    return out
