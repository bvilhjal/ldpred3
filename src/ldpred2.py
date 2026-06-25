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

# Optional Numba acceleration. The Gibbs sampler's inner sweep is pure Python
# and dominates runtime; JIT-compiling it gives a large speed-up. If Numba is
# not installed we fall back to a no-op decorator and run the identical code in
# pure Python (just slower).
try:
    from numba import njit as _njit

    HAVE_NUMBA = True

    def _jit(func):
        return _njit(cache=True)(func)

except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def _jit(func):
        return func


__all__ = [
    "standardize_betas",
    "ldpred2_inf",
    "ldpred2_grid",
    "ldpred2_auto",
    "ldpred2_by_blocks",
    "AutoResult",
    "SparseLD",
    "sparsify_ld",
    "block_diagonal_ld",
    "optimal_ld_blocks",
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


@dataclass
class SparseLD:
    """A symmetric LD matrix stored in compressed-sparse-row (CSR) form.

    Real LD is banded -- most off-diagonal entries are ~0 -- so storing only the
    non-zeros and updating only non-zero neighbours turns the sampler's hot-path
    rank-1 update from O(block_size) into O(bandwidth). Build one with
    :func:`sparsify_ld`. The diagonal (1.0) is always kept. Arrays use the
    layout numba expects: ``indptr``/``indices`` int32, ``data`` float32.
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    m: int

    @property
    def nnz(self):
        return int(self.data.shape[0])

    @property
    def density(self):
        return self.nnz / float(self.m * self.m)


def sparsify_ld(corr, threshold=1e-3, max_dist=None, shrink=1.0):
    """Build a :class:`SparseLD` from a dense LD matrix.

    Entries are kept when ``|r| >= threshold`` and (if ``max_dist`` is given)
    within ``max_dist`` SNPs of the diagonal; the diagonal is always kept. This
    mirrors how LDpred2 windows/thresholds LD in practice.

    Dropping off-diagonal entries (especially by hard distance banding) can make
    the matrix lose positive-definiteness, which **destabilises the Gibbs
    sampler** (effects can diverge with fixed ``h2``). Setting ``shrink`` < 1
    multiplies the kept off-diagonal entries by that factor (diagonal stays 1),
    restoring diagonal dominance / validity -- the standard regularisation for a
    windowed LD matrix. ``shrink`` is recommended whenever you band the LD before
    sampling; the infinitesimal solver is unaffected (its ridge handles it).

    Parameters
    ----------
    corr : ndarray, shape (m, m)
        Dense symmetric LD matrix.
    threshold : float
        Drop off-diagonal entries with absolute correlation below this.
    max_dist : int or None
        If set, also drop entries more than this many SNPs apart (banding).
    shrink : float
        Multiply kept off-diagonal entries by this (diagonal kept at 1.0).
    """
    corr = np.asarray(corr)
    m = corr.shape[0]
    indptr = np.zeros(m + 1, dtype=np.int32)
    idx_parts = []
    data_parts = []
    for j in range(m):
        row = corr[j]
        keep = np.abs(row) >= threshold
        if max_dist is not None:
            lo = max(0, j - max_dist)
            hi = min(m, j + max_dist + 1)
            band = np.zeros(m, dtype=bool)
            band[lo:hi] = True
            keep &= band
        keep[j] = True                      # always keep the diagonal
        cols = np.flatnonzero(keep)
        vals = row[cols].astype(np.float32)
        if shrink != 1.0:
            vals = vals * np.float32(shrink)
            vals[cols == j] = np.float32(1.0)   # restore the diagonal
        idx_parts.append(cols.astype(np.int32))
        data_parts.append(vals)
        indptr[j + 1] = indptr[j] + cols.shape[0]
    indices = (np.concatenate(idx_parts) if idx_parts
               else np.empty(0, np.int32)).astype(np.int32)
    data = (np.concatenate(data_parts) if data_parts
            else np.empty(0, np.float32)).astype(np.float32)
    return SparseLD(indptr, indices, data, m)


def block_diagonal_ld(blocks):
    """Assemble per-block dense LD matrices into one block-diagonal SparseLD.

    ``blocks`` is a sequence of ``(corr_block, idx)`` with ``corr_block`` a dense
    ``(k, k)`` LD matrix and ``idx`` the variants' global positions (the blocks
    must tile ``0 .. m-1``). Running a *single* model on the result gives
    genome-wide behaviour -- in particular LDpred2-auto then estimates ``h2`` and
    ``p`` jointly across all variants (global hyper-parameters), rather than
    independently per block. It is also one compiled call instead of one per
    block. Each block keeps all its entries (exact block-diagonal, so positive-
    definiteness is preserved).
    """
    blocks = sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0]))
    m = 0
    data_parts = []
    idx_parts = []
    row_nnz = []
    for corr_block, idx in blocks:
        idx = np.asarray(idx)
        k = idx.shape[0]
        if isinstance(corr_block, SparseLD):
            raise ValueError("block_diagonal_ld expects dense blocks")
        cb = np.asarray(corr_block, dtype=np.float32)
        data_parts.append(cb.ravel())                       # k*k, row-major
        idx_parts.append(np.tile(idx.astype(np.int32), k))  # block cols per row
        row_nnz.append(np.full(k, k, dtype=np.int64))
        m += k
    data = np.concatenate(data_parts).astype(np.float32)
    indices = np.concatenate(idx_parts).astype(np.int32)
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.concatenate(row_nnz), out=indptr[1:])
    return SparseLD(indptr.astype(np.int32), indices, data, m)


def _cost_diff_dense(corr, window, m):
    """Difference array whose prefix sum gives the per-boundary cut LD^2 (dense)."""
    diff = np.zeros(m + 1)
    for i in range(m):
        hi = i + window + 1
        if hi > m:
            hi = m
        for j in range(i + 1, hi):
            r = corr[i, j]
            r2 = r * r
            diff[i + 1] += r2
            diff[j + 1] -= r2
    return diff


def _cost_diff_sparse(indptr, indices, data, window, m):
    """Difference array of per-boundary cut LD^2 (sparse CSR)."""
    diff = np.zeros(m + 1)
    for i in range(m):
        for idx in range(indptr[i], indptr[i + 1]):
            j = indices[idx]
            if j > i and (j - i) <= window:
                r = data[idx]
                r2 = r * r
                diff[i + 1] += r2
                diff[j + 1] -= r2
    return diff


_cost_diff_dense_jit = _jit(_cost_diff_dense)
_cost_diff_sparse_jit = _jit(_cost_diff_sparse)


def optimal_ld_blocks(corr, max_size, min_size=1, window=None):
    """Split a region into near-independent LD blocks (Prive 2022, Bioinformatics).

    Implements the optimal LD-splitting idea behind bigsnpr's ``snp_ldsplit``:
    choose consecutive block boundaries that **minimise the total squared LD
    falling between blocks** (the LD discarded when blocks are treated as
    independent), subject to ``min_size <= block size <= max_size``. Boundaries
    therefore land in low-LD valleys (recombination hotspots), so a
    block-diagonal approximation loses far less LD than fixed-size blocks of the
    same maximum size -- improving accuracy and shrinking per-block storage.

    The cost of a boundary ``b`` is ``sum r^2`` over pairs ``(i, j)`` with
    ``i < b <= j`` and ``j - i <= window``. Total cost is additive over
    boundaries when ``min_size >= window`` (no LD pair straddles two boundaries),
    which an O(m) dynamic program then minimises exactly (a good approximation
    otherwise). Ref: https://doi.org/10.1093/bioinformatics/btab519

    Parameters
    ----------
    corr : ndarray (m, m) or SparseLD
        LD matrix for a single region / chromosome.
    max_size : int
        Maximum block size (variants).
    min_size : int
        Minimum block size. For an exact cost, use ``min_size >= window``.
    window : int or None
        LD window: the max ``|i-j|`` with non-negligible LD. Defaults to
        ``max_size``.

    Returns
    -------
    blocks : list of (start, end)
        Block boundaries (``end`` exclusive) tiling ``0 .. m``.
    cost : float
        Total discarded between-block LD^2 for the chosen split.
    """
    if window is None:
        window = max_size
    if isinstance(corr, SparseLD):
        m = corr.m
        diff = _cost_diff_sparse_jit(corr.indptr, corr.indices, corr.data,
                                     int(window), m)
    else:
        corr = np.ascontiguousarray(corr, dtype=np.float64)
        m = corr.shape[0]
        diff = _cost_diff_dense_jit(corr, int(window), m)
    if m == 0:
        return [], 0.0
    cost_cut = np.cumsum(diff)[:m + 1]
    cost_cut[m] = 0.0                      # closing the final block costs nothing
    max_size = min(int(max_size), m)
    min_size = max(1, min(int(min_size), max_size))

    # DP: E[b] = min cost to split [0, b) into valid blocks ending with a
    # boundary at b. Sliding-window minimum of E[a] over the allowed previous
    # boundaries a in [b - max_size, b - min_size].
    INF = np.inf
    E = np.full(m + 1, INF)
    E[0] = 0.0
    back = np.zeros(m + 1, dtype=np.int64)
    from collections import deque
    dq = deque()                           # indices a with increasing E[a]
    for b in range(1, m + 1):
        a_new = b - min_size
        if a_new >= 0 and E[a_new] < INF:
            while dq and E[dq[-1]] >= E[a_new]:
                dq.pop()
            dq.append(a_new)
        lo = b - max_size
        while dq and dq[0] < lo:
            dq.popleft()
        if dq:
            a = dq[0]
            E[b] = E[a] + cost_cut[b]
            back[b] = a

    # Backtrack the boundaries.
    bounds = []
    b = m
    while b > 0:
        a = int(back[b])
        bounds.append((a, b))
        b = a
    bounds.reverse()
    return bounds, float(E[m])


def ldpred2_inf(corr, beta_hat, n_eff, h2):
    """LDpred2 infinitesimal model (closed form).

    Assumes every variant is causal with effects drawn from
    ``beta ~ N(0, h2 / m)``. The posterior mean then has the closed form::

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
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    N = float(np.median(n))
    ridge = m / (h2 * N)

    if isinstance(corr, SparseLD):
        # Solve (R + ridge*I) x = beta_hat by conjugate gradient; the system is
        # SPD (R is a correlation matrix and ridge>0) and only needs matvecs.
        return _cg_solve(corr, ridge, beta_hat)

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
    Ap = np.empty(m)
    tol2 = tol * tol * float(b @ b)
    for _ in range(max_iter):
        _sparse_matvec_jit(indptr, indices, data, pvec, Ap)
        Ap += ridge * pvec
        alpha = rs / float(pvec @ Ap)
        x += alpha * pvec
        r -= alpha * Ap
        rs_new = float(r @ r)
        if rs_new <= tol2:
            break
        pvec = r + (rs_new / rs) * pvec
        rs = rs_new
    return x


def _gibbs_kernel(corr, beta_hat, n, h2, p, burn_in, num_iter, sparse,
                  estimate_hyper, h2_min, h2_max, seed, init_beta, tol,
                  check_every):
    """Numeric core of the point-normal Gibbs sampler (JIT-compiled if numba).

    Takes only plain numeric / array arguments so it compiles under
    ``numba.njit``. Uses the legacy global ``np.random`` (seeded here): its
    ``random`` / ``standard_normal`` streams are identical between the compiled
    and pure-Python paths; ``beta`` (used only for the -auto p-update) may differ
    slightly between the two but yields an equally valid sampler.

    ``init_beta`` warm-starts the chain (e.g. from LDpred2-inf). When ``tol`` > 0,
    the sampler stops early once the running posterior mean's relative RMS change
    over ``check_every`` sweeps falls below ``tol`` (adaptive stopping).

    Returns ``(avg_beta, h2_path, p_path, count)``; ``count`` is the number of
    post-burn-in sweeps actually used, and the paths are truncated to it.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]

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
        # Variance of a causal effect under the slab.
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)    # constant within iteration
        # Per-SNP posterior quantities are vectors of length m.
        post_var = c1 / (n * c1 + 1.0)               # = 1 / (n + 1/c1)
        post_sd = np.sqrt(post_var)
        half_log_term = 0.5 * np.log1p(n * c1)
        n_post_var = n * post_var                    # post_mean = this * residual
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

            pv = post_var[j]
            post_mean = n_post_var[j] * res_beta_j

            # Posterior inclusion probability via the log-odds of null vs causal.
            log_odds = (log_prior_odds + half_log_term[j]
                        - 0.5 * post_mean * post_mean / pv)
            postp = 1.0 / (1.0 + np.exp(log_odds))

            # Rao-Blackwellized estimate: accumulate the conditional posterior
            # mean E[beta_j | rest] = postp * post_mean rather than the sampled
            # value. Same expectation, lower Monte-Carlo variance (as in the
            # original LDpred). The *sampled* value below still drives the chain.
            post_means[j] = postp * post_mean

            if sparse and postp < 0.5:
                new = 0.0
            elif unif[j] < postp:
                new = post_mean + gauss[j] * post_sd[j]
                nb_causal += 1
            else:
                new = 0.0

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


def _gibbs_kernel_sparse(indptr, indices, data, beta_hat, n, h2, p, burn_in,
                         num_iter, sparse, estimate_hyper, h2_min, h2_max, seed,
                         init_beta, tol, check_every):
    """Sparse (CSR) counterpart of :func:`_gibbs_kernel`.

    Identical point-normal Gibbs / Rao-Blackwellized sampler (incl. warm start
    and adaptive stopping), but the LD matrix is stored as CSR
    (``indptr``/``indices``/``data``), so the rank-1 update and the resync touch
    only the non-zero neighbours of each SNP -- O(bandwidth) rather than O(m).
    The diagonal must be present in the CSR structure.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]

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
        log_prior_odds = np.log1p(-p) - np.log(p)
        post_var = c1 / (n * c1 + 1.0)
        post_sd = np.sqrt(post_var)
        half_log_term = 0.5 * np.log1p(n * c1)
        n_post_var = n * post_var
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

            pv = post_var[j]
            post_mean = n_post_var[j] * res_beta_j
            log_odds = (log_prior_odds + half_log_term[j]
                        - 0.5 * post_mean * post_mean / pv)
            postp = 1.0 / (1.0 + np.exp(log_odds))
            post_means[j] = postp * post_mean

            if sparse and postp < 0.5:
                new = 0.0
            elif unif[j] < postp:
                new = post_mean + gauss[j] * post_sd[j]
                nb_causal += 1
            else:
                new = 0.0

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


def _gibbs_sampler(corr, beta_hat, n, h2, p, *, burn_in, num_iter, sparse,
                   seed, estimate_hyper, h2_bounds, shrink_corr,
                   warm_start=False, tol=0.0, check_every=50):
    """Prepare arguments and dispatch to the (optionally JIT-compiled) kernel.

    ``corr`` may be a dense ndarray or a :class:`SparseLD`; the matching dense or
    sparse kernel is used. With ``warm_start`` the chain is initialised from the
    LDpred2-inf solution; with ``tol`` > 0 the sampler stops early once the
    running estimate converges. Returns ``(avg_beta, h2_path, p_path, count)``.
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    h2_min, h2_max = h2_bounds
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]

    # Warm start from the (cheap) infinitesimal solution, else cold start.
    if warm_start:
        init_beta = np.ascontiguousarray(
            ldpred2_inf(corr, beta_hat, n, h2), dtype=np.float64)
    else:
        init_beta = np.zeros(beta_hat.shape[0])

    if isinstance(corr, SparseLD):
        if shrink_corr != 1.0:
            raise ValueError("shrink_corr is only supported for dense LD")
        return _gibbs_kernel_sparse_jit(
            corr.indptr, corr.indices, corr.data, beta_hat, n, float(h2),
            float(p), int(burn_in), int(num_iter), bool(sparse),
            bool(estimate_hyper), float(h2_min), float(h2_max), int(seed),
            init_beta, float(tol), int(check_every),
        )

    # Single-precision, contiguous LD matrix. ``corr`` is symmetric, so row j
    # (a contiguous slice) is also column j -- used for the rank-1 update. float32
    # halves the memory traffic of that (bandwidth-bound) hot loop with no
    # meaningful accuracy cost; it also matches bigsnpr, which stores LD in single
    # precision. The accumulator Rb and effects stay in float64.
    corr = np.ascontiguousarray(corr, dtype=np.float32)

    # Optionally shrink off-diagonal LD to stabilise the sampler (LDpred2
    # exposes this; default 1.0 = no shrinkage). Copy so the caller's matrix is
    # untouched.
    if shrink_corr != 1.0:
        corr = corr * np.float32(shrink_corr)
        np.fill_diagonal(corr, np.float32(1.0))

    return _gibbs_kernel_jit(
        corr, beta_hat, n, float(h2), float(p), int(burn_in), int(num_iter),
        bool(sparse), bool(estimate_hyper), float(h2_min), float(h2_max),
        int(seed), init_beta, float(tol), int(check_every),
    )


def ldpred2_grid(corr, beta_hat, n_eff, h2, p, *, burn_in=100, num_iter=400,
                 sparse=False, shrink_corr=1.0, warm_start=False, tol=0.0,
                 check_every=50, seed=None):
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
    if not isinstance(corr, SparseLD):
        corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    avg_beta, _, _, _ = _gibbs_sampler(
        corr, beta_hat, n, h2, p,
        burn_in=burn_in, num_iter=num_iter, sparse=sparse, seed=seed,
        estimate_hyper=False, h2_bounds=(1e-6, 1.0), shrink_corr=shrink_corr,
        warm_start=warm_start, tol=tol, check_every=check_every,
    )
    return avg_beta


@dataclass
class AutoResult:
    """Result of :func:`ldpred2_auto`."""

    beta_est: np.ndarray
    h2_est: float
    p_est: float
    n_iter: int = 0
    h2_path: np.ndarray = field(default=None, repr=False)
    p_path: np.ndarray = field(default=None, repr=False)


def ldpred2_auto(corr, beta_hat, n_eff, *, h2_init=0.1, p_init=0.1,
                 burn_in=200, num_iter=200, shrink_corr=1.0,
                 h2_bounds=(1e-4, 1.0), warm_start=False, tol=0.0,
                 check_every=50, seed=None):
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
    if not isinstance(corr, SparseLD):
        corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    avg_beta, h2_path, p_path, count = _gibbs_sampler(
        corr, beta_hat, n, h2_init, p_init,
        burn_in=burn_in, num_iter=num_iter, sparse=False, seed=seed,
        estimate_hyper=True, h2_bounds=h2_bounds, shrink_corr=shrink_corr,
        warm_start=warm_start, tol=tol, check_every=check_every,
    )
    return AutoResult(
        beta_est=avg_beta,
        h2_est=float(np.mean(h2_path)),
        p_est=float(np.mean(p_path)),
        n_iter=int(count),
        h2_path=h2_path,
        p_path=p_path,
    )


def ldpred2_by_blocks(blocks, beta_hat, n_eff, method="auto",
                      sparsify=False, ld_threshold=1e-3, ld_max_dist=None,
                      global_hyper=True, **kwargs):
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

    funcs = {"inf": ldpred2_inf, "grid": ldpred2_grid, "auto": ldpred2_auto}
    if method not in funcs:
        raise ValueError(f"method must be one of {sorted(funcs)}")

    # LDpred2-auto with GLOBAL hyper-parameters: assemble one block-diagonal
    # matrix and run a single auto fit, so h2 and p are estimated jointly across
    # all variants (matching bigsnpr) rather than noisily per block. Falls back
    # to per-block when global_hyper is off.
    if method == "auto" and global_hyper:
        ld = block_diagonal_ld([(cb, np.asarray(idx)) for cb, idx in blocks])
        kwargs.pop("h2", None)                       # auto estimates h2 itself
        return ldpred2_auto(ld, beta_hat, n, **kwargs).beta_est

    # Total h2 (if given) is split across blocks proportionally to block size.
    total_h2 = kwargs.pop("h2", None)

    for corr_block, idx in blocks:
        idx = np.asarray(idx)
        k = idx.shape[0]
        if sparsify and not isinstance(corr_block, SparseLD):
            corr_block = sparsify_ld(corr_block, threshold=ld_threshold,
                                     max_dist=ld_max_dist)
        block_kwargs = dict(kwargs)
        if method in ("inf", "grid"):
            block_kwargs["h2"] = (total_h2 * k / m) if total_h2 is not None else 0.1
        res = funcs[method](corr_block, beta_hat[idx], n[idx], **block_kwargs)
        out[idx] = res.beta_est if isinstance(res, AutoResult) else res
    return out
