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
    from numba import njit as _njit, prange

    HAVE_NUMBA = True

    def _jit(func):
        return _njit(cache=True)(func)

    def _jit_parallel(func):
        return _njit(cache=True, parallel=True)(func)

    def _set_threads(ncores):
        from numba import set_num_threads
        set_num_threads(int(ncores))

except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False
    prange = range

    def _jit(func):
        return func

    def _jit_parallel(func):
        return func

    def _set_threads(ncores):
        pass


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
    if np.any(beta_se < 0):
        raise ValueError("beta_se must be non-negative")
    if np.any(n_eff <= 0):
        raise ValueError("n_eff must be positive")
    scale = np.sqrt(n_eff * beta_se ** 2 + beta ** 2)
    # Guard the degenerate beta == 0 and beta_se == 0 case (scale == 0) -> 0.
    beta_std = np.divide(beta, scale, out=np.zeros_like(beta, dtype=float),
                         where=scale > 0)
    return beta_std, scale


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
        warnings.warn(f"ldpred2_inf conjugate gradient did not converge in "
                      f"{max_iter} iterations (residual {rs_new ** 0.5:.2e})",
                      RuntimeWarning)
    return x


def _gibbs_kernel(corr, beta_hat, n, h2, p, burn_in, num_iter, sparse,
                  estimate_hyper, h2_min, h2_max, seed, init_beta, tol,
                  check_every, allow_jump_sign):
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
            postp = _stable_postp(log_odds)

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

            # Robustness guard (Privé et al.): forbid an effect flipping sign in
            # one step -- a major source of divergence on noisy / ill-conditioned
            # LD. The proposal is set to zero instead.
            if (not allow_jump_sign and old != 0.0 and new != 0.0
                    and (new > 0.0) != (old > 0.0)):
                new = 0.0
                nb_causal -= 1

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
                         h2_min, h2_max, seed, sample_every, allow_jump_sign):
    """Auto Gibbs kernel that also retains thinned *sampled* effect vectors.

    A trimmed copy of :func:`_gibbs_kernel` (dense, ``estimate_hyper`` always on,
    no sparse / warm-start / adaptive-stop branches) that, in addition to the
    Rao-Blackwellized posterior mean and the ``h2``/``p`` paths, stores the
    sampled ``curr_beta`` every ``sample_every`` post-burn-in sweeps. Those
    samples drive the LDpred2-auto predictive-r2 estimator (Privé et al. 2023).

    Returns ``(avg_beta, h2_path, p_path, beta_samples, n_saved)``.
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]
    curr_beta = np.zeros(m)
    avg_beta = np.zeros(m)
    post_means = np.zeros(m)
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
        post_var = c1 / (n * c1 + 1.0)
        post_sd = np.sqrt(post_var)
        half_log_term = 0.5 * np.log1p(n * c1)
        n_post_var = n * post_var
        nb_causal = 0

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
            pv = post_var[j]
            post_mean = n_post_var[j] * res_beta_j
            log_odds = (log_prior_odds + half_log_term[j]
                        - 0.5 * post_mean * post_mean / pv)
            postp = _stable_postp(log_odds)
            post_means[j] = postp * post_mean
            if unif[j] < postp:
                new = post_mean + gauss[j] * post_sd[j]
                nb_causal += 1
            else:
                new = 0.0
            if (not allow_jump_sign and old != 0.0 and new != 0.0
                    and (new > 0.0) != (old > 0.0)):
                new = 0.0
                nb_causal -= 1
            delta = new - old
            if delta != 0.0:
                cj = corr[j]
                for i in range(m):
                    Rb[i] += cj[i] * delta
                curr_beta[j] = new

        p = np.random.beta(1.0 + nb_causal, 1.0 + m - nb_causal)
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
    return avg_beta, h2_path[:count], p_path[:count], beta_samples[:n_saved], n_saved


_gibbs_kernel_sample_jit = _jit(_gibbs_kernel_sample)


def _gibbs_kernel_sparse(indptr, indices, data, beta_hat, n, h2, p, burn_in,
                         num_iter, sparse, estimate_hyper, h2_min, h2_max, seed,
                         init_beta, tol, check_every, allow_jump_sign):
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
            postp = _stable_postp(log_odds)
            post_means[j] = postp * post_mean

            if sparse and postp < 0.5:
                new = 0.0
            elif unif[j] < postp:
                new = post_mean + gauss[j] * post_sd[j]
                nb_causal += 1
            else:
                new = 0.0

            if (not allow_jump_sign and old != 0.0 and new != 0.0
                    and (new > 0.0) != (old > 0.0)):
                new = 0.0
                nb_causal -= 1

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

    # When N is constant the per-SNP posterior constants collapse to scalars;
    # computing them once per sweep avoids m sqrt/log calls and length-m
    # allocations -- a big saving at genome scale. Vectors are kept for the rare
    # per-variant-N case.
    post_var = np.zeros(m)
    post_sd = np.zeros(m)
    half_log_term = np.zeros(m)
    n_post_var = np.zeros(m)

    for it in range(n_iter_total):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        if n_const:
            nc1 = n[0] * c1
            pv0 = c1 / (nc1 + 1.0)
            psd0 = np.sqrt(pv0)
            half0 = 0.5 * np.log1p(nc1)
            npv0 = n[0] * pv0
        else:
            post_var = c1 / (n * c1 + 1.0)
            post_sd = np.sqrt(post_var)
            half_log_term = 0.5 * np.log1p(n * c1)
            n_post_var = n * post_var
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
                if n_const:
                    pv = pv0
                    psd = psd0
                    half = half0
                    post_mean = npv0 * res_beta_j
                else:
                    pv = post_var[gj]
                    psd = post_sd[gj]
                    half = half_log_term[gj]
                    post_mean = n_post_var[gj] * res_beta_j
                log_odds = (log_prior_odds + half
                            - 0.5 * post_mean * post_mean / pv)
                postp = _stable_postp(log_odds)
                post_means[gj] = postp * post_mean

                if sparse and postp < 0.5:
                    new = 0.0
                elif unif[gj] < postp:
                    new = post_mean + gauss[gj] * psd
                    nb_causal += 1
                else:
                    new = 0.0

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
    post_var = np.zeros(m)
    post_sd = np.zeros(m)
    half_log_term = np.zeros(m)
    n_post_var = np.zeros(m)

    for it in range(n_iter_total):
        c1 = h2 / (m * p)
        log_prior_odds = np.log1p(-p) - np.log(p)
        if n_const:
            nc1 = n[0] * c1
            pv0 = c1 / (nc1 + 1.0)
            psd0 = np.sqrt(pv0)
            half0 = 0.5 * np.log1p(nc1)
            npv0 = n[0] * pv0
        else:
            post_var = c1 / (n * c1 + 1.0)
            post_sd = np.sqrt(post_var)
            half_log_term = 0.5 * np.log1p(n * c1)
            n_post_var = n * post_var

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
                if n_const:
                    pv = pv0
                    psd = psd0
                    half = half0
                    post_mean = npv0 * res_beta_j
                else:
                    pv = post_var[gj]
                    psd = post_sd[gj]
                    half = half_log_term[gj]
                    post_mean = n_post_var[gj] * res_beta_j
                log_odds = (log_prior_odds + half
                            - 0.5 * post_mean * post_mean / pv)
                postp = _stable_postp(log_odds)
                post_means[gj] = postp * post_mean

                if sparse and postp < 0.5:
                    new = 0.0
                elif unif[gj] < postp:
                    new = post_mean + gauss[gj] * psd
                    nb_causal += 1
                else:
                    new = 0.0

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
            init_beta[idx] = ldpred2_inf(cb, beta_hat[idx], n[idx], h2 * k / m)
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


def _gibbs_one_sweep(corr, beta_hat, n, curr_beta, Rb, post_means, unif, gauss,
                     c1, log_prior_odds, sparse, n_const, n0, resync):
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

    if n_const:
        nc1 = n0 * c1
        pv0 = c1 / (nc1 + 1.0)
        psd0 = np.sqrt(pv0)
        half0 = 0.5 * np.log1p(nc1)
        npv0 = n0 * pv0

    nb_causal = 0
    for lj in range(k):
        old = curr_beta[lj]
        res_beta_j = beta_hat[lj] - Rb[lj] + old
        if n_const:
            pv = pv0
            psd = psd0
            half = half0
            post_mean = npv0 * res_beta_j
        else:
            nj = n[lj]
            nc1 = nj * c1
            pv = c1 / (nc1 + 1.0)
            psd = np.sqrt(pv)
            half = 0.5 * np.log1p(nc1)
            post_mean = nj * pv * res_beta_j
        log_odds = log_prior_odds + half - 0.5 * post_mean * post_mean / pv
        postp = _stable_postp(log_odds)
        post_means[lj] = postp * post_mean

        if sparse and postp < 0.5:
            new = 0.0
        elif unif[lj] < postp:
            new = post_mean + gauss[lj] * psd
            nb_causal += 1
        else:
            new = 0.0

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


def _gibbs_blocks_stream(blocks, beta_hat, n, h2, p, *, burn_in, num_iter,
                         sparse, seed, estimate_hyper, h2_bounds,
                         warm_start=False, tol=0.0, check_every=50):
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

    # float32 blocks, de-duplicated by object identity (shared blocks -> 1 copy).
    cache = {}
    fblocks = []
    for cb, idx in sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0])):
        key = id(cb)
        if key not in cache:
            cache[key] = np.ascontiguousarray(cb, dtype=np.float32)
        idx = np.asarray(idx)
        fblocks.append((cache[key], int(idx[0]), int(idx.shape[0])))

    curr_beta = np.zeros(m)
    if warm_start:
        for cbf, start, k in fblocks:
            sl = slice(start, start + k)
            curr_beta[sl] = ldpred2_inf(cbf, beta_hat[sl], n[sl], h2 * k / m)
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
        for cbf, start, k in fblocks:
            sl = slice(start, start + k)
            nbc, gvb = _gibbs_one_sweep_jit(
                cbf, beta_hat[sl], n[sl], curr_beta[sl], Rb[sl], post_means[sl],
                unif[sl], gauss[sl], c1, log_prior_odds, bool(sparse),
                n_const, n0, resync)
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


def _gibbs_sampler(corr, beta_hat, n, h2, p, *, burn_in, num_iter, sparse,
                   seed, estimate_hyper, h2_bounds, shrink_corr,
                   warm_start=False, tol=0.0, check_every=50,
                   allow_jump_sign=True):
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
            init_beta, float(tol), int(check_every), bool(allow_jump_sign),
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
        bool(allow_jump_sign),
    )


def ldpred2_grid(corr, beta_hat, n_eff, h2, p, *, burn_in=100, num_iter=400,
                 sparse=False, shrink_corr=1.0, warm_start=False, tol=0.0,
                 check_every=50, allow_jump_sign=True, seed=None):
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
    _check_h2_p(h2=h2, p=p)
    if not isinstance(corr, SparseLD):
        corr = np.asarray(corr, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    avg_beta, _, _, _ = _gibbs_sampler(
        corr, beta_hat, n, h2, p,
        burn_in=burn_in, num_iter=num_iter, sparse=sparse, seed=seed,
        estimate_hyper=False, h2_bounds=(1e-6, 1.0), shrink_corr=shrink_corr,
        warm_start=warm_start, tol=tol, check_every=check_every,
        allow_jump_sign=allow_jump_sign,
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
                 check_every=50, allow_jump_sign=True, seed=None):
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
    _check_h2_p(h2=h2_init, p=p_init)
    lo, hi = h2_bounds
    if not (0 < lo <= hi):
        raise ValueError("h2_bounds must satisfy 0 < lower <= upper")
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
        allow_jump_sign=allow_jump_sign,
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
                      global_hyper=True, ncores=1, **kwargs):
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
        if ncores and ncores > 1:
            # Multicore: packed blocks + prange (more memory, parallel sweeps).
            avg_beta, _, _, _ = _gibbs_blocks(blk, beta_hat, n, h2_init, p_init,
                                              ncores=ncores, **common)
        else:
            # Single core: streaming sampler (low memory).
            avg_beta, _, _, _ = _gibbs_blocks_stream(blk, beta_hat, n, h2_init,
                                                     p_init, **common)
        return avg_beta

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
