"""
LD representations and construction utilities for LDpred3.

Holds the sparse-LD container and the routines that build / reshape LD matrices
independently of the samplers:

* :class:`SparseLD` -- a banded LD matrix in CSR form (the sampler only touches
  non-zero neighbours).
* :func:`sparsify_ld` -- threshold / band a dense LD matrix into a ``SparseLD``.
* :func:`block_diagonal_ld` -- pack per-block dense LD into one block-diagonal
  ``SparseLD``.
* :func:`optimal_ld_blocks` -- recombination-aware block splitting (Privé 2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._numba import _jit

__all__ = ["SparseLD", "sparsify_ld", "block_diagonal_ld", "optimal_ld_blocks",
           "shrink_ld_blocks", "LowRankLD", "lowrank_ld"]


def shrink_ld_blocks(blocks, n_ref, *, max_shrink=0.5, intensity=1.0, min_block=1):
    """Size-aware spectral shrinkage of per-block LD toward the identity.

    A block's sample LD estimated from a finite reference panel of ``n_ref``
    individuals carries noise that grows with the block size ``k`` relative to
    ``n_ref`` (Marchenko-Pastur: the noise in the eigenvalues scales with
    ``k / n_ref``). Small blocks (``k << n_ref``) are well estimated, but large
    blocks (``k`` approaching or exceeding ``n_ref``) are noise-dominated: the
    sample eigenvalues are inflated/spread, which makes the Gibbs sampler over-fit
    and inflates ``h2``.

    Each block is shrunk toward the identity by
    ``alpha = min(max_shrink, intensity * k / n_ref)`` --
    ``R <- (1 - alpha) R + alpha I`` with the diagonal kept at 1. Because
    ``alpha`` grows with ``k / n_ref``, this **regularises large blocks while
    leaving small, well-estimated ones essentially untouched** -- a uniform
    eigenvalue shrinkage (every eigenvalue ``lam -> (1-alpha) lam + alpha``) that,
    unlike low-rank PC truncation, does not preserve the Marchenko-Pastur-inflated
    top eigenvalues. Returns a new list of ``(R, idx)`` blocks (``SparseLD`` blocks
    are passed through unchanged).

    Parameters
    ----------
    blocks : list of (ndarray, idx)
        Dense per-block LD and the variants' positions.
    n_ref : int
        Number of individuals the reference LD was estimated from.
    max_shrink : float, default 0.5
        Cap on the per-block shrinkage intensity.
    intensity : float, default 1.0
        Scales the ``k / n_ref`` shrinkage (use < 1 to shrink more gently).
    min_block : int, default 1
        Blocks smaller than this are never shrunk.
    """
    if not n_ref or n_ref <= 0:
        return list(blocks)
    out = []
    for cb, idx in blocks:
        idx = np.asarray(idx)
        if isinstance(cb, SparseLD):
            out.append((cb, idx))
            continue
        k = idx.shape[0]
        alpha = min(max_shrink, intensity * k / float(n_ref)) if k >= min_block else 0.0
        if alpha > 0.0:
            R = np.asarray(cb, dtype=np.float64)
            R = (1.0 - alpha) * R + alpha * np.eye(k)
            np.fill_diagonal(R, 1.0)
            out.append((R.astype(np.float32), idx))
        else:
            out.append((np.asarray(cb, dtype=np.float32), idx))
    return out


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


@dataclass
class LowRankLD:
    """A low-rank LD approximation ``R ~= U @ U.T`` with unit diagonal.

    Realistic LD is close to low rank, so keeping the top eigenvectors captures
    it at a fraction of the memory and lets the sampler work in the r-dimensional
    eigenspace: the residual ``(R beta)`` is recovered from ``s = U.T beta``
    (length r) as ``U @ s``, and each effect update touches only ``s`` -- O(r)
    per SNP, O(k*r) memory, vs O(k^2) dense. ``U`` (k x r, float32) already
    absorbs ``sqrt(eigenvalue)`` and is row-scaled so ``(U U.T)_jj = 1``. Build
    with :func:`lowrank_ld`.
    """

    U: np.ndarray
    m: int

    @property
    def rank(self):
        return int(self.U.shape[1])

    @property
    def density(self):
        return self.U.size / float(self.m * self.m)


def lowrank_ld(corr, variance=0.99, max_rank=None, min_eig=1e-6):
    """Build a :class:`LowRankLD` from a dense LD matrix by eigen-truncation.

    Keeps the top eigenvectors until they explain ``variance`` of the total
    spectrum (capped at ``max_rank``), folds ``sqrt(eigenvalue)`` into ``U`` and
    row-scales so the reconstruction has unit diagonal. This is the SBayesRC-style
    low-rank LD: on realistic LD it matches the dense fit at a fraction of the
    memory (the spectrum is concentrated), whereas distance banding discards real
    long-range structure.

    Parameters
    ----------
    corr : ndarray (m, m)
        Dense symmetric LD matrix.
    variance : float, default 0.99
        Keep the fewest top eigenvectors explaining this fraction of ``sum(eig)``.
    max_rank : int or None
        Hard cap on the kept rank.
    min_eig : float
        Floor on kept eigenvalues (numerical safety; they are already > 0).
    """
    corr = np.asarray(corr, dtype=float)
    m = corr.shape[0]
    w, V = np.linalg.eigh(corr)
    w = w[::-1]; V = V[:, ::-1]                      # descending
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        r = 1
    else:
        r = int(np.searchsorted(np.cumsum(w), variance * total) + 1)
    r = max(1, min(r, m))
    if max_rank is not None:
        r = min(r, int(max_rank))
    wk = np.maximum(w[:r], min_eig)
    U = V[:, :r] * np.sqrt(wk)                       # R ~= U U^T
    d = np.sqrt(np.clip((U * U).sum(axis=1), 1e-12, None))
    U = U / d[:, None]                              # unit diagonal
    return LowRankLD(U.astype(np.float32), m)


def sparsify_ld(corr, threshold=1e-3, max_dist=None, shrink=1.0):
    """Build a :class:`SparseLD` from a dense LD matrix.

    Entries are kept when ``|r| >= threshold`` and (if ``max_dist`` is given)
    within ``max_dist`` SNPs of the diagonal; the diagonal is always kept. This
    mirrors how LDpred3 windows/thresholds LD in practice.

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
    genome-wide behaviour -- in particular LDpred3-auto then estimates ``h2`` and
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
