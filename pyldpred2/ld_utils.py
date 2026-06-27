"""
LD representations and construction utilities for LDpred2.

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

__all__ = ["SparseLD", "sparsify_ld", "block_diagonal_ld", "optimal_ld_blocks"]


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
