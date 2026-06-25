"""
Build LD (linkage-disequilibrium) correlation blocks from a genotype panel.

LDpred2 operates on per-block SNP correlation matrices. This module estimates
those from a reference panel of genotypes -- in-sample (the target cohort) for a
quick analysis, or an external panel passed separately. Variants are split into
contiguous blocks, never spanning a chromosome boundary, and within each block
the correlation is computed from standardized (mean-imputed, z-scored) dosages.

The returned ``blocks`` -- a list of ``(R, idx)`` pairs, ``R`` a ``float32``
correlation matrix and ``idx`` the column indices it covers -- plug directly
into :func:`ldpred2.ldpred2_by_blocks`.
"""

from __future__ import annotations

import numpy as np

from .prs import standardize_dosage

__all__ = ["compute_ld_blocks"]


def _block_bounds(chrom, block_size):
    """Yield (start, stop) column ranges of <= block_size, split by chromosome."""
    n = len(chrom)
    start = 0
    while start < n:
        c = chrom[start]
        stop = start
        while stop < n and chrom[stop] == c and (stop - start) < block_size:
            stop += 1
        yield start, stop
        start = stop


def compute_ld_blocks(dosage, *, chrom=None, block_size=500, ridge=0.0):
    """Estimate per-block LD correlation matrices from a genotype panel.

    Parameters
    ----------
    dosage : array_like, shape (n_ref, n_variants)
        Reference-panel dosages (``-1`` = missing), variants in genomic order.
    chrom : array_like, optional
        Per-variant chromosome labels; blocks never straddle a change in label.
        If omitted, all variants are treated as one chromosome.
    block_size : int, default 500
        Maximum variants per block.
    ridge : float, default 0.0
        If > 0, shrink each block towards the identity:
        ``R <- (1 - ridge) * R + ridge * I``. Guarantees positive-definiteness
        for downstream solvers when the panel has perfect-LD duplicates.

    Returns
    -------
    blocks : list of (ndarray float32, ndarray int)
        ``(R, idx)`` per block, ready for ``ldpred2_by_blocks``.
    """
    dosage = np.asarray(dosage)
    n_variants = dosage.shape[1]
    if chrom is None:
        chrom = np.zeros(n_variants, dtype=np.int8)
    else:
        chrom = np.asarray(chrom)
    if len(chrom) != n_variants:
        raise ValueError("chrom must have one label per variant")
    if not 0 < block_size:
        raise ValueError("block_size must be positive")

    blocks = []
    for start, stop in _block_bounds(chrom, block_size):
        idx = np.arange(start, stop)
        Z = standardize_dosage(dosage[:, start:stop])
        R = (Z.T @ Z) / Z.shape[0]
        if ridge > 0:
            R *= (1.0 - ridge)
            R[np.diag_indices_from(R)] += ridge
        np.fill_diagonal(R, 1.0)
        blocks.append((R.astype(np.float32), idx))
    return blocks
