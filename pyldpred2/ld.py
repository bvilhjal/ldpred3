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

__all__ = ["compute_ld_blocks", "save_ld_blocks", "load_ld_blocks"]


def save_ld_blocks(path, blocks, variant_ids):
    """Save computed LD ``blocks`` and the variant IDs they cover to ``path``.

    ``blocks`` is the ``[(R, idx), ...]`` list from :func:`compute_ld_blocks`;
    ``variant_ids`` are the IDs of the variants in column order (one per
    column the blocks tile). Stored as a compressed ``.npz`` so a later run can
    reload the LD instead of recomputing it (see :func:`load_ld_blocks`).
    """
    ids = np.asarray(variant_ids, dtype=object).astype(str)
    sizes = np.array([R.shape[0] for R, _ in blocks], dtype=np.int64)
    if int(sizes.sum()) != len(ids):
        raise ValueError("variant_ids length does not match the blocks' columns")
    arrays = {f"R{i}": np.asarray(R, dtype=np.float32)
              for i, (R, _) in enumerate(blocks)}
    np.savez_compressed(path, ids=ids, sizes=sizes, **arrays)


def load_ld_blocks(path):
    """Load LD blocks saved by :func:`save_ld_blocks`.

    Returns ``(blocks, variant_ids)`` with ``blocks`` a ``[(R, idx), ...]`` list
    (contiguous ``idx`` reconstructed from the stored block sizes) ready for
    :func:`ldpred2.ldpred2_by_blocks`, and ``variant_ids`` the column-order IDs
    the caller should align its summary statistics to.
    """
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        sizes = z["sizes"]
        blocks, start = [], 0
        for i, k in enumerate(sizes):
            k = int(k)
            R = z[f"R{i}"]
            blocks.append((R, np.arange(start, start + k)))
            start += k
    return blocks, ids


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
