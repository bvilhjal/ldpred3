"""
Build LD (linkage-disequilibrium) correlation blocks from a genotype panel.

LDpred3 operates on per-block SNP correlation matrices. This module estimates
those from a reference panel of genotypes -- in-sample (the target cohort) for a
quick analysis, or an external panel passed separately. Variants are split into
contiguous blocks, never spanning a chromosome boundary, and within each block
the correlation is computed from standardized (mean-imputed, z-scored) dosages.

The returned ``blocks`` -- a list of ``(R, idx)`` pairs, ``R`` a ``float32``
correlation matrix and ``idx`` the column indices it covers -- plug directly
into :func:`ldpred3.ldpred3_by_blocks`.
"""

from __future__ import annotations

import numpy as np

from .prs import standardize_dosage
from .ld_utils import sparsify_ld, SparseLD, lowrank_ld, LowRankLD

__all__ = ["compute_ld_blocks", "save_ld_blocks", "load_ld_blocks"]


def save_ld_blocks(path, blocks, variant_ids):
    """Save computed LD ``blocks`` and the variant IDs they cover to ``path``.

    ``blocks`` is the ``[(R, idx), ...]`` list from :func:`compute_ld_blocks`;
    ``variant_ids`` are the IDs of the variants in column order (one per
    column the blocks tile). Stored as a compressed ``.npz`` so a later run can
    reload the LD instead of recomputing it (see :func:`load_ld_blocks`).
    """
    ids = np.asarray(variant_ids, dtype=object).astype(str)
    sizes, kinds, arrays = [], [], {}
    for i, (R, _) in enumerate(blocks):
        if isinstance(R, SparseLD):           # banded CSR (memory-efficient on disk)
            kinds.append(1); sizes.append(R.m)
            arrays[f"R{i}_indptr"] = R.indptr
            arrays[f"R{i}_indices"] = R.indices
            arrays[f"R{i}_data"] = R.data
        elif isinstance(R, LowRankLD):        # low-rank factor U (k x r)
            kinds.append(2); sizes.append(R.m)
            arrays[f"R{i}_U"] = R.U
        else:
            kinds.append(0); sizes.append(R.shape[0])
            arrays[f"R{i}"] = np.asarray(R, dtype=np.float32)
    sizes = np.array(sizes, dtype=np.int64)
    if int(sizes.sum()) != len(ids):
        raise ValueError("variant_ids length does not match the blocks' columns")
    np.savez_compressed(path, ids=ids, sizes=sizes,
                        kinds=np.array(kinds, dtype=np.int8), **arrays)


def load_ld_blocks(path):
    """Load LD blocks saved by :func:`save_ld_blocks`.

    Returns ``(blocks, variant_ids)`` with ``blocks`` a ``[(R, idx), ...]`` list
    (contiguous ``idx`` reconstructed from the stored block sizes) ready for
    :func:`ldpred3.ldpred3_by_blocks`, and ``variant_ids`` the column-order IDs
    the caller should align its summary statistics to.
    """
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        sizes = z["sizes"]
        kinds = z["kinds"] if "kinds" in z else np.zeros(len(sizes), np.int8)
        blocks, start = [], 0
        for i, k in enumerate(sizes):
            k = int(k)
            if int(kinds[i]) == 1:
                R = SparseLD(z[f"R{i}_indptr"], z[f"R{i}_indices"],
                             z[f"R{i}_data"], k)
            elif int(kinds[i]) == 2:
                R = LowRankLD(z[f"R{i}_U"], k)
            else:
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


def compute_ld_blocks(dosage, *, chrom=None, block_size=500, ridge=0.0,
                      sparse=False, ld_threshold=1e-3, max_dist=None,
                      lowrank=False, lowrank_variance=0.99, lowrank_max_rank=None):
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
    sparse : bool, default False
        If True, store each block as a banded :class:`~ldpred3.SparseLD`
        (thresholded at ``ld_threshold`` and, if ``max_dist`` is set, banded to
        that window) instead of a dense matrix. The dense block is built
        transiently and discarded, so **persistent** memory is O(k·bandwidth)
        rather than O(k²) -- essential for large blocks (thousands of SNPs) at
        genome scale. The sampler consumes these via ``global_hyper=False`` (the
        dense global-hyper path requires dense blocks).
    ld_threshold : float, default 1e-3
        Drop off-diagonal entries with ``|r| < ld_threshold`` (sparse only).
    max_dist : int or None
        If set, also band each block to ``|i-j| <= max_dist`` (sparse only).
    lowrank : bool, default False
        If True, store each block as a :class:`~ldpred3.LowRankLD` (top
        eigenvectors to ``lowrank_variance`` of the spectrum). Persistent memory
        is O(k·rank); on **realistic** LD this matches the dense fit at a fraction
        of the memory (preferred over banding, which discards long-range LD).
        Fit via ``global_hyper`` auto (the eigenspace streaming kernel).
    lowrank_variance : float, default 0.99
        Spectrum fraction to keep (low-rank only).
    lowrank_max_rank : int or None
        Hard cap on the kept rank per block (low-rank only).

    Returns
    -------
    blocks : list of (R, idx)
        ``R`` is a ``float32`` dense matrix, or a ``SparseLD`` when
        ``sparse=True``; ``idx`` are the column indices the block covers.
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
        if sparse and lowrank:
            raise ValueError("use either sparse or lowrank, not both")
        if lowrank:
            # Build dense transiently, store top-rank eigizmodes (O(k*rank)).
            blocks.append((lowrank_ld(R, variance=lowrank_variance,
                                      max_rank=lowrank_max_rank), idx))
        elif sparse:
            # Build dense transiently, store banded -> persistent O(k*bandwidth).
            blocks.append((sparsify_ld(R, threshold=ld_threshold,
                                       max_dist=max_dist), idx))
        else:
            blocks.append((R.astype(np.float32), idx))
    return blocks
