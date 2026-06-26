"""pyLDpred2 — a NumPy-only LDpred2 implementation and PRS pipeline.

Public API re-exported here for convenience::

    from pyldpred2 import ldpred2_auto, run_ldpred2_prs, ldpred2_auto_infer
"""

from .ldpred2 import (
    standardize_betas,
    ldpred2_inf,
    ldpred2_grid,
    ldpred2_auto,
    ldpred2_by_blocks,
    AutoResult,
    SparseLD,
    sparsify_ld,
    block_diagonal_ld,
    optimal_ld_blocks,
)
from .pipeline import (
    run_ldpred2_prs,
    PRSResult,
    preflight_prs,
    score_from_weights,
    ScoreResult,
    load_genotypes,
)
from .infer import ldpred2_auto_infer, InferResult
from .annot import ldpred2_auto_annot, ldpred2_auto_annot_blocks, AnnotResult

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
    "run_ldpred2_prs",
    "PRSResult",
    "preflight_prs",
    "score_from_weights",
    "ScoreResult",
    "load_genotypes",
    "ldpred2_auto_infer",
    "InferResult",
    "ldpred2_auto_annot",
    "ldpred2_auto_annot_blocks",
    "AnnotResult",
]

__version__ = "0.1.0"
