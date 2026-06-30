"""LDpred3 — a NumPy-only implementation of LDpred2 and a full PRS pipeline.

Public API re-exported here for convenience::

    from ldpred3 import ldpred3_auto, run_ldpred3_prs, ldpred3_auto_infer
"""

from .ldpred3 import (
    standardize_betas,
    ldpred3_inf,
    ldpred3_grid,
    ldpred3_auto,
    ldpred3_by_blocks,
    AutoResult,
    SparseLD,
    sparsify_ld,
    block_diagonal_ld,
    optimal_ld_blocks,
    shrink_ld_blocks,
    LowRankLD,
    lowrank_ld,
)
from .pipeline import (
    run_ldpred3_prs,
    PRSResult,
    run_finemap,
    preflight_prs,
    score_from_weights,
    ScoreResult,
    load_genotypes,
)
from .infer import ldpred3_auto_infer, InferResult
from .annot import ldpred3_auto_annot, ldpred3_auto_annot_blocks, AnnotResult
from .ldsc import ld_scores, ldsc_h2, LDSCResult, ldsc_rg, LDSCRgResult
from .bivariate import (ldpred3_auto_bivariate, ldpred3_auto_bivariate_blocks,
                        BivariateResult)
from .finemap import (ldpred3_pip, single_signal_finemap, finemap_by_blocks,
                      FineMapResult, CredibleSet)

__all__ = [
    "standardize_betas",
    "ldpred3_inf",
    "ldpred3_grid",
    "ldpred3_auto",
    "ldpred3_by_blocks",
    "AutoResult",
    "SparseLD",
    "sparsify_ld",
    "block_diagonal_ld",
    "optimal_ld_blocks",
    "shrink_ld_blocks",
    "LowRankLD",
    "lowrank_ld",
    "run_ldpred3_prs",
    "run_finemap",
    "PRSResult",
    "preflight_prs",
    "score_from_weights",
    "ScoreResult",
    "load_genotypes",
    "ldpred3_auto_infer",
    "InferResult",
    "ldpred3_auto_annot",
    "ldpred3_auto_annot_blocks",
    "AnnotResult",
    "ld_scores",
    "ldsc_h2",
    "LDSCResult",
    "ldsc_rg",
    "LDSCRgResult",
    "ldpred3_auto_bivariate",
    "ldpred3_auto_bivariate_blocks",
    "BivariateResult",
    "ldpred3_pip",
    "single_signal_finemap",
    "finemap_by_blocks",
    "FineMapResult",
    "CredibleSet",
]

__version__ = "0.1.0"
