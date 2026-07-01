"""LDpred3 — a NumPy-only implementation of LDpred2 and a full PRS pipeline.

Public API re-exported here for convenience::

    from ldpred3 import ldpred3_auto, run_ldpred3_prs, ldpred3_auto_infer

Names are imported **lazily** (PEP 562): ``import ldpred3`` does not pull in the
pipeline / IO / analysis submodules until you first touch one of their symbols,
so importing just the core sampler stays cheap and avoids optional coupling.
"""

import importlib

__version__ = "0.2.0"

# public name -> submodule it lives in
_EXPORTS = {
    "ldpred3": [
        "standardize_betas", "ldpred3_inf", "ldpred3_grid", "ldpred3_auto",
        "ldpred3_laplace", "ldpred3_by_blocks", "maf_slab_weights", "AutoResult",
        "SparseLD", "sparsify_ld", "block_diagonal_ld", "optimal_ld_blocks",
        "shrink_ld_blocks", "LowRankLD", "lowrank_ld",
    ],
    "pipeline": [
        "run_ldpred3_prs", "PRSResult", "run_finemap", "preflight_prs",
        "score_from_weights", "ScoreResult", "load_genotypes",
    ],
    "infer": ["ldpred3_auto_infer", "InferResult"],
    "annot": ["ldpred3_auto_annot", "ldpred3_auto_annot_blocks", "AnnotResult"],
    "ldsc": ["ld_scores", "ldsc_h2", "LDSCResult", "ldsc_rg", "LDSCRgResult",
             "partition_h2"],
    "bivariate": ["ldpred3_auto_bivariate", "ldpred3_auto_bivariate_blocks",
                  "BivariateResult"],
    "finemap": ["ldpred3_pip", "single_signal_finemap", "finemap_by_blocks",
                "FineMapResult", "CredibleSet"],
    "impute": ["impute_sumstats_blocks", "ImputeResult"],
    "scale": ["n_eff_case_control", "h2_liability", "standardize_prs"],
    "lassosum": ["lassosum2", "Lassosum2Result"],
}

# name -> module, for the lazy loader
_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = ["__version__", *_NAME_TO_MODULE]


def __getattr__(name):
    """Import the owning submodule on first access (PEP 562)."""
    mod = _NAME_TO_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = obj          # cache so subsequent access skips __getattr__
    return obj


def __dir__():
    return sorted(__all__)
