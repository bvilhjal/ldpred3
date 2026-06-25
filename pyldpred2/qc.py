"""
Quality control for GWAS summary statistics, before LDpred2.

Two stages, matching the bigsnpr / LDpred2 tutorial workflow:

1. :func:`qc_sumstats` — filters that need only the sumstats themselves:
   * drop non-finite or non-positive-SE rows,
   * drop duplicated variants,
   * per-variant **sample-size** filter (``N ≥ frac · max(N)``) — meta-analysis
     variants seen in few cohorts are unreliable,
   * **MAF** filter (when an effect-allele-frequency column is present),
   * **INFO** / imputation-quality filter (when present),
   * **chi-square outlier** filter (drop implausibly large ``z = β/se``).

2. :func:`sd_consistency_mask` — the key LDpred2 diagnostic, which needs a
   reference panel. It compares the standard deviation implied by the sumstats,
   ``sd_ss ≈ 1/√(N·se² + β²)``, against the genotype SD from the reference
   panel, ``sd_ref = √(2·f·(1−f))``. Variants where the two disagree signal a
   wrong ``N``, an allele/strand error, or bad imputation and are removed. This
   runs *after* harmonisation, on the matched variants.

Each function returns a boolean keep-mask and a per-filter count log.
"""

from __future__ import annotations

import numpy as np

__all__ = ["qc_sumstats", "sd_consistency_mask"]


def qc_sumstats(ss, *, min_n_ratio=0.7, min_maf=0.01, min_info=0.7,
                max_chisq=None, drop_duplicates=True):
    """Sumstats-only QC. Returns ``(keep_mask, log)`` over ``len(ss)`` variants.

    Parameters
    ----------
    ss : Sumstats
    min_n_ratio : float, default 0.7
        Keep variants with ``n_eff >= min_n_ratio * max(n_eff)``.
    min_maf : float, default 0.01
        Minor-allele-frequency threshold; applied only where ``eaf`` is present.
    min_info : float, default 0.7
        Imputation-quality threshold; applied only where ``info`` is present.
    max_chisq : float, optional
        Drop variants with ``(beta/se)**2 > max_chisq``. ``None`` disables it.
    drop_duplicates : bool, default True
        Drop variants whose id (or chrom:pos:alleles) appears more than once.
    """
    n = len(ss)
    keep = np.ones(n, dtype=bool)
    log = {"n_input": n}

    finite = np.isfinite(ss.beta) & np.isfinite(ss.se) & (ss.se > 0)
    log["n_drop_nonfinite"] = int((keep & ~finite).sum())
    keep &= finite

    if drop_duplicates:
        # Any variant key occurring more than once is dropped entirely.
        keys = [ss.id[i] if ss.id[i] else
                (str(ss.chrom[i]), int(ss.pos[i]), ss.ea[i], ss.oa[i])
                for i in range(n)]
        counts = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        dup = np.array([counts[k] > 1 for k in keys])
        log["n_drop_duplicate"] = int((keep & dup).sum())
        keep &= ~dup

    if np.any(np.isfinite(ss.n_eff)):
        nmax = np.nanmax(ss.n_eff)
        bad = ss.n_eff < (min_n_ratio * nmax)
        log["n_drop_lowN"] = int((keep & bad).sum())
        keep &= ~bad

    if np.any(np.isfinite(ss.eaf)):
        maf = np.minimum(ss.eaf, 1 - ss.eaf)
        bad = np.isfinite(ss.eaf) & (maf < min_maf)
        log["n_drop_lowMAF"] = int((keep & bad).sum())
        keep &= ~bad

    if np.any(np.isfinite(ss.info)):
        bad = np.isfinite(ss.info) & (ss.info < min_info)
        log["n_drop_lowINFO"] = int((keep & bad).sum())
        keep &= ~bad

    if max_chisq is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            chisq = (ss.beta / ss.se) ** 2
        bad = np.isfinite(chisq) & (chisq > max_chisq)
        log["n_drop_chisq_outlier"] = int((keep & bad).sum())
        keep &= ~bad

    log["n_kept"] = int(keep.sum())
    return keep, log


def sd_consistency_mask(beta, se, n_eff, af_ref, *,
                        sd_ratio_bounds=(0.5, 2.0), min_sd_ref=0.05):
    """LDpred2 SD-consistency check against reference allele frequencies.

    Parameters
    ----------
    beta, se, n_eff : array_like
        Per-variant summary statistics (already harmonised).
    af_ref : array_like
        Allele frequency of the *same counted allele* in the reference panel.
    sd_ratio_bounds : (lo, hi), default (0.5, 2.0)
        Keep variants whose ``sd_ss / sd_ref`` ratio lies in ``[lo, hi]``.
    min_sd_ref : float, default 0.05
        Drop variants whose reference SD is below this (near-monomorphic).

    Returns
    -------
    keep : ndarray of bool
    log : dict
    diag : dict with ``sd_ss`` and ``sd_ref`` arrays (for plotting/inspection)
    """
    beta = np.asarray(beta, float); se = np.asarray(se, float)
    n_eff = np.asarray(n_eff, float); af_ref = np.asarray(af_ref, float)

    sd_ref = np.sqrt(2.0 * af_ref * (1.0 - af_ref))
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_ss = 1.0 / np.sqrt(n_eff * se ** 2 + beta ** 2)
    # Put sd_ss on the same scale as sd_ref (it is defined up to a constant for
    # standardized effects); anchor on the bulk via the median ratio.
    finite = np.isfinite(sd_ss) & np.isfinite(sd_ref) & (sd_ref > 0)
    if finite.any():
        scale = np.median(sd_ref[finite] / sd_ss[finite])
        sd_ss = sd_ss * scale

    lo, hi = sd_ratio_bounds
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sd_ss / sd_ref
    keep = finite & (sd_ref >= min_sd_ref) & (ratio >= lo) & (ratio <= hi)

    log = {
        "n_input": int(beta.size),
        "n_drop_sd_inconsistent": int((finite & ~keep).sum()),
        "n_drop_nonfinite_or_mono": int((~finite | (sd_ref < min_sd_ref)).sum()),
        "n_kept": int(keep.sum()),
    }
    return keep, log, {"sd_ss": sd_ss, "sd_ref": sd_ref}
