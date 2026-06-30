"""
Quality control for GWAS summary statistics, before LDpred3.

Two stages, matching the bigsnpr / LDpred2 tutorial workflow:

1. :func:`qc_sumstats` — filters that need only the sumstats themselves:
   * drop non-finite or non-positive-SE rows,
   * drop duplicated variants,
   * per-variant **sample-size** filter (``N ≥ frac · max(N)``) — meta-analysis
     variants seen in few cohorts are unreliable,
   * **MAF** filter (when an effect-allele-frequency column is present),
   * **INFO** / imputation-quality filter (when present),
   * **chi-square outlier** filter (drop implausibly large ``z = β/se``).

2. :func:`sd_consistency_mask` — the key LDpred3 diagnostic, which needs a
   reference panel. It compares the standard deviation implied by the sumstats,
   ``sd_ss ≈ 1/√(N·se² + β²)``, against the genotype SD from the reference
   panel, ``sd_ref = √(2·f·(1−f))``. Variants where the two disagree signal a
   wrong ``N``, an allele/strand error, or bad imputation and are removed. This
   runs *after* harmonisation, on the matched variants.

3. :func:`impute_n_eff` — the *correction* counterpart of the SD check (Privé,
   Arbel, Aschard & Vilhjálmsson, *HGG Advances* 2022): rather than dropping
   variants with a wrong ``N``, recover the per-variant effective sample size
   from ``se`` and the reference allele frequency, ``N_j ∝ 1/(se_j²·2f_j(1−f_j))``.
   Useful when a GWAS reports only a global (or constant / misspecified) ``N``.

Each function returns a boolean keep-mask and a per-filter count log.
"""

from __future__ import annotations

import numpy as np

__all__ = ["qc_sumstats", "sd_consistency_mask", "dentist_outlier_mask",
           "impute_n_eff"]


def dentist_outlier_mask(blocks, z, *, p_cutoff=5e-8, ridge=0.01, n_iter=20,
                         min_block=3, min_neighbor_r=0.1):
    """DENTIST-style LD-consistency outlier filter (Chen et al., *Nat Commun* 2021).

    Within each LD block, test whether each variant's z-score is consistent with
    the value predicted from its LD neighbours. Under the LD model the studentized
    leave-one-out residual ``T_j = (Omega z)_j^2 / Omega_jj`` (with
    ``Omega = (R + ridge·I)^-1``) is ~``chi^2_1``; the variant with the largest
    ``T`` above the ``p_cutoff`` threshold is flagged as LD-inconsistent — the
    signature of an allele/strand error, a local LD-reference mismatch or bad
    imputation — and dropped.

    Removal is **iterative and one-at-a-time per block**: a single corrupt variant
    inflates the residuals of every LD neighbour it tags, so removing all variants
    above threshold in one pass would discard a whole haplotype around one error.
    Instead each pass drops only the single worst variant per block and recomputes
    on the survivors (DENTIST's actual scheme), repeating up to ``n_iter`` times
    until no block exceeds the threshold.

    Only variants that **have an LD neighbour** (some survivor with
    ``|r| >= min_neighbor_r``) are removal candidates. This is essential: with no
    neighbour the residual collapses to the variant's own z-score, so a region of
    near-independent variants (or an in-sample LD matrix close to the identity)
    would otherwise have *every* genome-wide-significant hit flagged as an
    "outlier". An uncorroborated association is left in place — there is simply no
    LD evidence to call it inconsistent.

    Parameters
    ----------
    blocks : list of (R, idx)
        Per-block LD matrices and the variants' positions in ``z``.
    z : array_like (m,)
        Marginal z-scores (``beta_hat / se``), covariance ``R`` under the model.
    p_cutoff : float, default 5e-8
        Two-sided p-value for the chi-square_1 threshold (stringent by design).
    ridge : float, default 0.01
        Ridge added to ``R`` before inversion (stabilises a noisy reference LD).
    n_iter : int, default 20
        Maximum number of single-removal passes (stops early once clean).
    min_block : int, default 3
        Skip blocks with fewer surviving variants than this.
    min_neighbor_r : float, default 0.1
        A variant is only a removal candidate if some surviving block-mate has
        ``|r| >= min_neighbor_r`` with it. Guards against flagging uncorroborated
        signals in low-LD / near-identity regions.

    Returns
    -------
    keep : ndarray of bool (m,)
    log : dict

    Notes
    -----
    Even among corroborated variants this can flag a *genuine* signal that
    disagrees with its neighbours (a poorly tagged independent association). It is
    therefore a deliberate trade-off — keep ``p_cutoff`` stringent and treat it as
    optional cleaning, not a default.
    """
    from statistics import NormalDist
    thr = NormalDist().inv_cdf(1.0 - p_cutoff / 2.0) ** 2      # chi^2_1 cutoff
    z = np.asarray(z, dtype=float)
    m = z.shape[0]
    keep = np.ones(m, dtype=bool)
    fblocks = [(np.asarray(R, dtype=float), np.asarray(idx)) for R, idx in blocks]

    # Blocks tile disjoint variant ranges, so a block's residuals only change
    # when it drops one of its own variants. Once a block makes no removal (or
    # can no longer test any variant) it is settled for good, so we mark it
    # inactive and skip its (otherwise identical) re-inversion on later passes.
    n_pass = 0
    active = np.ones(len(fblocks), dtype=bool)
    for _ in range(max(1, int(n_iter))):
        flagged = False
        for bi, (R, idx) in enumerate(fblocks):
            if not active[bi]:
                continue
            local = keep[idx]
            if int(local.sum()) < min_block:
                active[bi] = False
                continue
            Rk = R[np.ix_(local, local)]
            zk = z[idx[local]]
            # A variant is testable only if it has a neighbour in LD; otherwise
            # its residual is just its own z and any strong hit looks "wrong".
            offdiag = np.abs(Rk) - np.eye(Rk.shape[0])
            has_nbr = offdiag.max(axis=1) >= min_neighbor_r
            if not has_nbr.any():
                active[bi] = False
                continue
            omega = np.linalg.inv(Rk + ridge * np.eye(Rk.shape[0]))
            t = omega @ zk
            stat = t * t / np.maximum(np.diag(omega), 1e-12)
            stat = np.where(has_nbr, stat, -np.inf)   # only corroborated variants
            j = int(np.argmax(stat))
            if stat[j] > thr:
                # Drop only the single worst variant in this block; its LD
                # neighbours are re-tested (on the survivors) next pass.
                keep[idx[local][j]] = False
                flagged = True
            else:
                active[bi] = False   # clean: residuals won't change again
        n_pass += 1
        if not flagged:
            break

    log = {"n_input": m, "n_drop_dentist": m - int(keep.sum()),
           "n_kept": int(keep.sum()), "n_pass": n_pass}
    return keep, log


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
    """LDpred3 SD-consistency check against reference allele frequencies.

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


def impute_n_eff(se, af_ref, n_total, *, anchor_quantile=0.99, info=None):
    """Recover the per-variant effective sample size from ``se`` and frequency.

    Privé, Arbel, Aschard & Vilhjálmsson (*HGG Advances* 2022): the standard error
    of a marginal effect relates to the sample size and the (reference) allele
    frequency by ``se_j ≈ sd_y / (√(2 f_j (1−f_j)) · √N_j)``, so the per-variant
    effective ``N`` can be read back off the summary statistics::

        N_j  ∝  1 / (se_j² · 2 f_j (1−f_j))

    The unknown proportionality constant (the phenotype-variance scale ``sd_y²``,
    and the per-allele vs standardized ``se`` scale) is fixed by **anchoring a high
    quantile of the imputed N to the reported total** ``n_total`` — the best-typed
    common variants carry ≈ the full sample. The result is clipped at ``n_total``
    (a variant cannot have more than the whole sample).

    This is the *correction* counterpart of :func:`sd_consistency_mask`: where a
    GWAS reports only a global (or constant, or misspecified) ``N``, the true
    per-variant N — which varies with imputation quality, missingness and
    meta-analysis cohort overlap — is what the LDpred sampler needs, since it sets
    each variant's likelihood precision. Replacing rather than dropping keeps
    variants the SD filter would otherwise discard.

    Parameters
    ----------
    se : array_like (m,)
        Reported standard errors of the marginal effects (any consistent scale;
        the anchoring is scale-invariant).
    af_ref : array_like (m,)
        Allele frequency of the counted allele in the reference panel.
    n_total : float
        Reported total (effective) GWAS sample size, used only to set the scale.
    anchor_quantile : float, default 0.99
        The quantile of the raw imputed N matched to ``n_total`` (robust to the
        few noisy top variants a hard max would key on).
    info : array_like (m,), optional
        Imputation-quality (INFO) scores. When given, the imputed N is multiplied
        by ``info`` as well — relevant only if ``se`` does not already reflect the
        imputation uncertainty (it usually does, so leave unset by default).

    Returns
    -------
    n_imp : ndarray (m,)
        Imputed per-variant effective sample size, in ``(0, n_total]``.
    log : dict
        Summary diagnostics (median imputed N, fraction far below ``n_total``).
    """
    se = np.asarray(se, dtype=float)
    af_ref = np.asarray(af_ref, dtype=float)
    n_total = float(n_total)
    sd_ref2 = 2.0 * af_ref * (1.0 - af_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_raw = 1.0 / (se ** 2 * sd_ref2)
    finite = np.isfinite(n_raw) & (n_raw > 0)
    if not finite.any():
        raise ValueError("no variants with finite se and polymorphic af_ref to "
                         "impute N from")
    ref = np.quantile(n_raw[finite], anchor_quantile)
    scale = n_total / ref if ref > 0 else 1.0
    n_imp = n_raw * scale
    if info is not None:
        n_imp = n_imp * np.asarray(info, dtype=float)
    # Non-finite (se=0 or monomorphic) fall back to the reported total; clip to
    # the physical ceiling (a variant cannot exceed the whole sample).
    n_imp = np.where(np.isfinite(n_imp) & (n_imp > 0), n_imp, n_total)
    n_imp = np.minimum(n_imp, n_total)

    log = {
        "n_input": int(se.size),
        "n_total": n_total,
        "median_imputed_n": float(np.median(n_imp)),
        "frac_below_half": float(np.mean(n_imp < 0.5 * n_total)),
    }
    return n_imp, log
