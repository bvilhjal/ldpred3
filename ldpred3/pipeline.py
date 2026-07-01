"""
End-to-end LDpred3 PRS pipeline.

Ties the pieces together:

    GWAS sumstats  +  genotypes (PLINK)
          |                |
          |  read_sumstats |  read_plink
          v                v
        harmonise (align effect alleles to A1, drop ambiguous/mismatched)
          |
          v
        per-block LD from a reference panel  (in-sample or external)
          |
          v
        ldpred3_by_blocks  (inf / grid / auto)  -> adjusted weights
          |
          v
        prs_score  ->  one polygenic score per target individual

Usage from Python::

    from pipeline import run_ldpred3_prs
    res = run_ldpred3_prs("gwas.txt.gz", "target", method="auto")
    res.scores            # per-individual PRS
    res.harmonize_log     # QC counts

or from the command line::

    python -m pipeline --sumstats gwas.txt.gz --plink target \\
        --method auto --out scores.txt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genotype_io import read_plink, read_bim, read_fam, read_bed, _strip_ext
from .bgen_io import read_bgen
from .sumstats import Sumstats, read_sumstats, detect_columns
from .harmonize import harmonize, diagnose_match
from .ld import compute_ld_blocks, save_ld_blocks, load_ld_blocks
from .prs import prs_score, allele_frequency, dosage_stats
from .qc import (qc_sumstats, sd_consistency_mask, dentist_outlier_mask,
                 impute_n_eff)
from .ldpred3 import (standardize_betas, ldpred3_by_blocks, shrink_ld_blocks,
                      SparseLD, LowRankLD)
from .infer import ldpred3_auto_infer
from .lassosum import lassosum2
from .annot import ldpred3_auto_annot_blocks, read_annotations

__all__ = ["PRSResult", "ScoreResult", "run_ldpred3_prs", "run_finemap",
           "preflight_prs", "score_from_weights", "load_genotypes"]


def _inference_dict(res):
    """The h2/p/r2 summary dict stored in ``PRSResult.inference``."""
    return {"h2_est": res.h2_est, "h2_ci": res.h2_ci,
            "p_est": res.p_est, "p_ci": res.p_ci,
            "r2_est": res.r2_est, "r2_ci": res.r2_ci,
            "n_chains_kept": res.n_chains_kept}


def load_genotypes(path, *, sample_path=None, variant_ids=None):
    """Read genotypes from a PLINK prefix or a ``.bgen`` file (auto-detected).

    ``variant_ids`` restricts the read to those variants (by rsID) — via seek
    for PLINK, via a filtered scan for BGEN — so only the requested SNPs are
    loaded from a biobank-scale fileset.
    """
    if str(path).endswith(".bgen"):
        return read_bgen(path, sample_path=sample_path, variant_ids=variant_ids)
    return read_plink(path, variant_ids=variant_ids)


@dataclass
class PRSResult:
    """Output of :func:`run_ldpred3_prs`."""

    scores: np.ndarray          # (n_target,) per-individual PRS
    sample_fid: np.ndarray
    sample_iid: np.ndarray
    beta_adjusted: np.ndarray   # (n_matched,) standardized LDpred3 weights
    var_index: np.ndarray       # genotype columns the weights apply to
    harmonize_log: dict
    qc_log: dict = None         # sumstats + SD-consistency QC counts
    inference: dict = None      # h2/p/r2 estimates (+CIs) if infer=True
    enrichment: dict = None     # annotation enrichment if method="annot"
    variant_id: np.ndarray = None     # IDs of the scored variants (weight order)
    effect_allele: np.ndarray = None  # allele each weight counts (genotype A1)
    other_allele: np.ndarray = None
    chrom: np.ndarray = None
    pos: np.ndarray = None
    af: np.ndarray = None       # A1 frequency in the fit cohort (for frozen scaling)
    sd: np.ndarray = None       # A1 dosage SD in the fit cohort (for frozen scaling)

    def write_weights(self, path):
        """Write the fitted weights as a reusable table.

        Columns: ``ID CHR POS A1 A2 WEIGHT`` where ``A1`` is the allele the
        weight counts and ``WEIGHT`` is the standardized LDpred3 effect. When the
        fit-cohort allele frequency / dosage SD are known, two more columns
        ``AF_REF SD_REF`` are appended so :func:`score_from_weights` can reapply
        the *same* standardization to another cohort (``scaling="frozen"``). Feed
        the file to :func:`score_from_weights` to score a new cohort without
        refitting.
        """
        have_scale = self.af is not None and self.sd is not None
        with open(path, "w") as fh:
            fh.write("ID\tCHR\tPOS\tA1\tA2\tWEIGHT"
                     + ("\tAF_REF\tSD_REF\n" if have_scale else "\n"))
            cols = [self.variant_id, self.chrom, self.pos, self.effect_allele,
                    self.other_allele, self.beta_adjusted]
            if have_scale:
                cols += [self.af, self.sd]
            for row in zip(*cols):
                if have_scale:
                    vid, c, p, a1, a2, w, af, sd = row
                    fh.write(f"{vid}\t{c}\t{p}\t{a1}\t{a2}\t{w:.8g}"
                             f"\t{af:.8g}\t{sd:.8g}\n")
                else:
                    vid, c, p, a1, a2, w = row
                    fh.write(f"{vid}\t{c}\t{p}\t{a1}\t{a2}\t{w:.8g}\n")
        return path

    def __repr__(self):
        nm = self.harmonize_log.get("n_matched", len(self.beta_adjusted))
        inf = ""
        if self.inference is not None:
            i = self.inference
            inf = (f", h2={i['h2_est']:.3f}, p={i['p_est']:.4g}, "
                   f"r2={i['r2_est']:.3f}")
        if self.enrichment:
            top = max(self.enrichment.items(), key=lambda kv: abs(kv[1]))
            inf += f", top_annot={top[0]}={top[1]:+.2f}"
        return (f"PRSResult(n_samples={len(self.scores)}, "
                f"n_variants={nm}{inf})")


def run_ldpred3_prs(sumstats, plink, *, method="auto", block_size=500,
                    n_eff=None, ld_prefix=None, ld_ridge=0.0,
                    ld_cache=None, ld_out=None,
                    sample_path=None, ld_sample_path=None, subset_to_sumstats=True,
                    qc=True, qc_params=None, sd_check=True,
                    impute_n=False, impute_n_params=None,
                    dentist=False, dentist_params=None,
                    ld_shrink=False, ld_shrink_params=None,
                    ld_sparse=False, ld_sparse_params=None,
                    ld_lowrank=False, ld_lowrank_params=None, ld_stream=False,
                    annotations=None, annot_params=None,
                    auto_chains=1, ldsc_init=False, alpha=-1.0,
                    infer=False, infer_max_variants=30000, infer_params=None,
                    sumstats_cols=None, **ldpred3_kwargs):
    """Run the full sumstats -> LDpred3 -> PRS pipeline.

    Parameters
    ----------
    sumstats : str
        Path to the GWAS summary-statistics file.
    plink : str
        PLINK fileset prefix for the **target** genotypes to be scored.
    method : {"auto", "grid", "inf", "annot", "lassosum2", "laplace"}, default "auto"
        LDpred3 model. ``"laplace"`` is the Bayesian-lasso (Laplace-prior)
        posterior-mean sampler — the Bayesian counterpart of ``lassosum2`` (which
        is the same prior's mode). ``"lassosum2"`` is the penalised-regression (L1, sparse)
        alternative, tuned by pseudo-validation (no validation cohort); the
        bigsnpr workflow keeps whichever of auto / lassosum2 predicts better.
    auto_chains : int, default 1
        With ``method="auto"``, average this many quality-filtered chains for the
        PRS weights — the robust LDpred2-auto estimator of Privé et al. (2023) —
        instead of a single chain. ``>1`` enables it (~10 recommended); the same
        multi-chain run also yields h²/p/r², so ``infer=True`` then costs nothing
        extra. Default ``1`` keeps the single-chain behaviour.
    block_size : int, default 500
        Maximum variants per LD block.
    n_eff : float, optional
        GWAS sample size, if the sumstats file lacks an N column.
    ld_prefix : str, optional
        PLINK prefix of an external LD reference panel. If omitted, LD is
        estimated in-sample from the target genotypes.
    ld_ridge : float, default 0.0
        Ridge shrinkage applied to each LD block (see :func:`compute_ld_blocks`).
    ld_cache : str, optional
        Path to LD blocks saved by a previous run (``ld_out``). When given, the
        LD is loaded instead of recomputed and the harmonised variants are
        aligned to the cached set (the cache is authoritative — the SD-check and
        LD computation are skipped). Lets you sweep methods / re-score without
        rebuilding the LD.
    ld_out : str, optional
        Save the computed LD blocks (and their variant IDs) to this ``.npz`` for
        later reuse via ``ld_cache``.
    qc : bool, default True
        Apply sumstats-only QC (:func:`qc.qc_sumstats`) before harmonisation.
    qc_params : dict, optional
        Overrides for the QC thresholds (e.g. ``{"min_maf": 0.005}``).
    sd_check : bool, default True
        After harmonisation, drop variants failing the LDpred3 SD-consistency
        check against the reference panel (:func:`qc.sd_consistency_mask`).
    impute_n : bool, default False
        Replace the per-variant sample size with one **imputed from ``se`` and the
        reference allele frequency** (:func:`qc.impute_n_eff`, Privé et al. HGG
        Advances 2022), anchored to the reported total. Use when the GWAS reports
        only a global / constant / misspecified ``N`` — the LDpred likelihood
        needs the true per-variant precision. Applies to a freshly-computed LD
        (not with ``ld_cache``).
    impute_n_params : dict, optional
        Overrides for :func:`qc.impute_n_eff` (e.g. ``{"anchor_quantile": 0.95}``).
    dentist : bool, default False
        After building the LD blocks, drop variants flagged by the DENTIST-style
        LD-consistency filter (:func:`qc.dentist_outlier_mask`) and rebuild the
        blocks on the survivors. Off by default — it can remove genuine,
        poorly-tagged independent signals along with true errors. Ignored (with
        a warning) when ``ld_cache`` is given, since the cached blocks are
        authoritative; rebuild the cache from a ``dentist=True`` run to apply it.
    dentist_params : dict, optional
        Overrides for the DENTIST thresholds (e.g. ``{"p_cutoff": 1e-6}``).
    ld_shrink : bool, default False
        Apply size-aware spectral shrinkage of the LD blocks toward the identity
        (:func:`ld_utils.shrink_ld_blocks`): each block is shrunk by
        ``alpha = min(max_shrink, k / n_ref)`` so large blocks (noise-dominated
        when ``k`` approaches the reference-panel size) are regularised while
        small, well-estimated blocks are left alone. Helps on a finite / noisy LD
        reference (reduces sampler over-fit and h² inflation).
    ld_shrink_params : dict, optional
        Overrides for the shrinkage (e.g. ``{"max_shrink": 0.3, "intensity": 0.5}``).
    ld_sparse : bool, default False
        Store the LD blocks as banded :class:`~ldpred3.SparseLD` (built and fit
        with O(k·bandwidth) memory instead of O(k²)). Essential at genome scale /
        thousands of SNPs per block. The auto sampler fits these directly via the
        streaming kernel. Not compatible with ``dentist`` (which needs dense LD).
    ld_sparse_params : dict, optional
        Overrides for the banding, e.g. ``{"max_dist": 500, "ld_threshold": 1e-3}``.
    ld_lowrank : bool, default False
        Store the LD blocks as low-rank :class:`~ldpred3.LowRankLD` (top
        eigenvectors), fit by the eigenspace streaming auto sampler at O(k·rank)
        memory. On **realistic** LD this matches the dense fit at a fraction of
        the memory (preferred over ``ld_sparse`` banding, which discards
        long-range LD). Not compatible with ``ld_sparse`` or ``dentist``.
    ld_lowrank_params : dict, optional
        Overrides, e.g. ``{"lowrank_variance": 0.995, "lowrank_max_rank": 1000}``.
        Set ``lowrank_min_size`` (CLI ``--ld-lowrank-min-size``) for a **mixed**
        representation: only blocks at least that large are compressed, smaller
        ones stay dense — near-dense speed genome-wide, compressing just the few
        big blocks that need it.
    ld_stream : bool, default False
        When writing ``ld_out``, store a memory-mappable LD cache (dense /
        low-rank). A later run with ``ld_cache`` then **streams blocks from disk**
        (memmap), so resident memory is ~O(one block) and an LD larger than RAM
        still fits. Build once with ``ld_lowrank=True, ld_out=…, ld_stream=True``;
        reuse cheaply with ``ld_cache=…``.
    alpha : float, default -1.0
        Exponent of the MAF-dependent effect-size prior (Privé et al. 2023):
        each variant's slab variance is scaled by ``[2f(1-f)]^(1+alpha)`` from the
        target allele frequency ``f``. ``-1`` (default) is the flat prior and
        reproduces the original sampler exactly; more negative concentrates effect
        on common variants, less negative on rarer ones. ``method="auto"``/
        ``"grid"`` only (not multi-chain auto, lassosum2 or annot).
    sumstats_cols : dict, optional
        Column overrides forwarded to :func:`read_sumstats`.
    **ldpred3_kwargs
        Forwarded to :func:`ldpred3_by_blocks` (e.g. ``ncores``, ``num_iter``).

    Returns
    -------
    PRSResult
    """
    ss = read_sumstats(sumstats, n_eff=n_eff, **(sumstats_cols or {}))

    qc_log = {}
    if qc:
        keep, qc_log = qc_sumstats(ss, **(qc_params or {}))
        ss = ss.subset(keep)
        if len(ss) == 0:
            raise ValueError("all GWAS variants were removed by sumstats QC")

    # Read only the (QC'd) GWAS variants from the genotypes when possible.
    vids = set(ss.id) if subset_to_sumstats else None
    geno = load_genotypes(plink, sample_path=sample_path, variant_ids=vids)
    if subset_to_sumstats and geno.n_variants == 0:
        # IDs didn't line up (e.g. chr:pos-style genotype IDs) — read in full.
        geno = load_genotypes(plink, sample_path=sample_path)

    h = harmonize(ss, geno.variants)
    if len(h) == 0:
        diag = diagnose_match(ss, geno.variants)
        raise ValueError(
            "no GWAS variants matched the genotypes after harmonisation. "
            f"Diagnosis: {diag['message']} "
            f"(rsID overlap {diag['rsid_overlap']}, position overlap "
            f"{diag['pos_overlap_normalized']}).")

    # LD reference: external panel (matched + allele-recoded) or in-sample.
    h, target_dos, ld_dos, chrom = _external_ld_dosage(
        ss, geno, h, vids=vids, ld_prefix=ld_prefix,
        ld_sample_path=ld_sample_path, subset_to_sumstats=subset_to_sumstats)

    if ld_cache is not None:
        # Cached LD is authoritative: align the harmonised variants to its column
        # order (the cache was built post-SD-check), then skip SD + recompute.
        if dentist:
            import warnings
            warnings.warn(
                "dentist=True is ignored when ld_cache is given: the cached LD "
                "blocks are used as-is. Rebuild the cache (ld_out=) from a run "
                "with dentist=True to apply the filter.", stacklevel=2)
        blocks, cached_ids = load_ld_blocks(ld_cache)
        pos_of = {vid: i for i, vid in enumerate(geno.variants.id[h.var_index])}
        missing = [c for c in cached_ids if c not in pos_of]
        if missing:
            raise ValueError(
                f"ld_cache has {len(missing)} variant(s) absent from the current "
                f"harmonised set (e.g. {missing[:3]}); the inputs/QC changed, so "
                f"the cache no longer applies — rebuild it with ld_out=")
        order = np.array([pos_of[c] for c in cached_ids], dtype=np.int64)
        h = _subset_harmonized(h, order)
        target_dos = geno.dosage[:, h.var_index]
    else:
        # SD-consistency QC: compare sumstats-implied SD to reference-panel SD.
        if sd_check:
            af_ref = allele_frequency(ld_dos)
            keep_sd, sd_log, _ = sd_consistency_mask(h.beta, h.se, h.n_eff, af_ref)
            qc_log["sd_consistency"] = sd_log
            if keep_sd.sum() == 0:
                raise ValueError("all variants failed the SD-consistency check")
            h = _subset_harmonized(h, keep_sd)
            target_dos = target_dos[:, keep_sd]
            ld_dos = ld_dos[:, keep_sd]
            chrom = chrom[keep_sd]

        # Per-variant effective-N imputation (Privé et al., HGG Advances 2022):
        # recover N_j from se + reference allele frequency, anchored to the
        # reported total, and use it as the sampler's per-variant N. Corrects a
        # global / constant / misspecified reported N (the precision the LDpred
        # likelihood needs). Replaces rather than drops, so no variants are lost.
        if impute_n:
            af_ref_i = allele_frequency(ld_dos)
            n_total = float(np.nanmax(h.n_eff))
            n_imp, n_log = impute_n_eff(h.se, af_ref_i, n_total,
                                        **(impute_n_params or {}))
            qc_log["impute_n"] = n_log
            h.n_eff = n_imp

        if (ld_sparse or ld_lowrank) and dentist:
            raise ValueError("dentist requires dense LD blocks and is not "
                             "compatible with ld_sparse / ld_lowrank")
        # Compact LD representations keep persistent memory sub-O(k²) for large
        # blocks (genome / sequencing scale): banded SparseLD (O(k·bandwidth)) or
        # low-rank LowRankLD (O(k·rank), preferred on realistic LD). The dense
        # block is built transiently per block and discarded.
        if ld_lowrank:
            ld_kw = dict(lowrank=True, **(ld_lowrank_params or {}))
        elif ld_sparse:
            ld_kw = dict(sparse=True, **(ld_sparse_params or {}))
        else:
            ld_kw = {}
        blocks = compute_ld_blocks(ld_dos, chrom=chrom, block_size=block_size,
                                   ridge=ld_ridge, **ld_kw)

        # Optional DENTIST-style LD-consistency outlier removal. Catches
        # variants whose z-score disagrees with its LD neighbours (allele/strand
        # errors, LD-reference mismatch). Off by default — it can also drop
        # genuine, poorly-tagged independent signals (see qc.dentist_outlier_mask).
        if dentist:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = h.beta / h.se
            keep_dt, dt_log = dentist_outlier_mask(blocks, z,
                                                   **(dentist_params or {}))
            qc_log["dentist"] = dt_log
            if keep_dt.sum() == 0:
                raise ValueError("all variants failed the DENTIST check")
            if not keep_dt.all():
                h = _subset_harmonized(h, keep_dt)
                target_dos = target_dos[:, keep_dt]
                ld_dos = ld_dos[:, keep_dt]
                chrom = chrom[keep_dt]
                blocks = compute_ld_blocks(ld_dos, chrom=chrom,
                                           block_size=block_size, ridge=ld_ridge)

        # Optional size-aware spectral shrinkage of the LD blocks toward the
        # identity. A block's sample LD from n_ref reference individuals is
        # noise-dominated when the block is large relative to n_ref; shrinking
        # those (and leaving small, well-estimated blocks alone) stabilises the
        # sampler and reduces h2 over-fitting on a finite LD panel.
        if ld_shrink:
            n_ref = ld_dos.shape[0]
            blocks = shrink_ld_blocks(blocks, n_ref,
                                      **(ld_shrink_params or {}))
            sizes = [int(np.asarray(idx).shape[0]) for _, idx in blocks]
            qc_log["ld_shrink"] = {"n_ref": int(n_ref), "n_blocks": len(blocks),
                                   "max_block": int(max(sizes)) if sizes else 0}

        if ld_out is not None:
            save_ld_blocks(ld_out, blocks, geno.variants.id[h.var_index],
                           mmap=ld_stream)

    beta_std, _ = standardize_betas(h.beta, h.se, h.n_eff)

    # Optionally seed the sampler's heritability from LD Score regression (the
    # bigsnpr workflow): h2_init for auto, the fixed h2 for inf/grid. Needs dense
    # blocks (LDSC's r^2 is dense); skipped otherwise.
    if ldsc_init and not any(isinstance(R, (SparseLD, LowRankLD))
                             for R, _ in blocks):
        from .ldsc import ld_scores, ldsc_h2
        ell = ld_scores(blocks)
        chisq = h.n_eff * beta_std ** 2          # z^2 = N * beta_std^2
        h2_ldsc = float(min(max(ldsc_h2(chisq, ell, h.n_eff).h2, 1e-3), 1.0))
        qc_log["ldsc_h2_init"] = h2_ldsc
        if method == "auto":
            ldpred3_kwargs.setdefault("h2_init", h2_ldsc)
        elif method in ("inf", "grid"):
            ldpred3_kwargs.setdefault("h2", h2_ldsc)

    # MAF-dependent effect-size prior (Privé et al. 2023): scale each variant's
    # slab variance by [2f(1-f)]^(1+alpha). alpha=-1 (default) is the flat prior
    # and reproduces the original sampler bit-for-bit. Only the dense grid / auto
    # by-blocks path supports it; reject the combinations that route elsewhere.
    if alpha != -1.0:
        if method not in ("auto", "grid"):
            raise ValueError("alpha (the MAF-dependent prior) applies to "
                             "method='auto' or 'grid' only")
        if method == "auto" and auto_chains and int(auto_chains) > 1:
            raise ValueError("alpha (the MAF-dependent prior) is not supported "
                             "with the multi-chain auto estimator (auto_chains>1)")
        ldpred3_kwargs["af"] = allele_frequency(target_dos)
        ldpred3_kwargs["alpha"] = alpha
        # the MAF prior runs per-block (the global pooled-hyperparameter auto
        # path doesn't carry per-variant slab weights).
        if method == "auto":
            ldpred3_kwargs.setdefault("global_hyper", False)
        qc_log["maf_prior_alpha"] = float(alpha)

    enrichment = None
    inference = None
    if method == "annot":
        if annotations is None:
            raise ValueError("method='annot' requires annotations=<file or array>")
        matched_ids = geno.variants.id[h.var_index]
        if isinstance(annotations, str):
            A, annot_names = read_annotations(annotations, matched_ids)
        else:
            A, annot_names = np.asarray(annotations, dtype=float), None
        ares = ldpred3_auto_annot_blocks(blocks, beta_std, h.n_eff, A,
                                         annotation_names=annot_names,
                                         **(annot_params or {}))
        beta_adj = ares.beta_est
        enrichment = ares.enrichment
    elif method == "lassosum2":
        # Penalised-regression PRS (sparse, L1) over the same LD; picks (s, λ) by
        # pseudo-validation — no validation cohort needed. Needs dense blocks.
        if any(isinstance(R, (SparseLD, LowRankLD)) for R, _ in blocks):
            raise ValueError("method='lassosum2' needs dense LD blocks "
                             "(not ld_sparse / ld_lowrank)")
        lres = lassosum2(blocks, beta_std, **(ldpred3_kwargs or {}))
        beta_adj = lres.beta_est
        qc_log["lassosum2"] = {"s": lres.best_s, "lambda": lres.best_lambda,
                               "pseudoval": lres.best_score,
                               "n_nonzero": lres.n_nonzero}
    elif method == "auto" and auto_chains and int(auto_chains) > 1:
        # Robust multi-chain LDpred3-auto PRS (Privé et al. 2023): run several
        # chains, drop the non-converged ones and average the survivors, rather
        # than scoring from a single chain. The same run also yields h2/p/r2, so
        # --infer adds no extra cost.
        _ip = dict(infer_params or {})
        _ip["n_chains"] = int(auto_chains)
        res = ldpred3_auto_infer(blocks, beta_std, h.n_eff,
                                 ncores=ldpred3_kwargs.get("ncores", 1), **_ip)
        beta_adj = res.beta_est
        if infer:
            inference = _inference_dict(res)
    else:
        beta_adj = ldpred3_by_blocks(blocks, beta_std, h.n_eff, method=method,
                                     **ldpred3_kwargs)
    scores = prs_score(target_dos, beta_adj, standardize=True)
    # Freeze the fit-cohort standardization (per-variant mean/SD) so the same
    # weights can be reapplied on a fixed scale to other cohorts.
    fit_mean, fit_sd = dosage_stats(target_dos)

    if infer and inference is None:
        # Streaming (block-diagonal) inference -- no dense genome-wide LD, so no
        # size cap. (infer_max_variants is kept for backwards compatibility.)
        res = ldpred3_auto_infer(blocks, beta_std, h.n_eff,
                                 ncores=ldpred3_kwargs.get("ncores", 1),
                                 **(infer_params or {}))
        inference = _inference_dict(res)

    gv = geno.variants
    # n_matched is the initial harmonised count; record how many actually
    # survived QC / SD-check / DENTIST / LD-cache alignment and were scored.
    final_log = dict(h.log)
    final_log["n_final"] = int(len(h))
    return PRSResult(
        scores=scores,
        sample_fid=geno.samples.fid,
        sample_iid=geno.samples.iid,
        beta_adjusted=beta_adj,
        var_index=h.var_index,
        harmonize_log=final_log,
        qc_log=qc_log,
        inference=inference,
        enrichment=enrichment,
        variant_id=gv.id[h.var_index],
        effect_allele=gv.a1[h.var_index],
        other_allele=gv.a2[h.var_index],
        chrom=gv.chrom[h.var_index],
        pos=gv.pos[h.var_index],
        af=fit_mean / 2.0,
        sd=fit_sd,
    )


def _subset_harmonized(h, mask):
    from .harmonize import Harmonized
    return Harmonized(
        var_index=h.var_index[mask], beta=h.beta[mask], se=h.se[mask],
        n_eff=h.n_eff[mask], flipped=h.flipped[mask], log=h.log)


def _external_ld_dosage(ss, geno, h, *, vids, ld_prefix, ld_sample_path,
                        subset_to_sumstats):
    """LD dosages aligned and allele-recoded to the harmonised target variants.

    With no external panel, in-sample LD == the target dosages. With an
    ``ld_prefix`` panel: load it, restrict both panels to their shared variants,
    and recode reference dosages to ``2 - dosage`` wherever the panel counts the
    opposite allele to the target/beta (detected via the harmonisation flip
    flags, so strand flips are handled too). Returns
    ``(h, target_dos, ld_dos, chrom)`` -- ``h`` / ``target_dos`` may be re-subset
    to the shared variants. Shared by the PRS and fine-mapping pipelines so both
    use exactly the same orientation logic.
    """
    target_dos = geno.dosage[:, h.var_index]
    chrom = geno.variants.chrom[h.var_index]
    if ld_prefix is None:
        return h, target_dos, target_dos, chrom

    ref = load_genotypes(ld_prefix, sample_path=ld_sample_path, variant_ids=vids)
    if subset_to_sumstats and ref.n_variants == 0:
        ref = load_genotypes(ld_prefix, sample_path=ld_sample_path)
    href = harmonize(ss, ref.variants)
    common = np.intersect1d(geno.variants.id[h.var_index],
                            ref.variants.id[href.var_index])
    if len(common) == 0:
        raise ValueError("LD reference shares no variants with the target")
    tmask = np.isin(geno.variants.id[h.var_index], common)
    h = _subset_harmonized(h, tmask)
    target_dos = geno.dosage[:, h.var_index]
    chrom = geno.variants.chrom[h.var_index]
    ref_order = {vid: i for i, vid in enumerate(ref.variants.id[href.var_index])}
    ref_pos = [ref_order[v] for v in geno.variants.id[h.var_index]]
    ref_cols = href.var_index[ref_pos]
    ld_dos = ref.dosage[:, ref_cols].astype(float, copy=True)
    recode = h.flipped != href.flipped[ref_pos]
    if np.any(recode):
        x = ld_dos[:, recode]
        ld_dos[:, recode] = np.where(np.isfinite(x), 2.0 - x, x)
    return h, target_dos, ld_dos, chrom


def _read_regions(regions):
    """Normalise ``regions`` to a list of ``(chrom, start, end, name)``.

    Accepts a BED-like file path (``chrom start end [name]``, tab/space
    separated, ``#`` comments) or an in-memory list of ``(chrom, start, end[,
    name])`` tuples.
    """
    if not isinstance(regions, str):
        out = []
        for r in regions:
            name = r[3] if len(r) > 3 else f"{r[0]}:{int(r[1])}-{int(r[2])}"
            out.append((str(r[0]), int(r[1]), int(r[2]), name))
        return out
    out = []
    with open(regions) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            name = f[3] if len(f) > 3 else f"{f[0]}:{int(f[1])}-{int(f[2])}"
            out.append((str(f[0]), int(f[1]), int(f[2]), name))
    return out


def _write_finemap(out, res, ids, chrom, pos, beta_std, n_eff):
    """Write ``<out>.pip.tsv`` (per-variant) and ``<out>.cs.tsv`` (credible sets)."""
    z = beta_std * np.sqrt(n_eff)
    with open(f"{out}.pip.tsv", "w") as fh:
        fh.write("variant_id\tchrom\tpos\tpip\tposterior_mean\tposterior_sd\t"
                 "z\tbeta_std\tn_eff\n")
        for i in range(len(ids)):
            fh.write(f"{ids[i]}\t{chrom[i]}\t{int(pos[i])}\t{res.pip[i]:.6g}\t"
                     f"{res.posterior_mean[i]:.6g}\t{res.posterior_sd[i]:.6g}\t"
                     f"{z[i]:.6g}\t{beta_std[i]:.6g}\t{n_eff[i]:.6g}\n")
    with open(f"{out}.cs.tsv", "w") as fh:
        fh.write("cs_id\tsignal\tcoverage\tn_variants\tlead_variant\tlead_pip\t"
                 "purity_min_abs_r\tpurity_mean_abs_r\tvariants\n")
        for k, cs in enumerate(res.credible_sets):
            members = ";".join(str(ids[v]) for v in cs.variants)
            lead = cs.lead_variant if cs.lead_variant is not None else \
                str(ids[cs.variants[int(np.argmax(cs.pip))]])
            fh.write(f"CS{k + 1}\t{cs.signal}\t{cs.coverage:.4g}\t"
                     f"{len(cs.variants)}\t{lead}\t{cs.lead_pip:.4g}\t"
                     f"{cs.purity_min_abs_r:.4g}\t{cs.purity_mean_abs_r:.4g}\t"
                     f"{members}\n")


def run_finemap(sumstats, plink, *, regions=None, out=None, n_eff=None,
                ld_prefix=None, ld_ridge=0.0, block_size=500, sample_path=None,
                ld_sample_path=None, subset_to_sumstats=True, qc=True,
                qc_params=None, sd_check=True, dentist=False, dentist_params=None,
                only_significant=None, max_signals=10, coverage=0.95,
                min_abs_corr=0.5, ncores=1, sumstats_cols=None, **pip_kw):
    """Genome-wide / per-region fine-mapping from a GWAS file + target genotypes.

    Shares the PRS pipeline's read / QC / harmonise / external-LD machinery (so
    allele orientation, the ``2 - dosage`` recoding and SD/DENTIST QC are
    identical), then runs LDpred3-PIP fine-mapping (:func:`ldpred3.finemap_by_blocks`)
    over the LD blocks (or only the loci in ``regions``). Writes ``<out>.pip.tsv``
    and ``<out>.cs.tsv`` when ``out`` is given. Returns the genome-wide
    :class:`~ldpred3.FineMapResult`.

    Parameters
    ----------
    regions : str or list, optional
        Restrict fine-mapping to these loci: a BED-like file (``chrom start end
        [name]``) or a list of ``(chrom, start, end)`` tuples. ``None`` fine-maps
        the whole genome (every LD block).
    only_significant : float, optional
        Skip LD blocks with no variant below this two-sided p-value (e.g.
        ``5e-8``) -- the usual "fine-map loci around hits" mode. ``None`` (default)
        fine-maps every block.
    ld_prefix : str, optional
        External LD reference-panel prefix; in-sample LD from the target if omitted.
    """
    from .finemap import finemap_by_blocks

    ss = read_sumstats(sumstats, n_eff=n_eff, **(sumstats_cols or {}))
    qc_log = {}
    if qc:
        keep, qc_log = qc_sumstats(ss, **(qc_params or {}))
        ss = ss.subset(keep)
        if len(ss) == 0:
            raise ValueError("all GWAS variants were removed by sumstats QC")

    vids = set(ss.id) if subset_to_sumstats else None
    geno = load_genotypes(plink, sample_path=sample_path, variant_ids=vids)
    if subset_to_sumstats and geno.n_variants == 0:
        geno = load_genotypes(plink, sample_path=sample_path)
    h = harmonize(ss, geno.variants)
    if len(h) == 0:
        diag = diagnose_match(ss, geno.variants)
        raise ValueError(
            "no GWAS variants matched the genotypes after harmonisation. "
            f"Diagnosis: {diag['message']} "
            f"(rsID overlap {diag['rsid_overlap']}, position overlap "
            f"{diag['pos_overlap_normalized']}).")

    h, target_dos, ld_dos, chrom = _external_ld_dosage(
        ss, geno, h, vids=vids, ld_prefix=ld_prefix,
        ld_sample_path=ld_sample_path, subset_to_sumstats=subset_to_sumstats)
    pos = geno.variants.pos[h.var_index]
    ids = geno.variants.id[h.var_index]

    if sd_check:
        af_ref = allele_frequency(ld_dos)
        keep_sd, sd_log, _ = sd_consistency_mask(h.beta, h.se, h.n_eff, af_ref)
        qc_log["sd_consistency"] = sd_log
        if keep_sd.sum() == 0:
            raise ValueError("all variants failed the SD-consistency check")
        h = _subset_harmonized(h, keep_sd)
        ld_dos = ld_dos[:, keep_sd]; chrom = chrom[keep_sd]
        pos = pos[keep_sd]; ids = ids[keep_sd]

    if regions is not None:
        inreg = np.zeros(len(ids), dtype=bool)
        for c, s, e, _name in _read_regions(regions):
            inreg |= (chrom == c) & (pos >= s) & (pos <= e)
        if not inreg.any():
            raise ValueError("no harmonised variants fall in the given regions")
        h = _subset_harmonized(h, inreg)
        ld_dos = ld_dos[:, inreg]; chrom = chrom[inreg]
        pos = pos[inreg]; ids = ids[inreg]

    blocks = compute_ld_blocks(ld_dos, chrom=chrom, block_size=block_size,
                               ridge=ld_ridge)
    if dentist:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = h.beta / h.se
        keep_dt, dt_log = dentist_outlier_mask(blocks, z, **(dentist_params or {}))
        qc_log["dentist"] = dt_log
        if keep_dt.sum() == 0:
            raise ValueError("all variants failed the DENTIST check")
        if not keep_dt.all():
            h = _subset_harmonized(h, keep_dt)
            ld_dos = ld_dos[:, keep_dt]; chrom = chrom[keep_dt]
            pos = pos[keep_dt]; ids = ids[keep_dt]
            blocks = compute_ld_blocks(ld_dos, chrom=chrom, block_size=block_size,
                                       ridge=ld_ridge)

    beta_std, _ = standardize_betas(h.beta, h.se, h.n_eff)
    res = finemap_by_blocks(blocks, beta_std, h.n_eff,
                            only_significant=only_significant, variant_ids=ids,
                            max_signals=max_signals, coverage=coverage,
                            min_abs_corr=min_abs_corr, ncores=ncores, **pip_kw)
    res.diagnostics.update(variant_ids=ids, chrom=chrom, pos=pos, qc_log=qc_log)
    if out is not None:
        _write_finemap(out, res, ids, chrom, pos, beta_std, h.n_eff)
    return res


def preflight_prs(sumstats, plink, *, n_eff=None, sample_path=None,
                  qc=True, qc_params=None, subset_to_sumstats=True,
                  sumstats_cols=None):
    """Fast preflight: detect columns, match IDs and preview harmonisation.

    Reads the sumstats and the (matched) genotype variants and runs QC +
    harmonisation, but does **not** compute LD or fit a model — so it returns in
    seconds and surfaces the usual late failures (wrong column mapping, IDs that
    don't line up, mass allele mismatch) up front. Returns a report ``dict`` with
    ``columns`` / ``missing`` / ``n_sumstats`` / ``qc`` / ``harmonize`` /
    ``warnings``; nothing is written.
    """
    header, mapping = detect_columns(sumstats, **(sumstats_cols or {}))
    warnings = []
    missing = [f for f in ("ea", "oa") if f not in mapping]
    if "beta" not in mapping and "or" not in mapping:
        missing.append("beta/or")
    if "n_eff" not in mapping and n_eff is None:
        missing.append("n_eff (or pass n_eff=)")
    if missing:
        return {"columns": mapping, "header": header, "missing": missing,
                "warnings": ["could not resolve required columns; "
                             "pass them via sumstats_cols"]}

    ss = read_sumstats(sumstats, n_eff=n_eff, **(sumstats_cols or {}))
    n_raw = len(ss)
    qc_log = {}
    if qc:
        keep, qc_log = qc_sumstats(ss, **(qc_params or {}))
        ss = ss.subset(keep)
    if len(ss) == 0:
        return {"columns": mapping, "header": header, "missing": [],
                "n_sumstats": n_raw, "qc": qc_log,
                "warnings": ["all variants removed by sumstats QC"]}

    vids = set(ss.id) if subset_to_sumstats else None
    geno = load_genotypes(plink, sample_path=sample_path, variant_ids=vids)
    if subset_to_sumstats and geno.n_variants == 0:
        geno = load_genotypes(plink, sample_path=sample_path)
        warnings.append("sumstats IDs did not match genotype IDs; matched by "
                        "position instead (check ID conventions / build)")
    h = harmonize(ss, geno.variants)
    if h.log["n_matched"] == 0:
        warnings.append("no variants matched after harmonisation — check build "
                        "/ allele coding")
    elif h.log["n_dropped_mismatch"] > 0.5 * h.log["n_sumstats"]:
        warnings.append("over half of variants dropped as allele-mismatched — "
                        "likely a build or strand problem")
    return {"columns": mapping, "header": header, "missing": [],
            "n_sumstats": n_raw, "n_after_qc": len(ss), "qc": qc_log,
            "n_genotype_variants_read": geno.n_variants,
            "harmonize": h.log, "warnings": warnings}


@dataclass
class ScoreResult:
    """Output of :func:`score_from_weights`."""

    scores: np.ndarray
    sample_fid: np.ndarray
    sample_iid: np.ndarray
    n_weights: int
    n_matched: int

    def __repr__(self):
        return (f"ScoreResult(n_samples={len(self.scores)}, "
                f"n_matched={self.n_matched}/{self.n_weights})")


def _score_plink_streamed(prefix, n_samples, n_total, var_index, beta,
                          *, mean=None, sd=None, chunk=1000):
    """PRS over variant-chunks from a ``.bed`` without the full dosage matrix.

    Reads each chunk of the matched variant columns (seek-based), standardises it
    (in-cohort, or with a frozen ``mean``/``sd``) and accumulates
    ``dosage_chunk @ beta_chunk``. Peak memory is O(n_samples · chunk) instead of
    O(n_samples · n_variants) — so a biobank-scale target is scored without
    materialising hundreds of GB.
    """
    bed = _strip_ext(prefix) + ".bed"
    var_index = np.asarray(var_index, dtype=np.int64)
    scores = np.zeros(n_samples)
    for s in range(0, var_index.size, chunk):
        cols = var_index[s:s + chunk]
        dos = read_bed(bed, n_samples, n_total, variant_idx=cols)
        if mean is not None:
            scores += prs_score(dos, beta[s:s + chunk],
                                mean=mean[s:s + chunk], sd=sd[s:s + chunk])
        else:
            scores += prs_score(dos, beta[s:s + chunk], standardize=True)
    return scores


def score_from_weights(weights, plink, *, sample_path=None, scaling="target",
                       chunk=1000):
    """Score a target cohort from a saved weights file — no LD, no refit.

    ``weights`` is a path written by :meth:`PRSResult.write_weights` (columns
    ``ID CHR POS A1 A2 WEIGHT``, optionally ``AF_REF SD_REF``). The weights are
    harmonised to the target's alleles (sign-flipped where the alleles are
    swapped) and applied as a standardized polygenic score.

    ``scaling`` chooses the genotype standardization:

    * ``"target"`` (default): standardize using *this* cohort's allele
      frequencies / SD — fine for within-cohort ranking.
    * ``"frozen"``: reuse the fit cohort's ``AF_REF``/``SD_REF`` from the file,
      so two cohorts with different allele frequencies are scored on the *same*
      scale (portable / comparable). Requires those columns.

    A **PLINK** target is *streamed*: only the ``.bim``/``.fam`` are read up front
    and the ``.bed`` is scored in ``chunk``-variant blocks, so peak memory is
    O(n_samples · chunk) rather than O(n_samples · n_variants) — a biobank-scale
    cohort is scored without materialising the full dosage matrix. BGEN uses the
    full-load path.
    """
    if scaling not in ("target", "frozen"):
        raise ValueError("scaling must be 'target' or 'frozen'")
    ids, chrom, pos, a1, a2, w, af_ref, sd_ref = [], [], [], [], [], [], [], []
    with open(weights) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split("\t") if "\t" in header else header.split()
        has_scale = "AF_REF" in cols and "SD_REF" in cols
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t") if "\t" in line else line.split()
            ids.append(f[0]); chrom.append(str(f[1])); pos.append(int(f[2]))
            a1.append(f[3].upper()); a2.append(f[4].upper()); w.append(float(f[5]))
            if has_scale:
                af_ref.append(float(f[6])); sd_ref.append(float(f[7]))
    if scaling == "frozen" and not has_scale:
        raise ValueError("scaling='frozen' needs AF_REF/SD_REF columns; the "
                         "weights file was written without them (re-fit and "
                         "write_weights, or use scaling='target')")
    m = len(ids)
    ss = Sumstats(
        id=np.array(ids, dtype=object), chrom=np.array(chrom, dtype=object),
        pos=np.array(pos, dtype=np.int64), ea=np.array(a1, dtype=object),
        oa=np.array(a2, dtype=object), beta=np.array(w, dtype=float),
        se=np.ones(m), n_eff=np.ones(m),
        eaf=np.full(m, np.nan), info=np.full(m, np.nan))

    # PLINK is streamed (read the .bim/.fam only, then accumulate the score over
    # variant-chunks of the .bed); BGEN keeps the full-load path.
    is_bgen = str(plink).endswith(".bgen")
    if is_bgen:
        geno = load_genotypes(plink, sample_path=sample_path, variant_ids=set(ids))
        if geno.n_variants == 0:
            geno = load_genotypes(plink, sample_path=sample_path)
        h = harmonize(ss, geno.variants)
        variants_id, fid, iid = geno.variants.id, geno.samples.fid, geno.samples.iid
        n_samples = len(fid)
    else:
        pref = _strip_ext(plink)
        variants = read_bim(pref + ".bim")
        samples = read_fam(pref + ".fam")
        h = harmonize(ss, variants)
        variants_id, fid, iid = variants.id, samples.fid, samples.iid
        n_samples, n_total = len(fid), len(variants)
    if len(h) == 0:
        diag = diagnose_match(ss, read_bim(_strip_ext(plink) + ".bim")
                              if not is_bgen else geno.variants)
        raise ValueError("no weights matched the target genotypes. "
                         f"Diagnosis: {diag['message']}")

    mean = sd = None
    if scaling == "frozen":
        # AF_REF/SD_REF count the weight's A1; where harmonisation flipped the
        # allele the target dosage counts the other allele, so AF -> 1-AF (the
        # SD is unchanged for g vs 2-g). Frozen mean = 2*AF.
        by_id = {i: (a, s) for i, a, s in zip(ids, af_ref, sd_ref)}
        sel = variants_id[h.var_index]
        af = np.array([by_id[i][0] for i in sel])
        sd = np.array([by_id[i][1] for i in sel])
        af = np.where(h.flipped, 1.0 - af, af)
        mean = 2.0 * af

    if is_bgen:
        dos = geno.dosage[:, h.var_index]
        scores = (prs_score(dos, h.beta, mean=mean, sd=sd) if mean is not None
                  else prs_score(dos, h.beta, standardize=True))
    else:
        scores = _score_plink_streamed(plink, n_samples, n_total, h.var_index,
                                       h.beta, mean=mean, sd=sd, chunk=chunk)
    return ScoreResult(scores=scores, sample_fid=fid, sample_iid=iid,
                       n_weights=m, n_matched=len(h))


def _write_scores(path, fids, iids, scores, percentiles=False):
    """Write FID/IID/PRS, optionally with standardized Z and percentile columns."""
    extra = ""
    if percentiles:
        from .scale import standardize_prs
        z, pct = standardize_prs(scores)
    with open(path, "w") as fh:
        fh.write("FID\tIID\tPRS" + ("\tZ\tPCT\n" if percentiles else "\n"))
        for i, (fid, iid, s) in enumerate(zip(fids, iids, scores)):
            if percentiles:
                extra = f"\t{z[i]:.4g}\t{pct[i]:.4g}"
            fh.write(f"{fid}\t{iid}\t{s:.6g}{extra}\n")


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LDpred3 PRS pipeline")
    ap.add_argument("--sumstats", help="GWAS sumstats file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plink", help="target PLINK prefix (.bed/.bim/.fam)")
    g.add_argument("--bgen", help="target BGEN file (.bgen)")
    ap.add_argument("--sample", default=None, help="BGEN .sample file")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "grid", "inf", "annot", "lassosum2",
                             "laplace"],
                    help="LDpred3 model (default: auto; annot when "
                         "--annotations is given)")
    ap.add_argument("--annotations", default=None,
                    help="per-SNP annotation table (for --method annot)")
    ap.add_argument("--block-size", type=int, default=500,
                    help="max variants per LD block (default: 500)")
    ap.add_argument("--n-eff", type=float, default=None,
                    help="effective sample size, if the sumstats lack an N column")
    ap.add_argument("--n-cases", type=float, default=None,
                    help="case count (with --n-controls) -> effective N "
                         "4/(1/Ncase+1/Ncontrol) for a binary-trait GWAS")
    ap.add_argument("--n-controls", type=float, default=None,
                    help="control count (see --n-cases)")
    ap.add_argument("--ld-prefix", default=None, help="external LD panel prefix")
    ap.add_argument("--ld-ridge", type=float, default=0.0,
                    help="shrink each LD block towards the identity by this "
                         "fraction (default: 0.0)")
    ap.add_argument("--ld-out", default=None, help="save computed LD blocks (.npz)")
    ap.add_argument("--ld-cache", default=None,
                    help="reuse LD blocks saved earlier with --ld-out")
    ap.add_argument("--ncores", type=int, default=1,
                    help="threads for the Gibbs sampler (Numba; default: 1)")
    ap.add_argument("--no-qc", action="store_true", help="skip sumstats QC")
    ap.add_argument("--dentist", action="store_true",
                    help="apply the DENTIST LD-consistency outlier filter "
                         "(off by default; can drop poorly-tagged true signals)")
    ap.add_argument("--no-sd-check", action="store_true",
                    help="skip the SD-consistency QC")
    ap.add_argument("--impute-n", action="store_true",
                    help="impute per-variant N from se + reference frequency "
                         "(use when the GWAS reports only a global/constant N)")
    ap.add_argument("--ld-shrink", action="store_true",
                    help="size-aware spectral shrinkage of large LD blocks "
                         "toward the identity (helps on a finite LD panel)")
    ap.add_argument("--ld-sparse", action="store_true",
                    help="store LD blocks as banded SparseLD (O(k*bandwidth) "
                         "memory; for genome-scale / large blocks)")
    ap.add_argument("--ld-max-dist", type=int, default=None,
                    help="band half-width for --ld-sparse (variants; default: "
                         "threshold only)")
    ap.add_argument("--ld-lowrank", action="store_true",
                    help="store LD blocks as low-rank LowRankLD (O(k*rank) "
                         "memory; preferred for realistic / sequencing-scale LD)")
    ap.add_argument("--ld-lowrank-var", type=float, default=0.99,
                    help="spectrum fraction kept by --ld-lowrank (default: 0.99)")
    ap.add_argument("--ld-lowrank-min-size", type=int, default=0,
                    help="with --ld-lowrank, only compress blocks >= this size; "
                         "smaller blocks stay dense (mixed; default: 0 = all)")
    ap.add_argument("--ld-stream", action="store_true",
                    help="write a memory-mappable LD cache (with --ld-out) so a "
                         "later --ld-cache run streams blocks from disk")
    ap.add_argument("--ldsc-init", action="store_true",
                    help="seed the sampler's h2 from LD Score regression "
                         "(h2_init for auto, fixed h2 for inf/grid)")
    ap.add_argument("--auto-chains", type=int, default=1,
                    help="with --method auto: average this many filtered chains "
                         "for a more robust PRS (Privé 2023; >1 enables it, ~10 "
                         "recommended). Default 1 = single chain.")
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="MAF-dependent effect-size prior exponent (Privé 2023): "
                         "slab variance scales as [2f(1-f)]^(1+alpha). Default -1 "
                         "= flat prior (unchanged); auto/grid only.")
    ap.add_argument("--infer", action="store_true",
                    help="also infer h2 / polygenicity / predictive r2 "
                         "(streams block-diagonal LD; works with dense, "
                         "--ld-sparse and --ld-lowrank blocks)")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only: detect columns, match IDs and preview "
                         "harmonisation, then exit (no LD, no fit, no output)")
    ap.add_argument("--save-weights", default=None,
                    help="also write the fitted weights to this file")
    ap.add_argument("--weights", default=None,
                    help="score the target from a saved weights file "
                         "(no sumstats / LD / refit needed)")
    ap.add_argument("--scaling", choices=("target", "frozen"), default="target",
                    help="with --weights: 'target' standardizes by this cohort "
                         "(default); 'frozen' reuses the fit cohort's AF_REF/"
                         "SD_REF for a portable scale (needs those columns)")
    ap.add_argument("--score-chunk", type=int, default=1000,
                    help="with --weights on PLINK: variants per streamed chunk "
                         "(lower = less memory at biobank scale; default 1000)")
    ap.add_argument("--prs-percentiles", action="store_true",
                    help="also write standardized PRS (Z) and percentile (PCT) "
                         "columns to the scores output")
    ap.add_argument("--out", help="output scores file (or fine-map output prefix)")
    ap.add_argument("--finemap", action="store_true",
                    help="fine-map instead of computing a PRS: write per-variant "
                         "PIPs (<out>.pip.tsv) and credible sets (<out>.cs.tsv)")
    ap.add_argument("--regions", default=None,
                    help="with --finemap: BED-like file (chrom start end [name]) "
                         "restricting fine-mapping to these loci")
    ap.add_argument("--finemap-coverage", type=float, default=0.95,
                    help="credible-set coverage target (default: 0.95)")
    ap.add_argument("--finemap-max-signals", type=int, default=10,
                    help="max signals per locus to fine-map (default: 10)")
    ap.add_argument("--finemap-min-purity", type=float, default=0.5,
                    help="min |r| purity for a credible set (default: 0.5)")
    ap.add_argument("--finemap-only-significant", type=float, default=None,
                    help="only fine-map LD blocks with a variant below this "
                         "p-value (e.g. 5e-8); default: all blocks")
    args = ap.parse_args(argv)
    target = args.plink or args.bgen

    # Binary-trait effective N from case/control counts (overrides --n-eff).
    if args.n_cases is not None or args.n_controls is not None:
        if args.n_cases is None or args.n_controls is None:
            ap.error("--n-cases and --n-controls must be given together")
        from .scale import n_eff_case_control
        args.n_eff = n_eff_case_control(args.n_cases, args.n_controls)
        print(f"effective N (case/control) = {args.n_eff:.0f}")

    # Mode 1: score directly from a saved weights file (no sumstats/LD/refit).
    if args.weights:
        if not args.out:
            ap.error("--weights requires --out")
        sr = score_from_weights(args.weights, target, sample_path=args.sample,
                                scaling=args.scaling, chunk=args.score_chunk)
        _write_scores(args.out, sr.sample_fid, sr.sample_iid, sr.scores,
                      percentiles=args.prs_percentiles)
        print(f"matched {sr.n_matched} / {sr.n_weights} weights; "
              f"wrote {len(sr.scores)} PRS to {args.out}")
        return

    if not args.sumstats:
        ap.error("--sumstats is required (unless scoring with --weights)")

    # Mode 2: preflight only.
    if args.dry_run:
        rep = preflight_prs(args.sumstats, target, n_eff=args.n_eff,
                            sample_path=args.sample, qc=not args.no_qc)
        print("detected columns: "
              + ", ".join(f"{k}={v}" for k, v in rep["columns"].items()))
        if rep["missing"]:
            print("MISSING required columns: " + ", ".join(rep["missing"]))
        if "harmonize" in rep:
            h = rep["harmonize"]
            print(f"sumstats: {rep['n_sumstats']} variants"
                  + (f" -> {rep['n_after_qc']} after QC" if args.no_qc is False
                     else ""))
            print(f"matched {h['n_matched']} / {h['n_sumstats']} to the target "
                  f"({h['n_flipped']} flipped, {h['n_dropped_ambiguous']} "
                  f"ambiguous, {h['n_dropped_mismatch']} mismatched, "
                  f"{h['n_unmatched']} unmatched)")
        for w in rep["warnings"]:
            print(f"WARNING: {w}")
        return

    if not args.out:
        ap.error("--out is required")

    # Mode 2.5: fine-mapping (PIPs + credible sets) instead of a PRS.
    if args.finemap:
        fm = run_finemap(
            args.sumstats, target, regions=args.regions, out=args.out,
            block_size=args.block_size, n_eff=args.n_eff, sample_path=args.sample,
            ld_prefix=args.ld_prefix, ld_ridge=args.ld_ridge, ncores=args.ncores,
            qc=not args.no_qc, sd_check=not args.no_sd_check, dentist=args.dentist,
            only_significant=args.finemap_only_significant,
            coverage=args.finemap_coverage, max_signals=args.finemap_max_signals,
            min_abs_corr=args.finemap_min_purity)
        d = fm.diagnostics
        print(f"fine-mapped {d.get('n_blocks_finemapped', '?')}/"
              f"{d.get('n_blocks', '?')} blocks; {len(fm.credible_sets)} credible "
              f"sets (sum PIP {fm.n_signals_est:.1f}); "
              f"wrote {args.out}.pip.tsv and {args.out}.cs.tsv")
        return

    # Mode 3: full run. Default to method=annot when annotations are supplied.
    method = args.method
    if args.annotations and method == "auto":
        method = "annot"
        print("note: --annotations given, using --method annot")

    res = run_ldpred3_prs(
        args.sumstats, target, method=method,
        block_size=args.block_size, n_eff=args.n_eff, sample_path=args.sample,
        ld_prefix=args.ld_prefix, ld_ridge=args.ld_ridge, ncores=args.ncores,
        ld_out=args.ld_out, ld_cache=args.ld_cache,
        annotations=args.annotations,
        qc=not args.no_qc, sd_check=not args.no_sd_check,
        impute_n=args.impute_n,
        dentist=args.dentist, ld_shrink=args.ld_shrink,
        ld_sparse=args.ld_sparse,
        ld_sparse_params=({"max_dist": args.ld_max_dist}
                          if args.ld_max_dist else None),
        ld_lowrank=args.ld_lowrank,
        ld_lowrank_params={"lowrank_variance": args.ld_lowrank_var,
                           "lowrank_min_size": args.ld_lowrank_min_size},
        ld_stream=args.ld_stream,
        auto_chains=args.auto_chains, ldsc_init=args.ldsc_init,
        alpha=args.alpha, infer=args.infer)

    if args.save_weights:
        res.write_weights(args.save_weights)
        print(f"wrote {len(res.beta_adjusted)} weights to {args.save_weights}")

    _write_scores(args.out, res.sample_fid, res.sample_iid, res.scores,
                  percentiles=args.prs_percentiles)

    q = res.qc_log or {}
    if "n_input" in q:
        sd = q.get("sd_consistency", {})
        print(f"QC: {q['n_input']} -> {q['n_kept']} variants "
              f"(lowN {q.get('n_drop_lowN', 0)}, dup {q.get('n_drop_duplicate', 0)}"
              f", nonfinite {q.get('n_drop_nonfinite', 0)}"
              + (f", SD-inconsistent {sd.get('n_drop_sd_inconsistent', 0)}"
                 if sd else "") + ")")
    log = res.harmonize_log
    print(f"matched {log['n_matched']} / {log['n_sumstats']} GWAS variants "
          f"({log['n_flipped']} flipped, {log['n_dropped_ambiguous']} ambiguous,"
          f" {log['n_dropped_mismatch']} mismatched, "
          f"{log['n_unmatched']} unmatched)")
    if log.get("n_final") is not None and log["n_final"] != log["n_matched"]:
        print(f"  {log['n_final']} variants scored after QC / filtering")
    if res.inference is not None:
        i = res.inference
        print(f"inferred h2={i['h2_est']:.3f} {tuple(round(x, 3) for x in i['h2_ci'])}"
              f"  p={i['p_est']:.4f} {tuple(round(x, 4) for x in i['p_ci'])}"
              f"  predictive r2={i['r2_est']:.3f} "
              f"{tuple(round(x, 3) for x in i['r2_ci'])}")
    if res.enrichment:
        top = sorted(res.enrichment.items(), key=lambda kv: -abs(kv[1]))[:8]
        print("annotation enrichment: "
              + ", ".join(f"{nm}={c:+.2f}" for nm, c in top))
    print(f"wrote {len(res.scores)} PRS to {args.out}")


if __name__ == "__main__":
    _main()
