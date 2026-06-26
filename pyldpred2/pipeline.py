"""
End-to-end LDpred2 PRS pipeline.

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
        ldpred2_by_blocks  (inf / grid / auto)  -> adjusted weights
          |
          v
        prs_score  ->  one polygenic score per target individual

Usage from Python::

    from pipeline import run_ldpred2_prs
    res = run_ldpred2_prs("gwas.txt.gz", "target", method="auto")
    res.scores            # per-individual PRS
    res.harmonize_log     # QC counts

or from the command line::

    python -m pipeline --sumstats gwas.txt.gz --plink target \\
        --method auto --out scores.txt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genotype_io import read_plink
from .bgen_io import read_bgen
from .sumstats import Sumstats, read_sumstats, detect_columns
from .harmonize import harmonize
from .ld import compute_ld_blocks, save_ld_blocks, load_ld_blocks
from .prs import prs_score, allele_frequency
from .qc import qc_sumstats, sd_consistency_mask
from .ldpred2 import standardize_betas, ldpred2_by_blocks
from .infer import ldpred2_auto_infer
from .annot import ldpred2_auto_annot_blocks, read_annotations

__all__ = ["PRSResult", "ScoreResult", "run_ldpred2_prs", "preflight_prs",
           "score_from_weights", "load_genotypes"]


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
    """Output of :func:`run_ldpred2_prs`."""

    scores: np.ndarray          # (n_target,) per-individual PRS
    sample_fid: np.ndarray
    sample_iid: np.ndarray
    beta_adjusted: np.ndarray   # (n_matched,) standardized LDpred2 weights
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

    def write_weights(self, path):
        """Write the fitted weights as a reusable table.

        Columns: ``ID CHR POS A1 A2 WEIGHT`` where ``A1`` is the allele the
        weight counts and ``WEIGHT`` is the standardized LDpred2 effect. Feed the
        file to :func:`score_from_weights` to score a new cohort without
        refitting.
        """
        with open(path, "w") as fh:
            fh.write("ID\tCHR\tPOS\tA1\tA2\tWEIGHT\n")
            for vid, c, p, a1, a2, w in zip(
                    self.variant_id, self.chrom, self.pos,
                    self.effect_allele, self.other_allele, self.beta_adjusted):
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


def run_ldpred2_prs(sumstats, plink, *, method="auto", block_size=500,
                    n_eff=None, ld_prefix=None, ld_ridge=0.0,
                    ld_cache=None, ld_out=None,
                    sample_path=None, ld_sample_path=None, subset_to_sumstats=True,
                    qc=True, qc_params=None, sd_check=True,
                    annotations=None, annot_params=None,
                    infer=False, infer_max_variants=30000, infer_params=None,
                    sumstats_cols=None, **ldpred2_kwargs):
    """Run the full sumstats -> LDpred2 -> PRS pipeline.

    Parameters
    ----------
    sumstats : str
        Path to the GWAS summary-statistics file.
    plink : str
        PLINK fileset prefix for the **target** genotypes to be scored.
    method : {"auto", "grid", "inf"}, default "auto"
        LDpred2 model.
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
        After harmonisation, drop variants failing the LDpred2 SD-consistency
        check against the reference panel (:func:`qc.sd_consistency_mask`).
    sumstats_cols : dict, optional
        Column overrides forwarded to :func:`read_sumstats`.
    **ldpred2_kwargs
        Forwarded to :func:`ldpred2_by_blocks` (e.g. ``ncores``, ``num_iter``).

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
        raise ValueError("no GWAS variants matched the genotypes after "
                         "harmonisation; check IDs/alleles/build")

    target_dos = geno.dosage[:, h.var_index]
    chrom = geno.variants.chrom[h.var_index]

    # LD reference: external panel (matched to the same variants) or in-sample.
    if ld_prefix is not None:
        ref = load_genotypes(ld_prefix, sample_path=ld_sample_path,
                             variant_ids=vids)
        if subset_to_sumstats and ref.n_variants == 0:
            ref = load_genotypes(ld_prefix, sample_path=ld_sample_path)
        href = harmonize(ss, ref.variants)
        # Restrict to variants present in both target-matched and ref-matched.
        common = np.intersect1d(geno.variants.id[h.var_index],
                                ref.variants.id[href.var_index])
        if len(common) == 0:
            raise ValueError("LD reference shares no variants with the target")
        tmask = np.isin(geno.variants.id[h.var_index], common)
        h = _subset_harmonized(h, tmask)
        target_dos = geno.dosage[:, h.var_index]
        chrom = geno.variants.chrom[h.var_index]
        ref_order = {vid: i for i, vid in enumerate(ref.variants.id[href.var_index])}
        ref_cols = href.var_index[[ref_order[v] for v in
                                   geno.variants.id[h.var_index]]]
        ld_dos = ref.dosage[:, ref_cols]
    else:
        ld_dos = target_dos

    if ld_cache is not None:
        # Cached LD is authoritative: align the harmonised variants to its column
        # order (the cache was built post-SD-check), then skip SD + recompute.
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

        blocks = compute_ld_blocks(ld_dos, chrom=chrom, block_size=block_size,
                                   ridge=ld_ridge)
        if ld_out is not None:
            save_ld_blocks(ld_out, blocks, geno.variants.id[h.var_index])

    beta_std, _ = standardize_betas(h.beta, h.se, h.n_eff)

    enrichment = None
    if method == "annot":
        if annotations is None:
            raise ValueError("method='annot' requires annotations=<file or array>")
        matched_ids = geno.variants.id[h.var_index]
        if isinstance(annotations, str):
            A, annot_names = read_annotations(annotations, matched_ids)
        else:
            A, annot_names = np.asarray(annotations, dtype=float), None
        ares = ldpred2_auto_annot_blocks(blocks, beta_std, h.n_eff, A,
                                         annotation_names=annot_names,
                                         **(annot_params or {}))
        beta_adj = ares.beta_est
        enrichment = ares.enrichment
    else:
        beta_adj = ldpred2_by_blocks(blocks, beta_std, h.n_eff, method=method,
                                     **ldpred2_kwargs)
    scores = prs_score(target_dos, beta_adj, standardize=True)

    inference = None
    if infer:
        m_tot = len(h)
        if m_tot > infer_max_variants:
            raise ValueError(
                f"inference assembles a dense {m_tot}x{m_tot} LD matrix; that "
                f"exceeds infer_max_variants={infer_max_variants}. Run it on a "
                f"chromosome / curated SNP set, or raise the limit.")
        dense = np.zeros((m_tot, m_tot), dtype=np.float32)
        for R, idx in blocks:
            dense[np.ix_(idx, idx)] = R
        res = ldpred2_auto_infer(dense, beta_std, h.n_eff,
                                 ncores=ldpred2_kwargs.get("ncores", 1),
                                 **(infer_params or {}))
        inference = {"h2_est": res.h2_est, "h2_ci": res.h2_ci,
                     "p_est": res.p_est, "p_ci": res.p_ci,
                     "r2_est": res.r2_est, "r2_ci": res.r2_ci,
                     "n_chains_kept": res.n_chains_kept}

    gv = geno.variants
    return PRSResult(
        scores=scores,
        sample_fid=geno.samples.fid,
        sample_iid=geno.samples.iid,
        beta_adjusted=beta_adj,
        var_index=h.var_index,
        harmonize_log=h.log,
        qc_log=qc_log,
        inference=inference,
        enrichment=enrichment,
        variant_id=gv.id[h.var_index],
        effect_allele=gv.a1[h.var_index],
        other_allele=gv.a2[h.var_index],
        chrom=gv.chrom[h.var_index],
        pos=gv.pos[h.var_index],
    )


def _subset_harmonized(h, mask):
    from .harmonize import Harmonized
    return Harmonized(
        var_index=h.var_index[mask], beta=h.beta[mask], se=h.se[mask],
        n_eff=h.n_eff[mask], flipped=h.flipped[mask], log=h.log)


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


def score_from_weights(weights, plink, *, sample_path=None):
    """Score a target cohort from a saved weights file — no LD, no refit.

    ``weights`` is a path written by :meth:`PRSResult.write_weights` (columns
    ``ID CHR POS A1 A2 WEIGHT``). The weights are harmonised to the target's
    alleles (sign-flipped where the alleles are swapped) and applied as a
    standardized polygenic score. This is the cheap path for applying an existing
    model to a new cohort.
    """
    ids, chrom, pos, a1, a2, w = [], [], [], [], [], []
    with open(weights) as fh:
        header = fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t") if "\t" in line else line.split()
            ids.append(f[0]); chrom.append(str(f[1])); pos.append(int(f[2]))
            a1.append(f[3].upper()); a2.append(f[4].upper()); w.append(float(f[5]))
    m = len(ids)
    ss = Sumstats(
        id=np.array(ids, dtype=object), chrom=np.array(chrom, dtype=object),
        pos=np.array(pos, dtype=np.int64), ea=np.array(a1, dtype=object),
        oa=np.array(a2, dtype=object), beta=np.array(w, dtype=float),
        se=np.ones(m), n_eff=np.ones(m),
        eaf=np.full(m, np.nan), info=np.full(m, np.nan))

    geno = load_genotypes(plink, sample_path=sample_path, variant_ids=set(ids))
    if geno.n_variants == 0:
        geno = load_genotypes(plink, sample_path=sample_path)
    h = harmonize(ss, geno.variants)
    if len(h) == 0:
        raise ValueError("no weights matched the target genotypes")
    scores = prs_score(geno.dosage[:, h.var_index], h.beta, standardize=True)
    return ScoreResult(scores=scores, sample_fid=geno.samples.fid,
                       sample_iid=geno.samples.iid, n_weights=m,
                       n_matched=len(h))


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LDpred2 PRS pipeline")
    ap.add_argument("--sumstats", help="GWAS sumstats file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plink", help="target PLINK prefix (.bed/.bim/.fam)")
    g.add_argument("--bgen", help="target BGEN file (.bgen)")
    ap.add_argument("--sample", default=None, help="BGEN .sample file")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "grid", "inf", "annot"])
    ap.add_argument("--annotations", default=None,
                    help="per-SNP annotation table (for --method annot)")
    ap.add_argument("--block-size", type=int, default=500)
    ap.add_argument("--n-eff", type=float, default=None)
    ap.add_argument("--ld-prefix", default=None, help="external LD panel prefix")
    ap.add_argument("--ld-ridge", type=float, default=0.0)
    ap.add_argument("--ld-out", default=None, help="save computed LD blocks (.npz)")
    ap.add_argument("--ld-cache", default=None,
                    help="reuse LD blocks saved earlier with --ld-out")
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--no-qc", action="store_true", help="skip sumstats QC")
    ap.add_argument("--no-sd-check", action="store_true",
                    help="skip the SD-consistency QC")
    ap.add_argument("--infer", action="store_true",
                    help="also infer h2 / polygenicity / predictive r2 "
                         "(dense; for chromosome / curated-SNP scale)")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only: detect columns, match IDs and preview "
                         "harmonisation, then exit (no LD, no fit, no output)")
    ap.add_argument("--save-weights", default=None,
                    help="also write the fitted weights to this file")
    ap.add_argument("--weights", default=None,
                    help="score the target from a saved weights file "
                         "(no sumstats / LD / refit needed)")
    ap.add_argument("--out", help="output scores file")
    args = ap.parse_args(argv)
    target = args.plink or args.bgen

    # Mode 1: score directly from a saved weights file (no sumstats/LD/refit).
    if args.weights:
        if not args.out:
            ap.error("--weights requires --out")
        sr = score_from_weights(args.weights, target, sample_path=args.sample)
        with open(args.out, "w") as fh:
            fh.write("FID\tIID\tPRS\n")
            for fid, iid, s in zip(sr.sample_fid, sr.sample_iid, sr.scores):
                fh.write(f"{fid}\t{iid}\t{s:.6g}\n")
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

    # Mode 3: full run. Default to method=annot when annotations are supplied.
    method = args.method
    if args.annotations and method == "auto":
        method = "annot"
        print("note: --annotations given, using --method annot")

    res = run_ldpred2_prs(
        args.sumstats, target, method=method,
        block_size=args.block_size, n_eff=args.n_eff, sample_path=args.sample,
        ld_prefix=args.ld_prefix, ld_ridge=args.ld_ridge, ncores=args.ncores,
        ld_out=args.ld_out, ld_cache=args.ld_cache,
        annotations=args.annotations,
        qc=not args.no_qc, sd_check=not args.no_sd_check, infer=args.infer)

    if args.save_weights:
        res.write_weights(args.save_weights)
        print(f"wrote {len(res.beta_adjusted)} weights to {args.save_weights}")

    with open(args.out, "w") as fh:
        fh.write("FID\tIID\tPRS\n")
        for fid, iid, s in zip(res.sample_fid, res.sample_iid, res.scores):
            fh.write(f"{fid}\t{iid}\t{s:.6g}\n")

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
