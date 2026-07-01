"""Command-line interface for the LDpred3 pipeline.

This is the thin argparse layer over the library API in :mod:`ldpred3.pipeline`
(kept separate so importing the library does not pull in argparse or CLI glue).
The console-script entry point ``ldpred3`` and ``python -m ldpred3`` both call
:func:`main`.
"""

from __future__ import annotations

import argparse

from .pipeline import (run_ldpred3_prs, score_from_weights, preflight_prs,
                       run_finemap)


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


def _build_parser():
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
    return ap


def main(argv=None):
    ap = _build_parser()
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
    main()
