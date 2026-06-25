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

from genotype_io import read_plink
from bgen_io import read_bgen
from sumstats import read_sumstats
from harmonize import harmonize
from ld import compute_ld_blocks
from prs import prs_score
from ldpred2 import standardize_betas, ldpred2_by_blocks

__all__ = ["PRSResult", "run_ldpred2_prs", "load_genotypes"]


def load_genotypes(path, *, sample_path=None):
    """Read genotypes from a PLINK prefix or a ``.bgen`` file (auto-detected)."""
    if str(path).endswith(".bgen"):
        return read_bgen(path, sample_path=sample_path)
    return read_plink(path)


@dataclass
class PRSResult:
    """Output of :func:`run_ldpred2_prs`."""

    scores: np.ndarray          # (n_target,) per-individual PRS
    sample_fid: np.ndarray
    sample_iid: np.ndarray
    beta_adjusted: np.ndarray   # (n_matched,) standardized LDpred2 weights
    var_index: np.ndarray       # genotype columns the weights apply to
    harmonize_log: dict


def run_ldpred2_prs(sumstats, plink, *, method="auto", block_size=500,
                    n_eff=None, ld_prefix=None, ld_ridge=0.0,
                    sample_path=None, ld_sample_path=None,
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
    sumstats_cols : dict, optional
        Column overrides forwarded to :func:`read_sumstats`.
    **ldpred2_kwargs
        Forwarded to :func:`ldpred2_by_blocks` (e.g. ``ncores``, ``num_iter``).

    Returns
    -------
    PRSResult
    """
    geno = load_genotypes(plink, sample_path=sample_path)
    ss = read_sumstats(sumstats, n_eff=n_eff, **(sumstats_cols or {}))
    h = harmonize(ss, geno.variants)
    if len(h) == 0:
        raise ValueError("no GWAS variants matched the genotypes after "
                         "harmonisation; check IDs/alleles/build")

    target_dos = geno.dosage[:, h.var_index]
    chrom = geno.variants.chrom[h.var_index]

    # LD reference: external panel (matched to the same variants) or in-sample.
    if ld_prefix is not None:
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

    blocks = compute_ld_blocks(ld_dos, chrom=chrom, block_size=block_size,
                               ridge=ld_ridge)
    beta_std, _ = standardize_betas(h.beta, h.se, h.n_eff)
    beta_adj = ldpred2_by_blocks(blocks, beta_std, h.n_eff, method=method,
                                 **ldpred2_kwargs)
    scores = prs_score(target_dos, beta_adj, standardize=True)
    return PRSResult(
        scores=scores,
        sample_fid=geno.samples.fid,
        sample_iid=geno.samples.iid,
        beta_adjusted=beta_adj,
        var_index=h.var_index,
        harmonize_log=h.log,
    )


def _subset_harmonized(h, mask):
    from harmonize import Harmonized
    return Harmonized(
        var_index=h.var_index[mask], beta=h.beta[mask], se=h.se[mask],
        n_eff=h.n_eff[mask], flipped=h.flipped[mask], log=h.log)


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LDpred2 PRS pipeline")
    ap.add_argument("--sumstats", required=True, help="GWAS sumstats file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plink", help="target PLINK prefix (.bed/.bim/.fam)")
    g.add_argument("--bgen", help="target BGEN file (.bgen)")
    ap.add_argument("--sample", default=None, help="BGEN .sample file")
    ap.add_argument("--method", default="auto", choices=["auto", "grid", "inf"])
    ap.add_argument("--block-size", type=int, default=500)
    ap.add_argument("--n-eff", type=float, default=None)
    ap.add_argument("--ld-prefix", default=None, help="external LD panel prefix")
    ap.add_argument("--ld-ridge", type=float, default=0.0)
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--out", required=True, help="output scores file")
    args = ap.parse_args(argv)

    res = run_ldpred2_prs(
        args.sumstats, args.plink or args.bgen, method=args.method,
        block_size=args.block_size, n_eff=args.n_eff, sample_path=args.sample,
        ld_prefix=args.ld_prefix, ld_ridge=args.ld_ridge, ncores=args.ncores)

    with open(args.out, "w") as fh:
        fh.write("FID\tIID\tPRS\n")
        for fid, iid, s in zip(res.sample_fid, res.sample_iid, res.scores):
            fh.write(f"{fid}\t{iid}\t{s:.6g}\n")

    log = res.harmonize_log
    print(f"matched {log['n_matched']} / {log['n_sumstats']} GWAS variants "
          f"({log['n_flipped']} flipped, {log['n_dropped_ambiguous']} ambiguous,"
          f" {log['n_dropped_mismatch']} mismatched, "
          f"{log['n_unmatched']} unmatched)")
    print(f"wrote {len(res.scores)} PRS to {args.out}")


if __name__ == "__main__":
    _main()
