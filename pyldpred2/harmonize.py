"""
Harmonise GWAS summary statistics against a genotype variant table.

This aligns each GWAS variant to the genotype's counted (A1) allele so that the
returned per-variant ``beta`` can be applied directly to A1 dosages. It is the
step where a silent allele mix-up would flip effect signs and destroy a PRS, so
the rules are explicit:

* **Match** GWAS variants to genotype variants by rsID when available, else by
  ``chrom:pos``.
* **Same alleles, same order** (``ea==A1``): keep ``beta``.
* **Same alleles, swapped** (``ea==A2``): flip the sign of ``beta`` (it now
  counts A1).
* **Strand flip**: if the complement of the GWAS alleles matches, apply the
  same order/swap logic on the complemented alleles.
* **Strand-ambiguous (palindromic) A/T and C/G SNPs**: dropped by default,
  because strand cannot be resolved from alleles alone.
* **Allele mismatch / unmatched**: dropped.

A summary of how many variants fell into each bucket is returned in ``.log``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Harmonized", "harmonize"]

_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _complement(allele):
    """Reverse-complement a (single- or multi-base) allele; None if non-ACGT."""
    try:
        return "".join(_COMP[b] for b in reversed(allele))
    except KeyError:
        return None


def _is_palindromic(ea, oa):
    return (ea, oa) in (("A", "T"), ("T", "A"), ("C", "G"), ("G", "C"))


@dataclass
class Harmonized:
    """Result of harmonising sumstats to a genotype variant table.

    ``var_index`` indexes the genotype :class:`~genotype_io.VariantTable`; the
    parallel ``beta``/``se``/``n_eff`` arrays are aligned to the A1 allele of
    those variants.
    """

    var_index: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n_eff: np.ndarray
    flipped: np.ndarray
    log: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.var_index)


def _build_index(variants):
    """rsID -> row, and (chrom, pos) -> row, for the genotype variants."""
    by_id, by_pos = {}, {}
    for i in range(len(variants)):
        vid = variants.id[i]
        if vid and vid != ".":
            by_id.setdefault(vid, i)
        by_pos.setdefault((str(variants.chrom[i]), int(variants.pos[i])), i)
    return by_id, by_pos


def harmonize(sumstats, variants, *, drop_ambiguous=True):
    """Align ``sumstats`` to ``variants``; see module docstring for the rules."""
    by_id, by_pos = _build_index(variants)

    idx, betas, ses, ns, flips = [], [], [], [], []
    n_unmatched = n_ambiguous = n_mismatch = n_flip = n_strand = 0
    seen = set()

    for k in range(len(sumstats)):
        sid = sumstats.id[k]
        gi = by_id.get(sid) if sid else None
        if gi is None:
            gi = by_pos.get((str(sumstats.chrom[k]), int(sumstats.pos[k])))
        if gi is None:
            n_unmatched += 1
            continue
        if gi in seen:        # one genotype variant matched once (first wins)
            continue

        ea, oa = sumstats.ea[k], sumstats.oa[k]
        g1, g2 = str(variants.a1[gi]).upper(), str(variants.a2[gi]).upper()

        if drop_ambiguous and _is_palindromic(ea, oa):
            n_ambiguous += 1
            continue

        flip = None
        if (ea, oa) == (g1, g2):
            flip = False
        elif (ea, oa) == (g2, g1):
            flip = True
        else:
            cea, coa = _complement(ea), _complement(oa)
            if cea is not None and coa is not None:
                if (cea, coa) == (g1, g2):
                    flip = False; n_strand += 1
                elif (cea, coa) == (g2, g1):
                    flip = True; n_strand += 1

        if flip is None:
            n_mismatch += 1
            continue

        beta = sumstats.beta[k]
        if not np.isfinite(beta) or not np.isfinite(sumstats.se[k]):
            n_mismatch += 1
            continue
        if flip:
            beta = -beta
            n_flip += 1

        seen.add(gi)
        idx.append(gi); betas.append(beta); ses.append(sumstats.se[k])
        ns.append(sumstats.n_eff[k]); flips.append(flip)

    order = np.argsort(idx)        # keep genotype-column order
    idx = np.array(idx, dtype=np.int64)[order]
    log = {
        "n_sumstats": len(sumstats),
        "n_genotype_variants": len(variants),
        "n_matched": len(idx),
        "n_flipped": n_flip,
        "n_strand_flipped": n_strand,
        "n_dropped_ambiguous": n_ambiguous,
        "n_dropped_mismatch": n_mismatch,
        "n_unmatched": n_unmatched,
    }
    return Harmonized(
        var_index=idx,
        beta=np.array(betas, dtype=float)[order],
        se=np.array(ses, dtype=float)[order],
        n_eff=np.array(ns, dtype=float)[order],
        flipped=np.array(flips, dtype=bool)[order],
        log=log,
    )
