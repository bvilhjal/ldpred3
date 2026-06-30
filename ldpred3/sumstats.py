"""
Reader for GWAS summary-statistics files.

GWAS are distributed as delimited text with wildly inconsistent column names,
so this reader maps a large set of common aliases onto a canonical schema and
lets the caller override any column explicitly. The canonical fields are:

==============  ====================================================
field           meaning
==============  ====================================================
``id``          variant identifier (rsID), may be empty
``chrom``       chromosome (string), may be empty
``pos``         base-pair position (int), may be 0
``ea``          effect allele -- the allele ``beta`` is expressed per copy of
``oa``          other (non-effect) allele
``beta``        per-allele effect on the trait (log-OR for binary traits)
``se``          standard error of ``beta``
``n_eff``       (effective) sample size
``eaf``         effect-allele frequency (optional; used for palindrome QC)
==============  ====================================================

Effects given as odds ratios are converted to ``beta = log(OR)``. A missing
``se`` is recovered from ``beta`` and the p-value when possible.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import math

import numpy as np

__all__ = ["Sumstats", "read_sumstats", "detect_columns"]

# Column aliases (lower-cased) -> canonical field.
_ALIASES = {
    "id": ["snp", "rsid", "rs", "id", "variant_id", "markername", "snpid",
           "marker", "rs_id"],
    "chrom": ["chr", "chrom", "chromosome", "#chrom", "hg19chr", "chr_name"],
    "pos": ["bp", "pos", "position", "base_pair_location", "bp_position",
            "pos_b37", "base_pair"],
    "ea": ["a1", "effect_allele", "ea", "allele1", "alt", "effectallele",
           "tested_allele", "inc_allele"],
    "oa": ["a2", "other_allele", "oa", "nea", "allele0", "allele2", "ref",
           "noneffect_allele", "dec_allele"],
    "beta": ["beta", "b", "effect", "effect_size", "effects", "log_odds"],
    "or": ["or", "odds_ratio", "oddsratio"],
    "se": ["se", "standard_error", "stderr", "standarderror", "sebeta",
           "se_beta", "logor_se"],
    "pval": ["p", "pval", "p_value", "pvalue", "p-value", "p.value", "p_bolt_lmm",
             "p_value_association", "p_wald", "pval_nominal"],
    "n_eff": ["n_eff", "neff", "n", "sample_size", "totalsamplesize",
              "n_samples", "n_total", "obs_ct", "n_complete_samples"],
    "eaf": ["eaf", "freq", "frq", "effect_allele_frequency", "a1freq",
            "freq1", "maf", "af", "effect_allele_freq"],
    "info": ["info", "imputation_info", "imp_info", "rsq", "r2", "info_score",
             "imputation_quality", "minimac_r2"],
}


@dataclass
class Sumstats:
    """Parsed GWAS summary statistics as parallel arrays."""

    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    ea: np.ndarray
    oa: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    n_eff: np.ndarray
    eaf: np.ndarray
    info: np.ndarray

    def __len__(self):
        return len(self.beta)

    def subset(self, mask):
        """Return a new Sumstats keeping only rows where ``mask`` is True."""
        mask = np.asarray(mask)
        return Sumstats(*(getattr(self, f)[mask] for f in (
            "id", "chrom", "pos", "ea", "oa", "beta", "se", "n_eff",
            "eaf", "info")))


def _open(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _sniff_delimiter(header_line):
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return None      # whitespace; handled with str.split


def _build_colmap(header, overrides):
    """Map canonical field -> column index using aliases + user overrides."""
    lower = [h.strip().lower() for h in header]
    colmap = {}
    for field, aliases in _ALIASES.items():
        for a in aliases:
            if a in lower:
                colmap[field] = lower.index(a)
                break
    # Explicit overrides win (value may be a column name or an integer index).
    for field, col in (overrides or {}).items():
        if isinstance(col, int):
            colmap[field] = col
        else:
            colmap[field] = lower.index(col.strip().lower())
    return colmap


def detect_columns(path, **col_overrides):
    """Peek at the header and report the detected column mapping.

    Returns ``(header, mapping)`` where ``header`` is the raw column list and
    ``mapping`` is ``{canonical_field: column_name}`` for every field that was
    resolved (via aliases or ``col_overrides``). Reads only the first line — use
    it for a fast preflight before committing to a full run.
    """
    with _open(path) as fh:
        first = fh.readline().rstrip("\n")
    delim = _sniff_delimiter(first)
    header = first.split(delim) if delim else first.split()
    colmap = _build_colmap(header, col_overrides)
    return header, {field: header[i] for field, i in colmap.items()}


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return math.nan


def read_sumstats(path, *, n_eff=None, **col_overrides):
    """Read a GWAS summary-statistics file into a :class:`Sumstats`.

    Parameters
    ----------
    path : str
        Path to a delimited text file (optionally gzipped). Delimiter (tab,
        comma or whitespace) is auto-detected.
    n_eff : float or str, optional
        Sample size to use when the file has no per-variant N column. A **string**
        is treated as the name (or 0-based index) of the per-variant N column to
        use, i.e. an override for an unrecognised ``N`` header.
    **col_overrides
        Force a canonical field to a particular column, e.g.
        ``read_sumstats(..., ea="A1", beta="effect")`` or by index
        ``read_sumstats(..., beta=6)``.

    Returns
    -------
    Sumstats
    """
    if isinstance(n_eff, str):           # treat a string n_eff as a column override
        col_overrides["n_eff"] = n_eff
        n_eff = None
    with _open(path) as fh:
        first = fh.readline().rstrip("\n")
        delim = _sniff_delimiter(first)
        header = first.split(delim) if delim else first.split()
        colmap = _build_colmap(header, col_overrides)

        for req in ("ea", "oa"):
            if req not in colmap:
                raise ValueError(
                    f"could not find a '{req}' allele column in {path}; "
                    f"header was {header}. Pass it explicitly, e.g. {req}=...")
        if "beta" not in colmap and "or" not in colmap:
            raise ValueError(
                f"no effect column (beta/OR) found in {path}; header {header}")
        if "n_eff" not in colmap and n_eff is None:
            raise ValueError(
                f"no sample-size column found in {path}; pass n_eff=...")

        ids, chroms, poss, eas, oas = [], [], [], [], []
        betas, ses, ns, eafs, infos = [], [], [], [], []

        def get(fields, key, default=""):
            i = colmap.get(key)
            return fields[i] if i is not None and i < len(fields) else default

        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            f = raw.split(delim) if delim else raw.split()

            if "beta" in colmap:
                beta = _to_float(get(f, "beta"))
            else:
                orv = _to_float(get(f, "or"))
                beta = math.log(orv) if orv > 0 else math.nan

            se = _to_float(get(f, "se")) if "se" in colmap else math.nan
            if math.isnan(se) and "pval" in colmap and not math.isnan(beta):
                p = _to_float(get(f, "pval"))
                if 0 < p < 1 and beta != 0:
                    from statistics import NormalDist
                    # Two-sided z = Phi^-1(1 - p/2) = -Phi^-1(p/2). Use the second
                    # form: for tiny p (common in GWAS), 1 - p/2 rounds to exactly
                    # 1.0 and inv_cdf(1.0) raises, whereas p/2 stays representable.
                    half = p / 2.0
                    if half > 0.0:
                        z = -NormalDist().inv_cdf(half)
                        if math.isfinite(z) and z > 0:
                            se = abs(beta) / z

            n = _to_float(get(f, "n_eff")) if "n_eff" in colmap else float(n_eff)

            ids.append(get(f, "id"))
            chroms.append(str(get(f, "chrom")))
            poss.append(int(_to_float(get(f, "pos", "0")) or 0))
            eas.append(get(f, "ea").upper())
            oas.append(get(f, "oa").upper())
            betas.append(beta); ses.append(se); ns.append(n)
            eafs.append(_to_float(get(f, "eaf")) if "eaf" in colmap else math.nan)
            infos.append(_to_float(get(f, "info")) if "info" in colmap else math.nan)

    return Sumstats(
        id=np.array(ids, dtype=object),
        chrom=np.array(chroms, dtype=object),
        pos=np.array(poss, dtype=np.int64),
        ea=np.array(eas, dtype=object),
        oa=np.array(oas, dtype=object),
        beta=np.array(betas, dtype=float),
        se=np.array(ses, dtype=float),
        n_eff=np.array(ns, dtype=float),
        eaf=np.array(eafs, dtype=float),
        info=np.array(infos, dtype=float),
    )
