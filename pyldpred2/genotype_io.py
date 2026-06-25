"""
Genotype file readers for the PRS pipeline (NumPy-only, no third-party deps).

Currently implemented:

* PLINK 1 binary filesets (``.bed`` / ``.bim`` / ``.fam``) -- the de-facto
  standard hard-call genotype format, and what ``bigsnpr`` ingests.

The reader returns a :class:`Genotypes` bundle: a dosage matrix of shape
``(n_samples, n_variants)`` plus aligned variant and sample tables. Dosages
count the **A1 allele** (the 5th column of the ``.bim``); missing calls are
encoded as ``-1``. Effect-allele alignment against GWAS summary statistics is
handled later, during harmonisation, so this layer stays a faithful, lossless
view of the file.

A minimal :func:`write_plink` is included so the format can be round-tripped in
tests without an external PLINK install.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

__all__ = [
    "VariantTable",
    "SampleTable",
    "Genotypes",
    "read_bim",
    "read_fam",
    "read_bed",
    "read_plink",
    "write_plink",
]

# PLINK 1 .bed magic number and SNP-major mode byte.
_BED_MAGIC = bytes([0x6C, 0x1B])
_BED_SNP_MAJOR = 0x01

# Decode table: a packed 2-bit PLINK code -> A1 dosage.
#   0b00 = homozygous A1      -> 2
#   0b01 = missing            -> -1
#   0b10 = heterozygous       -> 1
#   0b11 = homozygous A2      -> 0
_CODE_TO_DOSAGE = np.array([2, -1, 1, 0], dtype=np.int8)
# Reverse map (A1 dosage + missing) -> 2-bit code, for writing.
_DOSAGE_TO_CODE = {2: 0b00, 1: 0b10, 0: 0b11, -1: 0b01}

# (256, 4) lookup: each byte holds 4 samples, sample 0 in the lowest two bits.
_BYTE_TO_DOSAGES = np.empty((256, 4), dtype=np.int8)
for _b in range(256):
    for _s in range(4):
        _BYTE_TO_DOSAGES[_b, _s] = _CODE_TO_DOSAGE[(_b >> (2 * _s)) & 0b11]


@dataclass
class VariantTable:
    """Per-variant metadata, columns of a ``.bim`` file (parallel arrays)."""

    chrom: np.ndarray   # str
    id: np.ndarray      # str (rsID or chr:pos:a1:a2)
    cm: np.ndarray      # float, genetic position (centimorgans); 0 if absent
    pos: np.ndarray     # int, base-pair position
    a1: np.ndarray      # str, first/counted allele (effect allele in many GWAS)
    a2: np.ndarray      # str, second allele

    def __len__(self):
        return len(self.id)


@dataclass
class SampleTable:
    """Per-sample metadata, columns of a ``.fam`` file (parallel arrays)."""

    fid: np.ndarray     # str, family ID
    iid: np.ndarray     # str, individual ID
    sex: np.ndarray     # int
    pheno: np.ndarray   # float

    def __len__(self):
        return len(self.iid)


@dataclass
class Genotypes:
    """A genotype matrix with aligned variant/sample tables.

    ``dosage`` is ``int8`` of shape ``(n_samples, n_variants)`` counting the A1
    allele, with ``-1`` for missing calls.
    """

    dosage: np.ndarray
    variants: VariantTable
    samples: SampleTable

    @property
    def n_samples(self):
        return self.dosage.shape[0]

    @property
    def n_variants(self):
        return self.dosage.shape[1]


def _strip_ext(prefix):
    """Allow callers to pass either ``foo`` or ``foo.bed``."""
    for ext in (".bed", ".bim", ".fam"):
        if prefix.endswith(ext):
            return prefix[: -len(ext)]
    return prefix


def read_bim(path):
    """Read a PLINK ``.bim`` variant table into a :class:`VariantTable`."""
    chrom, vid, cm, pos, a1, a2 = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split()
            if len(f) < 6:
                raise ValueError(f"{path}: expected 6 columns, got {len(f)}")
            chrom.append(f[0]); vid.append(f[1]); cm.append(float(f[2]))
            pos.append(int(f[3])); a1.append(f[4]); a2.append(f[5])
    return VariantTable(
        chrom=np.array(chrom, dtype=object),
        id=np.array(vid, dtype=object),
        cm=np.array(cm, dtype=float),
        pos=np.array(pos, dtype=np.int64),
        a1=np.array(a1, dtype=object),
        a2=np.array(a2, dtype=object),
    )


def read_fam(path):
    """Read a PLINK ``.fam`` sample table into a :class:`SampleTable`."""
    fid, iid, sex, pheno = [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split()
            if len(f) < 6:
                raise ValueError(f"{path}: expected 6 columns, got {len(f)}")
            fid.append(f[0]); iid.append(f[1])
            sex.append(int(f[4]))
            try:
                pheno.append(float(f[5]))
            except ValueError:        # PLINK uses -9 / NA for missing phenotype
                pheno.append(np.nan)
    return SampleTable(
        fid=np.array(fid, dtype=object),
        iid=np.array(iid, dtype=object),
        sex=np.array(sex, dtype=np.int64),
        pheno=np.array(pheno, dtype=float),
    )


def read_bed(path, n_samples, n_variants):
    """Decode a PLINK 1 ``.bed`` file into an ``int8`` dosage matrix.

    Returns an array of shape ``(n_samples, n_variants)`` counting A1, with
    ``-1`` for missing. ``n_samples`` and ``n_variants`` come from the matching
    ``.fam`` and ``.bim``.
    """
    bytes_per_variant = (n_samples + 3) // 4
    with open(path, "rb") as fh:
        magic = fh.read(2)
        mode = fh.read(1)
        if magic != _BED_MAGIC:
            raise ValueError(f"{path}: not a PLINK .bed file (bad magic)")
        if mode != bytes([_BED_SNP_MAJOR]):
            raise ValueError(f"{path}: only SNP-major .bed files are supported")
        raw = np.fromfile(fh, dtype=np.uint8)
    expected = bytes_per_variant * n_variants
    if raw.size != expected:
        raise ValueError(
            f"{path}: expected {expected} genotype bytes for "
            f"{n_variants} variants x {n_samples} samples, got {raw.size}")
    raw = raw.reshape(n_variants, bytes_per_variant)
    # Expand each byte to its 4 samples, then trim padding to n_samples.
    dos = _BYTE_TO_DOSAGES[raw].reshape(n_variants, bytes_per_variant * 4)
    dos = dos[:, :n_samples]
    return np.ascontiguousarray(dos.T)        # -> (n_samples, n_variants)


def read_plink(prefix):
    """Read a full PLINK 1 fileset given a path prefix (with or without ext)."""
    prefix = _strip_ext(prefix)
    variants = read_bim(prefix + ".bim")
    samples = read_fam(prefix + ".fam")
    dosage = read_bed(prefix + ".bed", len(samples), len(variants))
    return Genotypes(dosage=dosage, variants=variants, samples=samples)


def write_plink(prefix, dosage, variants, samples):
    """Write a PLINK 1 fileset (for tests / round-tripping).

    ``dosage`` is ``(n_samples, n_variants)`` counting A1 with ``-1`` missing.
    ``variants`` / ``samples`` are :class:`VariantTable` / :class:`SampleTable`.
    """
    prefix = _strip_ext(prefix)
    n_samples, n_variants = dosage.shape
    if len(variants) != n_variants or len(samples) != n_samples:
        raise ValueError("dosage shape does not match variant/sample tables")

    with open(prefix + ".bim", "w") as fh:
        for i in range(n_variants):
            fh.write(f"{variants.chrom[i]}\t{variants.id[i]}\t{variants.cm[i]:g}\t"
                     f"{variants.pos[i]}\t{variants.a1[i]}\t{variants.a2[i]}\n")
    with open(prefix + ".fam", "w") as fh:
        for i in range(n_samples):
            ph = samples.pheno[i]
            ph_s = "-9" if np.isnan(ph) else f"{ph:g}"
            fh.write(f"{samples.fid[i]}\t{samples.iid[i]}\t0\t0\t"
                     f"{samples.sex[i]}\t{ph_s}\n")

    bytes_per_variant = (n_samples + 3) // 4
    code_lut = np.zeros(4, dtype=np.uint8)    # index by (dosage+1): -1,0,1,2
    code_lut[-1 + 1] = _DOSAGE_TO_CODE[-1]
    code_lut[0 + 1] = _DOSAGE_TO_CODE[0]
    code_lut[1 + 1] = _DOSAGE_TO_CODE[1]
    code_lut[2 + 1] = _DOSAGE_TO_CODE[2]
    out = np.zeros((n_variants, bytes_per_variant), dtype=np.uint8)
    padded = np.zeros((n_variants, bytes_per_variant * 4), dtype=np.int8)
    padded[:, :n_samples] = dosage.T          # missing padding stays 0 -> homA1
    codes = code_lut[padded + 1]              # (n_variants, 4*bpv) 2-bit codes
    for s in range(4):
        out |= codes[:, s::4] << np.uint8(2 * s)
    with open(prefix + ".bed", "wb") as fh:
        fh.write(_BED_MAGIC + bytes([_BED_SNP_MAJOR]))
        out.tofile(fh)
