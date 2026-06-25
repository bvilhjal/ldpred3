"""
BGEN genotype reader (NumPy + stdlib ``zlib`` only).

Supports the common case found in UK-Biobank-style imputed data:

* BGEN **v1.2, layout 2**
* compression: **none** or **zlib** (zstd needs the optional ``zstandard``
  package and raises a clear error here)
* **biallelic, unphased, diploid** variants

For each variant the per-sample genotype probabilities ``(P(AA), P(AB), P(BB))``
are decoded and reduced to an A1 **dosage** ``2*P(AA) + P(AB)`` -- counting the
*first* listed allele, to match the PLINK reader's convention. Dosages are
continuous in ``[0, 2]``; missing samples become ``NaN``. The reader returns the
same :class:`~genotype_io.Genotypes` bundle as the PLINK reader, with a
``float32`` dosage matrix.

A restricted :func:`write_bgen` (integer dosages -> one-hot probabilities)
is provided so the format round-trips in tests without an external tool.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from .genotype_io import VariantTable, SampleTable, Genotypes

__all__ = ["read_bgen", "write_bgen"]

_MAGIC = b"bgen"


def _decompress(comp, raw, expected_len):
    if comp == 0:
        return raw
    if comp == 1:
        return zlib.decompress(raw)
    if comp == 2:
        raise NotImplementedError(
            "this BGEN uses zstd compression; install the 'zstandard' package "
            "and extend bgen_io, or recompress with zlib/none")
    raise ValueError(f"unknown BGEN compression flag {comp}")


def _decode_probs_to_dosage(block, n_samples):
    """Decode one layout-2 unphased diploid biallelic genotype block -> dosage."""
    n = struct.unpack_from("<I", block, 0)[0]
    k = struct.unpack_from("<H", block, 4)[0]
    pmin = block[6]
    pmax = block[7]
    if n != n_samples:
        raise ValueError("variant sample count disagrees with header")
    if k != 2:
        raise NotImplementedError("only biallelic variants are supported")
    ploidy_miss = np.frombuffer(block, np.uint8, count=n_samples, offset=8)
    off = 8 + n_samples
    phased = block[off]
    nbits = block[off + 1]
    off += 2
    if phased != 0:
        raise NotImplementedError("only unphased variants are supported")
    ploidy = ploidy_miss & 0x3F
    missing = (ploidy_miss & 0x80) != 0
    if np.any(ploidy[~missing] != 2):
        raise NotImplementedError("only diploid samples are supported")

    n_values = 2 * n_samples            # (Z-1)=2 stored probs per sample
    bits = np.unpackbits(
        np.frombuffer(block, np.uint8, offset=off), bitorder="little")
    need = n_values * nbits
    if bits.size < need:
        raise ValueError("truncated BGEN probability block")
    vals = bits[:need].reshape(n_values, nbits).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(nbits, dtype=np.uint64))
    ints = vals @ weights
    probs = ints.astype(np.float64) / float((1 << nbits) - 1)
    probs = probs.reshape(n_samples, 2)
    p_aa, p_ab = probs[:, 0], probs[:, 1]
    dosage = 2.0 * p_aa + p_ab          # count allele 0 (A1)
    dosage[missing] = np.nan
    return dosage.astype(np.float32)


def read_bgen(path, sample_path=None):
    """Read a BGEN v1.2 (layout 2) file into a :class:`Genotypes` bundle.

    ``sample_path`` (an Oxford ``.sample`` file) supplies sample IDs when the
    BGEN itself has no embedded sample-identifier block.
    """
    with open(path, "rb") as fh:
        data = fh.read()

    offset = struct.unpack_from("<I", data, 0)[0]
    header_len = struct.unpack_from("<I", data, 4)[0]
    n_variants = struct.unpack_from("<I", data, 8)[0]
    n_samples = struct.unpack_from("<I", data, 12)[0]
    magic = data[16:20]
    if magic not in (_MAGIC, b"\x00\x00\x00\x00"):
        raise ValueError(f"{path}: not a BGEN file (bad magic {magic!r})")
    flags = struct.unpack_from("<I", data, 4 + header_len - 4)[0]
    compression = flags & 0x3
    layout = (flags >> 2) & 0xF
    has_sample_ids = (flags >> 31) & 0x1
    if layout != 2:
        raise NotImplementedError(f"{path}: only BGEN layout 2 is supported")

    pos = 4 + header_len
    sample_ids = None
    if has_sample_ids:
        pos += 4                                  # length of sample-id block
        n2 = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        ids = []
        for _ in range(n2):
            ln = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            ids.append(data[pos:pos + ln].decode())
            pos += ln
        sample_ids = ids

    pos = 4 + offset                              # start of variant blocks

    chrom, vid, posn, a1, a2 = [], [], [], [], []
    dosage = np.empty((n_samples, n_variants), dtype=np.float32)

    def read_str():
        nonlocal pos
        ln = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        s = data[pos:pos + ln].decode()
        pos += ln
        return s

    for v in range(n_variants):
        vid.append(read_str())                    # variant ID
        rsid = read_str()                         # rsID
        vid[-1] = rsid or vid[-1]
        chrom.append(read_str())
        posn.append(struct.unpack_from("<I", data, pos)[0]); pos += 4
        k = struct.unpack_from("<H", data, pos)[0]; pos += 2
        alleles = []
        for _ in range(k):
            ln = struct.unpack_from("<I", data, pos)[0]; pos += 4
            alleles.append(data[pos:pos + ln].decode()); pos += ln
        a1.append(alleles[0]); a2.append(alleles[1] if k > 1 else "")

        clen = struct.unpack_from("<I", data, pos)[0]; pos += 4
        if compression != 0:
            ulen = struct.unpack_from("<I", data, pos)[0]
            comp = data[pos + 4:pos + clen]
            block = _decompress(compression, comp, ulen)
        else:
            block = data[pos:pos + clen]
        pos += clen
        dosage[:, v] = _decode_probs_to_dosage(block, n_samples)

    if sample_ids is None:
        sample_ids = _read_sample_file(sample_path, n_samples)

    samples = SampleTable(
        fid=np.array(sample_ids, dtype=object),
        iid=np.array(sample_ids, dtype=object),
        sex=np.zeros(n_samples, dtype=np.int64),
        pheno=np.full(n_samples, np.nan))
    variants = VariantTable(
        chrom=np.array(chrom, dtype=object),
        id=np.array(vid, dtype=object),
        cm=np.zeros(n_variants),
        pos=np.array(posn, dtype=np.int64),
        a1=np.array(a1, dtype=object),
        a2=np.array(a2, dtype=object))
    return Genotypes(dosage=dosage, variants=variants, samples=samples)


def _read_sample_file(sample_path, n_samples):
    if sample_path is None:
        return [f"sample_{i}" for i in range(n_samples)]
    ids = []
    with open(sample_path) as fh:
        lines = [ln.split() for ln in fh if ln.strip()]
    # Oxford .sample: 2 header rows, then ID_1 ID_2 ... per sample.
    for f in lines[2:]:
        ids.append(f[1] if len(f) > 1 else f[0])
    if len(ids) != n_samples:
        raise ValueError(".sample file sample count disagrees with BGEN")
    return ids


# --------------------------------------------------------------------------- #
# Minimal writer (integer dosages -> one-hot probabilities) for tests.
# --------------------------------------------------------------------------- #
def write_bgen(path, dosage, variants, samples, *, nbits=8, compression=1):
    """Write a restricted BGEN v1.2/layout-2 file (for tests / round-tripping).

    ``dosage`` counts A1 and must be integer 0/1/2 or NaN (missing); each call
    is encoded as one-hot genotype probabilities.
    """
    n_samples, n_variants = dosage.shape
    maxv = (1 << nbits) - 1

    sample_block = struct.pack("<I", n_samples)
    for iid in samples.iid:
        b = str(iid).encode()
        sample_block += struct.pack("<H", len(b)) + b
    sample_block = struct.pack("<I", len(sample_block) + 4) + sample_block

    variant_bytes = b""
    for v in range(n_variants):
        rs = str(variants.id[v]).encode()
        ch = str(variants.chrom[v]).encode()
        al1 = str(variants.a1[v]).encode()
        al2 = str(variants.a2[v]).encode()
        vb = (struct.pack("<H", 0) +                      # empty variant ID
              struct.pack("<H", len(rs)) + rs +
              struct.pack("<H", len(ch)) + ch +
              struct.pack("<I", int(variants.pos[v])) +
              struct.pack("<H", 2) +
              struct.pack("<I", len(al1)) + al1 +
              struct.pack("<I", len(al2)) + al2)

        ploidy = np.full(n_samples, 2, dtype=np.uint8)
        col = dosage[:, v].astype(float)
        miss = ~np.isfinite(col) | (col < 0)
        ploidy[miss] |= 0x80
        # one-hot probs: dosage 2->(1,0), 1->(0,1), 0->(0,0); missing->(0,0)
        d = np.where(miss, 0, col).astype(int)
        p0 = (d == 2).astype(np.uint64) * maxv
        p1 = (d == 1).astype(np.uint64) * maxv
        vals = np.empty(2 * n_samples, dtype=np.uint64)
        vals[0::2] = p0
        vals[1::2] = p1
        bitrows = ((vals[:, None] >> np.arange(nbits, dtype=np.uint64)) & 1
                   ).astype(np.uint8)
        prob_bytes = np.packbits(bitrows.reshape(-1), bitorder="little").tobytes()

        block = (struct.pack("<I", n_samples) + struct.pack("<H", 2) +
                 bytes([2, 2]) + ploidy.tobytes() + bytes([0, nbits]) +
                 prob_bytes)
        if compression == 1:
            comp = zlib.compress(block)
            gb = struct.pack("<I", len(comp) + 4) + struct.pack("<I", len(block)) + comp
        else:
            gb = struct.pack("<I", len(block)) + block
        variant_bytes += vb + gb

    header = (struct.pack("<I", 20) + struct.pack("<I", n_variants) +
              struct.pack("<I", n_samples) + _MAGIC +
              struct.pack("<I", (1 << 31) | (2 << 2) | (compression & 0x3)))
    offset = len(header) + len(sample_block)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", offset))
        fh.write(header)
        fh.write(sample_block)
        fh.write(variant_bytes)
