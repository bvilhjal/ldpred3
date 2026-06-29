"""Tests for the PLINK genotype reader/writer."""

import os
import sys

import numpy as np


from ldpred3.genotype_io import (         # noqa: E402
    VariantTable, SampleTable, read_plink, write_plink, read_bed,
    _BED_MAGIC, _BED_SNP_MAJOR,
)


def _make_tables(n_samples, n_variants, rng):
    variants = VariantTable(
        chrom=np.array([str(1 + i % 22) for i in range(n_variants)], dtype=object),
        id=np.array([f"rs{i}" for i in range(n_variants)], dtype=object),
        cm=np.zeros(n_variants),
        pos=np.arange(1, n_variants + 1, dtype=np.int64) * 1000,
        a1=np.array(["A"] * n_variants, dtype=object),
        a2=np.array(["G"] * n_variants, dtype=object),
    )
    samples = SampleTable(
        fid=np.array([f"F{i}" for i in range(n_samples)], dtype=object),
        iid=np.array([f"I{i}" for i in range(n_samples)], dtype=object),
        sex=rng.integers(1, 3, n_samples),
        pheno=rng.standard_normal(n_samples),
    )
    return variants, samples


def test_roundtrip_all_genotype_states(tmp_path):
    rng = np.random.default_rng(0)
    n_samples, n_variants = 13, 7      # 13 % 4 != 0 -> exercises byte padding
    dosage = rng.integers(-1, 3, size=(n_samples, n_variants)).astype(np.int8)
    # Guarantee every state (incl. missing) is present somewhere.
    dosage[0, 0] = -1; dosage[1, 0] = 0; dosage[2, 0] = 1; dosage[3, 0] = 2
    variants, samples = _make_tables(n_samples, n_variants, rng)

    prefix = str(tmp_path / "geno")
    write_plink(prefix, dosage, variants, samples)
    g = read_plink(prefix)

    assert g.dosage.shape == (n_samples, n_variants)
    assert g.dosage.dtype == np.int8
    np.testing.assert_array_equal(g.dosage, dosage)
    np.testing.assert_array_equal(g.variants.pos, variants.pos)
    np.testing.assert_array_equal(g.variants.a1, variants.a1)
    np.testing.assert_array_equal(g.samples.iid, samples.iid)


def test_read_plink_variant_subset_via_seek(tmp_path):
    rng = np.random.default_rng(7)
    n_samples, n_variants = 17, 12
    dosage = rng.integers(-1, 3, size=(n_samples, n_variants)).astype(np.int8)
    variants, samples = _make_tables(n_samples, n_variants, rng)
    prefix = str(tmp_path / "geno")
    write_plink(prefix, dosage, variants, samples)

    wanted = ["rs2", "rs5", "rs9"]                 # subset, out of order
    g = read_plink(prefix, variant_ids=wanted)
    # Returned in file order (rs2, rs5, rs9 -> columns 2, 5, 9).
    np.testing.assert_array_equal(g.variants.id, ["rs2", "rs5", "rs9"])
    np.testing.assert_array_equal(g.dosage, dosage[:, [2, 5, 9]])
    # Unknown IDs are simply skipped.
    g2 = read_plink(prefix, variant_ids=["rs5", "nope"])
    np.testing.assert_array_equal(g2.variants.id, ["rs5"])
    np.testing.assert_array_equal(g2.dosage, dosage[:, [5]])


def test_bed_decode_known_bits(tmp_path):
    """Decode a hand-built .bed against the documented 2-bit mapping."""
    # 4 samples, 1 variant. Codes (sample0..3): hom A1, missing, het, hom A2.
    #   00, 01, 10, 11  -> packed low-to-high in one byte: 0b11100100 = 0xE4
    path = str(tmp_path / "x.bed")
    with open(path, "wb") as fh:
        fh.write(_BED_MAGIC + bytes([_BED_SNP_MAJOR]) + bytes([0xE4]))
    dos = read_bed(path, n_samples=4, n_variants=1)
    np.testing.assert_array_equal(dos[:, 0], [2, -1, 1, 0])


def test_bad_magic_raises(tmp_path):
    path = str(tmp_path / "bad.bed")
    with open(path, "wb") as fh:
        fh.write(bytes([0x00, 0x00, 0x01, 0x00]))
    try:
        read_bed(path, 4, 1)
    except ValueError as e:
        assert "magic" in str(e)
    else:
        raise AssertionError("expected ValueError on bad magic")


def test_wrong_size_raises(tmp_path):
    path = str(tmp_path / "short.bed")
    with open(path, "wb") as fh:
        fh.write(_BED_MAGIC + bytes([_BED_SNP_MAJOR]) + bytes([0x00]))
    try:
        read_bed(path, n_samples=4, n_variants=5)   # needs 5 bytes, has 1
    except ValueError as e:
        assert "expected" in str(e)
    else:
        raise AssertionError("expected ValueError on truncated .bed")
