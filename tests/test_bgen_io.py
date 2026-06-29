"""Tests for the BGEN v1.2 reader/writer."""

import os
import sys

import numpy as np
import pytest


from ldpred3.genotype_io import VariantTable, SampleTable, write_plink, read_plink  # noqa: E402
from ldpred3.bgen_io import read_bgen, write_bgen                                    # noqa: E402


def _tables(n_samples, n_variants):
    variants = VariantTable(
        chrom=np.array(["1"] * n_variants, dtype=object),
        id=np.array([f"rs{i}" for i in range(n_variants)], dtype=object),
        cm=np.zeros(n_variants),
        pos=np.arange(1, n_variants + 1, dtype=np.int64) * 50,
        a1=np.array(["A"] * n_variants, dtype=object),
        a2=np.array(["G"] * n_variants, dtype=object))
    samples = SampleTable(
        fid=np.array([f"s{i}" for i in range(n_samples)], dtype=object),
        iid=np.array([f"s{i}" for i in range(n_samples)], dtype=object),
        sex=np.zeros(n_samples, dtype=np.int64),
        pheno=np.full(n_samples, np.nan))
    return variants, samples


@pytest.mark.parametrize("compression", [0, 1])
@pytest.mark.parametrize("nbits", [8, 16])
def test_bgen_roundtrip(tmp_path, compression, nbits):
    rng = np.random.default_rng(0)
    n_samples, n_variants = 11, 6
    dosage = rng.integers(0, 3, size=(n_samples, n_variants)).astype(np.int8)
    dosage[0, 0] = -1                      # a missing call
    variants, samples = _tables(n_samples, n_variants)

    path = str(tmp_path / "g.bgen")
    write_bgen(path, dosage, variants, samples, nbits=nbits,
               compression=compression)
    g = read_bgen(path)

    assert g.dosage.shape == (n_samples, n_variants)
    exp = dosage.astype(float)
    exp[exp < 0] = np.nan
    np.testing.assert_allclose(g.dosage, exp, atol=1e-4, equal_nan=True)
    np.testing.assert_array_equal(g.variants.id, variants.id)
    np.testing.assert_array_equal(g.samples.iid, samples.iid)


def test_bgen_embedded_sample_ids(tmp_path):
    variants, samples = _tables(5, 3)
    dosage = np.array([[2, 1, 0]] * 5, dtype=np.int8)
    path = str(tmp_path / "g.bgen")
    write_bgen(path, dosage, variants, samples)
    g = read_bgen(path)
    np.testing.assert_array_equal(g.samples.iid, samples.iid)


def test_bgen_variant_subset(tmp_path):
    rng = np.random.default_rng(9)
    n_samples, n_variants = 14, 10
    dosage = rng.integers(0, 3, size=(n_samples, n_variants)).astype(np.int8)
    variants, samples = _tables(n_samples, n_variants)
    path = str(tmp_path / "g.bgen")
    write_bgen(path, dosage, variants, samples)

    full = read_bgen(path)
    sub = read_bgen(path, variant_ids=["rs1", "rs7", "rs8"])
    np.testing.assert_array_equal(sub.variants.id, ["rs1", "rs7", "rs8"])
    np.testing.assert_allclose(sub.dosage, full.dosage[:, [1, 7, 8]], atol=1e-4)


def test_bgen_matches_plink_dosage(tmp_path):
    """Same genotypes via PLINK and BGEN must give identical dosages."""
    rng = np.random.default_rng(3)
    n_samples, n_variants = 20, 8
    dosage = rng.integers(0, 3, size=(n_samples, n_variants)).astype(np.int8)
    variants, samples = _tables(n_samples, n_variants)

    write_plink(str(tmp_path / "p"), dosage, variants, samples)
    write_bgen(str(tmp_path / "b.bgen"), dosage, variants, samples)
    gp = read_plink(str(tmp_path / "p"))
    gb = read_bgen(str(tmp_path / "b.bgen"))

    np.testing.assert_allclose(gb.dosage, gp.dosage.astype(float), atol=1e-4)
