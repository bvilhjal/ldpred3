"""Usability features: preflight, weight save/reuse, LD caching."""

import numpy as np

from ldpred3.pipeline import (run_ldpred3_prs, preflight_prs,
                                score_from_weights)

from test_pipeline import _simulate


def test_preflight_reports_columns_and_match(tmp_path):
    prefix, ss_path, _ = _simulate(tmp_path, m=300, seed=1)
    rep = preflight_prs(ss_path, prefix)
    assert rep["missing"] == []
    # alias detection: SNP/A1/A2/BETA/SE/N -> canonical fields
    assert rep["columns"]["beta"] == "BETA"
    assert rep["columns"]["ea"] == "A1"
    assert rep["columns"]["n_eff"] == "N"
    assert rep["harmonize"]["n_matched"] > 250          # most variants match
    assert rep["warnings"] == []


def test_preflight_flags_missing_columns(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("SNP\tBETA\n" + "rs1\t0.1\n")        # no allele / N columns
    rep = preflight_prs(str(bad), "ignored")
    assert "ea" in rep["missing"] and "oa" in rep["missing"]


def test_weights_roundtrip_reproduces_scores(tmp_path):
    prefix, ss_path, _ = _simulate(tmp_path, m=400, seed=2)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=200)
    wpath = str(tmp_path / "prs.weights.txt")
    res.write_weights(wpath)

    sr = score_from_weights(wpath, prefix)
    assert sr.n_matched == len(res.beta_adjusted)
    # scoring from saved weights reproduces the pipeline's scores exactly
    assert np.allclose(sr.scores, res.scores, atol=1e-6)


def test_weights_scoring_handles_allele_flips(tmp_path):
    """Weights should still apply after the target swaps A1/A2."""
    prefix, ss_path, _ = _simulate(tmp_path, m=300, seed=3)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=150)
    wpath = str(tmp_path / "w.txt")
    res.write_weights(wpath)
    base = score_from_weights(wpath, prefix)

    # Build a target whose alleles are swapped relative to the weights and whose
    # dosage is therefore 2-g; the harmonised score should match the original.
    from ldpred3.genotype_io import read_plink, write_plink, VariantTable
    g = read_plink(prefix)
    V = g.variants
    swapped = VariantTable(chrom=V.chrom, id=V.id, cm=V.cm, pos=V.pos,
                           a1=V.a2, a2=V.a1)
    dos = g.dosage.copy()
    dos[dos >= 0] = 2 - dos[dos >= 0]
    sw_prefix = str(tmp_path / "swapped")
    write_plink(sw_prefix, dos, swapped, g.samples)

    sw = score_from_weights(wpath, sw_prefix)
    assert np.allclose(sw.scores, base.scores, atol=1e-6)


def test_ld_cache_reproduces_fresh_run(tmp_path):
    prefix, ss_path, _ = _simulate(tmp_path, m=400, seed=4)
    cache = str(tmp_path / "ld.npz")
    fresh = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                            ld_out=cache, seed=1)
    cached = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                             ld_cache=cache, seed=1)
    # same variants, same weights, same scores from the reloaded LD
    assert np.array_equal(fresh.variant_id, cached.variant_id)
    assert np.allclose(fresh.beta_adjusted, cached.beta_adjusted, atol=1e-6)
    assert np.allclose(fresh.scores, cached.scores, atol=1e-6)


def test_ld_cache_rejects_changed_variant_set(tmp_path):
    import pytest
    prefix, ss_path, _ = _simulate(tmp_path, m=400, seed=5)
    cache = str(tmp_path / "ld.npz")
    run_ldpred3_prs(ss_path, prefix, method="inf", block_size=200, ld_out=cache)
    # Truncate the sumstats so the harmonised set lacks the cached variants:
    # the cache no longer applies and should raise a clear error.
    lines = open(ss_path).read().splitlines()
    small = str(tmp_path / "small.txt")
    open(small, "w").write("\n".join(lines[:50]) + "\n")
    with pytest.raises(ValueError, match="cache"):
        run_ldpred3_prs(small, prefix, method="inf", block_size=200,
                        ld_cache=cache)
