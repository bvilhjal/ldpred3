"""Tests for the sumstats reader and harmonisation rules."""


import numpy as np


from ldpred3.sumstats import read_sumstats                      # noqa: E402
from ldpred3.harmonize import harmonize, _complement, _is_palindromic   # noqa: E402
from ldpred3.genotype_io import VariantTable                    # noqa: E402


def _write(tmp_path, text, name="ss.txt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_read_sumstats_tab_and_aliases(tmp_path):
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP\tN\n"
        "rs1\t1\t100\tA\tG\t0.10\t0.02\t1e-6\t1000\n"
        "rs2\t1\t200\tC\tT\t-0.05\t0.01\t2e-7\t1000\n")
    ss = read_sumstats(path)
    assert len(ss) == 2
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2"])
    np.testing.assert_allclose(ss.beta, [0.10, -0.05])
    np.testing.assert_allclose(ss.se, [0.02, 0.01])
    np.testing.assert_array_equal(ss.ea, ["A", "C"])


def test_read_sumstats_or_to_beta(tmp_path):
    path = _write(tmp_path,
        "SNP,A1,A2,OR,SE,N\n"
        "rs1,A,G,1.105170918,0.02,500\n")     # log(OR) = 0.1
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.beta, [0.1], atol=1e-6)


def test_read_sumstats_se_from_pvalue(tmp_path):
    # No SE column: recover from beta and p (z = 1.959964 at p=0.05).
    path = _write(tmp_path,
        "SNP A1 A2 BETA P N\n"
        "rs1 A G 0.1 0.05 1000\n")
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.se, [0.1 / 1.959964], rtol=1e-4)


def test_read_sumstats_se_from_tiny_pvalue(tmp_path):
    # Very small p-values (common in GWAS): 1 - p/2 rounds to 1.0 and the naive
    # inv_cdf(1.0) raises. The reader must still recover a finite, sane SE.
    path = _write(tmp_path,
        "SNP A1 A2 BETA P N\n"
        "rs1 A G 0.5 1e-300 100000\n"
        "rs2 A G 0.2 5e-20 100000\n")
    ss = read_sumstats(path)
    assert np.all(np.isfinite(ss.se)) and np.all(ss.se > 0)
    # larger z (smaller p) -> smaller SE for comparable beta scale
    assert ss.se[0] < ss.se[1]


def test_read_sumstats_external_n(tmp_path):
    path = _write(tmp_path, "SNP A1 A2 BETA SE\nrs1 A G 0.1 0.02\n")
    ss = read_sumstats(path, n_eff=2000)
    np.testing.assert_allclose(ss.n_eff, [2000])


def _variants():
    return VariantTable(
        chrom=np.array(["1", "1", "1", "1", "1"], dtype=object),
        id=np.array(["rs1", "rs2", "rs3", "rs4", "rs5"], dtype=object),
        cm=np.zeros(5),
        pos=np.array([100, 200, 300, 400, 500], dtype=np.int64),
        a1=np.array(["A", "C", "A", "A", "G"], dtype=object),   # counted allele
        a2=np.array(["G", "T", "C", "G", "C"], dtype=object),
    )


def test_harmonize_alignment_rules(tmp_path):
    # rs1: ea==A1            -> keep beta
    # rs2: ea==A2 (swapped)  -> flip sign
    # rs3: strand flip (T/A complements A/T... actually use C/G via comp)
    # rs4: palindromic A/T   -> dropped (ambiguous)
    # rs5: allele mismatch   -> dropped
    path = _write(tmp_path,
        "SNP A1 A2 BETA SE N\n"
        "rs1 A G 0.20 0.01 1000\n"     # exact
        "rs2 T C -0.30 0.01 1000\n"    # swapped (A1=C,A2=T) -> ea=T==A2 -> flip
        "rs3 T G 0.40 0.01 1000\n"     # geno A/C; comp(T)=A,comp(G)=C -> A/C keep
        "rs4 A T 0.50 0.01 1000\n"     # geno A/G but ss A/T palindrome -> ambiguous
        "rs5 C A 0.60 0.01 1000\n")    # geno G/C, ss C/A -> mismatch
    ss = read_sumstats(path)
    h = harmonize(ss, _variants())

    kept = dict(zip(h.var_index.tolist(), h.beta.tolist()))
    assert 0 in kept and np.isclose(kept[0], 0.20)     # rs1 keep
    assert 1 in kept and np.isclose(kept[1], 0.30)     # rs2 flipped -0.30 -> +0.30
    assert 2 in kept and np.isclose(kept[2], 0.40)     # rs3 strand, same order
    assert 3 not in kept                               # rs4 dropped ambiguous
    assert 4 not in kept                               # rs5 dropped mismatch
    assert h.log["n_flipped"] == 1
    assert h.log["n_strand_flipped"] == 1
    assert h.log["n_dropped_ambiguous"] == 1
    assert h.log["n_dropped_mismatch"] == 1
    # var_index is sorted (genotype-column order)
    assert list(h.var_index) == sorted(h.var_index)


def test_harmonize_match_by_position_when_no_id(tmp_path):
    path = _write(tmp_path,
        "CHR BP A1 A2 BETA SE N\n"
        "1 200 C T 0.7 0.01 1000\n")    # no rsID -> match by chrom:pos -> rs2
    ss = read_sumstats(path)
    h = harmonize(ss, _variants())
    assert list(h.var_index) == [1]
    np.testing.assert_allclose(h.beta, [0.7])


def test_complement_and_palindrome_helpers():
    assert _complement("AC") == "GT"
    assert _complement("N") is None
    assert _is_palindromic("A", "T")
    assert not _is_palindromic("A", "G")
