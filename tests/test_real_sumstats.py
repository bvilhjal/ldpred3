"""Parsing real-world GWAS summary-statistics formats.

These fixtures mimic the header conventions of actual tools/consortia (GWAS
Catalog harmonised, BOLT-LMM, METAL, PGC/case-control OR files, SAIGE) rather
than the idealised synthetic format. They exercise the column-alias map, OR->beta
and SE-from-p conversions, allele upper-casing, the gzip path and a few messy
rows. No network / large files: small inline fixtures stand in for the real
multi-GB downloads.
"""

import gzip
import math

import numpy as np

from pyldpred2.sumstats import read_sumstats, detect_columns


def _write(tmp_path, text, name="ss.txt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_gwas_catalog_harmonised(tmp_path):
    # GWAS Catalog harmonised TSV: variant_id / base_pair_location / *_allele /
    # standard_error / p_value / effect_allele_frequency.
    path = _write(tmp_path,
        "variant_id\tchromosome\tbase_pair_location\teffect_allele\tother_allele"
        "\tbeta\tstandard_error\tp_value\teffect_allele_frequency\tn\n"
        "rs1\t1\t10000\tA\tG\t0.0321\t0.0067\t1.2e-6\t0.41\t250000\n"
        "rs2\t2\t20000\tT\tC\t-0.0150\t0.0070\t3.2e-2\t0.12\t250000\n")
    ss = read_sumstats(path)
    assert len(ss) == 2
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2"])
    np.testing.assert_array_equal(ss.chrom, ["1", "2"])
    np.testing.assert_array_equal(ss.pos, [10000, 20000])
    np.testing.assert_array_equal(ss.ea, ["A", "T"])
    np.testing.assert_array_equal(ss.oa, ["G", "C"])
    np.testing.assert_allclose(ss.beta, [0.0321, -0.0150])
    np.testing.assert_allclose(ss.se, [0.0067, 0.0070])
    np.testing.assert_allclose(ss.eaf, [0.41, 0.12])
    np.testing.assert_allclose(ss.n_eff, [250000, 250000])


def test_bolt_lmm(tmp_path):
    # BOLT-LMM: ALLELE1 is the effect allele, ALLELE0 the other; P_BOLT_LMM;
    # A1FREQ; INFO; no per-variant N column (passed externally).
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tGENPOS\tALLELE1\tALLELE0\tA1FREQ\tINFO\tBETA\tSE\tP_BOLT_LMM\n"
        "rs10\t1\t500\t0.0\tA\tG\t0.27\t0.98\t0.012\t0.003\t6.1e-5\n"
        "rs11\t1\t900\t0.0\tC\tT\t0.55\t0.91\t-0.008\t0.004\t4.5e-2\n")
    ss = read_sumstats(path, n_eff=400000)
    np.testing.assert_array_equal(ss.ea, ["A", "C"])     # ALLELE1 -> effect
    np.testing.assert_array_equal(ss.oa, ["G", "T"])     # ALLELE0 -> other
    np.testing.assert_allclose(ss.beta, [0.012, -0.008])
    np.testing.assert_allclose(ss.eaf, [0.27, 0.55])
    np.testing.assert_allclose(ss.info, [0.98, 0.91])
    np.testing.assert_allclose(ss.n_eff, [400000, 400000])


def test_metal_lowercase_alleles(tmp_path):
    # METAL: MarkerName / Allele1 / Allele2 / Effect / StdErr / P-value, with
    # lower-case alleles (METAL emits them lower-case) -> must be upper-cased.
    path = _write(tmp_path,
        "MarkerName\tAllele1\tAllele2\tEffect\tStdErr\tP-value\tTotalSampleSize\n"
        "rs5\ta\tg\t0.20\t0.05\t6e-5\t80000\n")
    ss = read_sumstats(path)
    np.testing.assert_array_equal(ss.ea, ["A"])
    np.testing.assert_array_equal(ss.oa, ["G"])
    np.testing.assert_allclose(ss.beta, [0.20])
    np.testing.assert_allclose(ss.n_eff, [80000])


def test_pgc_or_whitespace(tmp_path):
    # PGC-style case/control, whitespace-delimited, effect as an odds ratio.
    path = _write(tmp_path,
        "SNPID CHR POS A1 A2 OR SE P Neff\n"
        "rs7 6 3000 A G 2.0 0.08 1e-9 50000\n")
    ss = read_sumstats(path)
    np.testing.assert_allclose(ss.beta, [math.log(2.0)], atol=1e-9)   # OR -> log OR
    np.testing.assert_allclose(ss.n_eff, [50000])
    np.testing.assert_array_equal(ss.ea, ["A"])


def test_saige_se_from_pvalue(tmp_path):
    # No SE column: derive it from BETA and the p-value (z = Phi^-1(1-p/2)).
    p = 1e-8
    path = _write(tmp_path,
        "SNP,CHR,POS,Allele1,Allele2,BETA,p.value,N\n"
        f"rs9,3,400,A,G,0.10,{p},120000\n")
    ss = read_sumstats(path)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - p / 2.0)
    np.testing.assert_allclose(ss.se, [0.10 / z], rtol=1e-6)


def test_gzipped_file(tmp_path):
    # Real sumstats are usually shipped gzipped.
    p = tmp_path / "gwas.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("SNP\tA1\tA2\tBETA\tSE\tN\n"
                 "rs1\tA\tG\t0.1\t0.02\t1000\n"
                 "rs2\tC\tT\t-0.2\t0.03\t1000\n")
    ss = read_sumstats(str(p))
    assert len(ss) == 2
    np.testing.assert_allclose(ss.beta, [0.1, -0.2])


def test_detect_columns_on_bolt(tmp_path):
    # The preflight column detector should map a real BOLT header correctly.
    path = _write(tmp_path,
        "SNP\tCHR\tBP\tALLELE1\tALLELE0\tA1FREQ\tBETA\tSE\tP_BOLT_LMM\n"
        "rs1\t1\t1\tA\tG\t0.3\t0.01\t0.002\t1e-3\n")
    header, mapping = detect_columns(path)
    assert mapping["ea"] == "ALLELE1"
    assert mapping["oa"] == "ALLELE0"
    assert mapping["beta"] == "BETA"
    assert mapping["pval"] == "P_BOLT_LMM"
    assert mapping["eaf"] == "A1FREQ"


def test_explicit_column_override(tmp_path):
    # Ambiguous/non-standard header resolved by explicit overrides.
    path = _write(tmp_path,
        "id\tref\talt\tEFF\tSEBETA\tnsamp\n"          # 'ref'/'alt' swapped vs usual
        "rs1\tG\tA\t0.05\t0.01\t1000\n")
    ss = read_sumstats(path, ea="alt", oa="ref", beta="EFF",
                       se="SEBETA", n_eff="nsamp")
    np.testing.assert_array_equal(ss.ea, ["A"])
    np.testing.assert_array_equal(ss.oa, ["G"])
    np.testing.assert_allclose(ss.beta, [0.05])
    np.testing.assert_allclose(ss.n_eff, [1000])


def test_varying_per_variant_sample_size(tmp_path):
    # Meta-analysis sumstats: N differs per SNP. It must parse per-variant and
    # flow through the samplers / LDSC; and the per-SNP N path must reduce to the
    # constant fast path when N happens to be constant.
    from pyldpred2 import ldpred2_by_blocks, ld_scores, ldsc_h2

    path = _write(tmp_path,
        "SNP\tA1\tA2\tBETA\tSE\tN\n"
        "rs0\tA\tG\t0.05\t0.02\t10000\n"
        "rs1\tC\tT\t0.10\t0.02\t30000\n"
        "rs2\tA\tT\t0.15\t0.02\t60000\n")
    ss = read_sumstats(path)
    np.testing.assert_array_equal(ss.n_eff, [10000, 30000, 60000])
    assert ss.n_eff.min() != ss.n_eff.max()

    m, k = 200, 100
    rng = np.random.default_rng(0)
    blocks = [(np.eye(k, dtype=np.float32), np.arange(b * k, (b + 1) * k))
              for b in range(2)]
    bhat = rng.normal(0, 0.01, m)
    n_vec = np.linspace(8000, 60000, m)
    be = ldpred2_by_blocks(blocks, bhat, n_vec, method="auto",
                           burn_in=40, num_iter=60, seed=1)
    assert np.all(np.isfinite(be))                       # varying N runs
    h = ldsc_h2(n_vec * bhat ** 2, ld_scores(blocks), n_vec, n_blocks=20)
    assert np.isfinite(h.h2)                             # LDSC accepts a vector N

    # constant vector N == scalar N (per-SNP path matches the fast path)
    v = ldpred2_by_blocks(blocks, bhat, np.full(m, 3e4), method="auto",
                          burn_in=40, num_iter=60, seed=2)
    s = ldpred2_by_blocks(blocks, bhat, 3e4, method="auto",
                          burn_in=40, num_iter=60, seed=2)
    np.testing.assert_allclose(v, s)


def test_messy_rows_blank_and_extra_columns(tmp_path):
    # Blank lines tolerated; unknown trailing columns ignored.
    path = _write(tmp_path,
        "SNP\tA1\tA2\tBETA\tSE\tN\tEXTRA\tNOTE\n"
        "rs1\tA\tG\t0.1\t0.02\t1000\t99\thello\n"
        "\n"
        "rs2\tC\tT\t-0.2\t0.03\t1000\t98\tworld\n")
    ss = read_sumstats(path)
    assert len(ss) == 2
    np.testing.assert_array_equal(ss.id, ["rs1", "rs2"])
    np.testing.assert_allclose(ss.beta, [0.1, -0.2])
