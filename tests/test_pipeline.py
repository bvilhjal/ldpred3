"""End-to-end pipeline test: simulate -> GWAS -> PLINK+sumstats -> PRS."""

import os
import sys

import numpy as np


from pyldpred2.genotype_io import VariantTable, SampleTable, write_plink   # noqa: E402
from pyldpred2.bgen_io import write_bgen                                   # noqa: E402
from pyldpred2.pipeline import run_ldpred2_prs                             # noqa: E402


def _simulate(tmp_path, n_train=4000, n_target=1500, m=600, p_causal=0.1,
              h2=0.5, seed=0):
    """Make a train cohort (for GWAS) and a target cohort sharing variants."""
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.1, 0.9, m)

    def draw(n):
        return rng.binomial(2, af, size=(n, m)).astype(np.int8)

    G_tr, G_te = draw(n_train), draw(n_target)

    # True standardized effects on a sparse set of causal variants.
    causal = rng.random(m) < p_causal
    if not causal.any():
        causal[rng.integers(m)] = True
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())

    def standardize(G):
        Z = G.astype(float)
        Z -= Z.mean(0); Z /= Z.std(0)
        return Z

    Ztr = standardize(G_tr)
    g_tr = Ztr @ beta
    y_tr = g_tr + rng.normal(0, np.sqrt(1 - g_tr.var()), n_train)

    # Marginal GWAS on the training cohort (per standardized variant).
    bhat = (Ztr.T @ y_tr) / n_train
    se = np.sqrt((y_tr.var() - bhat ** 2 * 0) / n_train)  # ~ 1/sqrt(N)
    se = np.full(m, 1 / np.sqrt(n_train))

    # True target genetic value (for evaluation).
    g_te = standardize(G_te) @ beta

    # Variant + sample tables.
    a1 = np.array(["A"] * m, dtype=object)
    a2 = np.array(["G"] * m, dtype=object)
    variants = VariantTable(
        chrom=np.array(["1"] * m, dtype=object),
        id=np.array([f"rs{i}" for i in range(m)], dtype=object),
        cm=np.zeros(m), pos=np.arange(1, m + 1, dtype=np.int64) * 100,
        a1=a1, a2=a2)

    def samples(n, tag):
        return SampleTable(
            fid=np.array([f"{tag}{i}" for i in range(n)], dtype=object),
            iid=np.array([f"{tag}{i}" for i in range(n)], dtype=object),
            sex=np.ones(n, dtype=np.int64), pheno=np.full(n, np.nan))

    prefix = str(tmp_path / "target")
    smp = samples(n_target, "T")
    write_plink(prefix, G_te, variants, smp)
    write_bgen(str(tmp_path / "target.bgen"), G_te, variants, smp)

    ss_path = str(tmp_path / "gwas.txt")
    with open(ss_path, "w") as fh:
        fh.write("SNP\tA1\tA2\tBETA\tSE\tN\n")
        for i in range(m):
            fh.write(f"rs{i}\tA\tG\t{bhat[i]:.6g}\t{se[i]:.6g}\t{n_train}\n")

    return prefix, ss_path, g_te


def test_end_to_end_prs_predicts_genetic_value(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, seed=1)
    res = run_ldpred2_prs(ss_path, prefix, method="auto", block_size=200,
                          num_iter=150, burn_in=50, seed=1)

    assert res.scores.shape[0] == len(g_te)
    assert res.harmonize_log["n_matched"] == 600
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 vs true genetic value too low: {r2:.3f}"


def test_end_to_end_prs_via_bgen(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, seed=1)
    res = run_ldpred2_prs(ss_path, str(tmp_path / "target.bgen"), method="auto",
                          block_size=200, num_iter=150, burn_in=50, seed=1)
    assert res.scores.shape[0] == len(g_te)
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"BGEN PRS R^2 vs true genetic value too low: {r2:.3f}"


def test_end_to_end_inf_runs(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=2)
    res = run_ldpred2_prs(ss_path, prefix, method="inf", block_size=200)
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.10


def test_subset_to_sumstats_matches_full_read(tmp_path):
    # Reading only the GWAS variants must give the same PRS as a full read.
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=7)
    full = run_ldpred2_prs(ss_path, prefix, method="inf", block_size=150,
                           subset_to_sumstats=False)
    sub = run_ldpred2_prs(ss_path, prefix, method="inf", block_size=150,
                          subset_to_sumstats=True)
    np.testing.assert_allclose(full.scores, sub.scores, rtol=1e-6, atol=1e-6)


def test_allele_flip_is_corrected(tmp_path):
    """Swapping A1/A2 in the sumstats must not change the PRS (sign realigned)."""
    prefix, ss_path, g_te = _simulate(tmp_path, m=300, seed=3)
    res0 = run_ldpred2_prs(ss_path, prefix, method="inf", block_size=150)

    flipped = str(tmp_path / "gwas_flip.txt")
    with open(ss_path) as fin, open(flipped, "w") as fout:
        fout.write(fin.readline())               # header
        for line in fin:
            snp, a1, a2, beta, se, n = line.split()
            fout.write(f"{snp}\t{a2}\t{a1}\t{-float(beta):.6g}\t{se}\t{n}\n")
    res1 = run_ldpred2_prs(flipped, prefix, method="inf", block_size=150)

    np.testing.assert_allclose(res0.scores, res1.scores, rtol=1e-6, atol=1e-6)
