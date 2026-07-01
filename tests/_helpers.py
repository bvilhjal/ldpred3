"""Shared test fixtures/helpers (imported by several test modules).

Kept in its own module rather than imported from one test file into another, so
the dependency between test modules is explicit.
"""

import numpy as np

from ldpred3.genotype_io import VariantTable, SampleTable, write_plink
from ldpred3.bgen_io import write_bgen


def simulate_prs_dataset(tmp_path, n_train=4000, n_target=1500, m=600,
                         p_causal=0.1, h2=0.5, seed=0):
    """Make a train cohort (for GWAS) and a target cohort sharing variants.

    Writes PLINK + BGEN target genotypes and a GWAS sumstats file under
    ``tmp_path``; returns ``(plink_prefix, sumstats_path, true_target_gv)``.
    """
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
