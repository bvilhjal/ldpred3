"""
Genotype-level simulation harness to benchmark the LDpred2 implementation.

The summary-statistic test (``tests/test_ldpred2.py``) feeds LDpred2 effects
drawn directly from its own model. This script is a tougher, more realistic
end-to-end check:

1. simulate genotypes with block LD structure (separate train / test samples),
2. simulate a phenotype under a chosen heritability ``h2`` and polygenicity
   ``p`` (fraction of causal variants),
3. run a marginal GWAS on the training sample,
4. estimate the LD matrix per block from the training genotypes,
5. fit LDpred2-inf / -grid / -auto,
6. build polygenic scores on the held-out test sample and report prediction
   accuracy (R^2 with the phenotype).

It sweeps a grid of polygenicity, heritability and GWAS sample size and prints
a results table (optionally saved to CSV).

Run::

    python src/simulate.py                 # default grid
    python src/simulate.py --quick         # small/fast grid
    python src/simulate.py --csv out.csv   # also save results
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ldpred2 import ldpred2_by_blocks, standardize_betas  # noqa: E402


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _ar1_chol(k, rho):
    """Cholesky factor of the AR(1) correlation matrix of size ``k``."""
    idx = np.arange(k)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(corr + 1e-8 * np.eye(k))


def simulate_genotypes(n, block_sizes, maf, rho, rng):
    """Simulate diploid genotype dosages (0/1/2) with within-block LD.

    LD is induced through a latent multivariate-normal haplotype model: for each
    block two latent Gaussian haplotypes with AR(1) correlation ``rho`` are
    drawn and thresholded at the MAF-implied quantile, then summed. The
    resulting genotype LD has realistic block structure (attenuated relative to
    the latent correlation, as with real tetrachoric thresholding).

    Returns
    -------
    G : ndarray, shape (n, m)
        Integer genotype dosages.
    blocks : list of ndarray
        Index arrays giving the columns belonging to each LD block.
    """
    m = int(np.sum(block_sizes))
    G = np.empty((n, m), dtype=np.float64)
    blocks = []
    col = 0
    for k in block_sizes:
        chol = _ar1_chol(k, rho)
        thresholds = -np.sqrt(2.0)  # placeholder, set per-SNP below
        block_maf = maf[col:col + k]
        # Per-SNP threshold so that P(latent > t) = maf.
        from math import sqrt  # local import to keep numpy-only top level
        thr = _norm_isf(block_maf)
        hap_sum = np.zeros((n, k))
        for _ in range(2):  # two haplotypes -> dosage 0/1/2
            z = rng.standard_normal((n, k)) @ chol.T
            hap_sum += (z > thr).astype(np.float64)
        G[:, col:col + k] = hap_sum
        blocks.append(np.arange(col, col + k))
        col += k
    return G, blocks


def _norm_isf(q):
    """Inverse survival function of the standard normal (vectorised).

    Uses the Acklam rational approximation so the module stays NumPy-only.
    Returns z such that P(Z > z) = q.
    """
    q = np.asarray(q, dtype=float)
    p = 1.0 - q  # want the p-th quantile
    # Acklam's algorithm.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    z = np.empty_like(p)

    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    if np.any(lo):
        ql = np.sqrt(-2 * np.log(p[lo]))
        z[lo] = (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
                ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
    if np.any(hi):
        qh = np.sqrt(-2 * np.log(1 - p[hi]))
        z[hi] = -(((((c[0] * qh + c[1]) * qh + c[2]) * qh + c[3]) * qh + c[4]) * qh + c[5]) / \
                ((((d[0] * qh + d[1]) * qh + d[2]) * qh + d[3]) * qh + 1)
    if np.any(mid):
        qm = p[mid] - 0.5
        r = qm * qm
        z[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * qm / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return z


def simulate_phenotype(G_std, h2, p, rng):
    """Simulate a phenotype from standardized genotypes.

    Causal variants are a random fraction ``p`` of all SNPs with effects
    ``N(0, h2 / m_causal)`` on the standardized-genotype scale. Environmental
    noise is added so the phenotype has ~unit variance and heritability ``h2``.

    Returns ``(y, true_beta, genetic_value)``.
    """
    n, m = G_std.shape
    is_causal = rng.random(m) < p
    m_causal = max(int(is_causal.sum()), 1)
    true_beta = np.zeros(m)
    true_beta[is_causal] = rng.normal(0.0, np.sqrt(h2 / m_causal), size=int(is_causal.sum()))

    g = G_std @ true_beta
    g *= np.sqrt(h2) / (g.std() + 1e-12)  # fix realised genetic variance to h2
    e = rng.normal(0.0, np.sqrt(1.0 - h2), size=n)
    y = g + e
    return y, true_beta, g


# --------------------------------------------------------------------------- #
# GWAS + LD
# --------------------------------------------------------------------------- #
def standardize_columns(G, mean=None, sd=None):
    """Standardize genotype columns; reuse train mean/sd on the test set."""
    if mean is None:
        mean = G.mean(axis=0)
    if sd is None:
        sd = G.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (G - mean) / sd, mean, sd


def run_gwas(G_std, y):
    """Marginal (single-SNP) GWAS on standardized genotypes & phenotype.

    Returns ``(beta_hat, beta_se, n)`` for the per-SNP simple regressions.
    """
    n = G_std.shape[0]
    ys = (y - y.mean()) / (y.std() + 1e-12)
    # With standardized x and y the slope equals the correlation.
    r = (G_std * ys[:, None]).mean(axis=0)
    r = np.clip(r, -0.9999, 0.9999)
    se = np.sqrt((1 - r ** 2) / (n - 2))
    return r, se, n


def ld_blocks_from_geno(G_std, blocks):
    """Per-block in-sample LD correlation matrices."""
    n = G_std.shape[0]
    out = []
    for idx in blocks:
        Xb = G_std[:, idx]
        corr = (Xb.T @ Xb) / n
        out.append((corr, idx))
    return out


def r2(pred, target):
    """Squared Pearson correlation (prediction R^2)."""
    if np.std(pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1] ** 2)


# --------------------------------------------------------------------------- #
# One replicate
# --------------------------------------------------------------------------- #
def run_one(n_train, h2, p, *, m=1000, block_size=100, n_test=2000,
            rho=0.6, maf_range=(0.05, 0.5), seed=0,
            burn_in=60, num_iter=150):
    """Run a single simulation replicate and return prediction R^2 per method."""
    rng = np.random.default_rng(seed)
    block_sizes = [block_size] * (m // block_size)
    m = int(sum(block_sizes))
    maf = rng.uniform(*maf_range, size=m)

    n_total = n_train + n_test
    G, blocks = simulate_genotypes(n_total, block_sizes, maf, rho, rng)
    G_train, G_test = G[:n_train], G[n_train:]

    G_train_std, mean, sd = standardize_columns(G_train)
    G_test_std, _, _ = standardize_columns(G_test, mean, sd)

    # Phenotype is generated on the (training-standardized) genotypes.
    y_train, true_beta, _ = simulate_phenotype(G_train_std, h2, p, rng)
    g_test = G_test_std @ true_beta
    e_test = rng.normal(0.0, np.sqrt(1 - h2), size=n_test)
    y_test = g_test + e_test

    # GWAS + LD on training data.
    beta_hat, beta_se, n = run_gwas(G_train_std, y_train)
    beta_std, scale = standardize_betas(beta_hat, beta_se, n)
    ld = ld_blocks_from_geno(G_train_std, blocks)

    results = {}

    def score(beta_adj):
        return r2(G_test_std @ beta_adj, y_test)

    # Baseline: raw marginal effects as a PRS (LD double-counting).
    results["marginal"] = score(beta_std)

    results["inf"] = score(
        ldpred2_by_blocks(ld, beta_std, n, method="inf", h2=h2)
    )
    results["grid"] = score(
        ldpred2_by_blocks(ld, beta_std, n, method="grid", h2=h2, p=p,
                          burn_in=burn_in, num_iter=num_iter, seed=seed)
    )
    auto_beta = ldpred2_by_blocks(ld, beta_std, n, method="auto",
                                  burn_in=burn_in, num_iter=num_iter, seed=seed)
    results["auto"] = score(auto_beta)

    # Theoretical ceiling: R^2 of the true genetic value with the phenotype.
    results["ceiling(h2)"] = r2(g_test, y_test)
    return results


# --------------------------------------------------------------------------- #
# Grid sweep
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="smaller/faster grid for a sanity check")
    parser.add_argument("--csv", default=None, help="write results to this CSV file")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    if args.quick:
        polygenicities = [0.01, 0.5]
        heritabilities = [0.5]
        sample_sizes = [3000]
        m, block_size = 500, 100
    else:
        polygenicities = [0.005, 0.05, 0.5]   # sparse -> highly polygenic
        heritabilities = [0.2, 0.5]
        sample_sizes = [2000, 5000, 10000]
        m, block_size = 1000, 100

    methods = ["marginal", "inf", "grid", "auto", "ceiling(h2)"]
    header = f"{'N':>7} {'h2':>5} {'p':>7} | " + " ".join(f"{x:>11}" for x in methods)
    print(f"Genotype-level LDpred2 benchmark  (m={m} SNPs, blocks of {block_size})")
    print(header)
    print("-" * len(header))

    rows = []
    for h2 in heritabilities:
        for p in polygenicities:
            for n_train in sample_sizes:
                res = run_one(n_train, h2, p, m=m, block_size=block_size,
                              seed=args.seed)
                rows.append({"N": n_train, "h2": h2, "p": p, **res})
                cells = " ".join(f"{res[k]:>11.3f}" for k in methods)
                print(f"{n_train:>7} {h2:>5} {p:>7} | {cells}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved results to {args.csv}")


if __name__ == "__main__":
    main()
