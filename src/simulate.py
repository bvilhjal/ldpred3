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

To scale to 10k-100k SNPs without exhausting memory, genotypes are stored as
``int8`` dosages (~8x smaller than float64) and every step (standardization,
GWAS, LD, PRS) is processed one LD block at a time, so a full float genotype
matrix is never materialised.

Run::

    python src/simulate.py                       # default accuracy grid
    python src/simulate.py --quick               # small/fast grid
    python src/simulate.py --csv out.csv         # grid + save results
    python src/simulate.py --scaling             # runtime/memory vs #SNPs
    python src/simulate.py --scaling --m 10000 50000 100000
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ldpred2 import HAVE_NUMBA, ldpred2_by_blocks, standardize_betas  # noqa: E402


# --------------------------------------------------------------------------- #
# Genotype simulation
# --------------------------------------------------------------------------- #
def _ar1_chol(k, rho):
    """Cholesky factor of the AR(1) correlation matrix of size ``k``."""
    idx = np.arange(k)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(corr + 1e-8 * np.eye(k))


def _norm_isf(q):
    """Inverse survival function of the standard normal (vectorised).

    Returns ``z`` such that ``P(Z > z) = q``. Uses Acklam's rational
    approximation so the module stays NumPy-only.
    """
    q = np.asarray(q, dtype=float)
    p = 1.0 - q
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


def simulate_genotypes(n, block_sizes, maf, rho, rng):
    """Simulate diploid genotype dosages (0/1/2) with within-block LD.

    LD is induced through a latent multivariate-normal haplotype model: two
    latent Gaussian haplotypes with AR(1) correlation ``rho`` per block are
    thresholded at the MAF-implied quantile and summed. Genotypes are returned
    as ``int8`` to keep large matrices small in memory.

    Returns ``(G, blocks)`` with ``G`` of shape ``(n, m)`` and ``blocks`` a list
    of column-index arrays.
    """
    m = int(np.sum(block_sizes))
    G = np.empty((n, m), dtype=np.int8)
    blocks = []
    col = 0
    for k in block_sizes:
        chol = _ar1_chol(k, rho)
        thr = _norm_isf(maf[col:col + k])
        hap_sum = np.zeros((n, k))
        for _ in range(2):  # two haplotypes -> dosage 0/1/2
            z = rng.standard_normal((n, k)) @ chol.T
            hap_sum += (z > thr)
        G[:, col:col + k] = hap_sum.astype(np.int8)
        blocks.append(np.arange(col, col + k))
        col += k
    return G, blocks


def _std_block(G, rows, idx, mean, sd):
    """Standardize a genotype sub-block to float64 (column z-scores)."""
    X = G[rows][:, idx].astype(np.float64)
    X -= mean[idx]
    X /= sd[idx]
    return X


def r2(pred, target):
    """Squared Pearson correlation (prediction R^2)."""
    if np.std(pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1] ** 2)


# --------------------------------------------------------------------------- #
# One replicate (block-streaming, memory efficient)
# --------------------------------------------------------------------------- #
def run_one(n_train, h2, p, *, m=1000, block_size=100, n_test=2000,
            rho=0.6, maf_range=(0.05, 0.5), seed=0,
            burn_in=60, num_iter=150,
            methods=("marginal", "inf", "grid", "auto")):
    """Run a single simulation replicate; return prediction R^2 per method."""
    rng = np.random.default_rng(seed)
    block_sizes = [block_size] * (m // block_size)
    m = int(sum(block_sizes))
    maf = rng.uniform(*maf_range, size=m)

    n_total = n_train + n_test
    G, blocks = simulate_genotypes(n_total, block_sizes, maf, rho, rng)
    train_rows = slice(0, n_train)
    test_rows = slice(n_train, n_total)

    # True sparse effects on the standardized-genotype scale.
    is_causal = rng.random(m) < p
    m_causal = max(int(is_causal.sum()), 1)
    true_beta = np.zeros(m)
    true_beta[is_causal] = rng.normal(0.0, np.sqrt(h2 / m_causal),
                                      size=int(is_causal.sum()))

    # Pass 1: training mean/sd per SNP + genetic values (train & test).
    mean = np.empty(m)
    sd = np.empty(m)
    g_train = np.zeros(n_train)
    g_test = np.zeros(n_test)
    for idx in blocks:
        Xtr = G[train_rows][:, idx].astype(np.float64)
        mu = Xtr.mean(axis=0)
        s = Xtr.std(axis=0)
        s[s < 1e-8] = 1.0
        mean[idx] = mu
        sd[idx] = s
        Xtr -= mu
        Xtr /= s
        bb = true_beta[idx]
        g_train += Xtr @ bb
        g_test += _std_block(G, test_rows, idx, mean, sd) @ bb

    # Fix the realised genetic variance to h2, then add environmental noise.
    sg = np.sqrt(h2) / (g_train.std() + 1e-12)
    true_beta *= sg
    g_train *= sg
    g_test *= sg
    y_train = g_train + rng.normal(0.0, np.sqrt(1.0 - h2), n_train)
    y_test = g_test + rng.normal(0.0, np.sqrt(1.0 - h2), n_test)
    ys = (y_train - y_train.mean()) / (y_train.std() + 1e-12)

    # Pass 2: marginal GWAS + per-block LD matrices.
    beta_std = np.empty(m)
    ld = []
    for idx in blocks:
        Xtr = _std_block(G, train_rows, idx, mean, sd)
        r = (Xtr * ys[:, None]).mean(axis=0)          # slope == correlation
        r = np.clip(r, -0.9999, 0.9999)
        se = np.sqrt((1 - r ** 2) / (n_train - 2))
        bs, _ = standardize_betas(r, se, n_train)
        beta_std[idx] = bs
        ld.append(((Xtr.T @ Xtr) / n_train, idx))

    n_vec = np.full(m, float(n_train))

    # Fit each method (full adjusted-beta vector).
    adj = {"marginal": beta_std}
    if "inf" in methods:
        adj["inf"] = ldpred2_by_blocks(ld, beta_std, n_vec, method="inf", h2=h2)
    if "grid" in methods:
        adj["grid"] = ldpred2_by_blocks(ld, beta_std, n_vec, method="grid",
                                        h2=h2, p=p, burn_in=burn_in,
                                        num_iter=num_iter, seed=seed)
    if "auto" in methods:
        adj["auto"] = ldpred2_by_blocks(ld, beta_std, n_vec, method="auto",
                                        burn_in=burn_in, num_iter=num_iter,
                                        seed=seed)

    # Pass 3: build PRS on the test sample, block by block.
    prs = {k: np.zeros(n_test) for k in adj}
    for idx in blocks:
        Xte = _std_block(G, test_rows, idx, mean, sd)
        for k, beta in adj.items():
            prs[k] += Xte @ beta[idx]

    results = {k: r2(prs[k], y_test) for k in adj}
    results["ceiling(h2)"] = r2(g_test, y_test)
    return results


def _peak_mem_gb():
    """Peak resident set size of this process, in GB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_grid(args):
    if args.quick:
        polygenicities = [0.01, 0.5]
        heritabilities = [0.5]
        sample_sizes = [3000]
        m, block_size = 500, 100
    else:
        polygenicities = [0.005, 0.05, 0.5]
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
        _write_csv(args.csv, rows)


def run_scaling(args):
    sizes = args.m or [10000, 50000, 100000]
    h2, p = args.h2, args.p
    methods = ["marginal", "inf", "grid", "auto", "ceiling(h2)"]

    print(f"LDpred2 scaling benchmark  (numba={'on' if HAVE_NUMBA else 'off'}, "
          f"N_train={args.n_train}, N_test={args.n_test}, blocks of "
          f"{args.block_size}, h2={h2}, p={p})")
    head = (f"{'#SNPs':>8} {'time(s)':>8} {'mem(GB)':>8} | "
            + " ".join(f"{x:>11}" for x in methods))
    print(head)
    print("-" * len(head))

    rows = []
    for m in sizes:
        t0 = time.time()
        res = run_one(args.n_train, h2, p, m=m, block_size=args.block_size,
                      n_test=args.n_test, seed=args.seed,
                      burn_in=args.burn_in, num_iter=args.num_iter)
        dt = time.time() - t0
        mem = _peak_mem_gb()
        rows.append({"m": m, "time_s": round(dt, 2), "peak_mem_gb": round(mem, 2),
                     **res})
        cells = " ".join(f"{res[k]:>11.3f}" for k in methods)
        print(f"{m:>8} {dt:>8.1f} {mem:>8.2f} | {cells}")

    if args.csv:
        _write_csv(args.csv, rows)


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="smaller/faster accuracy grid")
    parser.add_argument("--scaling", action="store_true",
                        help="benchmark runtime/memory/accuracy vs number of SNPs")
    parser.add_argument("--m", type=int, nargs="+", default=None,
                        help="number(s) of SNPs for --scaling mode")
    parser.add_argument("--n-train", type=int, default=8000, dest="n_train")
    parser.add_argument("--n-test", type=int, default=2000, dest="n_test")
    parser.add_argument("--block-size", type=int, default=200, dest="block_size")
    parser.add_argument("--h2", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--burn-in", type=int, default=50, dest="burn_in")
    parser.add_argument("--num-iter", type=int, default=100, dest="num_iter")
    parser.add_argument("--csv", default=None, help="write results to this CSV file")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    if args.scaling:
        run_scaling(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
