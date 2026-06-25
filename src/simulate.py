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


def simulate_genotypes_coalescent(n, m, block_size, *, Ne=10000,
                                  recomb_rate=1e-8, mut_rate=1e-8, min_maf=0.01,
                                  seed=None):
    """Simulate genotypes with *realistic* LD via a coalescent-with-recombination.

    Unlike the AR(1) latent model (smooth geometric LD decay), a coalescent
    simulation (msprime) produces realistic LD: haplotype blocks separated by
    recombination, high-LD plateaus, a heavy decay tail and sporadic long-range
    correlation -- the structure seen in real reference panels.

    Human-like defaults: Ne=10,000, recombination & mutation rate 1e-8 per bp
    per generation. The sequence length is grown until at least ``m`` common
    SNPs (MAF > ``min_maf``) are obtained; the first ``m`` are kept and cut into
    contiguous LD blocks of ``block_size``.

    Returns ``(G, blocks)`` with ``G`` int8 of shape ``(n, m')`` and ``blocks``
    a list of contiguous index arrays (``m'`` is ``m`` rounded down to a multiple
    of ``block_size``).
    """
    try:
        import msprime
    except ImportError as e:  # pragma: no cover
        raise ImportError("the coalescent LD model needs msprime "
                          "(pip install msprime)") from e

    rng = np.random.default_rng(seed)
    seq_len = max(1e6, m / 1200 * 1e6)   # ~1200 common SNPs per Mb to start
    G = None
    for _ in range(7):
        ms_seed = int(rng.integers(1, 2 ** 31 - 1))
        ts = msprime.sim_ancestry(
            samples=n, ploidy=2, population_size=Ne,
            recombination_rate=recomb_rate, sequence_length=int(seq_len),
            random_seed=ms_seed)
        mts = msprime.sim_mutations(ts, rate=mut_rate, random_seed=ms_seed,
                                    model=msprime.BinaryMutationModel())
        H = mts.genotype_matrix()                       # (sites, 2n), 0/1
        dos = (H[:, 0::2] + H[:, 1::2]).T               # (n, sites), 0/1/2
        af = dos.mean(axis=0) / 2.0
        dos = dos[:, (af > min_maf) & (af < 1 - min_maf)]
        if dos.shape[1] >= m:
            G = dos
            break
        seq_len *= 1.8
    if G is None or G.shape[1] < m:
        raise RuntimeError("coalescent simulation produced too few common SNPs; "
                           "increase sequence length / Ne")

    n_blocks = m // block_size
    m2 = n_blocks * block_size
    G = np.ascontiguousarray(G[:, :m2].astype(np.int8))
    blocks = [np.arange(i * block_size, (i + 1) * block_size)
              for i in range(n_blocks)]
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
            burn_in=60, num_iter=150, ld_model="ar1",
            methods=("marginal", "inf", "grid", "auto"),
            return_timing=False):
    """Run a single simulation replicate; return prediction R^2 per method.

    ``ld_model`` selects the LD structure: ``"ar1"`` (fast, idealized geometric
    decay) or ``"coalescent"`` (realistic LD via msprime: haplotype blocks,
    recombination hotspots, heavy decay tail, long-range LD).

    If ``return_timing`` is True, return ``(results, timing)`` where ``timing``
    splits ``prep_s`` (genotype sim + GWAS + LD construction, which scale with
    N) from ``fit_s`` (the LDpred2 algorithm per method, which does not).
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    if ld_model == "coalescent":
        G, blocks = simulate_genotypes_coalescent(n_total, m, block_size,
                                                  seed=seed)
        m = int(sum(len(b) for b in blocks))
    elif ld_model == "ar1":
        block_sizes = [block_size] * (m // block_size)
        m = int(sum(block_sizes))
        maf = rng.uniform(*maf_range, size=m)
        G, blocks = simulate_genotypes(n_total, block_sizes, maf, rho, rng)
    else:
        raise ValueError("ld_model must be 'ar1' or 'coalescent'")
    train_rows = slice(0, n_train)
    test_rows = slice(n_train, n_total)

    # True sparse effects on the standardized-genotype scale. Guarantee at least
    # one causal variant so very low polygenicity (p) still yields signal (which
    # also requires m >= ~1/p for the realised fraction to match p).
    is_causal = rng.random(m) < p
    if not is_causal.any():
        is_causal[rng.integers(m)] = True
    m_causal = int(is_causal.sum())
    true_beta = np.zeros(m)
    true_beta[is_causal] = rng.normal(0.0, np.sqrt(h2 / m_causal), size=m_causal)

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
    t_prep = time.time() - t_start   # sim + GWAS + LD build (scales with N)

    # Fit each method (full adjusted-beta vector). Time the LDpred2 algorithm
    # itself (it operates on sumstats + LD only, independent of N) separately
    # from the data-prep / GWAS / LD-construction above (which do depend on N).
    fit_time = {}
    adj = {"marginal": beta_std}
    if "inf" in methods:
        t = time.time()
        adj["inf"] = ldpred2_by_blocks(ld, beta_std, n_vec, method="inf", h2=h2)
        fit_time["inf"] = time.time() - t
    if "grid" in methods:
        t = time.time()
        adj["grid"] = ldpred2_by_blocks(ld, beta_std, n_vec, method="grid",
                                        h2=h2, p=p, burn_in=burn_in,
                                        num_iter=num_iter, seed=seed)
        fit_time["grid"] = time.time() - t
    if "auto" in methods:
        t = time.time()
        adj["auto"] = ldpred2_by_blocks(ld, beta_std, n_vec, method="auto",
                                        burn_in=burn_in, num_iter=num_iter,
                                        seed=seed)
        fit_time["auto"] = time.time() - t

    # Pass 3: build PRS on the test sample, block by block.
    prs = {k: np.zeros(n_test) for k in adj}
    for idx in blocks:
        Xte = _std_block(G, test_rows, idx, mean, sd)
        for k, beta in adj.items():
            prs[k] += Xte @ beta[idx]

    results = {k: r2(prs[k], y_test) for k in adj}
    results["ceiling(h2)"] = r2(g_test, y_test)

    timing = {"prep_s": t_prep, "fit_s": fit_time}
    return (results, timing) if return_timing else results


def _peak_mem_gb():
    """Peak resident set size of this process, in GB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def _warmup():
    """Trigger Numba JIT compilation so it doesn't pollute the first timing row."""
    if HAVE_NUMBA:
        run_one(500, 0.5, 0.05, m=200, block_size=100, n_test=100,
                burn_in=5, num_iter=5, seed=0)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_grid(args):
    # Realistic complex-trait architectures: heritability 0.3-0.5 and
    # polygenicity (fraction of causal variants) 1e-4 to 0.1. m is chosen large
    # enough (>= ~1/p) that the lowest polygenicity has causal variants.
    if args.quick:
        polygenicities = [0.001, 0.1]
        heritabilities = [0.5]
        sample_sizes = [5000]
        m, block_size = 5000, 200
    else:
        polygenicities = [0.0001, 0.001, 0.01, 0.1]
        heritabilities = [0.3, 0.5]
        sample_sizes = [5000, 20000]
        m, block_size = 10000, 200

    methods = ["marginal", "inf", "grid", "auto", "ceiling(h2)"]
    header = f"{'N':>7} {'h2':>5} {'p':>7} | " + " ".join(f"{x:>11}" for x in methods)
    print(f"Genotype-level LDpred2 benchmark  (m={m} SNPs, blocks of "
          f"{block_size}, LD={args.ld_model})")
    print(header)
    print("-" * len(header))

    rows = []
    for h2 in heritabilities:
        for p in polygenicities:
            for n_train in sample_sizes:
                res = run_one(n_train, h2, p, m=m, block_size=block_size,
                              ld_model=args.ld_model, seed=args.seed)
                rows.append({"N": n_train, "h2": h2, "p": p, **res})
                cells = " ".join(f"{res[k]:>11.3f}" for k in methods)
                print(f"{n_train:>7} {h2:>5} {p:>7} | {cells}")

    if args.csv:
        _write_csv(args.csv, rows)


def run_scaling(args):
    """Scale the number of SNPs; split prep time (scales with N) from fit time."""
    sizes = args.m or [10000, 50000, 100000]
    h2, p = args.h2, args.p
    methods = ["marginal", "inf", "grid", "auto", "ceiling(h2)"]

    print(f"LDpred2 #SNP-scaling benchmark  (numba={'on' if HAVE_NUMBA else 'off'}, "
          f"N_train={args.n_train}, blocks of {args.block_size}, h2={h2}, p={p})")
    head = (f"{'#SNPs':>8} {'prep(s)':>8} {'fit(s)':>7} {'mem(GB)':>8} | "
            + " ".join(f"{x:>11}" for x in methods))
    print(head)
    print("-" * len(head))

    rows = []
    for m in sizes:
        res, tm = run_one(args.n_train, h2, p, m=m, block_size=args.block_size,
                          n_test=args.n_test, seed=args.seed,
                          burn_in=args.burn_in, num_iter=args.num_iter,
                          ld_model=args.ld_model, return_timing=True)
        fit = sum(tm["fit_s"].values())
        mem = _peak_mem_gb()
        rows.append({"m": m, "prep_s": round(tm["prep_s"], 2),
                     "fit_s": round(fit, 2), "peak_mem_gb": round(mem, 2), **res})
        cells = " ".join(f"{res[k]:>11.3f}" for k in methods)
        print(f"{m:>8} {tm['prep_s']:>8.1f} {fit:>7.2f} {mem:>8.2f} | {cells}")

    if args.csv:
        _write_csv(args.csv, rows)


def run_ld_scaling(args):
    """Scale the LD block size at fixed total #SNPs.

    This isolates the axis the LDpred2 algorithm actually depends on. Larger LD
    blocks make each block's sampler/solve more expensive: the Gibbs samplers
    (grid/auto) grow ~linearly in block size for fixed m (the rank-1 updates
    cost grows with block size), while the infinitesimal solve is a dense linear
    system per block and grows ~quadratically in block size for fixed m.
    """
    block_sizes = args.block_sizes or [100, 250, 500, 1000, 2000]
    m = (args.m or [20000])[0]
    h2, p = args.h2, args.p
    _warmup()

    print(f"LDpred2 LD-block-size scaling  (numba={'on' if HAVE_NUMBA else 'off'}, "
          f"m={m} SNPs fixed, N_train={args.n_train}, h2={h2}, p={p})")
    head = (f"{'block':>6} {'#blocks':>8} {'prep(s)':>8} | "
            f"{'fit_inf':>9} {'fit_grid':>9} {'fit_auto':>9} | "
            f"{'R2_grid':>8} {'R2_auto':>8}")
    print(head)
    print("-" * len(head))

    rows = []
    for bs in block_sizes:
        res, tm = run_one(args.n_train, h2, p, m=m, block_size=bs,
                          n_test=args.n_test, seed=args.seed,
                          burn_in=args.burn_in, num_iter=args.num_iter,
                          ld_model=args.ld_model, return_timing=True)
        ft = tm["fit_s"]
        nblk = m // bs
        rows.append({"block_size": bs, "n_blocks": nblk,
                     "prep_s": round(tm["prep_s"], 2),
                     "fit_inf_s": round(ft["inf"], 3),
                     "fit_grid_s": round(ft["grid"], 3),
                     "fit_auto_s": round(ft["auto"], 3),
                     "R2_grid": res["grid"], "R2_auto": res["auto"]})
        print(f"{bs:>6} {nblk:>8} {tm['prep_s']:>8.1f} | "
              f"{ft['inf']:>9.3f} {ft['grid']:>9.3f} {ft['auto']:>9.3f} | "
              f"{res['grid']:>8.3f} {res['auto']:>8.3f}")

    if args.csv:
        _write_csv(args.csv, rows)


def run_n_independence(args):
    """Vary the GWAS sample size at fixed LD; the fit time should be flat.

    Demonstrates that the LDpred2 algorithm cost is independent of N: only the
    data-prep / GWAS / LD-construction time grows with N, while the sampler time
    (which sees only sumstats + LD) stays roughly constant.
    """
    n_list = args.n_list or [2000, 8000, 32000]
    m = (args.m or [10000])[0]
    h2, p = args.h2, args.p
    _warmup()

    print(f"LDpred2 N-independence check  (numba={'on' if HAVE_NUMBA else 'off'}, "
          f"m={m} SNPs, blocks of {args.block_size}, h2={h2}, p={p})")
    head = (f"{'N_train':>8} {'prep(s)':>8} | {'fit_grid(s)':>11} "
            f"{'fit_auto(s)':>11} | {'R2_grid':>8}")
    print(head)
    print("-" * len(head))

    rows = []
    for n in n_list:
        res, tm = run_one(n, h2, p, m=m, block_size=args.block_size,
                          n_test=args.n_test, seed=args.seed,
                          burn_in=args.burn_in, num_iter=args.num_iter,
                          ld_model=args.ld_model, return_timing=True)
        ft = tm["fit_s"]
        rows.append({"N_train": n, "prep_s": round(tm["prep_s"], 2),
                     "fit_grid_s": round(ft["grid"], 3),
                     "fit_auto_s": round(ft["auto"], 3), "R2_grid": res["grid"]})
        print(f"{n:>8} {tm['prep_s']:>8.1f} | {ft['grid']:>11.3f} "
              f"{ft['auto']:>11.3f} | {res['grid']:>8.3f}")

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
    parser.add_argument("--ld-scaling", action="store_true", dest="ld_scaling",
                        help="benchmark algorithm time vs LD block size (fixed #SNPs)")
    parser.add_argument("--n-independence", action="store_true",
                        dest="n_independence",
                        help="show the algorithm's fit time is independent of N")
    parser.add_argument("--m", type=int, nargs="+", default=None,
                        help="number(s) of SNPs (first value used as fixed m for "
                             "--ld-scaling / --n-independence)")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=None,
                        dest="block_sizes",
                        help="LD block sizes to sweep for --ld-scaling")
    parser.add_argument("--n-list", type=int, nargs="+", default=None,
                        dest="n_list",
                        help="GWAS sample sizes to sweep for --n-independence")
    parser.add_argument("--ld-model", choices=["ar1", "coalescent"], default="ar1",
                        dest="ld_model",
                        help="LD structure: ar1 (idealized) or coalescent "
                             "(realistic, needs msprime)")
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

    if args.ld_scaling:
        run_ld_scaling(args)
    elif args.n_independence:
        run_n_independence(args)
    elif args.scaling:
        run_scaling(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
