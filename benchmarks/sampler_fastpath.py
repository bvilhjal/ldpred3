"""Constant-N fast path in the Gibbs samplers: per-sweep vs per-SNP scalars.

The point-normal update needs, per SNP, a posterior variance / SD and a
half-log-normaliser (a ``sqrt`` and a ``log1p``) plus the log prior-odds. When
the GWAS sample size ``N`` is **shared across variants** (a scalar ``n_eff``) and
the slab / prior are uniform, those quantities are identical for every SNP, so
the samplers hoist them out of the per-SNP loop and compute them **once per
sweep** (``_pn_const_scalars``) — the same optimisation the batched / streaming
kernels already used, back-ported to the dense (``_gibbs_kernel``), sampling
(``_gibbs_kernel_sample``) and sparse (``_gibbs_kernel_sparse``) kernels. A
per-variant ``N`` (or a MAF slab / non-uniform ``prior_weights``) falls back to
the exact per-SNP path, bit-for-bit unchanged.

This isolates the saving with an in-tree A/B on one realistic-LD block: the
**identical** fit run twice — once with a constant ``N`` (fast path) and once
with ``N`` perturbed by 1e-6 so ``n.min() != n.max()`` selects the per-SNP path.
The perturbation is far below any effect on the fit (the genetic R² columns are
equal to three decimals), so the two runs do the *same* rank-1 residual work and
the only difference timed is the per-SNP ``sqrt``/``log1p``/prior-odds the fast
path hoists to once a sweep. The saving is largest for sparse ``p`` (few effects
change, so the rank-1 update is mostly skipped and the per-SNP scalar work
dominates the sweep) — the regime fine-mapping and the auto r²/inference chains
run in.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/sampler_fastpath.py

Self-contained (realistic coalescent LD via msprime); writes
``benchmarks/sampler_fastpath.csv``.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

M = 2000                    # one realistic-LD block
N_GWAS = 20000.0
H2 = 0.5
P_GRID = (1e-3, 1e-2, 1e-1)
BURN, ITER = 100, 300
REPS = 3


def build_problem(seed=0):
    """One dense realistic-LD block + standardized marginal effects."""
    from ldpred3.simulate import simulate_genotypes_coalescent
    from ldpred3.ld import compute_ld_blocks
    G, _ = simulate_genotypes_coalescent(5000, M, M, seed=seed)
    m = G.shape[1]
    R = compute_ld_blocks(G, block_size=m)[0][0]        # float32 (m, m)
    return R, m


def make_sumstats(R, m, p, seed):
    """True effects at causal fraction ``p`` and a GWAS-noise beta_hat."""
    rng = np.random.default_rng(seed)
    Rf = np.asarray(R, dtype=np.float64)
    beta = np.zeros(m)
    causal = rng.random(m) < p
    nc = int(causal.sum()) or 1
    causal = np.flatnonzero(causal)[:nc]
    beta[causal] = rng.standard_normal(causal.size)
    beta *= np.sqrt(H2 / max(float(beta @ (Rf @ beta)), 1e-12))   # scale to h2
    ch = np.linalg.cholesky(Rf + 1e-6 * np.eye(m))
    beta_hat = Rf @ beta + (ch @ rng.standard_normal(m)) / np.sqrt(N_GWAS)
    return beta, beta_hat, Rf


def genetic_r2(fit, true, Rf):
    """(β̂ᵀRβ)² / [(β̂ᵀRβ̂)(βᵀRβ)] — squared PRS/true-genetic-value correlation."""
    Rt = Rf @ true
    num = float(fit @ Rt) ** 2
    den = float(fit @ (Rf @ fit)) * float(true @ Rt)
    return num / den if den > 0 else 0.0


def best_time(fn, reps=REPS):
    fn()                                            # warm up (JIT compile)
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    try:
        import msprime  # noqa: F401
    except ImportError:
        print("sampler_fastpath needs msprime (pip install msprime); skipping.")
        return

    from ldpred3 import ldpred3_grid
    from ldpred3.finemap import ldpred3_pip

    R, m = build_problem(seed=0)
    n_const = np.full(m, N_GWAS)
    # N perturbed by 1e-6 (negligible for the fit) purely to make n.min() != n.max(),
    # which routes the sampler through the exact per-SNP path for the A/B.
    n_var = N_GWAS * (1.0 + 1e-6 * np.linspace(-1.0, 1.0, m))

    def grid(n, p):
        return ldpred3_grid(R, bh, n, h2=H2, p=p, burn_in=BURN, num_iter=ITER, seed=1)

    def pip(n, p):
        return ldpred3_pip(R, bh, n, p_init=p, burn_in=BURN, num_iter=ITER,
                           n_chains=2, seed=1).posterior_mean

    rows = []
    for p in P_GRID:
        beta, bh, Rf = make_sumstats(R, m, p, seed=100 + int(-np.log10(p)))
        for name, fn in (("grid", grid), ("finemap", pip)):
            t_const = best_time(lambda: fn(n_const, p))
            t_var = best_time(lambda: fn(n_var, p))
            r2_const = genetic_r2(fn(n_const, p), beta, Rf)
            r2_var = genetic_r2(fn(n_var, p), beta, Rf)
            rows.append({
                "method": name, "p": p, "m": m,
                "const_ms": t_const * 1e3, "var_ms": t_var * 1e3,
                "ratio": t_var / t_const if t_const else float("nan"),
                "r2_const": r2_const, "r2_var": r2_var,
            })

    print(f"Constant-N fast path, m={m}, N={int(N_GWAS)}, "
          f"burn_in={BURN}/num_iter={ITER}, best of {REPS}\n")
    print(f"{'method':>8} {'p':>7} | {'fast path':>9} {'per-SNP':>10} {'saved':>7} "
          f"| {'R² fast':>9} {'R² slow':>8}")
    print("-" * 66)
    for r in rows:
        print(f"{r['method']:>8} {r['p']:>7.0e} | {r['const_ms']:>7.1f}ms "
              f"{r['var_ms']:>8.1f}ms {1 - 1 / r['ratio']:>6.0%} "
              f"| {r['r2_const']:>9.3f} {r['r2_var']:>8.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampler_fastpath.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
