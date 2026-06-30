"""Inference evaluation: does LDpred3-auto-infer recover h² and polygenicity?

Self-contained (no external LD library). Simulates a GWAS on a coalescent
(realistic-LD) panel,
runs multi-chain LDpred3-auto inference (no validation cohort), and checks the
posterior median and 95% credible interval against the known truth:

  (A) Heritability h²: swept with p fixed.
  (B) Polygenicity p: swept with h² fixed.

For each true value it reports the mean estimate, the mean CI width, and the
**empirical coverage** -- the fraction of replicates whose 95% CI contains the
truth (should be ~0.95 if the intervals are calibrated).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/infer_recovery.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks
from ldpred3.infer import ldpred3_auto_infer

NB, K = 15, 200            # m = 3000
M = NB * K
N_REF = 8000               # LD reference panel size
N_GWAS = 40000
REPS = 10
CHAINS, BURN, ITER = 6, 150, 150


def simulate(h2, p, seed):
    rng = np.random.default_rng(seed)
    G, _ = simulate_genotypes_coalescent(N_REF, M, K, seed=seed)   # realistic LD
    blocks = compute_ld_blocks(G, block_size=K)
    Rf = [(R.astype(float), idx) for R, idx in blocks]

    def gv(a, b):
        return sum(a[ix] @ (R @ b[ix]) for R, ix in Rf)

    causal = rng.random(M) < p
    beta = np.zeros(M); beta[causal] = rng.standard_normal(int(causal.sum()))
    beta *= np.sqrt(h2 / gv(beta, beta))
    beta_hat = np.empty(M)
    for R, ix in Rf:
        chol = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        beta_hat[ix] = R @ beta[ix] + (chol @ rng.standard_normal(len(ix))) / np.sqrt(N_GWAS)
    return blocks, beta_hat


def infer(blocks, beta_hat, seed):
    return ldpred3_auto_infer(blocks, beta_hat, np.full(M, float(N_GWAS)),
                              n_chains=CHAINS, burn_in=BURN, num_iter=ITER, seed=seed)


t0 = time.time()
print(f"Inference recovery, coalescent LD, m={M} ({NB}x{K}), Nref={N_REF}, "
      f"N_gwas={N_GWAS}, {REPS} reps, {CHAINS} chains\n")

print("(A) Heritability h² (p fixed = 0.02):")
print(f"{'true h2':>8} | {'est (mean)':>10} | {'CI width':>9} | {'coverage':>8}")
print("-" * 46)
for h2 in (0.1, 0.3, 0.5, 0.8):
    ests, widths, hits = [], [], 0
    for rep in range(REPS):
        blocks, bh = simulate(h2, 0.02, 200 + rep)
        r = infer(blocks, bh, rep)
        ests.append(r.h2_est); widths.append(r.h2_ci[1] - r.h2_ci[0])
        hits += r.h2_ci[0] <= h2 <= r.h2_ci[1]
    print(f"{h2:>8.2f} | {np.mean(ests):>10.3f} | {np.mean(widths):>9.3f} | "
          f"{hits/REPS:>8.2f}")

print("\n(B) Polygenicity p (h² fixed = 0.5):")
print(f"{'true p':>8} | {'est (mean)':>10} | {'CI width':>9} | {'coverage':>8}")
print("-" * 46)
for p in (0.005, 0.02, 0.1):
    ests, widths, hits = [], [], 0
    for rep in range(REPS):
        blocks, bh = simulate(0.5, p, 300 + rep)
        r = infer(blocks, bh, rep)
        ests.append(r.p_est); widths.append(r.p_ci[1] - r.p_ci[0])
        hits += r.p_ci[0] <= p <= r.p_ci[1]
    print(f"{p:>8.3f} | {np.mean(ests):>10.4f} | {np.mean(widths):>9.4f} | "
          f"{hits/REPS:>8.2f}")

print(f"\n({time.time()-t0:.0f}s)")
