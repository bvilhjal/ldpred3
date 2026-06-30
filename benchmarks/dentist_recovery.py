"""DENTIST LD-consistency filter: does it recover accuracy, and what does it cost?

Self-contained (no external LD library). Simulates a genotype panel with block
LD, draws a sparse genetic architecture, and builds marginal summary statistics.
Then it measures the DENTIST filter (``qc.dentist_outlier_mask``, exposed as
``--dentist``) in the two regimes that matter:

  (A) Corrupted sumstats: spurious genome-wide-significant hits are planted at
      non-causal variants (an allele/strand error that inflates a null variant's
      z out of line with its LD neighbours). Reports genetic R2 of the auto PRS
      with vs without DENTIST, and how many of the planted errors it catches.
  (B) Clean sumstats: no errors planted. Reports how many *genuine* variants
      DENTIST drops (the false-positive cost behind the "off by default" choice).

Run single-core for stable numbers:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/dentist_recovery.py
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks
from ldpred3.qc import dentist_outlier_mask
from ldpred3 import ldpred3_by_blocks

NB, K = 20, 200            # 20 LD blocks of 200 SNPs -> m = 4000
M = NB * K
N_REF = 10000              # individuals used to estimate the LD panel
N_GWAS = 10000             # GWAS sample size
H2, P = 0.5, 0.05          # heritability, polygenicity (sparse)
N_CORRUPT = 30             # spurious hits planted at non-causal variants
CORRUPT_Z = 8.0            # z-score of each planted false association
REPS = 5


def build(seed):
    """One replicate: estimated LD blocks, true betas, marginal sumstats."""
    rng = np.random.default_rng(seed)
    G, _ = simulate_genotypes_coalescent(N_REF, M, K, seed=seed)   # realistic LD
    blocks = compute_ld_blocks(G, block_size=K)          # (R, idx) per block
    Rfull = [(R.astype(float), idx) for R, idx in blocks]

    def gv(a, b):                                         # genetic covariance
        return sum(a[ix] @ (R @ b[ix]) for R, ix in Rfull)

    causal = rng.random(M) < P
    beta = np.zeros(M)
    beta[causal] = rng.standard_normal(int(causal.sum()))
    beta *= np.sqrt(H2 / gv(beta, beta))

    beta_hat = np.empty(M)                                # marginal = R beta + noise
    for R, ix in Rfull:
        chol = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        beta_hat[ix] = R @ beta[ix] + (chol @ rng.standard_normal(len(ix))) / np.sqrt(N_GWAS)
    return blocks, Rfull, gv, beta, beta_hat, causal, rng


def subset_blocks(blocks, keep):
    """Restrict each (R, idx) block to survivors, re-indexed to tile 0..n_kept-1.

    Returns ``(relabelled_blocks, orig_idx)`` where ``orig_idx`` maps each new
    contiguous position back to its global variant index (to scatter weights).
    """
    out, orig = [], []
    offset = 0
    for R, idx in blocks:
        loc = keep[idx]
        k = int(loc.sum())
        if k:
            out.append((np.asarray(R)[np.ix_(loc, loc)], np.arange(offset, offset + k)))
            orig.append(np.asarray(idx)[loc])
            offset += k
    return out, np.concatenate(orig)


def fit_r2(blocks, beta_hat, keep, gv, beta):
    """Fit auto on the kept variants; score genetic R2 over ALL variants."""
    n = np.full(M, float(N_GWAS))
    if keep is None:
        be = ldpred3_by_blocks(blocks, beta_hat, n, method="auto",
                               burn_in=80, num_iter=150, seed=0)
    else:
        sub, kept_idx = subset_blocks(blocks, keep)
        be_sub = ldpred3_by_blocks(sub, beta_hat[kept_idx], n[kept_idx],
                                   method="auto", burn_in=80, num_iter=150, seed=0)
        be = np.zeros(M)                                  # dropped variants -> 0
        be[kept_idx] = be_sub
    num = gv(be, beta)
    den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


t0 = time.time()
print(f"DENTIST recovery, coalescent LD, m={M} ({NB}x{K}), Nref={N_REF}, "
      f"N_gwas={N_GWAS}, h2={H2}, p={P}, {REPS} reps\n")

cleanA, noflt, dent, caught, planted = [], [], [], [], []
false_drop = []
for rep in range(REPS):
    blocks, Rfull, gv, beta, beta_hat, causal, rng = build(1000 + rep)

    # (A) plant spurious hits at NON-causal variants (allele/strand error that
    # inflates a null variant's z out of line with its LD neighbours).
    noncausal = np.flatnonzero(~causal)
    bad_idx = rng.choice(noncausal, N_CORRUPT, replace=False)
    bad = beta_hat.copy()
    bad[bad_idx] = rng.choice([-1.0, 1.0], N_CORRUPT) * CORRUPT_Z / np.sqrt(N_GWAS)
    z_bad = bad * np.sqrt(N_GWAS)

    cleanA.append(fit_r2(blocks, beta_hat, None, gv, beta))      # no errors, no filter
    noflt.append(fit_r2(blocks, bad, None, gv, beta))            # errors, no filter
    keep_bad, _ = dentist_outlier_mask(blocks, z_bad)
    dent.append(fit_r2(blocks, bad, keep_bad, gv, beta))         # errors + DENTIST
    dropped = ~keep_bad
    caught.append(int(dropped[bad_idx].sum())); planted.append(N_CORRUPT)

    # (B) clean data: count genuine variants DENTIST drops (false positives).
    z_clean = beta_hat * np.sqrt(N_GWAS)
    keep_clean, log_clean = dentist_outlier_mask(blocks, z_clean)
    false_drop.append(int((~keep_clean).sum()))

print("(A) Corrupted sumstats (spurious hits planted at non-causal variants):")
print(f"{'condition':>26} | {'genetic R2':>10}")
print("-" * 41)
print(f"{'clean (no errors)':>26} | {np.mean(cleanA):>10.3f}")
print(f"{'corrupted, no filter':>26} | {np.mean(noflt):>10.3f}")
print(f"{'corrupted, --dentist':>26} | {np.mean(dent):>10.3f}")
print(f"\n  planted errors/rep: {np.mean(planted):.0f}; "
      f"caught by DENTIST: {np.mean(caught):.1f} "
      f"({100*np.mean(caught)/np.mean(planted):.0f}%)")

print(f"\n(B) Clean sumstats: genuine variants dropped (false positives): "
      f"{np.mean(false_drop):.1f} / {M} "
      f"({100*np.mean(false_drop)/M:.2f}%)")
print(f"\n({time.time()-t0:.0f}s)")
