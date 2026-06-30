"""MAF-dependent effect-size prior (`alpha` / `use_MLE`, Privé et al. 2023).

When the true effect sizes are coupled to allele frequency, the standard
point-normal prior (`alpha = -1`, equal variance per *standardised* effect) is
misspecified. LDpred3's `alpha` knob scales each variant's slab variance by
`[2f(1-f)]^(1+alpha)`; matching it to the true coupling recovers accuracy.

This benchmark simulates *realistic* genotypes (coalescent LD + genuine allele
frequencies), draws causal standardised effects with variance
`∝ [2f(1-f)]^(1+alpha_true)`, generates summary statistics, and fits LDpred3
`auto` across a grid of prior `alpha`. Metric: genetic R² = (β̂ᵀRβ)² /
[(β̂ᵀRβ̂)(βᵀRβ)] under population LD, averaged over replicates. Self-contained
(needs msprime); writes ``maf_alpha_prior.csv``.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3 import ldpred3_by_blocks                       # noqa: E402
from ldpred3.simulate import simulate_genotypes_coalescent  # noqa: E402

K = 500                       # SNPs per LD block
NB = 20                       # blocks  ->  M = 10,000 SNPs
M = NB * K
N_REF = 4000                  # individuals defining the population LD
H2 = 0.5
N_GWAS = 5000                 # GWAS sample size (low power: the prior matters)
P_CAUSAL = 0.1                # polygenic: the variance allocation matters
REPS = 8
TRUE_ALPHAS = [-1.0, -0.5, 0.0]       # truth: -1 = standard model
FIT_ALPHAS = [-1.0, -0.75, -0.5, -0.25, 0.0]

# --- one realistic genome: coalescent LD + real allele frequencies -----------
print(f"simulating {N_REF} individuals x {M} SNPs (coalescent) ...", flush=True)
t0 = time.time()
G, _blocks = simulate_genotypes_coalescent(N_REF, M, K, seed=7)
af = G.mean(axis=0) / 2.0                                   # per-SNP frequency
het = 2.0 * af * (1.0 - af)
Gs = (G - G.mean(0)) / G.std(0)

blocks, chols, idxs = [], [], []
for b in range(NB):
    ix = np.arange(b * K, (b + 1) * K)
    R = (Gs[:, ix].T @ Gs[:, ix]) / N_REF
    np.fill_diagonal(R, 1.0)
    blocks.append((R.astype(np.float32), ix))
    chols.append(np.linalg.cholesky(R + 1e-4 * np.eye(K)))
    idxs.append(ix)
print(f"  ({time.time() - t0:.0f}s)  M={M}, mean het={het.mean():.3f}", flush=True)


def make_beta(alpha_true, rng):
    """Causal standardised effects with var ∝ [2f(1-f)]^(1+alpha_true)."""
    beta = np.zeros(M)
    causal = rng.random(M) < P_CAUSAL
    sd = np.sqrt(np.maximum(het[causal], 1e-12) ** (1.0 + alpha_true))
    beta[causal] = rng.normal(0.0, 1.0, causal.sum()) * sd
    gv = sum(beta[ix] @ (blocks[b][0].astype(float) @ beta[ix])
             for b, ix in enumerate(idxs))
    if gv > 0:
        beta *= np.sqrt(H2 / gv)                            # fix total h2
    return beta


def sumstats(beta, rng):
    bhat = np.empty(M)
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        bhat[ix] = R @ beta[ix] + (chols[b] @ rng.standard_normal(K)) / np.sqrt(N_GWAS)
    return bhat


def genetic_r2(b_est, beta):
    num = den1 = den2 = 0.0
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        Rb = R @ beta[ix]
        num += b_est[ix] @ Rb
        den1 += b_est[ix] @ (R @ b_est[ix])
        den2 += beta[ix] @ Rb
    return float(num * num / (den1 * den2)) if den1 > 0 and den2 > 0 else 0.0


def fit_auto(bhat, n, alpha):
    # Per-block auto for every column so the *only* thing that varies is the
    # prior alpha (alpha=-1 reduces the slab weights to all-ones, i.e. the
    # standard model on the same per-block sampler -- a fair comparison).
    return ldpred3_by_blocks(blocks, bhat, n, method="auto", burn_in=80,
                             num_iter=200, seed=1, af=af, alpha=alpha,
                             global_hyper=False)


n_vec = np.full(M, float(N_GWAS))
rows = []
print(f"\nGenetic R2, realistic LD + MAF-coupled effects, m={M}, N={N_GWAS}, "
      f"h2={H2}, p={P_CAUSAL}, {REPS} reps\n", flush=True)
hdr = f"{'true alpha':>11} | " + " ".join(f"a={a:>5}" for a in FIT_ALPHAS) + "  | best"
print(hdr)
print("-" * len(hdr))
for at in TRUE_ALPHAS:
    per_fit = {fa: [] for fa in FIT_ALPHAS}
    for rep in range(REPS):
        rng = np.random.default_rng(2000 + rep)
        beta = make_beta(at, rng)
        bhat = sumstats(beta, rng)
        for fa in FIT_ALPHAS:
            per_fit[fa].append(genetic_r2(fit_auto(bhat, n_vec, fa), beta))
    means = [float(np.mean(per_fit[fa])) for fa in FIT_ALPHAS]
    best = FIT_ALPHAS[int(np.argmax(means))]
    print(f"{at:>11} | " + " ".join(f"{x:>7.3f}" for x in means)
          + f"  | {best:>+.2f}")
    rows.append([at] + [round(x, 4) for x in means] + [best])

with open("maf_alpha_prior.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true_alpha"] + [f"fit_alpha_{a}" for a in FIT_ALPHAS] + ["best_fit_alpha"])
    w.writerows(rows)
print(f"\n({time.time() - t0:.0f}s)  wrote maf_alpha_prior.csv", flush=True)
