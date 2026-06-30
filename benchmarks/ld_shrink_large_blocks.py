"""Size-aware LD shrinkage of large blocks on a finite reference panel.

Self-contained (coalescent genotypes). A block's sample LD from ``Nref`` reference
individuals is noise-dominated when the block is large relative to ``Nref``
(Marchenko-Pastur). This shows that shrinking *large* blocks toward the identity
-- ``alpha = min(max_shrink, k / Nref)`` per block, via
``ldpred3.shrink_ld_blocks`` -- recovers PRS accuracy and tames h² over-fitting,
while leaving small, well-estimated blocks essentially untouched.

The genome mixes small and large LD blocks. GWAS sumstats are drawn from the
*population* LD (a large sample); the model is fit with LD estimated from a
finite reference panel of ``Nref`` individuals. Compared: no shrinkage, a uniform
ridge, and the size-aware rule. Pure PC truncation is intentionally absent -- it
keeps the MP-inflated top eigenvalues and does not help here.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/ld_shrink_large_blocks.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3.ld import compute_ld_blocks
from ldpred3 import ldpred3_by_blocks, shrink_ld_blocks

SIZES = [100, 200, 300, 500, 800, 1100]   # mixed: small well-estimated + large noisy
M = sum(SIZES)
N_POP = 8000               # "population" sample -> true LD
N_GWAS = 50000
H2, P = 0.5, 0.01
REPS = 3
NREFS = [500, 1000, 2000]


def population(seed=0):
    # One coalescent region of M SNPs (realistic LD), partitioned into SIZES blocks.
    G, _ = simulate_genotypes_coalescent(N_POP, M, M, seed=seed)
    pop = compute_ld_blocks(G, chrom=_block_chrom(), block_size=max(SIZES))
    Rp = [(R.astype(float), idx) for R, idx in pop]
    chol = [np.linalg.cholesky(R + 1e-4 * np.eye(len(idx))) for R, idx in Rp]
    return G, Rp, chol


def _block_chrom():
    # one "chromosome" label per block so compute_ld_blocks honours SIZES exactly
    return np.concatenate([np.full(k, i, np.int32) for i, k in enumerate(SIZES)])


def gv(Rp, a, b):
    return sum(a[ix] @ (R @ b[ix]) for R, ix in Rp)


def r2(Rp, be, beta):
    num = gv(Rp, be, beta); den = gv(Rp, be, be) * gv(Rp, beta, beta)
    return float(num * num / den) if den > 0 else 0.0


Gpop, Rp, chol = population()
print(f"LD shrinkage of large blocks, coalescent LD, mixed blocks {SIZES} (m={M}), "
      f"N_gwas={N_GWAS}, h2={H2}, p={P}, {REPS} reps\n")
print(f"{'Nref':>5} | {'no-shrink':>9} | {'uniform .1':>10} | {'size-aware':>10} "
      f"|| {'h2 raw':>6} | {'h2 sz':>6}")
print("-" * 64)

for nref in NREFS:
    acc = {"raw": [], "uni": [], "sz": [], "hraw": [], "hsz": []}
    for rep in range(REPS):
        rng = np.random.default_rng(10 + rep)
        c = rng.random(M) < P
        beta = np.zeros(M); beta[c] = rng.standard_normal(int(c.sum()))
        beta *= np.sqrt(H2 / gv(Rp, beta, beta))
        bh = np.empty(M)
        for (R, ix), ch in zip(Rp, chol):
            bh[ix] = R @ beta[ix] + (ch @ rng.standard_normal(len(ix))) / np.sqrt(N_GWAS)
        # reference panel of nref individuals: a finite subsample of the
        # population (realistic finite-panel LD noise, Marchenko-Pastur).
        ref_rows = rng.choice(N_POP, nref, replace=False)
        ref = [(R.astype(np.float32), idx) for R, idx in
               compute_ld_blocks(Gpop[ref_rows], chrom=_block_chrom(),
                                 block_size=max(SIZES))]
        uni = [((1 - 0.1) * np.asarray(R, float) + 0.1 * np.eye(len(idx)), idx)
               for R, idx in ref]
        for u in uni:
            np.fill_diagonal(u[0], 1.0)
        sz = shrink_ld_blocks(ref, nref, max_shrink=0.5)

        def fit(bl):
            return ldpred3_by_blocks(bl, bh, np.full(M, float(N_GWAS)),
                                     method="auto", burn_in=100, num_iter=120, seed=rep)
        ber, beu, bes = fit(ref), fit(uni), fit(sz)
        acc["raw"].append(r2(Rp, ber, beta)); acc["uni"].append(r2(Rp, beu, beta))
        acc["sz"].append(r2(Rp, bes, beta))
        acc["hraw"].append(gv(Rp, ber, ber)); acc["hsz"].append(gv(Rp, bes, bes))
    m = {k: np.mean(v) for k, v in acc.items()}
    print(f"{nref:>5} | {m['raw']:>9.3f} | {m['uni']:>10.3f} | {m['sz']:>10.3f} "
          f"|| {m['hraw']:>6.2f} | {m['hsz']:>6.2f}")

print(f"\n(true h2 = {H2}; size-aware = min(0.5, k/Nref) per block)")
