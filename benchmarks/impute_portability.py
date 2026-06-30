"""Cross-ancestry portability: does imputation + annotation transfer better?

The portability hypothesis. A causal *functional* variant is shared across
populations, but its LD tags are not. If the discovery GWAS (population A) puts
the effect on A-specific tags, the PRS transfers poorly to population B. If
imputation makes the (untyped) functional variant available and a functional
annotation places the effect there, the PRS should transfer better -- because the
shared causal carries the same effect in B while the A-tags do not.

This simulates two populations with a coalescent split (msprime): shared
ancestral variants but **diverged LD**. Causals (shared across A and B, enriched
in a 20% functional annotation) are then **dropped from the GWAS** (untyped). The
GWAS and LD/imputation use population A; the PRS is scored as genetic R2 in
population A (in-sample) and population B (cross-ancestry), for four pipelines:

  drop/auto, drop/annot, impute/auto, impute/annot

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/impute_portability.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import (ldpred3_by_blocks, ldpred3_auto_annot_blocks,
                     impute_sumstats_blocks)

NB, K = 8, 300            # blocks / SNPs per block
N_IND = 1500             # diploid individuals per population (for population LD)
T_SPLIT = 3000           # generations since the A/B split (moderate divergence)
NE = 10000
N_GWAS = 50000
H2 = 0.5
FUNC_FRAC = 0.20
P_CAUSAL = 0.02
ENRICH = 12.0
REPS = 4
MIN_MAF = 0.05


def two_pop_block(seq_seed):
    """One region: population-A and -B dosages on the same common sites."""
    import msprime
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=NE)
    dem.add_population(name="B", initial_size=NE)
    dem.add_population(name="ANC", initial_size=NE)
    dem.add_population_split(time=T_SPLIT, derived=["A", "B"], ancestral="ANC")
    seq = max(5e5, K / 1000 * 1e6)
    rng = np.random.default_rng(seq_seed)
    for _ in range(8):
        s = int(rng.integers(1, 2 ** 31 - 1))
        ts = msprime.sim_ancestry(
            samples={"A": N_IND, "B": N_IND}, ploidy=2, demography=dem,
            recombination_rate=1e-8, sequence_length=int(seq), random_seed=s)
        mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=s,
                                    model=msprime.BinaryMutationModel())
        G = mts.genotype_matrix()                       # (sites, n_haplotypes)
        pop = np.array([ts.node(n).population for n in ts.samples()])
        Ah = G[:, pop == 0]; Bh = G[:, pop == 1]
        dosA = (Ah[:, 0::2] + Ah[:, 1::2]).T            # (N_IND, sites)
        dosB = (Bh[:, 0::2] + Bh[:, 1::2]).T
        afA = dosA.mean(0) / 2; afB = dosB.mean(0) / 2
        common = ((afA > MIN_MAF) & (afA < 1 - MIN_MAF) &
                  (afB > MIN_MAF) & (afB < 1 - MIN_MAF))
        if common.sum() >= K:
            sel = np.flatnonzero(common)[:K]
            return dosA[:, sel].astype(float), dosB[:, sel].astype(float)
        seq *= 1.8
    raise RuntimeError("not enough common sites; raise sequence length")


def corr(G):
    Gs = (G - G.mean(0)) / G.std(0)
    R = (Gs.T @ Gs) / G.shape[0]
    np.fill_diagonal(R, 1.0)
    return R


print(f"Cross-ancestry portability, two-pop coalescent (T_split={T_SPLIT}), "
      f"m={NB * K}, N_gwas={N_GWAS}, functional={FUNC_FRAC:.0%}, {REPS} reps")
print("building two-population LD ...", flush=True)
t0 = time.time()
RA, RB, cholA, idxs = [], [], [], []
for b in range(NB):
    dA, dB = two_pop_block(seq_seed=10 + b)
    Ra, Rb = corr(dA), corr(dB)
    RA.append(Ra); RB.append(Rb)
    cholA.append(np.linalg.cholesky(Ra + 1e-4 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))
M = NB * K
blocksA = [(RA[b].astype(np.float32), idxs[b]) for b in range(NB)]
rng0 = np.random.default_rng(0)
func = rng0.random(M) < FUNC_FRAC
A_annot = func.astype(float)[:, None]
print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)


def gv(Rlist, a, b):
    return sum(a[ix] @ (Rlist[i] @ b[ix]) for i, ix in enumerate(idxs))


def r2(Rlist, be, beta):
    num = gv(Rlist, be, beta); den = gv(Rlist, be, be) * gv(Rlist, beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def simulate(rng):
    base = np.where(func, ENRICH, 1.0)
    pc = np.clip(base / base.sum() * (P_CAUSAL * M), 0, 1)
    causal = rng.random(M) < pc
    beta = np.zeros(M); beta[causal] = rng.normal(0, 1, int(causal.sum()))
    beta *= np.sqrt(H2 / gv(RA, beta, beta))
    bhat = np.empty(M)
    for i, ix in enumerate(idxs):
        bhat[ix] = RA[i] @ beta[ix] + (cholA[i] @ rng.standard_normal(K)) / np.sqrt(N_GWAS)
    untyped = causal & func
    return beta, bhat, ~untyped


def subset_blocks(blocks, keep):
    out, orig, off = [], [], 0
    for R, idx in blocks:
        loc = keep[idx]; k = int(loc.sum())
        if k:
            out.append((np.asarray(R)[np.ix_(loc, loc)], np.arange(off, off + k)))
            orig.append(np.asarray(idx)[loc]); off += k
    return out, np.concatenate(orig)


def fit(method, blocks, bhat, n, A_in):
    if method == "auto":
        return ldpred3_by_blocks(blocks, bhat, n, method="auto",
                                 burn_in=80, num_iter=150, seed=1)
    return ldpred3_auto_annot_blocks(blocks, bhat, n, A_in, burn_in=80,
                                     num_iter=150, seed=1).beta_est


def run(pipe, method, beta, bhat, typed):
    n_full = np.full(M, float(N_GWAS))
    if pipe == "drop":
        sub, orig = subset_blocks(blocksA, typed)
        be = np.zeros(M); be[orig] = fit(method, sub, bhat[orig], n_full[orig], A_annot[orig])
    else:
        imp = impute_sumstats_blocks(bhat, blocksA, typed, N_GWAS)
        be = fit(method, blocksA, imp.beta_hat, imp.n_eff, A_annot)
    return r2(RA, be, beta), r2(RB, be, beta)


CASES = [("drop", "auto"), ("drop", "annot"), ("impute", "auto"), ("impute", "annot")]
print(f"{'pipeline':>14} | {'R2 pop A':>9} | {'R2 pop B':>9} | {'retained B/A':>12}")
print("-" * 54)
for pipe, meth in CASES:
    a, bb = [], []
    for rep in range(REPS):
        beta, bhat, typed = simulate(np.random.default_rng(800 + rep))
        ra, rb = run(pipe, meth, beta, bhat, typed)
        a.append(ra); bb.append(rb)
    ma, mb = np.mean(a), np.mean(bb)
    print(f"{pipe + '/' + meth:>14} | {ma:>9.3f} | {mb:>9.3f} | {mb / ma:>11.0%}")
print(f"\n(R2 pop A = in-discovery-population; R2 pop B = cross-ancestry transfer. "
      f"{time.time()-t0:.0f}s)")
