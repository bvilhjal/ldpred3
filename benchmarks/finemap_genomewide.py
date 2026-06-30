"""Genome-wide fine-mapping: `finemap_by_blocks` across a whole genome.

Builds a genome of many realistic-LD blocks (cycled from ``ld_library.npz``) with
a sparse set of strong causal variants scattered across blocks, then runs
LDpred3-PIP fine-mapping genome-wide and scores genome-wide credible-set
**coverage** (a set contains a true causal), **power** (causals captured) and
resolution. Two parts:

  (A) the realistic ``only_significant`` mode (fine-map only loci around
      genome-wide-significant hits) across genome sizes -- recovery + runtime;
  (B) all-blocks vs only_significant at one size -- the fixed sparse prior should
      keep null blocks from emitting spurious credible sets.

Needs ``ld_library.npz``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/finemap_genomewide.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import finemap_by_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
NLIB = libR.shape[0]
K = libR.shape[1]
N_GWAS = 100000
Z = 8.0                 # per-causal marginal z (clearly significant, fine-mappable)
SIGNAL_FRAC = 0.10      # fraction of blocks carrying a causal
SIG_P = 5e-8

chol = [np.linalg.cholesky(libR[i] + 1e-4 * np.eye(K)) for i in range(NLIB)]


def build_genome(nb, seed):
    rng = np.random.default_rng(seed)
    blocks, beta, causal = [], np.zeros(nb * K), []
    for b in range(nb):
        blocks.append((libR[b % NLIB].astype(np.float32),
                       np.arange(b * K, (b + 1) * K)))
        if rng.random() < SIGNAL_FRAC:
            gj = b * K + int(rng.integers(K))
            beta[gj] = rng.choice([-1.0, 1.0]) * Z / np.sqrt(N_GWAS)
            causal.append(gj)
    bhat = np.empty(nb * K)
    for b in range(nb):
        R = libR[b % NLIB]
        ix = slice(b * K, (b + 1) * K)
        bhat[ix] = R @ beta[ix] + (chol[b % NLIB] @ rng.standard_normal(K)) / np.sqrt(N_GWAS)
    return blocks, bhat, np.array(sorted(causal))


def score(res, causal):
    cset = set(int(c) for c in causal)
    sets = [set(int(v) for v in cs.variants) for cs in res.credible_sets]
    n_cs = len(sets)
    covered = sum(1 for s in sets if cset & s)
    found = sum(1 for c in causal if any(c in s for s in sets))
    sizes = [len(s) for s in sets]
    return n_cs, covered, found, sizes


# (A) only_significant mode across genome sizes -----------------------------
print(f"Genome-wide fine-mapping, realistic LD (blocks of {K}), N={N_GWAS}, "
      f"causal z={Z}, ~{SIGNAL_FRAC:.0%} of blocks carry a causal\n")
print("(A) only_significant=5e-8 (fine-map loci around hits) vs genome size:")
hdr = (f"{'#SNPs':>8} | {'#blocks':>7} | {'fmap':>5} | {'causals':>7} | "
       f"{'coverage':>8} | {'power':>6} | {'med|CS|':>7} | {'time(s)':>7}")
print(hdr); print("-" * len(hdr))
for nb in (200, 500, 1000):                       # 100k, 250k, 500k SNPs
    blocks, bhat, causal = build_genome(nb, seed=100 + nb)
    n = np.full(nb * K, float(N_GWAS))
    t0 = time.time()
    res = finemap_by_blocks(blocks, bhat, n, only_significant=SIG_P, seed=1)
    dt = time.time() - t0
    n_cs, covered, found, sizes = score(res, causal)
    cov = covered / n_cs if n_cs else float("nan")
    pw = found / len(causal) if len(causal) else float("nan")
    print(f"{nb * K:>8} | {nb:>7} | {res.diagnostics['n_blocks_finemapped']:>5} | "
          f"{len(causal):>7} | {cov:>8.2f} | {pw:>6.2f} | "
          f"{np.median(sizes):>7.0f} | {dt:>7.1f}")

# (B) all-blocks vs only_significant at one size ----------------------------
nb = 200
blocks, bhat, causal = build_genome(nb, seed=300)
n = np.full(nb * K, float(N_GWAS))
print(f"\n(B) all-blocks vs only_significant at m={nb * K} "
      f"({len(causal)} causals in {nb} blocks):")
print(f"{'mode':>18} | {'blocks fmap':>11} | {'#CS':>4} | {'false CS':>8} | "
      f"{'coverage':>8} | {'power':>6} | {'time(s)':>7}")
print("-" * 74)
for label, only in (("all blocks", None), ("only_significant", SIG_P)):
    t0 = time.time()
    res = finemap_by_blocks(blocks, bhat, n, only_significant=only, seed=1)
    dt = time.time() - t0
    n_cs, covered, found, sizes = score(res, causal)
    cov = covered / n_cs if n_cs else float("nan")
    pw = found / len(causal) if len(causal) else float("nan")
    print(f"{label:>18} | {res.diagnostics['n_blocks_finemapped']:>11} | {n_cs:>4} | "
          f"{n_cs - covered:>8} | {cov:>8.2f} | {pw:>6.2f} | {dt:>7.1f}")
print("\n(false CS = credible sets not containing a true causal; the fixed sparse "
      "prior keeps null blocks quiet even in all-blocks mode.)")
