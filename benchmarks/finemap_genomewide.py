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
from math import erfc, sqrt
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


def build_genome(nb, seed, z=Z):
    rng = np.random.default_rng(seed)
    blocks, beta, causal = [], np.zeros(nb * K), []
    for b in range(nb):
        blocks.append((libR[b % NLIB].astype(np.float32),
                       np.arange(b * K, (b + 1) * K)))
        if rng.random() < SIGNAL_FRAC:
            gj = b * K + int(rng.integers(K))
            beta[gj] = rng.choice([-1.0, 1.0]) * z / np.sqrt(N_GWAS)
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

# (B) WEAK causals: causal-strength sweep, both modes ----------------------
# The hard, realistic regime. With a genome-wide-significance gate the locus of a
# weak causal never reaches significance, so it is never fine-mapped (a DETECTION
# limit); fine-mapping every block instead exposes the FINE-MAPPER limit at low
# power (and its false-set cost). The two-sided p of each z marks the 5e-8 gate
# (|z| > 5.45).
nb = 200
print(f"\n(B) weak causals -- power by causal strength at m={nb * K} "
      f"(~{int(SIGNAL_FRAC * nb)} causals; 5e-8 gate is |z|>5.45):")
print(f"{'z':>4} | {'p2-sided':>9} || {'only-sig pow':>12} | {'cov':>4} | "
      f"{'|CS|':>4} || {'all-blk pow':>11} | {'cov':>4} | {'falseCS':>7} | {'|CS|':>4}")
print("-" * 86)
for z in (4.0, 5.0, 6.0, 8.0):
    blocks, bhat, causal = build_genome(nb, seed=300 + int(z), z=z)
    n = np.full(nb * K, float(N_GWAS))
    p2 = erfc(z / sqrt(2.0))
    out = {}
    for only in (SIG_P, None):
        res = finemap_by_blocks(blocks, bhat, n, only_significant=only, seed=1)
        n_cs, covered, found, sizes = score(res, causal)
        out[only] = (found / len(causal), covered / n_cs if n_cs else float("nan"),
                     n_cs - covered, np.median(sizes) if sizes else float("nan"))
    sg, ab = out[SIG_P], out[None]
    print(f"{z:>4.0f} | {p2:>9.1e} || {sg[0]:>12.2f} | {sg[1]:>4.2f} | "
          f"{sg[3]:>4.0f} || {ab[0]:>11.2f} | {ab[1]:>4.2f} | {ab[2]:>7} | {ab[3]:>4.0f}")

print("\nReading: weak causals (z=4-5) sit below the 5e-8 gate, so only-sig never "
      "fine-maps them (power -> 0) -- a DETECTION limit. Fine-mapping every block "
      "recovers some (all-blk pow) but at low power and a rising false-set cost -- "
      "the FINE-MAPPER limit. Both are targets for improvement (sub-threshold "
      "locus selection; PIP calibration at low power).")
