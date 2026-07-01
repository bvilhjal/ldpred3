"""LDpred3 scaling with SNP *density*, on realistic heterogeneous LD.

Four modelling choices make this a realistic genome-scale picture rather than a
tidy extrapolation:

1. **Block count is set by recombination, not density.** The human genome has
   ~1,700 approximately-independent, recombination-delimited LD blocks (Berisa &
   Pickrell 2016), and that count barely moves as you densify: going from an
   array (~1M SNPs) to imputed / WGS (~10M+) adds only ~1.2x more blocks per 10x
   SNPs. So block *size* grows ~linearly with #SNPs, instead of tiling ever more
   same-size blocks.

2. **The density lever is the mutation rate, on fixed chromosomes.** Each block
   is a *fixed* coalescent-with-recombination segment (msprime) -- a physical
   length and recombination rate drawn once and held across all densities -- and
   we densify by raising the **mutation rate**, i.e. finding more variants in the
   *same* recombination structure (exactly what array -> imputed -> WGS does).
   Segment lengths (log-normal) and recombination rates (0.5x-2x) vary across the
   pool, so block sizes and internal LD decay vary the way real chromosomes do.

3. **Realistic memory, four ways.** For each density we report the resident LD
   footprint under the representations a real run would actually use -- dense,
   banded (a genetic-distance window), low-rank, and on-disk / streaming -- not
   just the naive dense one. Dense and banded LD grow ~quadratically with density
   (a fixed cM window packs more SNPs as you densify), so they hit a RAM ceiling
   early; low-rank grows sub-quadratically (the block's effective rank is set by
   recombination, so redundant SNPs in tight LD collapse) and streaming keeps
   only one block resident.

4. Fit time (dense, single core) is measured for inf/grid/auto; per-block work is
   O(k^2) (Gibbs) / O(k^3) (inf's dense solve), so time grows super-linearly.

Writes ``ldpred3_scaling.csv`` and a two-panel figure ``ldpred3_scaling.png``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_ldpred3_scaling.py
    python benchmarks/bench_ldpred3_scaling.py 1000000 2000000    # specific sizes
"""
import sys, os, csv, json, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3.ld import sparsify_ld, lowrank_ld
from ldpred3.simulate import simulate_genotypes_by_mutation_rate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.environ.get("BENCH_WORK", "/tmp/bench_ldpred3_scaling")

# --- architecture + GWAS power (match bench_vs_bigsnpr for continuity) -------
H2, P, N = 0.5, 0.01, 50_000
BURN_IN, NUM_ITER = 100, 200
N_REF = 3000                 # individuals defining the coalescent (population) LD
POOL = 24                    # distinct chromosome segments, tiled to fill the genome

# --- coalescent chromosome geometry (fixed across densities) -----------------
NE, MIN_MAF = 10_000, 0.01
L_MEAN = 0.6e6               # mean segment physical length (bp) ~ an LD region
L_SIGMA = 0.4               # log-normal spread of segment lengths
REC_LO, REC_HI = 0.5, 2.0   # recombination-rate spread, x the 1e-8 default
# common SNPs per Mb at mut_rate=1e-8 (empirical for these Ne / MAF / n): ~1880
SNPS_PER_MB_AT_BASE = 1880.0

# --- realistic block model: ~1,700 blocks at 1M SNPs, x1.2 per 10x SNPs ------
NB_1M = 1700                 # Berisa & Pickrell 2016 (approx. independent blocks)
ALPHA = float(np.log10(1.2)) # block count grows as (m/1e6)^ALPHA  (~x1.2 / decade)
BAND_FRAC = 0.15             # banded window as a fraction of the block (~fixed cM)
LOWRANK_VAR = 0.995          # low-rank retained variance
RAM_GB = 15.0                # this machine, for the ceiling line

# One fixed set of "chromosomes": (length bp, recombination rate, ancestry seed).
_g = np.random.default_rng(20240701)
POOL_GEOM = [(float(L_MEAN * np.exp(_g.normal(0, L_SIGMA))),
              float(1e-8 * np.exp(_g.uniform(np.log(REC_LO), np.log(REC_HI)))),
              int(_g.integers(1, 2**31 - 1))) for _ in range(POOL)]
L_MEAN_ACTUAL = float(np.mean([L for L, _, _ in POOL_GEOM]))


def block_layout(nsnps):
    """(target n_blocks, mean block size) under the recombination-block model."""
    nb = int(round(NB_1M * (nsnps / 1e6) ** ALPHA))
    return nb, nsnps / nb


def mut_rate_for(mean_k):
    """Mutation rate that yields ~mean_k common SNPs in a mean-length segment."""
    return 1e-8 * mean_k / (SNPS_PER_MB_AT_BASE * L_MEAN_ACTUAL / 1e6)


SIZES = [int(float(a)) for a in sys.argv[1:]] or \
        [200_000, 500_000, 1_000_000, 2_000_000]   # dense LD caps ~2.6M on 15 GB


def sim_block(seq_len, recomb_rate, mut_rate, seed):
    """One fixed chromosome segment densified by the mutation rate -> LD matrix."""
    G = simulate_genotypes_by_mutation_rate(
        N_REF, seq_len, recomb_rate=recomb_rate, mut_rate=mut_rate,
        Ne=NE, min_maf=MIN_MAF, seed=seed).astype(np.float64)
    Gs = (G - G.mean(0)) / G.std(0)
    R = ((Gs.T @ Gs) / G.shape[0]).astype(np.float32)
    np.fill_diagonal(R, 1.0)
    return R


def make_pool(mean_k):
    """Simulate the fixed chromosomes at the mutation rate for this density."""
    mu = mut_rate_for(mean_k)
    return [sim_block(L, r, mu, seed) for (L, r, seed) in POOL_GEOM]


def tile_genome(pool, target_m):
    """Cycle the pool until the genome reaches target_m SNPs; return the plan."""
    tiling, m, b = [], 0, 0
    while m < target_m:
        i = b % len(pool)
        tiling.append(i); m += pool[i].shape[0]; b += 1
    return np.asarray(tiling), m


def rep_memory_gb(pool, tiling):
    """Resident LD footprint (GB) under each representation, exact for this genome."""
    counts = np.bincount(tiling, minlength=len(pool))
    dense = band = low = 0
    kmax = 0
    for i, R in enumerate(pool):
        k = R.shape[0]; c = int(counts[i]); kmax = max(kmax, k)
        dense += c * R.nbytes
        sp = sparsify_ld(R, threshold=1e-3, max_dist=max(1, round(BAND_FRAC * k)))
        band += c * (sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes)
        low += c * lowrank_ld(R, variance=LOWRANK_VAR).U.nbytes
    stream = kmax * kmax * 4          # one (largest) block resident
    return (dense / 1e9, band / 1e9, low / 1e9, stream / 1e9)


def build_beta_bhat(pool, tiling, rng):
    """Per-block causal betas scaled to H2, plus GWAS bhat, on the tiled genome."""
    sizes = [pool[i].shape[0] for i in tiling]
    off = np.concatenate([[0], np.cumsum(sizes)]); m = int(off[-1])
    chol = {i: np.linalg.cholesky(pool[i].astype(np.float64)
                                  + 1e-4 * np.eye(pool[i].shape[0]))
            for i in set(tiling.tolist())}
    beta = np.zeros(m)
    causal = rng.random(m) < P
    beta[causal] = rng.normal(0, 1, int(causal.sum()))
    gv = 0.0
    for b, i in enumerate(tiling):
        R = pool[i].astype(np.float64); bb = beta[off[b]:off[b + 1]]
        gv += bb @ (R @ bb)
    beta *= np.sqrt(H2 / gv)                     # total genetic variance == H2
    bhat = np.empty(m)
    for b, i in enumerate(tiling):
        R = pool[i].astype(np.float64); ix = slice(off[b], off[b + 1])
        k = pool[i].shape[0]
        bhat[ix] = R @ beta[ix] + (chol[i] @ rng.standard_normal(k)) / np.sqrt(N)
    return beta, bhat, off


# The dense fit runs in a subprocess that materialises the full genome (distinct
# per-block copies, so peak RSS reflects a real genome's dense LD footprint, not
# the small tiled pool), and also reports auto's phenotype-scale R2.
WORKER = r'''
import os, sys, json, time, resource
import numpy as np
sys.path.insert(0, %r)
from ldpred3 import ldpred3_by_blocks
d = np.load(os.path.join(%r, "scaling_in.npz"))
tiling = d["tiling"]; H2 = float(d["h2"]); P = float(d["p"]); N = float(d["n"])
BURN = int(d["burn"]); IT = int(d["it"]); bhat = d["bhat"]; beta_true = d["beta"]
pool = [d["R%%d" %% i] for i in range(int(d["pool"]))]
blocks = []; off = 0
for b in tiling:
    R = pool[int(b)].copy(); k = R.shape[0]          # distinct copy -> real footprint
    blocks.append((R, np.arange(off, off + k))); off += k
m = off; n = np.full(m, N)
def fit(mth):
    if mth == "inf":  return ldpred3_by_blocks(blocks, bhat, n, method="inf", h2=H2)
    if mth == "grid": return ldpred3_by_blocks(blocks, bhat, n, method="grid", h2=H2, p=P, burn_in=BURN, num_iter=IT)
    if mth == "auto": return ldpred3_by_blocks(blocks, bhat, n, method="auto", burn_in=BURN, num_iter=IT, seed=1, h2_init=H2, p_init=P)
fit("auto")   # warm JIT (not timed)
out = {}; betas = {}
for mth in ("inf", "grid", "auto"):
    t0 = time.perf_counter(); be = fit(mth); out[mth] = time.perf_counter()-t0; betas[mth] = np.asarray(be)
num = d1 = d2 = 0.0; ba = betas["auto"]
for (R, ix) in blocks:
    Rd = R.astype(np.float64); Rb = Rd @ beta_true[ix]
    num += ba[ix] @ Rb; d1 += ba[ix] @ (Rd @ ba[ix]); d2 += beta_true[ix] @ Rb
gr2 = (num*num)/(d1*d2) if d1 > 0 and d2 > 0 else 0.0
_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mem = _rss/1e9 if sys.platform == "darwin" else _rss/1e6   # macOS bytes / Linux KB -> GB
print("RESULT " + json.dumps({"time": out, "mem_gb": mem, "r2": gr2*H2, "m": m}))
'''


def run_dense_fit(pool, tiling, beta, bhat):
    os.makedirs(WORK, exist_ok=True)
    arrs = {"R%d" % i: R for i, R in enumerate(pool)}
    np.savez(os.path.join(WORK, "scaling_in.npz"), tiling=tiling, pool=len(pool),
             h2=H2, p=P, n=float(N), burn=BURN_IN, it=NUM_ITER,
             bhat=bhat, beta=beta, **arrs)
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    code = WORKER % (ROOT, WORK)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr); raise RuntimeError("ldpred3 worker failed")
    return json.loads([l for l in r.stdout.splitlines() if l.startswith("RESULT ")][-1][7:])


rows = []
for nsnps in SIZES:
    nb_t, mean_k = block_layout(nsnps)
    pool = make_pool(mean_k)
    tiling, m = tile_genome(pool, nsnps)
    nb = len(tiling)
    dense_gb, band_gb, low_gb, stream_gb = rep_memory_gb(pool, tiling)
    rng = np.random.default_rng(42)
    beta, bhat, off = build_beta_bhat(pool, tiling, rng)
    res = run_dense_fit(pool, tiling, beta, bhat)
    del pool
    t = res["time"]
    rows.append([m, nb, round(m / nb), round(dense_gb, 3), round(band_gb, 3),
                 round(low_gb, 3), round(stream_gb, 4), round(res["mem_gb"], 3),
                 round(t["inf"], 2), round(t["grid"], 2), round(t["auto"], 2),
                 round(res["r2"], 4)])
    print(f"{m:>9} | {nb:>4} blk mean {m//nb:>4} | dense {dense_gb:.2f} band {band_gb:.2f} "
          f"lowrank {low_gb:.2f} stream {stream_gb:.3f} GB | RSS {res['mem_gb']:.2f} | "
          f"inf {t['inf']:.1f} grid {t['grid']:.1f} auto {t['auto']:.1f} | R2 {res['r2']:.4f}",
          flush=True)
    with open(os.path.join(HERE, "ldpred3_scaling.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nsnps", "n_blocks", "mean_block", "dense_gb", "banded_gb",
                    "lowrank_gb", "stream_gb", "peak_rss_gb",
                    "inf_s", "grid_s", "auto_s", "auto_r2"])
        w.writerows(rows)

# ---- figure ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = np.array(rows, float)
mm = a[:, 0] / 1e6          # #SNPs in millions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for j, name, c in ((3, "dense", "C3"), (4, "banded", "C1"),
                   (5, "low-rank", "C0"), (6, "streaming (1 block)", "C2")):
    ax1.plot(mm, a[:, j], "o-", color=c, label=name)
ax1.axhline(RAM_GB, ls=":", color="k", label=f"{RAM_GB:.0f} GB RAM")
ax1.set_xlabel("#SNPs (millions)"); ax1.set_ylabel("resident LD memory (GB)")
ax1.set_title("LD memory vs SNP density\n(dense/banded ~quadratic, low-rank sub-quadratic)")
ax1.legend(); ax1.grid(alpha=.3)

for j, name in ((8, "inf"), (9, "grid"), (10, "auto")):
    ax2.plot(mm, a[:, j], "o-", label=name)
ax2.set_xlabel("#SNPs (millions)"); ax2.set_ylabel("dense fit time (s), single core")
ax2.set_title("Fit time vs SNP density\n(per-block O(k^2)-O(k^3))")
ax2.legend(); ax2.grid(alpha=.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "ldpred3_scaling.png"), dpi=130)
print("wrote ldpred3_scaling.csv and ldpred3_scaling.png")
