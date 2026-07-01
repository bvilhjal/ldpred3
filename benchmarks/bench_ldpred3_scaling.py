"""LDpred3 scaling with SNP *density*: LD memory, time and accuracy.

Four modelling choices make this a realistic genome-scale picture rather than a
tidy extrapolation:

1. **Block count is set by recombination, not density.** The human genome has
   ~1,700 approximately-independent, recombination-delimited LD blocks (Berisa &
   Pickrell 2016), and that count barely moves as you densify: 1M -> 10M SNPs
   adds only ~1.2x more blocks. So densifying makes each block *bigger*, it does
   not add proportionally more same-size blocks.

2. **The density lever is the mutation rate, on fixed chromosomes.** Each block
   is a *fixed* coalescent-with-recombination segment (msprime) -- a physical
   length and recombination rate drawn once and held across densities, varied
   across the pool (log-normal lengths, 0.5x-2x recombination) -- densified by
   raising the **mutation rate** (the reusable
   ``simulate_genotypes_by_mutation_rate`` primitive; a fixed seed keeps the
   genealogy identical, so it is the same chromosome with more variants).

3. **Memory, time *and* accuracy, per representation.** Memory alone is
   misleading -- a compact representation trades fit *time* and possibly
   *accuracy* for it. So for each density we report all three for **dense**,
   **low-rank**, and on-disk **streaming**. Streaming's robust, machine-
   independent wins are memory (one block resident) and accuracy (exact -- it is
   the same LD, just relocated to disk); its **time is highly hardware-dependent**
   (storage speed, RAM / page-cache size, filesystem). What we time here is the
   page-cached, compute-bound case (only while the cache fits in RAM) -- a lower
   bound; past RAM it is disk-I/O-bound and its wall-clock varies by orders of
   magnitude across setups, so treat the streaming *time* as illustrative, not a
   portable number. (Banded / windowed LD is omitted: within recombination-
   limited blocks any window narrow enough to save memory drops the within-block
   LD the adjustment needs, so it loses accuracy without a memory win.) R2 is
   always scored on the true (dense) population LD, so a representation's loss of
   accuracy shows up honestly.

4. ``auto`` is the common method (per-method inf/grid/auto timing is in
   ``method_scaling``); per-block Gibbs work is O(k^2), so time is super-linear.

Findings: dense LD grows ~quadratically and hits a fixed RAM ceiling early; the
block's effective rank is recombination-bounded, so **low-rank** grows
sub-quadratically and is essentially **lossless**, but its eigenspace fit
recomputes ``(R beta)_j`` per SNP and so costs several times the dense fit time --
the price of scaling past the dense RAM wall. Streaming keeps one block resident.

Writes ``ldpred3_scaling.csv`` and a three-panel figure ``ldpred3_scaling.png``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_ldpred3_scaling.py
    python benchmarks/bench_ldpred3_scaling.py 1000000 2000000    # specific sizes
"""
import sys, os, csv, json, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3.ld import lowrank_ld
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
LOWRANK_VAR = 0.995          # low-rank retained variance
RAM_GB = 15.0                # this machine, for the ceiling line
STREAM_FIT_MAX_GB = 4.0      # only time the on-disk stream fit while it fits in RAM

# One fixed set of "chromosomes": (length bp, recombination rate, ancestry seed).
_g = np.random.default_rng(20240701)
POOL_GEOM = [(float(L_MEAN * np.exp(_g.normal(0, L_SIGMA))),
              float(1e-8 * np.exp(_g.uniform(np.log(REC_LO), np.log(REC_HI)))),
              int(_g.integers(1, 2**31 - 1))) for _ in range(POOL)]
L_MEAN_ACTUAL = float(np.mean([L for L, _, _ in POOL_GEOM]))

DEFAULT_SIZES = [200_000, 500_000, 1_000_000, 2_000_000]   # dense LD caps ~2.4M/15 GB


def block_layout(nsnps):
    """(target n_blocks, mean block size) under the recombination-block model."""
    nb = int(round(NB_1M * (nsnps / 1e6) ** ALPHA))
    return nb, nsnps / nb


def mut_rate_for(mean_k):
    """Mutation rate that yields ~mean_k common SNPs in a mean-length segment."""
    return 1e-8 * mean_k / (SNPS_PER_MB_AT_BASE * L_MEAN_ACTUAL / 1e6)


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
    dense = low = 0
    kmax = 0
    for i, R in enumerate(pool):
        k = R.shape[0]; c = int(counts[i]); kmax = max(kmax, k)
        dense += c * R.nbytes
        low += c * lowrank_ld(R, variance=LOWRANK_VAR).U.nbytes
    stream = kmax * kmax * 4          # one (largest) block resident
    return (dense / 1e9, low / 1e9, stream / 1e9)


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


# The fit runs in a subprocess (clean numba). It rebuilds the pool as dense and
# as low-rank, cycles each to the full genome, and times an ``auto`` fit plus its
# prediction R2 -- so the recurring fit cost and accuracy sit next to the memory
# each representation would occupy.
WORKER = r'''
import os, sys, json, time, resource
import numpy as np
sys.path.insert(0, %r)
from ldpred3 import ldpred3_by_blocks
from ldpred3.ld import lowrank_ld, save_ld_blocks, load_ld_blocks
WORK = %r
d = np.load(os.path.join(WORK, "scaling_in.npz"))
tiling = d["tiling"]; H2 = float(d["h2"]); P = float(d["p"]); N = float(d["n"])
BURN = int(d["burn"]); IT = int(d["it"]); LVAR = float(d["lvar"]); DO_STREAM = bool(d["do_stream"])
bhat = d["bhat"]; beta_true = d["beta"]
dense_pool = [d["R%%d" %% i] for i in range(int(d["pool"]))]
low_pool = [lowrank_ld(R, variance=LVAR) for R in dense_pool]
off = np.concatenate([[0], np.cumsum([dense_pool[int(i)].shape[0] for i in tiling])])
m = int(off[-1]); n = np.full(m, N)
def genome(pool):
    return [(pool[int(tiling[b])], np.arange(off[b], off[b+1])) for b in range(len(tiling))]
def fit(pool):
    bl = genome(pool)
    ldpred3_by_blocks(bl, bhat, n, method="auto", burn_in=3, num_iter=3, seed=1, h2_init=H2, p_init=P)
    t0 = time.perf_counter()
    be = ldpred3_by_blocks(bl, bhat, n, method="auto", burn_in=BURN, num_iter=IT, seed=1, h2_init=H2, p_init=P)
    return time.perf_counter() - t0, np.asarray(be)
times = {}; betas = {}
times["dense"], betas["dense"] = fit(dense_pool)
times["lowrank"], betas["lowrank"] = fit(low_pool)
if DO_STREAM:
    # write the whole genome to an on-disk mmap cache, then fit reading one
    # block at a time (resident memory ~ O(one block)); this is the real disk
    # path, so its time includes I/O (page-cached below RAM, disk-bound above).
    cache = os.path.join(WORK, "stream_cache.npz")
    save_ld_blocks(cache, genome(dense_pool), [str(i) for i in range(m)], mmap=True)
    bl, _ = load_ld_blocks(cache)
    ldpred3_by_blocks(bl, bhat, n, method="auto", burn_in=3, num_iter=3, seed=1, h2_init=H2, p_init=P)
    t0 = time.perf_counter()
    be = ldpred3_by_blocks(bl, bhat, n, method="auto", burn_in=BURN, num_iter=IT, seed=1, h2_init=H2, p_init=P)
    times["stream"] = time.perf_counter() - t0; betas["stream"] = np.asarray(be)
# auto phenotype-scale R2 per representation, scored on the true (dense) LD
Rdense = [R.astype(np.float64) for R in dense_pool]
def pheno_r2(be):
    num = d1 = d2 = 0.0
    for b in range(len(tiling)):
        R = Rdense[int(tiling[b])]; ix = slice(off[b], off[b+1])
        Rb = R @ beta_true[ix]
        num += be[ix] @ Rb; d1 += be[ix] @ (R @ be[ix]); d2 += beta_true[ix] @ Rb
    return (num*num)/(d1*d2)*H2 if d1 > 0 and d2 > 0 else 0.0
r2 = {k: pheno_r2(v) for k, v in betas.items()}
_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mem = _rss/1e9 if sys.platform == "darwin" else _rss/1e6
print("RESULT " + json.dumps({"time": times, "mem_gb": mem, "r2": r2, "m": m}))
'''


def run_fits(pool, tiling, beta, bhat, do_stream):
    os.makedirs(WORK, exist_ok=True)
    arrs = {"R%d" % i: R for i, R in enumerate(pool)}
    np.savez(os.path.join(WORK, "scaling_in.npz"), tiling=tiling, pool=len(pool),
             h2=H2, p=P, n=float(N), burn=BURN_IN, it=NUM_ITER,
             lvar=LOWRANK_VAR, do_stream=do_stream, bhat=bhat, beta=beta, **arrs)
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    code = WORKER % (ROOT, WORK)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr); raise RuntimeError("ldpred3 worker failed")
    return json.loads([l for l in r.stdout.splitlines() if l.startswith("RESULT ")][-1][7:])


def main(sizes):
    rows = []
    for nsnps in sizes:
        nb_t, mean_k = block_layout(nsnps)
        pool = make_pool(mean_k)
        tiling, m = tile_genome(pool, nsnps)
        nb = len(tiling)
        dense_gb, low_gb, stream_gb = rep_memory_gb(pool, tiling)
        rng = np.random.default_rng(42)
        beta, bhat, off = build_beta_bhat(pool, tiling, rng)
        do_stream = dense_gb <= STREAM_FIT_MAX_GB      # skip the disk fit past RAM
        res = run_fits(pool, tiling, beta, bhat, do_stream)
        del pool
        t = res["time"]; r2 = res["r2"]
        stream_s = round(t["stream"], 2) if "stream" in t else float("nan")
        stream_r2 = round(r2["stream"], 4) if "stream" in r2 else float("nan")
        rows.append([m, nb, round(m / nb), round(dense_gb, 3), round(low_gb, 3),
                     round(stream_gb, 4), round(t["dense"], 2), round(t["lowrank"], 2),
                     stream_s, round(r2["dense"], 4), round(r2["lowrank"], 4), stream_r2])
        ss = f"{stream_s:.1f}" if stream_s == stream_s else "n/a"   # nan-safe
        print(f"{m:>9} | {nb:>4} blk mean {m//nb:>4} | mem dense {dense_gb:.2f} lowrank "
              f"{low_gb:.2f} stream {stream_gb:.3f} GB | auto dense {t['dense']:.1f} lowrank "
              f"{t['lowrank']:.1f} stream {ss} s | R2 dense {r2['dense']:.3f} lowrank "
              f"{r2['lowrank']:.3f}", flush=True)
        with open(os.path.join(HERE, "ldpred3_scaling.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["nsnps", "n_blocks", "mean_block", "dense_gb", "lowrank_gb",
                        "stream_gb", "dense_s", "lowrank_s", "stream_s",
                        "dense_r2", "lowrank_r2", "stream_r2"])
            w.writerows(rows)
    make_figure(rows)
    print("wrote ldpred3_scaling.csv and ldpred3_scaling.png")


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.array(rows, float)
    mm = a[:, 0] / 1e6          # #SNPs in millions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))

    for j, name, c in ((3, "dense", "C3"), (4, "low-rank", "C0"),
                       (5, "streaming (1 block)", "C2")):
        ax1.plot(mm, a[:, j], "o-", color=c, label=name)
    ax1.axhline(RAM_GB, ls=":", color="k", label=f"{RAM_GB:.0f} GB RAM")
    ax1.set_xlabel("#SNPs (millions)"); ax1.set_ylabel("resident LD memory (GB)")
    ax1.set_title("LD memory\n(dense ~quadratic, low-rank sub-quadratic)")
    ax1.legend(); ax1.grid(alpha=.3)

    for j, name, c, st in ((6, "dense", "C3", "o-"), (7, "low-rank", "C0", "o-"),
                           (8, "streaming (cached; I/O-bound past RAM)", "C2", "s--")):
        ax2.plot(mm, a[:, j], st, color=c, label=name)
    ax2.set_xlabel("#SNPs (millions)"); ax2.set_ylabel("auto fit time (s), single core")
    ax2.set_title("auto fit time\n(low-rank costs time; streaming time is HW-dependent)")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    for j, name, c, st in ((9, "dense", "C3", "o-"), (10, "low-rank", "C0", "o-"),
                           (11, "streaming", "C2", "s--")):
        ax3.plot(mm, a[:, j], st, color=c, label=name)
    ax3.set_xlabel("#SNPs (millions)"); ax3.set_ylabel("prediction R² (auto)")
    ax3.set_title("prediction accuracy\n(low-rank & streaming ~lossless)")
    ax3.legend(); ax3.grid(alpha=.3)

    fig.suptitle("LDpred3 vs SNP density: memory / time / accuracy by LD representation")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ldpred3_scaling.png"), dpi=130)


if __name__ == "__main__":
    sizes = [int(float(a)) for a in sys.argv[1:]] or DEFAULT_SIZES
    main(sizes)
