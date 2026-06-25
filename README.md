# iprs
iPSYCH PRS

## pyLDpred2

**pyLDpred2** (`src/ldpred2.py`) is a small, dependency-light (NumPy only, optional
Numba) Python implementation of the core
[LDpred2](https://doi.org/10.1093/bioinformatics/btaa1029) polygenic-score
models. It re-weights GWAS marginal effect sizes using an LD
(linkage-disequilibrium) correlation matrix.

### Summary

pyLDpred2 reproduces LDpred2-inf/-grid/-auto in pure Python and matches the
reference R implementation (`bigsnpr`) on prediction accuracy, while being much
more memory-efficient. It adds Numba JIT acceleration, a running-residual /
float32 / fused Gibbs sampler, Rao-Blackwellized estimates, warm-start and
adaptive stopping, a sparse/banded LD backend with an iterative `inf` solver,
optimal LD-block splitting (Privé 2022), and global hyper-parameters for `-auto`
via a **streaming** sampler that never materialises a genome-wide LD matrix.
Across 200k–2M SNPs (single core) it matches bigsnpr's accuracy exactly while
using **~2× less memory**, and is competitive on speed in a method-dependent way:
`-auto` is ~1.1–1.4× faster, `-inf` is roughly on par, and `-grid` is ~2× slower
than bigsnpr's compiled C++ sampler. Both scale to 2M SNPs single-threaded
(pyLDpred2 in ~4 GB, bigsnpr in ~8 GB). An optional multicore path for global
`-auto` (`ldpred2_by_blocks(..., ncores=k)`) parallelises the per-sweep block
loop with `numba.prange`, but the benchmarks below compare **single-core**
throughout. See the head-to-head against `bigsnpr` below.

### Benchmark vs bigsnpr (realistic LD, 200k–2M SNPs, single core)

The benchmark uses **realistic LD** — each block is a `k`-SNP correlation matrix
from a coalescent-with-recombination simulation (msprime: haplotype plateaus,
recombination valleys, a heavy decay tail and perfect-LD duplicates), not
idealized AR(1). Every method runs on a **single core** for both tools (NumPy
BLAS and R BLAS pinned to one thread); bigsnpr's on-disk SFBM is assembled
**incrementally** block-by-block (`as_SFBM` + `$add_columns()`, as in the
LDpred2 vignette) so the full correlation never sits in RAM.

![1-core method comparison vs bigsnpr](benchmarks/cores_1core_benchmark.png)

Wall-clock time (s), single core:

| #SNPs | inf py / big | grid py / big | auto py / big |
|-------|-------------:|--------------:|--------------:|
| 200k  | **3.1** / 5.0 | 3.4 / **1.5** | **1.8** / 2.5 |
| 500k  | **5.2** / 8.9 | 8.1 / **3.5** | **4.2** / 6.2 |
| 1M    | **10.4** / 13.3 | 16.0 / **6.9** | **9.1** / 12.4 |
| 2M    | 20.7 / **18.2** | 32.0 / **13.9** | **21.7** / 25.0 |

Peak memory (GB) — LD-dominated, so ~equal across the three methods:

| #SNPs | pyLDpred2 | bigsnpr |
|-------|----------:|--------:|
| 200k  | **0.73** | 1.06 |
| 500k  | **1.33** | 2.24 |
| 1M    | **2.31** | 4.24 |
| 2M    | **4.28** | 8.24 |

**Prediction accuracy is identical** between the two at every size and method
(e.g. auto R²_pheno 0.493/0.492 at 200k → 0.421/0.421 at 2M; h²=0.5).

The picture is method-dependent — there is no blanket "N× faster":

- **Memory:** pyLDpred2 is **~2× leaner** everywhere (`float32` LD + one block
  resident; bigsnpr's SFBM stores `float64` values plus per-entry indices).
- **`-auto`:** pyLDpred2 is **~1.1–1.4× faster** — its streaming global-hyper
  sampler is the strongest path.
- **`-inf`:** roughly on par — pyLDpred2 faster up to 1M, bigsnpr slightly faster
  at 2M.
- **`-grid`:** **bigsnpr is ~2× faster** here; its compiled C++ grid sampler
  beats pyLDpred2's per-block Python-orchestrated one. This is pyLDpred2's weak
  spot at fixed hyper-parameters.

Regenerate with `benchmarks/plot_methods_1core.py` (data in
`benchmarks/cores_1core_benchmark.csv`, R side in
`benchmarks/bench_bigsnpr_blocks.R`).

### End-to-end PRS pipeline (real data)

`src/pipeline.py` runs the whole workflow from GWAS summary statistics and
genotype files to one polygenic score per individual — no R, NumPy-only:

```
GWAS sumstats + genotypes (PLINK/BGEN)
  → QC sumstats (N / MAF / INFO / duplicates / chi-sq outliers)
  → read & harmonise (align effect alleles to A1, drop ambiguous/mismatched)
  → SD-consistency QC vs the reference panel
  → per-block LD from a reference panel (in-sample or external)
  → ldpred2 (inf / grid / auto)
  → per-individual PRS
```

From the command line:

```bash
python -m pipeline --sumstats gwas.txt.gz --plink target --method auto --out prs.txt
python -m pipeline --sumstats gwas.txt.gz --bgen  target.bgen --out prs.txt
```

or from Python:

```python
from pipeline import run_ldpred2_prs
res = run_ldpred2_prs("gwas.txt.gz", "target", method="auto")
res.scores          # per-individual PRS
res.harmonize_log   # QC: matched / flipped / ambiguous / mismatched counts
```

Supporting modules, each usable on its own:

| Module          | What it does                                                           |
|-----------------|-----------------------------------------------------------------------|
| `genotype_io`   | Read/write PLINK 1 `.bed/.bim/.fam` (2-bit decode, NumPy-only)         |
| `bgen_io`       | Read BGEN v1.2/layout-2 (uncompressed or zlib; biallelic diploid)      |
| `sumstats`      | Parse GWAS files with flexible column aliases (OR→β, SE-from-p)        |
| `qc`            | Sumstats QC: N / MAF / INFO / duplicate / chi-sq + SD-consistency      |
| `harmonize`     | Match variants + align effect alleles (swap-flip, strand, palindrome) |
| `ld`            | Per-block LD correlation matrices from a genotype panel               |
| `prs`           | Weighted polygenic scores with missing-call imputation                |

**Sumstats QC** runs by default (`qc=True`, disable with `--no-qc`). Two stages,
following the bigsnpr / LDpred2 tutorial:

* **Sumstats-only** (`qc.qc_sumstats`, before harmonisation): drop non-finite or
  non-positive-SE rows, duplicated variants, low-`N` variants (`N < 0.7·max N`),
  low-MAF (`< 0.01`, when an EAF column is present), low-INFO (`< 0.7`, when
  present) and chi-square outliers (optional `max_chisq`).
* **SD-consistency** (`qc.sd_consistency_mask`, after harmonisation): compare the
  SD implied by the sumstats, `sd_ss ≈ 1/√(N·se² + β²)`, against the reference
  genotype SD `sd_ref = √(2·f·(1−f))`, and drop variants where the ratio leaves
  `[0.5, 2]`. This catches a wrong `N`, allele errors or bad imputation that
  harmonisation cannot. `PRSResult.qc_log` reports the per-filter counts.

**Format notes.** Dosages count the A1 (first) allele; missing calls are `-1`
(PLINK, hard calls) or `NaN` (BGEN, dosages in `[0,2]`). Strand-ambiguous
(A/T, C/G) and allele-mismatched variants are dropped during harmonisation.
BGEN with zstd compression needs the optional `zstandard` package (a clear
error is raised otherwise).

### Models implemented

| Function          | Model                                              | Hyper-parameters     |
|-------------------|----------------------------------------------------|----------------------|
| `ldpred2_inf`     | Infinitesimal (all variants causal, closed form)   | `h2`                 |
| `ldpred2_grid`    | Point-normal / spike-and-slab (Gibbs sampler)      | `h2`, `p` (fixed)    |
| `ldpred2_auto`    | Point-normal, estimates `h2` and `p` automatically | none (self-tuning)   |

Helpers: `standardize_betas` (put GWAS effects on the correlation scale),
`ldpred2_by_blocks` (run a model per LD block, genome-wide),
`block_diagonal_ld` (assemble blocks into one matrix) and `optimal_ld_blocks`
(choose LD block boundaries, below).

### Optimal LD block splitting

Fixed-size LD blocks cut arbitrarily through high-LD regions. `optimal_ld_blocks`
implements [Privé (2022), *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btab519)
(`snp_ldsplit`): a dynamic program that places boundaries to **minimise the
squared LD discarded between blocks**, subject to a maximum block size — so cuts
land in low-LD valleys (recombination hotspots).

```python
blocks, discarded_ld2 = optimal_ld_blocks(corr, max_size=1000, window=300)
```

On a simulated chromosome with recombination hotspots (within-300 LD window):

| scheme | max block | LD retained |
|--------|-----------|-------------|
| fixed 500 | 500 | 87.7 % |
| optimal (max 500) | 499 | **94.4 %** |
| fixed 1000 | 1000 | 94.6 % |
| optimal (max 1000) | 900 | **97.7 %** |

Optimal blocks discard less than half the LD of fixed blocks of the same size —
and `optimal(max 500)` retains as much LD as `fixed(1000)`, i.e. the same fidelity
at half the block size (≈4× less per-block O(k²) work and memory). The downstream
prediction gain is modest; the win is mainly computational/validity, as in the
paper.

### Global hyper-parameters for `-auto`

`ldpred2_by_blocks(method="auto")` estimates `h2` and `p` **globally** by default
(`global_hyper=True`): the sampler **streams the LD blocks one at a time** (a
jitted per-block sweep), pooling the causal count and genetic variance across all
variants each iteration (as in bigsnpr). Estimating them per block instead
(`global_hyper=False`) is noisy when blocks hold few causal variants and **loses
accuracy at genome scale**. Streaming keeps the fast dense contiguous update,
has a constant-N fast path, and — critically — never materialises a packed
genome-wide LD matrix, so peak memory is just the LD plus O(m) state. At **2M
SNPs** this brought global auto from 137 s / 11.4 GB (packed) down to
**21 s / 0.34 GB** (~6× faster, ~34× less memory). On a 1M-SNP simulation
(block-diagonal LD, p=0.001, vs R `bigsnpr`):

| `-auto` variant | predictive R² | bigsnpr |
|-----------------|---------------|---------|
| global (default) | **0.941** | 0.942 |
| per block | 0.848 | 0.942 |

`inf` and `grid` already match bigsnpr (0.080/0.080 and 0.942/0.942 in the same
run); global hyper-parameters bring `-auto` to parity too.

### Sparse / banded LD

Real LD is banded — most off-diagonal entries are ~0 — so the LD can be stored
sparse (CSR) and the sampler/solver need only touch non-zero neighbours
(O(bandwidth) instead of O(block_size)). Build a `SparseLD` with `sparsify_ld`
and pass it to any model (or `ldpred2_by_blocks(..., sparsify=True)`):

```python
from ldpred2 import sparsify_ld, ldpred2_inf, ldpred2_auto
ld = sparsify_ld(corr, threshold=1e-3)        # drop |r| < 1e-3 (and/or max_dist=…)
beta = ldpred2_inf(ld, beta_hat, n_eff, h2)   # sparse CG solve; samplers also accept ld
```

On a clean population AR(1) block (m=4000, 0.47 % density) this gives, with
**identical results** (r = 1.000):

| method | dense | sparse | speedup |
|--------|-------|--------|---------|
| inf  | 2.77 s | 0.006 s | **444×** (CG vs dense O(m³) solve) |
| grid | 0.131 s | 0.070 s | 1.9× |
| auto | 0.138 s | 0.071 s | 1.9× |

Two important caveats:

* **In-sample LD has a noise floor (~1/√N)** that fills the matrix, so magnitude
  thresholding alone won't sparsify it — band by distance (`max_dist=`) to drop
  the spurious far-apart entries, as LDpred2/bigsnpr do.
* **Hard banding can break positive-definiteness**, which destabilises the
  *fixed-h² sampler* (`grid` can diverge; `auto` self-limits via its h² clamp;
  `inf`'s ridge is unaffected). Use `sparsify_ld(..., shrink=<1)` to restore
  diagonal dominance, or supply an already-valid windowed LD matrix.

### Fewer iterations: warm start & adaptive stopping

`ldpred2_grid`/`ldpred2_auto` accept:

* `warm_start=True` — initialise the chain from the LDpred2-inf solution instead
  of zeros, shortening burn-in. It pays for one `inf` solve up front, so it only
  helps when burn-in/mixing dominates **and** inf is cheap — i.e. paired with the
  sparse LD backend (CG inf). With a *dense* O(m³) inf solve it can cost more
  than it saves.
* `tol=<x>` (+ `check_every`) — **adaptive stopping**: end sampling once the
  running posterior mean's relative RMS change over `check_every` sweeps drops
  below `tol`, instead of always running `num_iter`. `AutoResult.n_iter` reports
  how many sweeps were used. On a fast-mixing block this reached the same
  accuracy as a fixed 2000-iteration run in ~100 iterations (~10× faster), with
  no loss (corr 1.000 vs the long run).

### Performance (optional Numba acceleration)

The Gibbs sampler maintains a running `R @ beta` vector (per-SNP residual is an
O(1) lookup; the O(m) rank-1 update is only paid when an effect changes), so it
scales sub-quadratically in block size for sparse traits. The rank-1 update is
the bandwidth-bound hot path, so it runs as a fused element loop over a
**single-precision** LD row (float32 halves the memory traffic, ~2× faster, with
no meaningful accuracy cost — and matches bigsnpr, which also stores LD in single
precision). The effects and the `R @ beta` accumulator stay in float64.

The posterior-mean estimate is **Rao-Blackwellized** (as in the original
LDpred): each sweep accumulates the conditional expectation
`E[beta_j | rest] = P(causal) · posterior_mean` rather than the sampled draw.
The sampled value still drives the Markov chain; only the *estimate* uses the
expectation, which has much lower Monte-Carlo variance (≈6–13× in fast-mixing
regimes), so fewer iterations are needed for the same accuracy. In extreme-LD
regions the benefit is smaller because chain mixing, not sampling noise, is the
bottleneck.

[Numba](https://numba.pydata.org/) is strongly recommended: when installed, the
inner sampler is JIT-compiled (and cached) for a large speed-up. Without it the
code still runs and gives identical results, but the sampler falls back to plain
Python loops and is much slower — fine for small problems / CI, not for genome-
wide runs.

```bash
pip install numba      # optional but strongly recommended
```

On a single core (see the benchmark table above for the full head-to-head),
the JIT-compiled samplers produce effects identical to bigsnpr's C++
`snp_ldpred2_{grid,auto}` (matching accuracy at every size). On speed the result
is method-dependent: `-auto` is ~1.1–1.4× faster than bigsnpr and `-inf` is on
par, but `-grid` is ~2× slower — bigsnpr's compiled fixed-hyper-parameter grid
sampler is hard to beat from Python. pyLDpred2's edge is memory (~2× leaner) and
the streaming global-`auto` path.

### Conventions

Effects are on the standardized scale, where the marginal effects relate to the
true joint effects through the LD matrix `R`:

```
beta_hat = R @ beta + noise,   noise ~ N(0, R / N)
```

with `N` the GWAS sample size. Use `standardize_betas(beta, beta_se, n_eff)` to
convert reported GWAS effects to this scale and to recover the back-transform.

### Quick example

```python
import numpy as np
from ldpred2 import standardize_betas, ldpred2_auto

# beta, beta_se, n_eff come from GWAS summary statistics for one LD block;
# corr is the (m x m) LD correlation matrix for those variants.
beta_hat, scale = standardize_betas(beta, beta_se, n_eff)

res = ldpred2_auto(corr, beta_hat, n_eff)
adjusted_beta = res.beta_est * scale          # back to the input scale
print(res.h2_est, res.p_est)                  # estimated heritability & causal frac
```

### Tests / demo

```bash
python tests/test_ldpred2.py     # prints recovery of true effects on simulated data
python -m pytest tests/          # run the assertions
```

On the bundled synthetic LD block, correlation with the true effects improves
from ~0.67 (raw marginal betas) to ~0.98 (inf) and ~0.99 (grid / auto).

### Genotype-level benchmark

`src/simulate.py` is a full end-to-end simulation: it generates genotypes with
block LD, simulates a phenotype under a chosen heritability and polygenicity,
runs a marginal GWAS, estimates the LD matrix from the training sample, fits
LDpred2, and reports **out-of-sample** prediction R² on a held-out test set. It
sweeps a grid of polygenicity × heritability × sample size.

To stay within memory at scale, genotypes are stored as `int8` dosages and
every step (standardization, GWAS, LD, PRS) is processed one LD block at a time,
so a full float genotype matrix is never materialised.

**LD model (`--ld-model`).** Two choices for the LD between SNPs:

* `ar1` (default): a latent-Gaussian model with geometric LD decay
  (`r ≈ ρ^dist`). Fast and dependency-free, but idealized — LD collapses to ~0
  within a handful of SNPs.
* `coalescent`: realistic LD from a coalescent-with-recombination simulation
  (via [msprime](https://tskit.dev/msprime), human-like Ne=10⁴ and 1e-8 recomb/
  mutation rates). This produces actual haplotype blocks, recombination
  hotspots, a heavy LD decay tail and sporadic long-range LD — the structure of
  real reference panels (mean r² stays ~0.02 at 200 SNPs apart, vs ~0 for AR(1)).

LDpred2's advantage over the raw marginal PRS is *larger* under realistic LD
(e.g. h²=0.5, p=0.01: marginal 0.21 → grid/auto 0.43 with coalescent LD, vs
0.32 → 0.50 with AR(1)), because realistic long-range LD inflates the naive
score that LDpred2's LD-adjustment removes.

```bash
python src/simulate.py --quick                      # fast (AR(1))
python src/simulate.py --quick --ld-model coalescent  # realistic LD (needs msprime)
python src/simulate.py --csv sim.csv      # full accuracy grid, save results
```

Representative results (m=10000 SNPs, blocks of 200, AR(1) LD; prediction R² vs
phenotype, from `python src/simulate.py --csv …`):

| N | h² | p (causal) | marginal | inf | grid | auto | ceiling |
|---|----|-----------|---------|-----|------|------|---------|
| 5000  | 0.5 | 0.001 | 0.097 | 0.100 | 0.465 | 0.465 | 0.475 |
| 20000 | 0.5 | 0.001 | 0.254 | 0.262 | 0.489 | 0.489 | 0.489 |
| 20000 | 0.5 | 0.1   | 0.245 | 0.265 | 0.417 | 0.417 | 0.512 |
| 20000 | 0.3 | 0.01  | 0.135 | 0.139 | 0.301 | 0.300 | 0.311 |

Takeaways: LDpred2 always beats the raw marginal baseline; accuracy rises with
heritability and sample size; `grid`/`auto` approach the ceiling for sparse
architectures and remain the best across the grid. The infinitesimal model only
modestly beats the marginal score here — its all-causal prior leaves accuracy on
the table whenever the trait is even mildly sparse.

### Scaling: what the algorithm actually depends on

The LDpred2 *algorithm* works from summary statistics + the LD matrix, so its
cost is **independent of the GWAS sample size N** and is driven instead by the
**LD structure (block size)**. The benchmarks below separate the algorithm's
`fit` time from the simulation/GWAS/LD-construction `prep` time (which does scale
with N). All measured on a 4-core / 15 GB box, Numba on, h²=0.5, p=0.01.

**Independent of N** (`--n-independence`, m=10000, blocks of 200): the fit time
is flat while prep grows with N.

| N_train | prep (s) | fit_grid (s) | fit_auto (s) |
|---------|---------|--------------|--------------|
| 2000   | 4.1  | 0.200 | 0.367 |
| 8000   | 10.4 | 0.199 | 0.305 |
| 32000  | 46.9 | 0.195 | 0.270 |

**Driven by LD block size** (`--ld-scaling`, m=20000 fixed, N=8000): larger LD
blocks make each block's solve/sampler costlier. The infinitesimal model is a
dense linear solve per block (≈O(m·k²), grows fast), whereas the Gibbs samplers
stay nearly flat for sparse traits thanks to the running-residual update.

| block size | #blocks | fit_inf (s) | fit_grid (s) | fit_auto (s) |
|-----------|---------|-------------|--------------|--------------|
| 100   | 200 | 0.076 | 0.379 | 0.664 |
| 250   | 80  | 0.105 | 0.402 | 0.680 |
| 500   | 40  | 0.167 | 0.398 | 0.731 |
| 1000  | 20  | 0.347 | 0.410 | 0.468 |
| 2000  | 10  | 1.082 | 0.469 | 0.541 |

**Scaling #SNPs** (`--scaling`, N=8000, blocks of 200): with N fixed, total
runtime and memory grow ~linearly in #SNPs (≈1 ms/SNP; memory bounded by the
`int8` genotype matrix). Accuracy falls only because more SNPs/causal variants
dilute the fixed GWAS power — `grid` degrades gracefully while raw
`marginal`/`inf` collapse.

| #SNPs | prep (s) | fit (s) | peak mem (GB) | marginal | inf | grid | auto | ceiling |
|-------|---------|--------|---------------|---------|-----|------|------|---------|
| 10000  | ~10 | ~0.7 | 0.30 | 0.167 | 0.174 | 0.465 | 0.452 | 0.503 |
| 50000  | ~46 | ~3.5 | 0.74 | 0.051 | 0.050 | 0.316 | 0.264 | 0.485 |
| 100000 | ~98 | ~7   | 1.28 | 0.016 | 0.015 | 0.181 | 0.115 | 0.482 |

Practical takeaway: for dense data with long-range / large LD blocks, the
dense per-block LD storage and the infinitesimal solve become the bottleneck,
which motivates a banded / sparse-LD backend.
