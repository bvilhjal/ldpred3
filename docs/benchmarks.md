# Benchmarks

All benchmarks are single-core unless noted. Regenerate the bigsnpr comparison
with `benchmarks/plot_methods_1core.py` (data in
`benchmarks/cores_1core_benchmark.csv`, R side in
`benchmarks/bench_bigsnpr_blocks.R`).

## vs bigsnpr (realistic LD, 200k–2M SNPs, single core)

The benchmark uses **realistic LD** — each block is a `k`-SNP correlation matrix
from a coalescent-with-recombination simulation (msprime: haplotype plateaus,
recombination valleys, a heavy decay tail and perfect-LD duplicates), not
idealized AR(1). Every method runs on a **single core** for both tools (NumPy
BLAS and R BLAS pinned to one thread); bigsnpr's on-disk SFBM is assembled
**incrementally** block-by-block (`as_SFBM` + `$add_columns()`, as in the
LDpred2 vignette) so the full correlation never sits in RAM.

![1-core method comparison vs bigsnpr](../benchmarks/cores_1core_benchmark.png)

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

## End-to-end pipeline vs bigsnpr

Beyond the per-block accuracy check above, the **whole pipeline** was validated
against bigsnpr: the same simulated PLINK target + GWAS sumstats + in-sample LD
were run through pyLDpred2's complete pipeline (QC → harmonise → per-block LD →
`-auto` → scoring) and through bigsnpr's `snp_ldpred2_auto`, and the
per-individual polygenic scores compared.

| metric | result |
|--------|--------|
| PRS correlation (pyLDpred2 vs bigsnpr) | **r = 0.9995** |
| R² vs true genetic value | 0.567 (pyLDpred2) / 0.575 (bigsnpr) |

So the pipeline glue — allele harmonisation, QC, LD construction and scoring —
reproduces bigsnpr's polygenic scores essentially exactly. (Validation against a
downloaded public GWAS + 1000 Genomes reference is the natural next step; it adds
real-data quirks the simulation can't, but needs multi-GB inputs.)

## Methods by genetic architecture (realistic LD)

How do the LDpred2 variants compare across genetic architectures? The genome is
100 distinct coalescent/msprime LD blocks of 500 SNPs (m=50,000, h²=0.5); for
each architecture we simulate true effects, generate summary statistics, fit
every method, and measure the **genetic R²** — the squared correlation between
the PRS and the true genetic value under population LD,
`(β̂ᵀRβ)² / [(β̂ᵀRβ̂)(βᵀRβ)]` — averaged over 5 replicates. `grid` is given the
oracle `(h²,p)`; `annot` gets one functional annotation (informative only in the
last row). Regenerate with `benchmarks/bench_methods.py` /
`benchmarks/plot_methods_arch.py`.

![Methods by architecture](../benchmarks/methods_arch_benchmark.png)

Genetic R² at **N = 10,000** (the lower-power regime separates the methods):

| architecture | marginal | inf | grid | auto | annot |
|--------------|---------:|----:|-----:|-----:|------:|
| infinitesimal       | 0.451 | **0.532** | 0.531 | 0.526 | 0.527 |
| sparse (p=0.01)     | 0.460 | 0.541 | **0.747** | 0.746 | 0.747 |
| polygenic (p=0.2)   | 0.442 | 0.530 | **0.531** | 0.528 | 0.529 |
| major locus         | 0.459 | 0.533 | **0.684** | 0.672 | 0.672 |
| annotation-enriched | 0.457 | 0.536 | 0.646 | 0.644 | **0.662** |

Genetic R² at **N = 50,000** (higher power; everything shifts up and compresses):

| architecture | marginal | inf | grid | auto | annot |
|--------------|---------:|----:|-----:|-----:|------:|
| infinitesimal       | 0.565 | **0.794** | 0.792 | 0.789 | 0.789 |
| sparse (p=0.01)     | 0.575 | 0.797 | 0.941 | 0.942 | **0.942** |
| polygenic (p=0.2)   | 0.559 | 0.791 | **0.797** | 0.797 | 0.796 |
| major locus         | 0.574 | 0.794 | **0.904** | 0.903 | 0.903 |
| annotation-enriched | 0.577 | 0.796 | 0.900 | 0.901 | **0.908** |

Takeaways:

- **The raw marginal PRS is always far behind** — the LD adjustment is the
  first-order win (≈0.45→0.53 at N=10k, ≈0.57→0.79 at N=50k).
- **`inf` is architecture-robust but flat**: it is the best model *only* under a
  truly infinitesimal (or near-infinitesimal polygenic) architecture, and leaves
  large gains on the table whenever the trait is sparse or has major loci.
- **`grid`/`auto` win decisively on sparse and major-locus** architectures
  (e.g. 0.75/0.68 vs 0.53 for `inf` at N=10k) — the spike-and-slab captures
  concentrated signal. **`auto` matches the oracle `grid`** (handed the true
  `h²` and `p`) without any hyper-parameters — the practical default.
- **`annot` matches `auto` when the annotation is uninformative and beats it
  when it carries signal.** On the annotation-enriched architecture it is the
  best method at both power levels (N=10k: 0.662 vs grid 0.646; N=50k: 0.908 vs
  0.901), and it never falls behind `auto` elsewhere. The lift from a *single*
  binary annotation is modest — SBayesRC's larger real-data gains come from many
  S-LDSC-calibrated annotations — but it is consistent and free of the
  "garbage-in" penalty a *fixed* bad prior would carry.

> **Convergence note (why this is the corrected table).** An earlier run with a
> lazy annotation-map update (`theta_every=10`) had `annot` *underperforming*
> `auto` at N=10k — e.g. enriched 0.60 vs 0.64. That was an artifact: with short
> chains the `p_j = sigmoid(Aθ)` map had not converged, so it over-estimated the
> global `p` (effective p ≈ 0.04 vs a true 0.02) and **over-shrank** the effects.
> Updating `θ` every sweep (now the default — the IRLS step is cheap for a
> handful of annotations) lets the map and the effects co-adapt; the learned
> enrichment then reaches its true value (θ_func ≈ 1.7) and the anomaly
> disappears. A diagnostic confirmed the fix is purely about convergence: more
> iterations *without* frequent θ updates also fixed it, but added nothing on top
> of `theta_every=1`.

### Per-method running time

Fit time on the same setup (m=50,000 = 100 coalescent blocks of 500, single
core, burn-in 80 / 200 sampling sweeps; `inf` is a direct per-block solve).
Regenerate with `benchmarks/timing_bench.py`.

| method | fit time (s) |
|--------|-------------:|
| inf    | 0.56 |
| auto   | 2.13 |
| grid   | 2.20 |
| annot  | 3.99 |

`inf` is cheapest (one linear solve per block, no sampling). `grid`/`auto` are
the spike-and-slab Gibbs samplers and cost about the same. `annot` is ~1.9×
`auto`: it runs the same per-block effect sweeps plus a logistic annotation-map
update every sweep.

**Cost of the annotation learner (`annot`).** The θ-update is an `O(m·K²)` IRLS
solve in the number of annotations `K`, run every `theta_every` sweeps. Fit time
(s) at m=50,000:

| #annotations K | `theta_every=1` (default) | `theta_every=10` |
|---------------:|--------------------------:|-----------------:|
| 1   | 4.0  | 2.6 |
| 5   | 4.5  | 2.6 |
| 20  | 6.4  | 2.8 |
| 50  | 10.5 | 3.1 |
| 100 | 22.9 | 4.4 |

So the convergence-correct default (`theta_every=1`) is nearly free for a handful
of annotations but its `O(K²)` per-sweep cost takes over by `K ≈ 50`; with many
annotations raise `theta_every` to amortise it. (Persisting the running `R@β`
residual across chunks — rather than rebuilding it each θ-update — keeps the
default cheap; without it `annot` was ~3× `auto` instead of ~1.9×.)

## Genotype-level simulation

`pyldpred2/simulate.py` is a full end-to-end simulation: it generates genotypes with
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
python -m pyldpred2.simulate --quick                        # fast (AR(1))
python -m pyldpred2.simulate --quick --ld-model coalescent  # realistic LD (needs msprime)
python -m pyldpred2.simulate --csv sim.csv                  # full accuracy grid, save results
```

Representative results (m=10000 SNPs, blocks of 200, AR(1) LD; prediction R² vs
phenotype):

| N | h² | p (causal) | marginal | inf | grid | auto | ceiling |
|---|----|-----------|---------|-----|------|------|---------|
| 5000  | 0.5 | 0.001 | 0.097 | 0.100 | 0.465 | 0.465 | 0.475 |
| 20000 | 0.5 | 0.001 | 0.254 | 0.262 | 0.489 | 0.489 | 0.489 |
| 20000 | 0.5 | 0.1   | 0.245 | 0.265 | 0.417 | 0.417 | 0.512 |
| 20000 | 0.3 | 0.01  | 0.135 | 0.139 | 0.301 | 0.300 | 0.311 |

LDpred2 always beats the raw marginal baseline; accuracy rises with heritability
and sample size; `grid`/`auto` approach the ceiling for sparse architectures and
remain best across the grid. The infinitesimal model only modestly beats the
marginal score — its all-causal prior leaves accuracy on the table whenever the
trait is even mildly sparse.

## Scaling: what the algorithm actually depends on

The LDpred2 *algorithm* works from summary statistics + the LD matrix, so its
cost is **independent of the GWAS sample size N** and is driven instead by the
**LD structure (block size)**. The benchmarks below separate the algorithm's
`fit` time from the simulation/GWAS/LD-construction `prep` time (which does scale
with N). Measured on a 4-core / 15 GB box, Numba on, h²=0.5, p=0.01.

**Independent of N** (`--n-independence`, m=10000, blocks of 200): fit time is
flat while prep grows with N.

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

Practical takeaway: for dense data with long-range / large LD blocks, the dense
per-block LD storage and the infinitesimal solve become the bottleneck, which
motivates the banded / sparse-LD backend (see [algorithm.md](algorithm.md)).
