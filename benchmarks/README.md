# Benchmarks

Measurement / reproducibility scripts (not part of the `pytest` suite). Results
and discussion live in [`docs/benchmarks.md`](../docs/benchmarks.md) and
[`docs/inference.md`](../docs/inference.md). Run single-core for stable timings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/<script>.py
```

**Publication figures.** `make_paper_figures.py` assembles the headline results
into one multi-page PDF (`benchmarks/figures.pdf`): the bigsnpr time/memory
comparison, the cold-init `auto` accuracy/time comparison, and
accuracy-by-architecture (from the committed CSVs), plus h²/
polygenicity inference recovery, inference cross-checks (h² LDSC-vs-LDpred3-auto
and predictive-r² estimated-vs-realized), bivariate analysis (genetic-correlation
recovery and weak-trait prediction gain), DENTIST recovery, sparse/banded LD,
optimal block splitting, the Numba speed-up and multi-core scaling (computed from
a self-contained simulation and cached to `figdata_*.csv`). Needs `matplotlib`;
the LDSC/bivariate pages need realistic LD via `msprime`, and the performance
page needs `numba` — each is skipped with a note if its dependency is absent.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/make_paper_figures.py
```

**LD library.** Most scripts simulate from a cached coalescent LD library,
`ld_library.npz` (100 blocks × 500×500 correlation matrices, ~100 MB `float32`,
**not committed**), expected in the working directory. Generate it once with

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/make_ld_library.py
```

(needs msprime; ~15 s) or substitute your own `{"R": array of shape
(n_blocks, k, k)}`. The "self-contained" scripts below need no external data.

## Methods: accuracy

| Script | What it measures | Needs LD lib |
|--------|------------------|:---:|
| `bench_methods.py` | Genetic R² of marginal/inf/grid/auto/annot across genetic architectures (→ `methods_arch_benchmark.{csv,png}`) | ✓ |
| `sweep_p_h2_n.py` | PRS accuracy swept over polygenicity, heritability and sample size | ✓ |
| `robustness_ld_and_n.py` | Sensitivity to LD-reference-panel size and to a misspecified GWAS `N` | ✓ |
| `diagnose_annot.py` | Why `annot` under-converges at low power (the `theta_every` finding) | ✓ |

## Methods: running time & memory

| Script | What it measures | Needs LD lib |
|--------|------------------|:---:|
| `timing_bench.py` | Per-method fit time at m=50k; `annot` cost vs #annotations and `theta_every` | ✓ |
| `method_scaling.py` | **All methods compared on accuracy + time + memory together, 50k→1M SNPs** (marginal/inf/grid/auto/annot; → `method_scaling.{csv,png}`, per-size subprocess for clean RSS) | ✓ |
| `bench_ldpred3_scaling.py` | LDpred3-only fit time + peak memory 200k→4M SNPs (genome-scale ceiling; → `ldpred3_scaling.{csv,png}`) | ✓ |
| `bench_vs_bigsnpr.py` | From-scratch LDpred3-vs-bigsnpr driver (200k–2M, 1 core): shared sim → both tools → time / peak memory / accuracy (→ `cores_1core_benchmark.csv`) | ✓ |
| `bench_cold_init.py` | `auto` cold-started for **both** tools (no oracle hyper-parameters), the realistic scenario (→ `cold_init_auto.csv`) | ✓ |
| `bench_bigsnpr_blocks.R` | bigsnpr (R reference) side of the time/memory/accuracy comparison | — |
| `plot_methods_1core.py` | Plots the 1-core LDpred3-vs-bigsnpr comparison (→ `cores_1core_benchmark.png`) | self-contained (reads CSV) |
| `plot_methods_arch.py` | Renders the methods-by-architecture figure from the CSV | self-contained (reads CSV) |

## Features: QC, LD representation & performance

All **self-contained** (no `ld_library.npz`): they simulate a genotype panel
internally with realistic **coalescent** LD (`simulate_genotypes_coalescent`), so
they run anywhere `msprime` is installed (`pip install msprime`).

| Script | What it measures |
|--------|------------------|
| `infer_recovery.py` | LDpred3-auto-infer: h² and polygenicity recovery vs truth, CI width and empirical 95% coverage (no validation cohort) |
| `finemap_recovery.py` | LDpred3-PIP fine-mapping: credible-set coverage, power and resolution (set size) vs signal strength / #causal, clean vs reference-panel LD, and vs the ABF baseline |
| `finemap_architectures.py` | Fine-mapping **methods × architectures**: LDpred3-PIP vs ABF vs marginal-top across single / two-signal (independent & linked) / polygenic-background / major+sparse loci — coverage, power, resolution |
| `finemap_genomewide.py` | **Genome-wide** fine-mapping (`finemap_by_blocks`) over 100k–500k SNPs: genome-wide coverage/power/resolution + runtime, and `only_significant` vs all-blocks (false-set rate). *(needs `ld_library.npz`)* |
| `impute_annot.py` | **LD imputation × functional annotations**: drop functional causals, impute, fit auto/annot, matched vs mismatched LD — prediction R² (misspecification fix) and localisation of untyped functional causals. *(needs `ld_library.npz`)* |
| `dentist_recovery.py` | DENTIST filter: PRS R² recovered after planted allele/strand errors, error catch-rate, and false-drop cost on clean data |
| `ld_shrink_large_blocks.py` | Size-aware LD shrinkage (`shrink_ld_blocks`): R² and h² on a finite reference panel, no-shrink vs uniform vs size-aware, across Nref |
| `ld_memory_scaling.py` | Persistent LD storage dense O(k²) vs banded SparseLD O(k·w) across block sizes, with a 10M-SNP extrapolation (realistic coalescent LD) |
| `ld_lowrank.py` | Low-rank `LowRankLD` (eigen/PC) vs dense vs banded on realistic LD: genetic R² and memory (low-rank matches dense at ~24% memory; banding loses accuracy) |
| `ld_representations.py` | Memory **and running time** by LD representation (dense / banded / low-rank) on realistic large blocks: LD memory, build time, fit time, R² — the memory↓/time↑ trade-off |
| `sparse_ld_tradeoff.py` | Sparse / banded LD: storage (density) and fit time vs accuracy across thresholding / banding settings |
| `block_splitting.py` | `optimal_ld_blocks` vs fixed-size blocks: discarded between-block LD², per-block storage and accuracy |
| `numba_speedup.py` | Numba-JIT vs pure-Python fit time for the Gibbs sampler (runs itself twice, with/without JIT) |
| `cores_scaling.py` | Multi-core (`--ncores`) speed-up and parallel efficiency of the packed auto sampler |

## Inference: h² / genetic correlation

| Script | What it measures | Needs LD lib |
|--------|------------------|:---:|
| `compare_ldsc_infer.py` | Heritability: LDSC vs LDpred3-auto-infer vs truth (reference-panel LD) | ✓ |
| `infer_architectures.py` | h² (LDSC & LDpred3-auto) and polygenicity (LDpred3-auto) recovery **across four architectures** (infinitesimal / sparse / polygenic / major-locus), reference-panel LD | ✓ |
| `rg_architectures.py` | Genetic correlation (bivariate LDSC vs bivariate LDpred3) vs truth, swept over r_g **× the four architectures**, reference-panel LD | ✓ |
| `compare_bivariate_rg.py` | Genetic correlation: bivariate LDSC vs bivariate LDpred3 vs truth | ✓ |
| `inference_benchmark.py` | Accuracy **and** running time for all inference estimators (incl. a marginal no-LD baseline) | ✓ |
| `bivariate_demo.py` | Bivariate prediction gain for a weak trait across two-trait architectures | ✓ |
| `calibration.py` | 95% interval coverage for the inference methods (clean vs reference-panel LD) | ✓ |
| `sample_overlap.py` | Validates the overlap corrections (LDSC intercept, bivariate `cross_corr`) | ✓ |
| `infer_scaling.py` | Running time of LDpred3-auto inference, dense vs streaming, as m grows (and h² agreement) | ✓ |

All inference/robustness scripts fit with an LD matrix estimated from a finite
**reference panel** (not the true population LD that generates the GWAS) — the
mismatch that dominates real-world error.
