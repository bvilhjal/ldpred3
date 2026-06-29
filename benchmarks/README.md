# Benchmarks

Measurement / reproducibility scripts (not part of the `pytest` suite). Results
and discussion live in [`docs/benchmarks.md`](../docs/benchmarks.md) and
[`docs/inference.md`](../docs/inference.md). Run single-core for stable timings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/<script>.py
```

**LD library.** Most scripts simulate from a cached coalescent LD library,
`ld_library.npz` (100 blocks × 500×500 correlation matrices, ~200 MB, **not
committed**), expected in the working directory. Generate it once with msprime
(see `pyldpred2/simulate.py --ld-model coalescent`) or substitute your own
`{"R": array of shape (n_blocks, k, k)}`. The "self-contained" scripts below need
no external data.

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
| `bench_bigsnpr_blocks.R` | bigsnpr (R reference) side of the time/memory/accuracy comparison | — |
| `plot_methods_1core.py` | Plots the 1-core pyLDpred2-vs-bigsnpr comparison (→ `cores_1core_benchmark.png`) | self-contained (reads CSV) |
| `plot_methods_arch.py` | Renders the methods-by-architecture figure from the CSV | self-contained (reads CSV) |

## Features: QC, LD representation & performance

All **self-contained** (no `ld_library.npz`): they simulate an AR(1) genotype
panel internally, so they run anywhere.

| Script | What it measures |
|--------|------------------|
| `dentist_recovery.py` | DENTIST filter: PRS R² recovered after planted allele/strand errors, error catch-rate, and false-drop cost on clean data |
| `sparse_ld_tradeoff.py` | Sparse / banded LD: storage (density) and fit time vs accuracy across thresholding / banding settings |
| `block_splitting.py` | `optimal_ld_blocks` vs fixed-size blocks: discarded between-block LD², per-block storage and accuracy |
| `numba_speedup.py` | Numba-JIT vs pure-Python fit time for the Gibbs sampler (runs itself twice, with/without JIT) |
| `cores_scaling.py` | Multi-core (`--ncores`) speed-up and parallel efficiency of the packed auto sampler |

## Inference: h² / genetic correlation

| Script | What it measures | Needs LD lib |
|--------|------------------|:---:|
| `compare_ldsc_infer.py` | Heritability: LDSC vs LDpred2-auto-infer vs truth (reference-panel LD) | ✓ |
| `compare_bivariate_rg.py` | Genetic correlation: bivariate LDSC vs bivariate LDpred2 vs truth | ✓ |
| `inference_benchmark.py` | Accuracy **and** running time for all inference estimators (incl. a marginal no-LD baseline) | ✓ |
| `bivariate_demo.py` | Bivariate prediction gain for a weak trait across two-trait architectures | ✓ |
| `calibration.py` | 95% interval coverage for the inference methods (clean vs reference-panel LD) | ✓ |
| `sample_overlap.py` | Validates the overlap corrections (LDSC intercept, bivariate `cross_corr`) | ✓ |

All inference/robustness scripts fit with an LD matrix estimated from a finite
**reference panel** (not the true population LD that generates the GWAS) — the
mismatch that dominates real-world error.
