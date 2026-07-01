# Changelog

All notable changes to **LDpred3** are recorded here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Fine-mapping** (`docs/finemap.md`). The spike-and-slab sampler's per-SNP
  posterior inclusion probability (PIP) is turned into the standard fine-mapping
  outputs, reusing the same model / LD / QC as the PRS:
  - `ldpred3_pip` — per-locus PIPs, posterior effect mean/SD and **credible sets**
    (LD-clustered, purity-filtered, with tie-expansion so the 95% set stays
    calibrated); `single_signal_finemap` ABF baseline.
  - `finemap_by_blocks` — genome-wide driver over LD blocks (independent →
    parallel via `ncores`), with an `only_significant` loci filter.
  - `run_finemap` + CLI `--finemap` — GWAS file → `<out>.pip.tsv` (per-variant)
    and `<out>.cs.tsv` (credible sets); `--regions` (BED) /
    `--finemap-only-significant` / `--finemap-coverage`. Benchmark: credible-set
    coverage ~0.95 with median set size ~2 (`benchmarks/finemap_recovery.py`).
- **Scaling to millions of SNPs.** Composable LD representations for the
  genome / sequencing-scale regime (`docs/pipeline.md` → Scaling):
  - **Low-rank LD** (`LowRankLD`, `--ld-lowrank`): top-eigenvector blocks fit in
    the eigenspace at O(k·rank) memory — matches dense accuracy at ~¼ memory on
    realistic LD (the SBayesRC-style representation, validated vs banding which is
    lossy on realistic LD).
  - **Mixed** dense + low-rank by block size (`--ld-lowrank-min-size`): compress
    only the big blocks, keep small ones fast and dense.
  - **Banded `SparseLD`** construction + a sparse streaming sampler kernel
    (`--ld-sparse`/`--ld-max-dist`) for genuinely banded / array-like LD.
  - **On-disk LD streaming** (`--ld-stream`): memory-mappable cache so an LD
    larger than RAM streams from disk (fits bit-identical).
  - **Size-aware LD shrinkage** (`--ld-shrink`) toward the identity for large,
    noisy reference-panel blocks.
- **DENTIST LD-consistency filter** (`--dentist`, `qc.dentist_outlier_mask`).
- Complete CLI reference and output-format tables in `docs/pipeline.md`, a
  GWAS-sumstats input-format table (recognised column aliases), a `LICENSE`
  (MIT) and this changelog.
- Help text and documented defaults for previously bare CLI flags
  (`--method`, `--block-size`, `--n-eff`, `--ld-ridge`, `--ncores`).
- Self-contained benchmarks for the LD representations, DENTIST, inference and
  bivariate analysis; the publication figure generator
  (`benchmarks/make_paper_figures.py`) now covers them.

### Changed
- **Gibbs samplers: constant-N / uniform-prior fast path** in the dense
  (`_gibbs_kernel`), sampling (`_gibbs_kernel_sample`) and sparse
  (`_gibbs_kernel_sparse`) kernels, back-porting the hoist the batched/streaming
  kernels already used. When N is shared across variants (scalar `n_eff`) and the
  slab/prior are uniform, the per-SNP posterior scalars (`sqrt`/`log1p`) and the
  log prior-odds are computed once per sweep instead of per SNP. Output is
  **bit-identical** (same arithmetic, same RNG stream); ~1.3–1.5× faster in the
  sparse-`p` regime that dominates fine-mapping (`ldpred3_pip`) and the auto
  inference / r² chains (`ldpred3_auto_infer`). Falls back to the exact per-SNP
  path for per-variant N, MAF slab weights or non-uniform `prior_weights`.
- `prs._as_float_with_nan` copies the dosage matrix once (`np.array`) instead of
  twice (`np.asarray().copy()` up-cast then duplicated integer dosages), halving
  the transient memory of every standardization / scoring / LD-block build.
- `finemap._credible_sets` finds a signal's out-of-set neighbours with a boolean
  mask instead of `np.isin` (no internal sort); credible sets are unchanged.
- **Project renamed pyLDpred2 → LDpred3** (package `ldpred3`, CLI `ldpred3`, API
  `ldpred3_*`); citations to the LDpred2 method are kept.
- `qc.dentist_outlier_mask` skips blocks that have settled, avoiding redundant
  matrix re-inversions across passes (no change in output). (#30)
- `run_ldpred3_prs` now warns that `dentist=True` is ignored on the `ld_cache`
  path (cached LD is authoritative). (#30)

## [0.1.0]

First public version: a dependency-light (NumPy-only, optional Numba)
implementation of LDpred2 with a full summary-statistics → polygenic-score
pipeline, no R required.

### Added
- **DENTIST-style LD-consistency filter** (`qc.dentist_outlier_mask`,
  `--dentist`) that drops variants whose z-score disagrees with their LD
  neighbours; split the LD container / construction utilities into
  `ld_utils.py` and the Numba shim into `_numba.py`. (#29)
- **Streaming genome-wide LDpred3-auto inference** of h², polygenicity and
  predictive r² with no validation cohort, plus real-world sumstats tests. (#28, #27)
- **Bivariate (two-trait) LDpred3** and **LD Score regression** for h² and
  genetic correlation (`ldpred3_auto_bivariate`, `ldsc_h2`, `ldsc_rg`). (#24)
- **Annotation-informed priors** (SBayesRC-style, supplied or learned) wired
  into the pipeline via `--method annot`. (#20 and earlier)
- **Optimal LD-block splitting** (Privé 2022) and **sparse / banded LD**
  support for the sampler.
- **Usability**: `--dry-run` preflight, weight save/reuse (`--save-weights` /
  `--weights`), LD caching (`--ld-out` / `--ld-cache`), and a task-oriented
  user guide. (#21, #22, #23)
- **Benchmarks** vs `bigsnpr`: accuracy by genetic architecture, scaling to
  millions of SNPs, and robustness to LD-reference quality, sample size,
  sample overlap and interval calibration. (#24, #25, #26)
- I/O for PLINK 1 `.bed/.bim/.fam` and BGEN v1.2, flexible GWAS sumstats
  parsing (column aliases, OR→β, SE-from-p), harmonisation and sumstats QC.

[Unreleased]: https://github.com/bvilhjal/ldpred3/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bvilhjal/ldpred3/releases/tag/v0.1.0
