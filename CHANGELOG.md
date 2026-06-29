# Changelog

All notable changes to **LDpred3** are recorded here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Complete CLI reference and output-format tables in `docs/pipeline.md`, a
  GWAS-sumstats input-format table (recognised column aliases), a `LICENSE`
  (MIT) and this changelog.
- Help text and documented defaults for previously bare CLI flags
  (`--method`, `--block-size`, `--n-eff`, `--ld-ridge`, `--ncores`).

### Changed
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
