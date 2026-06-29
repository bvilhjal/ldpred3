# End-to-end PRS pipeline

`pyldpred2/pipeline.py` runs the whole workflow from GWAS summary statistics and
genotype files to one polygenic score per individual — no R, NumPy-only:

```
GWAS sumstats + genotypes (PLINK/BGEN)
  → QC sumstats (N / MAF / INFO / duplicates / chi-sq outliers)
  → read & harmonise (align effect alleles to A1, drop ambiguous/mismatched)
  → SD-consistency QC vs the reference panel
  → per-block LD from a reference panel (in-sample or external)
  → [optional] DENTIST LD-consistency outlier removal (--dentist)
  → ldpred2 (inf / grid / auto / annot)
  → per-individual PRS
```

### Annotation-informed PRS (`--method annot`)

Pass a per-SNP annotation table to learn an SBayesRC-style functional prior
genome-wide and report the enrichment:

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink target --method annot \
    --annotations annot.tsv --out prs.txt
# ... annotation enrichment: coding=+1.20, conserved=+0.80, ...
```

The annotation file is a delimited table with a SNP-id column (`SNP`/`rsid`/...)
and numeric annotation columns; rows are matched to the GWAS variants by ID
(variants absent from the file get all-zero annotations). The learned
enrichment coefficients are in `PRSResult.enrichment`. Like `--infer`, this uses
a streaming genome-wide learner, so it never materialises the genome-wide LD.

From the command line:

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink target --method auto --out prs.txt
pyldpred2-prs --sumstats gwas.txt.gz --bgen  target.bgen --out prs.txt
```

or from Python:

```python
from pyldpred2 import run_ldpred2_prs
res = run_ldpred2_prs("gwas.txt.gz", "target", method="auto")
res.scores          # per-individual PRS
res.harmonize_log   # matched / flipped / ambiguous / mismatched counts
res.qc_log          # per-filter QC counts
```

### Practical flags

| Flag / argument | What it does |
|-----------------|--------------|
| `--dry-run` / `preflight_prs(...)` | detect columns, match IDs, preview harmonisation, then exit — no LD, no fit |
| `--save-weights FILE` / `res.write_weights(FILE)` | write the fitted weights (`ID CHR POS A1 A2 WEIGHT`) for reuse |
| `--weights FILE` / `score_from_weights(FILE, target)` | score a cohort from saved weights — no sumstats, LD or refit |
| `--ld-out FILE` / `--ld-cache FILE` | save the computed LD blocks and reload them on later runs |

`--save-weights` + `--weights` is the standard discovery → application split:
fit once, then score any number of new cohorts cheaply. `--ld-out`/`--ld-cache`
makes re-runs (different method, QC sweep) skip LD construction; the cache is
keyed to its variant set and refuses to apply if the harmonised variants change.
When `--annotations` is given the method defaults to `annot`.

### CLI reference

Every `pyldpred2-prs` flag (run `pyldpred2-prs --help` for the canonical list):

| Flag | Default | What it does |
|------|---------|--------------|
| `--sumstats FILE` | — | GWAS summary statistics (required unless `--weights`). See [Sumstats input format](#sumstats-input-format). |
| `--plink PREFIX` | — | Target genotypes as PLINK 1 `.bed/.bim/.fam` (one of `--plink`/`--bgen` required). |
| `--bgen FILE` | — | Target genotypes as BGEN v1.2 (alternative to `--plink`). |
| `--sample FILE` | none | BGEN `.sample` file (sample IDs for `--bgen`). |
| `--out FILE` | — | Output scores file (required for a run; see [Outputs](#outputs)). |
| `--method {auto,grid,inf,annot}` | `auto` | LDpred2 model; see [Choosing a model](../README.md#choosing-a-model). |
| `--annotations FILE` | none | Per-SNP annotation table; switches `--method` to `annot`. |
| `--block-size N` | `500` | Maximum variants per LD block. |
| `--n-eff FLOAT` | none | Effective sample size, used when the sumstats have no `N` column. |
| `--ld-prefix PREFIX` | in-sample | External LD reference panel (PLINK prefix); default is the target itself. |
| `--ld-ridge FLOAT` | `0.0` | Shrink each LD block towards the identity by this fraction (stabilises noisy panels). |
| `--ld-out FILE` | none | Save the computed LD blocks to `.npz` for reuse. |
| `--ld-cache FILE` | none | Reuse LD blocks saved earlier with `--ld-out` (skips LD construction). |
| `--ncores N` | `1` | Threads for the Gibbs sampler (requires Numba). |
| `--no-qc` | off | Skip the sumstats-only QC stage. |
| `--no-sd-check` | off | Skip the SD-consistency QC stage. |
| `--dentist` | off | Apply the DENTIST LD-consistency outlier filter (see [Sumstats QC](#sumstats-qc)). |
| `--infer` | off | Also infer h² / polygenicity / predictive r² (see [inference.md](inference.md)). |
| `--dry-run` | off | Preflight only: detect columns, match IDs, preview harmonisation, then exit. |
| `--save-weights FILE` | none | Also write the fitted weights for reuse. |
| `--weights FILE` | none | Score the target from a saved weights file (no sumstats / LD / refit). |

## Outputs

| Produced by | File | Columns (tab-separated, with header) |
|-------------|------|--------------------------------------|
| `--out` | scores | `FID  IID  PRS` — one row per target individual (`PRS` to 6 significant figures) |
| `--save-weights` | weights | `ID  CHR  POS  A1  A2  WEIGHT` — one row per scored variant (`WEIGHT` to 8 significant figures) |

The weights file is what `--weights` / `score_from_weights` reads back, so a
fit-once-score-many workflow round-trips through it. From Python the same data is
on the result object: `res.scores` (the PRS array) and `res.write_weights(path)`.

## Supporting modules (each usable on its own)

| Module          | What it does                                                           |
|-----------------|-----------------------------------------------------------------------|
| `genotype_io`   | Read/write PLINK 1 `.bed/.bim/.fam` (2-bit decode, NumPy-only)         |
| `bgen_io`       | Read BGEN v1.2/layout-2 (uncompressed or zlib; biallelic diploid)      |
| `sumstats`      | Parse GWAS files with flexible column aliases (OR→β, SE-from-p)        |
| `qc`            | Sumstats QC: N / MAF / INFO / duplicate / chi-sq + SD-consistency + DENTIST |
| `harmonize`     | Match variants + align effect alleles (swap-flip, strand, palindrome) |
| `ld`            | Per-block LD correlation matrices from a genotype panel                |
| `prs`           | Weighted polygenic scores with missing-call imputation                |

## Sumstats QC

Runs by default (`qc=True`, disable with `--no-qc`). Two stages, following the
bigsnpr / LDpred2 tutorial:

* **Sumstats-only** (`qc.qc_sumstats`, before harmonisation): drop non-finite or
  non-positive-SE rows, duplicated variants, low-`N` variants (`N < 0.7·max N`),
  low-MAF (`< 0.01`, when an EAF column is present), low-INFO (`< 0.7`, when
  present) and chi-square outliers (optional `max_chisq`).
* **SD-consistency** (`qc.sd_consistency_mask`, after harmonisation): compare the
  SD implied by the sumstats, `sd_ss ≈ 1/√(N·se² + β²)`, against the reference
  genotype SD `sd_ref = √(2·f·(1−f))`, and drop variants where the ratio leaves
  `[0.5, 2]`. This catches a wrong `N`, allele errors or bad imputation that
  harmonisation cannot.
* **DENTIST LD-consistency** (`qc.dentist_outlier_mask`, opt-in with `--dentist`,
  after the LD blocks are built): within each LD block, test whether each
  variant's z-score agrees with the value predicted from its LD neighbours
  (studentized leave-one-out residual `T_j = (Ωz)_j²/Ω_jj ~ χ²₁`, `Ω=(R+ridge·I)⁻¹`).
  The single worst variant above the `5e-8` threshold is dropped, the LD blocks
  are rebuilt on the survivors, and the pass repeats — removing one variant at a
  time so a single corrupt SNP cannot take its whole tagged haplotype down with
  it. Catches allele/strand errors and local LD-reference mismatch that survive
  the SD-check.

  Two safeguards make it conservative. (1) Only variants with an LD neighbour
  (`|r| ≥ 0.1` with some block-mate) are removal candidates: with no neighbour the
  residual is just the variant's own z, so in a low-LD / near-identity region
  *every* genome-wide-significant hit would otherwise be flagged. (2) The χ²
  cutoff is stringent. It is **off by default** because, even so, it can drop a
  genuine but poorly-tagged independent association along with true errors. Tune
  via `dentist_params` (e.g. `{"p_cutoff": 1e-6}`).

## Sumstats input format

The reader (`sumstats.read_sumstats`) auto-detects columns from a tab-, comma- or
whitespace-delimited file (optionally `.gz`), matching a large set of
case-insensitive aliases onto a canonical schema. From Python you can override
any column explicitly with `sumstats_cols={...}` (e.g.
`run_ldpred2_prs(..., sumstats_cols={"beta": "EFFECT"})`).

| Field | Required? | Recognised aliases (case-insensitive) |
|-------|-----------|---------------------------------------|
| `id` | for matching | `snp`, `rsid`, `rs`, `id`, `variant_id`, `markername`, `snpid`, `marker`, `rs_id` |
| `ea` (effect allele) | **yes** | `a1`, `effect_allele`, `ea`, `allele1`, `alt`, `effectallele`, `tested_allele`, `inc_allele` |
| `oa` (other allele) | **yes** | `a2`, `other_allele`, `oa`, `nea`, `allele0`, `allele2`, `ref`, `noneffect_allele`, `dec_allele` |
| `beta` | one of `beta`/`or` | `beta`, `b`, `effect`, `effect_size`, `effects`, `log_odds` |
| `or` | one of `beta`/`or` | `or`, `odds_ratio`, `oddsratio` (converted to `beta = log(OR)`) |
| `se` | recommended | `se`, `standard_error`, `stderr`, `standarderror`, `sebeta`, `se_beta`, `logor_se` |
| `pval` | only if `se` absent | `p`, `pval`, `p_value`, `pvalue`, `p-value`, `p.value`, `p_bolt_lmm`, `p_value_association`, `p_wald`, `pval_nominal` |
| `n_eff` | yes, or pass `--n-eff` | `n_eff`, `neff`, `n`, `sample_size`, `totalsamplesize`, `n_samples`, `n_total`, `obs_ct`, `n_complete_samples` |
| `chrom` | optional | `chr`, `chrom`, `chromosome`, `#chrom`, `hg19chr`, `chr_name` |
| `pos` | optional | `bp`, `pos`, `position`, `base_pair_location`, `bp_position`, `pos_b37`, `base_pair` |
| `eaf` | optional (MAF QC) | `eaf`, `freq`, `frq`, `effect_allele_frequency`, `a1freq`, `freq1`, `maf`, `af`, `effect_allele_freq` |
| `info` | optional (INFO QC) | `info`, `imputation_info`, `imp_info`, `rsq`, `r2`, `info_score`, `imputation_quality`, `minimac_r2` |

Notes: effects given as odds ratios become `beta = log(OR)`. If `se` is missing
but `pval` is present, `se` is recovered from the (two-sided) p-value and `beta`.
If there is no sample-size column, pass `--n-eff` / `n_eff=`. Use `--dry-run` to
print the detected column mapping and the match/flip/ambiguous counts before
committing to a full run.

## Format / harmonisation notes

Dosages count the A1 (first) allele; missing calls are `-1` (PLINK, hard calls)
or `NaN` (BGEN, dosages in `[0,2]`). Strand-ambiguous (A/T, C/G) and
allele-mismatched variants are dropped during harmonisation. BGEN with zstd
compression needs the optional `zstandard` package (a clear error is raised
otherwise).
