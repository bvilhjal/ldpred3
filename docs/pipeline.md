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

## Format / harmonisation notes

Dosages count the A1 (first) allele; missing calls are `-1` (PLINK, hard calls)
or `NaN` (BGEN, dosages in `[0,2]`). Strand-ambiguous (A/T, C/G) and
allele-mismatched variants are dropped during harmonisation. BGEN with zstd
compression needs the optional `zstandard` package (a clear error is raised
otherwise).
