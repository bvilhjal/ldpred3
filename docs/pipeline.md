# End-to-end PRS pipeline

`pyldpred2/pipeline.py` runs the whole workflow from GWAS summary statistics and
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

## Supporting modules (each usable on its own)

| Module          | What it does                                                           |
|-----------------|-----------------------------------------------------------------------|
| `genotype_io`   | Read/write PLINK 1 `.bed/.bim/.fam` (2-bit decode, NumPy-only)         |
| `bgen_io`       | Read BGEN v1.2/layout-2 (uncompressed or zlib; biallelic diploid)      |
| `sumstats`      | Parse GWAS files with flexible column aliases (OR→β, SE-from-p)        |
| `qc`            | Sumstats QC: N / MAF / INFO / duplicate / chi-sq + SD-consistency      |
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

## Format / harmonisation notes

Dosages count the A1 (first) allele; missing calls are `-1` (PLINK, hard calls)
or `NaN` (BGEN, dosages in `[0,2]`). Strand-ambiguous (A/T, C/G) and
allele-mismatched variants are dropped during harmonisation. BGEN with zstd
compression needs the optional `zstandard` package (a clear error is raised
otherwise).
