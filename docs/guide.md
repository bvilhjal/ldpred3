# User guide

A task-oriented walkthrough: from a GWAS + a target dataset to a polygenic
score, how to pick a model, and how to read the output. For the maths and
internals see [algorithm.md](algorithm.md); for the full benchmarks see
[benchmarks.md](benchmarks.md).

**Jump to:** [what you need](#1-what-you-need) ·
[run it](#2-the-one-command-path-recommended) ·
[is it any good?](#3-is-the-prs-any-good-evaluating) ·
[choose a model](#4-choosing-a-model) ·
[your own LD blocks](#5-working-from-your-own-ld-blocks-library-path) ·
[annotations](#6-annotation-informed-prs-annot) ·
[h²/p/r² inference](#7-inferring-h-polygenicity-and-predictive-r-no-validation-set) ·
[reuse work](#8-re-using-work-saved-weights--cached-ld) ·
[scaling](#9-scaling--performance) · [troubleshooting](#10-troubleshooting)

> **In one line:** `pip install . numba` then
> `pyldpred2-prs --sumstats gwas.txt.gz --plink target --out prs.txt`
> (`auto` is the default model). Everything below is detail on the inputs, the
> model choice, and reading the output.

## 1. What you need

| Input | What it is | Notes |
|-------|------------|-------|
| **GWAS summary statistics** | per-variant marginal effect, SE (or p), allele(s), N | one text/`.gz` table; flexible column names (`sumstats` parses aliases, `OR→β`, SE-from-p) |
| **Target genotypes** | the individuals you want to score | PLINK `.bed/.bim/.fam` or BGEN v1.2 |
| **LD reference** | correlation between variants | by default computed **in-sample** from the target; or pass an external panel |

You do **not** need a validation/tuning cohort: `auto` self-tunes its
hyper-parameters, and `ldpred2_auto_infer` even estimates predictive r² without
one.

**LD reference & ancestry — the gotcha that quietly breaks PRS.** LDpred2's whole
job is to undo LD, so the LD reference must match the GWAS *and* the target
**ancestry**. Practical guidance:

- **In-sample LD (default)** — computed from the target — is fine when the target
  is reasonably large (thousands of individuals) and the same ancestry as the
  GWAS. It is the simplest path.
- **External panel** (`ld_prefix=` / `--ld-prefix`) — use a matched-ancestry
  reference (e.g. the relevant 1000 Genomes superpopulation) when the target is
  small, so the LD isn't estimated from a handful of people.
- **Cross-ancestry** (GWAS and target differ in ancestry) is *out of scope* for a
  single-population LDpred2 and will under-perform — that needs a cross-ancestry
  method. Keep GWAS, LD reference and target ancestry aligned.

## 2. The one-command path (recommended)

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink target --method auto --out prs.txt
```

This runs the whole pipeline: QC → harmonise alleles → per-block LD → LDpred2 →
one score per individual. The equivalent in Python, with the logs you should
check:

```python
from pyldpred2 import run_ldpred2_prs
res = run_ldpred2_prs("gwas.txt.gz", "target", method="auto")

res.scores          # np.ndarray, one PRS per individual (res.sample_iid for IDs)
res.qc_log          # how many variants each QC filter dropped
res.harmonize_log   # matched / flipped / ambiguous / mismatched counts
```

**Read the logs before trusting the scores.** If `harmonize_log` shows most
variants `mismatched`, your sumstats and genotypes use different builds or allele
codings; if `qc_log` drops almost everything, check the column mapping and `N`.
See [pipeline.md](pipeline.md) for every filter and file-format detail.

**Check inputs first with `--dry-run`.** Before committing to a genome-wide run,
preflight it — this detects the column mapping, matches IDs and previews
harmonisation in seconds, without computing LD or fitting:

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink target --dry-run
# detected columns: id=SNP, ea=A1, oa=A2, beta=BETA, se=SE, n_eff=N
# matched 31204 / 31875 to the target (12 flipped, 41 ambiguous, ...)
```

If a required column is mis-detected, pass it explicitly — by column name or
index — and re-check:

```python
run_ldpred2_prs("gwas.txt.gz", "target",
                sumstats_cols={"beta": "EFFECT", "ea": "ALT", "n_eff": 8})
```

A near-zero match rate in the preflight almost always means an ID convention or
build mismatch (e.g. `rs#` sumstats vs `chr:pos` genotype IDs — the pipeline
falls back to position matching, but only within the same build).

**Outputs.** The score file is a 3-column table, one row per individual:

```
FID     IID     PRS
fam1    ind1    0.0421
fam1    ind2   -0.0137
```

In Python the same numbers are `res.scores` (aligned to `res.sample_fid` /
`res.sample_iid`). `res.beta_adjusted` holds the per-variant weights (see
[§8](#8-re-using-work-saved-weights--cached-ld) to save and reuse them).

## 3. Is the PRS any good? (evaluating)

A PRS is only useful if it predicts. Two ways to judge it:

- **With a phenotype** (the gold standard): correlate the scores with a measured
  trait in the target or a held-out set. The squared Pearson correlation
  `cor(PRS, phenotype)²` is the realised prediction R² (for a binary trait, use
  the AUC or Nagelkerke R²). Always assess in individuals **not** in the GWAS.
- **Without a phenotype:** `--infer` / `ldpred2_auto_infer` estimates the PRS's
  expected out-of-sample r² from the summary statistics alone, with a credible
  interval ([§7](#7-inferring-h-polygenicity-and-predictive-r-no-validation-set)).
  It tracks the realised R² closely and falls toward 0 as GWAS power drops, so
  it's a useful a-priori check before you have outcome data.

If accuracy is far below the inferred r², suspect an input problem (ancestry
mismatch, wrong `N`, or LD that doesn't match the target) rather than the model.

## 4. Choosing a model

```
                trait is ~infinitesimal (highly polygenic, no big loci)?
                          │ yes                    │ no
                     ┌────┴─────┐            sparse / has major loci
                     │   inf    │                  │
                     └──────────┘         do you have trustworthy
                                          per-SNP functional annotations?
                                            │ yes              │ no
                                       ┌────┴────┐        ┌────┴────┐
                                       │  annot  │        │  auto   │
                                       └─────────┘        └─────────┘
```

| Model | When to use | Hyper-parameters |
|-------|-------------|------------------|
| **`auto`** | **the default** — self-tunes `h²` and `p`, matches the oracle `grid`, robust across architectures | none |
| **`inf`** | you believe the trait is truly infinitesimal, or you want the cheapest/most-robust baseline | `h2` |
| **`grid`** | you already know `h²` and `p` (e.g. from a previous fit) and want a fixed-hyper sampler | `h2`, `p` |
| **`annot`** | you have functional annotations (coding, conserved, enhancers, …) and want to exploit them | none (learns the map) |

Empirically (see [benchmarks.md](benchmarks.md)): the raw marginal PRS is always
far behind; `inf` is robust but flat and only wins under a truly infinitesimal
architecture; `grid`/`auto` win decisively on sparse / major-locus traits; `auto`
matches the oracle `grid` with no tuning; `annot` matches `auto` when the
annotation is uninformative and beats it when it carries signal.

## 5. Working from your own LD blocks (library path)

If you already have summary stats and an LD matrix for a region (or a whole
genome split into blocks), skip the pipeline and call the model directly. Effects
are on the **standardized scale** — `standardize_betas` converts reported GWAS
effects to it and gives you the back-transform:

```python
import numpy as np
from pyldpred2 import standardize_betas, ldpred2_auto, ldpred2_by_blocks

# one block: beta/beta_se/n_eff are GWAS stats, corr is the (m x m) LD matrix
beta_hat, scale = standardize_betas(beta, beta_se, n_eff)
res = ldpred2_auto(corr, beta_hat, n_eff)
adjusted_beta = res.beta_est * scale          # back to the input (per-allele) scale
print(res)                                    # AutoResult(h2_est=…, p_est=…)

# genome-wide: blocks is a list of (corr_block, index_array) tiling 0..m-1
beta_est = ldpred2_by_blocks(blocks, beta_hat, n_eff, method="auto")
```

`ldpred2_by_blocks(method="auto")` streams blocks one at a time and pools `h²`/`p`
globally, so the genome-wide LD is never materialised (this is the path that
scales to millions of SNPs). Build blocks with `block_diagonal_ld` or, better,
`optimal_ld_blocks` (cuts in recombination valleys — see [algorithm.md](algorithm.md)).

**Two correlated traits?** `ldpred2_auto_bivariate(corr, beta_hat1, beta_hat2,
n1, n2)` fits both jointly, learning their genetic correlation so a well-powered
trait sharpens a weaker correlated one (and reporting `res.rg`, `res.h2`). See
[algorithm.md](algorithm.md#bivariate-two-trait-ldpred2).

## 6. Annotation-informed PRS (`annot`)

Supply a per-SNP annotation table and the sampler learns an SBayesRC-style
functional prior `p_j = sigmoid(a_jᵀθ)` genome-wide, returning the learned
enrichment:

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink target --method annot \
    --annotations annot.tsv --out prs.txt
# ... annotation enrichment: coding=+1.20, conserved=+0.80, ...
```

```python
res = run_ldpred2_prs("gwas.txt.gz", "target", method="annot",
                      annotations="annot.tsv")
res.enrichment            # {"coding": 1.20, "conserved": 0.80, ...}
```

The annotation file is a delimited table with a SNP-id column (`SNP`/`rsid`/…)
and numeric annotation columns; rows are matched to the GWAS variants by ID
(variants absent from the file get all-zero annotations). A large positive
coefficient means that annotation enriches for causal variants. Because the map
is *learned*, an uninformative annotation harmlessly collapses to θ≈0 — there is
no "garbage-in" penalty like a fixed bad prior would carry.

**One knob that matters:** `theta_every` (how often the map is refit). The
default `1` (refit every sweep) is what makes the map converge within a normal
chain — don't raise it unless you have **many** annotations (≳50) and the
`O(m·K²)` refit cost starts to dominate. See the convergence note in
[benchmarks.md](benchmarks.md) for why.

## 7. Inferring h², polygenicity and predictive r² (no validation set)

To get heritability, polygenicity and the PRS's expected out-of-sample r² —
each with a credible interval and **without** a held-out cohort:

```python
from pyldpred2 import ldpred2_auto_infer
res = ldpred2_auto_infer(corr, beta_hat, n_eff, n_chains=10)
res.h2_est, res.h2_ci     # heritability + 95% CI
res.p_est,  res.p_ci      # polygenicity + 95% CI
res.r2_est, res.r2_ci     # predicted out-of-sample r² + 95% CI
```

or from the pipeline with `infer=True` / `--infer`. Inference **streams the LD
blocks**, so it runs genome-wide (no dense matrix is assembled). For an
independent h² check, `ldsc_h2` runs **LD Score regression** on the same summary
statistics (it agrees with the LDpred2-auto h² but is less precise — see
[inference.md](inference.md)). Full method and validation in
[inference.md](inference.md).

## 8. Re-using work: saved weights & cached LD

Two flags avoid redoing the expensive parts when you score more cohorts or
re-run with different settings.

**Save the fitted weights, then score new cohorts for free.** The weights (one
standardized effect per variant, with its allele) are the reusable product —
scoring another cohort from them skips LD construction and the whole LDpred2 fit:

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink discovery --save-weights prs.weights.txt --out d.txt
pyldpred2-prs --plink new_cohort --weights prs.weights.txt --out new.txt   # no sumstats / LD / refit
```

```python
res.write_weights("prs.weights.txt")                 # from a PRSResult
score_from_weights("prs.weights.txt", "new_cohort")  # -> ScoreResult
```

Weights are harmonised to each new target's alleles (sign-flipped where alleles
are swapped), so a cohort with the opposite A1/A2 coding still scores correctly.

**Cache the LD blocks** so re-runs (e.g. trying `grid` vs `auto`, or sweeping QC)
don't recompute LD:

```bash
pyldpred2-prs --sumstats gwas.txt.gz --plink target --method auto --ld-out ld.npz   --out auto.txt
pyldpred2-prs --sumstats gwas.txt.gz --plink target --method grid --ld-cache ld.npz --out grid.txt
```

The cache records the variant set it was built for; if your inputs/QC change so
the harmonised variants differ, the run stops with a clear error rather than
silently using stale LD — rebuild it with `--ld-out`.

## 9. Scaling & performance

- **Install Numba** (`pip install numba`). The inner sampler is JIT-compiled and
  cached; without it you get identical results but a much slower pure-Python
  loop — fine for CI, not for genome-wide runs.
- **Memory** is dominated by the LD. `auto`'s streaming sampler keeps only one
  block resident (float32), so peak memory is the LD plus `O(m)` state — ~4 GB at
  2M SNPs. Prefer `ldpred2_by_blocks(method="auto")` (global hyper) at genome
  scale.
- **Sparse / banded LD** (`sparsify_ld`, or `ldpred2_by_blocks(..., sparsify=True)`)
  makes `inf` a cheap CG solve and trims the samplers; band by **distance**
  (`max_dist=`) since in-sample LD has a ~1/√N noise floor. See
  [algorithm.md](algorithm.md).
- **Fewer iterations:** `warm_start=True` and adaptive stopping (`tol=`) on the
  samplers (algorithm.md).

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Most variants dropped at **harmonisation** | build/allele-coding mismatch between sumstats and genotypes; check strand and that both are the same genome build |
| Most variants dropped at **QC** | wrong column mapping (so `N`/SE parsed wrong), or a real `N`/MAF/INFO filter; inspect `res.qc_log` |
| **SD-consistency** drops many variants | a wrong/!per-variant `N`, allele errors, or bad imputation — the check compares sumstats-implied SD to the reference (see pipeline.md) |
| `auto` `p_est` looks too high for an **ultra-sparse** trait | `p` is unidentifiable below ~2 causal/1000 variants; `h²`/`r²` stay fine (inference.md) |
| `annot` **underperforms** `auto` | almost always under-converged map — keep `theta_every=1` (the default); raise iterations before raising `theta_every` |
| Sampler **diverges** on noisy/ill-conditioned LD | keep `allow_jump_sign=False` (default), or `sparsify_ld(..., shrink=<1)` to restore diagonal dominance (algorithm.md) |
| It's **slow** | check Numba is installed and the JIT cache is warm (first call compiles); pin BLAS threads for reproducible single-core timing |
