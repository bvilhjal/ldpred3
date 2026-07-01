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
[fine-mapping](#8-fine-mapping-which-variants-are-causal) ·
[reuse work](#9-re-using-work-saved-weights--cached-ld) ·
[scaling](#10-scaling--performance) · [troubleshooting](#11-troubleshooting)

> **In one line:** `pip install . numba` then
> `ldpred3 --sumstats gwas.txt.gz --plink target --out prs.txt`
> (`auto` is the default model). Everything below is detail on the inputs, the
> model choice, and reading the output.

**Common recipes** — find your task, copy the command, read the linked section for
detail. All build on the base command `ldpred3 --sumstats gwas.txt.gz --plink target`:

| You want to… | Add / change | More |
|--------------|--------------|------|
| Score a cohort (the default) | *(nothing — `auto` runs)* | [§2](#2-the-one-command-path-recommended) |
| Sanity-check inputs before a long run | `--dry-run` | [§2](#2-the-one-command-path-recommended) |
| Also estimate h² / polygenicity / r² | `--infer` | [§7](#7-inferring-h-polygenicity-and-predictive-r-no-validation-set) |
| Fine-map causal variants (PIPs + credible sets) | `--finemap` | [§8](#8-fine-mapping-which-variants-are-causal) |
| Use functional annotations | `--method annot --annotations annot.tsv` | [§6](#6-annotation-informed-prs-annot) |
| Fit once, then score more cohorts cheaply | `--save-weights w.txt`, then `--weights w.txt` | [§9](#9-re-using-work-saved-weights--cached-ld) |
| Make scores comparable **across** cohorts | `--weights w.txt --scaling frozen` | [§9](#9-re-using-work-saved-weights--cached-ld) |
| Cache LD to speed up re-runs | `--ld-out ld.npz` once, then `--ld-cache ld.npz` | [§9](#9-re-using-work-saved-weights--cached-ld) |
| Scale to millions of SNPs (sequencing) | `--ld-lowrank --ld-lowrank-min-size 1000 --ld-stream` | [§10](#10-scaling--performance) |

## 1. What you need

| Input | What it is | Notes |
|-------|------------|-------|
| **GWAS summary statistics** | per-variant marginal effect, SE (or p), allele(s), N | one text/`.gz` table; flexible column names (`sumstats` parses aliases, `OR→β`, SE-from-p) |
| **Target genotypes** | the individuals you want to score | PLINK `.bed/.bim/.fam` or BGEN v1.2 |
| **LD reference** | correlation between variants | by default computed **in-sample** from the target; or pass an external panel |

You do **not** need a validation/tuning cohort: `auto` self-tunes its
hyper-parameters, and `ldpred3_auto_infer` even estimates predictive r² without
one.

**LD reference & ancestry — the gotcha that quietly breaks PRS.** LDpred3's whole
job is to undo LD, so the LD reference must match the GWAS *and* the target
**ancestry**. Practical guidance:

- **In-sample LD (default)** — computed from the target — is fine when the target
  is reasonably large (thousands of individuals) and the same ancestry as the
  GWAS. It is the simplest path.
- **External panel** (`ld_prefix=` / `--ld-prefix`) — use a matched-ancestry
  reference (e.g. the relevant 1000 Genomes superpopulation) when the target is
  small, so the LD isn't estimated from a handful of people.

## 2. The one-command path (recommended)

```bash
ldpred3 --sumstats gwas.txt.gz --plink target --method auto --out prs.txt
```

This runs the whole pipeline: QC → harmonise alleles → per-block LD → LDpred3 →
one score per individual. The equivalent in Python, with the logs you should
check:

```python
from ldpred3 import run_ldpred3_prs
res = run_ldpred3_prs("gwas.txt.gz", "target", method="auto")

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
ldpred3 --sumstats gwas.txt.gz --plink target --dry-run
# detected columns: id=SNP, ea=A1, oa=A2, beta=BETA, se=SE, n_eff=N
# matched 31204 / 31875 to the target (12 flipped, 41 ambiguous, ...)
```

If a required column is mis-detected, pass it explicitly — by column name or
index — and re-check:

```python
run_ldpred3_prs("gwas.txt.gz", "target",
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
[§9](#9-re-using-work-saved-weights--cached-ld) to save and reuse them).

### Binary (case/control) traits

Two scale choices that quietly cost accuracy if skipped:

- **Effective sample size.** Pass the *effective* N, not the raw total —
  `--n-cases NCASE --n-controls NCONTROL` (or `ldpred3.n_eff_case_control`)
  computes `4/(1/Ncase + 1/Ncontrol)`. This is what the LDpred likelihood (and
  the `--impute-n` N-recovery) needs.
- **Liability-scale heritability.** A 0/1 phenotype gives *observed*-scale h²;
  convert it to the comparable, prevalence-aware liability scale with
  `ldpred3.h2_liability(h2_obs, prevalence, prop_cases=…)` (Lee et al. 2011).

For interpretable output, `--prs-percentiles` (or `ldpred3.standardize_prs`) adds
each individual's standardized PRS (Z) and percentile.

## 3. Is the PRS any good? (evaluating)

A PRS is only useful if it predicts. Two ways to judge it:

- **With a phenotype** (the gold standard): correlate the scores with a measured
  trait in the target or a held-out set. The squared Pearson correlation
  `cor(PRS, phenotype)²` is the realised prediction R² (for a binary trait, use
  the AUC or Nagelkerke R²). Always assess in individuals **not** in the GWAS.
- **Without a phenotype:** `--infer` / `ldpred3_auto_infer` estimates the PRS's
  expected out-of-sample r² from the summary statistics alone, with a credible
  interval ([§7](#7-inferring-h-polygenicity-and-predictive-r-no-validation-set)).
  It tracks the realised R² closely and falls toward 0 as GWAS power drops, so
  it's a useful a-priori check before you have outcome data.

If accuracy is far below the inferred r², suspect an input problem (ancestry
mismatch, wrong `N`, or LD that doesn't match the target) rather than the model.

## 4. Choosing a model

**Use `auto` unless you have a specific reason not to** — it self-tunes `h²` and
`p` and matches the oracle `grid` across architectures, with no tuning cohort.
Branch off it only for the cases below:

```
start →  auto   (self-tuning point-normal; the right answer for most traits)
           │
           ├─ trait is truly infinitesimal, or you want a cheap baseline   →  inf
           ├─ you already know h² and p (e.g. from a previous fit)         →  grid
           ├─ you have trustworthy per-SNP functional annotations          →  annot
           └─ want a sparse / lasso-style alternative to compare against   →  lassosum2  or  laplace
```

Every method takes standardized marginal effects + LD and returns adjusted
(posterior) effects; they differ in the **prior** on the true effects.

| Model | What it is | When to use | Tunes |
|-------|------------|-------------|-------|
| **`auto`** | point-normal (spike-and-slab) prior, `h²` & `p` self-tuned by the Gibbs sampler | **the default** — robust across architectures, no tuning cohort | `h²`, `p` |
| **`inf`** | infinitesimal (Gaussian) prior — *all* variants causal | a truly infinitesimal trait, or the cheapest, most robust baseline | — (needs `h²`) |
| **`grid`** | point-normal at **fixed** `h²`, `p` | you already know the hyper-parameters (e.g. from a previous `auto` fit) | — (needs `h²`, `p`) |
| **`annot`** | `auto` + an SBayesRC-style **learned** functional prior `p_j = σ(aⱼᵀθ)` | you have per-SNP annotations (coding, conserved, enhancers, …) | `h²`, per-annotation `θ` |
| **`lassosum2`** | L1-penalised regression (Laplace-prior *mode*); a `(shrink, λ)` grid picked by pseudo-validation | a sparse, no-MCMC complement — keep whichever of it / `auto` pseudo-validates better | `shrink`, `λ` (grid) |
| **`laplace`** | Bayesian lasso — Laplace-prior posterior **mean** via Gibbs, self-tuning the shrinkage | a robust dense-shrinkage alternative; the Bayesian counterpart of `lassosum2` | `λ` |

Empirically (see [benchmarks.md](benchmarks.md)): the raw marginal PRS is always
far behind; `inf` is robust but flat and only wins under a truly infinitesimal
architecture; `grid`/`auto` win decisively on sparse / major-locus traits; `auto`
matches the oracle `grid` with no tuning; `annot` matches `auto` when the
annotation is uninformative and beats it when it carries signal; `lassosum2` and
`laplace` are competitive sparse alternatives — no single method dominates every
trait, so comparing `auto` against one of them (they need no validation cohort)
is cheap insurance.

**Two modifiers that layer on `auto`/`grid`** (not separate methods):

- **`--auto-chains N`** (Privé 2023 robust auto): run `N` independent chains,
  drop the ones that fail a consistency filter, and average the rest — a more
  stable PRS at low power. `N≈10` is typical; the default `1` is a single chain.
- **`--alpha A`** (MAF-dependent prior): let the causal-effect variance scale as
  `[2f(1−f)]^(1+A)` instead of being flat. `A=−1` (default) is the standard
  LDpred model; `A<−1` up-weights rarer variants (the signature of negative
  selection). See [algorithm.md](algorithm.md#maf-dependent-slab-variance-alpha).

## 5. Working from your own LD blocks (library path)

If you already have summary stats and an LD matrix for a region (or a whole
genome split into blocks), skip the pipeline and call the model directly. Effects
are on the **standardized scale** — `standardize_betas` converts reported GWAS
effects to it and gives you the back-transform:

```python
import numpy as np
from ldpred3 import standardize_betas, ldpred3_auto, ldpred3_by_blocks

# one block: beta/beta_se/n_eff are GWAS stats, corr is the (m x m) LD matrix
beta_hat, scale = standardize_betas(beta, beta_se, n_eff)
res = ldpred3_auto(corr, beta_hat, n_eff)
adjusted_beta = res.beta_est * scale          # back to the input (per-allele) scale
print(res)                                    # AutoResult(h2_est=…, p_est=…)

# genome-wide: blocks is a list of (corr_block, index_array) tiling 0..m-1
beta_est = ldpred3_by_blocks(blocks, beta_hat, n_eff, method="auto")
```

`ldpred3_by_blocks(method="auto")` streams blocks one at a time and pools `h²`/`p`
globally, so the genome-wide LD is never materialised (this is the path that
scales to millions of SNPs). Build blocks with `block_diagonal_ld` or, better,
`optimal_ld_blocks` (cuts in recombination valleys — see [algorithm.md](algorithm.md)).

**Two correlated traits?** `ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2,
n1, n2)` fits both jointly, learning their genetic correlation so a well-powered
trait sharpens a weaker correlated one (and reporting `res.rg`, `res.h2`). See
[algorithm.md](algorithm.md#bivariate-two-trait-ldpred3).

## 6. Annotation-informed PRS (`annot`)

Supply a per-SNP annotation table and the sampler learns an SBayesRC-style
functional prior `p_j = sigmoid(a_jᵀθ)` genome-wide, returning the learned
enrichment:

```bash
ldpred3 --sumstats gwas.txt.gz --plink target --method annot \
    --annotations annot.tsv --out prs.txt
# ... annotation enrichment: coding=+1.20, conserved=+0.80, ...
```

```python
res = run_ldpred3_prs("gwas.txt.gz", "target", method="annot",
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
from ldpred3 import ldpred3_auto_infer
res = ldpred3_auto_infer(corr, beta_hat, n_eff, n_chains=10)
res.h2_est, res.h2_ci     # heritability + 95% CI
res.p_est,  res.p_ci      # polygenicity + 95% CI
res.r2_est, res.r2_ci     # predicted out-of-sample r² + 95% CI
```

or from the pipeline with `infer=True` / `--infer`. Inference **streams the LD
blocks**, so it runs genome-wide (no dense matrix is assembled). For an
independent h² check, `ldsc_h2` runs **LD Score regression** on the same summary
statistics (it agrees with the LDpred3-auto h² but is less precise — see
[inference.md](inference.md)). Full method and validation in
[inference.md](inference.md).

## 8. Fine-mapping: which variants are causal?

A PRS predicts a phenotype; **fine-mapping localises the signal** — at a
GWAS-significant locus, which SNP(s) actually drive it? LDpred3's spike-and-slab
sampler already gives each SNP its posterior probability of being causal (the
**PIP**), so fine-mapping reuses the same model, LD and QC as the PRS — no new
inputs.

```bash
# whole genome (every LD block) -> fm.pip.tsv + fm.cs.tsv
ldpred3 --finemap --sumstats gwas.txt.gz --plink target --out fm

# only loci around genome-wide-significant hits (faster; the usual workflow)
ldpred3 --finemap --sumstats gwas.txt.gz --plink target \
        --finemap-only-significant 5e-8 --out fm

# restrict to your own regions (BED: chrom start end [name])
ldpred3 --finemap --sumstats gwas.txt.gz --plink target --regions loci.bed --out fm
```

Two tables come out:

| file | one row per | key columns |
|------|-------------|-------------|
| `fm.pip.tsv` | variant | `pip`, posterior mean/SD, marginal `z` |
| `fm.cs.tsv` | credible set | `coverage`, `lead_variant`, `purity_min_abs_r`, member `variants` |

The **credible set** is the deliverable: the smallest set of variants that
contains the causal one with 95% probability. It is *calibrated* — a 95% set
contains the true causal variant ~95% of the time — whereas raw PIP values are
prior-dependent, so read the sets, not the absolute PIPs. A **purity** score
(min |r| among members) flags sets diluted by LD; tightly-linked proxies the data
cannot tell apart are kept together.

In Python:

```python
from ldpred3 import run_finemap
res = run_finemap("gwas.txt.gz", "target", only_significant=5e-8, out="fm")
for cs in res.credible_sets:
    print(cs.lead_variant, round(cs.lead_pip, 3), len(cs.variants))
```

Already have summary statistics and an LD matrix? Skip the files and call the
locus fine-mapper directly — `ldpred3_pip(corr, beta_hat, n_eff)` for one locus,
`finemap_by_blocks(blocks, beta_hat, n_eff)` genome-wide. Full reference and the
coverage benchmark: [finemap.md](finemap.md).

## 9. Re-using work: saved weights & cached LD

Two flags avoid redoing the expensive parts when you score more cohorts or
re-run with different settings.

**Save the fitted weights, then score new cohorts for free.** The weights (one
standardized effect per variant, with its allele) are the reusable product —
scoring another cohort from them skips LD construction and the whole LDpred3 fit:

```bash
ldpred3 --sumstats gwas.txt.gz --plink discovery --save-weights prs.weights.txt --out d.txt
ldpred3 --plink new_cohort --weights prs.weights.txt --out new.txt   # no sumstats / LD / refit
```

```python
res.write_weights("prs.weights.txt")                 # from a PRSResult
score_from_weights("prs.weights.txt", "new_cohort")  # -> ScoreResult
```

Weights are harmonised to each new target's alleles (sign-flipped where alleles
are swapped), so a cohort with the opposite A1/A2 coding still scores correctly.

**Comparing scores across cohorts? Freeze the scale.** A standardized PRS is
z-scored by each cohort's own allele frequencies, so the *same* weights give
scores on slightly different scales in two cohorts — fine for ranking people
*within* one cohort, but not for comparing a value between cohorts. Save the
weights (they then carry the fit cohort's `AF_REF`/`SD_REF`) and score with
`--scaling frozen` to reuse that one fixed scale everywhere:

```bash
ldpred3 --plink cohortB --weights prs.weights.txt --scaling frozen --out b.txt
```

```python
score_from_weights("prs.weights.txt", "cohortB", scaling="frozen")
```

Default is `scaling="target"` (each cohort's own) — switch to `frozen` only when
absolute, cross-cohort-comparable scores matter.

**Cache the LD blocks** so re-runs (e.g. trying `grid` vs `auto`, or sweeping QC)
don't recompute LD:

```bash
ldpred3 --sumstats gwas.txt.gz --plink target --method auto --ld-out ld.npz   --out auto.txt
ldpred3 --sumstats gwas.txt.gz --plink target --method grid --ld-cache ld.npz --out grid.txt
```

The cache records the variant set it was built for; if your inputs/QC change so
the harmonised variants differ, the run stops with a clear error rather than
silently using stale LD — rebuild it with `--ld-out`.

## 10. Scaling & performance

- **Install Numba** (`pip install numba`). The inner sampler is JIT-compiled and
  cached; without it you get identical results but a much slower pure-Python
  loop — fine for CI, not for genome-wide runs.
- **Memory** is dominated by the LD: the dense per-block matrices (`Σ kᵦ²`
  float32) are all held in RAM — fine for a curated ~2M-SNP panel (~4 GB), but it
  blows up when blocks reach thousands of SNPs.
- **Scaling to millions of SNPs (sequencing).** Use the composable LD levers —
  recombination-aware **block splitting** to bound block sizes, **low-rank LD**
  (`--ld-lowrank`) which fits huge blocks in ~¼ the memory *at matched accuracy on
  realistic LD*, a dense/low-rank **mixed** policy (`--ld-lowrank-min-size`) that
  compresses only the big blocks, and **on-disk streaming** (`--ld-stream`) for an
  LD larger than RAM. Full recipe and trade-offs:
  [pipeline.md → Scaling](pipeline.md#scaling-to-millions-of-snps). Note the
  honest trade-off: the compact representations cut memory but fit *slower* — they
  are for LD that would not fit dense, not a speed-up.
- **Sparse / banded LD** (`--ld-sparse` / `sparsify_ld`) makes `inf` a cheap CG
  solve and is a memory option for **genuinely banded** (array-like) LD — but on
  realistic LD distance banding **discards real long-range structure and loses
  accuracy**, so prefer `--ld-lowrank` there. Band by **distance** (`max_dist=`)
  since in-sample LD has a ~1/√N noise floor. See [algorithm.md](algorithm.md).
- **Fewer iterations:** `warm_start=True` and adaptive stopping (`tol=`) on the
  samplers (algorithm.md).

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Most variants dropped at **harmonisation** | build/allele-coding mismatch between sumstats and genotypes; check strand and that both are the same genome build |
| Most variants dropped at **QC** | wrong column mapping (so `N`/SE parsed wrong), or a real `N`/MAF/INFO filter; inspect `res.qc_log` |
| **SD-consistency** drops many variants | a wrong/!per-variant `N`, allele errors, or bad imputation — the check compares sumstats-implied SD to the reference (see pipeline.md) |
| Spurious **genome-wide hits** that disagree with their LD neighbours | likely allele/strand errors or an LD-reference mismatch the SD-check misses; try `--dentist` (off by default — it can also drop genuine poorly-tagged signals, so keep its cutoff stringent; see pipeline.md) |
| `auto` `p_est` looks too high for an **ultra-sparse** trait | `p` is unidentifiable below ~2 causal/1000 variants; `h²`/`r²` stay fine (inference.md) |
| `annot` **underperforms** `auto` | almost always under-converged map — keep `theta_every=1` (the default); raise iterations before raising `theta_every` |
| Sampler **diverges** on noisy/ill-conditioned LD | keep `allow_jump_sign=False` (default), or `sparsify_ld(..., shrink=<1)` to restore diagonal dominance (algorithm.md); `--ld-shrink` shrinks large noisy blocks toward the identity |
| **Out of memory** at genome / sequencing scale | the dense LD (`Σ kᵦ²`) is too big — use `--ld-lowrank` (low-rank LD, ~¼ memory at matched accuracy), `--ld-lowrank-min-size 1000` (compress only big blocks), and `--ld-out cache.npz --ld-stream` to stream the LD from disk ([scaling](pipeline.md#scaling-to-millions-of-snps)) |
| It's **slow** | check Numba is installed and the JIT cache is warm (first call compiles); pin BLAS threads for reproducible single-core timing; note `--ld-lowrank`/`--ld-sparse` trade fit speed for memory (use them only when dense won't fit) |
