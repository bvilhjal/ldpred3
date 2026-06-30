# Inferring h², polygenicity and predictive r² (LDpred3-auto)

`ldpred3_auto_infer` implements the inference machinery of
[Privé et al. (*AJHG* 2023)](https://doi.org/10.1016/j.ajhg.2023.10.010): it
runs many LDpred3-auto chains from log-spaced `p_init`, drops chains that failed
to converge (keeping those whose fitted effects `R·β̂` vary enough), and pools
the post-burn-in samples to estimate — with credible intervals and **no
validation set** —

* **h²** (SNP heritability) `= βᵀ R β`,
* **p** (polygenicity, the causal fraction), and
* the PRS's **out-of-sample predictive r²**, estimated as `E[b₁ᵀ R b₂]` over
  sampled effect vectors from *different* (independent) chains. If prediction
  were perfect `b₁ = b₂ = β` so `r² = h²`; with no power the draws are
  uncorrelated and `r² ≈ 0`.

```python
from ldpred3 import ldpred3_auto_infer
res = ldpred3_auto_infer(corr, beta_hat, n_eff, n_chains=10)
res.h2_est, res.h2_ci      # heritability + 95% CI
res.p_est,  res.p_ci       # polygenicity + 95% CI
res.r2_est, res.r2_ci      # predicted out-of-sample r² + 95% CI
```

It accepts either a **dense** LD matrix (one block, or a block-diagonal genome
via `block_diagonal_ld`) **or a list of per-block `(R, idx)` matrices**. The
blocks form is *streamed* — chains run one LD block at a time and the genome-wide
LD is never materialised — so it scales to genome-wide SNP counts. Pass
`ncores=k` to run the chains in parallel processes.

## From the pipeline

The end-to-end pipeline can run inference directly on the fitted LD with
`infer=True` (CLI `--infer`), reporting h²/p/r² alongside the scores:

```bash
ldpred3 --sumstats gwas.txt.gz --plink chr1 --infer --out prs.txt
# ... inferred h2=0.41 (0.39, 0.43)  p=0.012 (...)  predictive r2=0.18 (...)
```

```python
res = run_ldpred3_prs("gwas.txt.gz", "chr1", method="auto", infer=True)
res.inference   # {"h2_est", "h2_ci", "p_est", "p_ci", "r2_est", "r2_ci", ...}
```

The pipeline passes the per-block LD straight to the **streaming** inference, so
`--infer` runs genome-wide without assembling a dense LD matrix (the old
`infer_max_variants` cap is no longer enforced and is kept only for backwards
compatibility).

## Validation

Against independent train/test cohorts, the r² inferred from the training
summary statistics alone tracks the PRS's actual R² in the held-out cohort
(e.g. inferred 0.485 vs held-out 0.495 at N=8000), and falls toward 0 as power
drops — the central result of the paper.

## Polygenicity recovery

`p_est` tracks the true causal fraction closely across the realistic range
(within ~10–30 % for true `p` from 0.01 to 0.5):

| true p | ~#causal / 1000 | p_est |
|--------|-----------------|-------|
| 0.01 | 9 | 0.010 |
| 0.05 | 44 | 0.045 |
| 0.20 | 195 | 0.199 |
| 0.50 | 496 | 0.512 |

The exception is the **ultra-sparse limit**: at `p ≈ 0.002` (only ~2 causal
variants in 1000) `p` is essentially unidentifiable — there is too little signal
to estimate a *fraction* — and the estimate is upward-biased and high-variance.
This is inherent to the model, not specific to this implementation; h² and r²
remain well-estimated there.

## Cross-check: LD Score regression

The h² estimate has an independent external check — **LD Score regression**
(LDSC; Bulik-Sullivan et al. 2015), implemented here in `ldpred3.ldsc`. LDSC
fits `E[χ²_j] = intercept + (N·h²/M)·ℓ_j` where `ℓ_j = Σ_k r²_jk` is the variant's
LD score, recovering h² from the slope (the intercept measures confounding and
should be ~1):

```python
from ldpred3 import ld_scores, ldsc_h2
ell = ld_scores(blocks)                       # per-block LD matrices -> LD scores
res = ldsc_h2(n_eff * beta_hat**2, ell, n_eff)   # chi2 = (beta_hat/se)^2
res.h2, res.h2_se, res.intercept              # h2 (+jackknife SE) and confounding
```

Both methods estimate h² from the **same** summary statistics. The benchmark is
**realistic**: the GWAS is generated from the true population LD, but both methods
are fitted with an LD matrix/scores estimated from a finite **reference panel**
(`Nref=2000`) — the mismatch that dominates real-world error. On coalescent LD
(m=6000, N=50000, 5 reps), against the known true h²:

| architecture | h²_true | LDSC | LDpred3-auto |
|--------------|--------:|-----:|-------------:|
| infinitesimal | 0.20 | 0.215 ± 0.020 | 0.218 ± 0.002 |
| infinitesimal | 0.50 | 0.541 ± 0.047 | 0.554 ± 0.001 |
| sparse (p=0.01) | 0.20 | 0.239 ± 0.027 | 0.208 ± 0.004 |
| sparse (p=0.01) | 0.50 | 0.606 ± 0.059 | 0.530 ± 0.009 |

(± is the across-replicate SD.) Takeaways:

- **Reference-panel LD mismatch biases both estimators upward** (e.g. true
  h²=0.5 → ~0.54–0.55). This is the realistic regime — with the *true* LD both
  are essentially unbiased, so the bias is an LD-quality effect, not a flaw in
  either estimator. (LDSC is *more* biased under sparsity, 0.61 at sparse h²=0.5,
  where its infinitesimal `E[χ²]` assumption is most stressed.)
- **LDpred3-auto is much more precise** (often ~10× smaller SD): it uses the full
  LD likelihood, whereas LDSC is a two-parameter moment regression that discards
  most of the information. The trade-off is that its tiny SD makes the LD-mismatch
  bias the dominant error — so treat the point estimate as having a systematic
  component set by the LD reference quality, which the LDSC intercept and the
  across-method agreement help diagnose.

LDSC's value is its **robustness and speed** (a moment regression, no sampling)
and its intercept as a confounding diagnostic; LDpred3-auto's is **efficiency**.
Regenerate with `benchmarks/compare_ldsc_infer.py`.

### Across genetic architectures

Does inference hold up as the architecture changes? Fixing true h²=0.5 and
sweeping four architectures (reference-panel LD `Nref=2000`, m=6000, N=50000, 10
reps), estimating h² with both methods and polygenicity `p` with LDpred3-auto.
Regenerate with `benchmarks/infer_architectures.py`.

| architecture | LDSC h² | LDpred3 h² | p_true | LDpred3 p |
|--------------|--------:|-----------:|-------:|----------:|
| infinitesimal | 0.554 ± 0.060 | 0.547 ± 0.008 | 1.00  | 0.266 ± 0.041 |
| sparse (p=0.01) | 0.545 ± 0.143 | 0.527 ± 0.010 | 0.010 | 0.030 ± 0.004 |
| polygenic (p=0.2) | 0.575 ± 0.043 | 0.544 ± 0.005 | 0.200 | 0.188 ± 0.012 |
| major locus | 0.544 ± 0.188 | 0.548 ± 0.014 | 0.020 | 0.036 ± 0.006 |

- **h² inference is architecture-robust.** LDpred3-auto lands at ~0.53–0.55 for
  *every* architecture, very precisely; the uniform ~0.04 upward bias is the
  reference-panel LD mismatch (the same across architectures), not the
  architecture — so the 95% h² CI under-covers here for the bias reason in
  [interval calibration](#interval-calibration), not because any architecture
  breaks the estimator. LDSC is unbiased-on-average but far noisier, and its SD
  blows up on the sparse and major-locus architectures (0.14–0.19) where its
  infinitesimal `E[χ²]` assumption is most stressed.
- **Polygenicity recovery is architecture-dependent.** `p` is recovered well for
  the **polygenic** trait (0.188 vs 0.20) and order-correct for **sparse**
  (0.030 vs 0.010) and **major-locus** (0.036 vs 0.020, the few huge effects read
  as slightly more variants), but the spike-and-slab cannot represent a truly
  **infinitesimal** trait (p=1) and saturates around 0.27 — read `p` as "how
  concentrated", precise in the mid-polygenic range and a floor estimate at the
  infinitesimal limit.

The same holds for the **genetic correlation** between two traits: `ldsc_rg`
(cross-trait LD Score regression, `E[z₁z₂] = intercept + (√(N₁N₂)·ρ_g/M)·ℓ`)
cross-checks the `r_g` from `ldpred3_auto_bivariate`. Under the same realistic
reference-panel LD both are roughly unbiased and the bivariate sampler is ~2×
more precise (at true r_g=0.9, LDSC 0.86 ± 0.07 vs bivariate LDpred3 0.90 ± 0.04).
See [algorithm.md](algorithm.md#bivariate-two-trait-ldpred3) and
`benchmarks/compare_bivariate_rg.py`.

**Across architectures.** Sweeping true r_g over four architectures (shared
causal variants with bivariate-normal effects; ref-panel LD, m=6000, N₁=50k/N₂=20k,
4 reps; `benchmarks/rg_architectures.py`) — both estimators **track r_g across
every architecture**, with bivariate LDpred3 the more precise:

| architecture | r_g | bivariate LDSC | bivariate LDpred3 |
|--------------|----:|---------------:|------------------:|
| infinitesimal | 0.0 / 0.6 | 0.044 ± 0.090 / 0.625 ± 0.056 | 0.028 ± 0.043 / 0.623 ± 0.030 |
| sparse (p=0.01) | 0.0 / 0.6 | 0.042 ± 0.146 / 0.650 ± 0.106 | 0.006 ± 0.140 / 0.574 ± 0.129 |
| polygenic (p=0.2) | 0.0 / 0.6 | −0.055 ± 0.119 / 0.534 ± 0.081 | −0.026 ± 0.070 / 0.584 ± 0.043 |
| major locus | 0.0 / 0.6 | 0.081 ± 0.177 / 0.652 ± 0.044 | 0.080 ± 0.143 / 0.648 ± 0.057 |

- **r_g is architecture-robust** and **unbiased at r_g=0** (no spurious
  correlation) for both methods — unlike `h²` and `p`, the genetic *correlation*
  largely cancels the LD-mismatch and architecture effects (they hit numerator and
  denominator alike).
- **Bivariate LDpred3 is consistently more precise** (full-likelihood vs the
  moment regression) — e.g. at r_g=0.6, SD ~0.03–0.06 vs LDSC ~0.05–0.11.
- **Sparse traits are the hardest** for both (SD ~0.10–0.15: fewer shared causal
  variants carry the cross-trait signal); the **major-locus** and infinitesimal
  architectures are the most precise (a few large shared effects, or many, anchor
  the estimate).

## Accuracy vs running time

Both axes together, on the realistic reference-panel setup (single core, m=6000,
N₁=50000, N₂=20000, 5 reps; Numba warmed up). `benchmarks/inference_benchmark.py`:

| quantity | method | estimate (truth) | time / run |
|----------|--------|-----------------:|-----------:|
| h² = 0.50 | marginal — no LD | 8.52 ± 0.44 | **0.0001 s** |
| h² = 0.50 | LDSC (`ldsc_h2`) | 0.58 ± 0.13 | 0.03 s |
| h² = 0.50 | LDpred3-auto (`ldpred3_auto_infer`) | 0.52 ± 0.01 | 1.7 s |
| r_g = 0.60 | marginal — no LD | 0.58 ± 0.12 | **0.0001 s** |
| r_g = 0.60 | bivariate LDSC (`ldsc_rg`) | 0.61 ± 0.15 | 0.06 s |
| r_g = 0.60 | bivariate LDpred3 (`ldpred3_auto_bivariate`) | 0.58 ± 0.11 | 0.2 s |

("marginal — no LD" is the naive moment estimator that assumes SNPs are
independent, `h² = (mean χ² − 1)·M/N` and the analogous `r_g`; essentially free.)

- **For h², the LD adjustment is the whole game.** The no-LD estimate is ~17×
  too large (8.5 vs 0.5) because LD makes every causal variant's signal show up
  in its correlated neighbours, which the naive sum double-counts. LDSC (the LD
  scores) removes this for ~0.03 s; LDpred3-auto refines the point estimate
  further at a real time cost.
- **For r_g, LD matters far less.** The no-LD estimate (0.58 ± 0.12) is already
  good — about as tight as LDSC — because LD inflates the cross-covariance and
  both heritabilities *proportionally* and largely cancels in the ratio. So a
  fast marginal r_g is a reasonable first pass, where a marginal h² is useless.
- **The LDpred3 estimators are the most precise** (several-fold smaller SD than
  LDSC — up to ~13× for h²; the r_g gap is smaller)
  at a time cost: in this timing both run many MCMC chains, so they are slower
  than the moment regressions. (The h² timing above used the dense path; passing
  per-block LD makes `ldpred3_auto_infer` **stream** like the bivariate sampler,
  removing the dense `O(m²)` cost at genome scale.)

Use a marginal pass for a quick `r_g` sanity check, LDSC for a fast LD-correct h²
and the confounding intercept, and the LDpred3 estimators when precision matters
(reading their point estimates with the LD-mismatch bias in mind).

### Dense vs streaming inference

`ldpred3_auto_infer` accepts either a dense LD matrix or per-block `(R, idx)`.
The dense path is `O(m²)` per sweep; the streaming (blocks) path is
`O(m · block_size)`, so the gap widens with `m`. Same data, 6 chains, single core
(`benchmarks/infer_scaling.py`):

| m | dense (s) | streaming (s) | speed-up |
|------:|----------:|--------------:|---------:|
| 2000  | 0.29 | 0.11 | 2.6× |
| 4000  | 0.62 | 0.17 | 3.7× |
| 8000  | 1.91 | 0.30 | 6.4× |
| 16000 | 8.33 | 0.57 | **14.6×** |

Dense roughly **quadruples per doubling** of `m` (quadratic); streaming roughly
**doubles** (linear). The two agree on h² (≈0.52–0.55 here). By 16k SNPs dense is
already 8 s and a dense genome-wide matrix is infeasible (memory and time),
whereas streaming stays linear — so prefer the blocks form (which the pipeline's
`--infer` now uses by default) at any non-trivial size.

The blocks may also be **compact**: banded `SparseLD` (`O(k·bandwidth)`) or
low-rank `LowRankLD` (`O(k·rank)` eigenspace). `ldpred3_auto_infer` fits these in
their native representation — the sampler and the cross-chain r² products use the
banded CSR / eigenvectors directly, never densifying to `k × k` — so **h²/p/r²
inference scales the same way scoring does**. With `--infer`, `--ld-lowrank` (or
`--ld-sparse`) therefore caps memory at the per-block representation while still
returning the heritability, polygenicity and predictive-r² estimates. (`shrink_corr`
is only defined on dense blocks, so it is rejected with a compact LD.)

## Interval calibration

Do the nominal 95% intervals actually cover the truth 95% of the time? Coverage
over 40 replicates, under clean LD and under reference-panel LD
(`benchmarks/calibration.py`):

| 95% interval (truth) | clean LD | reference-panel LD |
|----------------------|---------:|-------------------:|
| LDpred3-auto h² (0.50) | 0.97 | **0.00** |
| LDpred3-auto p (0.01) | 0.82 | **0.00** |
| LDSC h² (0.50) | 0.90 | 0.93 |
| LDSC r_g (0.50) | 0.70 | 0.72 |

The headline is a real caution: **`ldpred3_auto_infer`'s intervals are
well-calibrated only when the LD matches.** Under a realistic reference panel its
coverage collapses to 0 — the LD-mismatch *bias* (≈0.04 for h²) dwarfs the
posterior SE (≈0.01), so the tight interval never reaches the truth. Treat the
LDpred3-auto interval as **precision, not accuracy**: it captures Monte-Carlo /
sampling uncertainty but not the systematic error set by the LD reference.
**LDSC's wider intervals stay honest** for h² (~0.9 in both conditions) because
they absorb that bias; its `r_g` interval under-covers somewhat (~0.7), so widen
it in practice. The robust uncertainty signal is **cross-method agreement** (and
the LDSC intercept), not the LDpred3 interval width.

## Sample overlap

Overlapping GWAS samples correlate the two traits' sampling noise, which inflates
a naive genetic correlation even when the traits are genetically independent.
Both estimators have a correction — LDSC a free cross-trait *intercept*,
`ldpred3_auto_bivariate` a `cross_corr` parameter — and both work
(`benchmarks/sample_overlap.py`, noise correlation ρ_e=0.5, N=10000, h²=0.3):

| true r_g | LDSC, intercept=0 | LDSC, free intercept | bivariate, cross_corr=0 | bivariate, cross_corr=ρ_e |
|---------:|------------------:|---------------------:|------------------------:|--------------------------:|
| 0.0 | 0.090 | −0.036 | 0.031 | −0.019 |
| 0.5 | 0.562 | 0.505 | 0.547 | 0.496 |

Uncorrected, overlap biases `r_g` upward (≈+0.06–0.09 at r_g=0); the free LDSC
intercept and the bivariate `cross_corr=ρ_e` each remove it. If your two GWAS
share samples, leave the LDSC intercept free and pass `cross_corr` (the
overlap-induced noise correlation, ≈ the cross-trait LDSC intercept) to the
bivariate sampler.
