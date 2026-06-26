# Inferring h², polygenicity and predictive r² (LDpred2-auto)

`ldpred2_auto_infer` implements the inference machinery of
[Privé et al. (*AJHG* 2023)](https://doi.org/10.1016/j.ajhg.2023.10.010): it
runs many LDpred2-auto chains from log-spaced `p_init`, drops chains that failed
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
from pyldpred2 import ldpred2_auto_infer
res = ldpred2_auto_infer(corr, beta_hat, n_eff, n_chains=10)
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
pyldpred2-prs --sumstats gwas.txt.gz --plink chr1 --infer --out prs.txt
# ... inferred h2=0.41 (0.39, 0.43)  p=0.012 (...)  predictive r2=0.18 (...)
```

```python
res = run_ldpred2_prs("gwas.txt.gz", "chr1", method="auto", infer=True)
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
(LDSC; Bulik-Sullivan et al. 2015), implemented here in `pyldpred2.ldsc`. LDSC
fits `E[χ²_j] = intercept + (N·h²/M)·ℓ_j` where `ℓ_j = Σ_k r²_jk` is the variant's
LD score, recovering h² from the slope (the intercept measures confounding and
should be ~1):

```python
from pyldpred2 import ld_scores, ldsc_h2
ell = ld_scores(blocks)                       # per-block LD matrices -> LD scores
res = ldsc_h2(n_eff * beta_hat**2, ell, n_eff)   # chi2 = (beta_hat/se)^2
res.h2, res.h2_se, res.intercept              # h2 (+jackknife SE) and confounding
```

Both methods estimate h² from the **same** summary statistics. The benchmark is
**realistic**: the GWAS is generated from the true population LD, but both methods
are fitted with an LD matrix/scores estimated from a finite **reference panel**
(`Nref=2000`) — the mismatch that dominates real-world error. On coalescent LD
(m=6000, N=50000, 5 reps), against the known true h²:

| architecture | h²_true | LDSC | LDpred2-auto |
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
- **LDpred2-auto is much more precise** (often ~10× smaller SD): it uses the full
  LD likelihood, whereas LDSC is a two-parameter moment regression that discards
  most of the information. The trade-off is that its tiny SD makes the LD-mismatch
  bias the dominant error — so treat the point estimate as having a systematic
  component set by the LD reference quality, which the LDSC intercept and the
  across-method agreement help diagnose.

LDSC's value is its **robustness and speed** (a moment regression, no sampling)
and its intercept as a confounding diagnostic; LDpred2-auto's is **efficiency**.
Regenerate with `benchmarks/compare_ldsc_infer.py`.

The same holds for the **genetic correlation** between two traits: `ldsc_rg`
(cross-trait LD Score regression, `E[z₁z₂] = intercept + (√(N₁N₂)·ρ_g/M)·ℓ`)
cross-checks the `r_g` from `ldpred2_auto_bivariate`. Under the same realistic
reference-panel LD both are roughly unbiased and the bivariate sampler is ~2×
more precise (at true r_g=0.9, LDSC 0.86 ± 0.07 vs bivariate LDpred2 0.90 ± 0.04).
See [algorithm.md](algorithm.md#bivariate-two-trait-ldpred2) and
`benchmarks/compare_bivariate_rg.py`.

## Accuracy vs running time

Both axes together, on the realistic reference-panel setup (single core, m=6000,
N₁=50000, N₂=20000, 5 reps; Numba warmed up). `benchmarks/inference_benchmark.py`:

| quantity | method | estimate (truth) | time / run |
|----------|--------|-----------------:|-----------:|
| h² = 0.50 | marginal — no LD | 9.60 ± 1.01 | **0.0001 s** |
| h² = 0.50 | LDSC (`ldsc_h2`) | 0.65 ± 0.16 | 0.03 s |
| h² = 0.50 | LDpred2-auto (`ldpred2_auto_infer`) | 0.54 ± 0.01 | 4.8 s |
| r_g = 0.60 | marginal — no LD | 0.62 ± 0.10 | **0.0001 s** |
| r_g = 0.60 | bivariate LDSC (`ldsc_rg`) | 0.57 ± 0.17 | 0.07 s |
| r_g = 0.60 | bivariate LDpred2 (`ldpred2_auto_bivariate`) | 0.63 ± 0.08 | 0.4 s |

("marginal — no LD" is the naive moment estimator that assumes SNPs are
independent, `h² = (mean χ² − 1)·M/N` and the analogous `r_g`; essentially free.)

- **For h², the LD adjustment is the whole game.** The no-LD estimate is ~19×
  too large (9.6 vs 0.5) because LD makes every causal variant's signal show up
  in its correlated neighbours, which the naive sum double-counts. LDSC (the LD
  scores) removes this for ~0.03 s; LDpred2-auto refines the point estimate
  further at a real time cost.
- **For r_g, LD matters far less.** The no-LD estimate (0.62 ± 0.10) is already
  good — even tighter than LDSC — because LD inflates the cross-covariance and
  both heritabilities *proportionally* and largely cancels in the ratio. So a
  fast marginal r_g is a reasonable first pass, where a marginal h² is useless.
- **The LDpred2 estimators are the most precise** (≈5–15× smaller SD than LDSC)
  at a time cost: in this timing both run many MCMC chains, so they are slower
  than the moment regressions. (The h² timing above used the dense path; passing
  per-block LD makes `ldpred2_auto_infer` **stream** like the bivariate sampler,
  removing the dense `O(m²)` cost at genome scale.)

Use a marginal pass for a quick `r_g` sanity check, LDSC for a fast LD-correct h²
and the confounding intercept, and the LDpred2 estimators when precision matters
(reading their point estimates with the LD-mismatch bias in mind).

## Interval calibration

Do the nominal 95% intervals actually cover the truth 95% of the time? Coverage
over 40 replicates, under clean LD and under reference-panel LD
(`benchmarks/calibration.py`):

| 95% interval (truth) | clean LD | reference-panel LD |
|----------------------|---------:|-------------------:|
| LDpred2-auto h² (0.50) | 0.97 | **0.00** |
| LDpred2-auto p (0.01) | 0.82 | **0.00** |
| LDSC h² (0.50) | 0.90 | 0.93 |
| LDSC r_g (0.50) | 0.70 | 0.72 |

The headline is a real caution: **`ldpred2_auto_infer`'s intervals are
well-calibrated only when the LD matches.** Under a realistic reference panel its
coverage collapses to 0 — the LD-mismatch *bias* (≈0.04 for h²) dwarfs the
posterior SE (≈0.01), so the tight interval never reaches the truth. Treat the
LDpred2-auto interval as **precision, not accuracy**: it captures Monte-Carlo /
sampling uncertainty but not the systematic error set by the LD reference.
**LDSC's wider intervals stay honest** for h² (~0.9 in both conditions) because
they absorb that bias; its `r_g` interval under-covers somewhat (~0.7), so widen
it in practice. The robust uncertainty signal is **cross-method agreement** (and
the LDSC intercept), not the LDpred2 interval width.

## Sample overlap

Overlapping GWAS samples correlate the two traits' sampling noise, which inflates
a naive genetic correlation even when the traits are genetically independent.
Both estimators have a correction — LDSC a free cross-trait *intercept*,
`ldpred2_auto_bivariate` a `cross_corr` parameter — and both work
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
