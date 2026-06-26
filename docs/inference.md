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

It operates on a dense LD matrix (one block, or a block-diagonal genome via
`block_diagonal_ld`). Pass `ncores=k` to run the chains in parallel processes.

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

Because the estimator is dense, the pipeline assembles a dense block-diagonal LD
and guards on size (`infer_max_variants`, default 30000) — so use it at
**chromosome / curated-SNP scale**. Streaming genome-wide inference (millions of
SNPs, as in bigsnpr's SFBM) is a future extension.

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

Both methods estimate h² from the **same** summary statistics. On coalescent LD
(m=6000, N=50000, 5 reps), against the known true h²:

| architecture | h²_true | LDSC | LDpred2-auto |
|--------------|--------:|-----:|-------------:|
| infinitesimal | 0.20 | 0.196 ± 0.017 | 0.199 ± 0.003 |
| infinitesimal | 0.50 | 0.494 ± 0.040 | 0.499 ± 0.004 |
| sparse (p=0.01) | 0.20 | 0.215 ± 0.031 | 0.199 ± 0.003 |
| sparse (p=0.01) | 0.50 | 0.543 ± 0.071 | 0.497 ± 0.005 |

(± is the across-replicate SD.) Two takeaways:

- **They agree with the truth and with each other** — an independent validation
  of the LDpred2-auto h². LDSC's intercept stays ~1 (no confounding simulated).
- **LDpred2-auto is far more precise** (≈5–15× smaller SD): it uses the full LD
  likelihood, whereas LDSC is a two-parameter moment regression that discards
  most of the information. LDSC also degrades more under **sparsity** (SD 0.071
  and a slight upward bias at sparse h²=0.5), because its infinitesimal
  `E[χ²]` assumption is stressed and a few large effects lever the slope —
  LDpred2-auto's spike-and-slab matches the architecture and stays tight.

LDSC's value is its **robustness and speed** (a moment regression, no sampling)
and its intercept as a confounding diagnostic; LDpred2-auto's is **efficiency**.
Regenerate with `benchmarks/compare_ldsc_infer.py`.
