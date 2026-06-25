# Inferring h², polygenicity and predictive r² (LDpred2-auto)

`infer.ldpred2_auto_infer` implements the inference machinery of
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
from infer import ldpred2_auto_infer
res = ldpred2_auto_infer(corr, beta_hat, n_eff, n_chains=10)
res.h2_est, res.h2_ci      # heritability + 95% CI
res.p_est,  res.p_ci       # polygenicity + 95% CI
res.r2_est, res.r2_ci      # predicted out-of-sample r² + 95% CI
```

It operates on a dense LD matrix (one block, or a block-diagonal genome via
`block_diagonal_ld`).

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
