# LD-based summary-statistic imputation (experimental)

A variant missing from the GWAS but present in the LD reference can have its
**standardized marginal effect** imputed from its typed neighbours. Under
LDpred3's model `β̂ = R·β + N(0, R/N)`, the missing statistic is a Gaussian
conditional mean, computed **per LD block** (so it streams on-the-fly):

```text
β̂_u   = R_ut · R_tt⁻¹ · β̂_t            (t = typed, u = untyped, within a block)
imp_r²_u = diag(R_ut · R_tt⁻¹ · R_tu)  ∈ [0, 1]   (imputation quality)
```

The imputed statistic is a **linear combination of the typed ones — it carries no
new information** — so it is **down-weighted** by its quality: the imputed variant
enters the sampler with effective sample size `N_u = N · imp_r²_u`. This is a
pure pre-processing layer ([`ldpred3/impute.py`](../ldpred3/impute.py)); the core
sampler is untouched.

```python
from ldpred3 import impute_sumstats_blocks, ldpred3_auto_annot_blocks

imp = impute_sumstats_blocks(beta_hat, blocks, typed_mask, n_eff=N)
beta = ldpred3_auto_annot_blocks(blocks, imp.beta_hat, imp.n_eff, A)  # annot prior
```

## Why it helps

It does two distinct things, both real (benchmark below):

1. **Corrects model misspecification → a prediction gain.** Dropping variants
   that are in LD with the typed ones and contribute to their marginals fits the
   wrong generative model (`β̂_t = R_tt·β_t + R_tu·β_u + ε`, but a typed-only fit
   uses only `R_tt`). Imputing the missing variants back restores the correct
   `R_full`, so the sampler attributes the signal properly. (This is why
   summary-stat methods want the sumstats aligned to the dense LD-reference
   variant set, not just the GWAS-typed subset.)
2. **Enables attribution — and this is where functional annotations matter.** An
   imputed statistic alone cannot say *which* of a set of LD-equivalent variants
   is causal. A **functional annotation** can: once the untyped functional variant
   is in the model (via imputation), the annotation-informed prior
   (`ldpred3_auto_annot_blocks`) pulls effect onto it rather than smearing it over
   the non-functional tags. This localises untyped causals — and is the mechanism
   behind cross-ancestry portability gains (the causal functional variant is often
   shared across populations even when its LD tags are not).

## Benchmark

`benchmarks/impute_annot.py` (coalescent LD, m=6000, causals enriched 12× in a
20% functional annotation, then the **functional causals are dropped from the
GWAS** — untyped). Four pipelines × matched/mismatched LD, scored for genetic R²
(prediction) and localisation (fraction of untyped functional causals that end up
the top |effect| in their LD neighbourhood):

| pipeline | matched R² | matched loc | mismatched R² | mismatched loc |
|----------|-----------:|------------:|--------------:|---------------:|
| drop / auto | 0.952 | 0.03 | 0.918 | 0.03 |
| drop / annot | 0.950 | 0.03 | 0.918 | 0.03 |
| impute / auto | 0.979 | 0.46 | 0.933 | 0.23 |
| **impute / annot** | **0.982** | **0.57** | **0.934** | **0.29** |

- **Imputation lifts prediction** (0.952 → 0.979 matched; 0.918 → 0.933
  mismatched) — the misspecification fix, not new information.
- **Annotation lifts attribution on top of imputation** (localisation 0.46 →
  0.57): the functional prior claims the untyped causal. The drop pipelines score
  ~0 — the variant is not in the model, so it can never be localised.
- The annotation's *prediction* gain is small here (0.979 → 0.982): in-sample,
  the smeared typed effect predicts about as well; the annotation's value is
  **localisation** (and, by the same mechanism, cross-ancestry transfer, which
  this single-population simulation does not exercise).

## Caveats

- The gain is bounded by **imputation quality** (`imp_r²`) — a poorly-tagged
  untyped variant cannot be reliably imputed or placed (`min_imp_r2` excludes
  them).
- A perfect-LD tie between an untyped functional variant and a typed
  non-functional one is broken **purely by the prior** — informative, but a wrong
  annotation mislocalises. The mitigation is that `ldpred3_auto_annot_blocks`
  **learns** the annotation weights from the data (S-LDSC-style), rather than
  trusting them blindly.
- Cross-ancestry **portability** is the larger prize (shared functional causal,
  population-specific tags); demonstrating it needs two LD panels and is left as
  future work.
