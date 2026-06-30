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
  **localisation**.

> **This is a best case.** It drops *exactly the causal* variants — the
> highest-value, most adversarial missingness. Under realistic **random**
> missingness the prediction gain is much smaller (next section). Read the
> attribution/localisation result as the durable finding; treat the prediction
> numbers here as an upper bound.

### Realistic (random) missingness

`benchmarks/impute_missingness.py` removes the adversarial setup: a **random**
fraction of variants is untyped (a variant is missing because of the array, not
because it is causal), and `ldpred3_auto_annot` is fit with vs without imputation.
Genetic R² (population LD), swept:

| sweep | no-impute → impute (Δ) |
|-------|------------------------|
| missingness 10% / 30% / 50% / 70% | 0.983→0.993 (+0.010) · 0.972→0.990 (+0.018) · 0.943→0.986 (+0.043) · 0.922→0.974 (**+0.052**) |
| polygenicity p = 0.001 / 0.01 / 0.1 | +0.015 · +0.018 · +0.010 |
| heritability h² = 0.2 / 0.5 / 0.8 | +0.026 · +0.018 · +0.016 |
| #SNPs = 3k / 6k / 12k | +0.011 · +0.018 · +0.021 |

Under random missingness imputation gives a **consistent but modest** gain —
**~1–2% genetic R², rising to ~5% at heavy (70%) missingness** — and it grows with
the missingness fraction and the #SNPs (both increase the model-misspecification
that imputation fixes). So the layer is worth having where the GWAS sumstats are
much sparser than the LD/target panel, but it is not a large, universal win; the
dramatic figures above are the drop-the-causals best case.

## Cross-ancestry portability

The bigger question is *transfer*: a causal functional variant is shared across
populations but its LD tags are not, so an effect placed on discovery-population
tags transfers poorly. `benchmarks/impute_portability.py` simulates two
populations with a coalescent split (msprime; shared variants, **diverged LD**),
runs the GWAS + LD + imputation in population A, drops the functional causals, and
scores genetic R² in **A (in-sample)** and **B (cross-ancestry)**:

| pipeline | R² pop A | R² pop B | retained B/A |
|----------|---------:|---------:|-------------:|
| drop / auto | 0.972 | 0.666 | 68% |
| drop / annot | 0.970 | 0.704 | 73% |
| **impute / auto** | 0.994 | **0.917** | **92%** |
| impute / annot | 0.993 | 0.893 | 90% |

The result is sharper than the naive "annotations drive portability" guess:

- **Imputation is the dominant portability lever** (retained transfer 68% → 92%).
  Specifying the model on the **shared** variant set lets the effect sit on
  variants that exist and carry the same effect in B, instead of A-specific tags
  that do not transfer. This is the same misspecification fix as above, and it is
  large cross-ancestry.
- **The annotation's value is attribution, not transfer.** On top of imputation it
  gives no clear transfer gain here (92% → 90%, within noise); its demonstrated
  benefit is **localising** the causal (the table above). Without imputation it
  nudges transfer up a little (68% → 73%) by shifting effect onto the *typed*
  functional variants.

So the two benefits are **distinct**: imputation fixes model specification (helps
in-sample accuracy *and* cross-ancestry transfer), while the functional annotation
helps you find *which* variant is causal.

> **Same best-case caveat.** This too drops *exactly the functional causals*, so
> the 92% vs 68% is an upper bound on the portability gain; with random
> missingness the cross-ancestry effect, like the in-sample one, is smaller. And
> the simulation is single-locus-architecture, clean-population LD — real transfer
> also involves allele-frequency and causal-effect heterogeneity it does not
> model. The robust, direction-of-effect claim is: *imputation > annotation* for
> transfer; the magnitude is setup-dependent.

## Running time

Imputation is a **one-time pre-step** (per LD block: a small dense solve), so it
amortises across every later fit / method — it is not a per-sweep cost.
`benchmarks/impute_timing.py` (blocks of 500, single core):

| #SNPs | impute | one `auto_annot` fit | impute / fit |
|------:|-------:|---------------------:|-------------:|
| 6k | 0.06 s | 0.17 s | 33% |
| 50k | 0.38 s | 1.50 s | 25% |
| 100k | 0.78 s | 2.88 s | 27% |

So ~**8 µs/SNP, linear in #SNPs, ≈¼–⅓ of a single fit**. And it gets **cheaper as
missingness rises** (0.55 s at 10% missing → 0.13 s at 70%, m=50k): the per-block
cost is `O(k_typed³)`, so more missing ⇒ a smaller typed block ⇒ a smaller solve —
it is cheapest exactly where the accuracy gain is largest. The `O(k_typed³)`
dependence does mean **large blocks are pricier** (k≈2000 is ~64× a k=500 block
per block); recombination-aware block splitting keeps that bounded.

## Caveats

- The gain is bounded by **imputation quality** (`imp_r²`) — a poorly-tagged
  untyped variant cannot be reliably imputed or placed (`min_imp_r2` excludes
  them).
- A perfect-LD tie between an untyped functional variant and a typed
  non-functional one is broken **purely by the prior** — informative, but a wrong
  annotation mislocalises. The mitigation is that `ldpred3_auto_annot_blocks`
  **learns** the annotation weights from the data (S-LDSC-style), rather than
  trusting them blindly.
- Cross-ancestry **portability** (above) is driven by the imputation /
  model-specification fix, not by the annotation — a useful correction to the
  initial intuition. The simulation is clean-population LD with a shared
  single-locus architecture; real transfer also involves allele-frequency and
  causal-effect heterogeneity it does not capture.
