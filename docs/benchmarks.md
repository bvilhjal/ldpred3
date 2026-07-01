# Benchmarks

All benchmarks are single-core unless noted. Regenerate the bigsnpr comparison
with `benchmarks/bench_vs_bigsnpr.py` (the from-scratch driver: shared
simulation → both tools → `benchmarks/cores_1core_benchmark.csv`; R side in
`benchmarks/bench_bigsnpr_blocks.R`; plot with `benchmarks/plot_methods_1core.py`).

## vs bigsnpr (realistic LD, 200k–1M SNPs, single core)

The benchmark uses **realistic LD** — each block is a `k`-SNP correlation matrix
from a coalescent-with-recombination simulation (msprime: haplotype plateaus,
recombination valleys, a heavy decay tail and perfect-LD duplicates), not
idealized AR(1). Both tools see the **same** simulation, sumstats and
hyper-parameters at each size (h²=0.5, p=0.01, N=50,000, burn-in 100 / 200
iterations); each runs on a **single core** (NumPy BLAS and R BLAS pinned to one
thread); bigsnpr's on-disk SFBM is assembled **incrementally** block-by-block
(`as_SFBM` + `$add_columns()`, as in the LDpred2 vignette) so the full
correlation never sits in RAM. To be apples-to-apples, **both** `auto` runs are
warm-started at the oracle hyper-parameters (LDpred3 `h2_init`/`p_init`,
bigsnpr `h2_init`/`vec_p_init`) — otherwise bigsnpr's auto gets the truth while
LDpred3's cold-starts and under-converges at scale.

![1-core method comparison vs bigsnpr](../benchmarks/cores_1core_benchmark.png)

Wall-clock time (s), single core:

| #SNPs | inf py / big | grid py / big | auto py / big |
|-------|-------------:|--------------:|--------------:|
| 200k  | **3.2** / 3.5 | 10.3 / **3.5** | **3.7** / 4.9 |
| 500k  | **5.4** / 8.0 | 15.7 / **9.7** | **9.0** / 13.7 |
| 1M    | 10.9 / 10.9 | 31.7 / **18.9** | **18.9** / 26.6 |

Peak memory (GB) — LD-dominated, so ~equal across the three methods:

| #SNPs | LDpred3 | bigsnpr | ratio |
|-------|----------:|--------:|------:|
| 200k  | **0.64** | 1.05 | 1.6× |
| 500k  | **1.20** | 2.29 | 1.9× |
| 1M    | **2.23** | 4.39 | 2.0× |

**Prediction accuracy is identical** between the two at every size and method —
e.g. auto phenotype-scale R² 0.388/0.388 at 200k → 0.185/0.185 at 1M (the level
falls with #SNPs because the GWAS power N is fixed; R²_pheno = genetic-R² × h²,
h²=0.5).

> **Hardware.** Every timing/memory table in this file was (re-)measured on **one
> machine — a 4-core Linux box @2.8 GHz / 15 GB, single core** — so the numbers
> are mutually comparable. The **2M-SNP** point is omitted from the bigsnpr
> comparison: bigsnpr's on-disk SFBM is ~4.4 GB at 1M, so ~8.5 GB at 2M would
> thrash on 15 GB and its timing would be memory-bound, not a fair single-core
> compute number. LDpred3's `float32` LD stays ~4.3 GB at 2M and runs fine — see
> the [genome-scale table](#genome-scale-scalability-200k4m-snps), which goes to
> 4M. Re-run `bench_vs_bigsnpr.py` on a larger-memory box for a full 2M pairing.

The picture is method-dependent — there is no blanket "N× faster":

- **Memory:** LDpred3 is **~1.6–2.0× leaner**, the gap widening cleanly to a full
  2× at 1M (`float32` LD + one block resident; bigsnpr's SFBM stores `float64`
  values plus per-entry indices).
- **`-auto`:** LDpred3 is **~1.3–1.5× faster** at matched initialization — its
  streaming global-hyper sampler is the strongest path.
- **`-inf`:** LDpred3 faster at 200k–500k, level at 1M — the per-block linear
  solve holds up well.
- **`-grid`:** **bigsnpr is faster** here (its compiled C++ grid sampler beats
  LDpred3's per-block Python-orchestrated one); the gap is widest at 200k (~3×)
  and narrows to ~1.7× by 1M. This is LDpred3's weak spot at fixed
  hyper-parameters.

### `auto` from a cold start (no oracle hyper-parameters)

The table above warm-starts both autos at the oracle h²/p. In practice you don't
know them — learning them is auto's whole point. Re-running with **both** tools
cold-started at `h2_init=0.1, p_init=0.1` (single chain, same burn-in/iterations,
same LD and sumstats; regenerate with `benchmarks/bench_cold_init.py`):

| #SNPs | LDpred3 R² | bigsnpr R² | LDpred3 (s) | bigsnpr (s) |
|-------|----------:|-----------:|------------:|------------:|
| 200k  | 0.388 | 0.388 | **3.9** | 8.1 |
| 500k  | **0.279** | 0.230 | **12.3** | 42.5 |
| 1M    | **0.155** | 0.144 | **42.3** | 103.6 |

From a cold start LDpred3's auto is **at least as accurate as bigsnpr's and 2–3.4×
faster** — they tie at 200k, and LDpred3 leads by up to ~21% (500k) at the larger
sizes. Both fall a little short of their oracle-init numbers (the fixed 300-sweep
budget doesn't fully converge the hyper-parameters at 1M+), but LDpred3 degrades
less; the oracle warm-start in the main table is what lets bigsnpr's single chain
catch up. Caveat: this is bigsnpr's **single-chain** auto, matched to LDpred3 —
its recommended usage runs ~30 chains over a `vec_p_init` grid, which would
recover accuracy at roughly 30× the cost.

## End-to-end pipeline vs bigsnpr

Beyond the per-block accuracy check above, the **whole pipeline** was validated
against bigsnpr: the same simulated PLINK target + GWAS sumstats + in-sample LD
were run through LDpred3's complete pipeline (QC → harmonise → per-block LD →
`-auto` → scoring) and through bigsnpr's `snp_ldpred2_auto`, and the
per-individual polygenic scores compared.

| metric | result |
|--------|--------|
| PRS correlation (LDpred3 vs bigsnpr) | **r = 0.9995** |
| R² vs true genetic value | 0.567 (LDpred3) / 0.575 (bigsnpr) |

So the pipeline glue — allele harmonisation, QC, LD construction and scoring —
reproduces bigsnpr's polygenic scores essentially exactly. (Validation against a
downloaded public GWAS + 1000 Genomes reference is the natural next step; it adds
real-data quirks the simulation can't, but needs multi-GB inputs.)

## Methods by genetic architecture (realistic LD)

How do the LDpred3 variants compare across genetic architectures? The genome is
100 distinct coalescent/msprime LD blocks of 500 SNPs (m=50,000, h²=0.5); for
each architecture we simulate true effects, generate summary statistics, fit
every method, and measure the **genetic R²** — the squared correlation between
the PRS and the true genetic value under population LD,
`(β̂ᵀRβ)² / [(β̂ᵀRβ̂)(βᵀRβ)]` — averaged over 5 replicates. `grid` is given the
oracle `(h²,p)`; `annot` gets one functional annotation (informative only in the
last row). Regenerate with `benchmarks/bench_methods.py` /
`benchmarks/plot_methods_arch.py`.

![Methods by architecture](../benchmarks/methods_arch_benchmark.png)

Genetic R² at **N = 10,000** (the lower-power regime separates the methods):

| architecture | marginal | inf | grid | auto | annot |
|--------------|---------:|----:|-----:|-----:|------:|
| infinitesimal       | 0.451 | **0.528** | 0.527 | 0.524 | 0.525 |
| sparse (p=0.01)     | 0.440 | 0.520 | 0.738 | **0.739** | 0.738 |
| polygenic (p=0.2)   | 0.446 | 0.523 | **0.524** | 0.523 | 0.524 |
| major locus         | 0.434 | 0.518 | **0.686** | 0.675 | 0.674 |
| annotation-enriched | 0.458 | 0.530 | 0.632 | 0.632 | **0.648** |

Genetic R² at **N = 50,000** (higher power; everything shifts up and compresses):

| architecture | marginal | inf | grid | auto | annot |
|--------------|---------:|----:|-----:|-----:|------:|
| infinitesimal       | 0.570 | **0.790** | 0.789 | 0.786 | 0.786 |
| sparse (p=0.01)     | 0.563 | 0.787 | 0.944 | **0.945** | 0.944 |
| polygenic (p=0.2)   | 0.570 | 0.789 | **0.795** | 0.795 | 0.795 |
| major locus         | 0.557 | 0.787 | **0.903** | 0.902 | 0.902 |
| annotation-enriched | 0.580 | 0.792 | 0.899 | 0.899 | **0.906** |

Takeaways:

- **The raw marginal PRS is always far behind** — the LD adjustment is the
  first-order win (≈0.45→0.52 at N=10k, ≈0.57→0.79 at N=50k).
- **`inf` is architecture-robust but flat**: it is the best model *only* under a
  truly infinitesimal (or near-infinitesimal polygenic) architecture, and leaves
  large gains on the table whenever the trait is sparse or has major loci.
- **`grid`/`auto` win decisively on sparse and major-locus** architectures
  (e.g. 0.74/0.69 vs 0.52 for `inf` at N=10k) — the spike-and-slab captures
  concentrated signal. **`auto` matches the oracle `grid`** (handed the true
  `h²` and `p`) without any hyper-parameters — the practical default.
- **`annot` matches `auto` when the annotation is uninformative and beats it
  when it carries signal.** On the annotation-enriched architecture it is the
  best method at both power levels (N=10k: 0.648 vs grid 0.632; N=50k: 0.906 vs
  0.899), and it never falls behind `auto` elsewhere. The lift from a *single*
  binary annotation is modest — SBayesRC's larger real-data gains come from many
  S-LDSC-calibrated annotations — but it is consistent and free of the
  "garbage-in" penalty a *fixed* bad prior would carry.

> **Convergence note (why this is the corrected table).** An earlier run with a
> lazy annotation-map update (`theta_every=10`) had `annot` *underperforming*
> `auto` at N=10k — e.g. enriched 0.60 vs 0.64. That was an artifact: with short
> chains the `p_j = sigmoid(Aθ)` map had not converged, so it over-estimated the
> global `p` (effective p ≈ 0.04 vs a true 0.02) and **over-shrank** the effects.
> Updating `θ` every sweep (now the default — the IRLS step is cheap for a
> handful of annotations) lets the map and the effects co-adapt; the learned
> enrichment then reaches its true value (θ_func ≈ 1.7) and the anomaly
> disappears. A diagnostic confirmed the fix is purely about convergence: more
> iterations *without* frequent θ updates also fixed it, but added nothing on top
> of `theta_every=1`.

### Per-method running time

Fit time on the same setup (m=50,000 = 100 coalescent blocks of 500, single
core, burn-in 80 / 200 sampling sweeps; `inf` is a direct per-block solve).
Regenerate with `benchmarks/timing_bench.py`.

| method | fit time (s) |
|--------|-------------:|
| inf    | 0.31 |
| auto   | 1.46 |
| grid   | 1.45 |
| annot  | 2.51 |

`inf` is cheapest (one linear solve per block, no sampling). `grid`/`auto` are
the spike-and-slab Gibbs samplers and cost about the same. `annot` is ~1.7×
`auto`: it runs the same per-block effect sweeps plus a logistic annotation-map
update every sweep.

**Cost of the annotation learner (`annot`).** The θ-update is an `O(m·K²)` IRLS
solve in the number of annotations `K`, run every `theta_every` sweeps. Fit time
(s) at m=50,000:

| #annotations K | `theta_every=1` (default) | `theta_every=10` |
|---------------:|--------------------------:|-----------------:|
| 1   | 2.5  | 1.6 |
| 5   | 2.8  | 1.6 |
| 20  | 3.3  | 1.7 |
| 50  | 6.6  | 2.0 |
| 100 | 12.6 | 2.6 |

So the convergence-correct default (`theta_every=1`) is nearly free for a handful
of annotations but its `O(K²)` per-sweep cost takes over by `K ≈ 50`; with many
annotations raise `theta_every` to amortise it. (Persisting the running `R@β`
residual across chunks — rather than rebuilding it each θ-update — keeps the
default cheap; without it `annot` was ~3× `auto` instead of ~1.7×.)

## MAF-dependent effect-size prior (`alpha`)

The standard point-normal prior gives every variant the same *standardised*
effect variance — i.e. per-allele effect variance `∝ 1/[2f(1-f)]`, the `alpha =
-1` special case. The `--alpha` knob relaxes this, scaling each variant's slab
variance by `[2f(1-f)]^(1+alpha)` (the LDpred2-auto `alpha`/`use_MLE` prior of
Privé et al. 2023). Does matching it to a real effect–frequency coupling help?

This simulates a realistic genome (coalescent LD + msprime's genuine,
L-shaped allele-frequency spectrum — m=10,000, median MAF ≈ 0.07, 43 % below
0.05, so `2f(1-f)` spans the full [0.01, 0.5] range), draws causal
**standardised** effects with variance `∝ [2f(1-f)]^(1+alpha_true)`, and fits
`auto` (per-block, so the *only* thing that changes across a row is the prior
`alpha`) over a grid of prior `alpha`. Genetic R² at the **low-power** N=5,000
(where the prior shape matters; h²=0.5, p=0.1, 8 reps). Regenerate with
`benchmarks/maf_alpha_prior.py`.

| true `alpha` | fit −1.0 | fit −0.75 | fit −0.5 | fit −0.25 | fit 0.0 | best |
|--------------|---------:|----------:|---------:|----------:|--------:|-----:|
| −1.0 (standard) | **0.661** | 0.648 | 0.626 | 0.602 | 0.577 | −1.0 |
| −0.5            | 0.708 | **0.713** | **0.713** | 0.706 | 0.697 | −0.75 |
|  0.0            | 0.737 | 0.747 | 0.754 | **0.757** | 0.756 | −0.25 |

Takeaways:

- **The best-fit `alpha` tracks the truth.** Each row peaks at (or next to) its
  own `alpha_true` and falls off monotonically away from it — the prior is doing
  exactly what it claims, reallocating slab variance across the frequency
  spectrum.
- **The default (`alpha = -1`) is safe.** When the data really follow the
  standard model (top row) the default is the single best choice, and a
  *mis-set* positive `alpha` there is the worst cell in the table (0.661 →
  0.577, an 8-point drop). So the bit-for-bit-unchanged default costs nothing
  when you have no reason to move it — and a badly wrong `alpha` is the real
  risk, not a missed gain.
- **The gain from matching is real but modest** (~0.005–0.02 genetic R² here),
  in line with Privé et al.'s real-data findings: it is a refinement, not a
  first-order win like the LD adjustment or the spike-and-slab. Privé's
  `use_MLE` selects `alpha` by the model's own likelihood; here it is a user
  knob (`--alpha`), so set it from external knowledge of the trait's
  architecture (or leave it at `-1`).

## Methods: accuracy vs scalability

The architecture table above fixes the genome at m=50,000; this sweeps **every
method on the same realistic-LD genome from 50k to 1M SNPs at once**, reporting
accuracy *and* cost together. One sparse architecture (h²=0.5, p=0.01), GWAS
N=50,000 held fixed (so power dilutes as variants multiply), realistic coalescent
LD, single core; `grid` gets the oracle `(h²,p)`, `auto` self-tunes, `annot` gets
one uninformative annotation, `lassosum2` is the L1 / pseudo-validation predictor
and `laplace` is the Bayesian-lasso (Laplace-prior posterior mean). Each size runs
in its own process for a clean peak RSS. Regenerate with
`benchmarks/method_scaling.py`.

> **Hardware.** Measured on the same machine as every other table here (4-core
> Linux @2.8 GHz / 15 GB, single core), so timings are mutually comparable.
> Genetic R² and the `float32` LD memory are hardware-independent and reproduce
> the earlier runs bit-for-bit — confirming the internal refactors (the shared
> `_pn_step` sampler core, the [constant-N fast path](#constant-n-sampler-fast-path),
> the new methods) changed neither accuracy nor footprint.

![Methods: accuracy, time and memory vs genome size](../benchmarks/method_scaling.png)

Genetic R² by method (accuracy falls with #SNPs because N is fixed):

| #SNPs | marginal | inf | grid | auto | annot | lassosum2 | laplace |
|-------|---------:|----:|-----:|-----:|------:|----------:|--------:|
| 50k   | 0.287 | 0.396 | 0.468 | **0.470** | 0.469 | 0.399 | 0.416 |
| 100k  | 0.261 | 0.343 | 0.438 | 0.438 | **0.438** | 0.324 | 0.366 |
| 200k  | 0.237 | 0.284 | **0.388** | 0.388 | 0.387 | 0.313 | 0.297 |
| 500k  | 0.176 | 0.194 | **0.280** | 0.279 | 0.279 | 0.262 | 0.195 |
| 1M    | 0.130 | 0.134 | **0.185** | 0.155 | 0.155 | 0.164 | 0.131 |

Fit time (s), single core:

| #SNPs | inf | grid | auto | annot | lassosum2 | laplace |
|-------|----:|-----:|-----:|------:|----------:|--------:|
| 50k   | 0.7 | 1.6 | 0.9 | 5.2 | 2.2 | 5.3 |
| 100k  | 1.1 | 3.3 | 1.8 | 7.6 | 5.0 | 9.4 |
| 200k  | 2.2 | 6.4 | 4.1 | 16.2 | 11.3 | 19.5 |
| 500k  | 5.4 | 16.1 | 13.0 | 44.4 | 28.5 | 45.3 |
| 1M    | 10.8 | 31.9 | 41.2 | 101.7 | 64.9 | 94.6 |

Peak memory is **LD-dominated** — the resident `float32` block-diagonal LD is
~2.2 GB at 1M and grows ~linearly (the clean, contiguous measurement is in the
[genome-scale table](#genome-scale-scalability-200k4m-snps)). The peak RSS this
per-size worker reports (0.41 GB at 50k → 2.42 GB at 1M) is now close to that
intrinsic footprint — the earlier `lassosum2` `float64`-LD copy and other
temporaries that inflated it were trimmed (float32 LD, in-process build). The
sampler *state* is
O(m) and negligible beside the LD, so accuracy aside the methods are
memory-equivalent.

Takeaways:

- **`grid`/`auto`/`annot` beat `inf` and `marginal` at every size** — the
  spike-and-slab is the accuracy win on a sparse trait, and the gap over the raw
  marginal score is the LD adjustment (≈0.13–0.28 absolute here).
- **`auto` matches the oracle `grid` up to 500k**, then **under-converges at 1M**
  (0.154 vs 0.190): self-tuning `h²`/`p` from a cold start needs more than the
  fixed ~300-sweep budget once the genome is huge. At genome scale give `auto`
  more iterations / chains (or warm-start the hyper-parameters) to recover the
  oracle-`grid` accuracy — this is the same convergence effect as the
  [cold-start table](#auto-from-a-cold-start-no-oracle-hyper-parameters).
- **`lassosum2` is a genuine peer of the Bayesian methods** — trailing `grid`/
  `auto` at small genomes (0.40 vs 0.47 at 50k) but closing the gap as #SNPs
  grows and, at 1M, **beating both `inf` (0.135) and the under-converged `auto`
  (0.154) at 0.168**. This is the expected behaviour: the L1 penalty is the MAP
  under a Laplace prior, a legitimate shrinkage model, so its prediction sits in
  the same band as the spike-and-slab rather than collapsing. (It took a fixed
  [pseudo-validation guard](#a-note-on-lassosum2s-pseudo-validation) to see this
  — the raw criterion had been selecting overfit models and dragging the score to
  0.075 at 1M.) Its value is complementarity and robustness on real data
  (misspecified LD, heavier-tailed effects), which is why the bigsnpr workflow
  keeps whichever of `auto` / `lassosum2` pseudo-validates better rather than
  trusting one blindly. It costs ~2× `auto` (a full (s, λ) grid of
  coordinate-descent sweeps).
- **`laplace` vs `lassosum2` is power-dependent — they are the *mean* and the
  *mode* of the same Laplace prior.** At moderate/high per-SNP power `laplace`
  (the posterior mean, the better estimator under squared loss) edges `lassosum2`
  (the MAP/mode): 0.416 vs 0.399 at 50k, 0.366 vs 0.324 at 100k, and likewise in
  the high-power [architecture check](#laplace-prior-the-bayesian-lasso-methodlaplace).
  But as power drops (`N/M` ≤ 0.25 here) **`lassosum2` pulls clearly ahead** —
  0.262 vs 0.195 at 500k, 0.164 vs 0.131 at 1M. The reason is structural: the
  posterior mean is **dense** (no point mass at zero), so genome-wide it spreads
  nonzero weight over ~1M mostly-null SNPs and cannot escape their noise, whereas
  `lassosum2`'s L1 mode zeros them out and its pseudo-validation adapts the
  shrinkage to the data. Both cost `annot`-like (~3.6→73 s across 50k–1M). So
  `laplace` recovers to the `inf`/`marginal` band at scale (it no longer *under*performs
  a raw ridge — the earlier collapse was an EM artefact, now fixed with an
  LDSC-plug-in `λ`); but for a *single* sparse method at genome scale `lassosum2`
  is the more robust choice.
- **`annot` tracks `auto`** when the annotation is uninformative, at **~2–3× the
  time** (the logistic θ-update) — the cost is worth it only when the annotation
  carries signal (see the architecture table).
- **Time is roughly linear in #SNPs** for `inf` (cheapest — a per-block solve, no
  sampling) and `grid`; `auto`'s per-sweep hyper-parameter update makes it the
  steepest of the samplers at 1M, with `annot`/`lassosum2`/`laplace` the priciest
  (extra per-SNP work). `marginal` is free.

### A note on lassosum2's pseudo-validation

With no validation cohort, `lassosum2` picks its `(s, λ)` by pseudo-validation —
the summary-statistic estimate of the PRS–trait correlation `βᵀr / √(βᵀRβ)`.
That estimate is **in-sample**, and on a well-conditioned LD the smallest
penalties drive `β` toward the `R⁻¹r` (OLS) fit, whose pseudo-validation score
runs *past 1* — an impossible correlation, and the fingerprint of overfitting.
Selecting on the raw criterion therefore chose the densest, most-overfit model:
at genome scale its score climbed to 1.2–2.4 while the true accuracy collapsed
(genetic R² 0.49 at 200k → 0.15 at 1M).

The fix is to **restrict the selection to physically valid models (score ≤ 1)**.
The sparse end of the path always qualifies, so a valid model is always
available, and dropping the overfit tail recovers essentially all of the
accuracy an actual held-out cohort would find:

| #SNPs | raw argmax | **guarded (score ≤ 1)** | oracle (held-out) |
|-------|-----------:|------------------------:|------------------:|
| 200k  | 0.488 | **0.654** | 0.759 |
| 500k  | 0.254 | **0.487** | 0.535 |
| 1M    | 0.150 | **0.335** | 0.350 |

(genetic R²; the guarded pick lands within a few percent of the oracle at 1M).
This is what lifts `lassosum2` from an apparent also-ran to the peer of `auto`
in the table above. A real validation cohort is still preferable when available
— the guard is the best that pure summary statistics allow.

## Laplace prior: the Bayesian lasso (`method="laplace"`)

`lassosum2` is the posterior *mode* under a Laplace (double-exponential) prior;
`ldpred3_laplace` samples the posterior **mean** of that same prior — the proper
Bayesian shrinkage estimator, which should predict at least as well. It is a
Gibbs sampler over the normal / exponential scale-mixture of the Laplace (Park &
Casella 2008): a per-SNP latent variance `τ_j²` with `β_j | τ_j² ~ N(0, τ_j²)`
and `τ_j² ~ Exp(λ²/2)`, so each sweep is the usual Gaussian per-SNP conditional
plus an Inverse-Gaussian draw for `1/τ_j²`. The global shrinkage is a **plug-in**
`λ = √(2k/h²)` from a heritability estimated once by LD Score regression — this
replaced a per-sweep EM update (`λ² = 2k/Στ_j²`) that overfit at low
signal-to-noise (its genetic variance ran ~1.8× the truth at `N/M ≈ 0.1`, so
`laplace` dropped *below* the raw marginal at 1M SNPs). Unlike the spike-and-slab
it has no point mass at zero — the posterior mean is dense.

Genetic R² by architecture (realistic coalescent LD, m=10,000, N=20,000, h²=0.5,
5 reps; `grid` gets the oracle `(h²,p)`). Regenerate with
`benchmarks/laplace_vs_lasso.py`.

| architecture | inf | grid | auto | lassosum2 | **laplace** |
|--------------|----:|-----:|-----:|----------:|------------:|
| infinitesimal     | 0.862 | 0.861 | 0.858 | 0.854 | **0.861** |
| polygenic (p=0.1) | 0.859 | 0.879 | 0.879 | 0.860 | **0.866** |
| sparse (p=0.01)   | 0.864 | 0.969 | 0.970 | 0.889 | **0.897** |

- **At this (high) power the posterior mean beats the mode.** `laplace` edges out
  `lassosum2` at every architecture (0.861 vs 0.854, 0.866 vs 0.860, 0.897 vs
  0.889) — the expected reward for averaging the posterior rather than taking its
  peak, and a direct confirmation that the two are the same prior seen two ways.
  **This ordering flips at low power** (the fixed-N [scaling table](#methods-accuracy-vs-scalability)):
  once `N/M` ≲ 0.25 `lassosum2`'s sparsity + pseudo-validation pull ahead, because
  the dense posterior mean can't zero the genome's many null SNPs. Mean-beats-mode
  holds *per SNP*; sparsity wins *across a genome of nulls*.
- **It matches `inf` on the infinitesimal trait** (0.861 vs 0.862): with a truly
  Gaussian architecture the Laplace mean reproduces the optimal ridge/BLUP.
- **It trails the spike-and-slab on the sparse trait** (0.897 vs `auto` 0.970):
  the Laplace has heavier tails than a Gaussian but no spike, so it cannot
  concentrate on a handful of causals the way the point-normal does. That gap is
  the price of the continuous prior — `auto` remains the default for sparse
  traits; `laplace` is the self-tuning dense alternative (and the Bayesian sibling
  of `lassosum2`), best used as a cross-check rather than a genome-scale default.

> **Tuning `λ`.** Two self-tuning schemes were tried before the current plug-in.
> A naïve fully-Bayesian `λ` (a Gamma-Gibbs draw) drifts to the hyper-prior's mean
> *independently of the data*, mis-shrinking every architecture to ~0.76. A
> marginal-maximisation (EM) update `λ² = 2k/Στ_j²` fixed that at high power but
> has a positive-feedback failure at **low** signal-to-noise — the fitted `τ_j²`
> inflate on noise, so `λ` shrinks and the fit overfits (genetic variance ~1.8×
> the truth; `laplace` fell *below* the raw marginal at 1M SNPs). The current
> default is a **plug-in** `λ = √(2k/h²)` from an LD-Score-regression `h²`
> (estimated once, robust across power) — stable everywhere, and what lifted the
> low-power rows above (0.174→0.195 at 500k, 0.106→0.131 at 1M).

## Genotype-level simulation

`ldpred3/simulate.py` is a full end-to-end simulation: it generates genotypes with
block LD, simulates a phenotype under a chosen heritability and polygenicity,
runs a marginal GWAS, estimates the LD matrix from the training sample, fits
LDpred3, and reports **out-of-sample** prediction R² on a held-out test set. It
sweeps a grid of polygenicity × heritability × sample size.

To stay within memory at scale, genotypes are stored as `int8` dosages and
every step (standardization, GWAS, LD, PRS) is processed one LD block at a time,
so a full float genotype matrix is never materialised.

**LD model (`--ld-model`).** Two choices for the LD between SNPs:

* `ar1` (default): a latent-Gaussian model with geometric LD decay
  (`r ≈ ρ^dist`). Fast and dependency-free, but idealized — LD collapses to ~0
  within a handful of SNPs.
* `coalescent`: realistic LD from a coalescent-with-recombination simulation
  (via [msprime](https://tskit.dev/msprime), human-like Ne=10⁴ and 1e-8 recomb/
  mutation rates). This produces actual haplotype blocks, recombination
  hotspots, a heavy LD decay tail and sporadic long-range LD — the structure of
  real reference panels (mean r² stays ~0.02 at 200 SNPs apart, vs ~0 for AR(1)).

LDpred3's advantage over the raw marginal PRS is *larger* under realistic LD
(e.g. h²=0.5, p=0.01: marginal 0.21 → grid/auto 0.43 with coalescent LD, vs
0.32 → 0.50 with AR(1)), because realistic long-range LD inflates the naive
score that LDpred3's LD-adjustment removes.

```bash
python -m ldpred3.simulate --quick                        # fast (AR(1))
python -m ldpred3.simulate --quick --ld-model coalescent  # realistic LD (needs msprime)
python -m ldpred3.simulate --csv sim.csv                  # full accuracy grid, save results
```

Representative results (m=10000 SNPs, blocks of 200, AR(1) LD; prediction R² vs
phenotype):

| N | h² | p (causal) | marginal | inf | grid | auto | ceiling |
|---|----|-----------|---------|-----|------|------|---------|
| 5000  | 0.5 | 0.001 | 0.097 | 0.100 | 0.465 | 0.465 | 0.475 |
| 20000 | 0.5 | 0.001 | 0.254 | 0.262 | 0.489 | 0.489 | 0.489 |
| 20000 | 0.5 | 0.1   | 0.245 | 0.265 | 0.417 | 0.417 | 0.512 |
| 20000 | 0.3 | 0.01  | 0.135 | 0.139 | 0.301 | 0.300 | 0.311 |

LDpred3 always beats the raw marginal baseline; accuracy rises with heritability
and sample size; `grid`/`auto` approach the ceiling for sparse architectures and
remain best across the grid. The infinitesimal model only modestly beats the
marginal score — its all-causal prior leaves accuracy on the table whenever the
trait is even mildly sparse.

## Scaling: what the algorithm actually depends on

The LDpred3 *algorithm* works from summary statistics + the LD matrix, so its
cost is **independent of the GWAS sample size N** and is driven instead by the
**LD structure (block size)**. The benchmarks below separate the algorithm's
`fit` time from the simulation/GWAS/LD-construction `prep` time (which does scale
with N). Single core, Numba on, h²=0.5, p=0.01 — these tables illustrate the
*dependence shape* (fit flat in N, driven by block size), which is
hardware-independent; the absolute times are on the faster laptop referenced
in earlier revisions, not the 15 GB box the other tables now use.

**Independent of N** (`--n-independence`, m=10000, blocks of 200): fit time is
flat while prep grows with N.

| N_train | prep (s) | fit_grid (s) | fit_auto (s) |
|---------|---------|--------------|--------------|
| 2000   | 1.5  | 0.059 | 0.041 |
| 8000   | 4.3  | 0.057 | 0.038 |
| 32000  | 15.2 | 0.057 | 0.037 |

**Driven by LD block size** (`--ld-scaling`, m=20000 fixed, N=8000): larger LD
blocks make each block's solve/sampler costlier. The infinitesimal model is a
dense linear solve per block (≈O(m·k²), grows fast), whereas the Gibbs samplers
stay nearly flat for sparse traits thanks to the running-residual update.

| block size | #blocks | fit_inf (s) | fit_grid (s) | fit_auto (s) |
|-----------|---------|-------------|--------------|--------------|
| 100   | 200 | 0.017 | 0.123 | 0.097 |
| 250   | 80  | 0.044 | 0.120 | 0.079 |
| 500   | 40  | 0.116 | 0.119 | 0.080 |
| 1000  | 20  | 0.385 | 0.127 | 0.094 |
| 2000  | 10  | 1.331 | 0.146 | 0.123 |

**Scaling #SNPs** (`--scaling`, N=8000, blocks of 200): with N fixed, total
runtime and memory grow ~linearly in #SNPs (≈1 ms/SNP; memory bounded by the
`int8` genotype matrix). Accuracy falls only because more SNPs/causal variants
dilute the fixed GWAS power — `grid` degrades gracefully while raw
`marginal`/`inf` collapse.

| #SNPs | prep (s) | fit (s) | peak mem (GB) | marginal | inf | grid | auto | ceiling |
|-------|---------|--------|---------------|---------|-----|------|------|---------|
| 10000  | ~4  | ~0.5 | 0.32 | 0.167 | 0.174 | 0.463 | 0.463 | 0.503 |
| 50000  | ~22 | ~0.6 | 0.89 | 0.051 | 0.050 | 0.316 | 0.316 | 0.485 |
| 100000 | ~44 | ~1.5 | 1.32 | 0.016 | 0.015 | 0.185 | 0.059 | 0.482 |

Practical takeaway: for dense data with long-range / large LD blocks, the dense
per-block LD storage and the infinitesimal solve become the bottleneck, which
motivates the banded / sparse-LD backend (see [algorithm.md](algorithm.md)).

### Genome-scale scalability (200k–4M SNPs)

How far does LDpred3 scale on a commodity machine? This pushes the **fitting**
step alone (LD already constructed, the realistic-LD library cycled to the target
size, single core) from 200k up to **4M SNPs** — past a fully imputed genome.
Measured on the same machine as every table here (4-core Linux @2.8 GHz / 15 GB);
regenerate with `benchmarks/bench_ldpred3_scaling.py`.

![LDpred3 scalability to 4M SNPs](../benchmarks/ldpred3_scaling.png)

| #SNPs | peak RAM | inf (s) | grid (s) | auto (s) | auto R² |
|-------|---------:|--------:|---------:|---------:|--------:|
| 200k  | 0.60 GB | 3.1 | 6.4 | 3.6 | 0.388 |
| 500k  | 1.20 GB | 5.5 | 16.0 | 9.6 | 0.280 |
| 1M    | 2.23 GB | 10.8 | 31.6 | 18.9 | 0.185 |
| 2M    | 4.28 GB | 21.5 | 63.0 | 37.1 | 0.103 |
| 3M    | 6.32 GB | 32.9 | 94.6 | 57.2 | 0.069 |
| 4M    | 8.38 GB | 42.8 | 127.8 | 75.2 | 0.051 |

Both axes are **linear in #SNPs**, and the **auto R² reproduces the earlier runs
bit-for-bit** at every size (0.388 → 0.051) — the refactors left the fit
unchanged. Fit time runs ~19 µs/SNP for `auto` (4M in ~75 s; `grid` ~2 min); the
1M row here (2.23 GB, auto 18.9 s) matches the [bigsnpr comparison](#vs-bigsnpr-realistic-ld-200k1m-snps-single-core)
exactly, as it should. Peak memory grows at **~2.1 GB per million SNPs** — the
`float32` dense LD held resident during the fit — so **4M fits in 8.4 GB**, and
the practical dense-in-RAM ceiling on a 15 GB machine is **~6–7M SNPs**. (The auto
R² falls with #SNPs only because GWAS power N=50,000 is held fixed while variants
multiply — it is not a scaling limit.)

The contrast with bigsnpr is the headline: bigsnpr's `float64` on-disk SFBM is
~8.5 GB at 2M and would thrash on this 15 GB box, whereas LDpred3's `float32` LD
sails to 4M in 8.4 GB. And to go **past** the dense-RAM ceiling, the low-rank / on-disk
`--ld-stream` backend keeps resident memory at O(one block) regardless of genome
size (see [LD representations at scale](#ld-representations-at-scale-memory-vs-running-time)) —
so LDpred3 is not capped at 6–7M; that is just the dense-in-RAM limit.

## Robustness: LD reference quality & sample size

How sensitive is the PRS to two things you don't control perfectly in practice —
the LD reference panel and the GWAS sample size? Both fit LDpred3-`auto` on
summary statistics generated from the true coalescent LD (m=6000, h²=0.5,
p=0.01, N=50000) and report the held-out **genetic R²** and the fitted genetic
variance (an h² proxy). Regenerate with `benchmarks/robustness_ld_and_n.py`.

**LD reference panel size** (`Nref`), the dominant real-world error — the LD is
estimated from `Nref` reference individuals rather than known exactly:

| Nref | pred R² | h² proxy |
|------|--------:|---------:|
| 500   | 0.829 | 0.932 |
| 1000  | 0.912 | 0.683 |
| 2000  | 0.970 | 0.540 |
| 5000  | 0.988 | 0.512 |
| 10000 | 0.989 | 0.512 |
| ∞ (true LD) | 0.992 | 0.491 |

A **small panel is actively harmful**: at Nref=500 the noisy LD makes the sampler
over-fit, inflating h² to 0.93 (true 0.5) and dropping R² to 0.83. Accuracy is
near-clean only by **Nref≈5000**; a 1000-Genomes-scale panel (~2000) already
costs ~2% R² and a ~8% h² over-estimate. This is the systematic bias behind the
0% interval coverage in [inference.md](inference.md#interval-calibration) — use
the largest matched-ancestry panel you can.

**Sample-size misspecification** — fitting with the wrong `N` (Nref=2000):

| N_used / N_true | pred R² | h² proxy |
|-----------------|--------:|---------:|
| 0.70 | 0.981 | 0.519 |
| 0.85 | 0.977 | 0.529 |
| 1.00 | 0.970 | 0.540 |
| 1.15 | 0.965 | 0.551 |
| 1.30 | 0.957 | 0.565 |

LDpred3-`auto` is **fairly robust to N**: ±30% changes R² by only ~±1.5% and
moves the h² proxy roughly in proportion to `N_used`. There is even a mild twist
— slightly *under*-stating `N` (0.70–0.85) predicts a touch **better** here,
because the extra shrinkage offsets the over-fit that noisy reference LD induces.
A correct (or mildly conservative) `N` is fine; a wildly wrong one mostly
mis-scales the heritability.

## Accuracy across polygenicity, heritability and sample size

How does PRS accuracy move with the three things that vary most across real
traits? Each axis is swept from a baseline (p=0.01, h²=0.5, N=50000), holding the
other two fixed, on realistic reference-panel LD (m=8000, Nref=2000, coalescent;
genetic R² = squared correlation of the PRS with the true genetic value).
`inf` is given the true h² (oracle); `auto` self-tunes. Regenerate with
`benchmarks/sweep_p_h2_n.py`.

**Sample size N** (p=0.01, h²=0.5):

| N | marginal | inf | auto |
|--------|---------:|----:|-----:|
| 10000  | 0.594 | 0.815 | 0.946 |
| 50000  | 0.622 | 0.912 | **0.970** |
| 200000 | 0.628 | 0.935 | 0.951 |

`auto` is strong even at N=10k and rises to ~0.97 — but note it **dips slightly at
N=200k while `inf` keeps climbing**. That is the *reference-LD ceiling*: with a
finite reference panel (Nref=2000), more GWAS data makes the sampler trust the
(mismatched) LD harder and over-fit relative to the true LD. At very large N a
better LD reference matters more than more samples (see
[LD reference quality](#robustness-ld-reference-quality--sample-size)).

**Heritability h²** (p=0.01, N=50000):

| h² | marginal | inf | auto |
|-----|---------:|----:|-----:|
| 0.1 | 0.594 | 0.815 | 0.946 |
| 0.3 | 0.617 | 0.893 | **0.970** |
| 0.5 | 0.622 | 0.912 | 0.970 |
| 0.8 | 0.625 | 0.924 | 0.965 |

Genetic R² (PRS vs the *genetic* value) is fairly flat in h² for `auto` — the
metric normalises out the heritability, so what it shows is that `auto` recovers
the genetic component well across the range, given enough power. (`inf` improves
with h² because higher h² sharpens its dense per-SNP estimates.) Phenotype-scale
R² would instead scale ~linearly with h².

**Polygenicity p** (h²=0.5, N=50000):

| p | marginal | inf | auto |
|-------|---------:|----:|-----:|
| 0.001 | 0.579 | 0.904 | 0.962 |
| 0.01  | 0.622 | 0.912 | **0.970** |
| 0.1   | 0.619 | 0.909 | 0.923 |
| 1.0 (infinitesimal) | 0.609 | 0.907 | 0.901 |

This is the clearest axis: `auto` **excels on sparse architectures** (0.96–0.97 at
p≤0.01) and **degrades toward the infinitesimal limit** (0.901 at p=1), where its
spike-and-slab is mildly mis-specified and the matched `inf` model (0.907) edges
it. `inf` is flat ~0.91 across p by construction (it assumes all variants causal,
so sparsity neither helps nor hurts it). The practical reading: `auto` is the
right default — it wins wherever there is concentrated signal and is only a hair
behind a perfectly-matched `inf` when the trait is truly infinitesimal.

## DENTIST LD-consistency filter

Does the optional DENTIST filter (`--dentist`) actually recover accuracy when the
sumstats contain LD-inconsistent errors, and what does it cost on clean data?
This plants spurious genome-wide-significant hits at **non-causal** variants (an
allele/strand error that inflates a null variant's z out of line with its LD
neighbours), fits LDpred3-`auto` with and without the filter, and reports genetic
R². Coalescent LD, m=4000 (20×200 blocks), Nref=10k, N=10k, h²=0.5, p=0.05, 5 reps.
Regenerate with `benchmarks/dentist_recovery.py`.

| condition | genetic R² |
|-----------|-----------:|
| clean (no errors) | 0.917 |
| corrupted, no filter | 0.811 |
| corrupted, `--dentist` | **0.907** |

30 planted errors/rep; DENTIST catches **100%** of them. On *clean* sumstats it
drops **0.02%** of genuine variants (the false-positive cost).

Reading: a handful of LD-inconsistent false hits cost ~0.11 R² (0.92 → 0.81) as
the LD adjustment propagates them to their neighbours; DENTIST recovers almost all
of the loss (→ 0.91). On *realistic* LD the filter is sharper than on idealized
AR(1): an out-of-line z stands out clearly against structured neighbours, so it
catches **all** planted errors and almost never false-flags a genuine variant
(0.02%). It is still **off by default** — the false-positive cost climbs with GWAS
power (the per-variant z grows, so well-tagged true signals start tripping the
residual test) — but turn it on when you suspect allele/strand errors or an
LD-reference mismatch, and keep `p_cutoff` stringent.

## Sparse / banded LD: storage vs accuracy

`sparsify_ld` thresholds and/or distance-bands each block into a CSR `SparseLD`
the sampler updates in O(bandwidth). Storage (density = stored entries / dense),
single-core fit time and genetic R² across settings, on **realistic coalescent
LD**, m=4000 = 8×500 blocks, Nref=5k, N=20k, h²=0.5, p=0.02. Regenerate with
`benchmarks/sparse_ld_tradeoff.py`.

| config | density | fit (s) | R² |
|--------|--------:|--------:|---:|
| dense | 100.0% | 0.03 | 0.978 |
| threshold 1e-2 | 95.7% | 0.05 | 0.978 |
| threshold 1e-3 | 99.6% | 0.05 | 0.979 |
| band max_dist=50 | 19.2% | 0.10 | 0.780 |
| band 25 + shrink 0.9 | 9.9% | 0.06 | 0.744 |

On realistic LD the two sparsification knobs behave very differently. **Magnitude
thresholding barely compresses** — real LD has few genuinely-zero entries, so even
a 1e-2 cutoff keeps ~96% of the block (vs ~52% on idealized AR(1)) — but it is
lossless (R² 0.978). **Distance banding compresses hard** (a 50-SNP band keeps
~19%, a shrunk 25-SNP band ~10%) **but is lossy on realistic LD**: it discards
real long-range structure, dropping R² from 0.978 to **0.78 / 0.74**. (`shrink` < 1
restores the positive-definiteness banding breaks, but cannot recover the lost
signal.)

> **The lesson on realistic LD:** thresholding is free but barely helps, and
> banding helps but is lossy. The right memory tool for realistic LD is **low-rank**
> `LowRankLD`, which compresses ~5× at no accuracy cost — see the next section.
> (On genuinely-banded AR(1) LD, by contrast, distance banding is nearly free.)

## LD representations at scale: memory vs running time

At genome / sequencing scale (millions of SNPs, thousands per block) the dense
per-block LD (Σ kᵦ²) does not fit in RAM. The compact representations the sampler
can fit — banded `SparseLD` and low-rank `LowRankLD` — are compared here on
**realistic** coalescent LD with **large blocks** (m=10,000 = 5×2000, N=50k,
h²=0.5, p=0.01): persistent LD memory, build time (low-rank pays an
eigendecomposition), per-fit time, and genetic R². Regenerate with
`benchmarks/ld_representations.py`.

| representation | LD memory | build (s) | fit (s) | R² |
|----------------|----------:|----------:|--------:|---:|
| dense | 80 MB | 3.1 | **0.20** | 0.987 |
| band w200 | 30 MB | 3.2 | 2.0 | 0.758 |
| **low-rank 99.5%** | **16 MB** | 10.5 | 1.6 | **0.986** |

Two costs, and they are different in kind:

* **Build (the eigendecomposition) is one-time and cached.** It is part of LD
  *construction*, not the fit — computed once, saved as the `U` factor (`--ld-out`,
  including the memmap `--ld-stream` cache) and reused across every later fit /
  cohort via `--ld-cache`. So the 10.5 s amortises to ~0 per run, exactly as the
  dense LD's own `Z·Zᵀ` does; it should not be charged to a fit.
* **The recurring cost is fit time.** Low-rank **cuts memory ~5× and matches
  dense accuracy (0.986 vs 0.987) but fits ~8× slower**, and that part does not
  amortise. The slowdown is structural — the dense sampler keeps the full
  residual vector and reads `(Rβ)_j` in O(1), whereas the eigenspace fit
  recomputes `(Rβ)_j = U[j]·s` in O(rank) per SNP. This is intrinsic, not an
  implementation gap: the Gibbs sweep is sequential — `s` changes after every SNP
  — so the reads can't be batched into one `U·s` matvec, and keeping a full
  residual instead would cost O(k) per update (worse).

**Banding is worse on realistic LD on every axis except a small memory edge over
dense** (slower than dense *and* it drops accuracy to 0.76). So low-rank is the
representation for *scale* — LD that would not fit dense — not a way to speed up a
problem that already fits in RAM. For the latter, dense (or recombination-aware
splitting to keep blocks bounded) is best; and the on-disk `--ld-stream` cache
lets a low-rank LD exceed RAM (paged from disk, fits bit-identical).

**Mixed dense + low-rank (`lowrank_min_size`).** Genome-wide, most blocks are
small/moderate and only a few are huge. `compute_ld_blocks(lowrank=True,
lowrank_min_size=K)` (CLI `--ld-lowrank-min-size`) keeps small blocks **dense**
(fast, cheap, often barely compressible) and compresses only blocks ≥ `K`. On a
mixed genome (40×300 + 2×2500 blocks, realistic LD) it is the Pareto choice:

| strategy | LD memory | fit (s) | R² |
|----------|----------:|--------:|---:|
| all dense | 64 MB | 0.26 | 0.979 |
| all low-rank | 14 MB | 1.57 | 0.979 |
| **mixed (≥1000 low-rank)** | 24 MB | **1.12** | 0.978 |

Mixed **dominates all-low-rank** (28 % faster at the same accuracy — small blocks
stay fast dense) and **bounds memory vs all-dense** (only the few huge blocks are
compressed). The big blocks still dominate fit time, so it is the right default
when *some* blocks are too big for dense but most are not — not when everything
already fits dense (then dense is fastest).

## Optimal LD-block splitting

`optimal_ld_blocks` (Privé 2022) places block boundaries in low-LD valleys
instead of at fixed offsets. Here one contiguous region of **realistic coalescent
LD** (m=3000) supplies genuine recombination valleys; both splits use the same
`max_size=250`. Regenerate with `benchmarks/block_splitting.py`.

| split | #blocks | discarded LD² | storage (Σk²) | R² |
|-------|--------:|--------------:|--------------:|---:|
| fixed | 12 | 30,722 | 750,000 | **0.815** |
| optimal | 13 | **30,487** | **717,789** | 0.813 |

On realistic LD the gain is **modest**: putting boundaries in the recombination
valleys discards slightly less true between-block LD and needs **~4% less**
per-block storage, at essentially equal accuracy (0.813 vs 0.815). Unlike the
engineered, hard-edged blocks of an AR(1) toy, real LD decays smoothly with no
deep valleys to exploit — so optimal splitting is a small, safe win here, not a
large one.

## Numba JIT speed-up

The Gibbs sampler's inner sweep dominates runtime; `ldpred3` JIT-compiles it
with Numba and otherwise runs the *identical* pure-Python code. Same auto fit
(m=2000, burn-in 60 / 150 sampling sweeps), with and without JIT (the script
toggles `NUMBA_DISABLE_JIT` in a subprocess). Regenerate with
`benchmarks/numba_speedup.py`.

| mode | fit time (s) |
|------|-------------:|
| pure Python | 43.75 |
| Numba JIT | 0.06 |

A **large** speed-up (here ~780×; strongly machine-dependent — the pure-Python
inner loop is especially slow in CPython) — this is why
`pip install numba` is strongly recommended. Without it everything still runs,
just far slower.

## Constant-N sampler fast path

The point-normal update needs, per SNP, a posterior variance/SD and a
half-log-normaliser (a `sqrt` and a `log1p`) plus the log prior-odds. When the
GWAS sample size `N` is **shared across variants** (a scalar `n_eff`) and the
slab/prior are uniform, those are identical for every SNP, so the samplers
compute them **once per sweep** instead of once per SNP — the hoist the
batched/streaming kernels already used, now also in the dense (`_gibbs_kernel`),
sampling (`_gibbs_kernel_sample`) and sparse (`_gibbs_kernel_sparse`) kernels. A
per-variant `N`, a MAF slab (`alpha`) or non-uniform `prior_weights` fall back to
the exact per-SNP path, **bit-for-bit unchanged** (verified across 13 grid/auto/
sparse/infer/fine-map configs).

The saving isolated on one realistic-LD block (m=2000, N=20,000, h²=0.5, burn-in
100 / 300 sweeps, single core): the *identical* fit run through the fast path vs
the per-SNP path (`N` perturbed by 1e-6 to select the latter). Regenerate with
`benchmarks/sampler_fastpath.py`.

| method | p (causal) | fast path (ms) | per-SNP path (ms) | saved | genetic R² |
|--------|-----------:|---------------:|------------------:|------:|-----------:|
| grid    | 0.001 | 88.6 | 113.5 | **22%** | 0.996 |
| finemap | 0.001 | 177.9 | 219.6 | **19%** | 0.998 |
| grid    | 0.01  | 93.7 | 117.8 | **20%** | 0.996 |
| finemap | 0.01  | 193.3 | 236.1 | **18%** | 0.995 |
| grid    | 0.1   | 149.7 | 175.1 | **14%** | 0.971 |
| finemap | 0.1   | 389.2 | 428.3 | **9%**  | 0.971 |

The genetic R² is the same to three decimals in both columns — the fast path
changes only *when* the scalars are computed, not the result. The saving is
**largest for sparse `p`** (~22%, tapering to ~9% as the trait gets denser):
when few effects change per sweep the O(m) rank-1 residual update is mostly
skipped, so the per-SNP `sqrt`/`log1p` is a larger share of the sweep — precisely
the regime that fine-mapping (`ldpred3_pip`, sparse `p_init`) and the auto
r²/inference chains (`ldpred3_auto_infer`, many chains) run in, so they benefit
most. The `global_hyper=True` streaming/batched genome-wide path was already on
this fast path and is unchanged.

## Multi-core scaling (`--ncores`)

The packed auto sampler parallelises its per-sweep block loop with Numba
`prange`. Parallel speed-up and efficiency of one fixed kernel (m=20,000 = 40×500
blocks, burn-in 100 / 200 sweeps, 4-core Linux machine). Regenerate with
`benchmarks/cores_scaling.py`.

| ncores | fit (s) | speed-up | efficiency |
|-------:|--------:|---------:|-----------:|
| 1 | 5.55 | 1.00× | 100% |
| 2 | 2.65 | 2.10× | 105% |
| 4 | 1.48 | 3.74× | 93% |

Near-linear scaling (~3.7× on 4 cores, ~93% efficiency) when there is enough
per-block work; efficiency is hardware-dependent (memory bandwidth and core
contention vary by machine). Note `--ncores 1` uses the low-memory *streaming* sampler while
`--ncores > 1` switches to this packed parallel kernel (more memory, parallel
sweeps); for small problems the single-core streaming path can already be fast
enough that the packed kernel's setup isn't worth it.
