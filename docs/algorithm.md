# Algorithm internals & LD options

## Global hyper-parameters for `-auto`

`ldpred3_by_blocks(method="auto")` estimates `h2` and `p` **globally** by default
(`global_hyper=True`): the sampler **streams the LD blocks one at a time** (a
jitted per-block sweep), pooling the causal count and genetic variance across all
variants each iteration (as in bigsnpr). Estimating them per block instead
(`global_hyper=False`) is noisy when blocks hold few causal variants and **loses
accuracy at genome scale**. Streaming keeps the fast dense contiguous update,
has a constant-N fast path, and — critically — never materialises a packed
genome-wide LD matrix, so peak memory is just the LD plus O(m) state. At **2M
SNPs** this brought global auto from 137 s / 11.4 GB (packed) down to
**21 s / 0.34 GB** (~6× faster, ~34× less memory). On a 1M-SNP simulation
(block-diagonal LD, p=0.001, vs R `bigsnpr`):

| `-auto` variant | predictive R² | bigsnpr |
|-----------------|---------------|---------|
| global (default) | **0.941** | 0.942 |
| per block | 0.848 | 0.942 |

`inf` and `grid` already match bigsnpr (0.080/0.080 and 0.942/0.942 in the same
run); global hyper-parameters bring `-auto` to parity too.

## Optimal LD block splitting

Fixed-size LD blocks cut arbitrarily through high-LD regions. `optimal_ld_blocks`
implements [Privé (2022), *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btab519)
(`snp_ldsplit`): a dynamic program that places boundaries to **minimise the
squared LD discarded between blocks**, subject to a maximum block size — so cuts
land in low-LD valleys (recombination hotspots).

```python
blocks, discarded_ld2 = optimal_ld_blocks(corr, max_size=1000, window=300)
```

On a simulated chromosome with recombination hotspots (within-300 LD window):

| scheme | max block | LD retained |
|--------|-----------|-------------|
| fixed 500 | 500 | 87.7 % |
| optimal (max 500) | 499 | **94.4 %** |
| fixed 1000 | 1000 | 94.6 % |
| optimal (max 1000) | 900 | **97.7 %** |

Optimal blocks discard less than half the LD of fixed blocks of the same size —
and `optimal(max 500)` retains as much LD as `fixed(1000)`, i.e. the same fidelity
at half the block size (≈4× less per-block O(k²) work and memory). The downstream
prediction gain is modest; the win is mainly computational/validity, as in the
paper.

## Sparse / banded LD

Real LD is banded — most off-diagonal entries are ~0 — so the LD can be stored
sparse (CSR) and the sampler/solver need only touch non-zero neighbours
(O(bandwidth) instead of O(block_size)). Build a `SparseLD` with `sparsify_ld`
and pass it to any model (or `ldpred3_by_blocks(..., sparsify=True)`):

```python
from ldpred3 import sparsify_ld, ldpred3_inf, ldpred3_auto
ld = sparsify_ld(corr, threshold=1e-3)        # drop |r| < 1e-3 (and/or max_dist=…)
beta = ldpred3_inf(ld, beta_hat, n_eff, h2)   # sparse CG solve; samplers also accept ld
```

On a clean population AR(1) block (m=4000, 0.47 % density) this gives, with
**identical results** (r = 1.000):

| method | dense | sparse | speedup |
|--------|-------|--------|---------|
| inf  | 2.77 s | 0.006 s | **444×** (CG vs dense O(m³) solve) |
| grid | 0.131 s | 0.070 s | 1.9× |
| auto | 0.138 s | 0.071 s | 1.9× |

At genome / sequencing scale (millions of SNPs, thousands per block) *all* blocks
are held in RAM, so persistent storage = Σ kᵦ² for dense blocks (≈160 GB for 10M
SNPs in 4000-blocks). `compute_ld_blocks(sparse=True, max_dist=w)` stores each
block banded (built dense transiently, then discarded), and the streaming auto
sampler fits `SparseLD` blocks directly via a CSR per-sweep kernel
(`_gibbs_one_sweep_sparse`) — so the accurate global-hyper fit runs at
O(k·bandwidth). The on-disk LD cache (`--ld-out` / `--ld-cache`) also stores the
CSR. Enable end-to-end with `--ld-sparse` (`--ld-max-dist w`).

**Banding is only lossless when LD really is short-range.** On realistic LD it is
**not**: banding discards genuine long-range structure. Measured on coalescent LD
(m=6000, 1000-SNP blocks; `benchmarks/ld_memory_scaling.py`,
`benchmarks/ld_shrink_large_blocks.py`):

| representation | genetic R² | memory |
|----------------|-----------:|-------:|
| dense | 0.992 | 100% |
| band w200 | 0.823 | 72% |
| **low-rank (PCs), 99.5% var** | **0.993** | **24%** |

So for realistic / sequencing-scale LD, **low-rank (eigen/PC) compression is the
right memory tool** — it matches dense accuracy at ~4× compression where banding
both loses accuracy and compresses less (SBayesRC's representation). This is
implemented: `compute_ld_blocks(lowrank=True, lowrank_variance=…)` (CLI
`--ld-lowrank`) stores each block as a `LowRankLD` (`R ≈ U Uᵀ`, unit diagonal,
top eigenvectors), and the global-hyper streaming auto sampler fits it **in the
r-dimensional eigenspace** via `_gibbs_one_sweep_lowrank`: it carries the block
residual as `s = Uᵀβ` (length r), recovers `(Rβ)_j = U[j]·s`, updates `s += Δ·U[j]`
on each effect change (O(r)), and uses `βᵀRβ = ‖s‖²` — so the fit is O(k·r) in
time and memory with no dense k×k. `ldpred3_inf` solves `LowRankLD` via Woodbury,
and the on-disk cache stores the `U` factor. Banding remains useful for genuinely
banded LD (e.g. AR(1)-like / some array data), and recombination-aware splitting
(`optimal_ld_blocks`) keeps blocks bounded regardless.

### On-disk LD streaming (`--ld-stream`)

Even compact blocks are all held in RAM by default. `save_ld_blocks(..., mmap=True)`
(pipeline `--ld-out --ld-stream`) writes the block payloads into one
memory-mappable `<cache>.dat.npy` sidecar; a later `--ld-cache` run then loads
each block as a **memmap view**, and the streaming sampler reads one block at a
time. Semantics to be precise about:

* The LD is **file-backed**, not on the Python heap — so it is **reclaimable
  under memory pressure and shareable across processes**, and an LD that exceeds
  RAM still runs (the OS pages it). Fits are bit-identical to the in-RAM path.
* On a machine with ample free RAM the OS caches all pages, so *resident* memory
  is similar to in-RAM — the benefit is enabling data > RAM and avoiding a
  per-process heap copy, not a smaller RSS when memory is plentiful.

Build once (ideally with `lowrank=True` to keep the cache small), then reuse it
cheaply across runs/cohorts: `--ld-out cache.npz --ld-stream` then
`--ld-cache cache.npz`. Supports dense and low-rank caches.

Two more caveats:

* **In-sample LD has a noise floor (~1/√N)** that fills the matrix, so magnitude
  thresholding alone won't sparsify it — band by distance (`max_dist=`) to drop
  the spurious far-apart entries, as LDpred2/bigsnpr do.
* **Hard banding can break positive-definiteness**, which destabilises the
  *fixed-h² sampler* (`grid` can diverge; `auto` self-limits via its h² clamp;
  `inf`'s ridge is unaffected). Use `sparsify_ld(..., shrink=<1)` to restore
  diagonal dominance, or supply an already-valid windowed LD matrix.

## Size-aware LD shrinkage (finite reference panels)

A block's sample LD estimated from `n_ref` reference individuals carries noise
that grows with the block size `k` relative to `n_ref` (Marchenko–Pastur: the
sample eigenvalues spread/inflate as `k/n_ref` grows). Small blocks
(`k ≪ n_ref`) are well estimated; **large blocks (`k` approaching or exceeding
`n_ref`) are noise-dominated**, and that noise makes the Gibbs sampler over-fit
and inflate `h²`.

`shrink_ld_blocks(blocks, n_ref)` (pipeline `--ld-shrink`) shrinks each block
toward the identity by `alpha = min(max_shrink, k/n_ref)` —
`R ← (1-alpha)·R + alpha·I`, diagonal kept 1 — so **large blocks are regularised
while small, well-estimated ones are left essentially untouched**. This is a
*uniform eigenvalue shrinkage* (`λ → (1-alpha)λ + alpha`).

A note on the spectral alternatives we tried (see
`benchmarks/ld_shrink_large_blocks.py`): on a finite panel,

* **PC truncation** (keep the top eigenvectors to a variance threshold, à la a
  naive low-rank LD) does **not** help as a drop-in — it preserves the
  Marchenko–Pastur-*inflated* top eigenvalues and, when `k < n_ref`, discards
  real signal directions.
* **RMT eigenvalue clipping** (flatten the noise bulk below the MP edge) helps,
  but **less** than the simpler size-aware shrinkage above.

so the shipped lever is the size-aware shrinkage. (Capturing SBayesRC's full PC
benefit would mean running the sampler in the *eigenspace* with a low-rank
likelihood rather than reconstructing `R̃` for the existing sampler — a larger
change, not yet implemented.)

## Fewer iterations: warm start & adaptive stopping

`ldpred3_grid`/`ldpred3_auto` accept:

* `warm_start=True` — initialise the chain from the LDpred3-inf solution instead
  of zeros, shortening burn-in. It pays for one `inf` solve up front, so it only
  helps when burn-in/mixing dominates **and** inf is cheap — i.e. paired with the
  sparse LD backend (CG inf). With a *dense* O(m³) inf solve it can cost more
  than it saves.
* `tol=<x>` (+ `check_every`) — **adaptive stopping**: end sampling once the
  running posterior mean's relative RMS change over `check_every` sweeps drops
  below `tol`, instead of always running `num_iter`. `AutoResult.n_iter` reports
  how many sweeps were used. On a fast-mixing block this reached the same
  accuracy as a fixed 2000-iteration run in ~100 iterations (~10× faster), with
  no loss (corr 1.000 vs the long run).

## Per-variant priors (annotation-informed, SBayesRC-style)

`ldpred3_grid` / `ldpred3_auto` accept `prior_weights` — a per-SNP relative
causal propensity from functional annotations. Each SNP's causal probability
becomes `p_j = p · prior_weights[j]` (clamped to `(0,1)`); with mean-1 weights
the expected causal count and `h²` stay coherent. SNPs in functionally
important regions (coding, conserved, enhancers, …) thus get a higher prior of
being causal — the core idea of SBayesRC.

```python
ldpred3_grid(corr, beta_hat, n_eff, h2, p, prior_weights=w)   # w_j >= 0, mean ~1
```

This injects **new information**, so unlike a change of slab shape it can
genuinely raise accuracy — but only when the annotations are trustworthy:

- **Informative** weights raise held-out R² (the gain grows as power drops, when
  more SNPs are borderline). On a single binary annotation in simulation the
  lift is small (~1–2% relative); SBayesRC's larger real-data gains come from
  many S-LDSC-calibrated annotations at genome scale.
- **Misleading / uninformative** weights *lower* accuracy, more so at low power.
  It is a "garbage-in" feature.
- Equal weights reproduce the uniform-`p` point-normal model bit-for-bit.

Only the inclusion probability is re-weighted here; the slab *variance* is left
global. Scaling the effect-size variance by annotation is a further knob, but it
must match a real annotation–effect-size relationship or it over-shrinks (it
hurt in simulations where effect size was annotation-independent).

### MAF-dependent slab variance (`alpha`)

A second, *allele-frequency*-based knob scales each variant's slab **variance**
(not its inclusion probability) by `[2f(1-f)]^(1+alpha)`, where `f` is the allele
frequency. This is the LDpred2-auto `alpha`/`use_MLE` prior of
[Privé et al. (2023)](https://doi.org/10.1016/j.ajhg.2022.10.010): it relaxes the
standard-genotype assumption that per-allele effect variance is exactly
proportional to `1/[2f(1-f)]` (i.e. `alpha = -1`, the default and the original
point-normal model). Real traits often prefer `alpha` somewhere in `[-1, -0.5]`,
putting relatively more variance on common variants.

```python
ldpred3_grid(corr, beta_hat, n_eff, h2, p, af=freq, alpha=-0.5)   # auto / grid
run_ldpred3_prs(sumstats, plink, method="auto", alpha=-0.5)        # or --alpha
```

- `alpha = -1` (default) leaves the sampler bit-for-bit unchanged — the weights
  are mean-normalised so the total `h²` budget is preserved.
- It is a change of slab *shape* only, so — unlike annotation priors — it injects
  no new information; it helps only when the true effect–MAF coupling departs from
  `alpha = -1`. Privé et al. select it by maximising the model's own likelihood
  across a small `alpha` grid; here it is a user-set knob.
- Runs through the dense per-block `auto` / `grid` path only (it forces
  `global_hyper=False` for `auto`); not supported with the multi-chain auto
  estimator, `lassosum2`, `annot`, or compact (sparse / low-rank) LD.

### Learning the annotation map (SBayesRC)

`ldpred3_auto_annot` learns the annotation→prior map *inside* the sampler, so
the weights need not be supplied: each SNP's causal probability is
`p_j = sigmoid(a_jᵀθ)` and `θ` is updated jointly with the effects. Two
strategies (`learn=`):

* **`"eb"`** — empirical-Bayes: a ridge-regularised logistic (Newton/IRLS) step
  on the posterior inclusion probabilities. NumPy-only, fast, stable.
* **`"probit"`** — fully Bayesian: a probit link with Albert–Chib (1993) data
  augmentation (vectorised normal CDF / inverse-CDF), giving a conjugate
  Gaussian draw of `θ`.

```python
from ldpred3 import ldpred3_auto_annot
res = ldpred3_auto_annot(corr, beta_hat, n_eff, annotations=A, learn="eb")
res.beta_est, res.theta      # effects + learned enrichment coefficients
```

A ridge penalty on the non-intercept coefficients keeps it stable with many
collinear annotations; the intercept absorbs the global `p`. In simulation the
sampler **recovers the enrichment of an informative annotation (θ≈+1) and
correctly ignores an irrelevant one (θ≈0)** — so, unlike a *fixed* bad prior, a
learned one automatically down-weights unhelpful annotations (no "garbage-in"
penalty), and the learned θ are directly interpretable as functional-enrichment
estimates.

**Update the map every sweep (`theta_every=1`, the default).** The `θ`-update
and the effect sweep are coupled — `θ` sets the per-SNP `p_j` the sweep uses, and
the sweep's posterior inclusion probabilities feed the next `θ`-update — so they
must co-adapt. With a *lazy* update (e.g. `theta_every=10`) the map lags the
chain and, at low per-SNP power with large `m`, settles at an inflated global `p`
(it reads the prior-smeared inclusion probabilities of null SNPs as signal),
which **over-shrinks** the effects: in the architecture benchmark this made
`annot` fall *below* plain `auto` at N=10k (enriched 0.60 vs 0.64). Updating `θ`
every sweep removes the lag — the learned enrichment reaches its true value and
`annot` recovers to ≥ `auto` everywhere (see
[benchmarks.md](benchmarks.md)). The IRLS step is an `O(m·K²)` solve, so it is
negligible for a handful of annotations; raise `theta_every` only when `K` is
large enough (≳50) that the per-sweep `θ` cost rivals the effect sweep.

Two further options complete the SBayesRC picture:

* **`learn_variance=True`** additionally learns an annotation → effect-*variance*
  map `σ²_j ∝ exp(a_jᵀφ)` (returned in `.phi` / `.variance_enrichment`). Being
  learned, `φ` collapses to ~0 when effect size is annotation-independent (no
  harm) and turns positive when functional SNPs carry larger effects.
* **`ldpred3_auto_annot_blocks`** is the genome-wide streaming version: the maps
  are global but the effect sweeps run one LD block at a time, so the
  genome-wide LD is never materialised (it matches the dense version on
  block-diagonal LD). This is what the pipeline's `--method annot` uses.

## lassosum2 — penalised regression (`method="lassosum2"`)

`lassosum2` is not a Bayesian sampler: it minimises a **penalised least-squares**
objective directly on the summary statistics (Mak 2017; Privé 2021),

```
argmin_β   (1−s)·βᵀRβ − 2·β̂ᵀβ + s·‖β‖² + 2λ·‖β‖₁
```

where `R` is the block LD, `β̂` the standardized marginal effects, `s ∈ (0,1]` is
an LD-shrinkage that blends `R` toward the identity, and `λ` is the L1 penalty
that drives a **sparse** solution (many exact zeros). It is solved by
coordinate descent, warm-started down a log-spaced `λ` path from `λ_max = max|β̂|`
(all-zero) so the whole path is cheap.

**No validation cohort needed.** The `(s, λ)` grid is scored by
**pseudo-validation** — a correlation between the candidate effects and the
summary statistics that estimates out-of-sample fit from the sumstats + LD alone
— and the best cell is returned (guarded so a degenerate over-fit score ≤ 1
cannot win). It is a fast, MCMC-free, sparse complement to `auto`: no single
method dominates every architecture, so fitting both and keeping the better
pseudo-validation is cheap insurance. Dense LD blocks only. CLI:
`--method lassosum2`; API: `lassosum2(blocks, beta_hat)` → `Lassosum2Result`.

## Laplace prior — the Bayesian lasso (`method="laplace"`)

The lasso (`lassosum2`) is the posterior *mode* under a Laplace (double-
exponential) prior on the effects; `ldpred3_laplace` samples the posterior
**mean** of that same prior — the proper Bayesian shrinkage estimator, generally
a better predictor than the mode. It is the Bayesian counterpart of `lassosum2`.

It uses the normal / exponential scale-mixture representation of the Laplace
(Park & Casella 2008): a per-SNP latent scale `τ_j²` with

```
β_j | τ_j²  ~  N(0, τ_j²)         τ_j²  ~  Exponential(λ²/2)   =>   β_j ~ Laplace(λ)
```

so each Gibbs sweep is the *same* Gaussian per-SNP conditional the point-normal
sampler already runs (prior variance `τ_j²` instead of the slab), plus an
Inverse-Gaussian draw for `1/τ_j² ~ InvGauss(λ/|β_j|, λ²)`. The estimate is the
Rao-Blackwellised average of the per-SNP conditional means over the post-burn-in
sweeps.

**Self-tuning `λ`.** The global shrinkage is set by marginal maximisation (an EM
step), `λ² = 2k / Σ τ_j²`, which converges to the value matching the fitted total
variance — no penalty grid, no validation cohort. A naïve fully-Bayesian `λ` (a
conditional Gamma draw) does *not* work here: with the extra scale-mixture layer
it drifts to the hyper-prior's mean independently of the data and mis-shrinks
(see [benchmarks](benchmarks.md#laplace-prior-the-bayesian-lasso-methodlaplace)).

Unlike the spike-and-slab there is no point mass at zero, so the posterior mean
is **dense** — heavier-tailed shrinkage than the infinitesimal Gaussian, but it
cannot concentrate on a few causals the way the point-normal does. It matches
`inf` on a truly infinitesimal trait, beats `lassosum2` across architectures
(the mean over the mode), and trails `auto`/`grid` on sparse traits. Dense LD
blocks only (per-block, like `inf`/`grid`). CLI: `--method laplace`.

## Bivariate (two-trait) LDpred3

`ldpred3_auto_bivariate` jointly fits **two traits that share one LD reference**.
Each variant takes one of **four** states — causal for neither trait, trait 1
only, trait 2 only, or **both** — with probabilities `(π₀₀, π₁₀, π₀₁, π₁₁)`. A
trait-1-causal effect is `N(0, s₁)`, a trait-2-causal one `N(0, s₂)`, and a
*both*-causal pair is `N(0, Σ)` with `Σ = [[s₁, s₁₂],[s₁₂, s₂]]`; the
off-diagonal `s₁₂` is the genetic covariance and the only place the traits
couple. Each Gibbs step evaluates the four bivariate-Gaussian likelihoods of the
residual estimate, samples a state, and draws the effects; `π` and `s₁₂` are
re-estimated each sweep, and `r_g = β₁ᵀRβ₂ / √(h²₁h²₂)` is reported.

```python
from ldpred3 import ldpred3_auto_bivariate
res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2)
res.beta1_est, res.beta2_est      # adjusted effects for the two traits
res.h2, res.rg                    # (h2_1, h2_2) and the genetic correlation
```

**Why per-trait states (and not one shared causal indicator).** An earlier
prototype used a single shared indicator (both traits causal at the same SNPs).
That helps when the assumption holds but **hurts** badly when it doesn't — with
disjoint causal variants it forced sharing and dropped the weak trait's accuracy
by ~0.1. The four-state model *learns* whether causal variants co-occur (`π₁₁`),
so disjoint traits drive `π₁₁ → 0` and the joint fit reduces to the independent
ones. Two further safeguards keep it honest: each trait's slab variance is capped
by its own **univariate** heritability (the weak trait's variance is otherwise
under-identified and inflates by borrowing from the strong one), and the variance
updates are damped.

The benchmark is **realistic**: the GWAS is generated from the true population
(coalescent) LD but fitted with an LD matrix estimated from a finite reference
panel (`Nref=2000`). For a genuinely under-powered trait 2 (N=2000, polygenic)
vs a well-powered trait 1 (N=100000), the gain grows with `r_g` and there is **no
harm** at low `r_g` or disjoint architectures:

| architecture | trait-2 alone | trait-2 joint | gain | r_g est |
|--------------|--------------:|--------------:|-----:|--------:|
| shared, r_g=0.0 | 0.641 | 0.636 | −0.005 | +0.02 |
| shared, r_g=0.3 | 0.647 | 0.641 | −0.006 | +0.39 |
| shared, r_g=0.6 | 0.655 | 0.694 | +0.039 | +0.67 |
| shared, r_g=0.9 | 0.658 | 0.830 | **+0.173** | +0.89 |
| disjoint causal | 0.630 | 0.610 | −0.020 | −0.08 |

The benefit is **real and large only where it should be** — a weak trait highly
correlated with a strong one — and negligible otherwise. It scales with how
under-powered trait 2 is: at N=1000 the rg=0.9 gain reaches ~+0.28, while for an
already well-powered trait 2 there is little to borrow and a small overhead, so
use the joint fit to boost an under-powered trait. (An earlier "fit with the true
LD" benchmark overstated the gains — they shrink markedly under realistic
reference-panel LD.) `ldpred3_auto_bivariate_blocks` is the streaming genome-wide
version; both GWAS must use the same LD/ancestry, and sample overlap is handled
via `cross_corr` (default 0). Regenerate with `benchmarks/bivariate_demo.py`.

**Genetic correlation vs bivariate LDSC.** The reported `r_g` has an independent
cross-check in `ldsc_rg` (cross-trait LD Score regression). Under the same
realistic reference-panel LD both are roughly unbiased from the same summary
statistics; bivariate LDpred3 is ~2× more precise (it uses the full LD
likelihood):

| true r_g | bivariate LDSC | bivariate LDpred3 |
|---------:|---------------:|------------------:|
| 0.0 | −0.04 ± 0.24 | −0.01 ± 0.15 |
| 0.3 | 0.29 ± 0.18 | 0.30 ± 0.16 |
| 0.6 | 0.59 ± 0.15 | 0.60 ± 0.13 |
| 0.9 | 0.86 ± 0.07 | 0.90 ± 0.04 |

(With the *true* LD the SEs are several-fold smaller and LDpred3's precision edge
larger; the reference-panel mismatch is what makes both noisier and narrows the
gap — the realistic picture.) Regenerate with
`benchmarks/compare_bivariate_rg.py`.

## Robustness: `allow_jump_sign`

`ldpred3_grid` / `ldpred3_auto` / `ldpred3_auto_infer` accept
`allow_jump_sign` (default `True`). Setting it `False` forbids a variant's
effect from flipping sign within a single Gibbs step (a sampled effect of the
opposite sign to the current one is set to zero instead). On noisy or
ill-conditioned LD this is a major source of divergence, and the guard — as in
the LDpred2-auto inference workflow (Privé et al.) — keeps the chain bounded.
It is exact for well-behaved problems (no flips occur) and only bites when the
sampler would otherwise oscillate.

The guard is honoured on every model path, including the **streaming** genome-wide
auto (`ldpred3_by_blocks(global_hyper=True)` and `--infer` on per-block / banded /
low-rank LD), so it is available at genome scale where ill-conditioned reference
LD makes it most useful — not just on a single dense block.

## Performance & Numba

The Gibbs sampler maintains a running `R @ beta` vector (per-SNP residual is an
O(1) lookup; the O(m) rank-1 update is only paid when an effect changes), so it
scales sub-quadratically in block size for sparse traits. The rank-1 update is
the bandwidth-bound hot path, so it runs as a fused element loop over a
**single-precision** LD row (float32 halves the memory traffic, ~2× faster, with
no meaningful accuracy cost — and matches bigsnpr, which also stores LD in single
precision). The effects and the `R @ beta` accumulator stay in float64.

The posterior-mean estimate is **Rao-Blackwellized** (as in the original
LDpred): each sweep accumulates the conditional expectation
`E[beta_j | rest] = P(causal) · posterior_mean` rather than the sampled draw.
The sampled value still drives the Markov chain; only the *estimate* uses the
expectation, which has much lower Monte-Carlo variance (≈6–13× in fast-mixing
regimes), so fewer iterations are needed for the same accuracy. In extreme-LD
regions the benefit is smaller because chain mixing, not sampling noise, is the
bottleneck.

[Numba](https://numba.pydata.org/) is strongly recommended: when installed, the
inner sampler is JIT-compiled (and cached) for a large speed-up. Without it the
code still runs and gives identical results, but the sampler falls back to plain
Python loops and is much slower — fine for small problems / CI, not for
genome-wide runs.

### Reproducibility

The Gibbs samplers are seeded and deterministic, but the **LD construction and
the inference post-processing use BLAS** (`Z.T @ Z`, the dense matmuls, `eigh`),
whose results can differ in the last bits across BLAS thread counts. For
bit-reproducible LD / r² across machines, pin the threads before running:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

(the test suite sets these for exactly this reason). The Gibbs effect estimates
themselves are reproducible from the seed regardless.

## References

Methods implemented in LDpred3 and the papers they come from.

**Core PRS models**

- **LDpred (original).** Vilhjálmsson BJ, Yang J, Finucane HK, *et al.* "Modeling
  Linkage Disequilibrium Increases Accuracy of Polygenic Risk Scores."
  *Am J Hum Genet* 97(4):576–592 (2015). doi:10.1016/j.ajhg.2015.09.001 —
  the point-normal (spike-and-slab) LD-aware PRS and the infinitesimal model.
- **LDpred2.** Privé F, Arbel J, Vilhjálmsson BJ. "LDpred2: better, faster,
  stronger." *Bioinformatics* 36(22–23):5424–5431 (2020).
  doi:10.1093/bioinformatics/btaa1029 — the `grid`, `auto`, and sparse variants,
  the `N(Rβ, R/N)` sampling model, and the `bigsnpr` reference implementation.
- **LDpred2-auto (disease architecture & inference).** Privé F, Albiñana C,
  Pasaniuc B, Vilhjálmsson BJ. "Inferring disease architecture and predictive
  ability with LDpred2-auto." *Am J Hum Genet* 110(12):2042–2055 (2023).
  doi:10.1016/j.ajhg.2023.10.010 — the robust multi-chain estimator
  (`--auto-chains`), the `use_MLE`/`α` MAF-dependent prior (`--alpha`), and the
  in-sample `h²` / polygenicity / predictive-`r²` estimators (`--infer`).

**Penalised-regression / Laplace-prior PRS**

- **lassosum.** Mak TSH, Porsch RM, Choi SW, Zhou X, Sham PC. "Polygenic scores
  via penalized regression on summary statistics." *Genet Epidemiol*
  41(6):469–480 (2017). doi:10.1002/gepi.22050 — the L1-penalised summary-statistic
  objective and pseudo-validation; `lassosum2` is the `bigsnpr` re-parameterisation
  (Privé *et al.* 2020).
- **Bayesian lasso.** Park T, Casella G. "The Bayesian Lasso." *J Am Stat Assoc*
  103(482):681–686 (2008). doi:10.1198/016214508000000337 — the normal/exponential
  scale-mixture Gibbs sampler and the marginal-maximisation `λ` update behind
  `method="laplace"`.
- **Scale mixtures of normals.** Andrews DF, Mallows CL. "Scale Mixtures of Normal
  Distributions." *J R Stat Soc B* 36(1):99–102 (1974) — the Laplace-as-mixture
  identity the sampler exploits.
- **Inverse-Gaussian variates.** Michael JR, Schucany WR, Haas RW. "Generating
  Random Variates Using Transformations with Multiple Roots." *Am Stat*
  30(2):88–90 (1976) — the `1/τ²` draw.

**Priors & annotations**

- **MAF/LD-dependent architecture (the `α`/`S` parameter).** Speed D, Cai N, *the
  UCLEB Consortium*, Johnson MR, Nejentsev S, Balding DJ. "Reevaluation of SNP
  heritability in complex human traits." *Nat Genet* 49:986–992 (2017).
  doi:10.1038/ng.3865; and Zeng J, de Vlaming R, Wu Y, *et al.* "Signatures of
  negative selection in the genetic architecture of human complex traits."
  *Nat Genet* 50:746–753 (2018). doi:10.1038/s41588-018-0101-4 — the effect-size /
  allele-frequency coupling formalised as `use_MLE` in LDpred2-auto (2023).
- **SBayesR / SBayesRC (annotation-informed priors).** Lloyd-Jones LR, Zeng J,
  *et al.* "Improved polygenic prediction by Bayesian multiple regression on
  summary statistics." *Nat Commun* 10:5086 (2019).
  doi:10.1038/s41467-019-12653-0; Zheng Z, Liu S, Sidorenko J, *et al.*
  "Leveraging functional genomic annotations and genome coverage to improve
  polygenic prediction of complex traits within and between ancestries."
  *Nat Genet* 56:767–777 (2024). doi:10.1038/s41588-024-01704-4 — the learned
  annotation→prior map behind `--method annot`.

**QC, LD, and summary-statistic corrections**

- **Sumstats misspecification (SD check & per-variant `N`).** Privé F, Arbel J,
  Aschard H, Vilhjálmsson BJ. "Identifying and correcting for misspecifications in
  GWAS summary statistics and polygenic scores." *HGG Adv* 3(4):100136 (2022).
  doi:10.1016/j.xhgg.2022.100136 — the SD-consistency filter and `--impute-n`.
- **DENTIST.** Chen W, Wu Y, Zheng Z, *et al.* "Improved analyses of GWAS summary
  statistics by reducing data heterogeneity and errors." *Nat Commun* 12:7117
  (2021). doi:10.1038/s41467-021-27438-7 — the LD-consistency outlier filter
  (`--dentist`).
- **Optimal LD-block splitting.** Privé F. "Optimal linkage disequilibrium
  splitting." *Bioinformatics* 38(1):255–256 (2022).
  doi:10.1093/bioinformatics/btab519 — `optimal_ld_blocks`.

**Heritability, inference & fine-mapping**

- **LD Score regression.** Bulik-Sullivan BK, Loh P-R, Finucane HK, *et al.*
  "LD Score regression distinguishes confounding from polygenicity in genome-wide
  association studies." *Nat Genet* 47(3):291–295 (2015). doi:10.1038/ng.3211 —
  `ld_scores` / `ldsc_h2` / `ldsc_rg` and `--ldsc-init`.
- **Liability-scale heritability.** Lee SH, Wray NR, Goddard ME, Visscher PM.
  "Estimating Missing Heritability for Disease from Genome-wide Association
  Studies." *Am J Hum Genet* 88(3):294–305 (2011).
  doi:10.1016/j.ajhg.2011.02.002 — `h2_liability`.
- **Approximate Bayes factor (fine-mapping).** Wakefield J. "Bayes factors for
  genome-wide association studies: comparison with P-values." *Genet Epidemiol*
  33(1):79–86 (2009). doi:10.1002/gepi.20359 — the single-SNP ABF in
  `single_signal_finemap`.
