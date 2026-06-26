# Algorithm internals & LD options

## Global hyper-parameters for `-auto`

`ldpred2_by_blocks(method="auto")` estimates `h2` and `p` **globally** by default
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
and pass it to any model (or `ldpred2_by_blocks(..., sparsify=True)`):

```python
from pyldpred2 import sparsify_ld, ldpred2_inf, ldpred2_auto
ld = sparsify_ld(corr, threshold=1e-3)        # drop |r| < 1e-3 (and/or max_dist=…)
beta = ldpred2_inf(ld, beta_hat, n_eff, h2)   # sparse CG solve; samplers also accept ld
```

On a clean population AR(1) block (m=4000, 0.47 % density) this gives, with
**identical results** (r = 1.000):

| method | dense | sparse | speedup |
|--------|-------|--------|---------|
| inf  | 2.77 s | 0.006 s | **444×** (CG vs dense O(m³) solve) |
| grid | 0.131 s | 0.070 s | 1.9× |
| auto | 0.138 s | 0.071 s | 1.9× |

Two important caveats:

* **In-sample LD has a noise floor (~1/√N)** that fills the matrix, so magnitude
  thresholding alone won't sparsify it — band by distance (`max_dist=`) to drop
  the spurious far-apart entries, as LDpred2/bigsnpr do.
* **Hard banding can break positive-definiteness**, which destabilises the
  *fixed-h² sampler* (`grid` can diverge; `auto` self-limits via its h² clamp;
  `inf`'s ridge is unaffected). Use `sparsify_ld(..., shrink=<1)` to restore
  diagonal dominance, or supply an already-valid windowed LD matrix.

## Fewer iterations: warm start & adaptive stopping

`ldpred2_grid`/`ldpred2_auto` accept:

* `warm_start=True` — initialise the chain from the LDpred2-inf solution instead
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

`ldpred2_grid` / `ldpred2_auto` accept `prior_weights` — a per-SNP relative
causal propensity from functional annotations. Each SNP's causal probability
becomes `p_j = p · prior_weights[j]` (clamped to `(0,1)`); with mean-1 weights
the expected causal count and `h²` stay coherent. SNPs in functionally
important regions (coding, conserved, enhancers, …) thus get a higher prior of
being causal — the core idea of SBayesRC.

```python
ldpred2_grid(corr, beta_hat, n_eff, h2, p, prior_weights=w)   # w_j >= 0, mean ~1
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

### Learning the annotation map (SBayesRC)

`ldpred2_auto_annot` learns the annotation→prior map *inside* the sampler, so
the weights need not be supplied: each SNP's causal probability is
`p_j = sigmoid(a_jᵀθ)` and `θ` is updated jointly with the effects. Two
strategies (`learn=`):

* **`"eb"`** — empirical-Bayes: a ridge-regularised logistic (Newton/IRLS) step
  on the posterior inclusion probabilities. NumPy-only, fast, stable.
* **`"probit"`** — fully Bayesian: a probit link with Albert–Chib (1993) data
  augmentation (vectorised normal CDF / inverse-CDF), giving a conjugate
  Gaussian draw of `θ`.

```python
from pyldpred2 import ldpred2_auto_annot
res = ldpred2_auto_annot(corr, beta_hat, n_eff, annotations=A, learn="eb")
res.beta_est, res.theta      # effects + learned enrichment coefficients
```

A ridge penalty on the non-intercept coefficients keeps it stable with many
collinear annotations; the intercept absorbs the global `p`. In simulation the
sampler **recovers the enrichment of an informative annotation (θ≈+1) and
correctly ignores an irrelevant one (θ≈0)** — so, unlike a *fixed* bad prior, a
learned one automatically down-weights unhelpful annotations (no "garbage-in"
penalty), and the learned θ are directly interpretable as functional-enrichment
estimates. It operates on a dense LD matrix; the genome-wide streaming path and
annotation-driven effect-*variance* are the remaining extensions.

## Robustness: `allow_jump_sign`

`ldpred2_grid` / `ldpred2_auto` / `ldpred2_auto_infer` accept
`allow_jump_sign` (default `True`). Setting it `False` forbids a variant's
effect from flipping sign within a single Gibbs step (a sampled effect of the
opposite sign to the current one is set to zero instead). On noisy or
ill-conditioned LD this is a major source of divergence, and the guard — as in
the LDpred2-auto inference workflow (Privé et al.) — keeps the chain bounded.
It is exact for well-behaved problems (no flips occur) and only bites when the
sampler would otherwise oscillate.

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
