# Fine-mapping with LDpred3 PIPs

LDpred3's spike-and-slab Gibbs sampler draws, for every SNP on every sweep, the
posterior probability that the SNP is causal (`postp`). Averaging it over the
post-burn-in sweeps **is** the per-SNP **posterior inclusion probability (PIP)** —
so fine-mapping reuses the exact same engine as PRS, on the same convention

```text
beta_hat = R @ beta + noise,   noise ~ N(0, R / N)
```

where `R` is LD and `N` the GWAS sample size. No separate model: the PIP is a
by-product of the sampler ([`ldpred3/finemap.py`](../ldpred3/finemap.py)).

## Per-locus

```python
from ldpred3 import ldpred3_pip, standardize_betas

beta_std, _ = standardize_betas(beta, beta_se, n_eff)   # GWAS -> standardized
res = ldpred3_pip(R, beta_std, n_eff, coverage=0.95, min_abs_corr=0.5)

res.pip                 # (m,) posterior inclusion probabilities
res.posterior_mean      # (m,) posterior effect means
res.credible_sets       # list[CredibleSet]: variants, coverage, purity, lead
res.n_signals_est       # ~ sum(pip): expected number of causal variants here
```

`R` may be a dense matrix, a banded `SparseLD` or a low-rank `LowRankLD` (compact
blocks are densified, since credible-set purity needs the within-locus
correlations).

**Fixed sparse prior (the default).** Re-estimating polygenicity `p` on one small
locus is unstable and inflates PIPs where there is no signal, so fine-mapping
holds `p` at a sparse value (`p_init=1e-3`, `estimate_p=False`) while the signal
strength `h²` still adapts. With it, a null locus yields **zero** credible sets
and PIPs near 0; a single causal variant gets PIP ≈ 1. Pass `estimate_p=True` to
recover the original per-locus auto behaviour.

## Credible sets and purity

`ldpred3_pip` returns one PIP per SNP (a marginal), not SuSiE's separable
per-effect assignment vectors, so signals are separated by LD: the highest-PIP
variant anchors a signal, its LD neighbours (`|r| ≥ min_abs_corr`) are gathered in
descending PIP order until the cumulative reaches `coverage`, and the set is
dropped if its **purity** (min pairwise `|r|` among members) falls below
`min_abs_corr`. This repeats `round(sum(pip))` times — the expected number of
causal variants in the locus.

**Tie-expansion (calibration).** LDpred3's spike-and-slab picks *one* of a set of
nearly indistinguishable proxies each sweep, so the marginal PIP over-concentrates
and a 95% set can collapse below true coverage. Each credible set therefore also
includes any variant in near-perfect LD (`|r| ≥ tie_r`, default 0.95) with its
lead — proxies the data cannot tell apart. This restores calibration (see the
benchmark below) while keeping sets sharp.

> **Calibration note.** Absolute PIP values depend on the prior; the **calibrated
> deliverable is credible-set coverage** (the 95% set contains the causal variant
> ~95% of the time), not the raw PIP. For tightly-linked multiple signals in one
> LD cluster, flat-PIP separation is weaker than a full per-effect model — a
> future upgrade can separate signals from the per-sweep sampled configurations.

## Benchmark

`benchmarks/finemap_recovery.py` (self-contained, coalescent LD; 400 SNPs/locus,
N=100k, 80 loci/cell) measures coverage, power and resolution vs signal strength:

| per-causal z | coverage | power | median \|CS\| |
|-------------:|---------:|------:|-------------:|
| 4  | 0.95 | 0.26 | 4 |
| 6  | 0.97 | 0.88 | 3 |
| 8  | 1.00 | 0.99 | 2 |
| 10 | 0.98 | 0.99 | 2 |

Credible-set **coverage is ~0.95+** across signal strengths; **power and
resolution improve as the signal strengthens** (median set size shrinks to ~2).
The headline is resolution at matched coverage — against the single-signal ABF
baseline (z=8, 1 causal):

| method | coverage | power | median \|CS\| |
|--------|---------:|------:|-------------:|
| LDpred3-PIP | 1.00 | 0.99 | **2** |
| ABF (single signal) | 1.00 | 1.00 | 380 |

ABF "covers" only by dumping the whole locus into one set; LDpred3-PIP localizes
to a handful of variants at the same coverage. Coverage is also robust to a finite
LD reference panel (1.00 clean → 0.95 at Nref=500).

### Across genetic architectures and methods

The same three metrics across five locus architectures (coalescent LD, 60
loci/cell, target z=7), comparing LDpred3-PIP, the ABF baseline and a naive
**marginal top-SNP** (the most significant variant as a size-1 set). Regenerate
with `benchmarks/finemap_architectures.py`.

| architecture | method | coverage | power | median \|CS\| |
|--------------|--------|---------:|------:|-------------:|
| single | LDpred3-PIP | 0.97 | 0.93 | **2** |
| single | ABF | 1.00 | 1.00 | 380 |
| single | marginal-top | 0.72 | 0.72 | 1 |
| two-independent | LDpred3-PIP | 0.93 | **0.90** | 2 |
| two-independent | ABF | 1.00 | 1.00 | 380 |
| two-independent | marginal-top | 0.60 | 0.30 | 1 |
| two-linked | LDpred3-PIP | 0.93 | 0.82 | 2 |
| two-linked | marginal-top | 0.60 | 0.30 | 1 |
| causal+background | LDpred3-PIP | 0.92 | 0.93 | 2 |
| major+sparse | LDpred3-PIP | 0.93 | 0.45 | 3 |

- **LDpred3-PIP is the only method that both localizes and finds multiple
  signals.** It keeps ~0.92–0.97 coverage with **median set size 2–3** across all
  architectures, and on the two-signal architectures it recovers ~1.8 sets/locus
  (power 0.82–0.90) — allelic heterogeneity the single-signal methods cannot see.
- **ABF covers (1.00) but never localizes** (median size 380 = the whole locus);
  its single-signal model also reports one set even when there are two.
- **marginal-top localizes (size 1) but misses signals**: power collapses to
  ~0.30 on two-signal architectures (it can only flag the lead variant) and its
  coverage is ~0.6–0.75 (the top SNP is often a proxy, not the causal).
- **`major+sparse` is the hard case** for power (0.45): every method nails the
  major variant, but the two small (z≈4) effects are near the detection floor.

## Genome-wide

Blocks are independent (block-diagonal LD), so genome-wide fine-mapping runs the
per-locus fine-mapper on every LD block and is embarrassingly parallel:

```python
from ldpred3 import compute_ld_blocks, finemap_by_blocks

blocks = compute_ld_blocks(dosage, block_size=500)      # or optimal_ld_blocks
gw = finemap_by_blocks(blocks, beta_std, n_eff,
                       only_significant=5e-8,            # fine-map loci around hits
                       ncores=4)
gw.pip                  # genome-wide PIP vector
gw.credible_sets        # credible sets, variant indices mapped to the genome
```

`only_significant=5e-8` fine-maps only blocks containing a genome-wide-significant
variant — the usual workflow, and much faster. `only_significant=None` fine-maps
every block; with the fixed sparse prior, null blocks correctly contribute no
credible sets.

## File-based pipeline

`run_finemap` takes a GWAS file + target genotypes and reuses the **same**
read / QC / harmonise / external-LD machinery as the PRS pipeline (identical
allele orientation and `2 - dosage` recoding, shared via
`_external_ld_dosage`), then fine-maps and writes two tables:

```python
from ldpred3 import run_finemap

res = run_finemap("gwas.txt.gz", "target",        # PLINK/BGEN prefix
                  regions="loci.bed",             # optional; else whole genome
                  only_significant=5e-8,           # optional locus filter
                  out="fm")                        # writes fm.pip.tsv, fm.cs.tsv
```

CLI (a flag on the main entry point, not a subcommand):

```bash
ldpred3 --finemap --sumstats gwas.txt.gz --plink target --out fm
ldpred3 --finemap --sumstats gwas.txt.gz --plink target --regions loci.bed \
        --finemap-only-significant 5e-8 --out fm
```

Outputs:

```text
fm.pip.tsv   variant_id chrom pos pip posterior_mean posterior_sd z beta_std n_eff
fm.cs.tsv    cs_id signal coverage n_variants lead_variant lead_pip
             purity_min_abs_r purity_mean_abs_r variants
```

`regions` is a BED-like file (`chrom start end [name]`) or a list of
`(chrom, start, end)` tuples; omit it to fine-map every LD block genome-wide.

## Baseline

`single_signal_finemap(R, beta_std, n_eff)` is a fast single-causal-variant
approximate Bayes factor (ABF): exact when a locus has one signal, and a useful
oracle/cross-check in tests.
