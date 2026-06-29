# LDpred3

A dependency-light (NumPy-only, optional Numba) Python implementation of
[LDpred2](https://doi.org/10.1093/bioinformatics/btaa1029) with a complete
polygenic-score pipeline: from GWAS summary statistics + genotypes to one score
per individual, no R required. It matches the reference implementation
(`bigsnpr`) on accuracy using **~2× less memory**, scales to 2M SNPs on a single
core, and needs **no validation cohort**.

> **New here?** Start with the **[user guide](docs/guide.md)** — it walks from a
> GWAS + target dataset to a polygenic score, helps you choose a model, and lists
> the common pitfalls.

### Install

```bash
pip install .            # installs ldpred3 and the `ldpred3` CLI (needs numpy)
pip install numba        # optional, strongly recommended — large sampler speed-up
pip install msprime      # optional, only for realistic-LD simulation
```

### Quick start

One command turns summary statistics + a genotype target into a polygenic score
(QC and allele harmonisation run automatically):

```bash
ldpred3 --sumstats gwas.txt.gz --plink target --out prs.txt
ldpred3 --sumstats gwas.txt.gz --bgen  target.bgen --out prs.txt
```

```python
from ldpred3 import run_ldpred3_prs
res = run_ldpred3_prs("gwas.txt.gz", "target")   # method="auto" by default
res.scores          # one PRS per individual
res.harmonize_log   # matched / flipped / ambiguous / mismatched counts
res.qc_log          # per-filter QC counts
```

Handy flags (full [CLI reference](docs/pipeline.md#cli-reference) and
[output formats](docs/pipeline.md#outputs) in the pipeline docs):

| flag | what it does |
|------|--------------|
| `--dry-run` | preflight inputs (column mapping, ID match, harmonisation) — no fitting |
| `--infer` | also estimate h², polygenicity and predictive r² |
| `--method annot --annotations a.tsv` | use functional-annotation priors |
| `--dentist` | drop LD-inconsistent variants (allele/strand errors, LD mismatch); off by default |
| `--save-weights w.txt` / `--weights w.txt` | save fitted weights / score a new cohort from them |
| `--ld-out f.npz` / `--ld-cache f.npz` | cache the LD to skip recomputing it on re-runs |

### Choosing a model

| Function | Model | When to use |
|----------|-------|-------------|
| `ldpred3_auto` | point-normal, self-tuning `h²` & `p` | **the default** — robust, no tuning |
| `ldpred3_inf` | infinitesimal (all variants causal) | a truly infinitesimal trait, or a cheap baseline |
| `ldpred3_grid` | point-normal at fixed `h²`, `p` | you already know the hyper-parameters |
| `ldpred3_auto_annot` | `auto` + a learned annotation prior | you have per-SNP functional annotations |

`auto` is the right choice for most traits; the
[guide's decision tree](docs/guide.md#4-choosing-a-model) covers the rest.

### What else it does

- **Heritability, polygenicity & out-of-sample r² — no validation set**
  (`ldpred3_auto_infer`, Privé et al. 2023), with an
  **[LD Score regression](docs/inference.md)** cross-check (`ldsc_h2`).
- **Genetic correlation & joint two-trait PRS** — `ldpred3_auto_bivariate` boosts
  a weak trait using a correlated well-powered one; `ldsc_rg` cross-checks the rg.
- **Annotation-informed priors** (SBayesRC-style, supplied or learned), **sparse
  / banded LD**, **optimal LD-block splitting**, and weight save/reuse + LD
  caching for fast re-runs.
- Internals (streaming genome-wide sampler, float32 LD, Numba JIT) are in
  [docs/algorithm.md](docs/algorithm.md); the full bigsnpr comparison, scaling and
  robustness studies are in [docs/benchmarks.md](docs/benchmarks.md).

### Working from your own LD blocks

If you already have summary statistics and an LD matrix, call a model directly.
Effects are on the **standardized scale**, where the marginal effects relate to
the true joint effects through the LD matrix `R`:

```
beta_hat = R @ beta + noise,   noise ~ N(0, R / N)
```

`standardize_betas` converts reported GWAS effects to this scale and gives the
back-transform:

```python
import numpy as np
from ldpred3 import standardize_betas, ldpred3_auto

beta_hat, scale = standardize_betas(beta, beta_se, n_eff)   # one LD block
res = ldpred3_auto(corr, beta_hat, n_eff)                   # corr: (m, m) LD
adjusted_beta = res.beta_est * scale                        # back to input scale
print(res.h2_est, res.p_est)
```

Use `ldpred3_by_blocks(...)` to run genome-wide, one LD block at a time.

### Tests

```bash
python -m pytest tests/        # full suite
```

### Documentation

- **[docs/guide.md](docs/guide.md)** — start here: choose a model, run the pipeline, read the output, troubleshoot
- [docs/pipeline.md](docs/pipeline.md) — pipeline, QC, file formats, CLI flags
- [docs/inference.md](docs/inference.md) — h² / polygenicity / r² / genetic-correlation inference
- [docs/algorithm.md](docs/algorithm.md) — sampler internals, sparse LD, LD splitting, bivariate model
- [docs/benchmarks.md](docs/benchmarks.md) — accuracy, speed, scaling and robustness benchmarks
- [CHANGELOG.md](CHANGELOG.md) — release history

### License

[MIT](LICENSE) © Bjarni Vilhjálmsson
