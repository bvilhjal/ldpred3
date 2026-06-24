# iprs
iPSYCH PRS

## LDpred2 (Python)

`src/ldpred2.py` is a small, dependency-light (NumPy only) Python implementation
of the core [LDpred2](https://doi.org/10.1093/bioinformatics/btaa1029)
polygenic-score models. LDpred2 re-weights GWAS marginal effect sizes using an
LD (linkage-disequilibrium) correlation matrix.

### Models implemented

| Function          | Model                                              | Hyper-parameters     |
|-------------------|----------------------------------------------------|----------------------|
| `ldpred2_inf`     | Infinitesimal (all variants causal, closed form)   | `h2`                 |
| `ldpred2_grid`    | Point-normal / spike-and-slab (Gibbs sampler)      | `h2`, `p` (fixed)    |
| `ldpred2_auto`    | Point-normal, estimates `h2` and `p` automatically | none (self-tuning)   |

Helpers: `standardize_betas` (put GWAS effects on the correlation scale) and
`ldpred2_by_blocks` (run a model independently per LD block, genome-wide).

### Conventions

Effects are on the standardized scale, where the marginal effects relate to the
true joint effects through the LD matrix `R`:

```
beta_hat = R @ beta + noise,   noise ~ N(0, R / N)
```

with `N` the GWAS sample size. Use `standardize_betas(beta, beta_se, n_eff)` to
convert reported GWAS effects to this scale and to recover the back-transform.

### Quick example

```python
import numpy as np
from ldpred2 import standardize_betas, ldpred2_auto

# beta, beta_se, n_eff come from GWAS summary statistics for one LD block;
# corr is the (m x m) LD correlation matrix for those variants.
beta_hat, scale = standardize_betas(beta, beta_se, n_eff)

res = ldpred2_auto(corr, beta_hat, n_eff)
adjusted_beta = res.beta_est * scale          # back to the input scale
print(res.h2_est, res.p_est)                  # estimated heritability & causal frac
```

### Tests / demo

```bash
python tests/test_ldpred2.py     # prints recovery of true effects on simulated data
python -m pytest tests/          # run the assertions
```

On the bundled synthetic LD block, correlation with the true effects improves
from ~0.67 (raw marginal betas) to ~0.98 (inf) and ~0.99 (grid / auto).

### Genotype-level benchmark

`src/simulate.py` is a full end-to-end simulation: it generates genotypes with
block LD, simulates a phenotype under a chosen heritability and polygenicity,
runs a marginal GWAS, estimates the LD matrix from the training sample, fits
LDpred2, and reports **out-of-sample** prediction R² on a held-out test set. It
sweeps a grid of polygenicity × heritability × sample size.

```bash
python src/simulate.py --quick            # fast sanity check
python src/simulate.py --csv sim.csv      # full grid, save results
```

Representative results (m=1000 SNPs, blocks of 100; prediction R² vs phenotype):

| N | h² | p (causal) | marginal | inf | grid | auto | ceiling |
|---|----|-----------|---------|-----|------|------|---------|
| 5000 | 0.5 | 0.005 | 0.280 | 0.331 | 0.452 | 0.451 | 0.455 |
| 5000 | 0.5 | 0.05  | 0.320 | 0.377 | 0.491 | 0.482 | 0.501 |
| 10000 | 0.5 | 0.5  | 0.368 | 0.460 | 0.475 | 0.471 | 0.537 |
| 10000 | 0.2 | 0.05 | 0.128 | 0.142 | 0.196 | 0.197 | 0.203 |

Takeaways: LDpred2 always beats the raw marginal baseline; accuracy rises with
heritability and sample size; `grid`/`auto` approach the ceiling for sparse
architectures, while `inf` is competitive for highly polygenic traits.
