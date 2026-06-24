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
