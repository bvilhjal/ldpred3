"""Golden (characterization) tests: pin the *exact* numerical output of the
estimators so a refactor that silently changes the math fails immediately.

The rest of the suite checks the samplers *statistically* — `h2` within ~0.1 of
the truth on a random simulation — which is deliberately loose and would let a
3-5% drift (a dropped term, a wrong scale, an off-by-one in the residual) pass
unnoticed. These tests instead compare against frozen arrays captured from
known-good code at a tight tolerance: they answer "does this still produce the
same numbers?", catching silent drift across the frequent sampler refactors.

The inputs are fully fixed (a small AR(1) LD block, a seeded `beta_hat`, seeded
chains) so the output is deterministic. The frozen values were verified to be
**bit-identical between the Numba and pure-Python paths**, so the same goldens
hold on both CI legs.

`ldpred3_inf` (linear solve) and `ldpred3_grid` (the dense Gibbs kernel sweep)
are deterministic; the streaming `auto` (effects + h2/p) and `ldpred3_auto_infer`
use NumPy's `default_rng` and are likewise reproducible. Dense `ldpred3_auto` is
*not* golden-tested directly: Numba's `np.random.beta` diverges from NumPy's, so
its per-sweep `p` draw is path-dependent — but its inner sweep is covered by the
`grid` golden (same `_gibbs_kernel`) and its hyper-parameter/Rao-Blackwell logic
by the streaming `auto` golden.
"""

import numpy as np

from ldpred3 import (ldpred3_inf, ldpred3_grid, ldpred3_by_blocks,   # noqa: E402
                     ldpred3_auto_infer)


def _fixtures():
    """Deterministic (R, beta_hat, blocks) shared by every golden test."""
    rng = np.random.default_rng(0)
    m = 20
    R = (0.4 ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))).astype(
        np.float64)
    beta_hat = rng.standard_normal(m) * 0.05
    blocks = [(R[:10, :10].astype(np.float32), np.arange(10)),
              (R[10:, 10:].astype(np.float32), np.arange(10, 20))]
    return R, beta_hat, blocks


# --- frozen outputs (captured from known-good code; see module docstring) ---
_INF = np.array([
    1.04607464929367226e-02, -2.69411086784802252e-02, 4.43881177290817441e-02,
    4.69549157521366618e-03, -4.76193144309874222e-02, 6.64710753698245001e-03,
    5.85223959241127356e-02, 5.06917104484134393e-02, -4.06525225109106991e-02,
    -5.54410472409121793e-02, -1.37121377060054218e-02, 7.18558429481322147e-02,
    -1.54444718706724743e-01, 6.85947941889747115e-02, -6.26144322746027548e-02,
    -8.10235130100025493e-03, -1.25825573411214222e-02, -1.85160833395538113e-02,
    1.11347161022490836e-02, 5.18772641585715164e-02,
])

_GRID = np.array([
    1.03498579002104326e-05, -6.22939706006986576e-04, 2.65616075261915149e-02,
    6.83396960534783887e-05, -3.91216637022260352e-02, 2.43836160974776750e-05,
    6.47656776540538881e-02, 3.13467404467837121e-02, -1.56226504154090354e-02,
    -6.79367579361051382e-02, -9.72293636294800157e-05, 6.79805506132258597e-02,
    -1.55997176703725399e-01, 6.98967101251834338e-02, -6.96852135616874979e-02,
    -1.03585024895822374e-04, -4.77756472764888460e-04, -4.59140960543514382e-04,
    6.54594820222845201e-06, 5.28009027756990434e-02,
])

_BLOCKS_AUTO = np.array([
    6.83416660895169465e-03, -2.23692154867563726e-02, 4.17259888785533259e-02,
    2.65807635593471366e-03, -4.29348948557226329e-02, 4.10017058479290528e-03,
    5.74434511261311537e-02, 4.78241146143608895e-02, -3.78112981317986832e-02,
    -5.66783904046004594e-02, -3.49264104472260598e-02, 6.50548314053285837e-02,
    -1.43823065701350089e-01, 6.10791941505039321e-02, -5.93604578802255853e-02,
    -6.95180179770651680e-03, -1.13496514939263795e-02, -1.47900033080116886e-02,
    7.65476238746960486e-03, 5.08026838208091996e-02,
])

# ldpred3_auto_infer(blocks, ..., n_chains=4, burn_in=60, num_iter=80, seed=7)
_INFER_H2 = 0.03753154333256639
_INFER_P = 0.8937249315174276
_INFER_R2 = 0.03642684482343874


def test_golden_inf():
    R, beta_hat, _ = _fixtures()
    got = ldpred3_inf(R, beta_hat, 10000, h2=0.3)
    np.testing.assert_allclose(got, _INF, rtol=1e-6, atol=1e-9)


def test_golden_grid():
    # Exercises the dense _gibbs_kernel inner sweep (estimate_hyper off).
    R, beta_hat, _ = _fixtures()
    got = ldpred3_grid(R, beta_hat, 10000, h2=0.3, p=0.1,
                       burn_in=50, num_iter=150, seed=42)
    np.testing.assert_allclose(got, _GRID, rtol=1e-6, atol=1e-9)


def test_golden_auto_blocks():
    # Streaming global-hyper auto: effects + the h2/p estimation path.
    _, beta_hat, blocks = _fixtures()
    got = ldpred3_by_blocks(blocks, beta_hat, 10000, method="auto",
                            burn_in=50, num_iter=150, seed=42)
    np.testing.assert_allclose(got, _BLOCKS_AUTO, rtol=1e-6, atol=1e-9)


def test_golden_infer_blocks():
    # Streaming inference: the h2 / p / predictive-r2 estimands.
    _, beta_hat, blocks = _fixtures()
    res = ldpred3_auto_infer(blocks, beta_hat, 10000, n_chains=4, burn_in=60,
                             num_iter=80, seed=7)
    np.testing.assert_allclose(res.h2_est, _INFER_H2, rtol=1e-6)
    np.testing.assert_allclose(res.p_est, _INFER_P, rtol=1e-6)
    np.testing.assert_allclose(res.r2_est, _INFER_R2, rtol=1e-6)
