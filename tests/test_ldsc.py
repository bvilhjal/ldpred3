"""LD Score regression: LD scores and heritability recovery."""

import numpy as np

from pyldpred2 import ld_scores, ldsc_h2, ldsc_rg


def _ar1(k, rho):
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    return (rho ** d).astype(np.float64)


def _varied_blocks(n_blocks, k, seed=0):
    """Block-diagonal AR(1) with rho varying per block, so LD scores span a real
    range (LDSC needs LD-score variation to identify the slope/intercept)."""
    rng = np.random.default_rng(seed)
    blocks, chols = [], []
    for b in range(n_blocks):
        rho = rng.uniform(0.0, 0.9)
        R = _ar1(k, rho)
        blocks.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        chols.append(np.linalg.cholesky(R))
    return blocks, chols


def _simulate(blocks, chols, k, h2, n, rng):
    m = sum(len(idx) for _, idx in blocks)
    beta = rng.normal(0, np.sqrt(h2 / m), m)            # infinitesimal
    bhat = np.empty(m)
    for (R, idx), chol in zip(blocks, chols):
        bhat[idx] = R.astype(float) @ beta[idx] + (chol @ rng.standard_normal(k)) / np.sqrt(n)
    return bhat


def test_ld_scores_basic():
    # Identity LD -> every LD score is exactly 1 (only the self term).
    R = np.eye(5, dtype=np.float32)
    assert np.allclose(ld_scores(R), 1.0)
    # AR(1): LD score exceeds 1 and grows with rho.
    assert ld_scores(_ar1(50, 0.5)).mean() > 1.0
    assert ld_scores(_ar1(50, 0.8)).mean() > ld_scores(_ar1(50, 0.5)).mean()


def test_ldsc_recovers_h2():
    k, nb, n = 200, 60, 40000
    blocks, chols = _varied_blocks(nb, k, seed=1)
    ell = ld_scores(blocks)
    for h2_true in (0.2, 0.5):
        ests = []
        for rep in range(6):
            rng = np.random.default_rng(100 + rep)
            bhat = _simulate(blocks, chols, k, h2_true, n, rng)
            res = ldsc_h2(n * bhat ** 2, ell, n, n_blocks=50)
            ests.append(res.h2)
        mean_h2 = np.mean(ests)
        # LDSC is noisy but should be roughly unbiased for h2.
        assert abs(mean_h2 - h2_true) < 0.08, (h2_true, mean_h2)


def test_ldsc_intercept_near_one_without_confounding():
    k, nb, n = 200, 60, 40000
    blocks, chols = _varied_blocks(nb, k, seed=2)
    ell = ld_scores(blocks)
    ints = []
    for rep in range(6):
        rng = np.random.default_rng(rep)
        bhat = _simulate(blocks, chols, k, 0.4, n, rng)
        ints.append(ldsc_h2(n * bhat ** 2, ell, n, n_blocks=50).intercept)
    # No stratification was simulated, so the intercept should average ~1.
    assert 0.9 < np.mean(ints) < 1.1, np.mean(ints)


def test_ldsc_rg_recovers_genetic_correlation():
    k, nb, n1, n2 = 200, 60, 40000, 20000
    blocks, chols = _varied_blocks(nb, k, seed=5)
    m = nb * k
    idxs = [np.arange(b * k, (b + 1) * k) for b in range(nb)]
    ell = ld_scores(blocks)

    def gv(a, b):
        return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))

    def sumstats(beta, n, rng):
        bh = np.empty(m)
        for i, ix in enumerate(idxs):
            bh[ix] = blocks[i][0].astype(float) @ beta[ix] + \
                (chols[i] @ rng.standard_normal(k)) / np.sqrt(n)
        return bh

    for rg_true in (0.0, 0.6):
        ests = []
        for rep in range(5):
            rng = np.random.default_rng(80 + rep)
            c = rng.random(m) < 0.05
            L = np.linalg.cholesky([[1, rg_true], [rg_true, 1]])
            raw = L @ rng.standard_normal((2, c.sum()))
            b1 = np.zeros(m); b2 = np.zeros(m); b1[c] = raw[0]; b2[c] = raw[1]
            b1 *= np.sqrt(0.5 / gv(b1, b1)); b2 *= np.sqrt(0.5 / gv(b2, b2))
            res = ldsc_rg(sumstats(b1, n1, rng), sumstats(b2, n2, rng), ell, n1, n2,
                          n_blocks=60)
            ests.append(res.rg)
        assert abs(np.mean(ests) - rg_true) < 0.15, (rg_true, np.mean(ests))


def test_ldsc_constrained_intercept():
    k, nb, n = 200, 60, 40000
    blocks, chols = _varied_blocks(nb, k, seed=3)
    ell = ld_scores(blocks)
    rng = np.random.default_rng(3)
    bhat = _simulate(blocks, chols, k, 0.4, n, rng)
    res = ldsc_h2(n * bhat ** 2, ell, n, constrain_intercept=1.0, n_blocks=50)
    assert res.intercept == 1.0
    assert res.intercept_se == 0.0
    assert abs(res.h2 - 0.4) < 0.1
