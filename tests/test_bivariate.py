"""Bivariate LDpred3-auto: rg / h2 recovery and cross-trait borrowing."""

import numpy as np

from ldpred3 import ldpred3_auto_bivariate_blocks, ldpred3_by_blocks


def _blocks(n_blocks=12, k=200, seed=0):
    rng = np.random.default_rng(seed)
    blocks, chols, idxs = [], [], []
    for b in range(n_blocks):
        rho = rng.uniform(0.0, 0.8)
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        R = (rho ** d).astype(np.float64)
        blocks.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        chols.append(np.linalg.cholesky(R + 1e-6 * np.eye(k)))
        idxs.append(np.arange(b * k, (b + 1) * k))
    return blocks, chols, idxs


def _gv(blocks, idxs, a, b):
    return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix])
               for i, ix in enumerate(idxs))


def _sim(blocks, chols, idxs, m, *, p, h2, rg, rng):
    """Shared-causal bivariate effects scaled to (h2[0], h2[1]) with corr rg."""
    causal = rng.random(m) < p
    nc = causal.sum()
    L = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    raw = (L @ rng.standard_normal((2, nc)))
    b1 = np.zeros(m); b2 = np.zeros(m)
    b1[causal] = raw[0]; b2[causal] = raw[1]
    b1 *= np.sqrt(h2[0] / _gv(blocks, idxs, b1, b1))
    b2 *= np.sqrt(h2[1] / _gv(blocks, idxs, b2, b2))
    return b1, b2


def _sumstats(blocks, chols, idxs, beta, n, k, rng):
    bhat = np.empty(beta.shape[0])
    for i, ix in enumerate(idxs):
        bhat[ix] = blocks[i][0].astype(float) @ beta[ix] + \
            (chols[i] @ rng.standard_normal(k)) / np.sqrt(n)
    return bhat


def _genetic_r2(b_est, beta, blocks, idxs):
    num = _gv(blocks, idxs, b_est, beta)
    den = _gv(blocks, idxs, b_est, b_est) * _gv(blocks, idxs, beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def test_recovers_rg_and_h2():
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=1)
    m = nb * k
    rgs, h1s, h2s = [], [], []
    for rep in range(3):
        rng = np.random.default_rng(10 + rep)
        b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.7, rng=rng)
        bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
        bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                            burn_in=120, num_iter=150, seed=rep)
        rgs.append(res.rg); h1s.append(res.h2[0]); h2s.append(res.h2[1])
    assert abs(np.mean(rgs) - 0.7) < 0.2, np.mean(rgs)
    assert abs(np.mean(h1s) - 0.5) < 0.12
    assert abs(np.mean(h2s) - 0.5) < 0.12


def test_rg_zero_is_recovered():
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=4)
    m = nb * k
    rng = np.random.default_rng(0)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.0, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                        burn_in=120, num_iter=150, seed=1)
    assert abs(res.rg) < 0.25, res.rg


def test_h2_cap_skips_prepass_and_validations():
    import pytest
    k, nb = 200, 8
    blocks, chols, idxs = _blocks(nb, k, seed=9)
    m = nb * k
    rng = np.random.default_rng(0)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)

    # h2_cap path (skips the univariate pre-pass) still recovers rg
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                        burn_in=80, num_iter=120,
                                        h2_cap=(0.5, 0.5), seed=1)
    assert abs(res.rg - 0.6) < 0.25

    with pytest.raises(ValueError, match="cross_corr"):
        ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                      cross_corr=1.0, h2_cap=(0.5, 0.5))

    overlap = [(blocks[0][0], np.arange(0, k)),
               (blocks[1][0], np.arange(k // 2, k // 2 + k))] + \
        [(blocks[i][0], np.arange(i * k, (i + 1) * k)) for i in range(2, nb)]
    with pytest.raises(ValueError, match="partition"):
        ldpred3_auto_bivariate_blocks(overlap, bh1, bh2, 40000, 40000,
                                      h2_cap=(0.5, 0.5))


def test_borrows_strength_for_low_power_trait():
    """With high rg, a low-N trait should predict better jointly than alone."""
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    N1, N2 = 100000, 3000       # trait 1 well powered, trait 2 weak
    bi, uni = [], []
    for rep in range(4):
        rng = np.random.default_rng(20 + rep)
        b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.9, rng=rng)
        bh1 = _sumstats(blocks, chols, idxs, b1, N1, k, rng)
        bh2 = _sumstats(blocks, chols, idxs, b2, N2, k, rng)
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, N1, N2,
                                            burn_in=120, num_iter=150, seed=rep)
        bi.append(_genetic_r2(res.beta2_est, b2, blocks, idxs))
        solo = ldpred3_by_blocks(blocks, bh2, np.full(m, float(N2)),
                                 method="auto", burn_in=120, num_iter=150, seed=rep)
        uni.append(_genetic_r2(solo, b2, blocks, idxs))
    assert np.mean(bi) > np.mean(uni) + 0.02, (np.mean(bi), np.mean(uni))


def test_bivariate_rejects_compact_blocks():
    # Compact (sparse / low-rank) LD blocks must fail loudly, not crash with a
    # cryptic float() TypeError inside np.ascontiguousarray.
    import pytest
    from ldpred3 import sparsify_ld, lowrank_ld
    rng = np.random.default_rng(0)
    R = (0.3 ** np.abs(np.subtract.outer(np.arange(40), np.arange(40)))).astype(float)
    b1 = rng.standard_normal(40) * 0.02
    b2 = rng.standard_normal(40) * 0.02
    for conv in (sparsify_ld, lowrank_ld):
        blocks = [(conv(R), np.arange(40))]
        with pytest.raises(NotImplementedError, match="dense LD"):
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 10000, 10000,
                                          burn_in=5, num_iter=5, h2_cap=(0.1, 0.1))
