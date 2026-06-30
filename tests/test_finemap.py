"""Tests for LDpred3-PIP fine-mapping (ldpred3/finemap.py)."""
import numpy as np
import pytest

from ldpred3 import (ldpred3_pip, single_signal_finemap, finemap_by_blocks,
                     FineMapResult, CredibleSet)


def ar1(m, rho):
    return (rho ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))).astype(float)


def sumstats(R, causal, eff, n, seed):
    """Standardized marginal effects beta_hat = R@beta + N(0, R/n)."""
    m = R.shape[0]
    rng = np.random.default_rng(seed)
    beta = np.zeros(m)
    for c, e in zip(causal, eff):
        beta[c] = e
    chol = np.linalg.cholesky(R + 1e-6 * np.eye(m))
    return R @ beta + (chol @ rng.standard_normal(m)) / np.sqrt(n)


N = 50000


def test_single_causal_top_pip_and_credible_set():
    R = ar1(60, 0.5)
    bh = sumstats(R, [30], [0.25], N, 1)
    res = ldpred3_pip(R, bh, N, seed=1)
    assert isinstance(res, FineMapResult)
    assert int(np.argmax(res.pip)) == 30          # top PIP is the causal variant
    assert res.pip[30] > 0.9
    assert len(res.credible_sets) == 1
    assert 30 in res.credible_sets[0].variants     # CS contains the causal
    assert res.credible_sets[0].coverage >= 0.95


def test_two_independent_signals_two_credible_sets():
    R = ar1(120, 0.5)                              # causals far apart -> low LD
    bh = sumstats(R, [20, 90], [0.25, 0.25], N, 2)
    res = ldpred3_pip(R, bh, N, seed=2)
    assert len(res.credible_sets) == 2
    leads = sorted(int(cs.variants[np.argmax(cs.pip)]) for cs in res.credible_sets)
    assert leads == [20, 90]
    assert res.n_signals_est == pytest.approx(2.0, abs=0.3)


def test_high_ld_cluster_credible_set_is_pure():
    R = ar1(40, 0.97)                              # one tight LD block
    bh = sumstats(R, [20], [0.18], N, 7)
    res = ldpred3_pip(R, bh, N, seed=7)
    assert len(res.credible_sets) >= 1
    cs = res.credible_sets[0]
    assert 20 in cs.variants                       # causal (or a tight proxy) is in the set
    assert cs.purity_min_abs_r >= 0.5              # purity filter holds


def test_null_locus_no_confident_credible_set():
    R = ar1(60, 0.5)
    bh = sumstats(R, [], [], N, 3)
    res = ldpred3_pip(R, bh, N, seed=3)
    assert len(res.credible_sets) == 0
    assert res.pip.max() < 0.3
    assert res.n_signals_est < 0.5


def test_determinism_fixed_seed():
    R = ar1(50, 0.5)
    bh = sumstats(R, [25], [0.25], N, 11)
    a = ldpred3_pip(R, bh, N, seed=123)
    b = ldpred3_pip(R, bh, N, seed=123)
    np.testing.assert_allclose(a.pip, b.pip)


def test_allele_flip_invariance():
    """Flipping a variant's reference allele flips the sign of beta_hat and the
    corresponding row/column of R; PIPs must be unchanged."""
    R = ar1(60, 0.5)
    bh = sumstats(R, [30], [0.25], N, 5)
    flip = np.ones(60)
    rng = np.random.default_rng(0)
    flip[rng.random(60) < 0.5] = -1.0              # flip a random half of variants
    Rf = R * np.outer(flip, flip)
    bhf = bh * flip
    a = ldpred3_pip(R, bh, N, seed=9)
    b = ldpred3_pip(Rf, bhf, N, seed=9)
    np.testing.assert_allclose(a.pip, b.pip, atol=1e-6)


def test_single_signal_abf_baseline():
    R = ar1(60, 0.5)
    bh = sumstats(R, [30], [0.25], N, 4)
    res = single_signal_finemap(R, bh, N)
    assert int(np.argmax(res.pip)) == 30
    assert res.pip.sum() == pytest.approx(1.0)     # ABF PIPs are a proper distribution


def test_compact_ld_densified():
    """LowRankLD / SparseLD blocks are densified and give the same answer."""
    from ldpred3 import lowrank_ld, sparsify_ld
    R = ar1(60, 0.6)
    bh = sumstats(R, [30], [0.25], N, 6)
    base = ldpred3_pip(R, bh, N, seed=3)
    lr = ldpred3_pip(lowrank_ld(R, variance=0.999), bh, N, seed=3)
    sp = ldpred3_pip(sparsify_ld(R, threshold=1e-4), bh, N, seed=3)
    assert int(np.argmax(lr.pip)) == 30
    assert int(np.argmax(sp.pip)) == 30


def _genome(n_blocks, k, signal_blocks, seed0=100):
    blocks, bh_all, truth, off = [], [], [], 0
    for b in range(n_blocks):
        Rb = ar1(k, 0.5)
        blocks.append((Rb.astype(np.float32), np.arange(off, off + k)))
        causal = [k // 2] if b in signal_blocks else []
        bh_all.append(sumstats(Rb, causal, [0.3] * len(causal), N, seed0 + b))
        truth += [off + c for c in causal]
        off += k
    return blocks, np.concatenate(bh_all), truth


def test_genome_wide_recovers_signal_blocks():
    blocks, bh, truth = _genome(5, 50, signal_blocks=(1, 3))
    gw = finemap_by_blocks(blocks, bh, N, seed=1)
    leads = sorted(int(cs.variants[np.argmax(cs.pip)]) for cs in gw.credible_sets)
    assert leads == sorted(truth)                  # exactly the two true signals
    for t in truth:
        assert gw.pip[t] > 0.9


def test_genome_wide_only_significant_skips_null_blocks():
    blocks, bh, truth = _genome(6, 50, signal_blocks=(2, 4))
    gw = finemap_by_blocks(blocks, bh, N, seed=1, only_significant=5e-8)
    assert gw.diagnostics["n_blocks_finemapped"] == 2   # only the 2 signal blocks
    leads = sorted(int(cs.variants[np.argmax(cs.pip)]) for cs in gw.credible_sets)
    assert leads == sorted(truth)


def test_credible_set_variants_mapped_to_genome():
    blocks, bh, truth = _genome(4, 50, signal_blocks=(2,))
    gw = finemap_by_blocks(blocks, bh, N, seed=1)
    assert len(gw.credible_sets) == 1
    # block 2 spans [100,150); the credible set indices must be global, not local.
    assert gw.credible_sets[0].variants.min() >= 100
    assert truth[0] in gw.credible_sets[0].variants
