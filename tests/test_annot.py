"""Tests for learning the annotation->prior map inside the sampler (SBayesRC)."""

import numpy as np
import pytest

from ldpred3 import (ldpred3_auto_annot, ldpred3_auto_annot_blocks,
                       ldpred3_auto)
from ldpred3.annot import _Phi, _Phi_inv, _truncnorm
from ldpred3.prs import standardize_dosage


# --------------------------------------------------------------------------- #
# Simulation helper
# --------------------------------------------------------------------------- #
def _geno(n, m, rho, rng):
    def hap():
        z = np.zeros((n, m)); z[:, 0] = rng.standard_normal(n)
        s = np.sqrt(1 - rho ** 2)
        for j in range(1, m):
            z[:, j] = rho * z[:, j - 1] + s * rng.standard_normal(n)
        return z
    thr = rng.uniform(-1, 1, m)
    return (hap() > thr).astype(float) + (hap() > thr).astype(float)


def _data(seed, N=2500, m=400, h2=0.5, p=0.05, enrich=12.0, extra_annot=None):
    rng = np.random.default_rng(seed)
    GA, GB = _geno(N, m, 0.6, rng), _geno(3000, m, 0.6, rng)
    ZA, ZB = standardize_dosage(GA), standardize_dosage(GB)
    R = (ZA.T @ ZA) / N; np.fill_diagonal(R, 1.0)
    func = (rng.random(m) < 0.2).astype(float)
    noise = (rng.random(m) < 0.3).astype(float)            # irrelevant
    base = np.where(func > 0, enrich, 1.0)
    causal = rng.random(m) < np.clip(base / base.sum() * (p * m), 0, 1)
    if not causal.any():
        causal[rng.integers(m)] = True
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
    gA = ZA @ beta
    y = gA + rng.normal(0, np.sqrt(max(1e-6, 1 - gA.var())), N)
    bhat = (ZA.T @ y) / N
    yte = ZB @ beta + rng.normal(0, np.sqrt(1 - h2), 3000)
    A = np.column_stack([func, noise])
    return R, bhat, N, A, ZB, yte


# --------------------------------------------------------------------------- #
# Learning behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("learn", ["eb", "probit"])
def test_learns_enrichment_and_ignores_noise(learn):
    tf = tn = 0.0
    for seed in range(4):
        R, bhat, N, A, _, _ = _data(seed)
        res = ldpred3_auto_annot(R, bhat, N, A, learn=learn, burn_in=80,
                                 num_iter=200, seed=1,
                                 annotation_names=["func", "noise"])
        tf += res.enrichment["func"]; tn += res.enrichment["noise"]
    tf /= 4; tn /= 4
    assert tf > 0.4, (learn, tf)
    assert tf > tn + 0.3, (learn, tf, tn)


def test_annot_predicts_at_least_as_well_as_uniform():
    r_uni = r_ann = 0.0
    for seed in range(4):
        R, bhat, N, A, ZB, yte = _data(seed)
        b_uni = ldpred3_auto(R, bhat, N, burn_in=80, num_iter=200, seed=1).beta_est
        b_ann = ldpred3_auto_annot(R, bhat, N, A, learn="eb", burn_in=80,
                                   num_iter=200, seed=1).beta_est
        r_uni += np.corrcoef(ZB @ b_uni, yte)[0, 1] ** 2
        r_ann += np.corrcoef(ZB @ b_ann, yte)[0, 1] ** 2
    assert r_ann / 4 >= r_uni / 4 - 0.01


def test_continuous_annotation_recovered():
    # A continuous annotation whose value drives causal probability.
    rng = np.random.default_rng(3)
    m = 400
    R = 0.5 ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    a = rng.normal(0, 1, m)                       # continuous annotation
    pr = np.clip(0.03 * np.exp(0.9 * a), 0, 1)    # higher a -> more causal
    causal = rng.random(m) < pr
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, 0.3, causal.sum())
    bhat = R @ beta + rng.standard_normal(m) / np.sqrt(8000)
    res = ldpred3_auto_annot(R, bhat, 8000, a, learn="eb", burn_in=80,
                             num_iter=200, seed=1, annotation_names=["score"])
    assert res.enrichment["score"] > 0.2


def test_learn_variance_recovers_effect_size_map():
    # Functional SNPs have larger effects -> variance coefficient phi_func > 0.
    pv = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        N, m, h2, p = 3000, 400, 0.5, 0.08
        G = _geno(N, m, 0.6, rng); Z = standardize_dosage(G)
        R = (Z.T @ Z) / N; np.fill_diagonal(R, 1.0)
        func = (rng.random(m) < 0.2).astype(float)
        causal = rng.random(m) < p
        sd = np.where(func > 0, 3.0, 1.0)
        beta = np.zeros(m)
        beta[causal] = rng.normal(0, 1, causal.sum()) * sd[causal]
        beta *= np.sqrt(h2 / (beta @ (R @ beta)))
        y = Z @ beta + rng.normal(0, np.sqrt(0.5), N)
        bhat = (Z.T @ y) / N
        res = ldpred3_auto_annot(R, bhat, N, func[:, None], learn="eb",
                                 learn_variance=True, burn_in=80, num_iter=200,
                                 seed=1, annotation_names=["func"])
        assert res.phi is not None
        pv += res.variance_enrichment["func"]
    assert pv / 4 > 0.25, pv / 4


def test_variance_off_gives_no_phi():
    R, bhat, N, A, _, _ = _data(0)
    res = ldpred3_auto_annot(R, bhat, N, A, learn_variance=False, burn_in=40,
                             num_iter=80, seed=1)
    assert res.phi is None and res.variance_enrichment is None


def test_intercept_only_runs_like_uniform():
    # No informative annotation (a constant column) -> still produces sane betas.
    R, bhat, N, A, ZB, yte = _data(0)
    const = np.zeros((bhat.shape[0], 1))          # uninformative constant
    res = ldpred3_auto_annot(R, bhat, N, const, learn="eb", burn_in=60,
                             num_iter=150, seed=1)
    assert np.all(np.isfinite(res.beta_est))
    assert res.theta.shape == (2,)               # intercept + the constant col


# --------------------------------------------------------------------------- #
# API ergonomics / determinism
# --------------------------------------------------------------------------- #
def test_deterministic_with_seed():
    R, bhat, N, A, _, _ = _data(1)
    a = ldpred3_auto_annot(R, bhat, N, A, learn="probit", burn_in=40,
                           num_iter=80, seed=7)
    b = ldpred3_auto_annot(R, bhat, N, A, learn="probit", burn_in=40,
                           num_iter=80, seed=7)
    np.testing.assert_array_equal(a.beta_est, b.beta_est)
    np.testing.assert_array_equal(a.theta, b.theta)


def test_1d_annotation_and_names_and_repr():
    R, bhat, N, A, _, _ = _data(0)
    a1 = A[:, 0]                                   # 1-D annotation
    res = ldpred3_auto_annot(R, bhat, N, a1, learn="eb", burn_in=40,
                             num_iter=80, seed=1, annotation_names=["coding"])
    assert res.annotation_names == ["intercept", "coding"]
    assert set(res.enrichment) == {"coding"}
    assert "coding=" in repr(res) and "h2_est=" in repr(res)


def test_supplied_intercept_not_duplicated():
    R, bhat, N, A, _, _ = _data(0)
    A_ic = np.column_stack([np.ones(bhat.shape[0]), A])   # already has intercept
    res = ldpred3_auto_annot(R, bhat, N, A_ic, learn="eb", burn_in=40,
                             num_iter=80, seed=1)
    assert res.theta.shape == (3,)               # intercept + 2 annotations


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validation_errors():
    R, bhat, N, A, _, _ = _data(0)
    m = bhat.shape[0]
    with pytest.raises(ValueError, match="learn"):
        ldpred3_auto_annot(R, bhat, N, A, learn="nope")
    with pytest.raises(ValueError, match="annotation_names"):
        ldpred3_auto_annot(R, bhat, N, A, annotation_names=["only_one"])
    with pytest.raises(ValueError, match="one row per variant"):
        ldpred3_auto_annot(R, bhat, N, A[:m - 1])
    with pytest.raises(ValueError, match="finite"):
        bad = A.copy(); bad[0, 0] = np.nan
        ldpred3_auto_annot(R, bhat, N, bad)
    with pytest.raises(ValueError, match="theta_every"):
        ldpred3_auto_annot(R, bhat, N, A, theta_every=0)
    with pytest.raises(ValueError, match="p_init"):
        ldpred3_auto_annot(R, bhat, N, A, p_init=1.5)


def test_streaming_matches_dense_on_block_diagonal():
    # Genome-wide streaming version == dense version on block-diagonal LD.
    rng = np.random.default_rng(2)
    m, nblk, N = 600, 3, 3000
    k = m // nblk
    blocks = []
    Rfull = np.zeros((m, m))
    G = np.empty((N, m))
    for b in range(nblk):
        Gb = _geno(N, k, 0.6, rng)
        Zb = standardize_dosage(Gb)
        Rb = (Zb.T @ Zb) / N; np.fill_diagonal(Rb, 1.0)
        idx = np.arange(b * k, (b + 1) * k)
        blocks.append((Rb.astype(np.float32), idx))
        Rfull[np.ix_(idx, idx)] = Rb; G[:, idx] = Gb
    Z = standardize_dosage(G)
    func = (rng.random(m) < 0.2).astype(float)
    causal = rng.random(m) < np.where(func > 0, 0.2, 0.01)
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(0.5 / causal.sum()), causal.sum())
    y = Z @ beta + rng.normal(0, np.sqrt(0.5), N)
    bhat = (Z.T @ y) / N
    A = func[:, None]
    d = ldpred3_auto_annot(Rfull, bhat, N, A, learn="eb", burn_in=60,
                           num_iter=150, seed=1, annotation_names=["func"])
    s = ldpred3_auto_annot_blocks(blocks, bhat, N, A, learn="eb", burn_in=60,
                                  num_iter=150, seed=1, annotation_names=["func"])
    assert np.corrcoef(d.beta_est, s.beta_est)[0, 1] > 0.99
    assert abs(d.enrichment["func"] - s.enrichment["func"]) < 0.3
    assert s.enrichment["func"] > 0.4


def test_streaming_rejects_bad_blocks():
    R = (0.5 ** np.abs(np.subtract.outer(np.arange(50), np.arange(50)))).astype(np.float32)
    blocks = [(R, np.arange(50))]
    with pytest.raises(ValueError, match="tile"):           # only covers 0..49
        ldpred3_auto_annot_blocks(blocks, np.zeros(60), 5000, np.ones((60, 1)))


def test_read_annotations_aligns_by_id(tmp_path):
    from ldpred3.annot import read_annotations
    p = tmp_path / "annot.tsv"
    p.write_text("SNP\tcoding\tconserved\n"
                 "rs1\t1\t0.5\n"
                 "rs3\t0\t0.2\n")               # rs0, rs2 absent -> zeros
    A, names = read_annotations(str(p), ["rs0", "rs1", "rs2", "rs3"])
    assert names == ["coding", "conserved"]
    np.testing.assert_array_equal(A[:, 0], [0, 1, 0, 0])
    np.testing.assert_allclose(A[:, 1], [0, 0.5, 0, 0.2])


def test_compact_ld_rejected():
    # Both compact representations (sparse / low-rank), via both entry points,
    # must raise NotImplementedError rather than crash with a cryptic TypeError.
    from ldpred3 import sparsify_ld, lowrank_ld
    R = 0.5 ** np.abs(np.subtract.outer(np.arange(40), np.arange(40)))
    for conv in (sparsify_ld, lowrank_ld):
        ld = conv(R)
        with pytest.raises(NotImplementedError, match="dense LD"):
            ldpred3_auto_annot(ld, np.zeros(40), 5000, np.ones((40, 1)))
        with pytest.raises(NotImplementedError, match="dense LD"):
            ldpred3_auto_annot_blocks([(ld, np.arange(40))], np.zeros(40), 5000,
                                      np.ones((40, 1)))


# --------------------------------------------------------------------------- #
# Numerical helpers used by the probit update
# --------------------------------------------------------------------------- #
def test_phi_and_phi_inv_accuracy():
    from statistics import NormalDist
    nd = NormalDist()
    xs = np.array([-3.0, -1.5, -0.3, 0.0, 0.7, 1.5, 3.0])
    ref = np.array([nd.cdf(x) for x in xs])
    np.testing.assert_allclose(_Phi(xs), ref, atol=2e-7)
    ps = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    ref_inv = np.array([nd.inv_cdf(p) for p in ps])
    np.testing.assert_allclose(_Phi_inv(ps), ref_inv, atol=1e-6)


def test_truncnorm_respects_sign():
    rng = np.random.default_rng(0)
    mu = np.linspace(-2, 2, 1000)
    gamma = np.tile([1.0, 0.0], 500)
    z = _truncnorm(mu, gamma, rng)
    assert np.all(z[gamma > 0] > 0)
    assert np.all(z[gamma == 0] < 0)
