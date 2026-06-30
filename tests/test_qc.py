"""Tests for sumstats QC."""


import numpy as np


from ldpred3.sumstats import Sumstats                              # noqa: E402
from ldpred3.qc import (                                           # noqa: E402
    qc_sumstats,
    sd_consistency_mask,
    dentist_outlier_mask,
)


def _ss(n, **over):
    d = dict(
        id=np.array([f"rs{i}" for i in range(n)], dtype=object),
        chrom=np.array(["1"] * n, dtype=object),
        pos=np.arange(1, n + 1, dtype=np.int64),
        ea=np.array(["A"] * n, dtype=object),
        oa=np.array(["G"] * n, dtype=object),
        beta=np.full(n, 0.1),
        se=np.full(n, 0.02),
        n_eff=np.full(n, 1000.0),
        eaf=np.full(n, 0.3),
        info=np.full(n, 0.99),
    )
    d.update(over)
    return Sumstats(**d)


def test_qc_drops_nonfinite_and_bad_se():
    ss = _ss(5)
    ss.beta[0] = np.nan
    ss.se[1] = 0.0
    ss.se[2] = -1.0
    keep, log = qc_sumstats(ss)
    assert log["n_drop_nonfinite"] == 3
    assert list(np.where(keep)[0]) == [3, 4]


def test_qc_low_n_filter():
    ss = _ss(4, n_eff=np.array([1000., 1000., 600., 1000.]))  # max 1000, 0.7->700
    keep, log = qc_sumstats(ss)
    assert log["n_drop_lowN"] == 1
    assert not keep[2]


def test_qc_maf_and_info():
    ss = _ss(4, eaf=np.array([0.3, 0.005, 0.3, 0.3]),
             info=np.array([0.99, 0.99, 0.5, 0.99]))
    keep, log = qc_sumstats(ss, min_maf=0.01, min_info=0.7)
    assert log["n_drop_lowMAF"] == 1 and not keep[1]
    assert log["n_drop_lowINFO"] == 1 and not keep[2]


def test_qc_duplicates_dropped():
    ss = _ss(4)
    ss.id[2] = "rs0"          # duplicate of index 0
    keep, log = qc_sumstats(ss)
    assert log["n_drop_duplicate"] == 2   # both copies dropped
    assert not keep[0] and not keep[2]


def test_qc_chisq_outlier():
    ss = _ss(3, beta=np.array([0.1, 5.0, 0.1]), se=np.full(3, 0.02))
    # z = beta/se -> (5, 250, 5); chisq (25, 62500, 25)
    keep, log = qc_sumstats(ss, max_chisq=1000)
    assert log["n_drop_chisq_outlier"] == 1 and not keep[1]


def test_qc_columns_absent_are_skipped():
    # No eaf/info available -> those filters are no-ops, nothing dropped.
    ss = _ss(3, eaf=np.full(3, np.nan), info=np.full(3, np.nan))
    keep, log = qc_sumstats(ss)
    assert keep.all()
    assert "n_drop_lowMAF" not in log and "n_drop_lowINFO" not in log


def test_sd_consistency_flags_wrong_n():
    rng = np.random.default_rng(0)
    n = 200
    af = rng.uniform(0.1, 0.9, n)
    # Consistent variants: se ~ 1/sqrt(N * 2f(1-f)) so sd_ss ~ sd_ref.
    N = np.full(n, 10000.0)
    se = 1.0 / np.sqrt(N * 2 * af * (1 - af))
    beta = rng.normal(0, 0.01, n)
    # Corrupt 20 variants with a 100x-too-small N (inflated se mismatch).
    bad = rng.choice(n, 20, replace=False)
    se[bad] *= 10
    keep, log, diag = sd_consistency_mask(beta, se, N, af)
    # Most corrupted variants should be flagged, most good ones kept.
    assert log["n_drop_sd_inconsistent"] >= 10
    assert keep.sum() > n - 40
    assert keep[~np.isin(np.arange(n), bad)].mean() > 0.8


def _ld_block(k, seed):
    """A single correlated LD block via a shared latent-factor genotype draw."""
    rng = np.random.default_rng(seed)
    nfac = max(2, k // 5)
    L = rng.standard_normal((k, nfac)) * 0.7
    G = rng.standard_normal((2000, nfac)) @ L.T + rng.standard_normal((2000, k))
    G = (G - G.mean(0)) / G.std(0)
    R = (G.T @ G) / G.shape[0]
    return R


def test_dentist_keeps_ld_consistent_z():
    # z generated as R @ beta (perfectly LD-consistent) -> nothing flagged.
    R = _ld_block(40, seed=1)
    rng = np.random.default_rng(2)
    beta = np.zeros(40)
    beta[rng.choice(40, 3, replace=False)] = rng.standard_normal(3) * 3
    z = R @ beta
    keep, log = dentist_outlier_mask([(R, np.arange(40))], z)
    assert keep.all()
    assert log["n_drop_dentist"] == 0


def test_dentist_catches_single_sign_flip():
    # One LD-inconsistent variant (sign-flipped) is removed; neighbours kept.
    R = _ld_block(40, seed=3)
    rng = np.random.default_rng(4)
    beta = np.zeros(40)
    lead = 10
    beta[lead] = 4.0
    beta[rng.choice(40, 2, replace=False)] = rng.standard_normal(2) * 2
    z = R @ beta
    z[lead] = -z[lead]                       # corrupt the lead variant
    keep, log = dentist_outlier_mask([(R, np.arange(40))], z)
    assert not keep[lead]                    # the error is caught
    # Single-worst-per-pass removal does not nuke the whole correlated block.
    assert keep.sum() >= 38


def test_dentist_skips_small_blocks():
    R = np.eye(2)
    z = np.array([10.0, -10.0])
    keep, log = dentist_outlier_mask([(R, np.arange(2))], z, min_block=3)
    assert keep.all()
