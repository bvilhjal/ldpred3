"""Tests for sumstats QC."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sumstats import Sumstats                              # noqa: E402
from qc import qc_sumstats, sd_consistency_mask            # noqa: E402


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
