"""Summary-statistic imputation from LD (SSimp-style), as a pre-processing layer.

A variant missing from the GWAS but present in the LD reference can have its
**standardized marginal effect** imputed from its typed neighbours, because under
LDpred3's model ``beta_hat = R @ beta + N(0, R/N)`` the missing statistic is a
Gaussian conditional mean::

    beta_hat_u   = R_ut R_tt^-1 beta_hat_t          (per LD block; t=typed, u=untyped)
    imp_r2_u     = diag(R_ut R_tt^-1 R_tu)  in [0,1]  (imputation quality)

The imputed statistic carries **no new information** (it is a linear combination
of the typed ones), so it must be **down-weighted** by its imputation quality:
the untyped variant enters the sampler with an effective sample size
``N_u = N * imp_r2_u``. This module only *prepares inputs* -- it returns an
augmented ``(beta_hat, n_eff)`` over the full variant set, which then feed the
**unchanged** sampler / fine-mapper. It is most useful in conjunction with
functional annotations (`ldpred3_auto_annot_blocks`): the annotation-informed
prior can redistribute effect onto an imputed *functional* variant that the
typed tags only smear over -- which matters for localisation and cross-ancestry
portability (see ``benchmarks/impute_annot.py``). The core sampler is untouched.

NumPy-only, and per LD block (``R_tt^-1`` is small), so it streams on-the-fly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ld_utils import SparseLD, LowRankLD


@dataclass
class ImputeResult:
    """Augmented summary statistics over the full (typed + untyped) variant set."""

    beta_hat: np.ndarray = field(repr=False)        # (m,) typed observed, untyped imputed
    n_eff: np.ndarray = field(repr=False)           # (m,) typed N, untyped N*imp_r2
    imp_r2: np.ndarray = field(repr=False)          # (m,) 1.0 at typed, [0,1] at untyped
    typed_mask: np.ndarray = field(repr=False)
    n_typed: int = 0
    n_imputed: int = 0

    def __repr__(self):
        impu = self.imp_r2[~self.typed_mask]
        mean_r2 = float(np.mean(impu)) if impu.size else float("nan")
        return (f"ImputeResult(n_typed={self.n_typed}, n_imputed={self.n_imputed}, "
                f"mean_imp_r2={mean_r2:.3f})")


def _dense(R):
    if isinstance(R, LowRankLD):
        D = np.asarray(R.U, float) @ np.asarray(R.U, float).T
        np.fill_diagonal(D, 1.0)
        return D
    if isinstance(R, SparseLD):
        m = int(R.m)
        D = np.zeros((m, m))
        for i in range(m):
            for q in range(R.indptr[i], R.indptr[i + 1]):
                D[i, R.indices[q]] = R.data[q]
        np.fill_diagonal(D, 1.0)
        return D
    return np.ascontiguousarray(R, dtype=np.float64)


def impute_sumstats_blocks(beta_hat, blocks, typed_mask, n_eff, *, ridge=1e-3,
                           min_imp_r2=0.0) -> ImputeResult:
    """Impute untyped variants' marginal effects from typed neighbours, per block.

    Parameters
    ----------
    beta_hat : array_like (m,)
        Standardized marginal effects; entries where ``typed_mask`` is False are
        ignored and overwritten by the imputed value.
    blocks : list of (R, idx)
        Per-block LD (dense / SparseLD / LowRankLD) partitioning ``0..m-1``; ``R``
        must span the **full** variant set (typed and untyped together).
    typed_mask : array_like of bool (m,)
        True where ``beta_hat`` is observed (typed in the GWAS).
    n_eff : float or array_like
        GWAS (effective) sample size of the typed variants.
    ridge : float, default 1e-3
        Tikhonov ridge added to ``R_tt`` before the solve (numerical stability).
    min_imp_r2 : float, default 0.0
        Untyped variants imputed below this quality keep ``beta_hat=0`` and
        ``n_eff=0`` (effectively excluded) — they cannot be imputed reliably.

    Returns
    -------
    ImputeResult with the augmented ``beta_hat`` / ``n_eff`` over all ``m``
    variants (down-weighted ``N*imp_r2`` at imputed variants), ready to feed the
    unchanged sampler or fine-mapper.
    """
    beta_hat = np.array(beta_hat, dtype=np.float64)        # copy (we overwrite untyped)
    typed_mask = np.asarray(typed_mask, dtype=bool)
    m = beta_hat.shape[0]
    n_eff = np.broadcast_to(np.asarray(n_eff, dtype=np.float64), (m,)).copy()
    imp_r2 = np.where(typed_mask, 1.0, 0.0)
    out_n = np.where(typed_mask, n_eff, 0.0)

    for R, idx in blocks:
        idx = np.asarray(idx)
        tb = typed_mask[idx]
        if tb.all() or not tb.any():        # nothing to impute / nothing to impute from
            continue
        D = _dense(R)
        t = np.flatnonzero(tb)
        u = np.flatnonzero(~tb)
        Rtt = D[np.ix_(t, t)] + ridge * np.eye(t.size)
        Rut = D[np.ix_(u, t)]                # (|u|, |t|)
        bt = beta_hat[idx[t]]
        # beta_hat_u = R_ut R_tt^-1 beta_hat_t
        x = np.linalg.solve(Rtt, bt)
        bhat_u = Rut @ x
        # imp_r2_u = diag(R_ut R_tt^-1 R_tu)
        M = np.linalg.solve(Rtt, Rut.T)      # (|t|, |u|)
        r2_u = np.clip(np.einsum("ij,ji->i", Rut, M), 0.0, 1.0)
        n_t = float(np.mean(n_eff[idx[t]]))  # typical typed N in this block
        keep = r2_u >= min_imp_r2
        gu = idx[u]
        beta_hat[gu] = np.where(keep, bhat_u, 0.0)
        imp_r2[gu] = np.where(keep, r2_u, 0.0)
        out_n[gu] = np.where(keep, n_t * r2_u, 0.0)

    return ImputeResult(beta_hat=beta_hat, n_eff=out_n, imp_r2=imp_r2,
                        typed_mask=typed_mask, n_typed=int(typed_mask.sum()),
                        n_imputed=int((~typed_mask).sum()))
