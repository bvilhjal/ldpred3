"""lassosum2: penalised-regression PRS from summary statistics + LD.

A complement to the Bayesian LDpred3 models (Mak et al. 2017; the lassosum2
variant in bigsnpr, Privé et al.). It minimises, over the standardized joint
effects ``β``::

    ½ βᵀ((1−s)R + sI)β  −  βᵀ r  +  λ‖β‖₁

where ``r`` are the standardized marginal effects (``beta_hat``), ``R`` the per-
block LD, ``s ∈ (0, 1]`` shrinks the LD toward the identity (regularisation /
robustness to a noisy reference), and ``λ`` is an L1 penalty giving a **sparse**
score. The L1 diagonal is 1 (since ``R_jj = 1``), so the coordinate-descent
update is a soft-threshold of the per-variant residual — reusing the same running
``Rβ`` the Gibbs sampler maintains.

lassosum2 fits a **grid** of ``(s, λ)`` and, with no validation cohort, picks the
best by **pseudo-validation** — the summary-statistic estimate of the PRS-trait
correlation ``βᵀr / √(βᵀRβ)`` (Privé et al.). The bigsnpr workflow runs this
alongside LDpred3-auto and keeps whichever predicts better; on some architectures
(very sparse, or a poor LD reference) the lasso wins.

NumPy-only, optional Numba; per block, so it streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._numba import _jit

__all__ = ["lassosum2", "Lassosum2Result"]


def _lassosum_sweep(R, r, beta, Rb, s, lam):
    """One coordinate-descent pass over a dense block; returns the max |Δβ|.

    Update: ``β_j ← softthr(r_j − (1−s)((Rβ)_j − β_j), λ)`` (the quadratic
    diagonal ``(1−s)·1 + s = 1``), then a rank-1 update of ``Rβ``.
    """
    k = beta.shape[0]
    max_change = 0.0
    for j in range(k):
        old = beta[j]
        resid = r[j] - (1.0 - s) * (Rb[j] - old)
        if resid > lam:
            new = resid - lam
        elif resid < -lam:
            new = resid + lam
        else:
            new = 0.0
        d = new - old
        if d != 0.0:
            cj = R[j]
            for i in range(k):
                Rb[i] += cj[i] * d
            beta[j] = new
            ad = d if d >= 0.0 else -d
            if ad > max_change:
                max_change = ad
    return max_change


_lassosum_sweep = _jit(_lassosum_sweep)


@dataclass
class Lassosum2Result:
    """Best lassosum2 fit plus the full ``(s, λ)`` grid.

    ``beta_est`` is the chosen (max-pseudo-validation) solution; ``best_s`` /
    ``best_lambda`` its hyper-parameters; ``grid`` the per-(s, λ) table of the
    pseudo-validation score and sparsity.
    """

    beta_est: np.ndarray = field(repr=False)
    best_s: float = 0.0
    best_lambda: float = 0.0
    best_score: float = 0.0
    n_nonzero: int = 0
    grid: list = field(default_factory=list, repr=False)

    def __repr__(self):
        return (f"Lassosum2Result(s={self.best_s:.2f}, lambda={self.best_lambda:.3g}, "
                f"pseudoval={self.best_score:.3f}, n_nonzero={self.n_nonzero})")


def _fblocks(blocks):
    out = []
    for R, idx in sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0])):
        idx = np.asarray(idx)
        out.append((np.ascontiguousarray(R, dtype=np.float64), idx))
    return out


def lassosum2(blocks, beta_hat, *, s_seq=(0.2, 0.5, 0.9), n_lambda=20,
             lambda_min_ratio=0.01, max_iter=100, tol=1e-4):
    """Fit lassosum2 over a ``(s, λ)`` grid; select by pseudo-validation.

    Parameters
    ----------
    blocks : list of (R, idx)
        Dense per-block LD partitioning ``0..m-1`` (as for the samplers).
    beta_hat : array_like (m,)
        Standardized marginal effects (``r`` in the objective).
    s_seq : sequence of float
        LD-shrinkage values to try (each in ``(0, 1]``; smaller = stronger LD).
    n_lambda : int
        Number of L1 penalties per ``s`` (a log-spaced path warm-started from the
        all-zero solution at ``λ_max = max|r|`` down to ``λ_max·lambda_min_ratio``).
    lambda_min_ratio, max_iter, tol : float/int
        Penalty-path floor, and the coordinate-descent budget / convergence.

    Returns
    -------
    Lassosum2Result
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    m = beta_hat.shape[0]
    fb = _fblocks(blocks)
    lam_max = float(np.max(np.abs(beta_hat))) if m else 0.0
    if lam_max <= 0.0:
        return Lassosum2Result(beta_est=np.zeros(m))
    lambdas = np.exp(np.linspace(np.log(lam_max),
                                 np.log(lam_max * lambda_min_ratio), int(n_lambda)))

    best = None
    grid = []
    for s in s_seq:
        s = float(s)
        if not 0.0 < s <= 1.0:
            raise ValueError("each s must be in (0, 1]")
        beta = np.zeros(m)                       # warm-start down the λ path
        Rb = np.zeros(m)
        for lam in lambdas:
            lam = float(lam)
            for _ in range(int(max_iter)):
                mc = 0.0
                for R, idx in fb:
                    sl = slice(int(idx[0]), int(idx[0]) + idx.shape[0])
                    mc = max(mc, _lassosum_sweep(R, beta_hat[sl], beta[sl],
                                                 Rb[sl], s, lam))
                if mc < tol:
                    break
            # pseudo-validation: betaᵀr / sqrt(betaᵀ R beta)
            bRb = float(beta @ Rb)
            score = float(beta @ beta_hat) / np.sqrt(bRb) if bRb > 1e-12 else 0.0
            nnz = int(np.count_nonzero(beta))
            grid.append({"s": s, "lambda": lam, "pseudoval": score, "n_nonzero": nnz})
            if best is None or score > best[0]:
                best = (score, s, lam, beta.copy(), nnz)

    score, s, lam, beta_est, nnz = best
    return Lassosum2Result(beta_est=beta_est, best_s=s, best_lambda=lam,
                           best_score=score, n_nonzero=nnz, grid=grid)
