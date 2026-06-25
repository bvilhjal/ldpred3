"""
LDpred2-auto inference of heritability, polygenicity and predictive r²
(Privé, Albiñana, Pasaniuc & Vilhjálmsson, *AJHG* 2023).

LDpred2-auto estimates ``h2`` and ``p`` within its Gibbs sampler. Running many
chains from different ``p_init`` values, discarding chains that failed to
converge, and pooling the post-burn-in samples gives robust point estimates
with credible intervals — and, remarkably, an estimate of the PRS's
**out-of-sample predictive r²** with no validation set.

The three estimands, all on the standardized (allele-correlation) scale where
genotypes and phenotype have unit variance:

* **h²** ``= βᵀ R β`` averaged over post-burn-in sweeps and kept chains.
* **p**  the causal fraction, averaged the same way.
* **r²** ``= E[ b₁ᵀ R b₂ ]`` over sampled effect vectors ``b₁``, ``b₂`` drawn
  from *different* chains (hence independent). If prediction were perfect
  ``b₁ = b₂ = β`` and ``r² = h²``; with no power the draws are uncorrelated and
  ``r² ≈ 0``.

Chain QC follows the paper: keep chains whose fitted marginal effects
``R β̂`` have a spread (range) of at least ``0.95 ×`` the 95th-percentile spread
across chains, dropping chains that collapsed to ~0.

This operates on a single (dense) LD matrix — one block, or a block-diagonal
genome assembled with :func:`ldpred2.block_diagonal_ld`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ldpred2 import _gibbs_kernel_sample_jit, _as_n_vector, _check_h2_p, SparseLD

__all__ = ["InferResult", "ldpred2_auto_infer"]


@dataclass
class InferResult:
    """Output of :func:`ldpred2_auto_infer`."""

    beta_est: np.ndarray            # posterior-mean effects (kept-chain average)
    h2_est: float                   # heritability, posterior median
    h2_ci: tuple                    # (2.5%, 97.5%)
    p_est: float                    # polygenicity, posterior median
    p_ci: tuple
    r2_est: float                   # out-of-sample predictive r², median
    r2_ci: tuple
    n_chains: int
    n_chains_kept: int


def _prep_corr(corr, shrink_corr):
    if isinstance(corr, SparseLD):
        raise NotImplementedError("ldpred2_auto_infer needs a dense LD matrix")
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    if shrink_corr != 1.0:
        corr = corr * np.float32(shrink_corr)
        np.fill_diagonal(corr, np.float32(1.0))
    return corr


def ldpred2_auto_infer(corr, beta_hat, n_eff, *, n_chains=10,
                       p_init_range=(1e-4, 0.2), h2_init=0.1,
                       burn_in=200, num_iter=200, sample_every=5,
                       shrink_corr=1.0, h2_bounds=(1e-4, 1.0),
                       qc=True, qc_frac=0.95, qc_quantile=0.95, seed=None):
    """Multi-chain LDpred2-auto with h²/p/r² inference.

    Parameters
    ----------
    corr : ndarray (m, m)
        Dense LD correlation matrix (one block or a block-diagonal genome).
    beta_hat : array_like (m,)
        Standardized marginal GWAS effects.
    n_eff : array_like or float
        GWAS sample size.
    n_chains : int, default 10
        Number of Gibbs chains, started from log-spaced ``p_init`` values.
    p_init_range : (lo, hi), default (1e-4, 0.2)
        Range of initial polygenicities across chains.
    burn_in, num_iter : int
        Per-chain burn-in and sampling sweeps.
    sample_every : int, default 5
        Thinning for the retained sampled effect vectors used by the r²
        estimator.
    shrink_corr : float, default 1.0
        Off-diagonal LD shrinkage (and the ``coef_shrink`` used in the r²
        matrix product), 1.0 = none.
    h2_bounds : (float, float)
        Clamp for the per-sweep h² estimate.
    qc : bool, default True
        Apply chain quality-control filtering.
    qc_frac, qc_quantile : float
        Keep chains whose fitted-effect range exceeds
        ``qc_frac * quantile(ranges, qc_quantile)``.
    seed : int or None

    Returns
    -------
    InferResult
    """
    _check_h2_p(h2=h2_init, p=p_init_range[0])
    if n_chains < 2:
        raise ValueError("need >= 2 chains for the cross-chain r² estimate")
    lo, hi = h2_bounds
    corr = _prep_corr(corr, shrink_corr)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)

    p_inits = np.exp(np.linspace(np.log(p_init_range[0]),
                                 np.log(p_init_range[1]), n_chains))
    ss = np.random.SeedSequence(seed)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_chains)]

    betas, h2_paths, p_paths, samples = [], [], [], []
    for c in range(n_chains):
        avg_beta, h2_path, p_path, bsamp, _ = _gibbs_kernel_sample_jit(
            corr, beta_hat, n, float(h2_init), float(p_inits[c]),
            int(burn_in), int(num_iter), float(lo), float(hi),
            seeds[c], int(sample_every))
        betas.append(avg_beta)
        h2_paths.append(h2_path)
        p_paths.append(p_path)
        samples.append(bsamp)

    # Chain QC: drop chains whose fitted marginal effects R*beta barely vary.
    ranges = np.array([np.ptp(corr @ b) for b in betas])
    keep = np.arange(n_chains)
    if qc and np.any(np.isfinite(ranges)) and ranges.max() > 0:
        thresh = qc_frac * np.quantile(ranges, qc_quantile)
        kept = np.where(ranges > thresh)[0]
        if kept.size >= 2:
            keep = kept

    beta_est = np.mean([betas[c] for c in keep], axis=0)

    h2_pool = np.concatenate([h2_paths[c] for c in keep])
    p_pool = np.concatenate([p_paths[c] for c in keep])
    h2_est = float(np.median(h2_pool))
    h2_ci = tuple(float(x) for x in np.quantile(h2_pool, [0.025, 0.975]))
    p_est = float(np.median(p_pool))
    p_ci = tuple(float(x) for x in np.quantile(p_pool, [0.025, 0.975]))

    # Out-of-sample r²: cross-chain products b_j^T R b_i over sampled effects.
    Rb = {}
    for c in keep:
        s = samples[c]
        if s.shape[0] == 0:
            continue
        prod = (corr @ s.T).T                      # (n_saved, m) = R b
        if shrink_corr != 1.0:
            prod = shrink_corr * prod + (1.0 - shrink_corr) * s
        Rb[c] = prod
    r2_vals = []
    kept_with_samples = [c for c in keep if c in Rb]
    for ii, ci in enumerate(kept_with_samples):
        for cj in kept_with_samples[ii + 1:]:
            cross = samples[cj] @ Rb[ci].T         # (n_j, n_i) of b_j^T R b_i
            r2_vals.append(cross.ravel())
    if r2_vals:
        r2_all = np.concatenate(r2_vals)
        r2_est = float(np.median(r2_all))
        r2_ci = tuple(float(x) for x in np.quantile(r2_all, [0.025, 0.975]))
    else:
        r2_est, r2_ci = float("nan"), (float("nan"), float("nan"))

    return InferResult(
        beta_est=beta_est,
        h2_est=h2_est, h2_ci=h2_ci,
        p_est=p_est, p_ci=p_ci,
        r2_est=r2_est, r2_ci=r2_ci,
        n_chains=n_chains, n_chains_kept=int(len(keep)),
    )
