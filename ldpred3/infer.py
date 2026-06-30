"""
LDpred3-auto inference of heritability, polygenicity and predictive r²
(Privé, Albiñana, Pasaniuc & Vilhjálmsson, *AJHG* 2023).

LDpred3-auto estimates ``h2`` and ``p`` within its Gibbs sampler. Running many
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

It accepts either a single **dense** LD matrix or a **list of per-block**
``(R, idx)`` matrices. The blocks form is *streamed* — chains run one block at a
time and the genome-wide LD is never materialised — so inference scales beyond
the dense path's ``infer_max_variants`` ceiling. Blocks may be dense, banded
:class:`~ldpred3.SparseLD` (O(k·bandwidth)) or low-rank
:class:`~ldpred3.LowRankLD` (O(k·rank) eigenspace), so h²/p/r² inference scales
the same way scoring does — the compact representation is used end to end (the
sampler and the r² cross-products), never densified to ``k × k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ldpred3 import (_gibbs_kernel_sample_jit, _gibbs_blocks_stream_sample,
                      _as_n_vector, _check_h2_p, SparseLD, LowRankLD)

__all__ = ["InferResult", "ldpred3_auto_infer"]


@dataclass
class InferResult:
    """Output of :func:`ldpred3_auto_infer`.

    Each estimate has a 95% credible interval (the ``*_ci`` tuples).
    ``print(result)`` shows a compact summary.
    """

    beta_est: np.ndarray            # posterior-mean effects (kept-chain average)
    h2_est: float                   # heritability, posterior median
    h2_ci: tuple                    # (2.5%, 97.5%)
    p_est: float                    # polygenicity, posterior median
    p_ci: tuple
    r2_est: float                   # out-of-sample predictive r², median
    r2_ci: tuple
    n_chains: int
    n_chains_kept: int

    def __repr__(self):
        def ci(lo, hi):
            return f"({lo:.3g}, {hi:.3g})"
        return (f"InferResult(h2={self.h2_est:.3f} {ci(*self.h2_ci)}, "
                f"p={self.p_est:.4g} {ci(*self.p_ci)}, "
                f"r2={self.r2_est:.3f} {ci(*self.r2_ci)}, "
                f"chains_kept={self.n_chains_kept}/{self.n_chains})")


def _is_blocks(corr):
    """True if ``corr`` is a list/tuple of ``(R, idx)`` blocks (not a matrix)."""
    return isinstance(corr, (list, tuple))


def _prep_corr(corr, shrink_corr):
    if isinstance(corr, SparseLD):
        raise NotImplementedError("ldpred3_auto_infer needs a dense or blocks LD")
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    if shrink_corr != 1.0:
        corr = corr * np.float32(shrink_corr)
        np.fill_diagonal(corr, np.float32(1.0))
    return corr


def _prep_blocks(blocks, shrink_corr):
    """Prepare ``(R, idx)`` blocks for the streaming sampler.

    Dense blocks become float32 with the off-diagonal optionally shrunk; banded
    :class:`SparseLD` and low-rank :class:`LowRankLD` blocks are passed through
    unchanged (they scale the inference the same way they scale scoring).
    ``shrink_corr`` only applies to dense blocks — it is not defined on the
    compact representations, so a non-unit value with any such block is rejected.
    """
    has_compact = any(isinstance(R, (SparseLD, LowRankLD)) for R, _ in blocks)
    if has_compact and shrink_corr != 1.0:
        raise ValueError("shrink_corr != 1 is only supported for dense LD "
                         "blocks, not SparseLD / LowRankLD")
    out = []
    for R, idx in blocks:
        if isinstance(R, (SparseLD, LowRankLD)):
            out.append((R, np.asarray(idx)))
            continue
        R = np.ascontiguousarray(R, dtype=np.float32)
        if shrink_corr != 1.0:
            R = R * np.float32(shrink_corr)
            np.fill_diagonal(R, np.float32(1.0))
        out.append((R, np.asarray(idx)))
    return out


def _sparse_block_rmatmul(R, Sb):
    """``Sb @ R`` for a symmetric banded :class:`SparseLD` block (no densify).

    ``R`` is stored in CSR; column ``j`` of the result gathers ``Sb`` over row
    ``j``'s non-zero neighbours -- O(nnz · rows), O(k · bandwidth) memory.
    """
    out = np.zeros_like(Sb)
    indptr, indices, data = R.indptr, R.indices, R.data
    for j in range(R.m):
        seg = slice(int(indptr[j]), int(indptr[j + 1]))
        cols = indices[seg]
        vals = data[seg].astype(Sb.dtype, copy=False)
        out[:, j] = Sb[:, cols] @ vals
    return out


def _blocks_matmul(blocks, S):
    """``S @ R`` for block-diagonal ``R`` (S is ``(rows, m)``).

    Each block is multiplied in its native representation, so a compact block is
    never densified to ``k × k``: dense uses ``Sb @ R``, low-rank uses
    ``(Sb @ U) @ Uᵀ`` (O(rows·k·rank)), banded uses a sparse gather.
    """
    out = np.zeros_like(S)
    for R, idx in blocks:
        Sb = S[:, idx]
        if isinstance(R, LowRankLD):
            U = R.U.astype(S.dtype, copy=False)
            out[:, idx] = (Sb @ U) @ U.T
        elif isinstance(R, SparseLD):
            out[:, idx] = _sparse_block_rmatmul(R, Sb)
        else:
            out[:, idx] = Sb @ R.astype(S.dtype, copy=False)
    return out


# Worker state for parallel chains. Set once per process by the pool
# initializer so the (read-only) LD is not re-pickled per chain.
_WORKER = {}


def _chain_init(corr, beta_hat, n, h2_init, burn_in, num_iter, lo, hi,
                sample_every, allow_jump_sign, blocks):
    _WORKER.update(corr=corr, beta_hat=beta_hat, n=n, h2_init=h2_init,
                   burn_in=burn_in, num_iter=num_iter, lo=lo, hi=hi,
                   sample_every=sample_every, allow_jump_sign=allow_jump_sign,
                   blocks=blocks)


def _chain_run(p_init_and_seed):
    p_init, seed = p_init_and_seed
    w = _WORKER
    if w["blocks"] is not None:                    # streaming block-diagonal path
        avg, h2p, pp, samp = _gibbs_blocks_stream_sample(
            w["blocks"], w["beta_hat"], w["n"], float(w["h2_init"]),
            float(p_init), burn_in=int(w["burn_in"]), num_iter=int(w["num_iter"]),
            seed=int(seed), h2_bounds=(float(w["lo"]), float(w["hi"])),
            sample_every=int(w["sample_every"]),
            allow_jump_sign=bool(w["allow_jump_sign"]))
        return avg, h2p, pp, samp
    avg, h2p, pp, samp, _ = _gibbs_kernel_sample_jit(   # dense path
        w["corr"], w["beta_hat"], w["n"], float(w["h2_init"]), float(p_init),
        int(w["burn_in"]), int(w["num_iter"]), float(w["lo"]), float(w["hi"]),
        int(seed), int(w["sample_every"]), bool(w["allow_jump_sign"]))
    return avg, h2p, pp, samp


def ldpred3_auto_infer(corr, beta_hat, n_eff, *, n_chains=10,
                       p_init_range=(1e-4, 0.2), h2_init=0.1,
                       burn_in=200, num_iter=200, sample_every=5, ncores=1,
                       allow_jump_sign=True,
                       shrink_corr=1.0, h2_bounds=(1e-4, 1.0),
                       qc=True, qc_frac=0.95, qc_quantile=0.95, seed=None):
    """Multi-chain LDpred3-auto with h²/p/r² inference.

    Parameters
    ----------
    corr : ndarray (m, m) or list of (R, idx)
        Either a dense LD correlation matrix (one block or a block-diagonal
        genome), or a list of per-block ``(R, idx)`` matrices that partition
        ``0..m-1``. The blocks form is streamed (the genome-wide LD is never
        materialised), so it scales past the dense path's size limit.
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
    blocks = None
    if _is_blocks(corr):
        blocks = _prep_blocks(corr, shrink_corr)
        m = sum(int(np.asarray(idx).shape[0]) for _, idx in blocks)
        apply_R = lambda S: _blocks_matmul(blocks, S)       # noqa: E731
    else:
        corr = _prep_corr(corr, shrink_corr)
        m = corr.shape[0]
        apply_R = lambda S: S @ corr                        # noqa: E731
    beta_hat = np.asarray(beta_hat, dtype=float)
    if beta_hat.shape[0] != m:
        raise ValueError(f"beta_hat length {beta_hat.shape[0]} != LD size {m}")
    n = _as_n_vector(n_eff, m)

    p_inits = np.exp(np.linspace(np.log(p_init_range[0]),
                                 np.log(p_init_range[1]), n_chains))
    ss = np.random.SeedSequence(seed)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_chains)]

    work = list(zip(p_inits, seeds))
    initargs = (corr if blocks is None else None, beta_hat, n, float(h2_init),
                int(burn_in), int(num_iter), float(lo), float(hi),
                int(sample_every), bool(allow_jump_sign), blocks)
    if ncores and ncores > 1 and n_chains > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(ncores),
                                 initializer=_chain_init,
                                 initargs=initargs) as ex:
            results = list(ex.map(_chain_run, work))
    else:
        _chain_init(*initargs)
        results = [_chain_run(w) for w in work]

    betas, h2_paths, p_paths, samples = [], [], [], []
    for avg_beta, h2_path, p_path, bsamp in results:
        betas.append(avg_beta)
        h2_paths.append(h2_path)
        p_paths.append(p_path)
        samples.append(bsamp)

    # Chain QC: drop chains whose fitted marginal effects R*beta barely vary.
    ranges = np.ptp(apply_R(np.asarray(betas)), axis=1)
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
        prod = apply_R(s.astype(float))            # (n_saved, m) = R b
        if blocks is None and shrink_corr != 1.0:
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
