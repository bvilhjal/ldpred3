"""Fine-mapping from LDpred3 posterior inclusion probabilities (PIPs).

LDpred3's spike-and-slab Gibbs sampler already draws, for every SNP on every
sweep, the posterior probability that the SNP is causal (``postp``). Averaging it
over the post-burn-in sweeps is exactly the **posterior inclusion probability**
(PIP) that fine-mapping needs -- so fine-mapping reuses the same engine as PRS,
on the same convention ``beta_hat = R @ beta + noise, noise ~ N(0, R / N)``.

This module turns those PIPs into the standard fine-mapping outputs:

* :func:`ldpred3_pip` -- per-locus PIPs, posterior effect mean/SD and
  **credible sets** (with a purity score), by running the auto sampler (several
  chains) on one locus.
* :func:`finemap_by_blocks` -- the **genome-wide** driver: run the per-locus
  fine-mapper on every LD block (blocks are independent, so this is
  embarrassingly parallel) and assemble one genome-wide PIP vector plus a
  concatenated credible-set table.
* :func:`single_signal_finemap` -- a fast single-causal-variant approximate Bayes
  factor (ABF) baseline, exact when a locus has one signal and a useful oracle in
  tests.

**These are heuristic LDpred3-PIP credible sets, not SuSiE-RSS.** LDpred3 yields
one *marginal* PIP per SNP, not SuSiE's per-effect assignment vectors, so a
credible set here is built by clustering LD neighbours around a lead SNP — a
useful localisation heuristic, but for **multiple signals overlapping in the same
LD block the marginal PIP cannot say which signal a SNP belongs to**, and set
coverage is not guaranteed calibrated (a tie-expansion step mitigates but does
not fully fix this; see ``docs/finemap.md``). Use SuSiE-RSS / FINEMAP for
calibrated multi-signal credible sets.

NumPy-only, consistent with the rest of LDpred3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ldpred3 import _gibbs_kernel_sample_jit, _as_n_vector
from .ld_utils import SparseLD, LowRankLD


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class CredibleSet:
    """One fine-mapped signal: the smallest variant set reaching ``coverage``."""

    signal: int
    variants: np.ndarray            # global variant indices (into the locus/genome)
    pip: np.ndarray = field(repr=False)
    coverage: float = 0.0           # achieved cumulative PIP within the set
    purity_min_abs_r: float = 1.0   # min |r| among set members (the SuSiE purity)
    purity_mean_abs_r: float = 1.0
    lead_variant: str | None = None
    lead_pip: float = np.nan

    def __repr__(self):
        lead = self.lead_variant if self.lead_variant is not None else \
            int(self.variants[np.argmax(self.pip)])
        return (f"CredibleSet(signal={self.signal}, size={len(self.variants)}, "
                f"lead={lead}, lead_pip={self.lead_pip:.3f}, "
                f"purity={self.purity_min_abs_r:.2f})")


@dataclass
class FineMapResult:
    """Fine-mapping output for a locus (or the genome, from the block driver)."""

    pip: np.ndarray = field(repr=False)
    posterior_mean: np.ndarray = field(repr=False)
    posterior_sd: np.ndarray = field(repr=False)
    credible_sets: list = field(default_factory=list)
    n_signals_est: float = 0.0      # ~ sum(pip): expected number of causal variants
    h2_est: float = 0.0             # locus-level (single locus) or pooled (genome)
    p_est: float = 0.0
    converged: bool = True
    n_iter: int = 0
    diagnostics: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"FineMapResult(n_variants={len(self.pip)}, "
                f"n_credible_sets={len(self.credible_sets)}, "
                f"n_signals_est={self.n_signals_est:.2f}, "
                f"max_pip={float(np.max(self.pip)) if len(self.pip) else 0.0:.3f})")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_dense_corr(corr):
    """Return a dense float64 LD matrix, densifying compact representations.

    Fine-mapping is a dense per-locus operation (credible-set purity and the
    sampler both need the within-locus correlations), so compact ``SparseLD`` /
    ``LowRankLD`` blocks are expanded here.
    """
    if isinstance(corr, LowRankLD):
        R = np.asarray(corr.U, dtype=np.float64) @ np.asarray(corr.U, dtype=np.float64).T
        np.fill_diagonal(R, 1.0)
        return R
    if isinstance(corr, SparseLD):
        m = int(corr.m)
        R = np.zeros((m, m), dtype=np.float64)
        indptr, indices, data = corr.indptr, corr.indices, corr.data
        for i in range(m):
            for q in range(indptr[i], indptr[i + 1]):
                R[i, indices[q]] = data[q]
        np.fill_diagonal(R, 1.0)
        return R
    return np.ascontiguousarray(corr, dtype=np.float64)


def _credible_sets(pip, R, coverage, min_abs_corr, max_signals, variant_ids,
                   tie_r=0.95):
    """Build credible sets from a flat PIP vector + dense LD.

    LDpred3 gives one marginal PIP per SNP, not SuSiE's separable per-effect
    assignment vectors, so signals are separated by LD here: take the highest-PIP
    unclaimed variant as a signal anchor, gather its LD neighbours
    (``|r| >= min_abs_corr``), accumulate their PIP in descending order until the
    cumulative reaches ``coverage``, then drop the set if its purity (min pairwise
    ``|r|``) is below ``min_abs_corr``. Repeat ``round(sum(pip))`` times (the
    expected number of causal variants in the locus), capped at ``max_signals``.
    """
    pip = np.asarray(pip, dtype=np.float64)
    m = pip.shape[0]
    absR = np.abs(np.asarray(R, dtype=np.float64))
    claimed = np.zeros(m, dtype=bool)
    n_sig = min(int(max_signals), int(round(float(pip.sum()))))
    sets = []
    for sig in range(n_sig):
        avail = np.flatnonzero(~claimed)
        if avail.size == 0:
            break
        anchor = int(avail[np.argmax(pip[avail])])
        if pip[anchor] <= 0.0:
            break
        nbr = avail[absR[anchor, avail] >= min_abs_corr]
        order = nbr[np.argsort(pip[nbr])[::-1]]
        members, csum = [], 0.0
        for j in order:
            members.append(int(j))
            csum += pip[j]
            if csum >= coverage:
                break
        members = np.asarray(members, dtype=int)
        # Tie-expansion: LDpred3's spike-and-slab picks one of a set of nearly
        # indistinguishable proxies, so the marginal PIP over-concentrates and the
        # set can collapse below true coverage. Add any unclaimed variant in
        # near-perfect LD (|r| >= tie_r) with the lead -- the data genuinely cannot
        # tell them apart, so a calibrated set must contain them.
        lead = int(members[np.argmax(pip[members])])
        # Unclaimed variants outside this set, via a direct boolean mask (avoids
        # np.isin's internal sort — members is a small subset of avail).
        in_members = np.zeros(m, dtype=bool)
        in_members[members] = True
        pool = avail[~in_members[avail]]
        ties = pool[absR[lead, pool] >= tie_r]
        if ties.size:
            members = np.concatenate([members, ties])
        claimed[members] = True
        if members.size > 1:
            sub = absR[np.ix_(members, members)]
            iu = np.triu_indices(members.size, 1)
            purity_min = float(sub[iu].min())
            purity_mean = float(sub[iu].mean())
        else:
            purity_min = purity_mean = 1.0
        if purity_min < min_abs_corr:          # impure -> not a clean signal
            continue
        lead = int(members[np.argmax(pip[members])])
        sets.append(CredibleSet(
            signal=len(sets), variants=members, pip=pip[members],
            coverage=float(csum), purity_min_abs_r=purity_min,
            purity_mean_abs_r=purity_mean,
            lead_variant=(str(variant_ids[lead]) if variant_ids is not None else None),
            lead_pip=float(pip[lead])))
    return sets


# --------------------------------------------------------------------------- #
# Per-locus fine-mapping
# --------------------------------------------------------------------------- #
def ldpred3_pip(corr, beta_hat, n_eff, *, h2_init=0.1, p_init=1e-3,
                burn_in=200, num_iter=500, n_chains=4, sample_every=5,
                h2_bounds=(1e-4, 1.0), allow_jump_sign=False,
                estimate_p=False, estimate_h2=True,
                coverage=0.95, min_abs_corr=0.5, tie_r=0.95, max_signals=10,
                variant_ids=None, seed=1) -> FineMapResult:
    """Fine-map one locus with LDpred3 posterior inclusion probabilities.

    Runs the LDpred3-auto spike-and-slab sampler (``n_chains`` independent chains)
    on a single locus and returns per-SNP PIPs, posterior effect mean/SD and
    credible sets.

    Fine-mapping uses a **fixed sparse prior** by default (``estimate_p=False``,
    ``p_init=1e-3``): re-estimating polygenicity on one small locus is unstable
    and inflates PIPs where there is no signal, so the causal fraction is held at
    a sparse genome-wide-style value while ``h2`` (the signal strength) still
    adapts. Pass ``estimate_p=True`` to recover the original per-locus auto
    behaviour.

    Parameters
    ----------
    corr : ndarray (m, m) or SparseLD or LowRankLD
        The locus LD matrix; compact representations are densified.
    beta_hat : array_like (m,)
        Standardized marginal effects (see :func:`ldpred3.standardize_betas`);
        i.e. ``beta_hat = R @ beta + N(0, R/N)``.
    n_eff : array_like or float
        GWAS (effective) sample size, scalar or per-variant.
    p_init : float, default 1e-3
        Causal fraction (held fixed unless ``estimate_p``).
    coverage : float, default 0.95
        Target cumulative PIP for each credible set.
    min_abs_corr : float, default 0.5
        Purity threshold: credible sets whose members are not mutually correlated
        at least this much are dropped (and it also bounds each signal's LD
        neighbourhood).
    tie_r : float, default 0.95
        A credible set always also includes any variant in near-perfect LD
        (``|r| >= tie_r``) with its lead -- proxies the data cannot distinguish,
        which keeps the 95% set calibrated when the spike-and-slab PIP
        over-concentrates on one of them.
    """
    R = _as_dense_corr(corr)
    m = R.shape[0]
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = _as_n_vector(n_eff, m)
    h2_min, h2_max = float(h2_bounds[0]), float(h2_bounds[1])

    pip = np.zeros(m)
    post_mean = np.zeros(m)
    h2s, ps = [], []
    sample_pool = []
    for c in range(int(n_chains)):
        avg, h2p, pp, samp, n_saved, chain_pip = _gibbs_kernel_sample_jit(
            R, beta_hat, n, float(h2_init), float(p_init),
            int(burn_in), int(num_iter), h2_min, h2_max,
            int(seed) + c, int(sample_every), bool(allow_jump_sign),
            bool(estimate_p), bool(estimate_h2))
        pip += chain_pip
        post_mean += avg
        h2s.append(np.median(h2p))
        ps.append(np.median(pp))
        if n_saved:
            sample_pool.append(np.asarray(samp))
    pip /= n_chains
    post_mean /= n_chains
    np.clip(pip, 0.0, 1.0, out=pip)

    if sample_pool:
        allsamp = np.concatenate(sample_pool, axis=0)
        post_sd = allsamp.std(axis=0)
    else:                                    # no thinned samples retained
        post_sd = np.zeros(m)

    cs = _credible_sets(pip, R, coverage, min_abs_corr, max_signals, variant_ids,
                        tie_r=tie_r)
    return FineMapResult(
        pip=pip, posterior_mean=post_mean, posterior_sd=post_sd,
        credible_sets=cs, n_signals_est=float(pip.sum()),
        h2_est=float(np.mean(h2s)), p_est=float(np.mean(ps)),
        converged=True, n_iter=int(num_iter),
        diagnostics={"n_chains": int(n_chains), "burn_in": int(burn_in),
                     "num_iter": int(num_iter), "m": int(m)})


def single_signal_finemap(corr, beta_hat, n_eff, *, prior_var=0.04,
                          coverage=0.95, variant_ids=None) -> FineMapResult:
    """Single-causal-variant approximate Bayes factor (ABF) fine-mapping.

    A fast closed-form baseline that assumes *exactly one* causal variant in the
    locus (Wakefield ABF on the marginal z-scores). Exact in that regime and a
    useful oracle/cross-check; for multi-signal loci use :func:`ldpred3_pip`. The
    LD matrix is used only to score credible-set purity, not in the ABF itself.
    """
    R = _as_dense_corr(corr)
    m = R.shape[0]
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    n = _as_n_vector(n_eff, m)
    z = beta_hat * np.sqrt(n)                       # marginal z-scores
    shat2 = 1.0 / n                                 # var of the standardized effect
    V = float(prior_var)
    # Wakefield (2009) log approximate Bayes factor:
    #   lABF = 0.5*log(s²/(s²+V)) + 0.5*z²·V/(s²+V),   z = β̂/s.
    # The signal lives entirely in the z² term, so it must NOT be divided by N
    # (an earlier version cancelled z²=β̂²·N against a spurious /N, which flattened
    # every PIP to ~uniform regardless of the signal strength).
    lbf = 0.5 * (np.log(shat2 / (shat2 + V))
                 + z * z * V / (shat2 + V))
    w = np.exp(lbf - lbf.max())
    pip = w / w.sum()
    post_var = 1.0 / (1.0 / V + 1.0 / shat2)
    mu = post_var * (beta_hat / shat2)
    # Posterior is a spike-and-slab mixture: 0 w.p. (1-pip), N(mu, post_var) w.p.
    # pip. Its variance includes the between-component term, not just pip*post_var.
    post_mean = pip * mu
    post_second = pip * (post_var + mu * mu)
    post_sd = np.sqrt(np.maximum(post_second - post_mean * post_mean, 0.0))
    cs = _credible_sets(pip, R, coverage, 0.0, 1, variant_ids)  # one signal, no purity drop
    return FineMapResult(
        pip=pip, posterior_mean=post_mean, posterior_sd=post_sd,
        credible_sets=cs, n_signals_est=1.0, converged=True, n_iter=0,
        diagnostics={"method": "abf", "prior_var": V, "m": int(m)})


# --------------------------------------------------------------------------- #
# Genome-wide fine-mapping
# --------------------------------------------------------------------------- #
# Module-level worker + initializer so the genome-wide beta_hat/n/ids are stashed
# once per process (not re-pickled per block); only the small per-block (R, idx)
# travels with each task. Mirrors the pool pattern in infer.py.
_FM_WORKER = {}


def _fm_init(beta_hat, n, variant_ids, kw):
    _FM_WORKER.clear()
    _FM_WORKER.update(beta_hat=beta_hat, n=n, variant_ids=variant_ids, kw=kw)


def _fm_run(job):
    R, idx = job
    w = _FM_WORKER
    vids = None if w["variant_ids"] is None else w["variant_ids"][idx]
    res = ldpred3_pip(R, w["beta_hat"][idx], w["n"][idx], variant_ids=vids, **w["kw"])
    return idx, res


def finemap_by_blocks(blocks, beta_hat, n_eff, *, only_significant=None,
                      variant_ids=None, max_signals=10, coverage=0.95,
                      min_abs_corr=0.5, ncores=1, **pip_kw) -> FineMapResult:
    """Genome-wide fine-mapping: run :func:`ldpred3_pip` on every LD block.

    ``blocks`` is a list of ``(R, idx)`` pairs that partition ``0..m-1`` (the
    output of :func:`ldpred3.compute_ld_blocks`); each block is an independent
    locus, so they are fine-mapped separately and (optionally) in parallel. The
    per-block PIPs / posterior moments are scattered back into genome-wide
    vectors, and the credible sets are concatenated with their variant indices
    mapped to the genome.

    Parameters
    ----------
    blocks : list of (R, idx)
        Per-block LD (dense / SparseLD / LowRankLD) and global column indices.
    beta_hat, n_eff : standardized marginal effects and GWAS N (genome-wide).
    only_significant : float or None
        If set, skip blocks with no variant at ``|z| `` above the two-sided
        p-value threshold (e.g. ``5e-8``) -- the usual "fine-map loci around
        genome-wide-significant hits" mode. ``None`` fine-maps every block.
    ncores : int
        Fine-map blocks in parallel across this many processes (blocks are
        independent). ``1`` runs serially.
    """
    beta_hat = np.ascontiguousarray(beta_hat, dtype=np.float64)
    m_total = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m_total)
    z_thresh = None
    if only_significant is not None:
        from math import erfc, sqrt
        # two-sided p < only_significant  <=>  |z| > z_thresh
        # invert via the complementary error function (bisection, NumPy-only).
        target = float(only_significant)
        lo, hi = 0.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if erfc(mid / sqrt(2.0)) > target:
                lo = mid
            else:
                hi = mid
        z_thresh = 0.5 * (lo + hi)

    jobs = []
    for R, idx in blocks:
        idx = np.asarray(idx)
        if z_thresh is not None:
            zb = np.abs(beta_hat[idx] * np.sqrt(n[idx]))
            if not np.any(zb > z_thresh):
                continue
        jobs.append((R, idx))

    vids_all = None if variant_ids is None else np.asarray(variant_ids)
    kw = dict(coverage=coverage, min_abs_corr=min_abs_corr,
              max_signals=max_signals, **pip_kw)
    init_args = (beta_hat, n, vids_all, kw)
    if int(ncores) > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(ncores),
                                 initializer=_fm_init, initargs=init_args) as ex:
            results = list(ex.map(_fm_run, jobs))
    else:
        _fm_init(*init_args)
        results = [_fm_run(j) for j in jobs]

    pip = np.zeros(m_total)
    post_mean = np.zeros(m_total)
    post_sd = np.zeros(m_total)
    credible_sets, h2s, ps = [], [], []
    for idx, res in results:
        pip[idx] = res.pip
        post_mean[idx] = res.posterior_mean
        post_sd[idx] = res.posterior_sd
        h2s.append(res.h2_est)
        ps.append(res.p_est)
        for cs in res.credible_sets:
            cs.signal = len(credible_sets)
            cs.variants = idx[cs.variants]      # map locus indices -> genome
            credible_sets.append(cs)

    return FineMapResult(
        pip=pip, posterior_mean=post_mean, posterior_sd=post_sd,
        credible_sets=credible_sets, n_signals_est=float(pip.sum()),
        h2_est=float(np.sum(h2s)) if h2s else 0.0,
        p_est=float(np.mean(ps)) if ps else 0.0,
        converged=True, n_iter=0,
        diagnostics={"n_blocks": len(blocks), "n_blocks_finemapped": len(jobs),
                     "only_significant": only_significant})
