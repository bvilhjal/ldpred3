"""Trait-scale helpers for binary (case/control) GWAS and PRS output.

LDpred3's sampler is scale-agnostic — it works on standardized marginal effects —
but two scale choices matter for *binary* traits, and getting them wrong quietly
costs accuracy:

* :func:`n_eff_case_control` — the **effective** sample size of a case/control
  GWAS, ``4 / (1/N_case + 1/N_control)``. Using the raw total ``N_case+N_control``
  (or a per-cohort constant) instead is the most common binary-trait mistake; the
  effective N is what the LDpred likelihood needs (and what the per-variant
  N-imputation anchors to).
* :func:`h2_liability` — convert an **observed-scale** SNP heritability (what the
  sampler / LDSC report for a 0/1 phenotype) to the **liability scale** (Lee et
  al., *AJHG* 2011), the comparable, prevalence-aware quantity.

Plus :func:`standardize_prs` for interpretable PRS output (z-scores / percentiles
against a reference distribution). NumPy + stdlib only.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

__all__ = ["n_eff_case_control", "h2_liability", "standardize_prs"]


def n_eff_case_control(n_case, n_control):
    """Effective sample size of a case/control GWAS: ``4/(1/N_case + 1/N_control)``.

    Equals the total ``N_case + N_control`` for a balanced study and tends to
    ``4·N_case`` when controls greatly outnumber cases — the effective number of
    informative observations, which is what should be passed as ``n_eff`` (and
    what the per-variant N-imputation anchors to). Accepts scalars or arrays.
    """
    n_case = np.asarray(n_case, dtype=float)
    n_control = np.asarray(n_control, dtype=float)
    if np.any(n_case <= 0) or np.any(n_control <= 0):
        raise ValueError("n_case and n_control must be positive")
    n = 4.0 / (1.0 / n_case + 1.0 / n_control)
    return float(n) if n.ndim == 0 else n


def h2_liability(h2_observed, prevalence, *, prop_cases=None):
    """Convert observed-scale SNP h² to the liability scale (Lee et al. 2011).

    A 0/1 (case/control) phenotype gives heritability on the *observed* scale,
    which depends on the case proportion in the study and is not comparable across
    studies. The liability-scale h² (an underlying-normal threshold model) is the
    standard, prevalence-aware quantity::

        h²_liab = h²_obs · [K(1−K)]² / (z² · P(1−P))

    with ``K`` the **population prevalence**, ``P`` the **case proportion in the
    GWAS** (``= K`` for a population-representative sample), and ``z = φ(t)`` the
    standard-normal density at the threshold ``t = Φ⁻¹(1−K)``.

    Parameters
    ----------
    h2_observed : float or array_like
        Observed-scale SNP heritability (e.g. ``ldpred3_auto_infer(...).h2_est``
        or ``ldsc_h2(...).h2`` fit on a 0/1 phenotype).
    prevalence : float
        Population disease prevalence ``K`` in ``(0, 1)``.
    prop_cases : float, optional
        Case proportion in the GWAS sample. Defaults to ``prevalence`` (a
        population sample); set it for an ascertained case/control study.
    """
    K = float(prevalence)
    if not 0.0 < K < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    P = K if prop_cases is None else float(prop_cases)
    if not 0.0 < P < 1.0:
        raise ValueError("prop_cases must be in (0, 1)")
    nd = NormalDist()
    t = nd.inv_cdf(1.0 - K)
    z = nd.pdf(t)
    factor = (K * (1.0 - K)) ** 2 / (z * z * P * (1.0 - P))
    h2 = np.asarray(h2_observed, dtype=float) * factor
    return float(h2) if h2.ndim == 0 else h2


def standardize_prs(scores, *, ref_mean=None, ref_sd=None):
    """Put a PRS on an interpretable scale: z-scores and percentiles.

    Returns ``(z, percentile)`` where ``z = (PRS − mean) / sd`` and ``percentile``
    is the empirical rank in ``[0, 1]``. By default the cohort's own mean/SD are
    used; pass ``ref_mean``/``ref_sd`` (e.g. from a reference cohort) to place a
    new cohort on the *same* scale — the standard way to report an individual's
    PRS as "X standard deviations / Yth percentile" comparably across cohorts.
    """
    s = np.asarray(scores, dtype=float)
    mu = float(np.mean(s)) if ref_mean is None else float(ref_mean)
    sd = float(np.std(s)) if ref_sd is None else float(ref_sd)
    z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
    # empirical percentile (average rank, ties shared), in [0, 1]
    order = np.argsort(np.argsort(s))
    percentile = (order + 0.5) / s.shape[0] if s.shape[0] else s
    return z, percentile
