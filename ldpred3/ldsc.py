"""
LD Score regression for SNP heritability (Bulik-Sullivan et al., *Nat Genet*
2015).

Under a polygenic model the expected chi-square of variant ``j`` is linear in its
**LD score** ``ell_j = sum_k r_jk^2``::

    E[chi2_j] = intercept + (N * h2 / M) * ell_j

so regressing the per-variant chi-squares on the LD scores recovers ``h2`` from
the slope, while the intercept measures confounding (population stratification /
cryptic relatedness) and should be ~1 in its absence.

This is a faithful univariate implementation: LD scores from per-block
correlation matrices, weighted least squares with the standard iterative
heteroscedasticity (``1 / 2 mu^2``) and overcounting (``1 / ell``) weights, and a
delete-a-block jackknife for the standard errors. NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ld_scores", "ldsc_h2", "LDSCResult", "ldsc_rg", "LDSCRgResult"]


def ld_scores(blocks, *, n_ref=None):
    """Per-variant LD scores ``ell_j = sum_k r_jk^2`` from LD blocks.

    Parameters
    ----------
    blocks : list of (ndarray, ndarray) or ndarray
        Either a single dense correlation matrix, or the ``[(R, idx), ...]``
        block list used elsewhere (``R`` an ``(k, k)`` correlation block, ``idx``
        the columns it covers, tiling ``0 .. m-1``).
    n_ref : int, optional
        Sample size of the panel the LD was estimated from. If given, each
        ``r^2`` is bias-adjusted (``r^2 - (1 - r^2) / (n_ref - 2)``) to remove the
        upward bias of in-sample ``r^2``; omit for population / noise-free LD.

    Returns
    -------
    ndarray
        LD score per variant (includes the self term ``r_jj^2 = 1``).
    """
    if isinstance(blocks, np.ndarray):
        blocks = [(blocks, np.arange(blocks.shape[0]))]
    m = sum(int(np.asarray(idx).shape[0]) for _, idx in blocks)
    ell = np.zeros(m)
    for R, idx in blocks:
        r2 = np.asarray(R, dtype=float) ** 2
        if n_ref is not None:
            r2 = r2 - (1.0 - r2) / (n_ref - 2.0)
        ell[np.asarray(idx)] = r2.sum(axis=1)
    return ell


@dataclass
class LDSCResult:
    """Output of :func:`ldsc_h2`."""

    h2: float                   # SNP heritability (regression slope * M / N)
    h2_se: float                # block-jackknife standard error
    intercept: float            # ~1 with no confounding
    intercept_se: float
    mean_chisq: float
    ratio: float                # (intercept - 1) / (mean_chisq - 1), confounding share

    @property
    def h2_ci(self):
        """Approximate 95% CI from the jackknife SE."""
        return (self.h2 - 1.96 * self.h2_se, self.h2 + 1.96 * self.h2_se)

    def __repr__(self):
        return (f"LDSCResult(h2={self.h2:.3f} ± {self.h2_se:.3f}, "
                f"intercept={self.intercept:.3f}, mean_chi2={self.mean_chisq:.2f})")


def _wls(x, y, w, constrain_intercept):
    """Weighted least squares; returns (slope, intercept)."""
    if constrain_intercept is None:
        X = np.column_stack([np.ones_like(x), x])
        WX = X * w[:, None]
        coef = np.linalg.solve(X.T @ WX, X.T @ (w * y))
        return float(coef[1]), float(coef[0])
    c = float(constrain_intercept)
    slope = np.sum(w * x * (y - c)) / np.sum(w * x * x)
    return float(slope), c


def _weights(pred_mean, ell_w):
    """Heteroscedasticity (1 / 2 mu^2) x overcounting (1 / ell) regression weights."""
    het = 1.0 / (2.0 * np.maximum(pred_mean, 1e-6) ** 2)
    over = 1.0 / np.maximum(ell_w, 1.0)
    return het * over


def ldsc_h2(chisq, ld_scores, n_eff, *, m_snps=None, n_blocks=200,
            constrain_intercept=None, n_iter=2):
    """Estimate SNP heritability by LD Score regression.

    Parameters
    ----------
    chisq : array_like (m,)
        Per-variant chi-square statistics, ``chi2_j = z_j^2 = (beta_hat / se)^2``
        (equivalently ``N * beta_hat^2`` for standardized effects).
    ld_scores : array_like (m,)
        LD scores from :func:`ld_scores`.
    n_eff : float or array_like
        GWAS sample size (scalar or per-variant).
    m_snps : int, optional
        Number of SNPs the heritability is defined over (default: ``len(chisq)``).
    n_blocks : int, default 200
        Number of contiguous jackknife blocks for the standard errors.
    constrain_intercept : float, optional
        Fix the intercept (e.g. ``1.0`` to assume no confounding) instead of
        estimating it.
    n_iter : int, default 2
        Number of weight-update iterations (weights depend on the fitted mean).

    Returns
    -------
    LDSCResult
    """
    chisq = np.asarray(chisq, dtype=float)
    ell = np.asarray(ld_scores, dtype=float)
    m = chisq.shape[0]
    N = np.asarray(n_eff, dtype=float)
    if N.ndim == 0:
        N = np.full(m, float(N))
    M = float(m_snps if m_snps is not None else m)

    x = N * ell / M               # slope on x is h2 directly
    ell_w = np.maximum(ell, 1.0)

    def fit(mask=None):
        xi, yi, li = (x, chisq, ell_w) if mask is None else (x[mask], chisq[mask], ell_w[mask])
        pred = np.ones_like(yi)
        slope = intercept = 0.0
        for _ in range(n_iter + 1):
            w = _weights(pred, li)
            slope, intercept = _wls(xi, yi, w, constrain_intercept)
            pred = np.maximum(intercept + slope * xi, 1.0)
        return slope, intercept

    h2, intercept = fit()

    nb = int(min(n_blocks, m))
    splits = np.array_split(np.arange(m), nb)
    h2_jk = np.empty(nb)
    int_jk = np.empty(nb)
    for b in range(nb):
        keep = np.ones(m, dtype=bool)
        keep[splits[b]] = False
        h2_jk[b], int_jk[b] = fit(keep)
    fac = (nb - 1) / nb
    h2_se = float(np.sqrt(fac * np.sum((h2_jk - h2_jk.mean()) ** 2)))
    int_se = (0.0 if constrain_intercept is not None
              else float(np.sqrt(fac * np.sum((int_jk - int_jk.mean()) ** 2))))

    mean_chisq = float(chisq.mean())
    ratio = (intercept - 1.0) / (mean_chisq - 1.0) if mean_chisq > 1.0 else float("nan")
    return LDSCResult(h2=float(h2), h2_se=h2_se, intercept=float(intercept),
                      intercept_se=int_se, mean_chisq=mean_chisq, ratio=ratio)


# --------------------------------------------------------------------------- #
# Bivariate (cross-trait) LD Score regression for genetic correlation.
# --------------------------------------------------------------------------- #
def _fit_slope(y, x, ell_w, n_iter, constrain):
    """Iterated WLS of y on x with LDSC heteroscedasticity/overcounting weights."""
    pred = np.ones_like(y)
    slope = intercept = 0.0
    for _ in range(n_iter + 1):
        slope, intercept = _wls(x, y, _weights(pred, ell_w), constrain)
        pred = np.maximum(intercept + slope * x, 1.0)
    return slope, intercept


@dataclass
class LDSCRgResult:
    """Output of :func:`ldsc_rg`."""

    rg: float                   # genetic correlation (can fall outside [-1,1] when noisy)
    rg_se: float                # block-jackknife standard error
    gcov: float                 # genetic covariance (cross-trait slope)
    gcov_intercept: float       # ~0 without sample overlap
    h2: tuple                   # (h2_1, h2_2) marginal heritabilities

    @property
    def rg_ci(self):
        return (self.rg - 1.96 * self.rg_se, self.rg + 1.96 * self.rg_se)

    def __repr__(self):
        return (f"LDSCRgResult(rg={self.rg:+.3f} ± {self.rg_se:.3f}, "
                f"gcov={self.gcov:+.4f}, h2=({self.h2[0]:.3f}, {self.h2[1]:.3f}))")


def ldsc_rg(beta_hat1, beta_hat2, ld_scores, n_eff1, n_eff2, *, m_snps=None,
            n_blocks=200, n_iter=2, constrain_intercept=None):
    """Genetic correlation by cross-trait LD Score regression.

    Fits ``E[z1_j z2_j] = intercept + (sqrt(N1 N2) rho_g / M) ell_j`` (with
    ``z_t = sqrt(N_t) beta_hat_t``), giving the genetic covariance ``rho_g`` from
    the slope; the intercept measures sample-overlap confounding (~0 for
    independent GWAS). The genetic correlation is
    ``r_g = rho_g / sqrt(h2_1 h2_2)`` with the marginal heritabilities from
    univariate LD Score regression. Standard errors are by block jackknife.

    Parameters
    ----------
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits (same variant order).
    ld_scores : array_like (m,)
        LD scores from :func:`ld_scores`.
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    constrain_intercept : float, optional
        Fix the cross-trait intercept (e.g. ``0.0`` for non-overlapping samples).

    Returns
    -------
    LDSCRgResult
        ``rg`` is undefined when a marginal heritability estimate is
        non-positive (low power / heavy noise); it is then large and not
        meaningful, as with the reference LDSC. Check ``.h2`` in that case.
    """
    b1 = np.asarray(beta_hat1, dtype=float)
    b2 = np.asarray(beta_hat2, dtype=float)
    ell = np.asarray(ld_scores, dtype=float)
    m = b1.shape[0]
    N1 = np.full(m, float(n_eff1)) if np.ndim(n_eff1) == 0 else np.asarray(n_eff1, float)
    N2 = np.full(m, float(n_eff2)) if np.ndim(n_eff2) == 0 else np.asarray(n_eff2, float)
    M = float(m_snps if m_snps is not None else m)

    chi1 = N1 * b1 * b1
    chi2 = N2 * b2 * b2
    cross = np.sqrt(N1 * N2) * b1 * b2
    x1 = N1 * ell / M
    x2 = N2 * ell / M
    xc = np.sqrt(N1 * N2) * ell / M
    ell_w = np.maximum(ell, 1.0)

    def fit(sel):
        h1, i1 = _fit_slope(chi1[sel], x1[sel], ell_w[sel], n_iter, None)
        h2, i2 = _fit_slope(chi2[sel], x2[sel], ell_w[sel], n_iter, None)
        pred1 = np.maximum(i1 + h1 * x1[sel], 1.0)
        pred2 = np.maximum(i2 + h2 * x2[sel], 1.0)
        # cross-trait regression weight ~ 1 / (ell * var(z1 z2)), var ~ pred1*pred2.
        w = 1.0 / (ell_w[sel] * np.maximum(pred1 * pred2, 1e-6))
        gcov, ic = _wls(xc[sel], cross[sel], w, constrain_intercept)
        return h1, h2, gcov, ic

    full = np.ones(m, dtype=bool)
    h1, h2, gcov, ic = fit(full)
    denom = np.sqrt(max(h1 * h2, 1e-12))
    rg = gcov / denom

    nb = int(min(n_blocks, m))
    splits = np.array_split(np.arange(m), nb)
    rg_jk = np.empty(nb)
    for b in range(nb):
        keep = full.copy()
        keep[splits[b]] = False
        hb1, hb2, gb, _ = fit(keep)
        rg_jk[b] = gb / np.sqrt(max(hb1 * hb2, 1e-12))
    rg_se = float(np.sqrt((nb - 1) / nb * np.sum((rg_jk - rg_jk.mean()) ** 2)))

    return LDSCRgResult(rg=float(rg), rg_se=rg_se, gcov=float(gcov),
                        gcov_intercept=float(ic), h2=(float(h1), float(h2)))
