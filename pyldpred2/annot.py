"""
Learn the annotation -> prior map inside the LDpred2-auto sampler (SBayesRC).

Each SNP's causal probability is modelled as ``p_j = sigmoid(a_j . theta)``
where ``a_j`` is its functional-annotation vector and ``theta`` is learned
jointly with the effects. The Gibbs sampler alternates:

1. an effect-update sweep (the usual point-normal LDpred2 step) using the
   current per-SNP ``p_j``;
2. an update of ``theta`` given the current causal pattern.

Two strategies for step 2 (``learn=``):

* ``"eb"``  — empirical-Bayes: a ridge-regularised logistic (Newton/IRLS) step
  on the posterior inclusion probabilities. NumPy-only, fast, stable.
* ``"probit"`` — fully Bayesian: a probit link with Albert & Chib (1993) data
  augmentation, giving a conjugate Gaussian draw of ``theta``.

The learned ``theta`` are directly interpretable as functional-enrichment
coefficients (large positive => the annotation enriches for causal variants).
This operates on a dense LD matrix (one block, or a block-diagonal genome).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ldpred2 import _jit, _stable_postp, _as_n_vector, SparseLD

__all__ = ["AnnotResult", "ldpred2_auto_annot"]


# --------------------------------------------------------------------------- #
# Jitted effect-update kernel for a chunk of sweeps at a fixed per-SNP p_j.
# --------------------------------------------------------------------------- #
def _annot_chunk(corr, beta_hat, n, p_j, h2, h2_min, h2_max, n_sweeps, seed,
                 init_beta):
    """Run ``n_sweeps`` point-normal sweeps with a fixed per-SNP causal prob.

    Returns ``(curr_beta, h2, pip_sum, rb_sum)`` where ``pip_sum`` /
    ``rb_sum`` accumulate the per-SNP posterior inclusion probability and the
    Rao-Blackwellised effect over the chunk. ``Rb`` is resynced from
    ``init_beta`` at the start (chunks are short and infrequent).
    """
    np.random.seed(seed)
    m = beta_hat.shape[0]
    curr_beta = init_beta.copy()
    Rb = np.zeros(m)
    for k in range(m):
        bk = curr_beta[k]
        if bk != 0.0:
            ck = corr[k]
            for i in range(m):
                Rb[i] += ck[i] * bk

    pbar = 0.0
    for j in range(m):
        pbar += p_j[j]
    pbar /= m
    c1 = h2 / (pbar * m)
    post_var = c1 / (n * c1 + 1.0)
    post_sd = np.sqrt(post_var)
    half_log = 0.5 * np.log1p(n * c1)
    n_post_var = n * post_var
    lpo = np.log1p(-p_j) - np.log(p_j)

    pip_sum = np.zeros(m)
    rb_sum = np.zeros(m)
    for it in range(n_sweeps):
        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)
        nb_causal = 0
        for j in range(m):
            old = curr_beta[j]
            res_beta_j = beta_hat[j] - Rb[j] + old
            pv = post_var[j]
            post_mean = n_post_var[j] * res_beta_j
            log_odds = lpo[j] + half_log[j] - 0.5 * post_mean * post_mean / pv
            postp = _stable_postp(log_odds)
            pip_sum[j] += postp
            rb_sum[j] += postp * post_mean
            if unif[j] < postp:
                new = post_mean + gauss[j] * post_sd[j]
                nb_causal += 1
            else:
                new = 0.0
            delta = new - old
            if delta != 0.0:
                cj = corr[j]
                for i in range(m):
                    Rb[i] += cj[i] * delta
                curr_beta[j] = new
        h2 = 0.0
        for i in range(m):
            h2 += curr_beta[i] * Rb[i]
        if h2 < h2_min:
            h2 = h2_min
        elif h2 > h2_max:
            h2 = h2_max
    return curr_beta, h2, pip_sum, rb_sum


_annot_chunk_jit = _jit(_annot_chunk)


# --------------------------------------------------------------------------- #
# Vectorised normal CDF / inverse CDF (for the probit / Albert-Chib update).
# --------------------------------------------------------------------------- #
def _Phi(x):
    """Standard-normal CDF (Abramowitz & Stegun 7.1.26 erf approximation)."""
    z = x / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(z))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
            + t * (-1.453152027 + t * 1.061405429))))
    erf = np.sign(z) * (1.0 - poly * np.exp(-z * z))
    return 0.5 * (1.0 + erf)


def _Phi_inv(p):
    """Standard-normal inverse CDF (Acklam's rational approximation)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.zeros_like(p)
    lo = p < plow; hi = p > phigh; mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid] - 0.5; r = q * q
    x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
             (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return x


def _truncnorm(mu, gamma, rng):
    """Sample N(mu, 1) truncated to (0, inf) where gamma==1, else (-inf, 0)."""
    u = rng.random(mu.shape)
    pnm = _Phi(-mu)
    a = np.where(gamma > 0, pnm, 0.0)
    b = np.where(gamma > 0, 1.0, pnm)
    q = np.clip(a + u * (b - a), 1e-12, 1 - 1e-12)
    return mu + _Phi_inv(q)


@dataclass
class AnnotResult:
    """Output of :func:`ldpred2_auto_annot`."""

    beta_est: np.ndarray            # posterior-mean effects
    h2_est: float                   # SNP heritability
    theta: np.ndarray               # learned annotation coefficients
    annotation_names: list = None   # optional column labels


def _add_intercept(A):
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if not np.allclose(A[:, 0], 1.0):
        A = np.column_stack([np.ones(A.shape[0]), A])
    return A


def ldpred2_auto_annot(corr, beta_hat, n_eff, annotations, *, learn="eb",
                       h2_init=0.1, p_init=0.1, burn_in=200, num_iter=200,
                       theta_every=10, ridge=5.0, h2_bounds=(1e-4, 1.0),
                       annotation_names=None, seed=None):
    """LDpred2-auto that learns a per-SNP prior from functional annotations.

    Parameters
    ----------
    corr : ndarray (m, m)
        Dense LD correlation matrix.
    beta_hat : array_like (m,)
        Standardized marginal effects.
    n_eff : array_like or float
        GWAS sample size.
    annotations : array_like (m, K)
        Per-SNP annotation matrix. An intercept column is added if absent.
    learn : {"eb", "probit"}
        ``"eb"`` ridge-logistic (IRLS) or ``"probit"`` Albert-Chib update.
    theta_every : int
        Number of effect sweeps between annotation-coefficient updates.
    ridge : float
        Ridge penalty on the non-intercept coefficients.
    Returns
    -------
    AnnotResult
    """
    if isinstance(corr, SparseLD):
        raise NotImplementedError("ldpred2_auto_annot needs a dense LD matrix")
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)
    A = _add_intercept(annotations)
    if A.shape[0] != m:
        raise ValueError("annotations must have one row per variant")
    K = A.shape[1]
    lo, hi = h2_bounds

    theta = np.zeros(K)
    theta[0] = np.log(p_init / (1 - p_init))
    pen = np.ones(K); pen[0] = 0.0
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    chunk_seeds = ss.generate_state(2 * (burn_in + num_iter) // max(theta_every, 1) + 4)

    curr = np.zeros(m); h2 = float(h2_init)
    avg = np.zeros(m); avg_rounds = 0
    done = 0; r = 0
    while done < burn_in + num_iter:
        ns = min(theta_every, burn_in + num_iter - done)
        p_j = np.clip(1.0 / (1.0 + np.exp(-(A @ theta))), 1e-5, 0.99)
        curr, h2, pip_sum, rb_sum = _annot_chunk_jit(
            corr, beta_hat, n, p_j, float(h2), float(lo), float(hi),
            int(ns), int(chunk_seeds[r % len(chunk_seeds)]), curr)
        if done >= burn_in:                       # post-burn-in: accumulate
            avg += rb_sum / ns
            avg_rounds += 1
        done += ns; r += 1

        # --- update theta ---
        pip = np.clip(pip_sum / ns, 1e-6, 1 - 1e-6)
        if learn == "eb":
            s = 1.0 / (1.0 + np.exp(-(A @ theta)))
            W = np.maximum(s * (1 - s), 1e-6)
            grad = A.T @ (pip - s) - ridge * pen * theta
            H = A.T @ (W[:, None] * A) + ridge * np.diag(pen) + 1e-6 * np.eye(K)
            theta = theta + np.linalg.solve(H, grad)
        elif learn == "probit":
            gamma = (pip > 0.5).astype(float)     # current causal indicators
            z = _truncnorm(A @ theta, gamma, rng)
            V = np.linalg.inv(A.T @ A + ridge * np.diag(pen) + 1e-6 * np.eye(K))
            mean = V @ (A.T @ z)
            theta = mean + np.linalg.cholesky(V) @ rng.standard_normal(K)
        else:
            raise ValueError("learn must be 'eb' or 'probit'")

    beta_est = avg / max(avg_rounds, 1)
    return AnnotResult(beta_est=beta_est, h2_est=float(h2), theta=theta,
                       annotation_names=annotation_names)
