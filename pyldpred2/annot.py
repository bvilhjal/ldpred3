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
def _annot_chunk(corr, beta_hat, n, p_j, slab_j, h2_min, h2_max, n_sweeps, seed,
                 init_beta):
    """Run ``n_sweeps`` point-normal sweeps at fixed per-SNP causal prob ``p_j``
    and per-SNP slab variance ``slab_j``.

    Returns ``(curr_beta, h2, pip_sum, rb_sum, m2_sum)``: the per-SNP posterior
    inclusion probability, the Rao-Blackwellised effect, and the posterior
    second moment ``postp*(mean^2 + var)`` (used to learn the variance map),
    summed over the chunk. ``Rb`` is resynced from ``init_beta`` at the start.
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

    post_var = slab_j / (n * slab_j + 1.0)
    post_sd = np.sqrt(post_var)
    half_log = 0.5 * np.log1p(n * slab_j)
    n_post_var = n * post_var
    lpo = np.log1p(-p_j) - np.log(p_j)

    pip_sum = np.zeros(m)
    rb_sum = np.zeros(m)
    m2_sum = np.zeros(m)
    h2 = 0.0
    for it in range(n_sweeps):
        unif = np.random.random(m)
        gauss = np.random.standard_normal(m)
        for j in range(m):
            old = curr_beta[j]
            res_beta_j = beta_hat[j] - Rb[j] + old
            pv = post_var[j]
            post_mean = n_post_var[j] * res_beta_j
            log_odds = lpo[j] + half_log[j] - 0.5 * post_mean * post_mean / pv
            postp = _stable_postp(log_odds)
            pip_sum[j] += postp
            rb_sum[j] += postp * post_mean
            m2_sum[j] += postp * (post_mean * post_mean + pv)
            if unif[j] < postp:
                new = post_mean + gauss[j] * post_sd[j]
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
    return curr_beta, h2, pip_sum, rb_sum, m2_sum


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
    """Output of :func:`ldpred2_auto_annot`.

    ``theta`` are the learned annotation coefficients (the first is the
    intercept). A large positive coefficient means that annotation enriches for
    causal variants. Use :attr:`enrichment` for a name->coefficient mapping or
    just ``print(result)`` for a summary.
    """

    beta_est: np.ndarray            # posterior-mean effects
    h2_est: float                   # SNP heritability
    theta: np.ndarray               # inclusion-map coefficients (incl intercept)
    phi: np.ndarray = None          # variance-map coefficients (if learn_variance)
    annotation_names: list = None   # column labels (incl "intercept")

    @property
    def enrichment(self):
        """``{name: coefficient}`` for the non-intercept inclusion coefficients."""
        names = self.annotation_names or [f"annot_{i}" for i in range(len(self.theta))]
        return {nm: float(t) for nm, t in zip(names[1:], self.theta[1:])}

    @property
    def variance_enrichment(self):
        """``{name: coefficient}`` for the effect-variance map (or None)."""
        if self.phi is None:
            return None
        names = self.annotation_names or [f"annot_{i}" for i in range(len(self.phi))]
        return {nm: float(t) for nm, t in zip(names[1:], self.phi[1:])}

    def __repr__(self):
        items = sorted(self.enrichment.items(), key=lambda kv: -abs(kv[1]))
        top = ", ".join(f"{nm}={c:+.2f}" for nm, c in items[:6])
        more = "" if len(items) <= 6 else f", (+{len(items) - 6} more)"
        return (f"AnnotResult(h2_est={self.h2_est:.3f}, n_variants="
                f"{len(self.beta_est)}, enrichment[{top}{more}])")


def _add_intercept(A):
    """Return ``(A_with_intercept, had_intercept)``; intercept is column 0."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    had = A.shape[1] >= 1 and np.allclose(A[:, 0], 1.0)
    if not had:
        A = np.column_stack([np.ones(A.shape[0]), A])
    return A, had


def ldpred2_auto_annot(corr, beta_hat, n_eff, annotations, *, learn="eb",
                       learn_variance=False, h2_init=0.1, p_init=0.1,
                       burn_in=200, num_iter=200, theta_every=10, ridge=5.0,
                       h2_bounds=(1e-4, 1.0), annotation_names=None, seed=None):
    """LDpred2-auto that learns a per-SNP prior from functional annotations.

    Each SNP's causal probability is ``p_j = sigmoid(a_j . theta)`` and ``theta``
    is learned jointly with the effects (the SBayesRC idea). The learned
    coefficients are returned as interpretable functional-enrichment estimates.

    Parameters
    ----------
    corr : ndarray (m, m)
        Dense LD correlation matrix.
    beta_hat : array_like (m,)
        Standardized marginal effects.
    n_eff : array_like or float
        GWAS sample size.
    annotations : array_like, shape (m,) or (m, K)
        Per-SNP annotation matrix (binary or continuous). An intercept column is
        added automatically if not already present.
    learn : {"eb", "probit"}, default "eb"
        ``"eb"`` ridge-logistic (Newton/IRLS) update on the posterior inclusion
        probabilities, or ``"probit"`` fully-Bayesian Albert-Chib update.
    learn_variance : bool, default False
        Also learn an annotation -> effect-*variance* map ``sigma2_j ∝
        exp(a_j . phi)`` (the second half of SBayesRC). ``phi`` is fit by ridge
        regression of the per-causal second moment on the annotations and
        returned in ``AnnotResult.phi`` / ``.variance_enrichment``. Because it is
        learned, it harmlessly collapses to ~0 when effect size is
        annotation-independent.
    h2_init, p_init : float
        Initial heritability and baseline causal fraction (the latter sets the
        intercept).
    burn_in, num_iter : int
        Burn-in and sampling sweeps.
    theta_every : int, default 10
        Effect sweeps between annotation-coefficient updates.
    ridge : float, default 5.0
        Ridge penalty on the non-intercept coefficients (stabilises many /
        collinear annotations).
    annotation_names : list of str, optional
        Names for the non-intercept annotations (length ``K``), used to label the
        returned enrichment estimates.
    seed : int or None

    Returns
    -------
    AnnotResult
        ``.beta_est`` (effects), ``.h2_est``, ``.theta`` (coefficients incl.
        intercept) and ``.enrichment`` (a ``{name: coefficient}`` mapping). A
        large positive coefficient means the annotation enriches for causal
        variants.

    Examples
    --------
    >>> res = ldpred2_auto_annot(corr, beta_hat, n_eff, A,
    ...                          annotation_names=["coding", "conserved"])
    >>> res.enrichment            # {"coding": 1.2, "conserved": 0.8}
    >>> res.beta_est              # adjusted effects to build the PRS
    """
    if isinstance(corr, SparseLD):
        raise NotImplementedError("ldpred2_auto_annot needs a dense LD matrix")
    if learn not in ("eb", "probit"):
        raise ValueError("learn must be 'eb' or 'probit'")
    if theta_every < 1:
        raise ValueError("theta_every must be >= 1")
    if not 0.0 < p_init < 1.0:
        raise ValueError("p_init must be in (0, 1)")
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    beta_hat = np.asarray(beta_hat, dtype=float)
    m = beta_hat.shape[0]
    n = _as_n_vector(n_eff, m)

    annot_in = np.asarray(annotations, dtype=float)
    if annot_in.ndim == 1:
        annot_in = annot_in[:, None]
    if annot_in.shape[0] != m:
        raise ValueError(
            f"annotations must have one row per variant "
            f"({m}); got {annot_in.shape[0]}")
    if not np.all(np.isfinite(annot_in)):
        raise ValueError("annotations must be finite (no NaN/inf)")
    A, _ = _add_intercept(annot_in)
    K = A.shape[1]

    # Labels name the non-intercept annotations (K-1 of them).
    n_user = K - 1
    if annotation_names is None:
        user_names = [f"annot_{i}" for i in range(n_user)]
    else:
        user_names = list(annotation_names)
        if len(user_names) != n_user:
            raise ValueError(
                f"annotation_names must have {n_user} entries "
                f"(one per non-intercept annotation); got {len(user_names)}")
    names = ["intercept"] + user_names
    lo, hi = h2_bounds

    theta = np.zeros(K)
    theta[0] = np.log(p_init / (1 - p_init))
    phi = np.zeros(K)                              # variance-map coefficients
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
        # per-SNP slab variance: exp(A phi) normalised so sum_j p_j slab_j = h2.
        rel = np.exp(np.clip(A @ phi, -10, 10)) if learn_variance else np.ones(m)
        slab_j = h2 * rel / max(np.sum(p_j * rel), 1e-12)
        curr, h2, pip_sum, rb_sum, m2_sum = _annot_chunk_jit(
            corr, beta_hat, n, p_j, slab_j, float(lo), float(hi),
            int(ns), int(chunk_seeds[r % len(chunk_seeds)]), curr)
        if done >= burn_in:                       # post-burn-in: accumulate
            avg += rb_sum / ns
            avg_rounds += 1
        done += ns; r += 1

        # --- update theta (inclusion map) ---
        pip = np.clip(pip_sum / ns, 1e-6, 1 - 1e-6)
        if learn == "eb":
            s = 1.0 / (1.0 + np.exp(-(A @ theta)))
            W = np.maximum(s * (1 - s), 1e-6)
            grad = A.T @ (pip - s) - ridge * pen * theta
            H = A.T @ (W[:, None] * A) + ridge * np.diag(pen) + 1e-6 * np.eye(K)
            theta = theta + np.linalg.solve(H, grad)
        else:                                     # "probit" (validated above)
            gamma = (pip > 0.5).astype(float)     # current causal indicators
            z = _truncnorm(A @ theta, gamma, rng)
            V = np.linalg.inv(A.T @ A + ridge * np.diag(pen) + 1e-6 * np.eye(K))
            mean = V @ (A.T @ z)
            theta = mean + np.linalg.cholesky(V) @ rng.standard_normal(K)

        # --- update phi (effect-variance map): ridge regression of the log
        #     per-causal second moment E[beta^2 | causal] on the annotations. ---
        if learn_variance:
            e2 = np.clip(m2_sum / np.maximum(pip_sum, 1e-8), 1e-12, None)
            wts = pip                              # weight by how causal a SNP is
            y = np.log(e2)
            WA = wts[:, None] * A
            Hphi = A.T @ WA + ridge * np.diag(pen) + 1e-6 * np.eye(K)
            gphi = A.T @ (wts * (y - A @ phi)) - ridge * pen * phi
            phi = phi + np.linalg.solve(Hphi, gphi)

    beta_est = avg / max(avg_rounds, 1)
    return AnnotResult(beta_est=beta_est, h2_est=float(h2), theta=theta,
                       phi=(phi if learn_variance else None),
                       annotation_names=names)
