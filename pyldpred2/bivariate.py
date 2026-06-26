"""
Bivariate LDpred2-auto: jointly fit two traits that share an LD reference.

Each variant falls in one of **four** latent states with probabilities
``(pi00, pi10, pi01, pi11)``: causal for neither trait, trait 1 only, trait 2
only, or **both**. A trait-1-causal effect is ``N(0, s1)``, a trait-2-causal one
``N(0, s2)``, and a *both*-causal pair is drawn from ``N(0, Sigma)`` with
``Sigma = [[s1, s12], [s12, s2]]`` -- the off-diagonal ``s12`` is the genetic
covariance and is the only place the traits couple. The Gibbs step evaluates the
four bivariate-Gaussian likelihoods of the residual estimate, samples the state,
then draws the effects; ``pi`` and ``(s1, s2, s12)`` are re-estimated each sweep.

This **per-trait** indicator (rather than a single shared one) is what makes the
joint model safe: whether the two traits' causal variants co-occur is *learned*
(``pi11``), not assumed. Two traits that share causal variants and are
genetically correlated let the better-powered one sharpen the other (via the
``both`` component); two traits with disjoint causal variants drive ``pi11 -> 0``
so the joint fit reduces to the independent ones and does no harm.

Both GWAS are assumed to use the **same** LD reference (same ancestry). Sample
overlap can be passed via ``cross_corr`` (the cross-trait correlation of the
sampling noise, i.e. the bivariate-LDSC intercept); the default 0 assumes
independent GWAS samples. NumPy only (optional Numba).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ldpred2 import _jit, _stable_postp, _as_n_vector, ldpred2_by_blocks

__all__ = ["BivariateResult", "ldpred2_auto_bivariate",
           "ldpred2_auto_bivariate_blocks"]

DAMP = 0.2          # damping factor for the variance-component updates


def _gv_blocks(fblocks, beta):
    """Genetic variance beta' R beta summed over (R, start, k) blocks."""
    g = 0.0
    for R, start, k in fblocks:
        sl = slice(start, start + k)
        g += float(beta[sl] @ (R.astype(np.float64) @ beta[sl]))
    return g


def _bivar_one_sweep(corr, bh1, bh2, n1, n2, curr1, curr2, rb1, rb2,
                     rbsum1, rbsum2, unif, z1, z2,
                     lpi00, lpi10, lpi01, lpi11, s1, s2, s12, cross_corr, resync):
    """One Gibbs sweep of the 4-state model over a block; mutates in place.

    States: 0 = null, 1 = trait-1 only, 2 = trait-2 only, 3 = both. Returns
    ``(c10, c01, c11, sum1sq, sum2sq, sum12, gv11, gv12, gv22)``: per-state counts
    and effect (co)moments for the hyper-parameter update, and the
    (co)heritability quadratics ``beta_t' R beta_u``. ``rbsum1/2`` accumulate the
    Rao-Blackwellised effects ``sum_state P(state) E[beta | state]``.
    """
    k = bh1.shape[0]
    if resync:                                   # rebuild R@beta to clear drift
        for i in range(k):
            rb1[i] = 0.0
            rb2[i] = 0.0
        for j in range(k):
            b1 = curr1[j]
            b2 = curr2[j]
            if b1 != 0.0 or b2 != 0.0:
                cj = corr[j]
                for i in range(k):
                    rb1[i] += cj[i] * b1
                    rb2[i] += cj[i] * b2

    c10 = 0
    c01 = 0
    c11 = 0
    sum1sq = 0.0
    sum2sq = 0.0
    sum12 = 0.0
    for j in range(k):
        b1 = curr1[j]
        b2 = curr2[j]
        d1 = bh1[j] - rb1[j] + b1                 # residual marginal estimates
        d2 = bh2[j] - rb2[j] + b2
        nn1 = n1[j]
        nn2 = n2[j]

        # noise covariance E of (d1, d2) and its inverse.
        E11 = 1.0 / nn1
        E22 = 1.0 / nn2
        E12 = cross_corr / np.sqrt(nn1 * nn2)
        Ei11 = E22 / (E11 * E22 - E12 * E12)
        Ei22 = E11 / (E11 * E22 - E12 * E12)
        Ei12 = -E12 / (E11 * E22 - E12 * E12)

        # log N(d; 0, E + Slab_state) for each of the 4 states (drop 2*pi const).
        # state 0: Cov = E
        det0 = E11 * E22 - E12 * E12
        q0 = (E22 * d1 * d1 - 2.0 * E12 * d1 * d2 + E11 * d2 * d2) / det0
        w0 = lpi00 - 0.5 * np.log(det0) - 0.5 * q0
        # state 1: Cov = E + diag(s1, 0)
        a11 = E11 + s1
        det1 = a11 * E22 - E12 * E12
        q1 = (E22 * d1 * d1 - 2.0 * E12 * d1 * d2 + a11 * d2 * d2) / det1
        w1 = lpi10 - 0.5 * np.log(det1) - 0.5 * q1
        # state 2: Cov = E + diag(0, s2)
        a22 = E22 + s2
        det2 = E11 * a22 - E12 * E12
        q2 = (a22 * d1 * d1 - 2.0 * E12 * d1 * d2 + E11 * d2 * d2) / det2
        w2 = lpi01 - 0.5 * np.log(det2) - 0.5 * q2
        # state 3: Cov = E + Sigma
        b11 = E11 + s1
        b22 = E22 + s2
        b12 = E12 + s12
        det3 = b11 * b22 - b12 * b12
        q3 = (b22 * d1 * d1 - 2.0 * b12 * d1 * d2 + b11 * d2 * d2) / det3
        w3 = lpi11 - 0.5 * np.log(det3) - 0.5 * q3

        wmax = w0
        if w1 > wmax:
            wmax = w1
        if w2 > wmax:
            wmax = w2
        if w3 > wmax:
            wmax = w3
        e0 = np.exp(w0 - wmax)
        e1 = np.exp(w1 - wmax)
        e2 = np.exp(w2 - wmax)
        e3 = np.exp(w3 - wmax)
        tot = e0 + e1 + e2 + e3
        p0 = e0 / tot
        p1 = e1 / tot
        p2 = e2 / tot
        p3 = e3 / tot

        # posterior effect means under each non-null state.
        # state 1 (trait-1 only): 1D posterior for beta1, beta2 = 0.
        prec1 = Ei11 + 1.0 / s1
        m1_1 = (Ei11 * d1 + Ei12 * d2) / prec1
        v1_1 = 1.0 / prec1
        # state 2 (trait-2 only)
        prec2 = Ei22 + 1.0 / s2
        m2_2 = (Ei22 * d2 + Ei12 * d1) / prec2
        v2_2 = 1.0 / prec2
        # state 3 (both): bivariate posterior, P = E^-1 + Sigma^-1.
        dS = s1 * s2 - s12 * s12
        Si11 = s2 / dS
        Si22 = s1 / dS
        Si12 = -s12 / dS
        P11 = Ei11 + Si11
        P12 = Ei12 + Si12
        P22 = Ei22 + Si22
        dP = P11 * P22 - P12 * P12
        V11 = P22 / dP
        V22 = P11 / dP
        V12 = -P12 / dP
        g1 = Ei11 * d1 + Ei12 * d2
        g2 = Ei12 * d1 + Ei22 * d2
        m1_3 = V11 * g1 + V12 * g2
        m2_3 = V12 * g1 + V22 * g2

        # Rao-Blackwell estimate: E[beta_t] = sum_state P(state) E[beta_t|state].
        rbsum1[j] += p1 * m1_1 + p3 * m1_3
        rbsum2[j] += p2 * m2_2 + p3 * m2_3

        # sample a state from (p0, p1, p2, p3).
        u = unif[j]
        if u < p0:
            new1 = 0.0
            new2 = 0.0
        elif u < p0 + p1:
            new1 = m1_1 + np.sqrt(v1_1) * z1[j]
            new2 = 0.0
            c10 += 1
            sum1sq += new1 * new1
        elif u < p0 + p1 + p2:
            new1 = 0.0
            new2 = m2_2 + np.sqrt(v2_2) * z2[j]
            c01 += 1
            sum2sq += new2 * new2
        else:
            L11 = np.sqrt(V11)
            L21 = V12 / L11
            t = V22 - L21 * L21
            L22 = np.sqrt(t) if t > 0.0 else 0.0
            new1 = m1_3 + L11 * z1[j]
            new2 = m2_3 + L21 * z1[j] + L22 * z2[j]
            c11 += 1
            sum1sq += new1 * new1
            sum2sq += new2 * new2
            sum12 += new1 * new2

        dlt1 = new1 - b1
        dlt2 = new2 - b2
        if dlt1 != 0.0 or dlt2 != 0.0:
            cj = corr[j]
            for i in range(k):
                cij = cj[i]
                rb1[i] += cij * dlt1
                rb2[i] += cij * dlt2
            curr1[j] = new1
            curr2[j] = new2

    gv11 = 0.0
    gv12 = 0.0
    gv22 = 0.0
    for i in range(k):
        gv11 += curr1[i] * rb1[i]
        gv12 += curr1[i] * rb2[i]
        gv22 += curr2[i] * rb2[i]
    return c10, c01, c11, sum1sq, sum2sq, sum12, gv11, gv12, gv22


_bivar_one_sweep_jit = _jit(_bivar_one_sweep)


@dataclass
class BivariateResult:
    """Output of :func:`ldpred2_auto_bivariate`.

    ``beta1_est`` / ``beta2_est`` are the posterior-mean (standardized) effects
    for the two traits, ``h2`` the pair of SNP heritabilities, ``rg`` the
    estimated genetic correlation, ``p`` the causal fraction and ``sigma`` the
    learned 2x2 effect covariance.
    """

    beta1_est: np.ndarray
    beta2_est: np.ndarray
    h2: tuple
    rg: float
    p: float
    sigma: np.ndarray

    def __repr__(self):
        return (f"BivariateResult(h2=({self.h2[0]:.3f}, {self.h2[1]:.3f}), "
                f"rg={self.rg:+.3f}, p={self.p:.4g}, "
                f"n_variants={len(self.beta1_est)})")


def ldpred2_auto_bivariate_blocks(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, *,
                                  h2_init=0.1, p_init=0.1, rg_init=0.0,
                                  cross_corr=0.0, burn_in=200, num_iter=200,
                                  h2_bounds=(1e-4, 1.0), h2_cap=None, seed=None):
    """Genome-wide (streaming) bivariate LDpred2-auto.

    ``blocks`` is the ``[(R, idx), ...]`` list (contiguous ``idx`` partitioning
    ``0..m-1``) used elsewhere; the two traits' summary statistics share it. The
    effect sweeps run one block at a time while ``pi`` and ``Sigma`` are pooled
    globally, so the genome-wide LD is never materialised.

    Parameters
    ----------
    blocks : list of (ndarray, ndarray)
        Per-block LD ``(R, idx)`` partitioning ``0..m-1``.
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits (same variant order).
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    h2_init, p_init, rg_init : float
        Initial heritability, causal fraction and genetic correlation.
    cross_corr : float, default 0.0
        Cross-trait correlation of the sampling noise (sample overlap); must lie
        in ``(-1, 1)``. 0 assumes independent GWAS samples.
    burn_in, num_iter : int
        Burn-in and sampling sweeps.
    h2_bounds : (float, float)
        Clamp range for the per-trait heritabilities.
    h2_cap : (float, float), optional
        Per-trait heritability ceilings used to anchor the slab variances. If
        omitted they are estimated with a univariate ``-auto`` pre-pass per trait
        (two extra fits); pass known heritabilities to skip that cost.
    seed : int or None

    Returns
    -------
    BivariateResult
    """
    if not -1.0 < cross_corr < 1.0:
        raise ValueError("cross_corr must be in (-1, 1)")
    bh1 = np.ascontiguousarray(beta_hat1, dtype=np.float64)
    bh2 = np.ascontiguousarray(beta_hat2, dtype=np.float64)
    m = bh1.shape[0]
    if bh2.shape[0] != m:
        raise ValueError("beta_hat1 and beta_hat2 must have the same length")
    n1 = _as_n_vector(n_eff1, m)
    n2 = _as_n_vector(n_eff2, m)

    fblocks = []
    for R, idx in sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0])):
        idx = np.asarray(idx)
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + idx.shape[0])):
            raise ValueError("each block must use contiguous indices")
        fblocks.append((np.ascontiguousarray(R, dtype=np.float32),
                        int(idx[0]), int(idx.shape[0])))
    starts = [s for _, s, _ in fblocks]
    ends = [s + k for _, s, k in fblocks]
    if (sum(k for _, _, k in fblocks) != m or starts[0] != 0
            or starts[1:] != ends[:-1] or ends[-1] != m):
        raise ValueError("blocks must partition 0..m-1 exactly once")

    lo, hi = h2_bounds
    M = float(m)

    # Anchor each trait's heritability ceiling to its own univariate-auto fit.
    # The weak trait's slab variance is poorly identified from its own data, so
    # without this it inflates by borrowing from a strong correlated trait; using
    # the univariate h2 as the cap keeps the joint scale honest. A caller that
    # already knows the heritabilities can pass h2_cap to skip the two fits.
    if h2_cap is None:
        b1u = ldpred2_by_blocks(blocks, bh1, n1, method="auto",
                                burn_in=burn_in, num_iter=num_iter, seed=seed)
        b2u = ldpred2_by_blocks(blocks, bh2, n2, method="auto",
                                burn_in=burn_in, num_iter=num_iter, seed=seed)
        h2_1c = _gv_blocks(fblocks, b1u)
        h2_2c = _gv_blocks(fblocks, b2u)
    else:
        h2_1c, h2_2c = h2_cap
    h2_1c = min(max(h2_1c, lo), hi)
    h2_2c = min(max(h2_2c, lo), hi)

    rng = np.random.default_rng(seed)
    curr1 = np.zeros(m); curr2 = np.zeros(m)
    rb1 = np.zeros(m); rb2 = np.zeros(m)
    avg1 = np.zeros(m); avg2 = np.zeros(m)
    count = 0
    gv_acc = np.zeros(3)

    # state probabilities (pi00, pi10, pi01, pi11) and slab variances.
    pi = np.array([1.0 - p_init, p_init / 3.0, p_init / 3.0, p_init / 3.0])
    s1 = s2 = float(h2_init) / max(p_init * M, 1.0)
    s12 = float(rg_init) * s1

    for it in range(burn_in + num_iter):
        resync = (it % 100 == 0)
        unif = rng.random(m)
        z1 = rng.standard_normal(m)
        z2 = rng.standard_normal(m)
        rbs1 = np.zeros(m); rbs2 = np.zeros(m)
        lpi = np.log(np.maximum(pi, 1e-300))
        c10 = c01 = c11 = 0
        S1 = S2 = S12 = 0.0
        gv11 = gv12 = gv22 = 0.0
        for R, start, k in fblocks:
            sl = slice(start, start + k)
            a10, a01, a11, s1sq, s2sq, s12s, g11, g12, g22 = _bivar_one_sweep_jit(
                R, bh1[sl], bh2[sl], n1[sl], n2[sl], curr1[sl], curr2[sl],
                rb1[sl], rb2[sl], rbs1[sl], rbs2[sl], unif[sl], z1[sl], z2[sl],
                float(lpi[0]), float(lpi[1]), float(lpi[2]), float(lpi[3]),
                float(s1), float(s2), float(s12), float(cross_corr), resync)
            c10 += a10; c01 += a01; c11 += a11
            S1 += s1sq; S2 += s2sq; S12 += s12s
            gv11 += g11; gv12 += g12; gv22 += g22

        # --- global hyper-parameter updates ---
        c00 = m - c10 - c01 - c11
        pi = rng.dirichlet([1.0 + c00, 1.0 + c10, 1.0 + c01, 1.0 + c11])
        n1c = c10 + c11
        n2c = c01 + c11
        # Damp the variance updates: a weak trait that borrows from a strong
        # correlated one can otherwise inflate its own slab in a feedback loop
        # (borrowed-large effects -> larger variance estimate -> more borrowing).
        if n1c > 0:
            s1 = (1.0 - DAMP) * s1 + DAMP * (S1 / n1c)
        if n2c > 0:
            s2 = (1.0 - DAMP) * s2 + DAMP * (S2 / n2c)
        if c11 > 0:
            s12 = (1.0 - DAMP) * s12 + DAMP * (S12 / c11)
        # cap each slab so implied per-trait h2 (= n_tc * s_t) <= univariate h2.
        s1 = min(max(s1, 1e-12), h2_1c / max(n1c, 1))
        s2 = min(max(s2, 1e-12), h2_2c / max(n2c, 1))
        mab = 0.999 * np.sqrt(s1 * s2)               # keep Sigma positive-definite
        s12 = min(max(s12, -mab), mab)

        if it >= burn_in:
            avg1 += rbs1; avg2 += rbs2
            gv_acc += (gv11, gv12, gv22)
            count += 1

    count = max(count, 1)
    g11, g12, g22 = gv_acc / count
    h2_1 = min(max(g11, lo), hi)
    h2_2 = min(max(g22, lo), hi)
    rg = g12 / np.sqrt(max(g11 * g22, 1e-12))
    rg = float(min(max(rg, -1.0), 1.0))
    return BivariateResult(beta1_est=avg1 / count, beta2_est=avg2 / count,
                           h2=(float(h2_1), float(h2_2)), rg=rg,
                           p=(float(pi[1] + pi[2] + pi[3])),
                           sigma=np.array([[s1, s12], [s12, s2]]))


def ldpred2_auto_bivariate(corr, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
    """Bivariate LDpred2-auto on a single dense LD matrix.

    Convenience wrapper over :func:`ldpred2_auto_bivariate_blocks` for one block
    (or a block-diagonal genome packed into one matrix). See that function and
    :class:`BivariateResult` for the parameters and output.
    """
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    m = corr.shape[0]
    return ldpred2_auto_bivariate_blocks([(corr, np.arange(m))], beta_hat1,
                                         beta_hat2, n_eff1, n_eff2, **kwargs)
