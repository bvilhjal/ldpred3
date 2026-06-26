"""
Bivariate LDpred2-auto: jointly fit two traits that share an LD reference.

Each variant is null (both effects 0) with probability ``1 - p`` or causal, in
which case its effect pair ``(beta1_j, beta2_j)`` is drawn from a bivariate
normal ``N(0, Sigma)`` with a 2x2 covariance ``Sigma`` whose off-diagonal is the
genetic covariance of the two traits. The Gibbs sampler updates each pair with an
explicit 2x2 mixture step (a bivariate Bayes factor for inclusion, then a
correlated draw), and re-estimates ``p`` and ``Sigma`` each sweep. When the two
traits are genetically correlated, the better-powered one sharpens the other's
effects through ``Sigma`` -- the point of a joint model.

Both GWAS are assumed to use the **same** LD reference (same ancestry). Sample
overlap can be passed via ``cross_corr`` (the cross-trait correlation of the
sampling noise, i.e. the bivariate-LDSC intercept); the default 0 assumes
independent GWAS samples. NumPy only (optional Numba).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ldpred2 import _jit, _stable_postp, _as_n_vector

__all__ = ["BivariateResult", "ldpred2_auto_bivariate",
           "ldpred2_auto_bivariate_blocks"]


def _bivar_one_sweep(corr, bh1, bh2, n1, n2, curr1, curr2, rb1, rb2,
                     rbsum1, rbsum2, unif, z1, z2, p, s11, s12, s22,
                     cross_corr, resync):
    """One Gibbs sweep over a block; mutates curr/rb/rbsum in place.

    Returns ``(n_causal, C11, C12, C22, gv11, gv12, gv22)``: the sampled causal
    count, the summed outer products of the sampled causal effect pairs (for the
    Sigma update) and the (co)heritability quadratic forms ``beta_t' R beta_u``.
    ``rbsum1/2`` accumulate the Rao-Blackwellised effects ``P(causal)*post_mean``.
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

    log_p_odds = np.log(1.0 - p) - np.log(p)     # log P(null)/P(causal) prior
    detS = s11 * s22 - s12 * s12
    Si11 = s22 / detS
    Si22 = s11 / detS
    Si12 = -s12 / detS

    nb = 0
    C11 = 0.0
    C12 = 0.0
    C22 = 0.0
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
        detE = E11 * E22 - E12 * E12
        Ei11 = E22 / detE
        Ei22 = E11 / detE
        Ei12 = -E12 / detE

        # Bayes factor causal vs null: d ~ N(0, E+Sigma) vs N(0, E).
        qB = Ei11 * d1 * d1 + 2.0 * Ei12 * d1 * d2 + Ei22 * d2 * d2
        A11 = E11 + s11
        A12 = E12 + s12
        A22 = E22 + s22
        detA = A11 * A22 - A12 * A12
        qA = (A22 * d1 * d1 - 2.0 * A12 * d1 * d2 + A11 * d2 * d2) / detA
        log_bf = -0.5 * (np.log(detA) - np.log(detE)) - 0.5 * (qA - qB)
        postp = _stable_postp(log_p_odds - log_bf)

        # posterior (given causal): precision P = E^-1 + Sigma^-1, mean V E^-1 d.
        P11 = Ei11 + Si11
        P12 = Ei12 + Si12
        P22 = Ei22 + Si22
        detP = P11 * P22 - P12 * P12
        V11 = P22 / detP
        V22 = P11 / detP
        V12 = -P12 / detP
        g1 = Ei11 * d1 + Ei12 * d2
        g2 = Ei12 * d1 + Ei22 * d2
        m1 = V11 * g1 + V12 * g2
        m2 = V12 * g1 + V22 * g2

        rbsum1[j] += postp * m1
        rbsum2[j] += postp * m2

        if unif[j] < postp:                       # sample N(m, V) via 2x2 chol
            L11 = np.sqrt(V11)
            L21 = V12 / L11
            t = V22 - L21 * L21
            L22 = np.sqrt(t) if t > 0.0 else 0.0
            new1 = m1 + L11 * z1[j]
            new2 = m2 + L21 * z1[j] + L22 * z2[j]
            nb += 1
            C11 += new1 * new1
            C12 += new1 * new2
            C22 += new2 * new2
        else:
            new1 = 0.0
            new2 = 0.0

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
    return nb, C11, C12, C22, gv11, gv12, gv22


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
                                  h2_bounds=(1e-4, 1.0), seed=None):
    """Genome-wide (streaming) bivariate LDpred2-auto.

    ``blocks`` is the ``[(R, idx), ...]`` list (contiguous ``idx`` tiling
    ``0..m-1``) used elsewhere; the two traits' summary statistics share it. The
    effect sweeps run one block at a time while ``p`` and ``Sigma`` are pooled
    globally, so the genome-wide LD is never materialised.
    """
    bh1 = np.ascontiguousarray(beta_hat1, dtype=np.float64)
    bh2 = np.ascontiguousarray(beta_hat2, dtype=np.float64)
    m = bh1.shape[0]
    if bh2.shape[0] != m:
        raise ValueError("beta_hat1 and beta_hat2 must have the same length")
    n1 = _as_n_vector(n_eff1, m)
    n2 = _as_n_vector(n_eff2, m)

    fblocks = []
    for R, idx in blocks:
        idx = np.asarray(idx)
        if idx.shape[0] > 1 and not np.array_equal(idx, np.arange(idx[0], idx[0] + idx.shape[0])):
            raise ValueError("blocks must use contiguous indices")
        fblocks.append((np.ascontiguousarray(R, dtype=np.float32),
                        int(idx[0]), int(idx.shape[0])))
    covered = sum(k for _, _, k in fblocks)
    if covered != m:
        raise ValueError("blocks must tile 0..m-1 exactly once")

    lo, hi = h2_bounds
    M = float(m)
    rng = np.random.default_rng(seed)
    curr1 = np.zeros(m); curr2 = np.zeros(m)
    rb1 = np.zeros(m); rb2 = np.zeros(m)
    avg1 = np.zeros(m); avg2 = np.zeros(m)
    count = 0
    gv_acc = np.zeros(3)

    p = float(p_init)
    s11 = s22 = float(h2_init) / max(p_init * M, 1.0)
    s12 = float(rg_init) * s11

    for it in range(burn_in + num_iter):
        resync = (it % 100 == 0)
        unif = rng.random(m)
        z1 = rng.standard_normal(m)
        z2 = rng.standard_normal(m)
        rbs1 = np.zeros(m); rbs2 = np.zeros(m)
        nb = 0
        C11 = C12 = C22 = 0.0
        gv11 = gv12 = gv22 = 0.0
        for R, start, k in fblocks:
            sl = slice(start, start + k)
            nbk, c11, c12, c22, g11, g12, g22 = _bivar_one_sweep_jit(
                R, bh1[sl], bh2[sl], n1[sl], n2[sl], curr1[sl], curr2[sl],
                rb1[sl], rb2[sl], rbs1[sl], rbs2[sl], unif[sl], z1[sl], z2[sl],
                float(p), float(s11), float(s12), float(s22),
                float(cross_corr), resync)
            nb += nbk
            C11 += c11; C12 += c12; C22 += c22
            gv11 += g11; gv12 += g12; gv22 += g22

        # --- global hyper-parameter updates ---
        p = float(rng.beta(1.0 + nb, 1.0 + m - nb))
        if nb > 0:
            s11 = C11 / nb; s12 = C12 / nb; s22 = C22 / nb
        cap = hi / max(nb, 1)                      # bound implied per-trait h2
        s11 = min(max(s11, 1e-12), cap)
        s22 = min(max(s22, 1e-12), cap)
        mab = 0.999 * np.sqrt(s11 * s22)           # keep Sigma positive-definite
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
                           h2=(float(h2_1), float(h2_2)), rg=rg, p=float(p),
                           sigma=np.array([[s11, s12], [s12, s22]]))


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
