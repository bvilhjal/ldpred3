"""Bivariate LDpred2-auto across two-trait architectures.

Fits a weak trait 2 (low N) on its own (univariate auto) and jointly with a
well-powered trait 1, reporting the genetic R2 of the trait-2 PRS and the
recovered rg. The joint model should help when the traits share causal structure
and/or are genetically correlated, and do no harm when their causal variants are
disjoint. Self-contained (heterogeneous AR(1) LD).
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ldpred2_auto_bivariate_blocks, ldpred2_by_blocks

K, NB = 200, 12
M = NB * K
N1, N2 = 100000, 3000
REPS = 6

rng0 = np.random.default_rng(2)
blocks, chols, idxs = [], [], []
for b in range(NB):
    rho = rng0.uniform(0.0, 0.8)
    d = np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    R = (rho ** d).astype(np.float64)
    blocks.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(np.linalg.cholesky(R + 1e-6 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))


def gv(a, b):
    return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def scale(b, h2):
    g = gv(b, b)
    return b * np.sqrt(h2 / g) if g > 0 else b


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = blocks[i][0].astype(float) @ beta[ix] + \
            (chols[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


def r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def shared(rg):
    def f(rng):
        c = rng.random(M) < 0.05
        L = np.linalg.cholesky([[1, rg], [rg, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
        b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
        return scale(b1, 0.5), scale(b2, 0.5)
    return f


def disjoint(rng):
    c1 = rng.random(M) < 0.05; c2 = rng.random(M) < 0.05
    b1 = np.zeros(M); b2 = np.zeros(M)
    b1[c1] = rng.standard_normal(c1.sum()); b2[c2] = rng.standard_normal(c2.sum())
    return scale(b1, 0.5), scale(b2, 0.5)


def partial(frac):
    def f(rng):
        c1 = rng.random(M) < 0.05
        sh = c1 & (rng.random(M) < frac); pv = (~c1) & (rng.random(M) < 0.05)
        b1 = np.zeros(M); b2 = np.zeros(M)
        b1[c1] = rng.standard_normal(c1.sum()); c2 = sh | pv; b2[c2] = rng.standard_normal(c2.sum())
        return scale(b1, 0.5), scale(b2, 0.5)
    return f


CASES = [("shared, rg=0.0", shared(0.0)), ("shared, rg=0.3", shared(0.3)),
         ("shared, rg=0.6", shared(0.6)), ("shared, rg=0.9", shared(0.9)),
         ("disjoint causal", disjoint), ("partial overlap 50%", partial(0.5))]

t0 = time.time()
print(f"trait2 genetic R2 (N1={N1}, N2={N2}, h2=0.5, m={M}, {REPS} reps)\n")
print(f"{'architecture':>22} | {'alone':>6} | {'joint':>6} | {'gain':>6} | {'rg_est':>6}")
print("-" * 60)
for label, simfn in CASES:
    solo, joint, rge = [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(300 + rep)
        b1, b2 = simfn(rng)
        bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
        res = ldpred2_auto_bivariate_blocks(blocks, bh1, bh2, N1, N2,
                                            burn_in=150, num_iter=200, seed=rep)
        s = ldpred2_by_blocks(blocks, bh2, np.full(M, float(N2)), method="auto",
                              burn_in=150, num_iter=200, seed=rep)
        joint.append(r2(res.beta2_est, b2)); solo.append(r2(s, b2)); rge.append(res.rg)
    a, j = np.mean(solo), np.mean(joint)
    print(f"{label:>22} | {a:>6.3f} | {j:>6.3f} | {j-a:>+6.3f} | {np.mean(rge):>+6.2f}")
print(f"\n({time.time()-t0:.0f}s)")
