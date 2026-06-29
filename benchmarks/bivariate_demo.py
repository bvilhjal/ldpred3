"""Bivariate LDpred3-auto across two-trait architectures, with realistic LD.

The GWAS is generated from the true population (coalescent) LD but fitted with an
LD matrix estimated from a finite reference panel (Nref) -- the mismatch that
dominates real-world error. A weak trait 2 (low N) is fit on its own (univariate
auto) and jointly with a well-powered trait 1; the joint model should help when
the traits share causal structure and/or are correlated, and do no harm when
their causal variants are disjoint. Needs ``ld_library.npz`` in the cwd.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ldpred3_auto_bivariate_blocks, ldpred3_by_blocks

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
N1, N2 = 100000, 2000          # trait 2 genuinely under-powered (where joint helps)
PCAUSAL = 0.1                  # polygenic
NREF = 2000
SHRINK = 0.05
REPS = 6

rng0 = np.random.default_rng(0)
pop, chol_pop, ref, idxs = [], [], [], []
for b in range(NB):
    Rp = libR[b].copy()
    cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T
    Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    ref.append((Rr.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chol_pop.append(cp); idxs.append(np.arange(b * K, (b + 1) * K))


def gv(a, b):
    return sum(a[ix] @ (pop[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def scale(b, h2):
    g = gv(b, b)
    return b * np.sqrt(h2 / g) if g > 0 else b


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i][0].astype(float) @ beta[ix] + \
            (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


def r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def shared(rg):
    def f(rng):
        c = rng.random(M) < PCAUSAL
        L = np.linalg.cholesky([[1, rg], [rg, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
        b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
        return scale(b1, 0.5), scale(b2, 0.5)
    return f


def disjoint(rng):
    c1 = rng.random(M) < PCAUSAL; c2 = rng.random(M) < PCAUSAL
    b1 = np.zeros(M); b2 = np.zeros(M)
    b1[c1] = rng.standard_normal(c1.sum()); b2[c2] = rng.standard_normal(c2.sum())
    return scale(b1, 0.5), scale(b2, 0.5)


def partial(frac):
    def f(rng):
        c1 = rng.random(M) < PCAUSAL
        sh = c1 & (rng.random(M) < frac); pv = (~c1) & (rng.random(M) < PCAUSAL)
        b1 = np.zeros(M); b2 = np.zeros(M)
        b1[c1] = rng.standard_normal(c1.sum()); c2 = sh | pv; b2[c2] = rng.standard_normal(c2.sum())
        return scale(b1, 0.5), scale(b2, 0.5)
    return f


CASES = [("shared, rg=0.0", shared(0.0)), ("shared, rg=0.3", shared(0.3)),
         ("shared, rg=0.6", shared(0.6)), ("shared, rg=0.9", shared(0.9)),
         ("disjoint causal", disjoint)]

t0 = time.time()
print(f"trait2 genetic R2, reference-panel LD (Nref={NREF}, coalescent); "
      f"N1={N1}, N2={N2}, h2=0.5, m={M}, {REPS} reps\n")
print(f"{'architecture':>22} | {'alone':>6} | {'joint':>6} | {'gain':>6} | {'rg_est':>6}")
print("-" * 60)
for label, simfn in CASES:
    solo, joint, rge = [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(300 + rep)
        b1, b2 = simfn(rng)
        bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
        res = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2,
                                            burn_in=150, num_iter=200, seed=rep)
        s = ldpred3_by_blocks(ref, bh2, np.full(M, float(N2)), method="auto",
                              burn_in=150, num_iter=200, seed=rep)
        joint.append(r2(res.beta2_est, b2)); solo.append(r2(s, b2)); rge.append(res.rg)
    a, j = np.mean(solo), np.mean(joint)
    print(f"{label:>22} | {a:>6.3f} | {j:>6.3f} | {j-a:>+6.3f} | {np.mean(rge):>+6.2f}")
print(f"\n({time.time()-t0:.0f}s)")
