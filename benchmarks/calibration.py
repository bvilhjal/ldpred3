"""Credible/confidence-interval calibration (coverage) for the inference methods.

For many replicates we form each method's nominal 95% interval and count how
often it contains the known truth. Well-calibrated intervals cover ~95% of the
time. We do this under two LD conditions:

  - clean : fit with the true population LD (no mismatch)
  - ref   : fit with an LD estimated from a finite reference panel (realistic)

The expectation is that the point-estimate bias from LD mismatch makes the tight
intervals *under-cover* in the ``ref`` condition. Needs ``ld_library.npz``.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ld_scores, ldsc_h2, ldsc_rg, ldpred2_auto_infer

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K, NB = 500, 6
M = NB * K
N1, N2 = 50000, 30000
NREF = 2000
SHRINK = 0.05
H2, P, RG = 0.5, 0.01, 0.5
REPS = 40

idxs = [np.arange(b * K, (b + 1) * K) for b in range(NB)]
rng0 = np.random.default_rng(0)
pop, chol_pop, ref = [], [], []
for b in range(NB):
    Rp = libR[b].copy(); cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T; Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), idxs[b])); chol_pop.append(cp)
    ref.append((Rr.astype(np.float32), idxs[b]))


def dense_of(blocks):
    d = np.zeros((M, M), dtype=np.float32)
    for R, ix in blocks:
        d[np.ix_(ix, ix)] = R
    return d


def gv(a, b):
    return sum(a[ix] @ (pop[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def make_beta(rng):
    c = rng.random(M) < P
    beta = np.zeros(M); beta[c] = rng.standard_normal(c.sum())
    return beta * np.sqrt(H2 / gv(beta, beta))


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i][0].astype(float) @ beta[ix] + (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


def sim_pair(rng):
    c = rng.random(M) < P
    L = np.linalg.cholesky([[1, RG], [RG, 1]]); raw = L @ rng.standard_normal((2, c.sum()))
    b1 = np.zeros(M); b2 = np.zeros(M); b1[c] = raw[0]; b2[c] = raw[1]
    return b1 * np.sqrt(H2 / gv(b1, b1)), b2 * np.sqrt(H2 / gv(b2, b2))


conds = {"clean": (pop, dense_of(pop), ld_scores(pop)),
         "ref": (ref, dense_of(ref), ld_scores(ref, n_ref=NREF))}

# warm up the infer kernel
_ = ldpred2_auto_infer(dense_of(ref), sumstats(make_beta(np.random.default_rng(9)), N1,
                       np.random.default_rng(9)), N1, n_chains=4, burn_in=30,
                       num_iter=30, seed=0)

cover = {c: {k: 0 for k in ("infer_h2", "infer_p", "ldsc_h2", "ldsc_rg")} for c in conds}
t0 = time.time()
for rep in range(REPS):
    rng = np.random.default_rng(1000 + rep)
    beta = make_beta(rng); bh = sumstats(beta, N1, rng)
    b1, b2 = sim_pair(rng); bh1 = sumstats(b1, N1, rng); bh2 = sumstats(b2, N2, rng)
    for cname, (blocks, dense, ell) in conds.items():
        ir = ldpred2_auto_infer(dense, bh, N1, n_chains=8, burn_in=100,
                                num_iter=120, seed=rep)
        cover[cname]["infer_h2"] += ir.h2_ci[0] <= H2 <= ir.h2_ci[1]
        cover[cname]["infer_p"] += ir.p_ci[0] <= P <= ir.p_ci[1]
        hr = ldsc_h2(N1 * bh ** 2, ell, N1, n_blocks=60)
        cover[cname]["ldsc_h2"] += hr.h2_ci[0] <= H2 <= hr.h2_ci[1]
        gr = ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=60)
        cover[cname]["ldsc_rg"] += gr.rg_ci[0] <= RG <= gr.rg_ci[1]

print(f"95% interval coverage over {REPS} reps (m={M}); target 0.95\n")
print(f"{'method (truth)':>22} | {'clean LD':>9} | {'ref-panel LD':>12}")
print("-" * 50)
labels = {"infer_h2": "LDpred2 h2 (0.50)", "infer_p": "LDpred2 p (0.01)",
          "ldsc_h2": "LDSC h2 (0.50)", "ldsc_rg": "LDSC rg (0.50)"}
for k, lab in labels.items():
    print(f"{lab:>22} | {cover['clean'][k] / REPS:>9.2f} | "
          f"{cover['ref'][k] / REPS:>12.2f}")
print(f"\n({time.time()-t0:.0f}s)")
