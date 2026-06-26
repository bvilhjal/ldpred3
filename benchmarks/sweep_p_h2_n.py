"""PRS accuracy across polygenicity, heritability and sample size.

Sweeps each of (p, h2, N) from a baseline, holding the other two fixed, and
reports the genetic R2 of the PRS for marginal / inf / auto, under realistic
reference-panel LD (GWAS generated from the true coalescent LD, fitted with an LD
estimated from a finite reference panel). `inf` is given the true h2 (oracle);
`auto` self-tunes. Needs ``ld_library.npz`` in the cwd.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from pyldpred2 import ldpred2_by_blocks

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K, NB = 500, 16
M = NB * K
NREF = 2000
SHRINK = 0.05
REPS = 5
P0, H0, N0 = 0.01, 0.5, 50000          # baseline

idxs = [np.arange(b * K, (b + 1) * K) for b in range(NB)]
rng0 = np.random.default_rng(0)
pop, chol_pop, ref = [], [], []
for b in range(NB):
    Rp = libR[b].copy(); cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T; Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), idxs[b])); chol_pop.append(cp)
    ref.append((Rr.astype(np.float32), idxs[b]))


def gv(a, b):
    return sum(a[ix] @ (pop[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))


def make_beta(p, h2, rng):
    beta = np.zeros(M)
    c = np.ones(M, bool) if p >= 1.0 else (rng.random(M) < p)
    if not c.any():
        c[rng.integers(M)] = True
    beta[c] = rng.standard_normal(c.sum())
    return beta * np.sqrt(h2 / gv(beta, beta))


def sumstats(beta, n, rng):
    bh = np.empty(M)
    for i, ix in enumerate(idxs):
        bh[ix] = pop[i][0].astype(float) @ beta[ix] + (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(n)
    return bh


def r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def fit(method, bh, n, h2):
    if method == "marginal":
        return bh
    if method == "inf":
        return ldpred2_by_blocks(ref, bh, np.full(M, float(n)), method="inf", h2=h2)
    return ldpred2_by_blocks(ref, bh, np.full(M, float(n)), method="auto",
                             burn_in=120, num_iter=150, seed=0)


METHODS = ["marginal", "inf", "auto"]


def cell(p, h2, n):
    acc = {m: [] for m in METHODS}
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        beta = make_beta(p, h2, rng); bh = sumstats(beta, n, rng)
        for m in METHODS:
            acc[m].append(r2(fit(m, bh, n, h2), beta))
    return [np.mean(acc[m]) for m in METHODS]


t0 = time.time()
print(f"PRS genetic R2, reference-panel LD (Nref={NREF}, coalescent), m={M}, "
      f"{REPS} reps. Baseline p={P0}, h2={H0}, N={N0}\n")
hdr = f"{'value':>10} | " + " ".join(f"{m:>8}" for m in METHODS)


def sweep(name, values, fn):
    print(f"== sweep {name} ==")
    print(hdr); print("-" * len(hdr))
    for v in values:
        means = fn(v)
        label = f"{v:g}"
        print(f"{label:>10} | " + " ".join(f"{x:>8.3f}" for x in means))
    print()


sweep("N (p=%.3g, h2=%.2g)" % (P0, H0), [10000, 50000, 200000],
      lambda n: cell(P0, H0, n))
sweep("h2 (p=%.3g, N=%d)" % (P0, N0), [0.1, 0.3, 0.5, 0.8],
      lambda h: cell(P0, h, N0))
sweep("p (h2=%.2g, N=%d)" % (H0, N0), [0.001, 0.01, 0.1, 1.0],
      lambda p: cell(p, H0, N0))
print(f"({time.time()-t0:.0f}s)")
