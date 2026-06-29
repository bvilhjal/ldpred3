"""Robustness benchmarks (auto PRS): LD-reference quality and N misspecification.

Both fit LDpred3-auto on summary statistics generated from the true population
(coalescent) LD, and measure the genetic R2 of the resulting PRS plus the fitted
genetic variance (a heritability proxy). Needs ``ld_library.npz`` in the cwd.

  (A) LD-reference quality: vary the size Nref of the reference panel the LD is
      estimated from (the dominant real-world error). Nref=inf uses the true LD.
  (B) N misspecification: fit with a wrong sample size N_used = factor * N_true.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ldpred3_by_blocks

LIB = np.load("ld_library.npz"); libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
N_TRUE = 50000
H2, P = 0.5, 0.01
REPS = 5
SHRINK = 0.05

idxs = [np.arange(b * K, (b + 1) * K) for b in range(NB)]
pop, chol_pop = [], []
for b in range(NB):
    Rp = libR[b].copy()
    pop.append((Rp.astype(np.float32), idxs[b]))
    chol_pop.append(np.linalg.cholesky(Rp + 1e-4 * np.eye(K)))


def make_ref(nref, seed):
    rng = np.random.default_rng(seed)
    blocks = []
    for b in range(NB):
        if nref is None:
            blocks.append((pop[b][0], idxs[b]))
            continue
        Z = rng.standard_normal((nref, K)) @ chol_pop[b].T
        Z = (Z - Z.mean(0)) / Z.std(0)
        Rr = (1 - SHRINK) * ((Z.T @ Z) / nref) + SHRINK * np.eye(K)
        blocks.append((Rr.astype(np.float32), idxs[b]))
    return blocks


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


def r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


t0 = time.time()
print(f"Robustness (auto PRS), coalescent LD, m={M}, N_true={N_TRUE}, "
      f"h2={H2}, p={P}, {REPS} reps\n")

print("(A) LD-reference quality (correct N):")
print(f"{'Nref':>8} | {'pred R2':>8} | {'h2 proxy':>9}")
print("-" * 32)
for nref in (500, 1000, 2000, 5000, 10000, None):
    ref = make_ref(nref, seed=0)
    r2s, h2s = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        beta = make_beta(rng); bh = sumstats(beta, N_TRUE, rng)
        be = ldpred3_by_blocks(ref, bh, np.full(M, float(N_TRUE)), method="auto",
                               burn_in=120, num_iter=150, seed=rep)
        r2s.append(r2(be, beta)); h2s.append(gv(be, be))
    label = "inf (true)" if nref is None else str(nref)
    print(f"{label:>8} | {np.mean(r2s):>8.3f} | {np.mean(h2s):>9.3f}")

print("\n(B) N misspecification (Nref=2000, true N=%d, true h2=%.2f):" % (N_TRUE, H2))
print(f"{'N_used/N':>8} | {'pred R2':>8} | {'h2 proxy':>9}")
print("-" * 32)
ref = make_ref(2000, seed=0)
for factor in (0.7, 0.85, 1.0, 1.15, 1.3):
    r2s, h2s = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        beta = make_beta(rng); bh = sumstats(beta, N_TRUE, rng)
        be = ldpred3_by_blocks(ref, bh, np.full(M, factor * N_TRUE), method="auto",
                               burn_in=120, num_iter=150, seed=rep)
        r2s.append(r2(be, beta)); h2s.append(gv(be, be))
    print(f"{factor:>8.2f} | {np.mean(r2s):>8.3f} | {np.mean(h2s):>9.3f}")
print(f"\n({time.time()-t0:.0f}s)")
