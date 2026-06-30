"""Decompose why `annot` underperforms `auto` at low power + large m.

Compares, on realistic coalescent LD, the genetic R2 AND the learned
hyper-parameters (h2, effective p) of:
  - auto                : plain global-hyper sampler (Beta-posterior p)
  - annot_zero          : annotation learner with an ALL-ZERO annotation
                          (= "auto via the EB learner"; isolates learner overhead)
  - annot_info          : with the real informative functional annotation
  - annot_info_iter     : more iterations (burn 200 / iter 500)
  - annot_info_ridge.5  : weaker ridge (0.5)
  - annot_info_theta1   : update theta every sweep
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/iprs")
from ldpred3 import ldpred3_by_blocks, ldpred3_auto_annot_blocks

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K = 500
NB = 40                       # m = 20000 (faster to iterate; effect is m-dependent)
M = NB * K
H2 = 0.5
REPS = 3
N = 10000

blocks, chols, idxs = [], [], []
for b in range(NB):
    R = libR[b % libR.shape[0]].copy()
    blocks.append((R.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chols.append(np.linalg.cholesky(R + 1e-4 * np.eye(K)))
    idxs.append(np.arange(b * K, (b + 1) * K))

rng0 = np.random.default_rng(0)
func = (rng0.random(M) < 0.2).astype(float)
A_info = func[:, None]
A_zero = np.zeros((M, 1))


def make_beta(model, rng):
    beta = np.zeros(M)
    if model == "sparse":
        c = rng.random(M) < 0.01; beta[c] = rng.normal(0, 1, c.sum())
    elif model == "annot_enriched":
        base = np.where(func > 0, 10.0, 1.0)
        c = rng.random(M) < np.clip(base / base.sum() * (0.02 * M), 0, 1)
        beta[c] = rng.normal(0, 1, c.sum())
    gv = sum(beta[ix] @ (blocks[b][0].astype(float) @ beta[ix])
             for b, ix in enumerate(idxs))
    if gv > 0:
        beta *= np.sqrt(H2 / gv)
    return beta


def sumstats(beta, rng):
    bhat = np.empty(M)
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        bhat[ix] = R @ beta[ix] + (chols[b] @ rng.standard_normal(K)) / np.sqrt(N)
    return bhat


def genetic_r2(b_est, beta):
    num = den1 = den2 = 0.0
    for b, ix in enumerate(idxs):
        R = blocks[b][0].astype(float)
        Rb = R @ beta[ix]
        num += b_est[ix] @ Rb
        den1 += b_est[ix] @ (R @ b_est[ix])
        den2 += beta[ix] @ Rb
    return float(num * num / (den1 * den2)) if den1 > 0 and den2 > 0 else 0.0


def eff_p(A, theta):
    return float(np.mean(1.0 / (1.0 + np.exp(-(A_with_int(A) @ theta)))))


def A_with_int(A):
    return np.column_stack([np.ones(A.shape[0]), A])


n = np.full(M, float(N))
MODELS = ["sparse", "annot_enriched"]
TRUE = {"sparse": (0.01, H2), "annot_enriched": (0.02, H2)}

t0 = time.time()
for model in MODELS:
    print(f"\n=== {model}  (true p≈{TRUE[model][0]}, h2={H2}, N={N}, m={M}) ===")
    print(f"{'config':>18} | {'gen R2':>7} | {'h2_est':>7} | {'eff_p':>8} | theta")
    print("-" * 75)
    acc = {}
    for rep in range(REPS):
        rng = np.random.default_rng(1000 + rep)
        beta = make_beta(model, rng)
        bhat = sumstats(beta, rng)

        # auto baseline
        be = ldpred3_by_blocks(blocks, bhat, n, method="auto",
                               burn_in=80, num_iter=200, seed=1)
        acc.setdefault("auto", []).append((genetic_r2(be, beta), np.nan, np.nan, None))

        def run_annot(tag, A, **kw):
            r = ldpred3_auto_annot_blocks(blocks, bhat, n, A, seed=1, **kw)
            acc.setdefault(tag, []).append(
                (genetic_r2(r.beta_est, beta), r.h2_est, eff_p(A, r.theta),
                 np.round(r.theta, 2)))

        run_annot("annot_zero", A_zero, burn_in=80, num_iter=200)
        run_annot("annot_info", A_info, burn_in=80, num_iter=200)
        run_annot("annot_info_iter", A_info, burn_in=200, num_iter=500)
        run_annot("annot_info_ridge.5", A_info, burn_in=80, num_iter=200, ridge=0.5)
        run_annot("annot_info_theta1", A_info, burn_in=80, num_iter=200, theta_every=1)

    for tag, vals in acc.items():
        r2 = np.mean([v[0] for v in vals])
        h2 = np.nanmean([v[1] for v in vals])
        ep = np.nanmean([v[2] for v in vals])
        th = vals[-1][3]
        h2s = f"{h2:7.3f}" if not np.isnan(h2) else "    -  "
        eps = f"{ep:8.4f}" if not np.isnan(ep) else "    -   "
        print(f"{tag:>18} | {r2:7.3f} | {h2s} | {eps} | {th}")

print(f"\n({time.time()-t0:.0f}s)")
