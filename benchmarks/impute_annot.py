"""Does LD imputation + functional annotations help recover untyped causals?

The hypothesis (the user's): an imputed marginal statistic carries no new
information, BUT in conjunction with a functional annotation it lets the
annotation-informed prior place effect onto an imputed *functional* variant that
the typed tags only smear over. This benchmark tests it.

Setup (realistic coalescent LD, ld_library): causal effects are concentrated on
an annotated ("functional") subset; the functional causal variants are then
**dropped from the GWAS** (untyped), keeping their tags. Four pipelines:

  drop  / auto   -- fit auto on the typed subset only (untyped effects = 0)
  drop  / annot  -- fit annot on the typed subset
  impute/ auto   -- impute the untyped variants, fit auto on the union
  impute/ annot  -- impute the untyped variants, fit annot on the union   <-- proposed

reported under matched LD (fit = population LD) and mismatched LD (fit =
reference panel), for two outcomes:

  R2     -- genetic R2 of the PRS vs the true genetic value (prediction)
  loc    -- localisation rate: fraction of the *untyped functional causals* that
            end up the top |effect| variant in their LD neighbourhood (attribution)

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/impute_annot.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3 import (ldpred3_by_blocks, ldpred3_auto_annot_blocks,
                     impute_sumstats_blocks)

LIB = np.load("ld_library.npz")
libR = LIB["R"].astype(np.float64)
K, NB = 500, 12
M = NB * K
N = 50000
NREF = 2000
SHRINK = 0.05
H2 = 0.5
FUNC_FRAC = 0.20            # fraction of variants that are "functional" (annotated)
P_CAUSAL = 0.01            # overall causal fraction
ENRICH = 12.0             # causal odds multiplier in functional variants
REPS = 4

rng0 = np.random.default_rng(0)
pop, chol_pop, ref, idxs = [], [], [], []
for b in range(NB):
    Rp = libR[b % libR.shape[0]].copy()
    cp = np.linalg.cholesky(Rp + 1e-4 * np.eye(K))
    Z = rng0.standard_normal((NREF, K)) @ cp.T
    Z = (Z - Z.mean(0)) / Z.std(0)
    Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
    pop.append((Rp.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    ref.append((Rr.astype(np.float32), np.arange(b * K, (b + 1) * K)))
    chol_pop.append(cp)
    idxs.append(np.arange(b * K, (b + 1) * K))

func = (rng0.random(M) < FUNC_FRAC)          # the functional annotation (fixed)
A = func.astype(float)[:, None]
Dpop = [pop[b][0].astype(np.float64) for b in range(NB)]


def gv(a, b):
    return sum(a[ix] @ (Dpop[i] @ b[ix]) for i, ix in enumerate(idxs))


def genetic_r2(be, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def simulate(rng):
    base = np.where(func, ENRICH, 1.0)
    pc = np.clip(base / base.sum() * (P_CAUSAL * M), 0, 1)
    causal = rng.random(M) < pc
    beta = np.zeros(M); beta[causal] = rng.normal(0, 1, int(causal.sum()))
    beta *= np.sqrt(H2 / gv(beta, beta))
    bhat = np.empty(M)
    for i, ix in enumerate(idxs):
        bhat[ix] = Dpop[i] @ beta[ix] + (chol_pop[i] @ rng.standard_normal(K)) / np.sqrt(N)
    # untyped = the functional causal variants (what we want to recover)
    untyped = causal & func
    typed = ~untyped
    return beta, bhat, typed, np.flatnonzero(untyped)


def subset_blocks(blocks, keep):
    out, orig, off = [], [], 0
    for R, idx in blocks:
        loc = keep[idx]
        k = int(loc.sum())
        if k:
            out.append((np.asarray(R)[np.ix_(loc, loc)], np.arange(off, off + k)))
            orig.append(np.asarray(idx)[loc])
            off += k
    return out, np.concatenate(orig) if orig else np.array([], int)


def localisation(be, beta_causal_untyped):
    """Fraction of untyped functional causals that are the top |effect| in their
    LD neighbourhood (|r|>=0.5) -- did the effect land on the (functional) causal?"""
    if beta_causal_untyped.size == 0:
        return float("nan")
    hits = 0
    for c in beta_causal_untyped:
        b = c // K
        R = Dpop[b]; loc = c - b * K
        nbr = idxs[b][np.abs(R[loc]) >= 0.5]
        if np.abs(be[c]) >= np.max(np.abs(be[nbr])) - 1e-12:
            hits += 1
    return hits / beta_causal_untyped.size


def fit(method, fit_blocks, bhat, n, A_in):
    if method == "auto":
        return ldpred3_by_blocks(fit_blocks, bhat, n, method="auto",
                                 burn_in=80, num_iter=150, seed=1)
    return ldpred3_auto_annot_blocks(fit_blocks, bhat, n, A_in, burn_in=80,
                                     num_iter=150, seed=1).beta_est


def run(pipeline, method, fit_lib, beta, bhat, typed, untyped_idx):
    n_full = np.full(M, float(N))
    if pipeline == "drop":
        sub, orig = subset_blocks(fit_lib, typed)
        be_sub = fit(method, sub, bhat[orig], n_full[orig], A[orig])
        be = np.zeros(M); be[orig] = be_sub
    else:                                       # impute the untyped, fit on the union
        imp = impute_sumstats_blocks(bhat, fit_lib, typed, N)
        be = fit(method, fit_lib, imp.beta_hat, imp.n_eff, A)
    return genetic_r2(be, beta), localisation(be, untyped_idx)


CASES = [("drop", "auto"), ("drop", "annot"), ("impute", "auto"), ("impute", "annot")]
t0 = time.time()
print(f"Imputation + annotations, coalescent LD, m={M}, N={N}, h2={H2}, "
      f"functional={FUNC_FRAC:.0%}, causals enriched {ENRICH:g}x in functional, "
      f"{REPS} reps\n")
for ld_label, fit_lib in (("matched LD (population)", pop),
                          ("mismatched LD (ref panel)", ref)):
    print(f"== {ld_label} ==")
    print(f"{'pipeline':>14} | {'genetic R2':>10} | {'localisation':>12}")
    print("-" * 44)
    for pipe, meth in CASES:
        r2s, locs = [], []
        for rep in range(REPS):
            rng = np.random.default_rng(700 + rep)
            beta, bhat, typed, untyped_idx = simulate(rng)
            r2, loc = run(pipe, meth, fit_lib, beta, bhat, typed, untyped_idx)
            r2s.append(r2); locs.append(loc)
        print(f"{pipe + '/' + meth:>14} | {np.mean(r2s):>10.3f} | "
              f"{np.nanmean(locs):>12.2f}")
    print()
print(f"(localisation = fraction of untyped functional causals that are the top "
      f"|effect| in their LD neighbourhood; drop pipelines score 0 -- the variant "
      f"is not in the model. {time.time()-t0:.0f}s)")
