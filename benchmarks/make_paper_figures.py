"""Generate a multi-page, publication-quality PDF of the key LDpred3 results.

One command -> ``benchmarks/figures.pdf`` with:

  p1  Running time & peak memory vs bigsnpr (1 core, realistic LD)
  p2  Accuracy by genetic architecture (N = 10k and 50k)
  p3  Inference evaluation: h² and polygenicity recovery vs truth (95% CIs)
  p4  Inference cross-checks: h² LDSC vs LDpred3-auto; predictive r² est-vs-realized
  p5  Bivariate analysis: genetic-correlation recovery; weak-trait prediction gain
  p6  DENTIST LD-consistency filter: accuracy recovery and error catch/false-drop
  p7  LD representation: sparse/banded storage-vs-accuracy; optimal block splitting
  p8  Performance: Numba JIT speed-up; multi-core scaling

Pages 4-5 need realistic (coalescent) LD and are skipped with a note unless
msprime is installed.

Pages 1-2 read the committed CSVs (``cores_1core_benchmark.csv`` /
``methods_arch_benchmark.csv``); they are skipped with a note if absent. Pages
3-5 compute their data from a self-contained AR(1) simulation and cache it to
``benchmarks/figdata_*.csv`` so re-runs are fast and the underlying numbers are a
reproducible artifact (delete the caches to recompute).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/make_paper_figures.py

Needs matplotlib (and numba for the performance page; that page degrades to a
note without it).
"""
import csv
import os
import sys
import time
import subprocess
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

from ldpred3.simulate import simulate_genotypes
from ldpred3.ld import compute_ld_blocks
from ldpred3.qc import dentist_outlier_mask
from ldpred3.infer import ldpred3_auto_infer
from ldpred3 import (ldpred3_by_blocks, sparsify_ld, optimal_ld_blocks,
                     ld_scores, ldsc_h2, ldsc_rg, ldpred3_auto_bivariate_blocks)
from ldpred3._numba import HAVE_NUMBA

try:  # realistic LD (coalescent) is needed for the LDSC comparisons
    import msprime  # noqa: F401
    from ldpred3.simulate import simulate_genotypes_coalescent
    HAVE_MSPRIME = True
except ImportError:  # pragma: no cover
    HAVE_MSPRIME = False

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
})
MCOLOR = {"inf": "#9467bd", "grid": "#1f77b4", "auto": "#2ca02c",
          "marginal": "#7f7f7f", "annot": "#d62728"}


# --------------------------------------------------------------------------- #
# Shared simulation helpers (self-contained AR(1) panel, no external LD lib).
# --------------------------------------------------------------------------- #
def simulate(nref, sizes, n_gwas, h2, p, rho, seed):
    """Return (blocks, gv, beta, beta_hat) for a simulated GWAS + estimated LD."""
    m = int(np.sum(sizes))
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, m)
    G, _ = simulate_genotypes(nref, list(sizes), maf, rho, rng)
    blocks = compute_ld_blocks(G, block_size=max(sizes))
    Rf = [(R.astype(float), idx) for R, idx in blocks]

    def gv(a, b):
        return sum(a[ix] @ (R @ b[ix]) for R, ix in Rf)

    causal = rng.random(m) < p
    beta = np.zeros(m); beta[causal] = rng.standard_normal(int(causal.sum()))
    beta *= np.sqrt(h2 / gv(beta, beta))
    beta_hat = np.empty(m)
    for R, ix in Rf:
        chol = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        beta_hat[ix] = R @ beta[ix] + (chol @ rng.standard_normal(len(ix))) / np.sqrt(n_gwas)
    return blocks, gv, beta, beta_hat, causal, rng


def geneticr2(be, gv, beta):
    num = gv(be, beta); den = gv(be, be) * gv(beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def cached(name, fields, compute):
    """Load benchmarks/figdata_<name>.csv if present, else compute + save it."""
    path = os.path.join(HERE, f"figdata_{name}.csv")
    if os.path.exists(path):
        with open(path) as fh:
            return list(csv.DictReader(fh))
    print(f"  computing {name} ...", flush=True)
    rows = compute()
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- #
# Data producers for the feature pages (cached to CSV).
# --------------------------------------------------------------------------- #
def data_dentist():
    NB, K, M, NREF, N, H2, P, RHO, NC, CZ, REPS = 20, 200, 4000, 10000, 10000, 0.5, 0.05, 0.9, 30, 8.0, 5

    def subset(blocks, keep):
        out, orig, off = [], [], 0
        for R, idx in blocks:
            loc = keep[idx]; k = int(loc.sum())
            if k:
                out.append((np.asarray(R)[np.ix_(loc, loc)], np.arange(off, off + k)))
                orig.append(np.asarray(idx)[loc]); off += k
        return out, np.concatenate(orig)

    def fit(blocks, bh, keep, gv, beta):
        n = np.full(M, float(N))
        if keep is None:
            be = ldpred3_by_blocks(blocks, bh, n, method="auto", burn_in=80, num_iter=150, seed=0)
        else:
            sub, ki = subset(blocks, keep)
            bes = ldpred3_by_blocks(sub, bh[ki], n[ki], method="auto", burn_in=80, num_iter=150, seed=0)
            be = np.zeros(M); be[ki] = bes
        return geneticr2(be, gv, beta)

    clean, noflt, dent, caught, fdrop = [], [], [], [], []
    for rep in range(REPS):
        blocks, gv, beta, bh, causal, rng = simulate(NREF, [K] * NB, N, H2, P, RHO, 1000 + rep)
        nc = np.flatnonzero(~causal); bad_idx = rng.choice(nc, NC, replace=False)
        bad = bh.copy(); bad[bad_idx] = rng.choice([-1.0, 1.0], NC) * CZ / np.sqrt(N)
        clean.append(fit(blocks, bh, None, gv, beta))
        noflt.append(fit(blocks, bad, None, gv, beta))
        keepb, _ = dentist_outlier_mask(blocks, bad * np.sqrt(N))
        dent.append(fit(blocks, bad, keepb, gv, beta))
        caught.append(int((~keepb)[bad_idx].sum()) / NC)
        keepc, _ = dentist_outlier_mask(blocks, bh * np.sqrt(N))
        fdrop.append(int((~keepc).sum()) / M)
    return [{"clean": np.mean(clean), "no_filter": np.mean(noflt), "dentist": np.mean(dent),
             "caught_frac": np.mean(caught), "falsedrop_frac": np.mean(fdrop)}]


def data_inference():
    """LDpred3-auto-infer recovery of h² and polygenicity p vs the truth.

    Sweeps the true value, runs multi-chain inference on a self-contained AR(1)
    panel, and records the posterior median + 95% CI. No validation cohort.
    """
    NB, K, M, NREF, N, RHO, REPS = 15, 200, 3000, 8000, 40000, 0.8, 3
    rows = []
    for h2 in (0.1, 0.2, 0.35, 0.5, 0.8):                  # h² sweep (fixed p=0.02)
        est, lo, hi = [], [], []
        for rep in range(REPS):
            blocks, _, _, bh, _, _ = simulate(NREF, [K] * NB, N, h2, 0.02, RHO, 200 + rep)
            r = ldpred3_auto_infer(blocks, bh, np.full(M, float(N)),
                                   n_chains=6, burn_in=150, num_iter=150, seed=rep)
            est.append(r.h2_est); lo.append(r.h2_ci[0]); hi.append(r.h2_ci[1])
        rows.append({"kind": "h2", "true": h2, "est": np.mean(est),
                     "lo": np.mean(lo), "hi": np.mean(hi)})
    for p in (0.005, 0.02, 0.1):                           # p sweep (fixed h²=0.5)
        est, lo, hi = [], [], []
        for rep in range(REPS):
            blocks, _, _, bh, _, _ = simulate(NREF, [K] * NB, N, 0.5, p, RHO, 300 + rep)
            r = ldpred3_auto_infer(blocks, bh, np.full(M, float(N)),
                                   n_chains=6, burn_in=150, num_iter=150, seed=rep)
            est.append(r.p_est); lo.append(r.p_ci[0]); hi.append(r.p_ci[1])
        rows.append({"kind": "p", "true": p, "est": np.mean(est),
                     "lo": np.mean(lo), "hi": np.mean(hi)})
    return rows


def coalescent_panel(seed, m=3000, k=500, n_ref=4000):
    """Realistic (coalescent/msprime) LD panel + sumstats / LD-score helpers.

    LDSC needs the LD-score distribution of *realistic* LD; the AR(1) panel used
    elsewhere is too smooth and makes the LDSC regression blow up. These pages
    therefore simulate a coalescent-with-recombination panel.
    """
    G, _ = simulate_genotypes_coalescent(n_ref, m, k, seed=seed)
    mm = G.shape[1]
    blocks = compute_ld_blocks(G, block_size=k)
    Rf = [(R.astype(float), idx) for R, idx in blocks]

    def gv(a, b):
        return sum(a[ix] @ (R @ b[ix]) for R, ix in Rf)

    def ss(beta, n, rng):
        bh = np.empty(mm)
        for R, ix in Rf:
            chol = np.linalg.cholesky(R + 1e-4 * np.eye(len(ix)))
            bh[ix] = R @ beta[ix] + (chol @ rng.standard_normal(len(ix))) / np.sqrt(n)
        return bh

    return blocks, mm, gv, ss, ld_scores(blocks, n_ref=n_ref)


def data_infer_ldsc():
    """h² recovery (LDSC vs LDpred3-auto) and predictive-r² est-vs-realized.

    Realistic coalescent LD. ``realized`` predictive r² = genetic-R²(PRS, truth)
    × h² (the out-of-sample phenotype r² the auto estimator targets).
    """
    blocks, m, gv, ss, ell = coalescent_panel(11)
    N, REPS = 50000, 5
    rows = []
    for h2 in (0.1, 0.3, 0.5, 0.8):
        hl, hi, est, lo, hi_ci, real = [], [], [], [], [], []
        for rep in range(REPS):
            rng = np.random.default_rng(400 + rep)
            c = rng.random(m) < 0.02
            beta = np.zeros(m); beta[c] = rng.standard_normal(int(c.sum()))
            beta *= np.sqrt(h2 / gv(beta, beta))
            bh = ss(beta, N, rng)
            hl.append(ldsc_h2((bh * np.sqrt(N)) ** 2, ell, N, n_blocks=60).h2)
            r = ldpred3_auto_infer(blocks, bh, np.full(m, float(N)),
                                   n_chains=5, burn_in=120, num_iter=120, seed=rep)
            hi.append(r.h2_est)
            est.append(r.r2_est); lo.append(r.r2_ci[0]); hi_ci.append(r.r2_ci[1])
            real.append(geneticr2(r.beta_est, gv, beta) * h2)
        rows.append({"true_h2": h2, "ldsc": np.mean(hl), "ldsc_sd": np.std(hl),
                     "auto": np.mean(hi), "auto_sd": np.std(hi),
                     "r2_est": np.mean(est), "r2_lo": np.mean(lo),
                     "r2_hi": np.mean(hi_ci), "r2_real": np.mean(real)})
    return rows


def data_bivariate():
    """Genetic-correlation recovery (LDpred3 vs LDSC) and weak-trait gain.

    Realistic coalescent LD. (A) sweep true rg, recover it; (B) a weak secondary
    trait (small N2) scored alone vs boosted by a correlated well-powered trait.
    """
    blocks, m, gv, ss, ell = coalescent_panel(12)
    rows = []
    N1, N2, P, H2, REPS = 50000, 30000, 0.02, 0.5, 5
    for rg in (0.0, 0.3, 0.6, 0.9):
        bp, ld = [], []
        for rep in range(REPS):
            rng = np.random.default_rng(60 + rep)
            c = rng.random(m) < P
            L = np.linalg.cholesky([[1, rg], [rg, 1]])
            raw = L @ rng.standard_normal((2, int(c.sum())))
            b1 = np.zeros(m); b2 = np.zeros(m); b1[c] = raw[0]; b2[c] = raw[1]
            b1 *= np.sqrt(H2 / gv(b1, b1)); b2 *= np.sqrt(H2 / gv(b2, b2))
            bh1 = ss(b1, N1, rng); bh2 = ss(b2, N2, rng)
            bp.append(ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, N1, N2,
                                                    burn_in=100, num_iter=120, seed=rep).rg)
            ld.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=60).rg)
        rows.append({"kind": "rg", "x": rg, "m1": np.mean(bp), "s1": np.std(bp),
                     "m2": np.mean(ld), "s2": np.std(ld)})
    N1g, N2g, Pg, H1, H2g, REPSg = 100000, 3000, 0.05, 0.5, 0.3, 5
    for rg in (0.0, 0.3, 0.6, 0.9):
        uni, bi = [], []
        for rep in range(REPSg):
            rng = np.random.default_rng(70 + rep)
            c = rng.random(m) < Pg
            L = np.linalg.cholesky([[1, rg], [rg, 1]])
            raw = L @ rng.standard_normal((2, int(c.sum())))
            b1 = np.zeros(m); b2 = np.zeros(m); b1[c] = raw[0]; b2[c] = raw[1]
            b1 *= np.sqrt(H1 / gv(b1, b1)); b2 *= np.sqrt(H2g / gv(b2, b2))
            bh1 = ss(b1, N1g, rng); bh2 = ss(b2, N2g, rng)
            be2u = ldpred3_by_blocks(blocks, bh2, np.full(m, float(N2g)),
                                     method="auto", burn_in=80, num_iter=120, seed=rep)
            res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, N1g, N2g,
                                                burn_in=100, num_iter=120, seed=rep)
            uni.append(geneticr2(be2u, gv, b2)); bi.append(geneticr2(res.beta2_est, gv, b2))
        rows.append({"kind": "gain", "x": rg, "m1": np.mean(uni), "s1": np.std(uni),
                     "m2": np.mean(bi), "s2": np.std(bi)})
    return rows


def data_sparse():
    NB, K, M, NREF, N, H2, P, RHO, REPS = 8, 500, 4000, 5000, 20000, 0.5, 0.02, 0.85, 3
    configs = [("dense", None, None, 1.0), ("thr 1e-2", 1e-2, None, 1.0),
               ("thr 1e-3", 1e-3, None, 1.0), ("band 50", 1e-4, 50, 1.0),
               ("band 25\n+shrink", 1e-4, 25, 0.9)]
    n = np.full(M, float(N))
    b0, gv0, be0, bh0, _, _ = simulate(NREF, [K] * NB, N, H2, P, RHO, 0)
    ldpred3_by_blocks(b0, bh0, n, method="auto", global_hyper=False, burn_in=10, num_iter=10, seed=0)
    sp0 = [(sparsify_ld(R, threshold=1e-2), idx) for R, idx in b0]
    ldpred3_by_blocks(sp0, bh0, n, method="auto", global_hyper=False, burn_in=10, num_iter=10, seed=0)
    rows = []
    for label, thr, md, shr in configs:
        dens, tt, r2 = [], [], []
        for rep in range(REPS):
            blocks, gv, beta, bh, _, _ = simulate(NREF, [K] * NB, N, H2, P, RHO, 100 + rep)
            if thr is None:
                fb = blocks; dens.append(1.0)
            else:
                fb = [(sparsify_ld(R, threshold=thr, max_dist=md, shrink=shr), idx) for R, idx in blocks]
                dens.append(sum(b.nnz for b, _ in fb) / float(M * K))
            t = time.time()
            be = ldpred3_by_blocks(fb, bh, n, method="auto", global_hyper=False, burn_in=60, num_iter=120, seed=0)
            tt.append(time.time() - t); r2.append(geneticr2(be, gv, beta))
        rows.append({"config": label, "density": np.mean(dens), "fit_s": np.mean(tt), "r2": np.mean(r2)})
    return rows


def data_splitting():
    SIZES = [137, 211, 89, 256, 170, 137]; M = sum(SIZES)
    MAXS, MINS, NREF, N, H2, P, RHO, REPS = 250, 30, 4000, 20000, 0.5, 0.02, 0.9, 3

    def discarded(R, bounds):
        R2 = R * R; tot = np.triu(R2, 1).sum()
        within = sum(np.triu(R2[s:e, s:e], 1).sum() for s, e in bounds)
        return float(tot - within)

    def fit(R, bounds, bh, beta, n):
        blocks = [(R[s:e, s:e].astype(np.float32), np.arange(s, e)) for s, e in bounds]
        be = ldpred3_by_blocks(blocks, bh, n, method="auto", global_hyper=False, burn_in=60, num_iter=120, seed=0)
        num = be @ (R @ beta); den = (be @ (R @ be)) * (beta @ (R @ beta))
        return float(num * num / den) if den > 0 else 0.0

    n = np.full(M, float(N))
    agg = {"fixed": [], "optimal": []}
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        maf = rng.uniform(0.05, 0.5, M)
        G, _ = simulate_genotypes(NREF, SIZES, maf, RHO, rng)
        Gs = (G - G.mean(0)) / G.std(0); R = (Gs.T @ Gs) / NREF
        causal = rng.random(M) < P; beta = np.zeros(M); beta[causal] = rng.standard_normal(int(causal.sum()))
        beta *= np.sqrt(H2 / (beta @ (R @ beta)))
        chol = np.linalg.cholesky(R + 1e-6 * np.eye(M)); bh = R @ beta + (chol @ rng.standard_normal(M)) / np.sqrt(N)
        fb = [(s, min(s + MAXS, M)) for s in range(0, M, MAXS)]
        ob, _ = optimal_ld_blocks(R, max_size=MAXS, min_size=MINS, window=MAXS)
        for name, bb in (("fixed", fb), ("optimal", ob)):
            agg[name].append((len(bb), discarded(R, bb), sum((e - s) ** 2 for s, e in bb), fit(R, bb, bh, beta, n)))
    return [{"split": nm, "nblocks": np.mean([a[0] for a in agg[nm]]),
             "discarded": np.mean([a[1] for a in agg[nm]]), "storage": np.mean([a[2] for a in agg[nm]]),
             "r2": np.mean([a[3] for a in agg[nm]])} for nm in ("fixed", "optimal")]


def data_numba():
    code = (
        "import time,numpy as np,sys;sys.path.insert(0,%r)\n"
        "from ldpred3.simulate import simulate_genotypes\n"
        "from ldpred3.ld import compute_ld_blocks\n"
        "from ldpred3 import ldpred3_by_blocks\n"
        "NB,K=10,200;M=NB*K;N=20000\n"
        "rng=np.random.default_rng(0);maf=rng.uniform(.05,.5,M)\n"
        "G,_=simulate_genotypes(5000,[K]*NB,maf,.8,rng);bl=compute_ld_blocks(G,block_size=K)\n"
        "beta=np.zeros(M);c=rng.random(M)<.02;beta[c]=rng.standard_normal(int(c.sum()))\n"
        "bh=np.empty(M)\n"
        "for R,ix in [(R.astype(float),idx) for R,idx in bl]:\n"
        " ch=np.linalg.cholesky(R+1e-6*np.eye(len(ix)));bh[ix]=R@beta[ix]+(ch@rng.standard_normal(len(ix)))/np.sqrt(N)\n"
        "n=np.full(M,float(N))\n"
        "ldpred3_by_blocks(bl,bh,n,method='auto',burn_in=5,num_iter=5)\n"
        "t=time.time();ldpred3_by_blocks(bl,bh,n,method='auto',burn_in=60,num_iter=150);print(time.time()-t)\n"
    ) % os.path.dirname(HERE)

    def run(disable):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        if disable:
            env["NUMBA_DISABLE_JIT"] = "1"
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(out.stderr)
        return float(out.stdout.strip().splitlines()[-1])

    return [{"mode": "pure_python", "fit_s": run(True)},
            {"mode": "numba_jit", "fit_s": run(False)}]


def data_cores():
    from ldpred3.ldpred3 import _gibbs_blocks
    NB, K, M, NREF, N, RHO = 40, 500, 20000, 2000, 50000, 0.8
    rng = np.random.default_rng(0); maf = rng.uniform(0.05, 0.5, M)
    G, _ = simulate_genotypes(NREF, [K] * NB, maf, RHO, rng)
    blocks = compute_ld_blocks(G, block_size=K)
    beta = np.zeros(M); c = rng.random(M) < 0.01; beta[c] = rng.standard_normal(int(c.sum()))
    bh = np.empty(M)
    for R, ix in [(R.astype(float), idx) for R, idx in blocks]:
        ch = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        bh[ix] = R @ beta[ix] + (ch @ rng.standard_normal(len(ix))) / np.sqrt(N)
    n = np.full(M, float(N))
    common = dict(sparse=False, seed=0, estimate_hyper=True, h2_bounds=(1e-4, 1.0))
    rows, t1 = [], None
    for nc in (1, 2, 4):
        _gibbs_blocks(blocks, bh, n, 0.1, 0.1, burn_in=5, num_iter=5, ncores=nc, **common)
        t = time.time()
        _gibbs_blocks(blocks, bh, n, 0.1, 0.1, burn_in=100, num_iter=200, ncores=nc, **common)
        dt = time.time() - t; t1 = t1 or dt
        rows.append({"ncores": nc, "fit_s": dt, "speedup": t1 / dt})
    return rows


# --------------------------------------------------------------------------- #
# Pages.
# --------------------------------------------------------------------------- #
def note_page(pdf, title, msg):
    fig, ax = plt.subplots(figsize=(11, 6)); ax.axis("off")
    ax.set_title(title, fontsize=13)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#666")
    pdf.savefig(fig); plt.close(fig)


def page_bigsnpr(pdf):
    path = os.path.join(HERE, "cores_1core_benchmark.csv")
    if not os.path.exists(path):
        return note_page(pdf, "vs bigsnpr", "cores_1core_benchmark.csv not found.")
    rows = []
    with open(path) as fh:
        for tool, me, m, t, mem, r2 in csv.reader(fh):
            rows.append((tool, me, int(m), float(t), float(mem), float(r2)))

    def series(tool, me, col):
        xy = [(m / 1e6, (t, mem)[col]) for tl, mm, m, t, mem, _ in rows if tl == tool and mm == me]
        xy.sort(); return [a for a, _ in xy], [b for _, b in xy]

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4.8))
    for me in ("inf", "grid", "auto"):
        x, y = series("LDpred3", me, 0); ax_t.plot(x, y, "-o", color=MCOLOR[me], lw=2, ms=5, label=f"py {me}")
        x, y = series("bigsnpr", me, 0); ax_t.plot(x, y, "--s", color=MCOLOR[me], lw=1.5, ms=4, alpha=0.8, label=f"big {me}")
    ax_t.set(title="Running time, 1 core", xlabel="SNPs (millions)", ylabel="wall-clock (s)")
    ax_t.legend(fontsize=8, ncol=3)
    x, y = series("LDpred3", "auto", 1); ax_m.plot(x, y, "-o", color="#1f77b4", lw=2, ms=6, label="LDpred3")
    x, y = series("bigsnpr", "auto", 1); ax_m.plot(x, y, "-^", color="#2ca02c", lw=2, ms=6, label="bigsnpr")
    ax_m.set(title="Peak memory", xlabel="SNPs (millions)", ylabel="peak RSS (GB)"); ax_m.legend(fontsize=9)
    fig.suptitle("LDpred3 vs bigsnpr — realistic LD (coalescent), N=100k, h²=0.5", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_arch(pdf):
    path = os.path.join(HERE, "methods_arch_benchmark.csv")
    if not os.path.exists(path):
        return note_page(pdf, "Accuracy by architecture", "methods_arch_benchmark.csv not found.")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    methods = ["marginal", "inf", "grid", "auto", "annot"]
    models = ["infinitesimal", "sparse", "polygenic", "major_locus", "annot_enriched"]
    labels = ["infinit.", "sparse", "polygenic", "major\nlocus", "annot\nenriched"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, N in zip(axes, ("10000", "50000")):
        sub = {r["model"]: r for r in rows if r["N"] == N}
        x = np.arange(len(models)); w = 0.16
        for i, me in enumerate(methods):
            vals = [float(sub[m][me]) for m in models]
            ax.bar(x + (i - 2) * w, vals, w, color=MCOLOR[me], label=me)
        ax.set(title=f"N = {int(N):,}", xticks=x, ylim=(0.4, 1.0))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("genetic R²") if N == "10000" else None
    axes[0].legend(fontsize=8, ncol=5, loc="upper left")
    fig.suptitle("Accuracy by genetic architecture (realistic LD, h²=0.5)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_inference(pdf):
    rows = [{k: r[k] for k in r} for r in data_infer_rows]
    h2 = [r for r in rows if r["kind"] == "h2"]
    pp = [r for r in rows if r["kind"] == "p"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))

    tx = [float(r["true"]) for r in h2]; ex = [float(r["est"]) for r in h2]
    yerr = [[float(r["est"]) - float(r["lo"]) for r in h2],
            [float(r["hi"]) - float(r["est"]) for r in h2]]
    lim = (0, 0.9)
    a1.plot(lim, lim, "--", color="#aaa", label="truth (y = x)")
    a1.errorbar(tx, ex, yerr=yerr, fmt="o", color="#1f77b4", capsize=3, ms=6,
                label="estimate ± 95% CI")
    a1.set(title="Heritability h² recovery", xlabel="true h²", ylabel="inferred h²",
           xlim=lim, ylim=lim); a1.legend(fontsize=9, loc="upper left")

    tx = [float(r["true"]) for r in pp]; ex = [float(r["est"]) for r in pp]
    yerr = [[max(1e-6, float(r["est"]) - float(r["lo"])) for r in pp],
            [float(r["hi"]) - float(r["est"]) for r in pp]]
    plim = (3e-3, 0.3)
    a2.plot(plim, plim, "--", color="#aaa", label="truth (y = x)")
    a2.errorbar(tx, ex, yerr=yerr, fmt="o", color="#2ca02c", capsize=3, ms=6,
                label="estimate ± 95% CI")
    a2.set(title="Polygenicity p recovery", xlabel="true p", ylabel="inferred p",
           xscale="log", yscale="log", xlim=plim, ylim=plim)
    a2.legend(fontsize=9, loc="upper left")
    fig.suptitle("Inference evaluation (LDpred3-auto-infer, no validation cohort)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_infer_ldsc(pdf):
    if not HAVE_MSPRIME:
        return note_page(pdf, "Inference: LDpred3-auto vs LDSC",
                         "Needs realistic (coalescent) LD — install msprime\n"
                         "(pip install msprime) to generate this page.")
    rows = [{k: float(r[k]) for k in r} for r in data_infer_ldsc_rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    tx = [r["true_h2"] for r in rows]
    lim = (0, 0.9)
    a1.plot(lim, lim, "--", color="#aaa", label="truth (y = x)")
    a1.errorbar(tx, [r["ldsc"] for r in rows], yerr=[r["ldsc_sd"] for r in rows],
                fmt="s", color="#ff7f0e", capsize=3, ms=6, label="LDSC")
    a1.errorbar(tx, [r["auto"] for r in rows], yerr=[r["auto_sd"] for r in rows],
                fmt="o", color="#1f77b4", capsize=3, ms=6, label="LDpred3-auto")
    a1.set(title="Heritability h²: LDSC vs LDpred3-auto", xlabel="true h²",
           ylabel="inferred h²", xlim=lim, ylim=lim); a1.legend(fontsize=9, loc="upper left")

    xr = [r["r2_real"] for r in rows]; yr = [r["r2_est"] for r in rows]
    yerr = [[r["r2_est"] - r["r2_lo"] for r in rows], [r["r2_hi"] - r["r2_est"] for r in rows]]
    hi = max(max(xr), max(yr)) * 1.1
    a2.plot([0, hi], [0, hi], "--", color="#aaa", label="y = x")
    a2.errorbar(xr, yr, yerr=yerr, fmt="o", color="#2ca02c", capsize=3, ms=6,
                label="estimate ± 95% CI")
    a2.set(title="Predictive r²: estimated vs realized", xlabel="realized out-of-sample r²",
           ylabel="inferred r² (no validation set)", xlim=(0, hi), ylim=(0, hi))
    a2.legend(fontsize=9, loc="upper left")
    fig.suptitle("Inference cross-checks (realistic coalescent LD)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_bivariate(pdf):
    if not HAVE_MSPRIME:
        return note_page(pdf, "Bivariate (genetic correlation)",
                         "Needs realistic (coalescent) LD — install msprime\n"
                         "(pip install msprime) to generate this page.")
    rows = [{k: (r[k] if k == "kind" else float(r[k])) for k in r} for r in data_bivariate_rows]
    rg = [r for r in rows if r["kind"] == "rg"]
    gain = [r for r in rows if r["kind"] == "gain"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))

    tx = [r["x"] for r in rg]; lim = (-0.15, 1.0)
    a1.plot([0, 1], [0, 1], "--", color="#aaa", label="truth (y = x)")
    a1.errorbar(tx, [r["m2"] for r in rg], yerr=[r["s2"] for r in rg],
                fmt="s", color="#ff7f0e", capsize=3, ms=6, label="bivariate LDSC")
    a1.errorbar(tx, [r["m1"] for r in rg], yerr=[r["s1"] for r in rg],
                fmt="o", color="#1f77b4", capsize=3, ms=6, label="bivariate LDpred3")
    a1.set(title="Genetic correlation r_g recovery", xlabel="true r_g",
           ylabel="inferred r_g", xlim=lim, ylim=lim); a1.legend(fontsize=9, loc="upper left")

    gx = [r["x"] for r in gain]
    a2.errorbar(gx, [r["m1"] for r in gain], yerr=[r["s1"] for r in gain],
                fmt="-o", color="#7f7f7f", capsize=3, ms=6, label="univariate (weak trait)")
    a2.errorbar(gx, [r["m2"] for r in gain], yerr=[r["s2"] for r in gain],
                fmt="-o", color="#2ca02c", capsize=3, ms=6, label="bivariate (boosted)")
    a2.set(title="Weak-trait prediction gain (N₂≪N₁)", xlabel="true r_g with the strong trait",
           ylabel="weak-trait genetic R²"); a2.legend(fontsize=9, loc="upper left")
    fig.suptitle("Bivariate analysis (realistic coalescent LD)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_dentist(pdf):
    d = data_dentist_rows
    r = {k: float(d[0][k]) for k in d[0]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    bars = ["clean\n(no errors)", "corrupted\nno filter", "corrupted\n--dentist"]
    vals = [r["clean"], r["no_filter"], r["dentist"]]
    cols = ["#2ca02c", "#d62728", "#1f77b4"]
    a1.bar(bars, vals, color=cols)
    for i, v in enumerate(vals):
        a1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    a1.set(title="PRS accuracy under planted errors", ylabel="genetic R²", ylim=(0, 1.0))
    a2.bar(["errors caught", "genuine dropped\n(clean data)"],
           [100 * r["caught_frac"], 100 * r["falsedrop_frac"]], color=["#1f77b4", "#d62728"])
    a2.text(0, 100 * r["caught_frac"] + 1, f"{100*r['caught_frac']:.0f}%", ha="center", fontsize=10)
    a2.text(1, 100 * r["falsedrop_frac"] + 1, f"{100*r['falsedrop_frac']:.2f}%", ha="center", fontsize=10)
    a2.set(title="Catch rate vs false-drop cost", ylabel="% of variants", ylim=(0, 105))
    fig.suptitle("DENTIST LD-consistency filter (--dentist)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_ld_repr(pdf):
    sp = [{k: r[k] for k in r} for r in data_sparse_rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = [r["config"] for r in sp]
    dens = [100 * float(r["density"]) for r in sp]
    r2 = [float(r["r2"]) for r in sp]
    x = np.arange(len(labels))
    a1.bar(x, dens, color="#9ecae1", label="density (% stored)")
    a1.set(title="Sparse / banded LD", ylabel="stored entries (%)", xticks=x)
    a1.set_xticklabels(labels, fontsize=8)
    a1b = a1.twinx(); a1b.plot(x, r2, "-o", color="#d62728", label="genetic R²")
    a1b.set_ylabel("genetic R²", color="#d62728"); a1b.set_ylim(min(r2) - 0.01, max(r2) + 0.005)
    a1b.grid(False); a1b.tick_params(axis="y", colors="#d62728")

    sd = {r["split"]: r for r in data_split_rows}
    metrics = ["discarded LD²", "storage Σk²"]
    fixed = [float(sd["fixed"]["discarded"]), float(sd["fixed"]["storage"])]
    opt = [float(sd["optimal"]["discarded"]), float(sd["optimal"]["storage"])]
    fixed_n = [1.0, 1.0]; opt_n = [opt[0] / fixed[0], opt[1] / fixed[1]]
    xs = np.arange(len(metrics)); w = 0.35
    a2.bar(xs - w / 2, fixed_n, w, color="#7f7f7f", label="fixed")
    a2.bar(xs + w / 2, opt_n, w, color="#2ca02c", label="optimal")
    for i, v in enumerate(opt_n):
        a2.text(i + w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    a2.set(title=f"LD-block splitting (R² {float(sd['fixed']['r2']):.3f}→{float(sd['optimal']['r2']):.3f})",
           ylabel="relative to fixed (lower better)", xticks=xs, ylim=(0, 1.2))
    a2.set_xticklabels(metrics, fontsize=9); a2.legend(fontsize=9)
    fig.suptitle("LD representation: storage vs accuracy", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


def page_perf(pdf):
    if not HAVE_NUMBA:
        return note_page(pdf, "Performance (Numba / cores)",
                         "Numba not installed — install it (pip install numba) to\n"
                         "measure the JIT speed-up and multi-core scaling.")
    nb = {r["mode"]: float(r["fit_s"]) for r in data_numba_rows}
    co = data_cores_rows
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    a1.bar(["pure Python", "Numba JIT"], [nb["pure_python"], nb["numba_jit"]],
           color=["#7f7f7f", "#2ca02c"]); a1.set_yscale("log")
    a1.set(title=f"Numba JIT speed-up (~{nb['pure_python']/nb['numba_jit']:.0f}×)",
           ylabel="fit time (s, log)")
    for i, v in enumerate([nb["pure_python"], nb["numba_jit"]]):
        a1.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=10)
    nc = [int(r["ncores"]) for r in co]; su = [float(r["speedup"]) for r in co]
    a2.plot(nc, nc, "--", color="#aaa", label="ideal (linear)")
    a2.plot(nc, su, "-o", color="#1f77b4", lw=2, ms=7, label="measured")
    for x, y in zip(nc, su):
        a2.text(x, y - 0.25, f"{y:.2f}×", ha="center", fontsize=9)
    a2.set(title="Multi-core scaling (packed auto sampler)", xlabel="threads (--ncores)",
           ylabel="speed-up", xticks=nc); a2.legend(fontsize=9)
    fig.suptitle("Performance", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    print("Generating figure data (cached to benchmarks/figdata_*.csv) ...")
    data_infer_rows = cached("inference", ["kind", "true", "est", "lo", "hi"], data_inference)
    if HAVE_MSPRIME:
        data_infer_ldsc_rows = cached("infer_ldsc",
            ["true_h2", "ldsc", "ldsc_sd", "auto", "auto_sd", "r2_est", "r2_lo", "r2_hi", "r2_real"],
            data_infer_ldsc)
        data_bivariate_rows = cached("bivariate", ["kind", "x", "m1", "s1", "m2", "s2"], data_bivariate)
    data_dentist_rows = cached("dentist", ["clean", "no_filter", "dentist", "caught_frac", "falsedrop_frac"], data_dentist)
    data_sparse_rows = cached("sparse", ["config", "density", "fit_s", "r2"], data_sparse)
    data_split_rows = cached("splitting", ["split", "nblocks", "discarded", "storage", "r2"], data_splitting)
    if HAVE_NUMBA:
        data_numba_rows = cached("numba", ["mode", "fit_s"], data_numba)
        data_cores_rows = cached("cores", ["ncores", "fit_s", "speedup"], data_cores)

    out = os.path.join(HERE, "figures.pdf")
    with PdfPages(out) as pdf:
        page_bigsnpr(pdf)
        page_arch(pdf)
        page_inference(pdf)
        page_infer_ldsc(pdf)
        page_bivariate(pdf)
        page_dentist(pdf)
        page_ld_repr(pdf)
        page_perf(pdf)
        meta = pdf.infodict()
        meta["Title"] = "LDpred3 benchmark figures"
        meta["Author"] = "LDpred3"
    print(f"wrote {out} ({time.time()-t0:.0f}s)")
