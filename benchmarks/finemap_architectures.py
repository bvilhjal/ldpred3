"""Fine-mapping methods compared across genetic architectures.

Three fine-mappers -- LDpred3-PIP (`ldpred3_pip`), the single-signal ABF baseline
(`single_signal_finemap`) and a naive **marginal top-SNP** (the most significant
variant as a size-1 credible set) -- are run on five locus architectures, on
realistic coalescent LD. Reports, per (architecture, method): credible-set
**coverage** (a set contains a true causal), **power** (true causals captured in
some set) and **resolution** (median set size).

The architectures exercise what separates the methods: a single signal, *two*
signals (independent or in LD, i.e. allelic heterogeneity), a signal on a
polygenic background, and a major + sparse mix. Self-contained (coalescent LD).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/finemap_architectures.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3 import ldpred3_pip, single_signal_finemap

K = 400
N_POP = 2000
N_GWAS = 100000
Z = 7.0                  # per-target marginal z
N_LOCI = 60
COVERAGE = 0.95
ARCHS = ["single", "two-independent", "two-linked", "causal+background", "major+sparse"]
METHODS = ["LDpred3-PIP", "ABF", "marginal-top"]


def _corr(G):
    Gs = (G - G.mean(0)) / G.std(0)
    R = (Gs.T @ Gs) / G.shape[0]
    np.fill_diagonal(R, 1.0)
    return R


def _place(rng, nc, mindist):
    cs, pool = [], list(range(K))
    rng.shuffle(pool)
    for j in pool:
        if all(abs(j - c) >= mindist for c in cs):
            cs.append(j)
        if len(cs) == nc:
            break
    return sorted(cs)


def make_locus(arch, seed):
    """Return (R_pop, beta_hat, target_causals) for one locus of the architecture."""
    rng = np.random.default_rng(seed)
    G, _ = simulate_genotypes_coalescent(N_POP, K, K, seed=seed)
    R = _corr(G)
    beta = np.zeros(K)
    b0 = Z / np.sqrt(N_GWAS)
    if arch == "single":
        tgt = _place(rng, 1, K)
        beta[tgt] = rng.choice([-1.0, 1.0]) * b0
    elif arch == "two-independent":
        tgt = _place(rng, 2, K // 2)
        beta[tgt] = rng.choice([-1.0, 1.0], 2) * b0
    elif arch == "two-linked":                      # two nearby causals (in LD)
        c0 = int(rng.integers(30, K - 40)); off = int(rng.integers(8, 25))
        tgt = [c0, c0 + off]
        beta[tgt] = rng.choice([-1.0, 1.0], 2) * b0
    elif arch == "causal+background":               # one signal on a polygenic bg
        tgt = _place(rng, 1, K)
        bg = rng.random(K) < 0.1
        beta[bg] += rng.normal(0, 0.6 / np.sqrt(N_GWAS), int(bg.sum()))
        beta[tgt] = rng.choice([-1.0, 1.0]) * b0
    else:                                           # major + sparse: 1 huge + 2 small
        tgt = _place(rng, 3, 20)
        beta[tgt[0]] = rng.choice([-1.0, 1.0]) * 12.0 / np.sqrt(N_GWAS)
        beta[tgt[1:]] = rng.choice([-1.0, 1.0], 2) * 4.0 / np.sqrt(N_GWAS)
    chol = np.linalg.cholesky(R + 1e-6 * np.eye(K))
    bhat = R @ beta + (chol @ rng.standard_normal(K)) / np.sqrt(N_GWAS)
    return R, bhat, np.array(sorted(tgt))


def credible_sets(method, R, bhat, _t):
    if method == "LDpred3-PIP":
        return [cs.variants for cs in ldpred3_pip(R, bhat, N_GWAS,
                coverage=COVERAGE, seed=1).credible_sets]
    if method == "ABF":
        return [cs.variants for cs in single_signal_finemap(R, bhat, N_GWAS,
                coverage=COVERAGE).credible_sets]
    top = int(np.argmax(np.abs(bhat * np.sqrt(N_GWAS))))   # marginal top SNP
    return [np.array([top])]


def run(arch, method):
    cs_tot = cs_cov = tg_tot = tg_found = 0
    sizes = []
    for rep in range(N_LOCI):
        R, bhat, tgt = make_locus(arch, 4000 + rep)
        sets = credible_sets(method, R, bhat, tgt)
        tset = set(int(c) for c in tgt)
        cs_tot += len(sets)
        cs_cov += sum(1 for s in sets if tset & set(int(v) for v in s))
        tg_tot += tgt.size
        tg_found += sum(1 for c in tgt
                        if any(int(c) in set(int(v) for v in s) for s in sets))
        sizes += [len(s) for s in sets]
    coverage = cs_cov / cs_tot if cs_tot else float("nan")
    power = tg_found / tg_tot if tg_tot else 0.0
    med = float(np.median(sizes)) if sizes else float("nan")
    return coverage, power, med, cs_tot / N_LOCI


t0 = time.time()
print(f"Fine-mapping methods across architectures, coalescent LD, K={K}/locus, "
      f"N={N_GWAS}, target z={Z}, {N_LOCI} loci/cell\n")
print(f"{'architecture':>18} | {'method':>13} | {'coverage':>8} | {'power':>6} "
      f"| {'med|CS|':>7} | {'sets/loc':>8}")
print("-" * 76)
for arch in ARCHS:
    for method in METHODS:
        cov, pw, med, spl = run(arch, method)
        print(f"{arch:>18} | {method:>13} | {cov:>8.2f} | {pw:>6.2f} | "
              f"{med:>7.0f} | {spl:>8.2f}")
    print()
print(f"(coverage = credible sets containing a true causal; power = causals "
      f"captured; {time.time()-t0:.0f}s)")
