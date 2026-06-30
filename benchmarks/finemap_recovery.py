"""Fine-mapping recovery: does LDpred3-PIP fine-mapping find the causal variants?

Self-contained (coalescent LD). Simulates many loci with realistic
coalescent-with-recombination LD, plants a known set of causal variants per
locus, builds standardized marginal sumstats, and runs LDpred3-PIP fine-mapping
(`ldpred3_pip`). Reports the quantities that define fine-mapping quality:

  * coverage   -- fraction of 95% credible sets that contain a true causal
                  variant (should be ~0.95 if calibrated)
  * power      -- fraction of true causal variants captured in some credible set
  * resolution -- median credible-set size (smaller = sharper localisation)
  * sets/locus -- mean number of credible sets reported
  * PIP@causal -- mean PIP on the true causal variants

(A) sweeps the number of causal variants per locus; (B) compares clean
(population) LD against a finite reference panel -- the real-world mismatch; (C)
contrasts with the single-causal ABF baseline.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/finemap_recovery.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/iprs")
from ldpred3.simulate import simulate_genotypes_coalescent
from ldpred3 import ldpred3_pip, single_signal_finemap

K = 400                    # SNPs per locus
N_POP = 2000               # coalescent sample defining the population LD
N_REF = 500                # finite reference-panel size (the mismatch case)
N_GWAS = 100000            # GWAS sample size
N_LOCI = 80                # replicate loci per cell (for a stable coverage estimate)
COVERAGE = 0.95
MIN_DIST = 20              # min spacing between planted causals (separable signals)


def _corr(G):
    Gs = (G - G.mean(0)) / G.std(0)
    R = (Gs.T @ Gs) / G.shape[0]
    np.fill_diagonal(R, 1.0)
    return R


def make_locus(n_causal, target_z, seed):
    """One coalescent locus: population LD, finite-panel LD, causals, sumstats."""
    rng = np.random.default_rng(seed)
    Gp, _ = simulate_genotypes_coalescent(N_POP, K, K, seed=seed)
    Rpop = _corr(Gp)
    ref_rows = rng.choice(N_POP, N_REF, replace=False)
    Rref = _corr(Gp[ref_rows])

    # plant causals at least MIN_DIST apart
    causal = []
    pool = list(range(K))
    rng.shuffle(pool)
    for j in pool:
        if all(abs(j - c) >= MIN_DIST for c in causal):
            causal.append(j)
        if len(causal) == n_causal:
            break
    causal = np.array(sorted(causal))

    beta = np.zeros(K)
    beta[causal] = (rng.choice([-1.0, 1.0], causal.size) * target_z / np.sqrt(N_GWAS))
    chol = np.linalg.cholesky(Rpop + 1e-6 * np.eye(K))
    beta_hat = Rpop @ beta + (chol @ rng.standard_normal(K)) / np.sqrt(N_GWAS)
    return Rpop, Rref, causal, beta_hat


def score(finemap_fn, R, beta_hat, causal):
    """Run a fine-mapper, return per-locus (n_cs, n_cs_covered, n_caus_found,
    cs_sizes, pip_at_causal)."""
    res = finemap_fn(R, beta_hat, N_GWAS)
    cset = set(int(c) for c in causal)
    n_cs = len(res.credible_sets)
    covered = sum(1 for cs in res.credible_sets if cset & set(int(v) for v in cs.variants))
    found = sum(1 for c in causal if any(int(c) in set(int(v) for v in cs.variants)
                                         for cs in res.credible_sets))
    sizes = [len(cs.variants) for cs in res.credible_sets]
    pip_caus = float(np.mean(res.pip[causal]))
    return n_cs, covered, found, sizes, pip_caus


def run(finemap_fn, n_causal, target_z, use_ref):
    cs_tot = cs_cov = caus_tot = caus_found = 0
    sizes, pips = [], []
    t0 = time.time()
    for rep in range(N_LOCI):
        Rpop, Rref, causal, beta_hat = make_locus(n_causal, target_z, 1000 + rep)
        R = Rref if use_ref else Rpop
        n_cs, cov, found, sz, pipc = score(finemap_fn, R, beta_hat, causal)
        cs_tot += n_cs; cs_cov += cov
        caus_tot += causal.size; caus_found += found
        sizes += sz; pips.append(pipc)
    dt = (time.time() - t0) / N_LOCI
    coverage = cs_cov / cs_tot if cs_tot else float("nan")
    power = caus_found / caus_tot if caus_tot else 0.0
    med_size = float(np.median(sizes)) if sizes else float("nan")
    return coverage, power, med_size, cs_tot / N_LOCI, float(np.mean(pips)), dt


_PIP = lambda R, bh, n: ldpred3_pip(R, bh, n, coverage=COVERAGE, seed=1)
_ABF = lambda R, bh, n: single_signal_finemap(R, bh, n, coverage=COVERAGE)

print(f"Fine-mapping recovery, coalescent LD, K={K}/locus, N_gwas={N_GWAS}, "
      f"{N_LOCI} loci/cell\n")

print("(A) LDpred3-PIP vs signal strength (per-causal z), 1 causal, clean LD:")
print(f"{'z':>5} | {'coverage':>8} | {'power':>6} | {'med|CS|':>7} | "
      f"{'sets/loc':>8} | {'PIP@caus':>8} | {'s/locus':>7}")
print("-" * 66)
for z in (4.0, 6.0, 8.0, 10.0):
    cov, pw, ms, spl, pc, dt = run(_PIP, 1, z, use_ref=False)
    print(f"{z:>5.0f} | {cov:>8.2f} | {pw:>6.2f} | {ms:>7.0f} | {spl:>8.2f} | "
          f"{pc:>8.3f} | {dt:>7.3f}")

print("\n(B) LDpred3-PIP vs #causal/locus (z=8, clean LD):")
print(f"{'#causal':>7} | {'coverage':>8} | {'power':>6} | {'med|CS|':>7} | {'sets/loc':>8}")
print("-" * 52)
for nc in (1, 2, 3):
    cov, pw, ms, spl, pc, dt = run(_PIP, nc, 8.0, use_ref=False)
    print(f"{nc:>7} | {cov:>8.2f} | {pw:>6.2f} | {ms:>7.0f} | {spl:>8.2f}")

print("\n(C) clean population LD vs finite reference panel (1 causal, z=8):")
print(f"{'LD':>18} | {'coverage':>8} | {'power':>6} | {'med|CS|':>7} | {'PIP@caus':>8}")
print("-" * 60)
for label, use_ref in (("population (clean)", False), (f"reference Nref={N_REF}", True)):
    cov, pw, ms, spl, pc, dt = run(_PIP, 1, 8.0, use_ref)
    print(f"{label:>18} | {cov:>8.2f} | {pw:>6.2f} | {ms:>7.0f} | {pc:>8.3f}")

print("\n(D) LDpred3-PIP vs single-signal ABF baseline (1 causal, z=8, clean LD):")
print(f"{'method':>16} | {'coverage':>8} | {'power':>6} | {'med|CS|':>7}")
print("-" * 50)
for label, fn in (("LDpred3-PIP", _PIP), ("ABF (single)", _ABF)):
    cov, pw, ms, spl, pc, dt = run(fn, 1, 8.0, use_ref=False)
    print(f"{label:>16} | {cov:>8.2f} | {pw:>6.2f} | {ms:>7.0f}")
