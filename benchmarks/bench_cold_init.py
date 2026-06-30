"""Cold-init auto: LDpred3 vs bigsnpr when NEITHER tool gets the oracle
hyper-parameters (the realistic way auto is used). Both start at
h2_init=0.1, p_init=0.1, single chain, identical burn-in / iterations, on the
same coalescent-LD simulation as bench_vs_bigsnpr.py. Writes cold_init_auto.csv
(one row per #SNPs: LDpred3 R2/time, bigsnpr R2/time).

Run:  python benchmarks/bench_cold_init.py [sizes...]
Needs bigsnpr (see bench_bigsnpr_blocks.R); set RSCRIPT / R_LIBS_USER if needed.
"""
import os, sys, csv, time, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmarks.bench_vs_bigsnpr as B
from ldpred3 import ldpred3_by_blocks

H2I, PI = 0.1, 0.1     # cold init for BOTH tools (agnostic, not the truth)
SIZES = [int(float(a)) for a in sys.argv[1:]] or [200_000, 500_000, 1_000_000, 2_000_000]


def ldpred3_auto_cold(blocks_views, bhat, nb):
    n = np.full(nb * B.K, float(B.N))
    blk = [(blocks_views[b], np.arange(b * B.K, (b + 1) * B.K)) for b in range(nb)]
    kw = dict(method="auto", burn_in=B.BURN_IN, num_iter=B.NUM_ITER, seed=1,
              h2_init=H2I, p_init=PI)
    ldpred3_by_blocks(blk, bhat, n, **kw)                       # warm JIT (untimed)
    t0 = time.perf_counter()
    be = ldpred3_by_blocks(blk, bhat, n, **kw)
    return np.asarray(be), time.perf_counter() - t0


def bigsnpr_auto_cold():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [os.environ.get("RSCRIPT", "Rscript"),
           os.path.join(B.HERE, "bench_bigsnpr_blocks.R"),
           str(B.H2), str(B.P), str(B.BURN_IN), str(B.NUM_ITER), B.WORK,
           str(H2I), str(PI)]                                   # cold init -> args 6-7
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr); raise RuntimeError("bigsnpr R side failed")
    t_auto = next(float(l.split()[2]) for l in (r.stdout + r.stderr).splitlines()
                  if l.startswith("TIME auto"))
    rb = np.genfromtxt(os.path.join(B.WORK, "r_betas.csv"), delimiter=",", names=True)
    return np.asarray(rb["auto"], float), t_auto


def main(sizes):
    rows = []
    for nsnps in sizes:
        rng = np.random.default_rng(42)                # same sim as bench_vs_bigsnpr
        blocks, beta, bhat = B.build(nsnps, rng)
        B.write_r_inputs(blocks, bhat)
        lp_b, lp_t = ldpred3_auto_cold(blocks, bhat, nsnps // B.K)
        bs_b, bs_t = bigsnpr_auto_cold()
        row = [nsnps, round(B.pheno_r2(lp_b, beta, blocks), 4), round(lp_t, 2),
               round(B.pheno_r2(bs_b, beta, blocks), 4), round(bs_t, 2)]
        rows.append(row)
        print(f"{nsnps:>9}: LDpred3 {row[1]} ({row[2]}s)   bigsnpr {row[3]} ({row[4]}s)", flush=True)
        with open(os.path.join(B.HERE, "cold_init_auto.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["nsnps", "ldpred3_r2", "ldpred3_s", "bigsnpr_r2", "bigsnpr_s"])
            w.writerows(rows)
    print("wrote cold_init_auto.csv")


if __name__ == "__main__":
    main(SIZES)
