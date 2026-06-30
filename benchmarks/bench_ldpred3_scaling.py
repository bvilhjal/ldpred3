"""LDpred3 scalability to genome scale (200k–4M SNPs), single core.

Reuses ``bench_vs_bigsnpr``'s shared realistic-LD simulation and its LDpred3
subprocess worker (which rebuilds the ``float32`` LD in-process and reports
wall-clock time and peak RSS), so this measures **only** LDpred3 across sizes —
no bigsnpr, no multi-GB disk round-trip. Writes ``ldpred3_scaling.csv`` and a
two-panel figure ``ldpred3_scaling.png`` (fit time and peak memory vs #SNPs).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_ldpred3_scaling.py
    python benchmarks/bench_ldpred3_scaling.py 1000000 2000000   # specific sizes
"""
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmarks.bench_vs_bigsnpr as B

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [int(float(a)) for a in sys.argv[1:]] or \
        [200_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000]
RAM_GB = 16.0   # this machine, for the ceiling line

rows = []
for nsnps in SIZES:
    rng = np.random.default_rng(42)
    blocks, beta, bhat = B.build(nsnps, rng)          # shared realistic-LD sim
    res, betas = B.run_ldpred3(blocks, bhat)          # subprocess: time + peak RSS
    t = res["time"]
    r2 = B.pheno_r2(betas["auto"], beta, blocks)
    rows.append([nsnps, round(res["mem_gb"], 3), round(t["inf"], 2),
                 round(t["grid"], 2), round(t["auto"], 2), round(r2, 4)])
    print(f"{nsnps:>9} | {res['mem_gb']:.2f} GB | inf {t['inf']:.1f} grid "
          f"{t['grid']:.1f} auto {t['auto']:.1f} | auto R2 {r2:.4f}", flush=True)
    with open(os.path.join(HERE, "ldpred3_scaling.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nsnps", "peak_gb", "inf_s", "grid_s", "auto_s", "auto_r2"])
        w.writerows(rows)

# ---- figure ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = np.array(rows, float)
m = a[:, 0] / 1e6          # #SNPs in millions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for j, name in ((2, "inf"), (3, "grid"), (4, "auto")):
    ax1.plot(m, a[:, j], "o-", label=name)
ax1.set_xlabel("#SNPs (millions)"); ax1.set_ylabel("fit time (s), single core")
ax1.set_title("LDpred3 fit time vs genome size"); ax1.legend(); ax1.grid(alpha=.3)

ax2.plot(m, a[:, 1], "o-", color="C3", label="peak RSS (float32 LD)")
ax2.axhline(RAM_GB, ls="--", color="gray", label=f"{RAM_GB:.0f} GB RAM")
ax2.set_xlabel("#SNPs (millions)"); ax2.set_ylabel("peak memory (GB)")
ax2.set_title("LDpred3 peak memory vs genome size"); ax2.legend(); ax2.grid(alpha=.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "ldpred3_scaling.png"), dpi=130)
print("wrote ldpred3_scaling.csv and ldpred3_scaling.png")
