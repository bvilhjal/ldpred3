"""Methods compared on BOTH accuracy and scalability, as the genome grows.

For each genome size (realistic coalescent LD cycled from ``ld_library.npz``,
one sparse architecture h2=0.5/p=0.01, N=50k, single core) every LDpred3 method
is fit and scored: marginal (no LD), inf, grid (oracle h2/p), auto (self-tuning)
and annot (one uninformative annotation). Each size runs in its own subprocess
(``_method_worker.py``) so the peak memory is clean. Reports genetic R2, fit time
and peak memory per method vs #SNPs, writes ``method_scaling.csv`` and a 3-panel
figure ``method_scaling.png``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/method_scaling.py
    python benchmarks/method_scaling.py 100000 500000      # custom sizes
"""
import os
import sys
import csv
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_method_worker.py")
METHODS = ["marginal", "inf", "grid", "auto", "annot", "lassosum2"]
SIZES = [int(float(a)) for a in sys.argv[1:]] or \
        [50_000, 100_000, 200_000, 500_000, 1_000_000]

env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

results = {}   # nsnps -> {"mem_gb":.., "methods": {meth: {time,r2}}}
for m in SIZES:
    r = subprocess.run([sys.executable, WORKER, str(m)], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"worker failed at m={m}")
    res = json.loads([ln for ln in r.stdout.splitlines()
                      if ln.startswith("RESULT ")][-1][7:])
    results[m] = res
    a = res["methods"]
    print(f"  m={m:>8}  mem={res['mem_gb']:.2f}GB  "
          + "  ".join(f"{k}:R2={a[k]['r2']:.3f}/{a[k]['time']:.1f}s"
                      for k in METHODS), flush=True)

rows = []
for m in SIZES:
    a = results[m]["methods"]
    for meth in METHODS:
        rows.append([m, meth, round(a[meth]["time"], 3), round(a[meth]["r2"], 4),
                     round(results[m]["mem_gb"], 3)])
with open(os.path.join(HERE, "method_scaling.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["nsnps", "method", "time_s", "genetic_r2", "peak_gb"])
    w.writerows(rows)


def _table(title, key, fmt):
    print(f"\n{title}")
    hdr = f"{'#SNPs':>8} | " + " ".join(f"{m:>8}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    for m in SIZES:
        a = results[m]["methods"]
        print(f"{m:>8} | " + " ".join(f"{a[meth][key]:>8{fmt}}" for meth in METHODS))


print("\nMethods: accuracy vs scalability on realistic coalescent LD "
      "(h2=0.5, p=0.01, N=50000, single core)")
_table("Genetic R2 by method vs #SNPs (accuracy; power dilutes as #SNPs grows):",
       "r2", ".3f")
_table("Fit time (s) by method vs #SNPs (scalability):", "time", ".1f")
print("\nPeak memory (GB, LD-dominated -> ~method-independent):")
for m in SIZES:
    print(f"{m:>8} | {results[m]['mem_gb']:.2f} GB")

# ---- figure ---------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array(SIZES) / 1e6
    colors = {"marginal": "C0", "inf": "C1", "grid": "C2",
              "auto": "C3", "annot": "C4", "lassosum2": "C5"}  # consistent across panels
    fig, (axr, axt, axm) = plt.subplots(1, 3, figsize=(14, 4.2))
    for meth in METHODS:
        r2 = [results[m]["methods"][meth]["r2"] for m in SIZES]
        tt = [results[m]["methods"][meth]["time"] for m in SIZES]
        axr.plot(x, r2, "o-", color=colors[meth], label=meth)
        if meth != "marginal":
            axt.plot(x, tt, "o-", color=colors[meth], label=meth)
    axr.set_xlabel("#SNPs (millions)"); axr.set_ylabel("genetic R²")
    axr.set_title("Accuracy vs genome size"); axr.legend(); axr.grid(alpha=.3)
    axt.set_xlabel("#SNPs (millions)"); axt.set_ylabel("fit time (s), single core")
    axt.set_title("Fit time vs genome size"); axt.legend(); axt.grid(alpha=.3)
    mem = [results[m]["mem_gb"] for m in SIZES]
    axm.plot(x, mem, "o-", color="C3")
    axm.set_xlabel("#SNPs (millions)"); axm.set_ylabel("peak memory (GB)")
    axm.set_title("Peak memory (LD-dominated)"); axm.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "method_scaling.png"), dpi=130)
    print("\nwrote method_scaling.csv and method_scaling.png")
except ImportError:
    print("\nwrote method_scaling.csv (matplotlib absent: no figure)")
