"""Plot genetic-R2 of LDpred2 variants across architectures (two power levels)."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.reader(open("methods_arch.csv")))
header = rows[0]
methods = header[2:]
data = {}                                  # data[N][model] = [values per method]
for r in rows[1:]:
    N = int(r[0]); model = r[1]
    data.setdefault(N, {})[model] = [float(x) for x in r[2:]]
Ns = sorted(data)
models = list(data[Ns[0]])
labels = {"infinitesimal": "infinitesimal", "sparse": "sparse (p=0.01)",
          "polygenic": "polygenic (p=0.2)", "major_locus": "major locus",
          "annot_enriched": "annotation-enriched"}
colors = {"marginal": "#999999", "inf": "#9467bd", "grid": "#1f77b4",
          "auto": "#2ca02c", "annot": "#ff7f0e"}

fig, axes = plt.subplots(1, len(Ns), figsize=(13, 5), sharey=True)
x = np.arange(len(models)); w = 0.16
for ax, N in zip(axes, Ns):
    for k, meth in enumerate(methods):
        vals = [data[N][m][k] for m in models]
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, vals, w,
               label=meth, color=colors.get(meth))
    ax.set_title(f"N = {N:,}")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models], rotation=20, ha="right",
                       fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.4, 1.0)
axes[0].set_ylabel("Genetic R² (PRS vs true genetic value)")
axes[-1].legend(frameon=False, fontsize=9, title="method")
fig.suptitle("LDpred2 variants on realistic (coalescent) LD, by genetic "
             "architecture — m=50000, h²=0.5", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("methods_arch.png", dpi=130)
print("wrote methods_arch.png")
