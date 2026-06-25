"""Plot 1-core method comparison: pyLDpred2 vs bigsnpr, inf/grid/auto."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
with open("cores_1core_benchmark.csv") as fh:
    for tool, method, m, t, mem, r2 in csv.reader(fh):
        rows.append((tool, method, int(m), float(t), float(mem), float(r2)))

mcolor = {"inf": "#9467bd", "grid": "#1f77b4", "auto": "#2ca02c"}

def series(tool, method, col):
    xs, ys = [], []
    for tl, me, m, t, mem, r2 in rows:
        if tl == tool and me == method:
            xs.append(m / 1e6); ys.append((t, mem)[col])
    return xs, ys

fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.8))

for method in ("inf", "grid", "auto"):
    x, y = series("pyLDpred2", method, 0)
    ax_t.plot(x, y, "-o", color=mcolor[method], lw=2, ms=6,
              label=f"pyLDpred2 {method}")
    x, y = series("bigsnpr", method, 0)
    ax_t.plot(x, y, "--s", color=mcolor[method], lw=1.6, ms=5, alpha=0.8,
              label=f"bigsnpr {method}")
ax_t.set_title("Running time, 1 core (solid = pyLDpred2, dashed = bigsnpr)")
ax_t.set_xlabel("Number of SNPs (millions)")
ax_t.set_ylabel("Wall-clock time (s)")
ax_t.grid(alpha=0.3); ax_t.legend(frameon=False, fontsize=8, ncol=3)

x, y = series("pyLDpred2", "auto", 1)
ax_m.plot(x, y, "-o", color="#1f77b4", lw=2, ms=7, label="pyLDpred2")
x, y = series("bigsnpr", "auto", 1)
ax_m.plot(x, y, "-^", color="#2ca02c", lw=2, ms=7, label="bigsnpr")
ax_m.set_title("Peak memory (LD-dominated, ~equal across methods)")
ax_m.set_xlabel("Number of SNPs (millions)")
ax_m.set_ylabel("Peak RSS (GB)")
ax_m.grid(alpha=0.3); ax_m.legend(frameon=False, fontsize=9)

fig.suptitle("1-core comparison, realistic LD (coalescent/msprime), N=100k, h²=0.5",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("cores_1core_benchmark.png", dpi=130)
print("wrote cores_1core_benchmark.png")
