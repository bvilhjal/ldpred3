"""Plot realistic-LD benchmark: pyLDpred2 (1 & 4 cores) vs bigsnpr (1 core)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
with open("cores_realistic_benchmark.csv") as fh:
    for label, m, t, mem, r2 in csv.reader(fh):
        rows.append((label, int(m), float(t), float(mem), float(r2)))

series = {
    "pyLDpred2 auto 1c": ("#1f77b4", "o", "pyLDpred2 (1 core, streaming)"),
    "pyLDpred2 auto 4c": ("#ff7f0e", "s", "pyLDpred2 (4 cores, packed)"),
    "bigsnpr auto 1c":   ("#2ca02c", "^", "bigsnpr (1 core)"),
}

def collect(name, col):
    xs, ys = [], []
    for label, m, t, mem, r2 in rows:
        if label == name:
            xs.append(m / 1e6); ys.append((t, mem)[col])
    return xs, ys

fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.8))
for col, ax, ttl, ylab in [(0, ax_t, "Running time (auto model)", "Wall-clock time (s)"),
                           (1, ax_m, "Peak memory (auto model)", "Peak RSS (GB)")]:
    for name, (color, marker, lab) in series.items():
        xs, ys = collect(name, col)
        ax.plot(xs, ys, color=color, marker=marker, label=lab, lw=2, ms=7)
    ax.set_title(ttl); ax.set_xlabel("Number of SNPs (millions)")
    ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)

fig.suptitle("Realistic LD (coalescent/msprime) — pyLDpred2 vs bigsnpr, N=100k, h²=0.5",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("cores_realistic_benchmark.png", dpi=130)
print("wrote cores_realistic_benchmark.png")
