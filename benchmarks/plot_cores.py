"""Plot pyLDpred2 (1 & 4 cores) vs bigsnpr (1 core): time and peak memory."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
with open("cores_benchmark.csv") as fh:
    for label, m, t, mem, r2 in csv.reader(fh):
        rows.append((label, int(m), float(t), float(mem), float(r2)))

series = {
    "pyLDpred2 auto 1c": dict(color="#1f77b4", marker="o"),
    "pyLDpred2 auto 4c": dict(color="#ff7f0e", marker="s"),
    "bigsnpr auto 1c":   dict(color="#2ca02c", marker="^"),
}
labels = {
    "pyLDpred2 auto 1c": "pyLDpred2 (1 core, streaming)",
    "pyLDpred2 auto 4c": "pyLDpred2 (4 cores, packed)",
    "bigsnpr auto 1c":   "bigsnpr (1 core)",
}

def collect(name, idx):
    xs, ys = [], []
    for label, m, t, mem, r2 in rows:
        if label == name:
            xs.append(m / 1e6)
            ys.append((t, mem)[idx - 2])
    return xs, ys

fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.8))

for name, sty in series.items():
    xs, ys = collect(name, 2)
    ax_t.plot(xs, ys, label=labels[name], **sty, lw=2, ms=7)
for name, sty in series.items():
    xs, ys = collect(name, 3)
    ax_m.plot(xs, ys, label=labels[name], **sty, lw=2, ms=7)

# mark bigsnpr OOM at 2M
ax_t.annotate("OOM", (2.0, 30), color="#2ca02c", fontsize=9, ha="center")
ax_m.annotate("OOM", (2.0, 13.5), color="#2ca02c", fontsize=9, ha="center")

ax_t.set_title("Running time (auto model)")
ax_t.set_xlabel("Number of SNPs (millions)")
ax_t.set_ylabel("Wall-clock time (s)")
ax_t.grid(alpha=0.3)
ax_t.legend(frameon=False, fontsize=9)

ax_m.set_title("Peak memory (auto model)")
ax_m.set_xlabel("Number of SNPs (millions)")
ax_m.set_ylabel("Peak RSS (GB)")
ax_m.grid(alpha=0.3)
ax_m.legend(frameon=False, fontsize=9)

fig.suptitle("pyLDpred2 (1 & 4 cores) vs bigsnpr (1 core) — distinct AR(1) LD blocks, N=100k, h²=0.5",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("cores_benchmark.png", dpi=130)
print("wrote cores_benchmark.png")
