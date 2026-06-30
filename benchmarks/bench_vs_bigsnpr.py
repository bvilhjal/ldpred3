"""Single-core LDpred3-vs-bigsnpr benchmark, regenerated from scratch.

Same shared simulation for both tools at each size: a block-diagonal genome of
realistic (coalescent) LD blocks cycled from ``ld_library.npz`` (100 blocks of
500), one sparse causal architecture (p, h2), and one set of standardized
marginal sumstats. Each tool is run in its own subprocess pinned to a single
core so peak RSS is isolated; identical burn-in / iterations / hyper-parameters
are passed to both. Writes ``cores_1core_benchmark.csv`` (the table + figure
source): one row per (tool, method, #SNPs) with wall-clock s, peak GB, and the
phenotype-scale prediction R2.

Run:  python benchmarks/bench_vs_bigsnpr.py            # full 200k-2M sweep
      python benchmarks/bench_vs_bigsnpr.py 200000     # one or more sizes
The R side needs bigsnpr (see bench_bigsnpr_blocks.R); set RSCRIPT / R_LIBS_USER
if R packages live in a user library.
"""
import os, sys, csv, json, time, shutil, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIBPATH = os.path.join(ROOT, "ld_library.npz")
WORK = os.environ.get("BENCH_WORK", "/tmp/bench_bigsnpr")
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")

K = 500                       # SNPs per block (matches the library)
H2, P, N = 0.5, 0.01, 50000   # architecture + GWAS power (fixed across sizes)
BURN_IN, NUM_ITER = 100, 200  # identical for both tools' grid/auto
SIZES = [200_000, 500_000, 1_000_000, 2_000_000]


def build(nsnps, rng):
    """Shared genome + standardized sumstats for a given #SNPs."""
    lib = np.load(LIBPATH)["R"].astype(np.float32)   # (100, 500, 500)
    nb = nsnps // K
    blocks = [lib[b % lib.shape[0]] for b in range(nb)]
    beta = np.zeros(nb * K)
    causal = rng.random(nb * K) < P
    beta[causal] = rng.normal(0, 1, int(causal.sum()))
    # scale to total genetic variance H2 (beta^T R beta)
    gv = sum(beta[b * K:(b + 1) * K] @ (blocks[b].astype(np.float64) @ beta[b * K:(b + 1) * K])
             for b in range(nb))
    beta *= np.sqrt(H2 / gv)
    bhat = np.empty(nb * K)
    for b in range(nb):
        Rb = blocks[b].astype(np.float64)
        L = np.linalg.cholesky(Rb + 1e-4 * np.eye(K))
        ix = slice(b * K, (b + 1) * K)
        bhat[ix] = Rb @ beta[ix] + (L @ rng.standard_normal(K)) / np.sqrt(N)
    return blocks, beta, bhat


def pheno_r2(b_est, beta, blocks):
    """Phenotype-scale R2 = genetic-R2 x h2 (var(y)=1), from population LD."""
    num = d1 = d2 = 0.0
    for b in range(len(blocks)):
        R = blocks[b].astype(np.float64)
        ix = slice(b * K, (b + 1) * K)
        Rb = R @ beta[ix]
        num += b_est[ix] @ Rb
        d1 += b_est[ix] @ (R @ b_est[ix])
        d2 += beta[ix] @ Rb
    gr2 = (num * num) / (d1 * d2) if d1 > 0 and d2 > 0 else 0.0
    return float(gr2 * H2)


def write_r_inputs(blocks, bhat):
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "sizes.txt"), "w") as fh:
        fh.write("\n".join(str(K) for _ in blocks) + "\n")
    with open(os.path.join(WORK, "blocks.bin"), "wb") as fh:
        for b in blocks:
            fh.write(np.asarray(b, np.float64).tobytes(order="F"))  # column-major for R
    import csv as _csv
    with open(os.path.join(WORK, "df_beta.csv"), "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["beta", "beta_se", "n_eff"])
        se = 1.0 / np.sqrt(N)
        for x in bhat:
            w.writerow([f"{x:.8g}", f"{se:.8g}", N])


LDPRED3_WORKER = r"""
import os, sys, json, time, resource
import numpy as np
sys.path.insert(0, %r)
from ldpred3 import ldpred3_by_blocks
d = np.load(os.path.join(%r, "ld_blocks.npz"))
nb = int(d["nb"]); K = int(d["K"]); H2 = float(d["h2"]); P = float(d["p"])
N = float(d["n"]); BURN = int(d["burn"]); IT = int(d["it"])
# Rebuild the LD in-process from the small library (no multi-GB serialization
# round-trip), so peak RSS reflects LDpred3's real float32 LD footprint.
src = np.load(str(d["libpath"]))["R"].astype(np.float32)   # ~100 blocks
full = np.empty((nb, K, K), np.float32)
for b in range(nb):
    full[b] = src[b %% src.shape[0]]
del src
blocks = [(full[b], np.arange(b*K,(b+1)*K)) for b in range(nb)]
bhat = d["bhat"]; n = np.full(nb*K, N)
def fit(m):
    if m=="inf":  return ldpred3_by_blocks(blocks,bhat,n,method="inf",h2=H2)
    if m=="grid": return ldpred3_by_blocks(blocks,bhat,n,method="grid",h2=H2,p=P,burn_in=BURN,num_iter=IT)
    # auto warm-started at the same oracle hyper-parameters bigsnpr's
    # snp_ldpred2_auto() gets via h2_init / vec_p_init -- apples-to-apples.
    if m=="auto": return ldpred3_by_blocks(blocks,bhat,n,method="auto",burn_in=BURN,num_iter=IT,seed=1,h2_init=H2,p_init=P)
fit("auto")  # warm JIT (not timed)
out={}; betas={}
for m in ("inf","grid","auto"):
    t0=time.perf_counter(); be=fit(m); out[m]=time.perf_counter()-t0; betas[m]=np.asarray(be)
mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6  # KB->GB
np.savez(os.path.join(%r,"ld_betas.npz"), **betas)
print("RESULT "+json.dumps({"time":out,"mem_gb":mem}))
"""


def run_ldpred3(blocks, bhat):
    np.savez(os.path.join(WORK, "ld_blocks.npz"), nb=len(blocks), K=K,
             h2=H2, p=P, n=float(N), burn=BURN_IN, it=NUM_ITER, bhat=bhat,
             libpath=LIBPATH)
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    code = LDPRED3_WORKER % (ROOT, WORK, WORK)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr); raise RuntimeError("ldpred3 worker failed")
    res = json.loads([l for l in r.stdout.splitlines() if l.startswith("RESULT ")][-1][7:])
    betas = np.load(os.path.join(WORK, "ld_betas.npz"))
    return res, {m: betas[m] for m in ("inf", "grid", "auto")}


def run_bigsnpr(beta, blocks):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["/usr/bin/time", "-v", RSCRIPT, os.path.join(HERE, "bench_bigsnpr_blocks.R"),
           str(H2), str(P), str(BURN_IN), str(NUM_ITER), WORK]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr); raise RuntimeError("bigsnpr R side failed")
    times, mem_gb = {}, float("nan")
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("TIME "):
            _, m, t = line.split(); times[m] = float(t)
        if "Maximum resident set size" in line:
            mem_gb = float(line.rsplit(" ", 1)[1]) / 1e6   # KB -> GB
    rb = np.genfromtxt(os.path.join(WORK, "r_betas.csv"), delimiter=",", names=True)
    betas = {m: np.asarray(rb[m], float) for m in ("inf", "grid", "auto")}
    return {"time": times, "mem_gb": mem_gb}, betas


def main(sizes):
    rows = []
    for nsnps in sizes:
        print(f"\n===== {nsnps:,} SNPs =====", flush=True)
        rng = np.random.default_rng(42)
        blocks, beta, bhat = build(nsnps, rng)
        write_r_inputs(blocks, bhat)

        t0 = time.time()
        lp_res, lp_beta = run_ldpred3(blocks, bhat)
        print(f"  LDpred3  t={lp_res['time']}  mem={lp_res['mem_gb']:.2f}GB  ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        bs_res, bs_beta = run_bigsnpr(beta, blocks)
        print(f"  bigsnpr  t={bs_res['time']}  mem={bs_res['mem_gb']:.2f}GB  ({time.time()-t0:.0f}s)", flush=True)

        for m in ("inf", "grid", "auto"):
            rows.append(["LDpred3", m, nsnps, round(lp_res["time"][m], 2),
                         round(lp_res["mem_gb"], 3), round(pheno_r2(lp_beta[m], beta, blocks), 4)])
            rows.append(["bigsnpr", m, nsnps, round(bs_res["time"][m], 2),
                         round(bs_res["mem_gb"], 3), round(pheno_r2(bs_beta[m], beta, blocks), 4)])
        # write incrementally so a long sweep is crash-safe
        with open(os.path.join(HERE, "cores_1core_benchmark.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    print("\nwrote cores_1core_benchmark.csv")


if __name__ == "__main__":
    args = [int(float(a)) for a in sys.argv[1:]] or SIZES
    main(args)
