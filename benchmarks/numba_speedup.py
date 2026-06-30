"""Numba JIT speed-up for the Gibbs sampler.

The inner sampling sweep dominates LDpred3's runtime; ``ldpred3`` JIT-compiles
it with Numba when available and otherwise runs the *identical* pure-Python code
(see ``ldpred3/_numba.py``). This script measures the gap.

It runs itself twice in subprocesses: once normally (JIT on) and once with
``NUMBA_DISABLE_JIT=1`` (Numba's njit becomes a no-op, so the same code runs in
pure Python). The reported speed-up is JIT-off-time / JIT-on-time.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/numba_speedup.py
"""
import os, sys, time, subprocess
sys.path.insert(0, "/home/user/iprs")

NB, K = 10, 200            # m = 2000 (kept modest so the pure-Python run is quick)
M = NB * K
N_GWAS = 20000
H2, P = 0.5, 0.02
BURN, ITER = 60, 150


def time_one_fit():
    """Build a problem and time a single auto fit (excludes JIT warm-up)."""
    import numpy as np
    from ldpred3.simulate import simulate_genotypes_coalescent
    from ldpred3.ld import compute_ld_blocks
    from ldpred3 import ldpred3_by_blocks
    rng = np.random.default_rng(0)
    G, _ = simulate_genotypes_coalescent(5000, M, K, seed=0)   # realistic LD
    blocks = compute_ld_blocks(G, block_size=K)
    beta = np.zeros(M); c = rng.random(M) < P
    beta[c] = rng.standard_normal(int(c.sum()))
    beta_hat = np.empty(M)
    for R, ix in [(R.astype(float), idx) for R, idx in blocks]:
        ch = np.linalg.cholesky(R + 1e-6 * np.eye(len(ix)))
        beta_hat[ix] = R @ beta[ix] + (ch @ rng.standard_normal(len(ix))) / np.sqrt(N_GWAS)
    n = np.full(M, float(N_GWAS))
    ldpred3_by_blocks(blocks, beta_hat, n, method="auto", burn_in=5, num_iter=5)  # warm-up
    t = time.time()
    ldpred3_by_blocks(blocks, beta_hat, n, method="auto", burn_in=BURN, num_iter=ITER)
    return time.time() - t


if os.environ.get("PYLDPRED2_BENCH_CHILD") == "1":
    from ldpred3._numba import HAVE_NUMBA
    dt = time_one_fit()
    print(f"{int(HAVE_NUMBA and not os.environ.get('NUMBA_DISABLE_JIT'))} {dt:.4f}")
    sys.exit(0)


def run_child(disable_jit):
    env = dict(os.environ, PYLDPRED2_BENCH_CHILD="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    if disable_jit:
        env["NUMBA_DISABLE_JIT"] = "1"
    out = subprocess.run([sys.executable, __file__], env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"child failed:\n{out.stderr}")
    active, dt = out.stdout.split()
    return int(active), float(dt)


try:
    import numba  # noqa: F401
    have = True
except ImportError:
    have = False

print(f"Numba speed-up, auto fit, m={M} ({NB}x{K}), burn_in={BURN}, num_iter={ITER}\n")
if not have:
    _, dt = run_child(disable_jit=True)
    print(f"Numba not installed; pure-Python auto fit: {dt:.2f}s")
    print("Install numba (`pip install numba`) to see the JIT speed-up.")
    sys.exit(0)

_, t_on = run_child(disable_jit=False)
_, t_off = run_child(disable_jit=True)
print(f"{'mode':>16} | {'fit time (s)':>12}")
print("-" * 32)
print(f"{'pure Python':>16} | {t_off:>12.2f}")
print(f"{'Numba JIT':>16} | {t_on:>12.2f}")
print(f"\nspeed-up: {t_off / t_on:.1f}x")
