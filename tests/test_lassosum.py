"""lassosum2: penalised-regression PRS from sumstats + LD."""

import numpy as np

from ldpred3 import lassosum2


def _sim(m=400, nblk=4, p=0.05, h2=0.5, N=20000, seed=0):
    rng = np.random.default_rng(seed)
    k = m // nblk
    blocks, R_full = [], np.zeros((m, m))
    for b in range(nblk):
        i = np.arange(k)
        rho = 0.5 + 0.3 * rng.random()
        Rb = (rho ** np.abs(i[:, None] - i[None, :]))
        blocks.append((Rb.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        R_full[b * k:(b + 1) * k, b * k:(b + 1) * k] = Rb
    causal = rng.random(m) < p
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())
    L = np.linalg.cholesky(R_full + 1e-6 * np.eye(m))
    beta_hat = R_full @ beta + (L @ rng.standard_normal(m)) / np.sqrt(N)
    return blocks, R_full, beta, beta_hat


def _genetic_corr(a, b, R):
    num = a @ (R @ b)
    da = a @ (R @ a); db = b @ (R @ b)
    return num / np.sqrt(da * db) if da > 0 and db > 0 else 0.0


def test_lassosum2_recovers_signal_and_is_sparse():
    blocks, R, beta, beta_hat = _sim(seed=1)
    res = lassosum2(blocks, beta_hat)
    # the selected score tracks the true genetic effect
    gc = _genetic_corr(res.beta_est, beta, R)
    assert gc > 0.4, f"lassosum2 genetic corr with truth too low: {gc:.3f}"
    # a lasso solution is sparse
    assert 0 < res.n_nonzero < beta_hat.size
    assert 0.0 < res.best_s <= 1.0
    assert res.best_score > 0


def test_lassosum2_lambda_controls_sparsity():
    # Across the grid, a larger penalty gives fewer non-zeros (for fixed s).
    blocks, R, beta, beta_hat = _sim(seed=2)
    res = lassosum2(blocks, beta_hat, s_seq=(0.5,), n_lambda=10)
    g = sorted([r for r in res.grid], key=lambda r: r["lambda"])  # ascending lambda
    nnz = [r["n_nonzero"] for r in g]
    # smaller lambda (front) is denser; the heaviest penalty (back) is sparsest
    assert nnz[0] >= nnz[-1]
    assert nnz[-1] <= 5          # heavy penalty -> very sparse / empty


def test_lassosum2_empty_signal():
    blocks = [(np.eye(10, dtype=np.float32), np.arange(10))]
    res = lassosum2(blocks, np.zeros(10))
    assert res.n_nonzero == 0
    assert np.all(res.beta_est == 0)
