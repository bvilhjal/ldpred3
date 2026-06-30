"""End-to-end pipeline test: simulate -> GWAS -> PLINK+sumstats -> PRS."""


import numpy as np


from ldpred3.genotype_io import VariantTable, SampleTable, write_plink   # noqa: E402
from ldpred3.bgen_io import write_bgen                                   # noqa: E402
from ldpred3.pipeline import run_ldpred3_prs                             # noqa: E402


def _simulate(tmp_path, n_train=4000, n_target=1500, m=600, p_causal=0.1,
              h2=0.5, seed=0):
    """Make a train cohort (for GWAS) and a target cohort sharing variants."""
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.1, 0.9, m)

    def draw(n):
        return rng.binomial(2, af, size=(n, m)).astype(np.int8)

    G_tr, G_te = draw(n_train), draw(n_target)

    # True standardized effects on a sparse set of causal variants.
    causal = rng.random(m) < p_causal
    if not causal.any():
        causal[rng.integers(m)] = True
    beta = np.zeros(m)
    beta[causal] = rng.normal(0, np.sqrt(h2 / causal.sum()), causal.sum())

    def standardize(G):
        Z = G.astype(float)
        Z -= Z.mean(0); Z /= Z.std(0)
        return Z

    Ztr = standardize(G_tr)
    g_tr = Ztr @ beta
    y_tr = g_tr + rng.normal(0, np.sqrt(1 - g_tr.var()), n_train)

    # Marginal GWAS on the training cohort (per standardized variant).
    bhat = (Ztr.T @ y_tr) / n_train
    se = np.sqrt((y_tr.var() - bhat ** 2 * 0) / n_train)  # ~ 1/sqrt(N)
    se = np.full(m, 1 / np.sqrt(n_train))

    # True target genetic value (for evaluation).
    g_te = standardize(G_te) @ beta

    # Variant + sample tables.
    a1 = np.array(["A"] * m, dtype=object)
    a2 = np.array(["G"] * m, dtype=object)
    variants = VariantTable(
        chrom=np.array(["1"] * m, dtype=object),
        id=np.array([f"rs{i}" for i in range(m)], dtype=object),
        cm=np.zeros(m), pos=np.arange(1, m + 1, dtype=np.int64) * 100,
        a1=a1, a2=a2)

    def samples(n, tag):
        return SampleTable(
            fid=np.array([f"{tag}{i}" for i in range(n)], dtype=object),
            iid=np.array([f"{tag}{i}" for i in range(n)], dtype=object),
            sex=np.ones(n, dtype=np.int64), pheno=np.full(n, np.nan))

    prefix = str(tmp_path / "target")
    smp = samples(n_target, "T")
    write_plink(prefix, G_te, variants, smp)
    write_bgen(str(tmp_path / "target.bgen"), G_te, variants, smp)

    ss_path = str(tmp_path / "gwas.txt")
    with open(ss_path, "w") as fh:
        fh.write("SNP\tA1\tA2\tBETA\tSE\tN\n")
        for i in range(m):
            fh.write(f"rs{i}\tA\tG\t{bhat[i]:.6g}\t{se[i]:.6g}\t{n_train}\n")

    return prefix, ss_path, g_te


def _write_ref(tmp_path, m, *, swap=None, n_ref=3000, seed=99, name="ref"):
    """External LD reference sharing the target's variant IDs. If ``swap`` is a
    boolean mask, those variants count the OTHER allele (A1/A2 exchanged and
    dosage -> 2 - dosage) — the orientation mismatch the pipeline must undo."""
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.1, 0.9, m)
    G = rng.binomial(2, af, size=(n_ref, m)).astype(np.int8)
    a1 = np.array(["A"] * m, dtype=object)
    a2 = np.array(["G"] * m, dtype=object)
    if swap is not None and np.any(swap):
        G = G.copy(); G[:, swap] = 2 - G[:, swap]
        a1 = a1.copy(); a2 = a2.copy()
        a1[swap], a2[swap] = a2[swap], a1[swap]
    variants = VariantTable(
        chrom=np.array(["1"] * m, dtype=object),
        id=np.array([f"rs{i}" for i in range(m)], dtype=object),
        cm=np.zeros(m), pos=np.arange(1, m + 1, dtype=np.int64) * 100, a1=a1, a2=a2)
    smp = SampleTable(
        fid=np.array([f"R{i}" for i in range(n_ref)], dtype=object),
        iid=np.array([f"R{i}" for i in range(n_ref)], dtype=object),
        sex=np.ones(n_ref, dtype=np.int64), pheno=np.full(n_ref, np.nan))
    prefix = str(tmp_path / name)
    write_plink(prefix, G, variants, smp)
    return prefix


def test_external_ld_reference_allele_orientation(tmp_path):
    # An LD-reference panel that counts the opposite allele for some variants
    # must give the SAME PRS as a correctly-oriented panel: the pipeline recodes
    # reference dosages to the target's counted allele before building LD.
    m = 400
    prefix, ss_path, _ = _simulate(tmp_path, m=m, seed=7)
    oriented = _write_ref(tmp_path, m, swap=None, name="ref_ok")
    swap = np.zeros(m, dtype=bool); swap[::3] = True          # flip 1/3 of variants
    swapped = _write_ref(tmp_path, m, swap=swap, name="ref_swap")

    kw = dict(method="auto", block_size=200, num_iter=120, burn_in=40, seed=1)
    a = run_ldpred3_prs(ss_path, prefix, ld_prefix=oriented, **kw)
    b = run_ldpred3_prs(ss_path, prefix, ld_prefix=swapped, **kw)
    # Same underlying reference genotypes, only re-oriented -> identical LD ->
    # (near-)identical scores. Without the recoding fix the flipped-sign LD
    # would change the posterior and the correlation would drop well below 1.
    assert np.corrcoef(a.scores, b.scores)[0, 1] > 0.999


def test_end_to_end_prs_predicts_genetic_value(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, seed=1)
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          num_iter=150, burn_in=50, seed=1)

    assert res.scores.shape[0] == len(g_te)
    assert res.harmonize_log["n_matched"] == 600
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 vs true genetic value too low: {r2:.3f}"


def test_end_to_end_prs_via_bgen(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, seed=1)
    res = run_ldpred3_prs(ss_path, str(tmp_path / "target.bgen"), method="auto",
                          block_size=200, num_iter=150, burn_in=50, seed=1)
    assert res.scores.shape[0] == len(g_te)
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"BGEN PRS R^2 vs true genetic value too low: {r2:.3f}"


def test_end_to_end_inf_runs(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=2)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=200)
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.10


def test_pipeline_infer_reports_h2_p_r2(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=4)
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          num_iter=120, burn_in=60, seed=1,
                          infer=True, infer_params={"n_chains": 6,
                                                    "burn_in": 100,
                                                    "num_iter": 120})
    assert res.inference is not None
    inf = res.inference
    assert 0 < inf["h2_est"] < 1.5
    assert 0 < inf["p_est"] <= 1
    assert inf["r2_ci"][0] <= inf["r2_est"] <= inf["r2_ci"][1]


def test_pipeline_infer_streams_past_old_cap(tmp_path):
    # Inference now streams block-diagonal LD, so it runs even when the number of
    # variants exceeds the old dense-assembly cap (no size-guard error).
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=5)
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          num_iter=120, burn_in=60, seed=1, infer=True,
                          infer_max_variants=100,          # below m=400; ignored now
                          infer_params={"n_chains": 6, "burn_in": 80,
                                        "num_iter": 100})
    assert res.inference is not None
    assert 0 < res.inference["h2_est"] < 1.5


def test_pipeline_dentist_runs_and_keeps_signal(tmp_path):
    # The DENTIST filter runs end-to-end, logs its counts, and (on clean
    # simulated data) leaves the PRS predictive.
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=8)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=200,
                          dentist=True)
    assert "dentist" in res.qc_log
    assert res.qc_log["dentist"]["n_kept"] <= res.qc_log["dentist"]["n_input"]
    # PRS still correlates with the held-out genetic value.
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 after DENTIST too low: {r2:.3f}"


def test_pipeline_auto_chains_robust_prs(tmp_path):
    # Multi-chain auto (Privé 2023) runs end-to-end, predicts, and reuses the run
    # for --infer (one InferResult drives both weights and h2/p/r2).
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=8)
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          auto_chains=4, infer=True,
                          infer_params={"burn_in": 60, "num_iter": 80})
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"multi-chain auto PRS R^2 too low: {r2:.3f}"
    # the multi-chain run also populated the inference dict
    assert res.inference is not None
    assert 0 < res.inference["h2_est"] < 1.5
    assert res.inference["n_chains_kept"] >= 2


def test_pipeline_ldsc_init_seeds_h2(tmp_path):
    # --ldsc-init seeds the sampler's h2 from LD Score regression; it runs
    # end-to-end, logs the seed, and stays predictive.
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=8)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=200,
                          ldsc_init=True)
    assert "ldsc_h2_init" in res.qc_log
    assert 0 < res.qc_log["ldsc_h2_init"] <= 1.0
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20


def test_pipeline_impute_n_runs_and_keeps_signal(tmp_path):
    # --impute-n runs end-to-end, logs its diagnostics, and (when the reported N
    # is already correct) leaves the PRS predictive — no harm.
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=8)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=200,
                          impute_n=True)
    assert "impute_n" in res.qc_log
    log = res.qc_log["impute_n"]
    assert 0 < log["median_imputed_n"] <= log["n_total"]
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 with impute_n too low: {r2:.3f}"


def test_pipeline_ld_shrink_runs_and_keeps_signal(tmp_path):
    # Size-aware LD shrinkage runs end-to-end, logs n_ref, and stays predictive.
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=9)
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          ld_shrink=True)
    assert "ld_shrink" in res.qc_log
    assert res.qc_log["ld_shrink"]["n_ref"] > 0
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 with LD shrink too low: {r2:.3f}"


def test_pipeline_ld_sparse_runs_and_keeps_signal(tmp_path):
    # Banded SparseLD blocks fit via the streaming auto kernel, stay predictive,
    # and round-trip through the on-disk cache.
    from ldpred3 import SparseLD
    from ldpred3.ld import load_ld_blocks
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=11)
    cache = tmp_path / "ld_sparse.npz"
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          ld_sparse=True, ld_sparse_params={"max_dist": 80},
                          ld_out=str(cache))
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 with sparse LD too low: {r2:.3f}"
    blocks, _ = load_ld_blocks(str(cache))       # cache stores banded CSR
    assert all(isinstance(R, SparseLD) for R, _ in blocks)


def test_pipeline_ld_stream_cache_roundtrip(tmp_path):
    # Build a memmap LD cache (--ld-stream), then a second run streams it from
    # disk (--ld-cache) and gives identical scores at O(one block) resident.
    import numpy as _np
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=13)
    cache = tmp_path / "ld_stream.npz"
    r1 = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                         ld_lowrank=True, ld_out=str(cache), ld_stream=True)
    assert (tmp_path / "ld_stream.npz.dat.npy").exists()   # memmap sidecar
    r2 = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                         ld_cache=str(cache))               # streams from disk
    # Same model -> essentially identical PRS.
    assert _np.corrcoef(r1.scores, r2.scores)[0, 1] > 0.999


def test_pipeline_ld_lowrank_runs_and_keeps_signal(tmp_path):
    # Low-rank LD blocks fit via the eigenspace streaming auto kernel, stay
    # predictive, and round-trip through the on-disk cache as factors.
    from ldpred3 import LowRankLD
    from ldpred3.ld import load_ld_blocks
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=12)
    cache = tmp_path / "ld_lr.npz"
    res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=200,
                          ld_lowrank=True,
                          ld_lowrank_params={"lowrank_variance": 0.99},
                          ld_out=str(cache))
    r2 = np.corrcoef(res.scores, g_te)[0, 1] ** 2
    assert r2 > 0.20, f"PRS R^2 with low-rank LD too low: {r2:.3f}"
    blocks, _ = load_ld_blocks(str(cache))
    assert all(isinstance(R, LowRankLD) for R, _ in blocks)


def test_pipeline_method_annot(tmp_path):
    # method="annot": reads an annotation file, learns enrichment, scores predict.
    rng = np.random.default_rng(3)
    n, m = 800, 300
    af = rng.uniform(0.1, 0.9, m)
    func = (rng.random(m) < 0.2).astype(int)
    G = rng.binomial(2, af, size=(n, m)).astype(np.int8)
    V = VariantTable(np.array(["1"] * m, object),
                     np.array([f"rs{i}" for i in range(m)], object),
                     np.zeros(m), np.arange(1, m + 1, dtype=np.int64) * 100,
                     np.array(["A"] * m, object), np.array(["G"] * m, object))
    S = SampleTable(np.array([f"I{i}" for i in range(n)], object),
                    np.array([f"I{i}" for i in range(n)], object),
                    np.ones(n, np.int64), np.full(n, np.nan))
    prefix = str(tmp_path / "t"); write_plink(prefix, G, V, S)
    ss = str(tmp_path / "gwas.txt")
    with open(ss, "w") as fh:
        fh.write("SNP\tA1\tA2\tBETA\tSE\tN\n")
        for i in range(m):
            b = rng.normal(0, 0.08) if func[i] else rng.normal(0, 0.02)
            fh.write(f"rs{i}\tA\tG\t{b:.5g}\t0.02\t5000\n")
    ann = str(tmp_path / "annot.tsv")
    with open(ann, "w") as fh:
        fh.write("SNP\tcoding\n")
        for i in range(m):
            fh.write(f"rs{i}\t{func[i]}\n")

    res = run_ldpred3_prs(ss, prefix, method="annot", annotations=ann,
                          block_size=100,
                          annot_params=dict(burn_in=60, num_iter=150, seed=1))
    assert res.enrichment is not None
    assert res.enrichment["coding"] > 0.3        # functional annotation enriched
    assert res.scores.shape[0] == n

    # method="annot" without annotations is an error.
    try:
        run_ldpred3_prs(ss, prefix, method="annot", block_size=100)
    except ValueError as e:
        assert "annotations" in str(e)
    else:
        raise AssertionError("expected ValueError when annotations missing")


def test_prsresult_repr_is_compact(tmp_path):
    prefix, ss_path, g_te = _simulate(tmp_path, m=300, seed=8)
    res = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=150)
    r = repr(res)
    assert r.startswith("PRSResult(") and "n_samples=" in r
    assert "\n" not in r and len(r) < 200          # no array dump
    # Reading only the GWAS variants must give the same PRS as a full read.
    prefix, ss_path, g_te = _simulate(tmp_path, m=400, seed=7)
    full = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=150,
                           subset_to_sumstats=False)
    sub = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=150,
                          subset_to_sumstats=True)
    np.testing.assert_allclose(full.scores, sub.scores, rtol=1e-6, atol=1e-6)


def test_allele_flip_is_corrected(tmp_path):
    """Swapping A1/A2 in the sumstats must not change the PRS (sign realigned)."""
    prefix, ss_path, g_te = _simulate(tmp_path, m=300, seed=3)
    res0 = run_ldpred3_prs(ss_path, prefix, method="inf", block_size=150)

    flipped = str(tmp_path / "gwas_flip.txt")
    with open(ss_path) as fin, open(flipped, "w") as fout:
        fout.write(fin.readline())               # header
        for line in fin:
            snp, a1, a2, beta, se, n = line.split()
            fout.write(f"{snp}\t{a2}\t{a1}\t{-float(beta):.6g}\t{se}\t{n}\n")
    res1 = run_ldpred3_prs(flipped, prefix, method="inf", block_size=150)

    np.testing.assert_allclose(res0.scores, res1.scores, rtol=1e-6, atol=1e-6)


def test_pipeline_infer_with_compact_ld(tmp_path):
    # --infer now scales with the LD representation: low-rank and banded blocks
    # flow through the streaming sampler and still report h2/p/r2.
    prefix, ss_path, _ = _simulate(tmp_path, m=300, seed=14)
    for kw in ({"ld_lowrank": True}, {"ld_sparse": True}):
        res = run_ldpred3_prs(ss_path, prefix, method="auto", block_size=150,
                              num_iter=60, burn_in=30, seed=1, infer=True,
                              infer_params={"n_chains": 4, "burn_in": 60,
                                            "num_iter": 80}, **kw)
        assert res.inference is not None
        assert 0 < res.inference["h2_est"] < 1.5
        assert res.inference["r2_ci"][0] <= res.inference["r2_est"] \
            <= res.inference["r2_ci"][1]
