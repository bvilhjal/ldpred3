"""Coalescent genotype simulation: the mutation-rate density lever."""

import numpy as np
import pytest


def test_mutation_rate_is_the_density_lever():
    """On a fixed segment, raising mut_rate finds more variants in the *same*
    recombination structure (array -> imputed -> WGS), deterministically."""
    pytest.importorskip("msprime")
    from ldpred3.simulate import simulate_genotypes_by_mutation_rate

    kw = dict(recomb_rate=1e-8, Ne=10_000, min_maf=0.01, seed=7)
    G_lo = simulate_genotypes_by_mutation_rate(500, 3e5, mut_rate=0.5e-8, **kw)
    G_hi = simulate_genotypes_by_mutation_rate(500, 3e5, mut_rate=2.0e-8, **kw)

    # int8 dosages, n rows preserved, valid genotype range
    assert G_lo.dtype == np.int8 and G_lo.shape[0] == 500
    assert set(np.unique(G_lo)).issubset({0, 1, 2})
    # denser mutation rate -> strictly more common SNPs on the same segment
    assert G_hi.shape[1] > G_lo.shape[1]
    # same seed + params -> deterministic
    again = simulate_genotypes_by_mutation_rate(500, 3e5, mut_rate=0.5e-8, **kw)
    assert np.array_equal(G_lo, again)
