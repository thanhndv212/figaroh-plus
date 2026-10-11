"""Base-parameter selection does not depend on column order or noise (#116)."""

import numpy as np
import pytest

from figaroh.tools.qrdecomposition import QRDecomposer, deterministic_column_pivots


def _tied_regressor(m=200, k=6, extra=4, seed=0):
    """Rank-k regressor whose independent columns have equal norms (exact
    ties for a pivoted QR) plus `extra` columns that are sign-combinations of
    them, also of norm ~1 (more ties for the first pivot)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((m, k)))
    cols = [Q[:, i] for i in range(k)]
    for j in range(extra):
        a, b = j % k, (j + 1) % k
        cols.append((Q[:, a] + Q[:, b]) / np.sqrt(2.0))
    W = np.column_stack(cols)
    names = [f"p{i:02d}" for i in range(k + extra)]
    return W, names


def _selected(decomposer, W, names, tau, method):
    res = decomposer.decompose(W, names, tau=tau, method=method)
    return res, sorted(names[i] for i in res.base_indices)


@pytest.mark.parametrize("method", ["pivoting", "double"])
def test_selection_ignores_column_order_and_noise(method):
    W, names = _tied_regressor()
    tau = W @ np.arange(1.0, W.shape[1] + 1)
    dec = QRDecomposer(tolerance=1e-8)
    ref_res, ref = _selected(dec, W, names, tau, method)
    ref_pred = ref_res.W_b @ ref_res.phi_b

    rng = np.random.default_rng(1)
    for trial in range(10):
        perm = rng.permutation(len(names))
        Wp = W[:, perm] + 1e-12 * rng.standard_normal(W.shape)
        res, got = _selected(dec, Wp, [names[i] for i in perm], tau, method)
        assert got == ref, f"trial {trial}"
        np.testing.assert_allclose(res.W_b @ res.phi_b, ref_pred, atol=1e-8)


def test_pivots_are_independent_of_input_order():
    W, names = _tied_regressor(seed=3)
    P, rank = deterministic_column_pivots(W, names)
    ref = [names[i] for i in P[:rank]]
    assert rank == 6
    perm = np.random.default_rng(5).permutation(len(names))
    P2, rank2 = deterministic_column_pivots(W[:, perm], [names[i] for i in perm])
    assert rank2 == rank
    assert [names[perm[i]] for i in P2[:rank2]] == ref


def test_pivots_pick_largest_residual_column():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 4)) * np.array([1.0, 5.0, 2.0, 0.5])
    P, rank = deterministic_column_pivots(W)
    assert rank == 4
    assert P[0] == 1  # largest norm first, as with LAPACK pivoting


def test_pivots_rank_deficient_and_empty():
    W = np.zeros((10, 3))
    P, rank = deterministic_column_pivots(W)
    assert rank == 0 and sorted(P.tolist()) == [0, 1, 2]
    P, rank = deterministic_column_pivots(np.zeros((10, 0)))
    assert rank == 0 and P.size == 0
