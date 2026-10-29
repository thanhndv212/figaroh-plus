"""Tests for Detmax.main_algo, the DETMAX single-exchange design (#90).

The candidate pool is rank-one information matrices x xᵀ, so a subset's
information matrix is non-singular only when its vectors span the space and
the D-optimal subset of a small pool can be found by enumeration.
"""

import itertools
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

pytest.importorskip("picos")

from figaroh.optimal.base_optimal_calibration import Detmax  # noqa: E402


def _rank_one_pool(n_candidates, dim, seed):
    rng = np.random.default_rng(seed)
    return {
        i: np.outer(x, x) for i, x in enumerate(rng.normal(size=(n_candidates, dim)))
    }


def _crit(pool, subset):
    return np.linalg.det(sum(pool[i] for i in subset)) ** (1 / pool[0].shape[0])


def test_known_d_optimal_subset_is_found_and_stored():
    # Two orthogonal unit directions beat any pair involving the
    # near-collinear or short candidates.
    vectors = [(1, 0), (0, 1), (1, 0.05), (0.3, 0), (0.7, 0.7)]
    pool = {i: np.outer(v, v) for i, v in enumerate(np.array(vectors, float))}
    detmax = Detmax(pool, 2)
    history = detmax.main_algo(init_set=[3, 4])
    assert sorted(detmax.cur_set) == [0, 1]
    assert sorted(detmax.opt_set) == [0, 1]
    assert history[-1] == pytest.approx(1.0)
    assert history[0] == pytest.approx(_crit(pool, [3, 4]))


@pytest.mark.parametrize("seed", range(8))
def test_result_is_a_distinct_local_optimum(seed):
    pool = _rank_one_pool(9, 3, seed)
    detmax = Detmax(pool, 4)
    random.seed(seed)
    history = detmax.main_algo()

    chosen = detmax.cur_set
    assert len(chosen) == 4 and len(set(chosen)) == 4  # no replicate
    assert history[-1] == pytest.approx(_crit(pool, chosen))
    assert all(b > a for a, b in zip(history, history[1:]))
    # DETMAX's stopping rule: after adding the best outside candidate,
    # no removal leaves a larger criterion than the subset already has.
    # (Not every pairwise swap: that is Fedorov's exchange.)
    outside = set(pool) - set(chosen)
    k = max(outside, key=lambda c: _crit(pool, chosen + [c]))
    added = chosen + [k]
    best_after = max(_crit(pool, [i for i in added if i != j]) for j in added)
    assert best_after <= history[-1] * (1 + 1e-9)


def test_reaches_the_enumerated_optimum_from_every_start():
    pool = _rank_one_pool(7, 2, seed=3)
    best = max(_crit(pool, s) for s in itertools.combinations(pool, 2))
    for start in itertools.combinations(pool, 2):
        history = Detmax(pool, 2).main_algo(init_set=list(start))
        assert history[-1] == pytest.approx(best)


def test_singular_start_terminates():
    # Every pair of the first three candidates is collinear: criterion 0.
    vectors = [(1, 0), (2, 0), (3, 0), (0, 1)]
    pool = {i: np.outer(v, v) for i, v in enumerate(np.array(vectors, float))}
    detmax = Detmax(pool, 2)
    history = detmax.main_algo(init_set=[0, 1])
    assert sorted(detmax.cur_set) == [2, 3]
    assert history[-1] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "n_chosen, init_set",
    [(0, None), (5, None), (2, [0, 0]), (2, [0]), (2, [0, 9])],
)
def test_invalid_size_or_start_raises(n_chosen, init_set):
    pool = _rank_one_pool(5, 2, seed=0)
    with pytest.raises(ValueError):
        Detmax(pool, n_chosen).main_algo(init_set=init_set)
