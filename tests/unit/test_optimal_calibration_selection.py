"""Tests for BaseOptimalCalibration's configuration-count selection rules.

Covers the two interchangeable strategies calculate_optimal_configurations()
can use to turn a SOCP weight ranking into a discrete configuration subset:
_select_by_threshold() (the original eps_opt cutoff) and _select_by_iroc()
(the new automatic minimal-count selection via D-optimality plateau
detection, see examples/talos_table_contact's report, Integration plan
step 8).

Both are tested directly against a synthetic candidate-configuration pool
(random SPD information matrices), bypassing BaseOptimalCalibration.__init__
(which needs a robot model + config file) since neither method touches
anything but self.w_dict_sort, self._subX_list and self.minNbChosen.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

pytest.importorskip("picos")
pytest.importorskip("cvxopt")

from figaroh.optimal.base_optimal_calibration import (
    BaseOptimalCalibration,
)  # noqa: E402


def _make_selector(subX_list, weights, min_chosen=1):
    """A bare BaseOptimalCalibration with just the attributes the
    selection helpers read, no robot/config/regressor machinery."""
    obj = object.__new__(BaseOptimalCalibration)
    obj._subX_list = subX_list
    obj.w_dict_sort = dict(
        reversed(sorted(dict(enumerate(weights)).items(), key=lambda kv: kv[1]))
    )
    obj.minNbChosen = min_chosen
    return obj


def _spanning_and_redundant_pool(n_dims=4, n_redundant=16, rng=None):
    """4 matrices that each dominate a distinct dimension (highly
    informative, in rank order), followed by many near-duplicates of the
    first one (near-zero marginal information once it's already
    included) -- built so the cumulative D-optimality criterion should
    rise sharply over the first n_dims configs and then plateau.
    """
    rng = rng or np.random.default_rng(0)
    matrices = []
    spanning_dir = None
    for i in range(n_dims):
        v = np.zeros(n_dims)
        v[i] = 10.0
        if i == 0:
            spanning_dir = v
        matrices.append(np.outer(v, v) + 1e-3 * np.eye(n_dims))
    for _ in range(n_redundant):
        v = spanning_dir * (1.0 + rng.normal(0, 1e-4))
        matrices.append(np.outer(v, v) + 1e-3 * np.eye(n_dims))
    return matrices


def _rank_order_weights(n):
    """Strictly decreasing weights, matching a pool already ordered from
    most to least informative -- i.e. what a real SOCP solve would
    produce (ties are vanishingly unlikely with a continuous solver, but
    would silently scramble the intended rank order in this synthetic
    pool if used here)."""
    return list(np.linspace(1.0, 0.05, n))


class TestSelectByThreshold:
    def test_keeps_only_weights_above_eps(self):
        subX = _spanning_and_redundant_pool()
        weights = [0.3, 0.3, 0.3, 0.1] + [1e-8] * 16
        selector = _make_selector(subX, weights, min_chosen=1)

        chosen = selector._select_by_threshold()

        assert selector.eps_opt == 1e-5
        assert set(chosen) == {0, 1, 2, 3}


class TestSelectByIroc:
    def test_stops_before_exhausting_redundant_candidates(self):
        subX = _spanning_and_redundant_pool()
        weights = _rank_order_weights(len(subX))
        selector = _make_selector(subX, weights, min_chosen=1)

        chosen = selector._select_by_iroc(rel_tol=1e-3, patience=3)

        assert len(chosen) >= selector.minNbChosen
        assert len(chosen) < len(subX), (
            "IROC selection should plateau well before the redundant "
            "tail of near-duplicate candidates is exhausted."
        )
        # The 4 genuinely spanning directions should all be included.
        assert {0, 1, 2, 3}.issubset(set(chosen))

    def test_o1_history_peaks_in_the_spanning_region(self):
        """O1(k) = det(sum of top-k weighted info matrices)^(1/n) / sqrt(k)
        -- the sqrt(k) normalization (matching plot()'s own formula)
        means it is *not* generally monotonic: it climbs sharply while
        each new configuration adds genuinely new information, then
        actually turns over once further configurations are redundant
        (same information, but the sqrt(k) penalty keeps growing). That
        turnover is exactly what makes elbow/plateau detection meaningful
        here, so this checks the shape directly: the criterion should
        peak at or shortly after the 4 genuinely spanning configurations,
        not out in the redundant tail.
        """
        n_dims = 4
        subX = _spanning_and_redundant_pool(n_dims=n_dims)
        weights = _rank_order_weights(len(subX))
        selector = _make_selector(subX, weights, min_chosen=1)

        selector._select_by_iroc(rel_tol=1e-3, patience=3)

        history = selector.o1_history
        assert len(history) >= n_dims
        peak_k = int(np.argmax(history)) + 1  # history[i] <-> k = i + 1
        assert peak_k <= n_dims + 1, (
            f"expected the D-optimality criterion to peak near the "
            f"{n_dims} spanning configurations, peaked at k={peak_k}"
        )

    def test_respects_min_chosen_floor(self):
        subX = _spanning_and_redundant_pool()
        weights = _rank_order_weights(len(subX))
        min_chosen = 8
        selector = _make_selector(subX, weights, min_chosen=min_chosen)

        chosen = selector._select_by_iroc(rel_tol=1e-3, patience=3)

        assert len(chosen) >= min_chosen

    def test_patience_delays_the_stop(self):
        """A larger patience should never select fewer configurations
        than a smaller one on the same pool (it requires a longer flat
        run before giving up)."""
        subX = _spanning_and_redundant_pool()
        weights = _rank_order_weights(len(subX))

        selector_low = _make_selector(subX, weights, min_chosen=1)
        chosen_low = selector_low._select_by_iroc(rel_tol=1e-3, patience=1)

        selector_high = _make_selector(subX, weights, min_chosen=1)
        chosen_high = selector_high._select_by_iroc(rel_tol=1e-3, patience=8)

        assert len(chosen_high) >= len(chosen_low)
