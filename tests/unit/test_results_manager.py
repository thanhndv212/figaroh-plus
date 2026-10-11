# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for ``figaroh.utils.results_manager``:

- The joint-major reshape fix in ``plot_identification_results()`` (a 1D
  torque array is flattened joint-major — all samples of joint 0, then
  joint 1, ... — not sample-major, so it must be reshaped as
  ``(n_joints, -1).T``, not ``(-1, 1)``).
- The ``plot_with_fallback()`` helper shared by every ``Base*.plot_results()``
  method.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # must precede pyplot and anything importing it
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from figaroh.utils.results_manager import (  # noqa: E402
    ResultsManager,
    plot_with_fallback,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotIdentificationResultsReshape:
    def test_1d_joint_major_array_reshaped_per_joint(self):
        n_joints = 3
        n_per_joint = 5
        joint_values = [10.0, 20.0, 30.0]
        # Joint-major flattened: all samples of joint 0, then joint 1, ...
        measured = np.concatenate([np.full(n_per_joint, v) for v in joint_values])
        identified = measured - 1.0

        result = {
            "task type": "identification",
            "torque processed": measured,
            "torque estimated": identified,
            "condition number": 10.0,
            "rmse norm (N/m)": 0.5,
        }
        manager = ResultsManager("identification", "test_robot", result)
        manager.plot_identification_results(
            n_joints=n_joints, joint_names=["j0", "j1", "j2"]
        )

        torque_ax = plt.gcf().axes[0]
        lines = torque_ax.get_lines()
        # Two lines per joint: measured + identified.
        assert len(lines) == n_joints * 2

        measured_lines = lines[0::2]
        for i, line in enumerate(measured_lines):
            assert np.allclose(line.get_ydata(), joint_values[i]), (
                f"joint {i} measured trace does not equal its own slice "
                "-- reshape likely mixed joints together"
            )

    def test_1d_array_without_n_joints_falls_back_to_single_column(self):
        measured = np.array([1.0, 2.0, 3.0])
        identified = np.array([1.1, 2.1, 3.1])
        result = {
            "task type": "identification",
            "torque processed": measured,
            "torque estimated": identified,
        }
        manager = ResultsManager("identification", "test_robot", result)
        manager.plot_identification_results()

        torque_ax = plt.gcf().axes[0]
        lines = torque_ax.get_lines()
        assert len(lines) == 2  # one measured + one identified trace

    def test_already_2d_array_unaffected_by_n_joints(self):
        # 2D input (n_samples, n_joints) must be used as-is regardless of
        # n_joints -- the reshape path is only for 1D input.
        measured = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        identified = measured - 0.1
        result = {
            "task type": "identification",
            "torque processed": measured,
            "torque estimated": identified,
        }
        manager = ResultsManager("identification", "test_robot", result)
        manager.plot_identification_results(n_joints=99)

        torque_ax = plt.gcf().axes[0]
        lines = torque_ax.get_lines()
        assert len(lines) == 4  # 2 joints x (measured + identified)


class TestPlotWithFallback:
    def test_primary_used_when_it_succeeds(self):
        calls = []
        plot_with_fallback(
            primary=lambda: calls.append("primary"),
            fallback=lambda: calls.append("fallback"),
            logger=logging.getLogger("test_results_manager"),
            context="unit-test",
        )
        assert calls == ["primary"]

    def test_fallback_triggers_on_primary_exception(self):
        calls = []

        def primary():
            calls.append("primary")
            raise RuntimeError("boom")

        def fallback():
            calls.append("fallback")

        plot_with_fallback(
            primary=primary,
            fallback=fallback,
            logger=logging.getLogger("test_results_manager"),
            context="unit-test",
        )
        # Primary attempted once, fallback exactly once -- no double-plotting.
        assert calls == ["primary", "fallback"]

    def test_fallback_not_called_when_primary_succeeds(self):
        fallback_calls = []
        plot_with_fallback(
            primary=lambda: None,
            fallback=lambda: fallback_calls.append("fallback"),
            logger=logging.getLogger("test_results_manager"),
            context="unit-test",
        )
        assert fallback_calls == []


def _trajectory_segments(nq, nv, n_samples=4, n_segments=2):
    """Segments whose every column holds a distinct constant, so a plotted
    trace identifies the column it came from (positions 100+i, velocities
    200+i, accelerations 300+i)."""
    t = [np.linspace(s, s + 1, n_samples) for s in range(n_segments)]
    pos = [np.tile(100.0 + np.arange(nq), (n_samples, 1)) for _ in t]
    vel = [np.tile(200.0 + np.arange(nv), (n_samples, 1)) for _ in t]
    acc = [np.tile(300.0 + np.arange(nv), (n_samples, 1)) for _ in t]
    return {"T_F": t, "P_F": pos, "V_F": vel, "A_F": acc}


class TestPlotOptimalTrajectoryResults:
    """#149: positions span ``nq`` columns, velocities/accelerations ``nv``."""

    def _plot(self, trajectories, **kwargs):
        manager = ResultsManager("optimal_trajectory", "test_robot")
        manager.plot_optimal_trajectory_results(
            trajectories=trajectories, condition_number=10.0, **kwargs
        )
        return plt.gcf()

    def test_nq_ne_nv_plots_each_quantity_from_its_own_columns(self, caplog):
        # A continuous joint first (2 position coordinates, 1 velocity), then
        # two revolute joints: nq=4, nv=3. Active: the two revolute joints.
        trajectories = _trajectory_segments(nq=4, nv=3)
        with caplog.at_level(logging.ERROR):
            fig = self._plot(
                trajectories,
                joint_names=["elbow", "wrist"],
                q_indices=[2, 3],
                v_indices=[1, 2],
            )

        assert not caplog.records
        assert len(fig.axes) == 2 * 3  # one row per active joint, no extras
        for row, (iq, iv) in enumerate([(2, 1), (3, 2)]):
            ax_pos, ax_vel, ax_acc = fig.axes[3 * row : 3 * row + 3]
            assert len(ax_pos.get_lines()) == 2  # one trace per segment
            for line in ax_pos.get_lines():
                assert np.allclose(line.get_ydata(), 100.0 + iq)
            for line in ax_vel.get_lines():
                assert np.allclose(line.get_ydata(), 200.0 + iv)
            for line in ax_acc.get_lines():
                assert np.allclose(line.get_ydata(), 300.0 + iv)
        assert "elbow" in fig.axes[0].get_ylabel()
        assert "wrist" in fig.axes[3].get_ylabel()
        assert all(ax.get_xlabel() == "Time (s)" for ax in fig.axes[-3:])

    def test_nq_eq_nv_without_indices_plots_every_column(self, caplog):
        trajectories = _trajectory_segments(nq=3, nv=3)
        with caplog.at_level(logging.ERROR):
            fig = self._plot(trajectories)

        assert not caplog.records
        assert len(fig.axes) == 3 * 3
        for row in range(3):
            for line in fig.axes[3 * row + 1].get_lines():
                assert np.allclose(line.get_ydata(), 200.0 + row)
        assert fig.axes[0].get_ylabel().startswith("Joint 1")

    def test_nq_ne_nv_without_indices_refuses_instead_of_misplotting(self, caplog):
        trajectories = _trajectory_segments(nq=4, nv=3)
        manager = ResultsManager("optimal_trajectory", "test_robot")
        with caplog.at_level(logging.ERROR):
            manager.plot_optimal_trajectory_results(
                trajectories=trajectories, condition_number=10.0
            )

        assert any("nq != nv" in r.getMessage() for r in caplog.records)
        assert not plt.get_fignums()  # nothing half-plotted

    def test_unpaired_indices_are_rejected(self):
        trajectories = _trajectory_segments(nq=4, nv=3)
        with pytest.raises(ValueError, match="together"):
            self._plot(trajectories, q_indices=[2, 3])
