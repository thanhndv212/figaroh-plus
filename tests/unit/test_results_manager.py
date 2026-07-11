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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from figaroh.utils.results_manager import ResultsManager, plot_with_fallback


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
        measured = np.concatenate(
            [np.full(n_per_joint, v) for v in joint_values]
        )
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
