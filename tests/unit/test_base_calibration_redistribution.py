"""Tests for BaseCalibration.redistribute_parameters.

Exercises the method in isolation (via BaseCalibration.__new__, bypassing
__init__'s robot/config-file requirements) since it only reads
self._C_param / self.var_ / self.calib_config -- no full calibration
pipeline needed. It replaces the implicit "1 representative base parameter
gets the fitted value, the rest of its redundant group stays at 0" deploy
behavior with a weighted minimum-norm lift across the whole group
(figaroh-plus#111); with equal expected sizes it is the Moore-Penrose
minimum-norm solution.
"""

import numpy as np
import pytest

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.utils.error_handling import CalibrationError
from figaroh.tools.qrdecomposition import (
    redistribute_min_norm,
    propagate_covariance_min_norm,
)


def _bare_calibration():
    """A BaseCalibration instance with __init__ skipped -- just enough
    attributes for redistribute_parameters() to run."""
    return BaseCalibration.__new__(BaseCalibration)


def _make_reduced_case(rng):
    """A small, known-redundant base-mapping case: 2 standard params
    (equal-coefficient duplicates) reduced to 1 base param."""
    M = np.array([[1.0, 1.0]])  # phi_base = theta_0 + theta_1
    full_names = ["d_px_joint1", "d_px_joint2"]
    row_names = ["d_px_joint1"]
    phi_base = np.array([10.0])
    C_base = np.array([[0.25]])  # some nonzero variance on the base param
    return M, full_names, row_names, phi_base, C_base


class TestRedistributeParameters:
    def test_raises_if_solve_not_run(self):
        calib = _bare_calibration()
        calib.calib_config = {}
        with pytest.raises(CalibrationError, match="solve"):
            calib.redistribute_parameters()

    def test_raises_if_var_missing(self):
        calib = _bare_calibration()
        calib.calib_config = {}
        calib._C_param = np.eye(1)
        # var_ deliberately not set
        with pytest.raises(CalibrationError, match="solve"):
            calib.redistribute_parameters()

    def test_raises_if_base_mapping_absent(self):
        calib = _bare_calibration()
        calib._C_param = np.eye(1)
        calib.var_ = np.array([1.0])
        calib.calib_config = {}  # create_param_list() never ran
        with pytest.raises(CalibrationError, match="create_param_list"):
            calib.redistribute_parameters()

    def test_redistributes_and_matches_direct_computation(self):
        rng = np.random.default_rng(0)
        M, full_names, row_names, phi_base, C_base = _make_reduced_case(rng)

        calib = _bare_calibration()
        calib.var_ = phi_base  # base_mapping_slice selects the whole vector
        calib._C_param = C_base
        calib.calib_config = {
            "base_mapping_matrix": M,
            "base_mapping_param_names": full_names,
            "base_mapping_row_names": row_names,
            "base_mapping_slice": (0, 1),
            "param_name": list(row_names),
        }

        result = calib.redistribute_parameters()

        expected_theta = redistribute_min_norm(M, phi_base)
        expected_C = propagate_covariance_min_norm(M, C_base)
        expected_std = np.sqrt(np.abs(np.diag(expected_C)))

        assert set(result.keys()) == set(full_names)
        for i, name in enumerate(full_names):
            assert result[name]["value"] == pytest.approx(expected_theta[i])
            assert result[name]["std_dev"] == pytest.approx(expected_std[i])

        # The known equal-coefficient duplicate pair: min-norm splits the
        # fitted value evenly, not one-hot (100%/0%, today's implicit
        # behavior for the parameter absent from param_name).
        assert "d_px_joint2" not in row_names  # confirms it's the eliminated one
        assert result["d_px_joint1"]["value"] == pytest.approx(5.0)
        assert result["d_px_joint2"]["value"] == pytest.approx(5.0)

    def test_respects_base_mapping_slice_offset(self):
        """base_mapping_slice must be honored positionally, not assumed
        to start at index 0 (e.g. elastic-gain params can precede the
        base-mapping block in calib_config['param_name'])."""
        rng = np.random.default_rng(1)
        M, full_names, row_names, phi_base, C_base = _make_reduced_case(rng)

        calib = _bare_calibration()
        # var_ has an unrelated leading entry (e.g. an elastic-gain param)
        # before the base-mapping block starts at index 1.
        calib.var_ = np.concatenate([[999.0], phi_base])
        full_C = np.array([[1.0, 0.0], [0.0, C_base[0, 0]]])
        calib._C_param = full_C
        calib.calib_config = {
            "base_mapping_matrix": M,
            "base_mapping_param_names": full_names,
            "base_mapping_row_names": row_names,
            "base_mapping_slice": (1, 2),
            "param_name": ["k_RZ_joint0"] + list(row_names),
        }

        result = calib.redistribute_parameters()

        assert result["d_px_joint1"]["value"] == pytest.approx(5.0)
        assert result["d_px_joint2"]["value"] == pytest.approx(5.0)


class TestWeightedLift:
    """Weighted lift with frame and dropped rows held at 0 (#111)."""

    def _calib(self, M, full_names, rows_full, kept, frame_rows, x, C):
        calib = _bare_calibration()
        calib.var_ = np.asarray(x, float)
        calib._C_param = np.asarray(C, float)
        keep = [rows_full.index(r) for r in kept]
        calib.calib_config = {
            "base_mapping_matrix": np.asarray(M)[keep],
            "base_mapping_matrix_full": np.asarray(M),
            "base_mapping_param_names": full_names,
            "base_mapping_row_names": list(kept),
            "base_mapping_row_names_full": list(rows_full),
            "base_frame_row_names": list(frame_rows),
            "base_mapping_slice": (0, len(kept)),
            "param_name": list(kept),
        }
        return calib

    def test_split_follows_expected_sizes(self):
        """A translation (1 mm) and a rotation (2 mrad) in one group share
        the fitted value in proportion to their prior variances."""
        calib = self._calib(
            [[1.0, 1.0]],
            ["d_px_j1", "d_phix_j2"],
            ["d_px_j1"],
            ["d_px_j1"],
            [],
            [10.0],
            [[0.25]],
        )
        r = calib.redistribute_parameters()
        assert r["d_px_j1"]["value"] == pytest.approx(2.0)  # 1e-6 / 5e-6
        assert r["d_phix_j2"]["value"] == pytest.approx(8.0)  # 4e-6 / 5e-6

    def test_frame_and_dropped_rows_held_at_zero(self):
        M = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        names = ["d_px_a", "d_px_b", "d_px_c", "d_px_d"]
        rows_full = ["d_px_a", "d_px_b", "d_px_c"]
        # row a carries the base frame, row b was dropped by the fit
        calib = self._calib(
            M, names, rows_full, ["d_px_a", "d_px_c"], ["d_px_a"], [0.3, 7.0], np.eye(2)
        )
        r = calib.redistribute_parameters()
        theta = np.array([r[n]["value"] for n in names])
        np.testing.assert_allclose(M @ theta, [0.0, 0.0, 7.0], atol=1e-12)
        # the frame row's fitted value (0.3) is not distributed onto joints
        assert r["d_px_a"]["value"] + r["d_px_b"]["value"] == pytest.approx(0.0)

    def test_unseen_directions_have_zero_conditional_std(self):
        calib = self._calib(
            [[1.0, 1.0]],
            ["d_px_j1", "d_px_j2"],
            ["d_px_j1"],
            ["d_px_j1"],
            ["d_px_j1"],
            [4.0],
            [[0.5]],
        )
        r = calib.redistribute_parameters()
        assert r["d_px_j1"] == {"value": 0.0, "std_dev": 0.0}
