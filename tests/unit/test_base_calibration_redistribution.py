"""Tests for BaseCalibration.redistribute_parameters.

Exercises the method in isolation (via BaseCalibration.__new__, bypassing
__init__'s robot/config-file requirements) since it only reads
self._C_param / self.var_ / self.calib_config -- no full calibration
pipeline needed. See TIAGO_CALIBRATION_ANALYSIS.md §8 for why this method
exists: it replaces the implicit "1 representative base parameter gets the
fitted value, the rest of its redundant group stays at 0" deploy behavior
with minimum-norm redistribution across the whole group.
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
