"""Calibration outlier exclusion (#98).

Samples whose position error exceeds ``outlier_eps`` (metres, from
``parameters.outlier_threshold``) are excluded and the rest refitted; the
evaluation covers the kept samples and names the excluded ones.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin  # noqa: F401
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from test_calibration_estimation import _calibrator

OUTLIER = 7


def _with_outlier(model, threshold):
    calib = _calibrator(model, "joint_offset", {}, n=30)
    n = calib.calib_config["NbSample"]
    calib.PEE_measured[0 * n + OUTLIER] += 0.10  # 10 cm on x
    calib.calib_config["outlier_eps"] = threshold
    calib.create_param_list()
    return calib


def test_gross_outlier_is_excluded_and_refitted(tiago_model):
    calib = _with_outlier(tiago_model, 0.02)
    result, excluded = calib.solve_optimisation()
    assert excluded == [OUTLIER]

    evaluation = calib._evaluate_solution(result, excluded)
    assert evaluation["excluded_samples"] == [OUTLIER]
    assert evaluation["excluded_sample_errors"][0] > 0.05
    assert evaluation["outlier_threshold"] == 0.02
    assert evaluation["n_outliers"] == 1
    # the kept samples fit to the noise level (0.2 mm per component)
    assert evaluation["rmse"] < 1e-3
    # the excluded sample's rows are zero and do not count as observations
    n_meas = len(calib.PEE_measured)
    assert calib.residual_dof == n_meas - 3 - len(result.x)


def test_without_threshold_the_outlier_stays_in_the_fit(tiago_model):
    calib = _with_outlier(tiago_model, None)
    result, excluded = calib.solve_optimisation()
    assert excluded == []
    evaluation = calib._evaluate_solution(result, excluded)
    assert evaluation["excluded_samples"] == []
    assert evaluation["rmse"] > 5e-3  # the 10 cm error dominates


def test_explicit_threshold_overrides_config(tiago_model):
    calib = _with_outlier(tiago_model, None)
    _, excluded = calib.solve_optimisation(outlier_threshold=0.02)
    assert excluded == [OUTLIER]


def test_clean_data_excludes_nothing(tiago_model):
    calib = _calibrator(tiago_model, "joint_offset", {}, n=30)
    calib.calib_config["outlier_eps"] = 0.02
    calib.create_param_list()
    _, excluded = calib.solve_optimisation()
    assert excluded == []


def test_one_fit_cannot_exclude(tiago_model):
    calib = _with_outlier(tiago_model, 0.02)
    _, excluded = calib.solve_optimisation(max_iterations=1)
    assert excluded == []


def test_position_errors_use_position_components_of_every_marker(tiago_model):
    calib = _calibrator(tiago_model, "joint_offset", {}, n=4)
    cfg = calib.calib_config
    # contact-like: z, roll, pitch measured; two markers
    cfg["measurability"] = [False, False, True, True, True, False]
    residuals = np.zeros((2, 3, 4))
    residuals[0, 1, 2] = 1.0  # roll of sample 2: not a position
    residuals[1, 0, 3] = 0.03  # z of sample 3, second marker
    calib.PEE_measured = residuals.ravel()
    errors = calib._sample_position_errors(residuals.ravel())
    np.testing.assert_allclose(errors, [0.0, 0.0, 0.0, 0.03])
    rows = calib._sample_rows([3])
    assert list(rows) == [3, 7, 11, 15, 19, 23]

    cfg["measurability"] = [False, False, False, True, True, True]
    assert calib._sample_position_errors(np.zeros(12)) is None
