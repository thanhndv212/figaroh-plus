"""Parameter standard errors from BaseCalibration.calc_stddev (#107).

``least_squares``' ``result.cost`` is ``0.5 * sum(fun**2)``; the residual
variance is ``sum(r**2) / (m - n)`` over the measurement residuals.
"""

from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares

from figaroh.calibration.base_calibration import BaseCalibration


def _bare(n_meas):
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.calib_config = {"NbSample": n_meas, "calibration_index": 1}
    calib.PEE_measured = np.zeros(n_meas)
    return calib


def test_variance_uses_measurement_rows_only():
    rng = np.random.default_rng(0)
    m, n = 40, 3
    J_meas = rng.normal(size=(m, n))
    r_meas = rng.normal(scale=2e-3, size=m)
    reg_rows = np.sqrt(0.01) * np.eye(n)  # e.g. a subclass regulariser
    result = SimpleNamespace(
        x=np.ones(n),
        fun=np.concatenate([r_meas, 0.5 * np.ones(n)]),
        jac=np.vstack([J_meas, reg_rows]),
    )
    result.cost = 0.5 * np.sum(result.fun**2)
    calib = _bare(m)

    calib.calc_stddev(result)

    sigma_sq = np.sum(r_meas**2) / (m - n)
    expected = sigma_sq * np.linalg.pinv(result.jac.T @ result.jac)
    np.testing.assert_allclose(calib._C_param, expected, rtol=1e-12)
    np.testing.assert_allclose(calib.std_dev, np.sqrt(np.diag(expected)), rtol=1e-12)


def test_standard_errors_match_sampling_spread():
    """Linear model, known noise: reported SE ~ spread of the estimates."""
    rng = np.random.default_rng(1)
    m, n, sigma = 60, 4, 1.5e-3
    J = rng.normal(size=(m, n))
    x_true = np.array([0.02, -0.01, 0.005, 0.03])
    estimates, reported = [], []
    for _ in range(300):
        y = J @ x_true + rng.normal(scale=sigma, size=m)
        result = least_squares(lambda x: J @ x - y, np.zeros(n), method="lm")
        calib = _bare(m)
        calib.calc_stddev(result)
        estimates.append(result.x)
        reported.append(calib.std_dev)
    spread = np.std(estimates, axis=0)
    mean_reported = np.mean(reported, axis=0)
    np.testing.assert_allclose(mean_reported, spread, rtol=0.15)
    np.testing.assert_allclose(
        mean_reported, sigma * np.sqrt(np.diag(np.linalg.inv(J.T @ J))), rtol=0.05
    )
    assert np.all(np.array(calib.std_pctg) > 0)
