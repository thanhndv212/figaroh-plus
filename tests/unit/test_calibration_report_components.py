"""Calibration report rows follow ``measurability`` (#100).

Residual rows are the measured pose components in order. A table-contact
calibration measuring ``[z, roll, pitch]`` must be reported as one
position row (mm) and two orientation rows (deg), with position and
orientation aggregates over those components only, and no ``inf``
uncertainty when there are no residual degrees of freedom.
"""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from figaroh.calibration.base_calibration import (
    BaseCalibration,
    _measured_components,
)
from figaroh.tools.report import _build_insights, _per_dof_section

CONTACT = [False, False, True, True, True, False]  # z, roll, pitch


def _calibration(measurability, n_samples, residuals_2d):
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.calib_config = {
        "measurability": measurability,
        "calibration_index": sum(measurability),
        "NbSample": n_samples,
        "param_name": ["p0", "p1"],
    }
    rng = np.random.default_rng(0)
    calib.PEE_measured = rng.normal(size=residuals_2d.size)
    stats = calib._compute_per_dof_stats(
        residuals_2d.flatten(), residuals_2d.shape[0], n_samples
    )
    calib.evaluation_metrics = {
        "optimization_success": True,
        "n_iterations": 3,
        "cost": 0.0,
        "n_outliers": 0,
        "outlier_percentage": 0.0,
        "per_dof_stats": stats,
        "param_stdev": [0.1, 0.2],
        "param_stddev_percentage": [1.0, 2.0],
        "residual_dof": 10,
    }
    calib._compute_validation_metrics = lambda: None
    return calib


def test_components_follow_measurability():
    comps = _measured_components({"measurability": CONTACT, "calibration_index": 3})
    assert comps["names"] == ["Z (mm)", "rx (deg)", "ry (deg)"]
    assert comps["pos_rows"] == [0] and comps["orient_rows"] == [1, 2]
    np.testing.assert_allclose(comps["scales"], [1000.0, 180 / np.pi, 180 / np.pi])


def test_contact_aggregates_use_measured_components_only():
    n = 50
    z = np.full(n, 0.5e-3)  # 0.5 mm
    roll = np.full(n, np.deg2rad(0.1))
    pitch = np.full(n, np.deg2rad(0.2))
    calib = _calibration(CONTACT, n, np.vstack([z, roll, pitch]))
    stats = calib.evaluation_metrics["per_dof_stats"]
    assert stats["dof_names"] == ["Z (mm)", "rx (deg)", "ry (deg)"]
    np.testing.assert_allclose(stats["rmse"], [0.5, 0.1, 0.2])
    overall = stats["overall"]
    assert overall["pos_rmse_mm"] == pytest.approx(0.5)
    assert overall["orient_rmse_deg"] == pytest.approx(np.hypot(0.1, 0.2))


def test_unmeasured_kind_is_nan_and_reported_as_such(capsys):
    n = 20
    position_only = [True, True, True, False, False, False]
    calib = _calibration(position_only, n, np.full((3, n), 1e-3))
    overall = calib.evaluation_metrics["per_dof_stats"]["overall"]
    assert np.isnan(overall["orient_rmse_deg"])
    calib.print_quality_report()
    out = capsys.readouterr().out
    assert "Orientation RMSE:  not measured" in out
    html = _per_dof_section(calib.evaluation_metrics["per_dof_stats"])
    assert "Position RMSE" in html and "Orientation RMSE" not in html


def test_terminal_report_labels_contact_rows(capsys):
    n = 10
    calib = _calibration(CONTACT, n, np.full((3, n), 1e-3))
    calib.print_quality_report()
    out = capsys.readouterr().out
    assert "Z (mm)" in out and "rx (deg)" in out and "ry (deg)" in out
    assert "X (mm)" not in out


def test_verify_selects_metrics_by_measured_kind():
    calib = _calibration(CONTACT, 10, np.full((3, 10), 1e-3))
    calib.results_data = {
        "validation_metrics": {
            "pos_rmse_calibrated_mm": 0.5,
            "orient_rmse_calibrated_deg": 0.2,
        }
    }
    calib._val_available = True
    calib.evaluation_metrics.update(condition_number=10.0, rmse=1e-3)
    verdict = calib.verify(scope="execution")
    assert verdict.metrics["position_rmse_mm"] == 0.5
    # three measured components, two of them rotations
    assert verdict.metrics["orientation_rmse_deg"] == 0.2
    assert verdict.compat["dof_names"] == ["Z (mm)", "rx (deg)", "ry (deg)"]


def test_zero_residual_dof_is_not_estimable(capsys):
    n = 4
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.calib_config = {"NbSample": n, "calibration_index": 1}
    calib.PEE_measured = np.zeros(n)
    result = SimpleNamespace(x=np.ones(n), fun=np.zeros(n), jac=np.eye(n))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero RuntimeWarning
        calib.calc_stddev(result)
    assert calib.residual_dof == 0
    assert all(np.isnan(calib.std_dev)) and calib._C_param is None

    eval_ = {"residual_dof": 0, "param_stdev": calib.std_dev}
    texts = [i["text"] for i in _build_insights(eval_, n, [], None)]
    assert any("not estimable" in t for t in texts)
