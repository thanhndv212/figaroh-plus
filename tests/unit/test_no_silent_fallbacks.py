"""No silent fallbacks (#264, plan item Q2.1).

A metric that cannot be computed is NaN, not a value that reads as a
result (correlation 1.0, improvement 0 %, R² 0). A step that fails raises
instead of returning something other than what was asked for.
"""

import builtins
import math
import sys

import numpy as np
import pytest

from figaroh.identification.base_identification import (
    BaseIdentification,
    _correlation,
)
from figaroh.tools.identification_report import _build_insights
from figaroh.tools.solver import LinearSolver


class _Ident(BaseIdentification):
    def load_trajectory_data(self, data_source=None):
        raise NotImplementedError


# -- correlation -------------------------------------------------------------


@pytest.mark.parametrize(
    "measured, estimated",
    [
        ([1.0], [1.0]),  # one sample
        ([], []),  # none
        ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]),  # constant measurement
        ([1.0, 2.0, 3.0], [0.5, 0.5, 0.5]),  # constant estimate
    ],
)
def test_undefined_correlation_is_nan(measured, estimated):
    assert math.isnan(_correlation(np.array(measured), np.array(estimated)))


def test_defined_correlation_is_pearson():
    x = np.array([1.0, 2.0, 3.0, 5.0])
    assert _correlation(x, 2 * x + 1) == pytest.approx(1.0)
    assert _correlation(x, -x) == pytest.approx(-1.0)


def test_quality_metrics_report_nan_correlation_for_one_sample():
    ident = _Ident.__new__(_Ident)
    ident.dynamic_regressor_base = np.array([[1.0]])
    ident.phi_base = np.array([2.0])
    ident.tau_noised = np.array([2.0])
    ident.tau_identif = np.array([2.0])

    ident._compute_quality_metrics()

    assert math.isnan(ident.correlation)


def test_report_flags_an_uncomputable_correlation():
    result = {"condition number": 10.0, "base parameters names": ["a"]}
    insights = _build_insights(result, None, ["a"], {"improvement_pct": 50.0})
    texts = [i["text"] for i in insights]
    assert any("could not be computed" in t for t in texts)
    assert not any("No issues detected" in t for t in texts)


# -- LinearSolver ------------------------------------------------------------


def test_r_squared_of_a_constant_target_is_nan():
    A = np.array([[1.0], [2.0], [3.0]])
    b = np.zeros(3)
    solver = LinearSolver(method="lstsq")
    solver.solve(A, b)
    assert math.isnan(solver.solver_info["r_squared"])


@pytest.mark.parametrize("method", ["lasso", "elastic_net"])
def test_sparse_methods_without_sklearn_raise(monkeypatch, method):
    real_import = builtins.__import__

    def no_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    for name in [m for m in sys.modules if m == "sklearn" or m.startswith("sklearn.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", no_sklearn)

    A = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ImportError, match=r"figaroh\[sklearn\]"):
        LinearSolver(method=method).solve(A, A @ np.array([1.0, 0.0, 2.0]))


# -- save_results ------------------------------------------------------------


class _FailingManager:
    def save_results(self, **kwargs):
        raise OSError("disk full")


def test_identification_save_failure_propagates(tmp_path):
    ident = _Ident.__new__(_Ident)
    ident.result = {"base parameters values": np.array([1.0])}
    ident.results_manager = _FailingManager()
    with pytest.raises(OSError, match="disk full"):
        ident.save_results(output_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_identification_basic_save_failure_propagates(tmp_path):
    ident = _Ident.__new__(_Ident)
    ident.result = {"base parameters values": np.array([1.0])}
    ident.results_manager = None
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    with pytest.raises(OSError):
        ident.save_results(output_dir=str(blocker))


def _calibration(manager):
    from figaroh.calibration.base_calibration import BaseCalibration

    calib = BaseCalibration.__new__(BaseCalibration)
    calib.result = object()
    calib.results_data = {"x": 1}
    calib.results_manager = manager
    return calib


def test_calibration_save_without_manager_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not saved"):
        _calibration(None).save_results(output_dir=str(tmp_path))


def test_calibration_save_failure_propagates(tmp_path):
    with pytest.raises(OSError, match="disk full"):
        _calibration(_FailingManager()).save_results(output_dir=str(tmp_path))


# -- physical-consistency projection -----------------------------------------

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None


@pytest.mark.skipif(pin is None, reason="Pinocchio not available")
def test_unbuildable_cad_constraints_are_an_error():
    pytest.importorskip("picos")
    from test_data_contract_wiring import _Ident as _WiredIdent
    from test_data_contract_wiring import _trajectory

    model = pin.buildSampleModelManipulator()
    ident = _WiredIdent(model)
    ident.trajectory_to_return = _trajectory(model)
    ident.initialize()
    ident.identif_config.update(
        reconstruction={"enabled": True, "method": "nullspace"},
        physical_consistency={
            "enabled": True,
            "skip_if_feasible": False,
            "cad_constraints": {"source": "urdf", "mass_scale_min": "not-a-number"},
        },
    )
    with pytest.raises(ValueError, match="not-a-number"):
        ident.solve(decimate=False, plotting=False)
