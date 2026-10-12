"""Compatibility paths deprecated in 1.0 (removal in 2.0) warn; replacements do not (#185)."""

import warnings
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from figaroh.tools import robotvisualization as rv
from figaroh.tools.qrdecomposition import QR_pivoting, QRDecomposer, double_QR
from figaroh.tools.robotcollisions import CollisionManager, CollisionWrapper
from figaroh.utils import results_manager as rm


def _no_deprecation(record):
    return [w for w in record if issubclass(w.category, DeprecationWarning)]


def _qr_inputs():
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal(20),
        rng.standard_normal((20, 5)),
        [f"p{i}" for i in range(5)],
    )


def test_qr_pivoting_warns():
    tau, W, params = _qr_inputs()
    with pytest.warns(DeprecationWarning, match="deprecated in 1.0.*2.0.*QRDecomposer"):
        QR_pivoting(tau, W, params)


def test_double_qr_warns():
    tau, W, params = _qr_inputs()
    with pytest.warns(DeprecationWarning, match="deprecated in 1.0.*2.0.*QRDecomposer"):
        double_QR(tau, W, params)


def test_qr_replacement_does_not_warn():
    tau, W, params = _qr_inputs()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        QRDecomposer().decompose_with_pivoting(tau, W, params)
        QRDecomposer().double_decomposition(tau, W, params)
    assert not _no_deprecation(rec)


def _robot():
    robot = Mock()
    robot.model.createData.return_value = Mock()
    robot.geom_model.createData.return_value = Mock()
    return robot


def test_collision_wrapper_warns_on_construction():
    with pytest.warns(DeprecationWarning, match="CollisionManager"):
        CollisionWrapper(_robot())


def test_collision_manager_does_not_warn():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        CollisionManager(_robot())
    assert not _no_deprecation(rec)


@pytest.mark.parametrize(
    "name, args",
    [
        ("place", (Mock(), "n", Mock())),
        ("display_COM", (Mock(), Mock(), Mock(), np.zeros(3), [1])),
        ("display_axes", (Mock(), Mock(), Mock(), np.zeros(3))),
        (
            "rotation_matrix_from_vectors",
            (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
        ),
        ("display_force", (Mock(), Mock(), Mock())),
        (
            "display_bounding_boxes",
            (Mock(), Mock(), Mock(), np.zeros(3), np.zeros(3), np.ones(3), [1]),
        ),
        ("display_joints", (Mock(), Mock(), Mock(), np.zeros(3))),
    ],
)
def test_visualization_wrappers_warn(monkeypatch, name, args):
    monkeypatch.setattr(rv, "RobotVisualizer", MagicMock())
    monkeypatch.setattr(rv.pin, "SE3ToXYZQUATtuple", lambda M: (0,) * 7)
    with pytest.warns(DeprecationWarning, match=f"{name} is deprecated in 1.0"):
        getattr(rv, name)(*args)


def test_visualizer_class_method_does_not_warn():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        viz = rv.RobotVisualizer(Mock(), Mock(), Mock())
        viz._rotation_from_vectors(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    assert not _no_deprecation(rec)


def test_results_manager_wrappers_warn(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(rm, "ResultsManager", lambda *a, **k: manager)
    with pytest.warns(
        DeprecationWarning, match="plot_calibration_results.*ResultsManager"
    ):
        rm.plot_calibration_results(None, None, None)
    with pytest.warns(
        DeprecationWarning, match="plot_identification_results.*ResultsManager"
    ):
        rm.plot_identification_results(None, None, None)
    with pytest.warns(DeprecationWarning, match="save_results.*ResultsManager"):
        rm.save_results({}, "calibration")


def test_results_manager_methods_do_not_warn(monkeypatch):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        rm.ResultsManager("calibration", "robot")
    assert not _no_deprecation(rec)


class _FakeIdent:
    from figaroh.identification.base_identification import BaseIdentification

    _apply_physical_consistency_if_enabled = (
        BaseIdentification._apply_physical_consistency_if_enabled
    )

    def __init__(self, cfg):
        self.identif_config = cfg


def test_flat_physical_consistency_key_warns():
    fake = _FakeIdent({"physical_consistency_enabled": False})
    with pytest.warns(
        DeprecationWarning,
        match="physical_consistency_enabled.*physical_consistency.enabled",
    ):
        fake._apply_physical_consistency_if_enabled({})


def test_nested_physical_consistency_key_does_not_warn():
    fake = _FakeIdent({"physical_consistency": {"enabled": False}})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fake._apply_physical_consistency_if_enabled({})
    assert not _no_deprecation(rec)
