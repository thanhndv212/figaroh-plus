"""A calibration whose solver failed is not exported (figaroh-plus#182, G-1).

A fit forced to report ``success=False`` records stage ``fit`` failed and sets
``STATUS = "FAILED"``; ``joint_corrections()`` (the URDF export gateway) and
``export_geometric_calibration_yaml`` then raise and record stage ``export``
failed. ``allow_failed_fit=True`` exports for diagnosis, with a warning and a
``fallback`` export stage. A converged fit is unaffected.
"""

import logging

import numpy as np
import pytest

try:
    import pinocchio as pin  # noqa: F401
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

import figaroh.calibration.base_calibration as bc
from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.calibration.calibration_tools import (
    calc_updated_fkm,
    random_joint_configuration,
)
from figaroh.calibration.config import unified_to_legacy_config
from figaroh.tools.geometric_calibration_export import (
    build_geometric_calibration,
    export_geometric_calibration_yaml,
)
from figaroh.utils.error_handling import CalibrationError


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


class _Calib(BaseCalibration):
    def cost_function(self, var):
        return (
            calc_updated_fkm(
                self.model, self.data, var, self.q_measured, self.calib_config
            )
            - self.PEE_measured
        )


def _stage(calib, name):
    return next(s for s in calib.stages if s.stage == name)


@pytest.fixture
def make_calib(tiago_model):
    def make():
        unified = {
            "joints": {},
            "kinematics": {
                "base_frame": "universe",
                "tool_frame": "wrist_ft_tool_link",
            },
            "parameters": {"calibration_level": "joint_offset", "estimation": {}},
            "measurements": {
                "markers": [
                    {
                        "reference_joint": "arm_7_joint",
                        "measurable_dof": [True] * 3 + [False] * 3,
                    }
                ]
            },
            "data": {"source_file": "unused.csv"},
        }
        cfg = unified_to_legacy_config(_Robot(tiago_model), unified)
        cfg.update(known_baseframe=False, known_tipframe=False, random_seed=0)
        cfg["coeff_regularize"] = 0.0
        rng = np.random.default_rng(0)
        q = np.array([random_joint_configuration(tiago_model, rng) for _ in range(30)])
        cfg["NbSample"] = len(q)
        calib = _Calib.__new__(_Calib)
        calib.model, calib.data, calib.calib_config = (
            tiago_model,
            tiago_model.createData(),
            cfg,
        )
        calib.q_measured = q
        calib.del_list_ = []
        calib.create_param_list()
        truth = np.random.default_rng(1).normal(scale=0.01, size=len(cfg["param_name"]))
        calib.PEE_measured = calc_updated_fkm(tiago_model, calib.data, truth, q, cfg)
        return calib

    return make


@pytest.fixture
def failed_fit(make_calib, monkeypatch):
    """A fit whose solver reports ``success=False``."""
    real = bc.least_squares

    def not_converged(*args, **kwargs):
        result = real(*args, **kwargs)
        result.success = False
        result.message = "forced non-convergence"
        return result

    monkeypatch.setattr(bc, "least_squares", not_converged)
    calib = make_calib()
    calib.solve(plotting=False, enable_logging=False)
    return calib


@pytest.fixture
def good_fit(make_calib):
    calib = make_calib()
    calib.solve(plotting=False, enable_logging=False)
    return calib


def test_status_of_a_failed_fit(failed_fit):
    assert _stage(failed_fit, "fit").status == "failed"
    assert failed_fit.STATUS == "FAILED"


def test_status_of_a_converged_fit(good_fit):
    assert _stage(good_fit, "fit").status == "ok"
    assert good_fit.STATUS == "CALIBRATED"


def test_joint_corrections_refuse_a_failed_fit(failed_fit):
    with pytest.raises(CalibrationError, match="export failed: the fit failed"):
        failed_fit.joint_corrections()
    export = _stage(failed_fit, "export")
    assert export.status == "failed"
    assert "forced non-convergence" in export.reason


def test_pal_export_refuses_a_failed_fit(failed_fit, tmp_path):
    out = tmp_path / "master_calibration.yaml"
    with pytest.raises(CalibrationError, match="export failed"):
        export_geometric_calibration_yaml(failed_fit, str(out))
    assert not out.exists()
    assert _stage(failed_fit, "export").status == "failed"
    with pytest.raises(CalibrationError, match="export failed"):
        build_geometric_calibration(failed_fit)


def test_override_exports_with_warning_and_record(failed_fit, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        corrections = failed_fit.joint_corrections(allow_failed_fit=True)
    assert corrections
    assert "failed fit" in caplog.text
    export = _stage(failed_fit, "export")
    assert export.status == "fallback"
    assert "diagnostic only" in export.reason
    assert failed_fit._run_provenance["export_override"]["allow_failed_fit"]

    out = tmp_path / "diag.yaml"
    export_geometric_calibration_yaml(failed_fit, str(out), allow_failed_fit=True)
    assert out.exists()


def test_a_converged_fit_exports_unchanged(good_fit, tmp_path):
    assert good_fit.joint_corrections()
    assert good_fit.joint_corrections(allow_failed_fit=True) == (
        good_fit.joint_corrections()
    )
    assert all(s.stage != "export" for s in good_fit.stages)
    export_geometric_calibration_yaml(good_fit, str(tmp_path / "ok.yaml"))
