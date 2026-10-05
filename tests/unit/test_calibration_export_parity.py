"""Calibrated vs exported/reloaded forward kinematics (figaroh-plus#62).

A known correction fixture on TIAGo: every joint parameter of the level,
an unknown base frame and tool point, noise-free measurements at 40 random
postures. For each calibration level and estimation method the fit is
exported with ``joint_corrections()`` (lifted and not), reloaded, and the
reloaded model with ``metrology_frames()`` applied outside it must predict
what the calibrated model predicts, on the training postures and on 30
postures not used for fitting. The metrology frames are never written into
the URDF.
"""

import copy

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration import estimation
from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.calibration.calibration_tools import (
    calc_updated_fkm,
    random_joint_configuration,
)
from figaroh.calibration.config import unified_to_legacy_config
from figaroh.calibration.parameter import BASE_TPL
from figaroh.tools.urdf_exporter import export_urdf
from figaroh.utils.error_handling import CalibrationError

TIP = ["pEEx_1", "pEEy_1", "pEEz_1"]
LEVELS = ["full_params", "joint_offset"]
METHODS = ["structural", "excitation", "map"]
# the exporter writes 12 significant digits; a convention error is >1e-6
PARITY_M = 1e-9


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


class _Calib(BaseCalibration):
    def cost_function(self, var):
        pee = calc_updated_fkm(
            self.model, self.data, var, self.q_measured, self.calib_config
        )
        return pee - self.PEE_measured


def _config(model, level, method=None):
    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {
            "calibration_level": level,
            "estimation": {} if method is None else {"method": method},
        },
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
    cfg = unified_to_legacy_config(_Robot(model), unified)
    cfg.update(known_baseframe=False, known_tipframe=False, random_seed=0)
    cfg["coeff_regularize"] = 0.0
    return cfg


def _fk(model, values, names, cfg, q):
    """Measured-point positions at ``q`` for parameters ``names``."""
    cfg = dict(cfg, param_name=list(names), NbSample=len(q))
    return calc_updated_fkm(model, model.createData(), np.asarray(values), q, cfg)


@pytest.fixture(scope="module")
def postures(tiago_model):
    rng = np.random.default_rng(0)
    train = np.array([random_joint_configuration(tiago_model, rng) for _ in range(40)])
    held_out = np.array(
        [random_joint_configuration(tiago_model, rng) for _ in range(30)]
    )
    return {"train": train, "held_out": held_out}


@pytest.fixture(scope="module", params=LEVELS)
def truth(request, tiago_model):
    """Known corrections of the expected sizes, frames of a few cm/rad."""
    level = request.param
    cfg = _config(tiago_model, level)
    joints = estimation.joint_candidates(tiago_model, cfg)
    names = list(BASE_TPL) + joints + TIP
    rng = np.random.default_rng(1)
    values = np.array(
        [
            rng.normal()
            * (
                0.05
                if estimation.prior_group(n, tiago_model) is None
                else estimation.DEFAULT_PRIORS[estimation.prior_group(n, tiago_model)]
            )
            for n in names
        ]
    )
    return {
        "level": level,
        "cfg": cfg,
        "names": names,
        "values": values,
        "joints": joints,
    }


@pytest.fixture(scope="module", params=METHODS)
def fit(request, truth, postures, tiago_model):
    cfg = _config(tiago_model, truth["level"], request.param)
    q = postures["train"]
    cfg["NbSample"] = len(q)
    calib = _Calib.__new__(_Calib)
    calib.model, calib.data, calib.calib_config = (
        tiago_model,
        tiago_model.createData(),
        cfg,
    )
    calib.q_measured = q
    calib.del_list_ = []
    calib.PEE_measured = _fk(tiago_model, truth["values"], truth["names"], cfg, q)
    calib.create_param_list()
    calib.solve(plotting=False, enable_logging=False)
    return request.param, calib


def _reload(urdf_path, corrections, tmp_path):
    out = tmp_path / "exported.urdf"
    export_urdf(str(urdf_path), corrections, output_path=str(out))
    return pin.buildModelFromUrdf(str(out)), out


def test_known_corrections_reload_exactly(
    truth, postures, tiago_model, tiago_urdf_path, tmp_path
):
    """The exporter's convention, without a fit: truth in, truth out."""
    names, values = truth["names"], truth["values"]
    corrections = {n: v for n, v in zip(names, values) if n in truth["joints"]}
    reloaded, _ = _reload(tiago_urdf_path, corrections, tmp_path)
    frames = [n for n in names if n not in corrections]
    frame_values = [values[names.index(n)] for n in frames]
    for q in postures.values():
        expected = _fk(tiago_model, values, names, truth["cfg"], q)
        got = _fk(reloaded, frame_values, frames, truth["cfg"], q)
        assert np.abs(got - expected).max() < PARITY_M


@pytest.mark.parametrize("lift", [True, False])
def test_exported_fit_reproduces_calibrated_fk(
    fit, lift, postures, tiago_model, tiago_urdf_path, tmp_path
):
    """Reloaded URDF + metrology frames == calibrated model, on postures
    used for fitting and not."""
    _, calib = fit
    cfg = calib.calib_config
    reloaded, _ = _reload(tiago_urdf_path, calib.joint_corrections(lift=lift), tmp_path)
    frames = calib.metrology_frames()
    assert set(frames) == set(calib._frame_param_names())
    for q in postures.values():
        calibrated = _fk(tiago_model, calib.var_, cfg["param_name"], cfg, q)
        got = _fk(reloaded, list(frames.values()), list(frames), cfg, q)
        assert np.abs(got - calibrated).max() < PARITY_M


def test_fit_recovers_known_corrections(fit, truth, postures, tiago_model):
    """Noise-free data: the fit predicts the truth on unused postures, i.e.
    it recovers the corrections in identifiable coordinates. ``structural``
    at ``full_params`` drops directions the data identifies (#113) and
    cannot."""
    method, calib = fit
    q = postures["held_out"]
    calibrated = _fk(
        tiago_model, calib.var_, calib.calib_config["param_name"], calib.calib_config, q
    )
    expected = _fk(tiago_model, truth["values"], truth["names"], truth["cfg"], q)
    error = np.abs(calibrated - expected).max()
    if method == "structural" and truth["level"] == "full_params":
        assert error > 1e-4
    else:
        assert error < 1e-9


def test_metrology_frames_are_not_written_into_the_urdf(fit, tiago_urdf_path, tmp_path):
    _, calib = fit
    corrections = calib.joint_corrections()
    frames = calib.metrology_frames()
    assert frames and not set(frames) & set(corrections)
    _, plain = _reload(tiago_urdf_path, corrections, tmp_path)
    with_frames = tmp_path / "with_frames.urdf"
    export_urdf(
        str(tiago_urdf_path), {**corrections, **frames}, output_path=str(with_frames)
    )
    assert with_frames.read_bytes() == plain.read_bytes()


@pytest.mark.parametrize("lift", [True, False])
def test_parameters_a_urdf_cannot_carry_are_rejected(fit, lift):
    """An elastic parameter in the fit: refuse, or drop only when asked."""
    _, calib = fit
    if lift and calib.calib_config.get("estimation_report") is None:
        pytest.skip("the structural lift names come from the base mapping")
    extra = "k_RZ_arm_2_joint"
    other = copy.copy(calib)
    other.calib_config = dict(calib.calib_config)
    other.calib_config["param_name"] = list(calib.calib_config["param_name"]) + [extra]
    if lift:
        report = dict(calib.calib_config["estimation_report"])
        report["candidates"] = list(report["candidates"]) + [extra]
        other.calib_config["estimation_report"] = report
        other._C_param = np.pad(calib._C_param, ((0, 1), (0, 1)))
    other.var_ = np.append(calib.var_, 1e-3)
    with pytest.raises(CalibrationError, match=extra):
        other.joint_corrections(lift=lift)
    kept = other.joint_corrections(lift=lift, drop_unsupported=True)
    assert kept == calib.joint_corrections(lift=lift)


def test_metrology_frames_requires_a_fit():
    calib = _Calib.__new__(_Calib)
    with pytest.raises(CalibrationError):
        calib.metrology_frames()


def test_pal_yaml_reproduces_calibrated_fk(
    fit, postures, tiago_model, tiago_urdf_path, tmp_path
):
    """The PAL geometric_calibration, applied to the nominal URDF as origin
    deltas, is the same model as the URDF export (#123): it was empty at
    ``joint_offset``, and TIAGo's arm_4/arm_5 origins sit at pitch -pi/2,
    where the rpy as written must be the reference."""
    from figaroh.tools.geometric_calibration_export import (
        build_geometric_calibration,
    )
    from test_geometric_calibration_export import apply_pal_yaml

    _, calib = fit
    cfg = calib.calib_config
    gc = build_geometric_calibration(calib, nominal_urdf=str(tiago_urdf_path))[
        "robot_state_publisher"
    ]["geometric_calibration"]
    assert gc
    reloaded = pin.buildModelFromUrdf(
        str(apply_pal_yaml(tiago_urdf_path, gc, tmp_path / "pal.urdf"))
    )
    frames = calib.metrology_frames()
    for q in postures.values():
        calibrated = _fk(tiago_model, calib.var_, cfg["param_name"], cfg, q)
        got = _fk(reloaded, list(frames.values()), list(frames), cfg, q)
        assert np.abs(got - calibrated).max() < PARITY_M
