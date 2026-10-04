"""Calibration gauge handling, initialisation and determinism (#102, #99).

When the base and tip frames are estimated, joint parameters they absorb
must be dropped on the actual measurement Jacobian; the frames start at a
closed-form estimate; and the selected parameter set must not depend on
Pinocchio's global random generator.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.calibration.calibration_tools import (
    calc_updated_fkm,
    calculate_identifiable_kinematics_model,
    drop_calibration_parameters,
    estimate_frames_closed_form,
    random_joint_configuration,
    select_identifiable_parameters,
)
from figaroh.calibration.config import unified_to_legacy_config
from figaroh.calibration.parameter import BASE_TPL, add_base_name


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


def _tiago_config(model, level, measurable):
    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {"calibration_level": level},
        "measurements": {
            "markers": [
                {"reference_joint": "arm_7_joint", "measurable_dof": measurable}
            ]
        },
        "data": {"source_file": "unused.csv"},
    }
    cfg = unified_to_legacy_config(_Robot(model), unified)
    cfg["known_baseframe"] = False
    cfg["known_tipframe"] = False
    return cfg


def _configurations(model, n, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([random_joint_configuration(model, rng) for _ in range(n)])


# ── determinism (#99) ───────────────────────────────────────────


def test_random_configuration_is_seeded_and_within_limits(tiago_model):
    a = random_joint_configuration(tiago_model, np.random.default_rng(3))
    pin.seed(42)  # global generator must not matter
    b = random_joint_configuration(tiago_model, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    for joint in tiago_model.joints[1:]:
        if joint.nq == 1:
            lo = tiago_model.lowerPositionLimit[joint.idx_q]
            hi = tiago_model.upperPositionLimit[joint.idx_q]
            if np.isfinite(lo) and np.isfinite(hi) and hi - lo <= 2 * np.pi:
                assert lo <= a[joint.idx_q] <= hi


def test_structural_regressor_ignores_global_rng(tiago_model):
    model, data = tiago_model, tiago_model.createData()
    cfg = _tiago_config(model, "full_params", [True] * 3 + [False] * 3)
    cfg["NbSample"] = 20
    pin.seed(0)
    first = calculate_identifiable_kinematics_model([], model, data, cfg)
    pin.seed(123)
    second = calculate_identifiable_kinematics_model([], model, data, cfg)
    np.testing.assert_array_equal(first, second)


# ── closed-form frame initialisation ────────────────────────────


@pytest.mark.parametrize("orient", [False, True])
def test_closed_form_recovers_base_and_tip(tiago_model, orient):
    model, data = tiago_model, tiago_model.createData()
    measurable = [True] * 3 + [orient] * 3
    cfg = _tiago_config(model, "full_params", measurable)
    tip = [
        f"{e}_1"
        for e, m in zip(
            ["pEEx", "pEEy", "pEEz", "phiEEx", "phiEEy", "phiEEz"], measurable
        )
        if m
    ]
    cfg["param_name"] = list(BASE_TPL) + tip
    q = _configurations(model, 25)
    cfg["NbSample"] = len(q)
    base = [1.5, -0.7, 0.4, 0.2, -0.1, 2.9]  # far from zero: yaw 166 deg
    tip_true = [0.12, -0.03, 0.2, 0.3, -0.2, 0.5][: len(tip)]
    truth = np.array(base + tip_true)
    PEE = calc_updated_fkm(model, data, truth, q, cfg)

    guess = estimate_frames_closed_form(model, data, q, PEE, cfg)

    est = np.array([guess[n] for n in cfg["param_name"]])
    np.testing.assert_allclose(est[:3], truth[:3], atol=1e-9)
    np.testing.assert_allclose(
        pin.rpy.rpyToMatrix(est[3:6]), pin.rpy.rpyToMatrix(truth[3:6]), atol=1e-9
    )
    np.testing.assert_allclose(est[6:9], truth[6:9], atol=1e-9)


def test_closed_form_skips_partial_position_measurement(tiago_model):
    model, data = tiago_model, tiago_model.createData()
    cfg = _tiago_config(model, "full_params", [False, False, True, True, True, False])
    cfg["param_name"] = list(BASE_TPL)
    q = _configurations(model, 5)
    assert estimate_frames_closed_form(model, data, q, np.zeros(3 * 5), cfg) == {}


# ── data-level identifiability ──────────────────────────────────


def test_select_drops_combinations_and_keeps_frames():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 40))
    J = np.column_stack([a, b, 2.0 * a - 0.5 * b, c, 1e3 * c])
    kept, dropped = select_identifiable_parameters(
        J, ["base_px", "base_py", "j1", "j2", "j3"], always_keep=["base_px", "base_py"]
    )
    assert kept == ["base_px", "base_py", "j2"]
    assert dropped == ["j1", "j3"]  # j1 = combination of frames; j3 ~ j2 (units)


def test_drop_parameters_keeps_base_mapping_aligned():
    cfg = {
        "param_name": ["base_px", "a", "b", "c", "pEEx_1"],
        "base_mapping_slice": (1, 4),
        "base_mapping_matrix": np.arange(9.0).reshape(3, 3),
        "base_mapping_row_names": ["a", "b", "c"],
    }
    drop_calibration_parameters(cfg, ["b"])
    assert cfg["param_name"] == ["base_px", "a", "c", "pEEx_1"]
    assert cfg["base_mapping_slice"] == (1, 3)
    assert cfg["base_mapping_row_names"] == ["a", "c"]
    np.testing.assert_array_equal(cfg["base_mapping_matrix"], [[0, 1, 2], [6, 7, 8]])


def test_add_base_name_shifts_joint_offset_slice():
    cfg = {
        "calib_model": "joint_offset",
        "param_name": ["x", "y"],
        "base_mapping_slice": (0, 2),
    }
    add_base_name(cfg)
    assert cfg["param_name"][:6] == list(BASE_TPL)
    assert cfg["base_mapping_slice"] == (6, 8)


def test_joint_offset_drops_parameters_absorbed_by_base(tiago_model):
    """TIAGo: torso (vertical prismatic) and arm_1 (vertical revolute) are
    absorbed by the base's z translation and yaw."""
    model = tiago_model
    cfg = _tiago_config(model, "joint_offset", [True] * 3 + [False] * 3)
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.model, calib.data, calib.calib_config = model, model.createData(), cfg
    q = _configurations(model, 30)
    cfg["NbSample"] = len(q)
    calib.q_measured = q

    calib.create_param_list()  # no measurements yet: structural set only
    rng = np.random.default_rng(2)
    truth = rng.normal(scale=0.01, size=len(cfg["param_name"]))
    calib.PEE_measured = calc_updated_fkm(model, calib.data, truth, q, cfg)

    cfg = _tiago_config(model, "joint_offset", [True] * 3 + [False] * 3)
    cfg["NbSample"] = len(q)
    calib.calib_config = cfg
    calib.create_param_list()  # with measurements: data-level selection

    dropped = cfg["absorbed_param_name"]
    assert "offsetPZ_torso_lift_joint" in dropped
    assert "offsetRZ_arm_1_joint" in dropped
    assert all(n in cfg["param_name"] for n in BASE_TPL)
    assert not any(n in cfg["param_name"] for n in dropped)


# ── orientation residual near the +-pi wrap ─────────────────────


def test_logmap_residual_across_yaw_wrap():
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.calib_config = {"measurability": [True] * 6, "NbSample": 1, "NbMarkers": 1}
    eps = 1e-3
    measured = np.array([0, 0, 0, 0, 0, np.pi - eps])
    estimated = np.array([0, 0, 0, 0, 0, -np.pi + eps])
    res = calib._compute_logmap_residuals(measured, estimated)
    assert np.linalg.norm(res[3:]) == pytest.approx(2 * eps, abs=1e-9)
