"""Deprecation warning tests for the legacy config parsers.

Both figaroh.calibration.config.get_param_from_yaml and
figaroh.identification.config.get_param_from_yaml are the single funnel
every legacy-format caller goes through (BaseCalibration/BaseIdentification's
load_param, figaroh.utils.config_parser's legacy branch, and custom scripts
that call them directly) -- so warning here covers every caller.
"""

import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.config import get_param_from_yaml as calib_get_param
from figaroh.identification.config import get_param_from_yaml as identif_get_param


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


LEGACY_CALIB = {
    "calib_level": "full_params",
    "non_geom": False,
    "base_frame": "base_link",
    "tool_frame": "link2",
    "markers": [
        {"ref_joint": "joint2", "measure": [True, True, True, False, False, False]}
    ],
    "free_flyer": False,
    "base_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "tip_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "coeff_regularize": 0.01,
    "outlier_eps": 0.05,
    "data_file": "data/calib.csv",
    "sample_configs_file": None,
    "nb_sample": 42,
}

LEGACY_IDENTIF = {
    "robot_params": [
        {
            "q_lim_def": 1.57,
            "dq_lim_def": 5.0,
            "ddq_lim_def": 20.0,
            "tau_lim_def": 4.0,
            "fv": [0.1, 0.2],
            "fs": [0.01, 0.02],
            "Ia": [0.001, 0.002],
            "offset": [0.0, 0.0],
            "Iam6": 0.5,
            "fvm6": 0.3,
            "fsm6": 0.1,
            "reduction_ratio": [50.0, 50.0],
            "ratio_essential": 30.0,
        }
    ],
    "problem_params": [
        {
            "is_external_wrench": False,
            "is_joint_torques": True,
            "force_torque": None,
            "external_wrench_offsets": False,
            "has_friction": True,
            "has_joint_offset": True,
            "has_actuator_inertia": True,
            "is_static_regressor": True,
            "is_inertia_regressor": True,
            "has_coupled_wrist": True,
            "embedded_forces": False,
        }
    ],
    "processing_params": [{"cut_off_frequency_butterworth": 100.0, "ts": 0.001}],
    "tls_params": [
        {"mass_load": 0.0, "which_body_loaded": 0, "sync_joint_motion": False}
    ],
}


def test_legacy_calibration_parser_warns(two_joint_urdf):
    robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
    with pytest.warns(DeprecationWarning, match="Legacy flat config format"):
        calib_get_param(robot, LEGACY_CALIB)


def test_legacy_identification_parser_warns(two_joint_urdf):
    robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
    with pytest.warns(DeprecationWarning, match="Legacy flat config format"):
        identif_get_param(robot, LEGACY_IDENTIF)


def test_unified_calibration_does_not_warn(two_joint_urdf, recwarn):
    """No false positive: parsing an already-unified task config (the
    dict shape unified_to_legacy_config expects) must not trigger the
    legacy-format deprecation warning."""
    from figaroh.calibration.config import unified_to_legacy_config

    robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
    unified_task_config = {
        "joints": {},
        "kinematics": {"base_frame": "base_link", "tool_frame": "link2"},
        "parameters": {"calibration_level": "full_params"},
        "measurements": {
            "markers": [{"reference_joint": "joint2", "measurable_dof": [True] * 6}]
        },
        "data": {"source_file": "data/calib.csv"},
    }
    unified_to_legacy_config(robot, unified_task_config)
    deprecation_warnings = [
        w for w in recwarn.list if issubclass(w.category, DeprecationWarning)
    ]
    assert not deprecation_warnings, [str(w.message) for w in deprecation_warnings]
