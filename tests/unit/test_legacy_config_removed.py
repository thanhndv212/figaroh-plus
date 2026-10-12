"""Removed in 1.0: the legacy flat config format and its deprecated parsers.

Loading a legacy flat config must raise an error naming the migration CLI
(``python -m figaroh.utils.config_migration``); the deprecated entry points
are gone.
"""

import pytest
import yaml

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

import figaroh.calibration.config as calib_config_mod
import figaroh.identification.config as identif_config_mod
import figaroh.utils.config_parser as config_parser
from figaroh.utils.config_parser import get_param_from_yaml, require_unified_config
from figaroh.utils.error_handling import ConfigurationError

CLI = "python -m figaroh.utils.config_migration"


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
    "coeff_regularize": 0.0,
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


@pytest.mark.parametrize(
    "module, name",
    [
        (calib_config_mod, "get_param_from_yaml"),
        (calib_config_mod, "get_param_from_yaml_legacy"),
        (calib_config_mod, "get_param_from_yaml_with_warning"),
        (identif_config_mod, "get_param_from_yaml"),
        (identif_config_mod, "get_param_from_yaml_legacy"),
        (identif_config_mod, "get_param_from_yaml_with_warning"),
        (config_parser, "get_calibration_param_from_yaml"),
        (config_parser, "get_identification_param_from_yaml"),
    ],
)
def test_deprecated_entry_points_are_removed(module, name):
    assert not hasattr(module, name)


@pytest.mark.parametrize("task", ["calibration", "identification"])
def test_legacy_dict_raises_with_migration_hint(two_joint_urdf, task):
    robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
    legacy = LEGACY_CALIB if task == "calibration" else LEGACY_IDENTIF
    for data in (legacy, {task: legacy}):
        with pytest.raises(ConfigurationError, match="config_migration") as exc:
            get_param_from_yaml(robot, data, task)
        assert CLI in str(exc.value)


def test_legacy_file_raises_with_migration_hint(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump({"calibration": LEGACY_CALIB}))
    with pytest.raises(ConfigurationError, match="legacy.yaml") as exc:
        require_unified_config(path)
    assert CLI in str(exc.value)


def test_missing_file_is_not_reported_as_legacy(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_unified_config(tmp_path / "absent.yaml")


def test_unified_calibration_does_not_warn(two_joint_urdf, recwarn):
    """Parsing an already-unified task config (the dict shape
    unified_to_legacy_config expects) raises no deprecation warning."""
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
