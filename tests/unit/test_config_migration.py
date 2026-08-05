"""Tests for figaroh.utils.config_migration (legacy -> unified config).

Regression-tests the two wiring fixes made alongside this converter:
free_flyer (kinematics.free_flying_base, not parameters.free_flyer) and
eye-hand (tasks.calibration.eye_hand.{camera_frame,reference_frame} ->
base_to_ref_frame/ref_frame), via the same round-trip self-check the CLI
tool uses.
"""

import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.utils.config_migration import (
    legacy_calibration_to_unified,
    legacy_identification_to_unified,
    merge_unified_sections,
    convert_legacy_to_unified,
    self_check,
)


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


LEGACY_CALIB_BASIC = {
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

LEGACY_IDENTIF_BASIC = {
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


class TestLegacyCalibrationToUnified:
    def test_basic_fields(self):
        unified = legacy_calibration_to_unified(LEGACY_CALIB_BASIC)
        task = unified["tasks"]["calibration"]
        assert task["parameters"]["calibration_level"] == "full_params"
        assert task["kinematics"]["base_frame"] == "base_link"
        assert task["kinematics"]["tool_frame"] == "link2"
        assert task["kinematics"]["free_flying_base"] is False
        assert task["data"]["number_of_samples"] == 42
        assert "eye_hand" not in task

    def test_eye_hand_fields(self):
        legacy = dict(LEGACY_CALIB_BASIC, base_to_ref_frame="link1", ref_frame="link1")
        unified = legacy_calibration_to_unified(legacy)
        eye_hand = unified["tasks"]["calibration"]["eye_hand"]
        assert eye_hand["enabled"] is True
        assert eye_hand["camera_frame"] == "link1"
        assert eye_hand["reference_frame"] == "link1"

    def test_free_flyer_true(self):
        legacy = dict(LEGACY_CALIB_BASIC, free_flyer=True)
        unified = legacy_calibration_to_unified(legacy)
        assert unified["tasks"]["calibration"]["kinematics"]["free_flying_base"] is True


class TestLegacyIdentificationToUnified:
    def test_mechanics_and_problem(self):
        unified = legacy_identification_to_unified(LEGACY_IDENTIF_BASIC)
        props = unified["robot"]["properties"]
        assert props["mechanics"]["friction_coefficients"]["viscous"] == [0.1, 0.2]
        assert props["coupling"]["has_coupled_wrist"] is True
        assert props["coupling"]["Iam6"] == 0.5

        problem = unified["tasks"]["identification"]["problem"]
        assert problem["model_components"]["friction"] is True
        assert problem["use_joint_torques"] is True

    def test_ts_to_sampling_frequency(self):
        unified = legacy_identification_to_unified(LEGACY_IDENTIF_BASIC)
        sp = unified["tasks"]["identification"]["signal_processing"]
        assert sp["sampling_frequency"] == pytest.approx(1.0 / 0.001)

    def test_unconsumed_fields_preserved_under_custom(self):
        unified = legacy_identification_to_unified(LEGACY_IDENTIF_BASIC)
        custom = unified["custom"]
        assert custom["legacy_problem_params"]["is_static_regressor"] is True
        assert custom["legacy_tls_params"]["sync_joint_motion"] is False


class TestMergeAndConvert:
    def test_merge_combines_tasks_and_robot_properties(self):
        calib = legacy_calibration_to_unified(LEGACY_CALIB_BASIC)
        identif = legacy_identification_to_unified(LEGACY_IDENTIF_BASIC)
        merged = merge_unified_sections(calib, identif)
        assert "calibration" in merged["tasks"]
        assert "identification" in merged["tasks"]
        assert merged["robot"]["properties"]["coupling"]["has_coupled_wrist"] is True

    def test_convert_full_legacy_config(self):
        legacy = {
            "calibration": LEGACY_CALIB_BASIC,
            "identification": LEGACY_IDENTIF_BASIC,
        }
        unified = convert_legacy_to_unified(legacy)
        assert unified["extends"].endswith("base_robot_config.yaml")
        assert unified["tasks"]["calibration"]["enabled"] is True
        assert unified["tasks"]["identification"]["enabled"] is True

    def test_convert_requires_at_least_one_section(self):
        with pytest.raises(ValueError):
            convert_legacy_to_unified({})

    def test_convert_unknown_template_raises(self):
        with pytest.raises(ValueError):
            convert_legacy_to_unified(
                {"calibration": LEGACY_CALIB_BASIC}, template="nonexistent"
            )


class TestSelfCheckRoundTrip:
    """Round-trip through the *real* unified_to_legacy_config /
    unified_to_legacy_identif_config -- the same check the CLI tool runs.
    Directly regression-tests the free_flyer and eye_hand wiring fixes.
    """

    def test_roundtrip_free_flyer(self, two_joint_urdf):
        robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
        legacy = {"calibration": dict(LEGACY_CALIB_BASIC, free_flyer=True)}
        unified = convert_legacy_to_unified(legacy)

        report = self_check(robot, legacy, unified)
        assert report["ok"], report["mismatches"]

    def test_roundtrip_eye_hand(self, two_joint_urdf):
        robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
        legacy = {
            "calibration": dict(
                LEGACY_CALIB_BASIC, base_to_ref_frame="link1", ref_frame="link1"
            )
        }
        unified = convert_legacy_to_unified(legacy)

        report = self_check(robot, legacy, unified)
        assert report["ok"], report["mismatches"]

    def test_roundtrip_identification(self, two_joint_urdf):
        robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
        legacy = {"identification": LEGACY_IDENTIF_BASIC}
        unified = convert_legacy_to_unified(legacy)

        report = self_check(robot, legacy, unified)
        assert report["ok"], report["mismatches"]

    def test_roundtrip_detects_real_mismatch(self, two_joint_urdf):
        """Sanity check that self_check isn't vacuously true: corrupting the
        unified doc after conversion must surface as a mismatch."""
        robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
        legacy = {"calibration": dict(LEGACY_CALIB_BASIC, free_flyer=True)}
        unified = convert_legacy_to_unified(legacy)
        unified["tasks"]["calibration"]["kinematics"]["free_flying_base"] = False

        report = self_check(robot, legacy, unified)
        assert not report["ok"]
        assert "calibration.free_flyer" in report["mismatches"]
