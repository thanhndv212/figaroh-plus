"""Tests for figaroh.tools.geometric_calibration_export.

Builds a minimal BaseCalibration stand-in (BaseCalibration.__new__,
bypassing __init__'s robot/config-file requirements — same pattern as
test_base_calibration_redistribution.py) with a fixed
redistribute_parameters() output, since this module only consumes that
method's return value plus calib_config["known_baseframe"]/
["base_mapping_row_names"].
"""

import yaml
import pytest

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.tools.geometric_calibration_export import (
    _pal_joint_name,
    build_geometric_calibration,
    export_geometric_calibration_yaml,
)


def _bare_calibration(redistributed, calib_config):
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.redistribute_parameters = lambda: redistributed
    calib.calib_config = calib_config
    return calib


class TestPalJointName:
    def test_strips_joint_suffix(self):
        assert _pal_joint_name("arm_right_2_joint") == "arm_right_2"

    def test_passes_through_without_suffix(self):
        assert _pal_joint_name("arm_right_2") == "arm_right_2"

    def test_only_strips_trailing_suffix(self):
        assert _pal_joint_name("joint_arm_1_joint") == "joint_arm_1"


class TestBuildGeometricCalibration:
    def test_maps_axes_to_pal_suffixes(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
            "d_py_arm_1_joint": {"value": 0.002, "std_dev": 0.0001},
            "d_pz_arm_1_joint": {"value": 0.003, "std_dev": 0.0001},
            "d_phix_arm_1_joint": {"value": 0.01, "std_dev": 0.001},
            "d_phiy_arm_1_joint": {"value": 0.02, "std_dev": 0.001},
            "d_phiz_arm_1_joint": {"value": 0.03, "std_dev": 0.001},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {
            "arm_1_dx": 0.001,
            "arm_1_dy": 0.002,
            "arm_1_dz": 0.003,
            "arm_1_droll": 0.01,
            "arm_1_dpitch": 0.02,
            "arm_1_dyaw": 0.03,
        }

    def test_excludes_non_joint_placement_categories(self):
        """joint_offset / elasticity-style names (from calib_model=
        'joint_offset' or non_geom=True) must not leak into the PAL
        geometric_calibration -- only d_p*/d_phi* joint_placement names
        belong there."""
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
            "offsetRX_arm_2_joint": {"value": 0.5, "std_dev": 0.01},
            "k_RZ_arm_3_joint": {"value": 0.05, "std_dev": 0.001},
            "not_a_known_param": {"value": 1.0, "std_dev": 0.1},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.001}

    def test_excludes_base_merged_block_when_baseframe_unknown(self):
        redistributed = {
            "d_px_torso_lift_joint": {"value": 0.01, "std_dev": 0.001},
            "d_py_torso_lift_joint": {"value": 0.02, "std_dev": 0.001},
            "d_pz_torso_lift_joint": {"value": 0.03, "std_dev": 0.001},
            "d_phix_torso_lift_joint": {"value": 0.04, "std_dev": 0.001},
            "d_phiy_torso_lift_joint": {"value": 0.05, "std_dev": 0.001},
            "d_phiz_torso_lift_joint": {"value": 0.06, "std_dev": 0.001},
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
        }
        calib_config = {
            "known_baseframe": False,
            "base_mapping_row_names": [
                "d_px_torso_lift_joint",
                "d_py_torso_lift_joint",
                "d_pz_torso_lift_joint",
                "d_phix_torso_lift_joint",
                "d_phiy_torso_lift_joint",
                "d_phiz_torso_lift_joint",
                "d_px_arm_1_joint",
            ],
        }
        calib = _bare_calibration(redistributed, calib_config)

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        # Only the 7th row_names entry (arm_1) survives -- the first 6
        # (torso_lift_joint, merged with the co-estimated base transform)
        # are excluded, matching every hand-curated master_calibration.yaml.
        assert gc == {"arm_1_dx": 0.001}

    def test_includes_first_joint_when_baseframe_known(self):
        """known_baseframe=True (or absent/default) means there's no
        co-estimated base transform to merge with -- nothing should be
        excluded on that basis."""
        redistributed = {
            "d_px_torso_lift_joint": {"value": 0.01, "std_dev": 0.001},
        }
        calib_config = {
            "known_baseframe": True,
            "base_mapping_row_names": ["d_px_torso_lift_joint"],
        }
        calib = _bare_calibration(redistributed, calib_config)

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"torso_lift_dx": 0.01}

    def test_min_sigma_filters_low_confidence_parameters(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.01, "std_dev": 0.001},  # 10 sigma
            "d_py_arm_1_joint": {"value": 0.001, "std_dev": 0.002},  # 0.5 sigma
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib, min_sigma=2.0)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.01}

    def test_min_sigma_none_includes_everything(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 10.0},  # tiny sigma
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib, min_sigma=None)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.001}

    def test_min_sigma_handles_zero_std_dev(self):
        """A parameter with exactly-zero std_dev is treated as maximally
        confident (sigma=inf), not a divide-by-zero crash."""
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib, min_sigma=100.0)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.001}


class TestExportGeometricCalibrationYaml:
    def test_writes_matching_pal_structure(self, tmp_path):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        out_file = tmp_path / "master_calibration.yaml"

        returned_path = export_geometric_calibration_yaml(calib, str(out_file))

        assert returned_path == str(out_file)
        assert out_file.exists()
        loaded = yaml.safe_load(out_file.read_text())
        assert loaded == build_geometric_calibration(calib)
        assert (
            loaded["robot_state_publisher"]["geometric_calibration"]["arm_1_dx"]
            == 0.001
        )

    def test_writes_header_comment(self, tmp_path):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        out_file = tmp_path / "master_calibration.yaml"

        export_geometric_calibration_yaml(
            calib, str(out_file), header_comment="48 samples, RMSE 8.99mm"
        )

        text = out_file.read_text()
        assert text.startswith("# 48 samples, RMSE 8.99mm\n")
        # Still valid YAML despite the leading comment.
        assert yaml.safe_load(text) is not None

    def test_forwards_min_sigma(self, tmp_path):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.01, "std_dev": 0.001},
            "d_py_arm_1_joint": {"value": 0.001, "std_dev": 0.002},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        out_file = tmp_path / "master_calibration_conservative.yaml"

        export_geometric_calibration_yaml(calib, str(out_file), min_sigma=2.0)

        loaded = yaml.safe_load(out_file.read_text())
        gc = loaded["robot_state_publisher"]["geometric_calibration"]
        assert gc == {"arm_1_dx": 0.01}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
