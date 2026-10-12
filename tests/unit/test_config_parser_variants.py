"""Regressions for #76: variant placement."""

import pytest
import yaml

from figaroh.utils.config_parser import UnifiedConfigParser

UNIFIED = {
    "robot": {"name": "arm"},
    "tasks": {
        "calibration": {
            "enabled": True,
            "type": "kinematic_calibration",
            "kinematics": {"base_frame": "universe", "tool_frame": "tool0"},
            "parameters": {"calibration_level": "full_params", "nb_samples": 30},
            "measurements": {"markers": []},
        },
    },
    "variants": {
        "quick": {
            "extends": "tasks.calibration",
            "parameters": {"calibration_level": "joint_offset"},
        },
        "renamed": {"robot": {"name": "arm_v2"}},
    },
}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "unified.yaml"
    path.write_text(yaml.safe_dump(UNIFIED))
    return path


def test_extending_variant_overrides_its_target_section(config_path):
    config = UnifiedConfigParser(config_path, variant="quick").parse()
    calibration = config["tasks"]["calibration"]
    assert calibration["parameters"]["calibration_level"] == "joint_offset"
    # Unchanged keys of the target are kept.
    assert calibration["parameters"]["nb_samples"] == 30
    assert calibration["kinematics"]["base_frame"] == "universe"


def test_extending_variant_does_not_leak_to_the_root(config_path):
    config = UnifiedConfigParser(config_path, variant="quick").parse()
    assert "parameters" not in config
    assert "kinematics" not in config
    assert "variants" not in config


def test_variant_without_extends_still_overrides_the_root(config_path):
    config = UnifiedConfigParser(config_path, variant="renamed").parse()
    assert config["robot"]["name"] == "arm_v2"
    assert config["tasks"]["calibration"]["parameters"]["calibration_level"] == (
        "full_params"
    )


def test_base_parse_is_unaffected_by_variants(config_path):
    config = UnifiedConfigParser(config_path).parse()
    assert config["tasks"]["calibration"]["parameters"]["calibration_level"] == (
        "full_params"
    )
