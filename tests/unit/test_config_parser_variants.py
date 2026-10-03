"""Regressions for #76: variant placement and sectioned legacy dispatch."""

from types import SimpleNamespace

import pytest
import yaml

from figaroh.utils.config_parser import (
    UnifiedConfigParser,
    _parse_legacy_format,
)
from figaroh.utils.error_handling import ConfigurationError

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


LEGACY_CALIBRATION = {"markers": [], "calib_level": "full_params"}
LEGACY_IDENTIFICATION = {"robot_params": [{}]}


@pytest.fixture
def parsers(monkeypatch):
    """Record which legacy parser receives which section."""
    import figaroh.calibration.calibration_tools as cal
    import figaroh.identification.identification_tools as idt

    seen = {}
    monkeypatch.setattr(
        cal, "get_param_from_yaml", lambda robot, data: seen.setdefault("cal", data)
    )
    monkeypatch.setattr(
        idt, "get_param_from_yaml", lambda robot, data: seen.setdefault("id", data)
    )
    return seen


ROBOT = SimpleNamespace()


def test_sectioned_legacy_file_dispatches_the_requested_section(parsers):
    sectioned = {
        "calibration": LEGACY_CALIBRATION,
        "identification": LEGACY_IDENTIFICATION,
    }
    assert _parse_legacy_format(ROBOT, sectioned, "calibration") is LEGACY_CALIBRATION
    assert (
        _parse_legacy_format(ROBOT, sectioned, "identification")
        is LEGACY_IDENTIFICATION
    )


def test_flat_legacy_section_is_still_accepted(parsers):
    assert (
        _parse_legacy_format(ROBOT, dict(LEGACY_CALIBRATION), "calibration")
        == LEGACY_CALIBRATION
    )


def test_auto_with_one_section_selects_it(parsers):
    assert (
        _parse_legacy_format(ROBOT, {"calibration": LEGACY_CALIBRATION}, "auto")
        is LEGACY_CALIBRATION
    )


def test_auto_with_both_sections_requires_an_explicit_task(parsers):
    sectioned = {
        "calibration": LEGACY_CALIBRATION,
        "identification": LEGACY_IDENTIFICATION,
    }
    with pytest.raises(ConfigurationError, match="specify task_type"):
        _parse_legacy_format(ROBOT, sectioned, "auto")


def test_missing_requested_section_is_rejected(parsers):
    with pytest.raises(ConfigurationError, match="Required sections not found"):
        _parse_legacy_format(
            ROBOT, {"identification": LEGACY_IDENTIFICATION}, "calibration"
        )
