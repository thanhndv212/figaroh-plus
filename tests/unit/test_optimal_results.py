# Copyright [2021-2025] Thanh Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for optimal-calibration result reporting and output location.

Covers three defects found by running ``optimal_config.py`` end to end:

* ``configuration_count`` counted dict keys (always 2) rather than the
  selected configurations, so the saved summary disagreed with stdout.
* The D-optimality determinant root was written under the key
  ``condition_number``, which is a different quantity entirely.
* ``tasks.optimal_configuration.output.output_file`` was ignored, because
  ``load_param`` only ever loads the ``calibration`` task.
"""

import textwrap

import pytest

from figaroh.optimal.base_optimal_calibration import BaseOptimalCalibration


class _Bare(BaseOptimalCalibration):
    """Construct the class without touching a robot model or config file.

    ``__init__`` needs a loaded robot and a parsable config; none of the
    behaviour under test does. Bypassing it keeps these unit tests fast and
    independent of Pinocchio and of any example config on disk.
    """

    def __init__(self):  # noqa: D107 - deliberately does not call super()
        pass


# --------------------------------------------------------------------------
# count_optimal_configurations  (was: len() over a dict of 2 keys)
# --------------------------------------------------------------------------


def test_count_returns_number_of_configurations_not_dict_keys():
    obj = _Bare()
    obj.optimal_configurations = {
        "calibration_joint_names": ["j1", "j2", "j3"],
        "calibration_joint_configurations": [[0.0] * 3 for _ in range(73)],
    }
    # The dict has 2 keys; the answer must be the 73 configurations.
    assert len(obj.optimal_configurations) == 2
    assert obj.count_optimal_configurations() == 73


@pytest.mark.parametrize("value", [None, {}])
def test_count_is_zero_when_nothing_selected(value):
    obj = _Bare()
    obj.optimal_configurations = value
    assert obj.count_optimal_configurations() == 0


def test_count_is_zero_when_key_absent():
    obj = _Bare()
    obj.optimal_configurations = {"calibration_joint_names": ["j1"]}
    assert obj.count_optimal_configurations() == 0


# --------------------------------------------------------------------------
# get_optimal_output_dir  (was: hardcoded "results")
# --------------------------------------------------------------------------


def test_output_dir_defaults_when_nothing_configured():
    obj = _Bare()
    obj._optimal_output_file = None
    assert obj.get_optimal_output_dir() == "results"


def test_output_dir_uses_configured_directory():
    obj = _Bare()
    obj._optimal_output_file = "data/optimal_configs/ur10_optimal_configs.yaml"
    assert obj.get_optimal_output_dir() == "data/optimal_configs"


def test_output_dir_falls_back_for_bare_filename():
    """A filename with no directory part must not yield an empty path."""
    obj = _Bare()
    obj._optimal_output_file = "configs.yaml"
    assert obj.get_optimal_output_dir() == "results"


def test_output_dir_default_is_overridable():
    obj = _Bare()
    obj._optimal_output_file = None
    assert obj.get_optimal_output_dir(default="out") == "out"


def test_output_dir_survives_missing_attribute():
    """Subclasses built before this attribute existed must still work."""
    obj = _Bare()
    assert obj.get_optimal_output_dir() == "results"


# --------------------------------------------------------------------------
# _load_optimal_output_file  (reads a task load_param never touches)
# --------------------------------------------------------------------------


def _write(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


UNIFIED_HEAD = """
    robot:
      name: "testbot"
    tasks:
      calibration:
        enabled: true
"""


def _unified(extra):
    """Return a minimal unified config with ``extra`` appended under tasks.

    Written as a call rather than an inline ``HEAD + "..."`` expression so
    black and this repo's flake8 hook agree on the formatting: black splits a
    long concatenation before the ``+``, which the hook reports as W503.
    """
    return UNIFIED_HEAD + extra


def test_reads_output_file_from_unified_config(tmp_path):
    cfg = _write(
        tmp_path,
        _unified(
            """
      optimal_configuration:
        enabled: true
        output:
          output_file: "data/optimal_configs/testbot.yaml"
    """
        ),
    )
    got = BaseOptimalCalibration._load_optimal_output_file(cfg)
    assert got == "data/optimal_configs/testbot.yaml"


@pytest.mark.parametrize("literal", ["None", "null", "~", '""'])
def test_yaml_none_like_values_are_treated_as_unset(tmp_path, literal):
    """A bare ``None`` in YAML parses as the *string* "None", not a null.

    That string previously slipped past the ``is None`` guard and was then
    treated as a file path, producing a confusing
    "Unsupported file format: None" error.
    """
    cfg = _write(
        tmp_path,
        _unified(
            f"""
      optimal_configuration:
        output:
          output_file: {literal}
    """
        ),
    )
    assert BaseOptimalCalibration._load_optimal_output_file(cfg) is None


def test_missing_optimal_block_yields_none(tmp_path):
    cfg = _write(tmp_path, UNIFIED_HEAD)
    assert BaseOptimalCalibration._load_optimal_output_file(cfg) is None


def test_legacy_config_yields_none(tmp_path):
    """Legacy flat configs have no tasks block; must not raise."""
    cfg = _write(
        tmp_path,
        """
        calibration:
          calib_level: full_params
          nb_sample: 10
        """,
    )
    assert BaseOptimalCalibration._load_optimal_output_file(cfg) is None


def test_unreadable_config_yields_none(tmp_path):
    missing = str(tmp_path / "does_not_exist.yaml")
    assert BaseOptimalCalibration._load_optimal_output_file(missing) is None
