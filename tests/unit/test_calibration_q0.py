"""Calibration must not overwrite robot.q0 (#125).

``calib_config["q0"]`` is ``robot.q0`` (the same array). ``load_data`` and
the structural parameter selection used to write each sample's joints into
it, leaving the robot's default configuration at the last sample.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.calibration.calibration_tools import calc_updated_fkm
from figaroh.calibration.config import unified_to_legacy_config
from figaroh.calibration.data_loader import load_data


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


@pytest.fixture
def robot_and_config(tiago_model):
    robot = _Robot(tiago_model)
    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {"calibration_level": "joint_offset"},
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
    cfg = unified_to_legacy_config(robot, unified)
    assert cfg["q0"] is robot.q0  # the aliasing this test guards against
    return robot, cfg


def test_load_data_leaves_q0_unchanged(robot_and_config, tiago_model, tmp_path):
    robot, cfg = robot_and_config
    nominal = robot.q0.copy()
    joints = [tiago_model.names[j] for j in cfg["actJoint_idx"]]
    rng = np.random.default_rng(0)
    rows = {j: rng.uniform(-0.5, 0.5, 4) for j in joints}
    rows.update({"x1": np.zeros(4), "y1": np.zeros(4), "z1": np.zeros(4)})
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    _, q = load_data(str(path), tiago_model, cfg)

    np.testing.assert_array_equal(robot.q0, nominal)
    # the samples themselves are still built from q0 + the recorded joints
    np.testing.assert_allclose(q[-1, cfg["config_idx"]], [rows[j][-1] for j in joints])


def test_structural_selection_leaves_q0_unchanged(robot_and_config, tiago_model):
    robot, cfg = robot_and_config
    nominal = robot.q0.copy()
    calib = _Calib.__new__(_Calib)
    calib.model, calib.data, calib.calib_config = (
        tiago_model,
        tiago_model.createData(),
        cfg,
    )
    cfg.update(known_baseframe=False, known_tipframe=False, NbSample=20)
    calib.create_param_list()

    np.testing.assert_array_equal(robot.q0, nominal)
