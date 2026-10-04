"""Held-out calibration data (``validation_data_file``), #105.

The unified config key must reach ``calib_config``, and loading the
validation CSV must not change the training sample count: ``load_data``
writes ``calib_config["NbSample"]``, which the solver and FK use.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.calibration.config import unified_to_legacy_config


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


def _unified(validation_file=""):
    return {
        "joints": {},
        "kinematics": {"base_frame": "base_link", "tool_frame": "link2"},
        "parameters": {"calibration_level": "full_params"},
        "measurements": {
            "markers": [
                {
                    "reference_joint": "joint2",
                    "measurable_dof": [True, True, True, False, False, False],
                }
            ]
        },
        "data": {"source_file": "train.csv", "validation_data_file": validation_file},
    }


def _write_csv(path, n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "y1": rng.normal(size=n),
            "z1": rng.normal(size=n),
            "joint1": rng.uniform(-1, 1, n),
            "joint2": rng.uniform(-1, 1, n),
        }
    )
    df.to_csv(path, index=False)
    return str(path)


def _calibration(model, calib_config, train_path):
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.model, calib.data = model, model.createData()
    calib.calib_config = calib_config
    calib._data_path = train_path
    calib.del_list_ = []
    return calib


@pytest.mark.parametrize("value, expected", [("val.csv", "val.csv"), ("", None)])
def test_config_maps_validation_data_file(two_joint_urdf, value, expected):
    robot = _Robot(pin.buildModelFromUrdf(two_joint_urdf))
    calib_config = unified_to_legacy_config(robot, _unified(value))
    assert calib_config["validation_data_file"] == expected


@pytest.mark.parametrize("n_train, n_val", [(7, 12), (12, 7)])
def test_validation_keeps_training_sample_count(
    tmp_path, two_joint_urdf, n_train, n_val
):
    model = pin.buildModelFromUrdf(two_joint_urdf)
    train = _write_csv(tmp_path / "train.csv", n_train, 0)
    val = _write_csv(tmp_path / "val.csv", n_val, 1)
    calib_config = unified_to_legacy_config(_Robot(model), _unified(val))
    calib = _calibration(model, calib_config, train)

    calib.load_data_set()

    assert calib._val_available
    assert calib.calib_config["NbSample"] == n_train
    assert calib.q_measured.shape[0] == n_train
    assert calib.PEE_measured.size == 3 * n_train
    assert calib._q_val.shape[0] == n_val
    assert calib._data_path == train

    # validation metrics evaluate every validation sample, and leave the
    # training count untouched
    calib.calib_config["param_name"] = ["d_px_joint2"]
    calib.LM_result = SimpleNamespace(x=np.array([0.01]))
    metrics = calib._compute_validation_metrics()
    assert metrics["validation_source"] == "validation_data"
    assert metrics["n_val_samples"] == n_val
    assert calib.calib_config["NbSample"] == n_train


def test_unloadable_validation_file_warns(tmp_path, two_joint_urdf, caplog):
    model = pin.buildModelFromUrdf(two_joint_urdf)
    train = _write_csv(tmp_path / "train.csv", 5, 0)
    missing = str(tmp_path / "missing.csv")
    calib_config = unified_to_legacy_config(_Robot(model), _unified(missing))
    calib = _calibration(model, calib_config, train)

    with caplog.at_level(logging.WARNING):
        calib.load_data_set()

    assert not getattr(calib, "_val_available", False)
    assert calib.calib_config["NbSample"] == 5
    assert "missing.csv" in caplog.text
