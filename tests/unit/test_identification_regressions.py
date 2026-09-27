"""Regression tests for identification bugs #11-#13.

#11 get_standard_parameters paired each joint with the previous body.
#12 regressor extra-column layout assumed every extra block was enabled.
#13 unified signal_processing rates never reached the filters.
"""

import itertools
import types

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.identification.base_identification import BaseIdentification
from figaroh.identification.config import (
    _extract_signal_processing_params,
)
from figaroh.identification.parameter import (
    add_standard_additional_parameters,
    get_standard_parameters,
)
from figaroh.tools.regressor import build_regressor_basic


class _Ident(BaseIdentification):
    """Concrete stand-in; tests set only the attributes they need."""

    def load_trajectory_data(self, data_source=None):
        raise NotImplementedError


@pytest.fixture(scope="module")
def arm():
    model = pin.buildSampleModelManipulator()
    robot = types.SimpleNamespace(model=model, data=model.createData())
    rng = np.random.default_rng(0)
    n = 20
    q = rng.uniform(-np.pi, np.pi, (n, model.nq))
    v = rng.uniform(-1, 1, (n, model.nv))
    a = rng.uniform(-3, 3, (n, model.nv))
    return robot, q, v, a


def _params(model, cfg):
    params = get_standard_parameters(model, cfg)
    if cfg["has_friction"] or cfg["has_actuator_inertia"] or cfg["has_joint_offset"]:
        params.update(add_standard_additional_parameters(model, cfg))
    return params


# -- #11 ---------------------------------------------------------------------


def test_standard_parameters_read_each_joints_own_body(arm):
    robot, *_ = arm
    model = robot.model
    params = get_standard_parameters(model, {})
    for jid in range(1, model.njoints):
        name = model.names[jid]
        expected = model.inertias[jid].toDynamicParameters()
        assert params[f"m_{name}"] == pytest.approx(model.inertias[jid].mass)
        assert params[f"mx_{name}"] == pytest.approx(expected[1])
        assert params[f"Izz_{name}"] == pytest.approx(expected[9])


def test_standard_parameters_reproduce_rnea(arm):
    """W @ phi_std must equal the model's own inverse dynamics."""
    robot, q, v, a = arm
    model = robot.model
    cfg = {
        "has_friction": False,
        "has_actuator_inertia": False,
        "has_joint_offset": False,
        "is_joint_torques": True,
        "act_idxv": list(range(model.nv)),
    }
    W = build_regressor_basic(robot, q, v, a, cfg)
    phi = np.array(list(get_standard_parameters(model, cfg).values()))
    tau = np.stack([pin.rnea(model, robot.data, q[i], v[i], a[i]) for i in range(len(q))])
    # Regressor rows are joint-major: row j * N + i is joint j, sample i.
    np.testing.assert_allclose(W @ phi, tau.T.ravel(), atol=1e-9)


# -- #12 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "friction, actuator_inertia, joint_offset",
    list(itertools.product([False, True], repeat=3)),
)
def test_regressor_extra_columns_match_parameter_keys(
    arm, friction, actuator_inertia, joint_offset
):
    robot, q, v, a = arm
    model = robot.model
    nv = model.nv
    cfg = {
        "has_friction": friction,
        "has_actuator_inertia": actuator_inertia,
        "has_joint_offset": joint_offset,
        "is_joint_torques": True,
        "act_idxv": list(range(nv)),
        "fv": [0.3] * nv,
        "fs": [0.2] * nv,
        "Ia": [0.05] * nv,
        "off": [0.1] * nv,
    }
    W = build_regressor_basic(robot, q, v, a, cfg)
    params = _params(model, cfg)
    n_extra = 2 * friction + actuator_inertia + joint_offset
    assert W.shape[1] == len(params) == (10 + n_extra) * nv

    # Disabled blocks contribute no keys.
    assert any(k.startswith("fv_") for k in params) == friction
    assert any(k.startswith("Ia_") for k in params) == actuator_inertia
    assert any(k.startswith("off_") for k in params) == joint_offset

    # Each column is labelled with the parameter it multiplies.
    tau = np.stack([pin.rnea(model, robot.data, q[i], v[i], a[i]) for i in range(len(q))])
    if friction:
        tau = tau + 0.3 * v + 0.2 * np.sign(v)
    if actuator_inertia:
        tau = tau + 0.05 * a
    if joint_offset:
        tau = tau + 0.1
    phi = np.array(list(params.values()))
    np.testing.assert_allclose(W @ phi, tau.T.ravel(), atol=1e-9)


# -- #13 ---------------------------------------------------------------------


def test_signal_processing_rates_reach_filter_params():
    cfg = {}
    _extract_signal_processing_params(
        cfg,
        {"sampling_frequency": 50.0, "cutoff_frequency": 4.0, "filter_order": 3},
    )
    fp = cfg["filter_config"]["filter_params"]
    assert fp["f_sample"] == 50.0
    assert fp["f_butter"] == 4.0
    assert fp["nbutter"] == 3


def test_signal_processing_explicit_filter_params_win():
    cfg = {}
    _extract_signal_processing_params(
        cfg,
        {
            "sampling_frequency": 500.0,
            "cutoff_frequency": 50.0,
            "filter_params": {"f_butter": 10, "med_fil": 3},
        },
    )
    fp = cfg["filter_config"]["filter_params"]
    assert fp == {"f_butter": 10, "med_fil": 3, "f_sample": 500.0, "nbutter": 4}


def test_signal_processing_rejects_cutoff_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        _extract_signal_processing_params(
            {}, {"sampling_frequency": 50.0, "cutoff_frequency": 30.0}
        )


def test_filter_uses_configured_rate():
    """The filter must attenuate by the configured rate, not a 100 Hz default."""
    fs, fc = 50.0, 4.0
    cfg = {}
    _extract_signal_processing_params(
        cfg, {"sampling_frequency": fs, "cutoff_frequency": fc}
    )
    t = np.arange(0, 20, 1 / fs)
    x = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz: well inside the 4 Hz band
    ident = object.__new__(_Ident)
    mid = slice(len(t) // 4, 3 * len(t) // 4)

    def gain(filter_params):
        y = ident._apply_filters(x, **filter_params).ravel()
        return np.ptp(y[mid]) / np.ptp(x[mid])

    assert gain(cfg["filter_config"]["filter_params"]) > 0.95
    # The pre-fix behaviour (f_sample left at 100 Hz) halves the cutoff and
    # visibly attenuates the same signal.
    assert gain({"f_butter": fc}) < 0.7
