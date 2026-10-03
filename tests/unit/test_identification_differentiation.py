"""Independent motion oracles for configuration-to-tangent differentiation."""

import numpy as np
import pinocchio as pin
import pytest

from figaroh.backends.pinocchio import PinocchioBackend
from figaroh.identification.identification_tools import (
    calculate_first_second_order_differentiation,
)


@pytest.mark.parametrize("variable_dt", [False, True])
@pytest.mark.parametrize("use_backend", [False, True])
@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
def test_all_manipulator_coordinates_and_legacy_sample_alignment(
    variable_dt, use_backend, flags
):
    model = pin.buildSampleModelManipulator()
    intervals = np.array([0.01, 0.02, 0.015, 0.03, 0.025, 0.01, 0.02, 0.015])
    if not variable_dt:
        intervals[:] = 0.01
    times = np.r_[0.0, np.cumsum(intervals)]
    acceleration = np.arange(1, model.nv + 1) * 0.1
    initial_velocity = np.arange(1, model.nv + 1) * 0.02
    positions = (
        0.5 * times[:, None] ** 2 * acceleration + times[:, None] * initial_velocity
    )
    original = positions.copy()
    midpoint_times = (times[:-1] + times[1:]) / 2
    config = dict(ts=0.01, is_joint_torques=flags[0], is_external_wrench=flags[1])
    backend = PinocchioBackend.from_model(model) if use_backend else None
    q, dq, ddq = calculate_first_second_order_differentiation(
        None if use_backend else model,
        positions,
        config,
        dt=intervals if variable_dt else None,
        backend=backend,
    )

    np.testing.assert_array_equal(positions, original)
    np.testing.assert_array_equal(q, positions[:-2])
    expected_velocity = midpoint_times[:-1, None] * acceleration + initial_velocity
    np.testing.assert_allclose(dq, expected_velocity, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        ddq, np.broadcast_to(acceleration, dq.shape), atol=1e-11, rtol=1e-11
    )


@pytest.mark.parametrize("joint_type", ["free_flyer", "continuous"])
@pytest.mark.parametrize("use_backend", [False, True])
@pytest.mark.parametrize("variable_dt", [False, True])
def test_non_euclidean_configuration_uses_nv(joint_type, use_backend, variable_dt):
    model = pin.Model()
    joint = (
        pin.JointModelFreeFlyer()
        if joint_type == "free_flyer"
        else pin.JointModelRUBZ()
    )
    model.addJoint(0, joint, pin.SE3.Identity(), "moving_joint")
    assert model.nq != model.nv
    intervals = np.array([0.02, 0.01, 0.03, 0.015, 0.025])
    if not variable_dt:
        intervals[:] = 0.02
    times = np.r_[0.0, np.cumsum(intervals)]
    acceleration = np.arange(1, model.nv + 1) * 0.2
    # One-parameter subgroup: exact tangent secants commute, including rotation.
    q = np.array(
        [
            pin.integrate(model, pin.neutral(model), 0.5 * t * t * acceleration)
            for t in times
        ]
    )
    backend = PinocchioBackend.from_model(model) if use_backend else None
    trimmed, dq, ddq = calculate_first_second_order_differentiation(
        None if use_backend else model,
        q,
        dict(ts=0.02, is_joint_torques=True, is_external_wrench=False),
        dt=intervals if variable_dt else None,
        backend=backend,
    )
    assert trimmed.shape == (len(times) - 2, model.nq)
    assert dq.shape == ddq.shape == (len(times) - 2, model.nv)
    midpoints = (times[:-1] + times[1:]) / 2
    np.testing.assert_allclose(dq, midpoints[:-1, None] * acceleration, atol=1e-11)
    np.testing.assert_allclose(ddq, np.broadcast_to(acceleration, dq.shape), atol=1e-10)


@pytest.mark.parametrize(
    "dt",
    [0.0, -0.01, np.nan, np.inf, [0.01, 0.0], [0.01, np.nan], [0.01], [[0.01, 0.01]]],
)
def test_invalid_intervals_raise_before_dividing(dt):
    model = pin.buildSampleModelManipulator()
    with pytest.raises(ValueError, match="timestep|dt"):
        calculate_first_second_order_differentiation(
            model, np.zeros((3, model.nq)), {}, dt=dt
        )


@pytest.mark.parametrize(
    "q", [np.zeros((2, 6)), np.zeros((3, 5)), np.zeros(6), np.full((3, 6), np.nan)]
)
def test_invalid_positions_are_rejected(q):
    with pytest.raises(ValueError, match="q|samples|finite"):
        calculate_first_second_order_differentiation(
            pin.buildSampleModelManipulator(), q, dict(ts=0.01)
        )


def test_three_samples_and_explicit_scalar_dt():
    model = pin.buildSampleModelManipulator()
    times = np.arange(3) * 0.01
    q = 0.5 * times[:, None] ** 2 * np.ones(model.nq)
    trimmed, dq, ddq = calculate_first_second_order_differentiation(
        model, q, {}, dt=0.01
    )
    assert trimmed.shape == dq.shape == ddq.shape == (1, model.nv)
    np.testing.assert_allclose(ddq, 1.0, atol=1e-12)


def test_continuous_joint_across_angle_wrap():
    model = pin.Model()
    model.addJoint(0, pin.JointModelRUBZ(), pin.SE3.Identity(), "continuous")
    angles = np.pi + np.arange(-2, 3) * 0.02
    q = np.column_stack((np.cos(angles), np.sin(angles)))
    _, dq, ddq = calculate_first_second_order_differentiation(model, q, dict(ts=0.01))
    np.testing.assert_allclose(dq, 2.0, atol=1e-12)
    np.testing.assert_allclose(ddq, 0.0, atol=1e-10)
