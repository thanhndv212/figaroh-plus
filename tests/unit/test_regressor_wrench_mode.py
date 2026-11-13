"""External-wrench mode of the basic regressor (#249).

The wrench rows are the free-flyer rows of the equations of motion. Joint
friction, actuator inertia and joint offsets do not appear in them, so the
mode refuses those blocks and builds inertial columns only. Floating-base
sample humanoid: 29 bodies, more than six, and more than six samples.
"""

import types

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.tools.regressor import WRENCH_COMPONENTS, build_regressor_basic

ALL = list(WRENCH_COMPONENTS)


@pytest.fixture(scope="module")
def humanoid():
    model = pin.buildSampleModelHumanoid()
    return types.SimpleNamespace(model=model, data=model.createData())


def _motion(model, n, seed=0):
    rng = np.random.default_rng(seed)
    pin.seed(seed)
    q = np.stack(
        [
            pin.randomConfiguration(model, -np.ones(model.nq), np.ones(model.nq))
            for _ in range(n)
        ]
    )
    return q, rng.standard_normal((n, model.nv)), rng.standard_normal((n, model.nv))


def _config(model, **overrides):
    config = dict(
        is_joint_torques=False,
        is_external_wrench=True,
        force_torque=ALL,
        has_friction=False,
        has_actuator_inertia=False,
        has_joint_offset=False,
        act_idxv=list(range(model.nv)),
    )
    return config | overrides


@pytest.mark.parametrize("n", [4, 8, 50])
@pytest.mark.parametrize(
    "block", ["has_friction", "has_actuator_inertia", "has_joint_offset"]
)
def test_joint_blocks_are_refused(humanoid, n, block):
    q, v, a = _motion(humanoid.model, n)
    with pytest.raises(ValueError, match=block):
        build_regressor_basic(
            humanoid, q, v, a, _config(humanoid.model, **{block: True})
        )


def test_every_enabled_block_is_named(humanoid):
    q, v, a = _motion(humanoid.model, 8)
    config = _config(humanoid.model, has_friction=True, has_joint_offset=True)
    with pytest.raises(ValueError, match="has_friction.*has_joint_offset"):
        build_regressor_basic(humanoid, q, v, a, config)


@pytest.mark.parametrize("n", [4, 50])
def test_regressor_is_the_base_wrench(humanoid, n):
    model = humanoid.model
    q, v, a = _motion(model, n)
    W = build_regressor_basic(humanoid, q, v, a, _config(model))
    bodies = len(model.inertias) - 1
    assert W.shape == (6 * n, 10 * bodies)

    phi = np.concatenate(
        [model.inertias[i].toDynamicParameters() for i in range(1, bodies + 1)]
    )
    data = model.createData()
    tau = np.stack([pin.rnea(model, data, q[i], v[i], a[i]) for i in range(n)])
    # component-major: all samples of Fx, then Fy, ...
    np.testing.assert_allclose(W @ phi, tau[:, :6].T.flatten(), atol=1e-9)


def test_only_requested_components_are_filled(humanoid):
    model = humanoid.model
    n = 8
    q, v, a = _motion(model, n)
    full = build_regressor_basic(humanoid, q, v, a, _config(model))
    part = build_regressor_basic(
        humanoid, q, v, a, _config(model, force_torque=["Fz", "Mx"])
    )
    assert part.shape == full.shape
    for c in range(6):
        rows = slice(c * n, (c + 1) * n)
        if c in (2, 3):
            np.testing.assert_array_equal(part[rows], full[rows])
        else:
            assert not part[rows].any()


@pytest.mark.parametrize(
    "force_torque", [None, [], "wrist_ft_sensor", ["wrist_ft_sensor"], ["All"]]
)
def test_force_torque_must_list_wrench_components(humanoid, force_torque):
    q, v, a = _motion(humanoid.model, 8)
    with pytest.raises(ValueError, match="wrench components"):
        build_regressor_basic(
            humanoid, q, v, a, _config(humanoid.model, force_torque=force_torque)
        )
