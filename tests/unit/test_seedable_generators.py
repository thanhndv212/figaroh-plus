"""Seedable random generators (#266, plan item Q2.4).

Each generator takes ``rng``: a seed or ``Generator`` makes it reproducible
on its own; ``None`` keeps drawing from NumPy's global state, so callers that
seed with ``np.random.seed`` get the numbers they got before.
"""

from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest

from figaroh.calibration.calibration_tools import initialize_variables
from figaroh.optimal import base_optimal_trajectory
from figaroh.optimal.base_optimal_trajectory import BaseOptimalTrajectory
from figaroh.tools.randomdata import generate_waypoints, generate_waypoints_fext
from figaroh.tools.rng import resolve_rng
from figaroh.utils.cubic_spline import WaypointsGeneration


@pytest.fixture(scope="module")
def arm():
    model = pin.buildSampleModelManipulator()
    return SimpleNamespace(model=model, q0=pin.neutral(model))


@pytest.fixture(scope="module")
def floating():
    model = pin.buildSampleModelHumanoid()
    return SimpleNamespace(model=model, q0=pin.neutral(model))


def _waypoint_generator(arm, rng):
    joints = list(arm.model.names[1:4])
    wp = WaypointsGeneration(arm, 4, joints, rng=rng)
    wp.gen_rand_pool(np.zeros((3, len(joints))))
    return wp


def _draws(arm, floating):
    """One callable per generator: ``rng -> array``."""

    def waypoints(rng):
        return _drawn(generate_waypoints(5, arm, -1.0, 1.0, rng=rng))

    def waypoints_fext(rng):
        return _drawn(generate_waypoints_fext(5, floating, -1.0, 1.0, rng=rng))

    def initial_guess(rng):
        calib_config = {"param_name": [f"p{i}" for i in range(6)]}
        return initialize_variables(calib_config, mode=1, seed=0.1, rng=rng)[0]

    def spline_waypoints(rng):
        wp = _waypoint_generator(arm, rng)
        return np.concatenate(wp.gen_rand_wp(vel_set_zero=False, acc_set_zero=False))

    return {
        "generate_waypoints": waypoints,
        "generate_waypoints_fext": waypoints_fext,
        "initialize_variables": initial_guess,
        "WaypointsGeneration.gen_rand_wp": spline_waypoints,
    }


GENERATORS = [
    "generate_waypoints",
    "generate_waypoints_fext",
    "initialize_variables",
    "WaypointsGeneration.gen_rand_wp",
]


def _drawn(qva):
    # generate_waypoints* start each array from an uninitialised np.empty
    # row; only the rows after it are drawn
    return np.concatenate([x[1:].ravel() for x in qva])


@pytest.mark.parametrize("name", GENERATORS)
def test_same_seed_same_draws(arm, floating, name):
    draw = _draws(arm, floating)[name]
    np.testing.assert_array_equal(draw(7), draw(7))
    gen_a, gen_b = np.random.default_rng(7), np.random.default_rng(7)
    np.testing.assert_array_equal(draw(gen_a), draw(gen_b))


@pytest.mark.parametrize("name", GENERATORS)
def test_different_seeds_differ(arm, floating, name):
    draw = _draws(arm, floating)[name]
    assert not np.array_equal(draw(1), draw(2))


@pytest.mark.parametrize("name", GENERATORS)
def test_seeded_draws_ignore_global_state(arm, floating, name):
    draw = _draws(arm, floating)[name]
    np.random.seed(0)
    first = draw(3)
    np.random.seed(12345)
    np.random.uniform(size=50)
    np.testing.assert_array_equal(first, draw(3))


@pytest.mark.parametrize("name", GENERATORS)
def test_default_follows_np_random_seed(arm, floating, name):
    draw = _draws(arm, floating)[name]
    np.random.seed(42)
    first = draw(None)
    np.random.seed(42)
    np.testing.assert_array_equal(first, draw(None))


def test_default_matches_the_legacy_global_draws(arm):
    # rng=None must give exactly the numbers np.random gave before #266
    np.random.seed(5)
    q, v, a = generate_waypoints(3, arm, -1.0, 1.0)
    np.random.seed(5)
    nq, nv = arm.model.nq, arm.model.nv
    for i in range(3):
        np.testing.assert_array_equal(
            q[i + 1], np.random.uniform(low=-1.0, high=1.0, size=(nq,))
        )
        np.testing.assert_array_equal(
            v[i + 1], np.random.uniform(low=-1.0, high=1.0, size=(nv,))
        )
        np.testing.assert_array_equal(
            a[i + 1], np.random.uniform(low=-1.0, high=1.0, size=(nv,))
        )


def test_resolve_rng():
    gen = np.random.default_rng(0)
    assert resolve_rng(gen) is gen
    assert resolve_rng(None) is np.random
    assert resolve_rng(np.random) is np.random
    assert isinstance(resolve_rng(3), np.random.Generator)
    assert isinstance(resolve_rng(np.random.SeedSequence(3)), np.random.Generator)


@pytest.mark.parametrize("seed", [None, 0])
def test_trajectory_shares_its_source_with_the_waypoints(arm, monkeypatch, seed):
    monkeypatch.setattr(
        base_optimal_trajectory,
        "load_param",
        lambda robot, config_file: ({"soft_lim": 0.0, "n_wps": 4}, {}),
    )
    joints = list(arm.model.names[1:4])
    traj = BaseOptimalTrajectory(arm, joints, config_file="unused.yaml", rng=seed)
    wp = WaypointsGeneration(arm, 4, joints, rng=traj.rng)
    assert wp.rng is traj.rng
    if seed is None:
        assert traj.rng is np.random
    else:
        assert isinstance(traj.rng, np.random.Generator)
