"""Collision constraint of the exciting-trajectory optimiser (#143).

A one-joint arm whose waypoints are clear of an obstacle while the motion
between them sweeps through it: the waypoint-only constraint misses it, the
check points and the whole-trajectory check do not.
"""

from types import SimpleNamespace

import coal
import numpy as np
import pinocchio as pin
import pytest

from figaroh.optimal.contraints import TrajectoryConstraintManager
from figaroh.tools.robotcollisions import add_non_adjacent_pairs
from figaroh.utils.cubic_spline import CubicSpline


def arm(with_pairs=False):
    """Revolute z joint with a 1 m link along x; obstacle at +y, 0.6 m out.

    An idle first joint (j0) carries no geometry, so the link is not
    adjacent to the world, where the obstacle is.
    """
    model = pin.Model()
    j0 = model.addJoint(
        0, pin.JointModelRevoluteUnaligned(1, 0, 0), pin.SE3.Identity(), "j0"
    )
    model.appendBodyToJoint(j0, pin.Inertia.FromSphere(0.1, 0.01), pin.SE3.Identity())
    jid = model.addJoint(
        j0, pin.JointModelRevoluteUnaligned(0, 0, 1), pin.SE3.Identity(), "j1"
    )
    model.appendBodyToJoint(
        jid, pin.Inertia.FromBox(1.0, 1.0, 0.1, 0.1), pin.SE3.Identity()
    )
    model.lowerPositionLimit[:] = -4.0
    model.upperPositionLimit[:] = 4.0
    model.velocityLimit[:] = 10.0
    model.effortLimit[:] = 100.0
    geom = pin.GeometryModel()
    link = pin.GeometryObject(
        "link",
        jid,
        pin.SE3(np.eye(3), np.array([0.5, 0.0, 0.0])),
        coal.Box(1.0, 0.1, 0.1),
    )
    obstacle = pin.GeometryObject(
        "obstacle",
        0,
        pin.SE3(np.eye(3), np.array([0.0, 0.6, 0.0])),
        coal.Box(0.2, 0.2, 0.2),
    )
    geom.addGeometryObject(link)
    geom.addGeometryObject(obstacle)
    if with_pairs:
        geom.addCollisionPair(pin.CollisionPair(0, 1))
    return SimpleNamespace(
        model=model,
        data=model.createData(),
        geom_model=geom,
        q0=pin.neutral(model),
        v0=np.zeros(model.nv),
        a0=np.zeros(model.nv),
    )


def manager(robot, **settings):
    CB = CubicSpline(robot, 3, ["j1"], 0)
    config = {"n_wps": 3, "freq": 20} | settings
    return TrajectoryConstraintManager(robot, CB, config, {}), CB


# waypoints 0 -> pi/2 + 1.2 -> pi: the link passes +y (the obstacle) in the
# first interval; the waypoints themselves are clear
WPS = np.array([[0.0, np.pi / 2 + 1.2, np.pi]])
TPS = np.matrix([[0.0], [2.0], [4.0]])
ZERO = np.zeros((1, 3))


def constraints(mgr, CB):
    t, p, _, _ = CB.get_full_config(20, TPS, WPS, ZERO, ZERO)
    return mgr._evaluate_collision_constraints(p, TPS, t)


def test_pairs_added_when_the_model_has_none(caplog):
    robot = arm()
    with caplog.at_level("INFO"):
        mgr, _ = manager(robot)
    assert len(robot.geom_model.collisionPairs) == mgr.n_pairs == 1
    assert "1 pairs" in caplog.text


def test_adjacent_bodies_are_not_paired():
    robot = arm()
    second = robot.model.addJoint(
        2,
        pin.JointModelRevoluteUnaligned(0, 0, 1),
        pin.SE3(np.eye(3), np.array([1.0, 0, 0])),
        "j2",
    )
    robot.geom_model.addGeometryObject(
        pin.GeometryObject("link2", second, pin.SE3.Identity(), coal.Box(0.1, 0.1, 0.1))
    )
    added = add_non_adjacent_pairs(robot.model, robot.geom_model)
    # link-obstacle and link2-obstacle; link-link2 are parent and child
    pairs = {(p.first, p.second) for p in robot.geom_model.collisionPairs}
    assert added == 2 and pairs == {(0, 1), (1, 2)}


def test_existing_pairs_are_kept():
    robot = arm(with_pairs=True)
    mgr, _ = manager(robot)
    assert mgr.n_pairs == 1


def test_no_geometry_warns(caplog):
    robot = arm()
    robot.geom_model = None
    with caplog.at_level("WARNING"):
        mgr, _ = manager(robot)
    assert mgr.n_pairs == 0 and "not enforced" in caplog.text
    assert mgr.trajectory_clear(TPS, WPS, ZERO, ZERO)


@pytest.mark.parametrize("screen", [None, 0.02])
def test_check_points_see_the_collision_between_waypoints(screen):
    waypoint_only, CB = manager(arm(), collision_screen=screen)
    assert np.all(constraints(waypoint_only, CB) >= waypoint_only.margin)
    assert not waypoint_only.trajectory_clear(TPS, WPS, ZERO, ZERO)

    checked, CB = manager(
        arm(), collision_screen=screen, collision_checks_per_interval=5
    )
    c = constraints(checked, CB)
    assert c[0] < checked.margin  # the first interval crosses the obstacle
    assert c[1] >= checked.margin


def test_screened_distances_match_exact_ones_within_the_screen():
    exact, _ = manager(arm())
    screened, _ = manager(arm(), collision_screen=0.3)
    for q in np.linspace(0.0, np.pi, 40):
        d = exact.clipped_distances(np.array([0.0, q]))
        np.testing.assert_allclose(
            screened.clipped_distances(np.array([0.0, q])),
            np.minimum(d, 0.3),
            atol=1e-9,
        )


def test_clear_trajectory_passes():
    mgr, _ = manager(arm())
    away = np.array([[0.0, -1.0, -2.0]])  # swings through -y, away from the obstacle
    assert mgr.trajectory_clear(TPS, away, ZERO, ZERO)
