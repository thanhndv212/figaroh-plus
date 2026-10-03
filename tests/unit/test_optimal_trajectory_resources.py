"""Resource regressions behind the optimal-trajectory examples.

The TIAGo and UR10 ``optimal_trajectory.py`` examples were killed by memory
exhaustion (a full m x m Q from pivoted QR on a tall regressor) and exceeded
their time budget (Richardson-extrapolated gradients, nv-times redundant RNEA
calls).  These tests pin the cheaper behaviour without allocating anything
large.
"""

from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
from scipy import linalg as scipy_linalg

from figaroh.optimal.base_optimal_trajectory import BaseTrajectoryIPOPTProblem
from figaroh.tools import qrdecomposition
from figaroh.tools.qrdecomposition import QR_pivoting, QRDecomposer, get_baseIndex
from figaroh.utils.cubic_spline import calc_torque


def _rank_deficient_regressor(m=60, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((m, 5))
    # Column 5 = col 0 + 2 * col 1, column 6 = -col 3: rank 5 of 7.
    return np.column_stack([W, W[:, 0] + 2 * W[:, 1], -W[:, 3]])


@pytest.fixture
def qr_modes(monkeypatch):
    """Record the ``mode`` of every pivoted scipy QR call in qrdecomposition."""
    modes = []

    def recording_qr(a, *args, **kwargs):
        if kwargs.get("pivoting"):
            modes.append(kwargs.get("mode", "full"))
        return scipy_linalg.qr(a, *args, **kwargs)

    monkeypatch.setattr(
        qrdecomposition,
        "linalg",
        SimpleNamespace(qr=recording_qr),
    )
    return modes


def test_pivoted_qr_never_builds_full_q(qr_modes):
    W = _rank_deficient_regressor()
    params = [f"p{i}" for i in range(W.shape[1])]
    tau = W @ np.arange(1.0, 8.0)

    get_baseIndex(W, params)
    QR_pivoting(tau, W, params)
    QRDecomposer().decompose(W, params, tau=tau, method="pivoting")
    QRDecomposer().get_base_mapping_matrix_pivoting(W, params)

    assert len(qr_modes) == 4
    assert "full" not in qr_modes


def test_base_index_matches_full_mode_qr():
    W = _rank_deficient_regressor()
    params = [f"p{i}" for i in range(W.shape[1])]

    _, R_full, P_full = scipy_linalg.qr(W, pivoting=True)
    rank = QRDecomposer()._find_rank(R_full)
    expected = tuple(sorted(P_full[:rank].tolist()))

    assert get_baseIndex(W, params) == expected
    assert len(expected) == 5


def test_calc_torque_matches_per_joint_rnea():
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    robot = SimpleNamespace(model=model, data=data)
    rng = np.random.default_rng(1)
    N = 7
    q = np.array([pin.randomConfiguration(model) for _ in range(N)])
    v = rng.standard_normal((N, model.nv))
    a = rng.standard_normal((N, model.nv))

    tau = calc_torque(N, robot, q, v, a)

    expected = np.zeros(model.nv * N)
    for i in range(N):
        tau_i = pin.rnea(model, data, q[i], v[i], a[i])
        for j in range(model.nv):
            expected[j * N + i] = tau_i[j]
    np.testing.assert_allclose(tau, expected)


def test_trajectory_gradient_uses_forward_differences():
    calls = []

    def objective_function(X, *args):
        calls.append(np.array(X))
        X = np.asarray(X)
        return float(X @ X + 3.0 * X[0])

    opt_traj = SimpleNamespace(objective_function=objective_function)
    problem = BaseTrajectoryIPOPTProblem(
        opt_traj, 2, 3, 10, None, None, None, None, None, None, None
    )
    X = np.array([0.5, -1.0, 2.0, 0.25])

    grad = problem.gradient(X)

    np.testing.assert_allclose(grad, 2 * X + np.array([3.0, 0, 0, 0]), atol=1e-4)
    # One base evaluation plus one per variable: no extrapolation sweep.
    assert len(calls) == len(X) + 1


class _StubConstraints:
    """Two waypoint variables in [-1, 1]; one constraint c = x0 + x1 >= 0."""

    def get_variable_bounds(self):
        return [-1.0, -1.0], [1.0, 1.0]

    def get_constraint_bounds(self, Ns):
        return [0.0], [2e19]

    def evaluate_constraints(self, Ns, X, opt_cb, *args):
        return np.array([X[0] + X[1]])


def _stub_problem(max_iterations=None):
    def objective_function(X, opt_cb, *args):
        opt_cb.update({"t_f": "t", "p_f": np.asarray(X), "v_f": "v", "a_f": "a"})
        return float(np.sum(np.asarray(X) ** 2))

    trajectory_config = {}
    if max_iterations is not None:
        trajectory_config["max_iterations"] = max_iterations
    opt_traj = SimpleNamespace(
        objective_function=objective_function,
        constraint_manager=_StubConstraints(),
        trajectory_config=trajectory_config,
    )
    # n_joints=1, n_wps=3 -> two decision variables.
    return BaseTrajectoryIPOPTProblem(
        opt_traj, 1, 3, 10, None, None, None, None, None, None, None
    )


def _fake_solver(status, x_opt, seen_configs):
    class FakeSolver:
        def __init__(self, problem, config):
            seen_configs.append(config)

        def solve(self):
            return status in (0, 1), {
                "x_opt": np.asarray(x_opt),
                "obj_val": 1.0,
                "status": status,
                "solve_time": 0.0,
            }

    return FakeSolver


@pytest.mark.parametrize(
    "x_opt, accepted",
    [([0.5, 0.25], True), ([0.5, -0.75], False)],
    ids=["feasible", "infeasible"],
)
def test_iteration_limit_keeps_only_feasible_iterate(monkeypatch, x_opt, accepted):
    from figaroh.optimal import base_optimal_trajectory as bot

    seen = []
    monkeypatch.setattr(bot, "RobotIPOPTSolver", _fake_solver(-1, x_opt, seen))
    problem = _stub_problem(max_iterations=17)

    success, results = problem.solve_with_waypoints(np.zeros((1, 3)))

    assert seen[0].max_iterations == 17
    assert success is accepted
    if accepted:
        assert results["iter_data"]["converged"] is False
        # Stored trajectory is rebuilt at the returned solution.
        np.testing.assert_allclose(results["p_f"], x_opt)


def test_converged_segment_is_marked_converged(monkeypatch):
    from figaroh.optimal import base_optimal_trajectory as bot

    seen = []
    monkeypatch.setattr(bot, "RobotIPOPTSolver", _fake_solver(0, [0.5, 0.25], seen))

    success, results = _stub_problem().solve_with_waypoints(np.zeros((1, 3)))

    assert seen[0].max_iterations == 200
    assert success is True
    assert results["iter_data"]["converged"] is True


def test_waypoint_steps_respect_velocity_limits():
    from figaroh.optimal.base_optimal_trajectory import BaseOptimalTrajectory

    t_s = 2.0
    v_max = np.array([0.07, 2.0])  # slow prismatic torso, fast arm joint
    stub = SimpleNamespace(
        trajectory_config={"t_s": t_s}, CB=SimpleNamespace(upper_dq=v_max)
    )
    wps = np.array(
        [
            [0.0, 0.3, -0.1, 0.25, 0.26],  # far beyond 0.07 m/s per 2 s
            [0.0, 0.5, 1.0, 0.8, 0.9],  # already feasible: unchanged
        ]
    )

    out = BaseOptimalTrajectory._limit_waypoint_steps(stub, wps)

    # A rest-to-rest cubic peaks at 1.5 * step / T.
    peak_v = 1.5 * np.abs(np.diff(out, axis=1)) / t_s
    assert np.all(peak_v <= v_max[:, None] + 1e-12)
    np.testing.assert_array_equal(out[1], wps[1])
    np.testing.assert_array_equal(out[:, 0], wps[:, 0])
    # Clamped waypoints move toward their predecessor, so stay in range.
    assert out[0].min() >= wps[0].min() and out[0].max() <= wps[0].max()
