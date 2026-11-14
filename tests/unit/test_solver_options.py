"""LinearSolver reports what it could not do instead of returning a result.

An option the selected method does not use is refused, a constrained solve
that did not converge raises, the robust method accepts an exact fit, and
per-sample weights never build an n x n matrix.
"""

import numpy as np
import pytest

from figaroh.tools.solver import LinearSolver, solve_linear_system  # noqa: E402


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 6))
    x = rng.standard_normal(6)
    return A, x, A @ x + 0.05 * rng.standard_normal(120)


# -- options a method does not use --


@pytest.mark.parametrize(
    "kwargs, named",
    [
        (dict(method="lstsq", bounds=[(0, None)] * 6), "bounds"),
        (dict(method="elastic_net", alpha=0.1, bounds=[(0, None)] * 6), "bounds"),
        (dict(method="ridge", alpha=0.1, weights=np.ones(120)), "weights"),
        (dict(method="lstsq", alpha=0.1), "alpha"),
        (dict(method="constrained", alpha=0.1), "alpha"),
        (dict(method="ridge", alpha=-1.0), "alpha"),
        (dict(method="ridge", alpha=0.1, l1_ratio=0.9), "l1_ratio"),
        (dict(method="lstsq", regularization="l2"), "regularization"),
        (dict(method="ridge", alpha=0.1, regularization="l1"), "regularization"),
        (
            dict(method="lstsq", constraints={"A_eq": np.eye(6), "b_eq": np.zeros(6)}),
            "A_eq",
        ),
        (dict(method="constrained", constraints={"L": np.eye(6)}), "L"),
        (dict(method="constrained", constraints={"A_eq": np.eye(6)}), "b_eq"),
        (dict(method="constrained", constraints={"b_ineq": np.zeros(6)}), "A_ineq"),
    ],
)
def test_unused_option_is_refused(kwargs, named):
    with pytest.raises(ValueError, match=named):
        LinearSolver(**kwargs)


def test_refusal_reaches_the_convenience_function(system):
    A, _, b = system
    with pytest.raises(ValueError, match="bounds"):
        solve_linear_system(A, b, method="svd", bounds=[(0, None)] * 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="lstsq"),
        dict(method="ridge", alpha=0.1, regularization="l2"),
        dict(method="tikhonov", alpha=0.1, constraints={"L": np.eye(6)}),
        dict(method="constrained", bounds=[(-10, 10)] * 6),
        dict(
            method="constrained",
            constraints={"A_ineq": np.eye(6), "b_ineq": np.full(6, 10.0)},
        ),
        dict(method="weighted", weights=np.ones(120)),
        dict(method="robust"),
    ],
)
def test_options_a_method_uses_are_accepted(system, kwargs):
    A, _, b = system
    assert LinearSolver(**kwargs).solve(A, b).shape == (6,)


# -- constrained --


def test_failed_constrained_solve_raises(system):
    A, _, b = system
    row = np.ones((1, 6))
    contradictory = {
        "A_eq": np.vstack([row, row]),
        "b_eq": np.array([1.0, 2.0]),
    }
    solver = LinearSolver(method="constrained", constraints=contradictory)
    with pytest.raises(RuntimeError, match="constrained solve failed"):
        solver.solve(A, b)
    assert solver.solver_info["success"] is False
    assert solver.solver_info["message"]


def test_constrained_solve_that_converges_returns(system):
    A, _, b = system
    solver = LinearSolver(method="constrained", bounds=[(0, None)] * 6)
    x = solver.solve(A, b)
    assert solver.solver_info["success"] and np.all(x >= 0)


# -- robust --


def test_robust_accepts_an_exact_fit(system):
    A, x_true, _ = system
    solver = LinearSolver(method="robust")
    x = solver.solve(A, A @ x_true)
    np.testing.assert_allclose(x, x_true, atol=1e-10)
    info = solver.solver_info
    assert info["n_iter"] == 0 and info["converged"]
    np.testing.assert_array_equal(info["weights"], np.ones(120))


def _robust_reference(A, b, max_iter, tol):
    """The same reweighting with an explicit diagonal weight matrix."""
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(max_iter):
        r = A @ x - b
        scale = 1.4826 * np.median(np.abs(r))
        u = np.abs(r / scale)
        W = np.diag(np.where(u <= 1.345, 1.0, 1.345 / u))
        x_new = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]
        done = np.linalg.norm(x_new - x) < tol
        x = x_new
        if done:
            break
    return x


def test_robust_matches_the_dense_weight_matrix(system):
    A, _, b = system
    b = b.copy()
    b[::15] += 20.0
    solver = LinearSolver(method="robust", max_iter=50)
    x = solver.solve(A, b)
    np.testing.assert_allclose(x, _robust_reference(A, b, 50, 1e-6), atol=1e-9)
    assert solver.solver_info["converged"]
    assert solver.solver_info["weights"].min() < 1.0


def test_robust_reports_an_unconverged_run(system):
    A, _, b = system
    b = b.copy()
    b[::15] += 20.0
    solver = LinearSolver(method="robust", max_iter=1, tol=1e-14)
    solver.solve(A, b)
    assert solver.solver_info["n_iter"] == 1
    assert not solver.solver_info["converged"]


# -- weighted --


def test_weighted_matches_the_dense_weight_matrix(system):
    A, _, b = system
    w = np.linspace(0.5, 2.0, 120)
    x = LinearSolver(method="weighted", weights=w).solve(A, b)
    W = np.diag(np.sqrt(w))
    np.testing.assert_allclose(
        x, np.linalg.lstsq(W @ A, W @ b, rcond=None)[0], atol=1e-10
    )


@pytest.mark.parametrize(
    "kwargs", [dict(method="weighted", weights=np.ones(120)), dict(method="robust")]
)
def test_no_n_by_n_weight_matrix_is_built(system, monkeypatch, kwargs):
    A, _, b = system
    b = b.copy()
    b[::15] += 20.0  # the robust method reweights

    def no_diag(*args, **kwargs):
        raise AssertionError("np.diag called: dense weight matrix")

    monkeypatch.setattr(np, "diag", no_diag)
    assert LinearSolver(**kwargs).solve(A, b).shape == (6,)


def test_weights_of_the_wrong_length_are_refused(system):
    A, _, b = system
    with pytest.raises(ValueError, match="weights have 5 rows"):
        LinearSolver(method="weighted", weights=np.ones(5)).solve(A, b)


# -- the identification entry point --


def test_custom_solver_refuses_an_unused_option():
    pin = pytest.importorskip("pinocchio")
    from test_data_contract_wiring import _Ident, _trajectory

    model = pin.buildSampleModelManipulator()
    ident = _Ident(model)
    ident.trajectory_to_return = _trajectory(model)
    ident.initialize()
    with pytest.raises(ValueError, match="bounds"):
        ident.solve_with_custom_solver(
            method="elastic_net", alpha=0.01, bounds=(0, None)
        )
