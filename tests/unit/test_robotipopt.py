#!/usr/bin/env python3
"""
Unit tests for the robotipopt module.

This module tests the generalized IPOPT framework for robotics applications,
including trajectory optimization and parameter identification problems.

The tests are based on the demonstration examples in demo_robotipopt_usage.py
and cover:
- Configuration classes and presets
- Abstract base classes and their implementations
- Solver initialization and validation
- Problem setup and error handling
- Integration scenarios with mocked IPOPT calls

To run these tests:
    cd /path/to/figaroh
    python -m pytest tests/unit/test_robotipopt.py -v

To run specific test classes:
    python -m pytest tests/unit/test_robotipopt.py::TestIPOPTConfig -v
    python -m pytest tests/unit/test_robotipopt.py::TestRobotIPOPTSolver -v

The tests use mocking to avoid requiring actual IPOPT optimization runs,
making them fast and reliable for continuous integration.
"""

import numpy as np
import pytest
from typing import List, Tuple
from unittest.mock import Mock, patch

# Import the module under test
from figaroh.tools.robotipopt import (
    IPOPTConfig,
    BaseOptimizationProblem,
    RobotIPOPTSolver,
    TrajectoryOptimizationProblem,
    create_trajectory_solver,
)


class SimpleTestProblem(BaseOptimizationProblem):
    """Simple optimization problem for testing."""

    def __init__(self, n_vars: int = 3, n_constraints: int = 2):
        super().__init__("SimpleTestProblem")
        self.n_vars = n_vars
        self.n_constraints = n_constraints

    def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
        """Get variable bounds."""
        lb = [-10.0] * self.n_vars
        ub = [10.0] * self.n_vars
        return lb, ub

    def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
        """Get constraint bounds."""
        if self.n_constraints == 0:
            return [], []
        cl = [-1.0] * self.n_constraints
        cu = [1.0] * self.n_constraints
        return cl, cu

    def get_initial_guess(self) -> List[float]:
        """Get initial guess."""
        return [0.0] * self.n_vars

    def objective(self, x: np.ndarray) -> float:
        """Quadratic objective function."""
        return np.sum(x**2)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Linear constraints."""
        if self.n_constraints == 0:
            return np.array([])
        elif self.n_constraints == 1:
            return np.array([np.sum(x)])
        else:
            return np.array([np.sum(x), np.sum(x**2) - 1.0])


class TrajectoryTestProblem(BaseOptimizationProblem):
    """Trajectory optimization problem for testing."""

    def __init__(self, n_waypoints: int = 4, n_joints: int = 2):
        super().__init__("TrajectoryTestProblem")
        self.n_wps = n_waypoints
        self.n_joints = n_joints
        self.n_vars = self.n_joints * self.n_wps

    def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
        """Get position bounds for all waypoints."""
        lb = [-np.pi] * self.n_vars
        ub = [np.pi] * self.n_vars
        return lb, ub

    def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
        """Get velocity constraint bounds."""
        # Velocity constraints between consecutive waypoints
        n_vel_constraints = self.n_joints * (self.n_wps - 1)
        cl = [-2.0] * n_vel_constraints
        cu = [2.0] * n_vel_constraints
        return cl, cu

    def get_initial_guess(self) -> List[float]:
        """Get initial guess (linear trajectory)."""
        x0 = []
        for i in range(self.n_wps):
            alpha = i / (self.n_wps - 1)
            waypoint = np.ones(self.n_joints) * alpha * 0.5
            x0.extend(waypoint.tolist())
        return x0

    def objective(self, x: np.ndarray) -> float:
        """Minimize trajectory smoothness (jerk)."""
        positions = x.reshape(self.n_wps, self.n_joints)

        if self.n_wps >= 3:
            # Minimize acceleration
            accel = positions[2:] - 2 * positions[1:-1] + positions[:-2]
            return np.sum(accel**2)
        else:
            # Minimize velocity
            return np.sum((positions[1:] - positions[:-1]) ** 2)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Velocity constraints between waypoints."""
        positions = x.reshape(self.n_wps, self.n_joints)
        velocities = positions[1:] - positions[:-1]
        return velocities.flatten()


class ParameterTestProblem(BaseOptimizationProblem):
    """Parameter identification problem for testing."""

    def __init__(self, regressor_matrix: np.ndarray, measured_data: np.ndarray):
        super().__init__("ParameterTestProblem")
        self.W = regressor_matrix
        self.y_measured = measured_data
        self.n_params = self.W.shape[1]

    def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
        """Get parameter bounds."""
        lb = [-100.0] * self.n_params
        ub = [100.0] * self.n_params
        return lb, ub

    def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
        """No constraints for least squares."""
        return [], []

    def get_initial_guess(self) -> List[float]:
        """Get initial guess using least squares."""
        try:
            params_ls = np.linalg.pinv(self.W) @ self.y_measured
            return params_ls.tolist()
        except np.linalg.LinAlgError:
            return [0.0] * self.n_params

    def objective(self, x: np.ndarray) -> float:
        """Minimize prediction error."""
        y_predicted = self.W @ x
        error = y_predicted - self.y_measured
        return np.sum(error**2)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """No constraints."""
        return np.array([])


class TestIPOPTConfig:
    """Test cases for IPOPTConfig class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = IPOPTConfig()

        assert config.tolerance == 1e-6
        assert config.acceptable_tolerance == 1e-4
        assert config.max_iterations == 3000
        assert config.print_level == 5
        assert config.hessian_approximation == "limited-memory"
        assert config.warm_start is True
        assert config.check_derivatives is True
        assert config.linear_solver == "mumps"
        assert isinstance(config.custom_options, dict)

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        custom_options = {b"mu_strategy": b"adaptive"}
        config = IPOPTConfig(
            tolerance=1e-3,
            max_iterations=500,
            print_level=3,
            custom_options=custom_options,
        )

        assert config.tolerance == 1e-3
        assert config.max_iterations == 500
        assert config.print_level == 3
        assert config.custom_options == custom_options

    def test_trajectory_optimization_config(self):
        """Test trajectory optimization preset."""
        config = IPOPTConfig.for_trajectory_optimization()

        assert config.tolerance == 1e-4
        assert config.acceptable_tolerance == 1e-3
        assert config.max_iterations == 1000
        assert config.print_level == 4
        assert config.hessian_approximation == "limited-memory"
        assert b"mu_strategy" in config.custom_options
        assert config.custom_options[b"mu_strategy"] == b"adaptive"

    def test_parameter_identification_config(self):
        """Test parameter identification preset."""
        config = IPOPTConfig.for_parameter_identification()

        assert config.tolerance == 1e-8
        assert config.acceptable_tolerance == 1e-6
        assert config.max_iterations == 5000
        assert config.print_level == 3
        assert config.hessian_approximation == "exact"
        assert b"mu_strategy" in config.custom_options
        assert config.custom_options[b"mu_strategy"] == b"monotone"

    def test_to_ipopt_options(self):
        """Test conversion to IPOPT options dictionary."""
        config = IPOPTConfig(
            tolerance=1e-5,
            max_iterations=100,
            print_level=2,
            custom_options={b"test_option": b"test_value"},
        )

        options = config.to_ipopt_options()

        assert options[b"tol"] == 1e-5
        assert options[b"max_iter"] == 100
        assert options[b"print_level"] == 2
        assert options[b"test_option"] == b"test_value"
        assert b"hessian_approximation" in options
        assert b"linear_solver" in options

    def test_output_file_option(self):
        """Test output file option handling."""
        config = IPOPTConfig(output_file="test_output.txt")
        options = config.to_ipopt_options()

        assert options[b"output_file"] == b"test_output.txt"


class TestBaseOptimizationProblem:
    """Test cases for BaseOptimizationProblem class."""

    def test_initialization(self):
        """Test problem initialization."""
        problem = SimpleTestProblem(n_vars=4, n_constraints=3)

        assert problem.name == "SimpleTestProblem"
        assert hasattr(problem, "iteration_data")
        assert hasattr(problem, "callback_data")
        assert hasattr(problem, "logger")

        # Check iteration data structure
        assert "iterations" in problem.iteration_data
        assert "obj_values" in problem.iteration_data
        assert "constraint_violations" in problem.iteration_data
        assert "solve_times" in problem.iteration_data

    def test_variable_bounds(self):
        """Test variable bounds retrieval."""
        problem = SimpleTestProblem(n_vars=3)
        lb, ub = problem.get_variable_bounds()

        assert len(lb) == 3
        assert len(ub) == 3
        assert all(x == -10.0 for x in lb)
        assert all(x == 10.0 for x in ub)

    def test_constraint_bounds(self):
        """Test constraint bounds retrieval."""
        problem = SimpleTestProblem(n_constraints=2)
        cl, cu = problem.get_constraint_bounds()

        assert len(cl) == 2
        assert len(cu) == 2
        assert all(x == -1.0 for x in cl)
        assert all(x == 1.0 for x in cu)

    def test_constraint_bounds_no_constraints(self):
        """Test constraint bounds with no constraints."""
        problem = SimpleTestProblem(n_constraints=0)
        cl, cu = problem.get_constraint_bounds()

        assert len(cl) == 0
        assert len(cu) == 0

    def test_initial_guess(self):
        """Test initial guess generation."""
        problem = SimpleTestProblem(n_vars=4)
        x0 = problem.get_initial_guess()

        assert len(x0) == 4
        assert all(x == 0.0 for x in x0)

    def test_objective_function(self):
        """Test objective function evaluation."""
        problem = SimpleTestProblem()
        x = np.array([1.0, 2.0, 3.0])
        obj_val = problem.objective(x)

        expected = 1.0**2 + 2.0**2 + 3.0**2  # 14.0
        assert abs(obj_val - expected) < 1e-10

    def test_constraints_function(self):
        """Test constraint function evaluation."""
        problem = SimpleTestProblem(n_constraints=2)
        x = np.array([1.0, 2.0, 3.0])
        constraints = problem.constraints(x)

        assert len(constraints) == 2
        assert abs(constraints[0] - 6.0) < 1e-10  # sum(x)
        assert abs(constraints[1] - 13.0) < 1e-10  # sum(x^2) - 1

    def test_constraints_function_no_constraints(self):
        """Test constraint function with no constraints."""
        problem = SimpleTestProblem(n_constraints=0)
        x = np.array([1.0, 2.0, 3.0])
        constraints = problem.constraints(x)

        assert len(constraints) == 0

    def test_gradient_computation(self):
        """Test gradient computation (automatic differentiation)."""
        problem = SimpleTestProblem()
        x = np.array([1.0, 2.0, 3.0])

        with patch("numdifftools.Gradient") as mock_gradient:
            mock_gradient.return_value.return_value = np.array([2.0, 4.0, 6.0])
            gradient = problem.gradient(x)

            assert len(gradient) == 3
            assert np.allclose(gradient, [2.0, 4.0, 6.0])

    def test_jacobian_computation(self):
        """Test Jacobian computation (automatic differentiation)."""
        problem = SimpleTestProblem(n_constraints=2)
        x = np.array([1.0, 2.0, 3.0])

        with patch("numdifftools.Jacobian") as mock_jacobian:
            mock_jacobian.return_value.return_value = np.array(
                [[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]]
            )
            jacobian = problem.jacobian(x)

            assert jacobian.shape == (2, 3)

    def test_hessian_default(self):
        """Test default Hessian implementation."""
        problem = SimpleTestProblem()
        x = np.array([1.0, 2.0, 3.0])
        lagrange = np.array([0.5, 0.5])
        obj_factor = 1.0

        hessian = problem.hessian(x, lagrange, obj_factor)
        assert hessian is False  # Default implementation

    def test_intermediate_callback(self):
        """Test intermediate callback for iteration tracking."""
        problem = SimpleTestProblem()

        # Simulate IPOPT callback
        result = problem.intermediate(
            alg_mod=1,
            iter_count=5,
            obj_value=10.5,
            inf_pr=1e-3,
            inf_du=1e-4,
            mu=1e-2,
            d_norm=0.1,
            regularization_size=0.0,
            alpha_du=1.0,
            alpha_pr=1.0,
            ls_trials=1,
        )

        assert result is True
        assert 5 in problem.iteration_data["iterations"]
        assert 10.5 in problem.iteration_data["obj_values"]
        assert 1e-3 in problem.iteration_data["constraint_violations"]


class TestRobotIPOPTSolver:
    """Test cases for RobotIPOPTSolver class."""

    def test_initialization(self):
        """Test solver initialization."""
        problem = SimpleTestProblem()
        config = IPOPTConfig()
        solver = RobotIPOPTSolver(problem, config)

        assert solver.problem == problem
        assert solver.config == config
        assert hasattr(solver, "logger")
        assert solver.last_solution is None
        assert solver.last_info is None
        assert isinstance(solver.solve_history, list)

    def test_initialization_default_config(self):
        """Test solver initialization with default config."""
        problem = SimpleTestProblem()
        solver = RobotIPOPTSolver(problem)

        assert isinstance(solver.config, IPOPTConfig)
        assert solver.config.tolerance == 1e-4  # trajectory optimization default

    def test_validate_problem_setup_valid(self):
        """Test problem validation with valid setup."""
        problem = SimpleTestProblem(n_vars=3, n_constraints=2)
        solver = RobotIPOPTSolver(problem)

        is_valid = solver.validate_problem_setup()
        assert is_valid is True

    def test_validate_problem_setup_dimension_mismatch(self):
        """Test problem validation with dimension mismatch."""
        problem = SimpleTestProblem()

        # Mock dimension mismatch
        def bad_bounds():
            return [-1.0], [1.0, 2.0]  # Different lengths

        problem.get_variable_bounds = bad_bounds
        solver = RobotIPOPTSolver(problem)

        is_valid = solver.validate_problem_setup()
        assert is_valid is False

    def test_validate_problem_setup_infinite_objective(self):
        """Test problem validation with infinite objective."""
        problem = SimpleTestProblem()

        # Mock infinite objective
        def bad_objective(x):
            return np.inf

        problem.objective = bad_objective
        solver = RobotIPOPTSolver(problem)

        is_valid = solver.validate_problem_setup()
        assert is_valid is False

    def test_get_solution_summary_no_solution(self):
        """Test solution summary with no solution."""
        problem = SimpleTestProblem()
        solver = RobotIPOPTSolver(problem)

        summary = solver.get_solution_summary()
        assert "No solution available" in summary

    @patch("cyipopt.Problem")
    def test_solve_success(self, mock_ipopt):
        """Test successful solve."""
        problem = SimpleTestProblem(n_vars=2, n_constraints=0)
        solver = RobotIPOPTSolver(problem)

        # Mock IPOPT solve
        mock_nlp = Mock()
        mock_ipopt.return_value = mock_nlp
        mock_nlp.solve.return_value = (
            np.array([0.5, -0.5]),  # solution
            {"status": 0, "status_msg": "Optimal", "obj_val": 0.25},  # info
        )

        success, results = solver.solve()

        assert success is True
        assert "x_opt" in results
        assert "obj_val" in results
        assert "solve_time" in results
        assert results["status"] == 0
        assert np.allclose(results["x_opt"], [0.5, -0.5])

    @patch("cyipopt.Problem")
    def test_solve_failure(self, mock_ipopt):
        """Test failed solve."""
        problem = SimpleTestProblem()
        solver = RobotIPOPTSolver(problem)

        # Mock IPOPT solve failure (status not in [-1, 0, 1])
        mock_nlp = Mock()
        mock_ipopt.return_value = mock_nlp
        mock_nlp.solve.return_value = (
            np.array([0.0, 0.0]),  # solution
            {"status": -2, "status_msg": "Failed", "obj_val": 1e10},  # info
        )

        success, results = solver.solve()

        assert success is False
        assert results["status"] == -2

    @patch("cyipopt.Problem")
    def test_solve_exception(self, mock_ipopt):
        """Test solve with exception."""
        problem = SimpleTestProblem()
        solver = RobotIPOPTSolver(problem)

        # Mock IPOPT exception
        mock_ipopt.side_effect = Exception("IPOPT error")

        success, results = solver.solve()

        assert success is False
        assert "error" in results
        assert "IPOPT error" in results["error"]


class TestTrajectoryOptimizationProblem:
    """Test cases for TrajectoryOptimizationProblem class."""

    def create_concrete_trajectory_problem(self):
        """Create concrete trajectory problem for testing."""

        class ConcreteTrajectoryProblem(TrajectoryOptimizationProblem):
            def get_variable_bounds(self):
                return [-1.0, -1.0], [1.0, 1.0]

            def get_constraint_bounds(self):
                return [], []

            def get_initial_guess(self):
                return [0.0, 0.0]

        return ConcreteTrajectoryProblem("TestTrajectory")

    def test_initialization(self):
        """Test trajectory problem initialization."""
        problem = self.create_concrete_trajectory_problem()

        assert problem.name == "TestTrajectory"
        assert hasattr(problem, "trajectory_data")
        assert "times" in problem.trajectory_data
        assert "positions" in problem.trajectory_data
        assert "velocities" in problem.trajectory_data
        assert "accelerations" in problem.trajectory_data

    def test_custom_objective_function(self):
        """Test setting custom objective function."""
        problem = self.create_concrete_trajectory_problem()

        def custom_obj(x):
            return np.sum(x**3)

        problem.set_objective_function(custom_obj)

        x = np.array([1.0, 2.0])
        obj_val = problem.objective(x)

        assert abs(obj_val - 9.0) < 1e-10  # 1^3 + 2^3 = 9

    def test_custom_constraint_function(self):
        """Test setting custom constraint function."""
        problem = self.create_concrete_trajectory_problem()

        def custom_cons(x):
            return np.array([np.sum(x) - 1.0])

        problem.set_constraint_function(custom_cons)

        x = np.array([0.3, 0.7])
        constraints = problem.constraints(x)

        assert len(constraints) == 1
        assert abs(constraints[0] - 0.0) < 1e-10  # 0.3 + 0.7 - 1.0 = 0

    def test_objective_not_implemented(self):
        """Test objective function without custom implementation."""
        problem = self.create_concrete_trajectory_problem()

        x = np.array([1.0, 2.0])

        with pytest.raises(NotImplementedError):
            problem.objective(x)

    def test_constraints_not_implemented(self):
        """Test constraints function without custom implementation."""
        problem = self.create_concrete_trajectory_problem()

        x = np.array([1.0, 2.0])

        with pytest.raises(NotImplementedError):
            problem.constraints(x)


class TestCreateTrajectorySolver:
    """Test cases for create_trajectory_solver factory function."""

    def test_factory_function(self):
        """Test trajectory solver creation via factory function."""

        def obj_func(x):
            return np.sum(x**2)

        def cons_func(x):
            return np.array([np.sum(x) - 1.0])

        def bounds_func():
            var_bounds = ([-2.0, -2.0], [2.0, 2.0])
            cons_bounds = ([0.0], [2.0])
            return var_bounds, cons_bounds

        def initial_func():
            return [0.5, 0.5]

        solver = create_trajectory_solver(
            objective_func=obj_func,
            constraint_func=cons_func,
            get_bounds_func=bounds_func,
            initial_guess_func=initial_func,
            name="TestFactory",
        )

        assert isinstance(solver, RobotIPOPTSolver)
        assert isinstance(solver.problem, TrajectoryOptimizationProblem)
        assert solver.problem.name == "TestFactory"

        # Test problem functions
        x = np.array([0.5, 0.5])
        obj_val = solver.problem.objective(x)
        assert abs(obj_val - 0.5) < 1e-10  # 0.5^2 + 0.5^2 = 0.5

        constraints = solver.problem.constraints(x)
        assert len(constraints) == 1
        assert abs(constraints[0] - 0.0) < 1e-10  # 0.5 + 0.5 - 1.0 = 0


class TestIntegrationExamples:
    """Integration tests based on demonstration examples."""

    def test_simple_trajectory_problem(self):
        """Test simple trajectory optimization problem."""
        problem = TrajectoryTestProblem(n_waypoints=4, n_joints=2)

        # Test problem setup
        lb, ub = problem.get_variable_bounds()
        assert len(lb) == 8  # 4 waypoints * 2 joints
        assert len(ub) == 8

        cl, cu = problem.get_constraint_bounds()
        assert len(cl) == 6  # 3 velocity constraints * 2 joints
        assert len(cu) == 6

        x0 = problem.get_initial_guess()
        assert len(x0) == 8

        # Test function evaluations
        x = np.array(x0)
        obj_val = problem.objective(x)
        assert np.isfinite(obj_val)

        constraints = problem.constraints(x)
        assert len(constraints) == 6
        assert all(np.isfinite(constraints))

    def test_parameter_identification_problem(self):
        """Test parameter identification problem."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples, n_params = 20, 5
        W = np.random.randn(n_samples, n_params)
        true_params = np.random.randn(n_params) * 5
        y_measured = W @ true_params + np.random.randn(n_samples) * 0.01

        problem = ParameterTestProblem(W, y_measured)

        # Test problem setup
        lb, ub = problem.get_variable_bounds()
        assert len(lb) == n_params
        assert len(ub) == n_params

        cl, cu = problem.get_constraint_bounds()
        assert len(cl) == 0  # No constraints
        assert len(cu) == 0

        x0 = problem.get_initial_guess()
        assert len(x0) == n_params

        # Test function evaluations
        x = np.array(x0)
        obj_val = problem.objective(x)
        assert np.isfinite(obj_val)

        constraints = problem.constraints(x)
        assert len(constraints) == 0

    @patch("cyipopt.Problem")
    def test_end_to_end_simple_problem(self, mock_ipopt):
        """Test end-to-end simple optimization."""
        problem = SimpleTestProblem(n_vars=2, n_constraints=0)
        config = IPOPTConfig.for_trajectory_optimization()
        config.print_level = 0  # Suppress output in tests
        solver = RobotIPOPTSolver(problem, config)

        # Mock successful IPOPT solve
        mock_nlp = Mock()
        mock_ipopt.return_value = mock_nlp
        mock_nlp.solve.return_value = (
            np.array([0.0, 0.0]),  # optimal solution for x^2 + y^2
            {"status": 0, "status_msg": "Optimal", "obj_val": 0.0},
        )

        # Validate and solve
        assert solver.validate_problem_setup() is True
        success, results = solver.solve()

        assert success is True
        assert results["obj_val"] == 0.0
        assert len(results["x_opt"]) == 2
        assert "solve_time" in results
        assert "iterations" in results


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_gradient_computation_error(self):
        """Test gradient computation with error."""
        problem = SimpleTestProblem()

        with patch("numdifftools.Gradient") as mock_gradient:
            mock_gradient.side_effect = Exception("Gradient error")
            x = np.array([1.0, 2.0, 3.0])
            gradient = problem.gradient(x)

            # Should return zeros on error
            assert len(gradient) == 3
            assert np.allclose(gradient, [0.0, 0.0, 0.0])

    def test_jacobian_computation_error(self):
        """Test Jacobian computation with error."""
        problem = SimpleTestProblem(n_constraints=2)

        with patch("numdifftools.Jacobian") as mock_jacobian:
            mock_jacobian.side_effect = Exception("Jacobian error")
            x = np.array([1.0, 2.0, 3.0])
            jacobian = problem.jacobian(x)

            # Should return identity matrix as fallback
            assert jacobian.shape == (2, 3)

    def test_constraint_validation_error(self):
        """Test constraint validation with error."""
        problem = SimpleTestProblem()

        # Mock constraint function that returns wrong dimension
        def bad_constraints(x):
            return np.array([1.0, 2.0, 3.0])  # Wrong dimension

        problem.constraints = bad_constraints
        solver = RobotIPOPTSolver(problem)

        is_valid = solver.validate_problem_setup()
        assert is_valid is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
