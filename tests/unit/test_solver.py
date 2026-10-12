# Copyright [2021-2026] Thanh Nguyen
# Copyright [2022-2024] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the linear solver module.

Tests cover:
- Basic least squares solving
- Regularization methods (Ridge, Lasso, Elastic Net)
- Constrained optimization
- Robust regression
- Weighted least squares
- Solution quality metrics
"""

import numpy as np
import pytest
from figaroh.tools.solver import LinearSolver, solve_linear_system


class TestLinearSolver:
    """Test suite for LinearSolver class."""

    @pytest.fixture
    def overdetermined_system(self):
        """Create a simple overdetermined linear system."""
        np.random.seed(42)
        n_samples = 200
        n_features = 20

        A = np.random.randn(n_samples, n_features)
        x_true = np.random.randn(n_features)
        b = A @ x_true + 0.1 * np.random.randn(n_samples)

        return A, b, x_true

    def test_lstsq_solver(self, overdetermined_system):
        """Test standard least squares solver."""
        A, b, x_true = overdetermined_system

        solver = LinearSolver(method="lstsq")
        x = solver.solve(A, b)

        # Check solution is close to true parameters
        assert x.shape == x_true.shape
        assert np.linalg.norm(x - x_true) < 1.0

        # Check solver info
        assert "rmse" in solver.solver_info
        assert "r_squared" in solver.solver_info
        assert "rank" in solver.solver_info

    def test_qr_solver(self, overdetermined_system):
        """Test QR decomposition solver."""
        A, b, x_true = overdetermined_system

        solver = LinearSolver(method="qr")
        x = solver.solve(A, b)

        assert x.shape == x_true.shape
        assert np.linalg.norm(x - x_true) < 1.0
        assert "rank" in solver.solver_info

    def test_svd_solver(self, overdetermined_system):
        """Test SVD solver."""
        A, b, x_true = overdetermined_system

        solver = LinearSolver(method="svd")
        x = solver.solve(A, b)

        assert x.shape == x_true.shape
        assert np.linalg.norm(x - x_true) < 1.0
        assert "singular_values" in solver.solver_info
        assert "condition_number" in solver.solver_info

    def test_ridge_regression(self, overdetermined_system):
        """Test Ridge regression."""
        A, b, x_true = overdetermined_system

        solver = LinearSolver(method="ridge", alpha=0.1)
        x = solver.solve(A, b)

        assert x.shape == x_true.shape
        assert solver.solver_info["regularization"] == "L2"
        assert solver.solver_info["alpha"] == 0.1

    def test_tikhonov_regularization(self, overdetermined_system):
        """Test Tikhonov regularization."""
        A, b, x_true = overdetermined_system

        solver = LinearSolver(method="tikhonov", alpha=0.05)
        x = solver.solve(A, b)

        assert x.shape == x_true.shape
        assert solver.solver_info["regularization"] == "Tikhonov"

    def test_constrained_with_bounds(self):
        """Test constrained optimization with box constraints."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        A = np.random.randn(n_samples, n_features)
        x_true = np.abs(np.random.randn(n_features))  # Positive params
        b = A @ x_true + 0.05 * np.random.randn(n_samples)

        # Enforce positivity constraint
        bounds = [(0, None) for _ in range(n_features)]

        solver = LinearSolver(method="constrained", bounds=bounds)
        x = solver.solve(A, b)

        # Check all parameters are non-negative
        assert np.all(x >= 0)
        assert x.shape == x_true.shape

    def test_robust_regression_with_outliers(self):
        """Test robust regression with outliers."""
        np.random.seed(42)
        n_samples = 200
        n_features = 15

        A = np.random.randn(n_samples, n_features)
        x_true = np.random.randn(n_features)
        b = A @ x_true + 0.1 * np.random.randn(n_samples)

        # Add outliers
        outlier_idx = np.random.choice(n_samples, size=20, replace=False)
        b[outlier_idx] += 5.0 * np.random.randn(20)

        # Standard least squares (affected by outliers)
        solver_lstsq = LinearSolver(method="lstsq")
        x_lstsq = solver_lstsq.solve(A, b)

        # Robust regression
        solver_robust = LinearSolver(method="robust", max_iter=50)
        x_robust = solver_robust.solve(A, b)

        # Robust should be closer to true parameters
        error_lstsq = np.linalg.norm(x_lstsq - x_true)
        error_robust = np.linalg.norm(x_robust - x_true)

        # Robust error should be smaller or comparable
        assert error_robust <= error_lstsq * 1.2
        assert "weights" in solver_robust.solver_info

    def test_weighted_least_squares(self):
        """Test weighted least squares."""
        np.random.seed(42)
        n_samples = 150
        n_features = 12

        A = np.random.randn(n_samples, n_features)
        x_true = np.random.randn(n_features)
        b = A @ x_true + 0.2 * np.random.randn(n_samples)

        # Give more weight to first half of samples
        weights = np.ones(n_samples)
        weights[: n_samples // 2] = 2.0

        solver = LinearSolver(method="weighted", weights=weights)
        x = solver.solve(A, b)

        assert x.shape == x_true.shape

    def test_solve_linear_system_convenience(self):
        """Test convenience function."""
        np.random.seed(42)
        A = np.random.randn(100, 10)
        b = np.random.randn(100)

        x, info = solve_linear_system(A, b, method="lstsq")

        assert x.shape == (10,)
        assert "rmse" in info
        assert "r_squared" in info

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be one of"):
            LinearSolver(method="invalid_method")

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        A = np.random.randn(100, 10)
        b = np.random.randn(50)  # Wrong size

        solver = LinearSolver()
        with pytest.raises(ValueError, match="must have same number of rows"):
            solver.solve(A, b)

    def test_underdetermined_warning(self):
        """Test warning for underdetermined systems."""
        A = np.random.randn(10, 20)  # More unknowns than equations
        b = np.random.randn(10)

        solver = LinearSolver()
        with pytest.warns(UserWarning, match="underdetermined"):
            solver.solve(A, b)

    def test_verbose_output(self, overdetermined_system, capsys):
        """Test verbose output."""
        A, b, _ = overdetermined_system

        solver = LinearSolver(method="lstsq", verbose=True)
        solver.solve(A, b)

        captured = capsys.readouterr()
        assert "Linear Solver" in captured.out
        assert "RMSE" in captured.out

    def test_solution_quality_metrics(self, overdetermined_system):
        """Test that solution quality metrics are computed."""
        A, b, _ = overdetermined_system

        solver = LinearSolver(method="lstsq")
        solver.solve(A, b)

        info = solver.solver_info

        # Check all required metrics are present
        assert "residual_norm" in info
        assert "rss" in info
        assert "rmse" in info
        assert "r_squared" in info
        assert "n_samples" in info
        assert "n_features" in info

        # Check values are reasonable
        assert info["rmse"] >= 0
        assert 0 <= info["r_squared"] <= 1
        assert info["n_samples"] == A.shape[0]
        assert info["n_features"] == A.shape[1]


class TestRobotIdentificationScenarios:
    """Test scenarios similar to robot parameter identification."""

    def test_large_overdetermined_system(self):
        """Test with large overdetermined system typical in robotics."""
        np.random.seed(42)
        n_samples = 2000  # Many trajectory samples
        n_features = 60  # Dynamic parameters

        A = np.random.randn(n_samples, n_features)
        x_true = np.abs(np.random.randn(n_features)) * 10
        b = A @ x_true + 0.5 * np.random.randn(n_samples)

        solver = LinearSolver(method="svd")
        x = solver.solve(A, b)

        assert x.shape == (n_features,)
        assert solver.solver_info["rmse"] < 1.0

    def test_ill_conditioned_system(self):
        """Test with ill-conditioned matrix (near-dependent columns)."""
        np.random.seed(42)
        n_samples = 500
        n_features = 30

        # Create near-dependent columns
        A_base = np.random.randn(n_samples, n_features // 2)
        A = np.hstack(
            [
                A_base,
                A_base + 0.01 * np.random.randn(n_samples, n_features // 2),
            ]
        )

        x_true = np.random.randn(n_features)
        b = A @ x_true + 0.1 * np.random.randn(n_samples)

        # Ridge should handle ill-conditioning better
        solver = LinearSolver(method="ridge", alpha=0.1)
        x = solver.solve(A, b)

        assert x.shape == x_true.shape
        # Condition number should be reported
        assert "condition_number" in solver.solver_info

    def test_positive_parameters_constraint(self):
        """Test enforcing positive parameters (like masses, inertias)."""
        np.random.seed(42)
        n_samples = 300
        n_features = 20

        A = np.random.randn(n_samples, n_features)
        x_true = np.abs(np.random.randn(n_features)) * 5
        b = A @ x_true + 0.2 * np.random.randn(n_samples)

        # Physical constraint: all parameters must be positive
        bounds = [(0, 100) for _ in range(n_features)]

        solver = LinearSolver(method="constrained", bounds=bounds)
        x = solver.solve(A, b)

        # Verify all parameters are positive and bounded
        assert np.all(x >= 0)
        assert np.all(x <= 100)

    def test_mixed_regularization_and_bounds(self):
        """Test combining regularization with bounds."""
        np.random.seed(42)
        n_samples = 400
        n_features = 25

        A = np.random.randn(n_samples, n_features)
        x_true = np.abs(np.random.randn(n_features))
        b = A @ x_true + 0.15 * np.random.randn(n_samples)

        # Use ridge first to get stable solution
        solver_ridge = LinearSolver(method="ridge", alpha=0.05)
        x_ridge = solver_ridge.solve(A, b)

        # Ridge solution quality
        assert x_ridge.shape == (n_features,)

        # Then apply bounds in constrained optimization
        bounds = [(0, None) for _ in range(n_features)]
        solver_constrained = LinearSolver(method="constrained", bounds=bounds)
        x_constrained = solver_constrained.solve(A, b)

        assert np.all(x_constrained >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
