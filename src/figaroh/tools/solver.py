# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

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
Multivariate linear solver for robot parameter identification.

This module provides advanced linear regression solvers optimized for
overdetermined systems (thin, tall matrices) with support for:
- Multiple regularization methods (Ridge, Lasso, Elastic Net, Tikhonov)
- Linear equality and inequality constraints
- Robust regression methods
- Physical parameter bounds
- Iterative refinement for improved accuracy
"""

import logging
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import warnings

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LinearSolver:
    """
    Advanced linear solver for overdetermined systems: Ax = b

    Optimized for robot identification where A is typically:
    - Dense (not sparse)
    - Large (many samples)
    - Thin (more rows than columns, overdetermined)

    Attributes:
        method (str): Solving method to use
        regularization (str): Type of regularization
        alpha (float): Regularization strength
        constraints (dict): Linear constraints on solution
        bounds (tuple): Box constraints on variables
        solver_info (dict): Information about the solution
    """

    METHODS = [
        "lstsq",  # Standard least squares (fastest)
        "qr",  # QR decomposition
        "svd",  # Singular value decomposition (most stable)
        "ridge",  # Ridge regression (L2 regularization)
        "lasso",  # Lasso regression (L1 regularization)
        "elastic_net",  # Elastic net (L1 + L2)
        "tikhonov",  # Tikhonov regularization (general L2)
        "constrained",  # Constrained least squares
        "robust",  # Robust regression (iterative reweighting)
        "weighted",  # Weighted least squares
    ]

    def __init__(
        self,
        method="lstsq",
        regularization=None,
        alpha=0.0,
        l1_ratio=0.5,
        weights=None,
        constraints=None,
        bounds=None,
        max_iter=1000,
        tol=1e-6,
        verbose=False,
    ):
        """
        Initialize linear solver.

        Args:
            method (str): Solving method (see METHODS)
            regularization (str): 'l1', 'l2', 'elastic_net', or None
            alpha (float): Regularization strength (>=0)
            l1_ratio (float): Elastic net mixing (0=Ridge, 1=Lasso)
            weights (ndarray): Sample weights for weighted least squares
            constraints (dict): Linear constraints:
                - 'A_eq': Equality constraint matrix (A_eq @ x = b_eq)
                - 'b_eq': Equality constraint vector
                - 'A_ineq': Inequality constraint matrix (A_ineq @ x <= b_ineq)
                - 'b_ineq': Inequality constraint vector
            bounds (tuple): Box constraints (lower, upper) for each variable
            max_iter (int): Maximum iterations for iterative methods
            tol (float): Convergence tolerance
            verbose (bool): Print solver information
        """
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")

        self.method = method
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights = weights
        self.constraints = constraints or {}
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.solver_info = {}

    def solve(self, A, b):
        """
        Solve the linear system Ax = b.

        Args:
            A (ndarray): Design matrix (n_samples, n_features)
            b (ndarray): Target vector (n_samples,)

        Returns:
            ndarray: Solution vector x (n_features,)

        Raises:
            ValueError: If inputs are incompatible
            np.linalg.LinAlgError: If solution fails
        """
        # Validate inputs
        A, b = self._validate_inputs(A, b)

        # Select and execute solving method
        if self.method == "lstsq":
            x = self._solve_lstsq(A, b)
        elif self.method == "qr":
            x = self._solve_qr(A, b)
        elif self.method == "svd":
            x = self._solve_svd(A, b)
        elif self.method == "ridge":
            x = self._solve_ridge(A, b)
        elif self.method == "lasso":
            x = self._solve_lasso(A, b)
        elif self.method == "elastic_net":
            x = self._solve_elastic_net(A, b)
        elif self.method == "tikhonov":
            x = self._solve_tikhonov(A, b)
        elif self.method == "constrained":
            x = self._solve_constrained(A, b)
        elif self.method == "robust":
            x = self._solve_robust(A, b)
        elif self.method == "weighted":
            x = self._solve_weighted(A, b)
        else:
            raise ValueError(f"Method {self.method} not implemented")

        # Compute solution quality metrics
        self._compute_solution_quality(A, b, x)

        if self.verbose:
            self._print_solution_info()

        return x

    def _validate_inputs(self, A, b):
        """Validate and prepare inputs."""
        # Convert to numpy arrays
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Check dimensions
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")

        if b.ndim == 1:
            b = b.reshape(-1, 1)
        elif b.ndim != 2:
            raise ValueError(f"b must be 1D or 2D, got shape {b.shape}")

        # Check compatibility
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"A and b must have same number of rows. "
                f"Got A: {A.shape}, b: {b.shape}"
            )

        # Warn if underdetermined
        if A.shape[0] < A.shape[1]:
            warnings.warn(
                f"System is underdetermined: {A.shape[0]} equations, "
                f"{A.shape[1]} unknowns. Solution may not be unique.",
                UserWarning,
            )

        return A, b.ravel()

    def _solve_lstsq(self, A, b):
        """Standard least squares using numpy."""
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        self.solver_info.update(
            {"residuals": residuals, "rank": rank, "singular_values": s}
        )
        return x

    def _solve_qr(self, A, b):
        """QR decomposition least squares."""
        Q, R = np.linalg.qr(A)
        x = linalg.solve_triangular(R, Q.T @ b)

        self.solver_info["rank"] = np.linalg.matrix_rank(R)
        return x

    def _solve_svd(self, A, b):
        """SVD-based least squares (most numerically stable)."""
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # Compute effective rank with tolerance
        tol = s[0] * max(A.shape) * np.finfo(float).eps
        rank = np.sum(s > tol)

        # Solve using pseudo-inverse
        s_inv = np.zeros_like(s)
        s_inv[:rank] = 1.0 / s[:rank]
        x = Vt.T @ (s_inv[:, np.newaxis] * (U.T @ b[:, np.newaxis]))

        self.solver_info.update(
            {
                "rank": rank,
                "singular_values": s,
                "condition_number": s[0] / s[rank - 1] if rank > 0 else np.inf,
            }
        )
        return x.ravel()

    def _solve_ridge(self, A, b):
        """Ridge regression (L2 regularization)."""
        n_features = A.shape[1]

        # Ridge: (A^T A + alpha*I) x = A^T b
        AtA = A.T @ A
        Atb = A.T @ b

        # Add regularization to diagonal
        AtA_reg = AtA + self.alpha * np.eye(n_features)

        x = linalg.solve(AtA_reg, Atb, assume_a="pos")

        self.solver_info["regularization"] = "L2"
        self.solver_info["alpha"] = self.alpha
        return x

    def _solve_lasso(self, A, b):
        """Lasso regression (L1 regularization) using coordinate descent."""
        try:
            from sklearn.linear_model import Lasso

            model = Lasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=False,
            )
            model.fit(A, b)

            self.solver_info["regularization"] = "L1"
            self.solver_info["alpha"] = self.alpha
            self.solver_info["n_iter"] = model.n_iter_
            return model.coef_
        except ImportError:
            warnings.warn(
                "sklearn not available, falling back to ridge regression",
                UserWarning,
            )
            return self._solve_ridge(A, b)

    def _solve_elastic_net(self, A, b):
        """Elastic Net (L1 + L2 regularization)."""
        try:
            from sklearn.linear_model import ElasticNet

            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=False,
            )
            model.fit(A, b)

            self.solver_info["regularization"] = "Elastic Net"
            self.solver_info["alpha"] = self.alpha
            self.solver_info["l1_ratio"] = self.l1_ratio
            self.solver_info["n_iter"] = model.n_iter_
            return model.coef_
        except ImportError:
            warnings.warn(
                "sklearn not available, falling back to ridge regression",
                UserWarning,
            )
            return self._solve_ridge(A, b)

    def _solve_tikhonov(self, A, b):
        """Tikhonov regularization with custom regularization matrix."""
        n_features = A.shape[1]

        # Allow custom regularization matrix L
        L = self.constraints.get("L", np.eye(n_features))

        # Tikhonov: (A^T A + alpha*L^T L) x = A^T b
        AtA = A.T @ A
        Atb = A.T @ b
        LtL = L.T @ L

        AtA_reg = AtA + self.alpha * LtL
        x = linalg.solve(AtA_reg, Atb, assume_a="pos")

        self.solver_info["regularization"] = "Tikhonov"
        self.solver_info["alpha"] = self.alpha
        return x

    def _solve_constrained(self, A, b):
        """Constrained least squares using scipy."""
        n_features = A.shape[1]

        # Objective: minimize ||Ax - b||^2
        def objective(x):
            residual = A @ x - b
            return 0.5 * np.sum(residual**2)

        def jacobian(x):
            return A.T @ (A @ x - b)

        # Initial guess
        x0 = np.zeros(n_features)

        # Prepare constraints for scipy
        constraints = []

        # Equality constraints: A_eq @ x = b_eq
        if "A_eq" in self.constraints and "b_eq" in self.constraints:
            A_eq = self.constraints["A_eq"]
            b_eq = self.constraints["b_eq"]
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: A_eq @ x - b_eq,
                    "jac": lambda x: A_eq,
                }
            )

        # Inequality constraints: A_ineq @ x <= b_ineq
        if "A_ineq" in self.constraints and "b_ineq" in self.constraints:
            A_ineq = self.constraints["A_ineq"]
            b_ineq = self.constraints["b_ineq"]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: b_ineq - A_ineq @ x,
                    "jac": lambda x: -A_ineq,
                }
            )

        # Solve with bounds and constraints
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=jacobian,
            bounds=self.bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.solver_info.update(
            {
                "success": result.success,
                "message": result.message,
                "n_iter": result.nit,
                "fun": result.fun,
            }
        )

        return result.x

    def _solve_robust(self, A, b):
        """Robust regression using iteratively reweighted least squares."""
        n_samples, n_features = A.shape

        # Initialize with standard least squares
        x = self._solve_lstsq(A, b)

        # Iterative reweighting
        for iteration in range(self.max_iter):
            # Compute residuals
            residuals = A @ x - b

            # Compute robust weights using Huber function
            scale = 1.4826 * np.median(np.abs(residuals))
            if scale < self.tol:
                break

            normalized_residuals = residuals / scale
            weights = np.where(
                np.abs(normalized_residuals) <= 1.345,
                1.0,
                1.345 / np.abs(normalized_residuals),
            )

            # Weighted least squares
            W = np.diag(weights)
            x_new = linalg.lstsq(W @ A, W @ b)[0]

            # Check convergence
            if np.linalg.norm(x_new - x) < self.tol:
                x = x_new
                break

            x = x_new

        self.solver_info.update({"n_iter": iteration + 1, "weights": weights})

        return x

    def _solve_weighted(self, A, b):
        """Weighted least squares."""
        if self.weights is None:
            raise ValueError(
                "Weights must be provided for weighted least squares"
            )

        W = np.sqrt(self.weights)
        if W.ndim == 1:
            W = np.diag(W)

        # Weighted problem: minimize ||W(Ax - b)||^2
        A_weighted = W @ A
        b_weighted = W @ b

        x = self._solve_lstsq(A_weighted, b_weighted)
        return x

    def _compute_solution_quality(self, A, b, x):
        """Compute quality metrics for the solution."""
        # Residuals
        residuals = A @ x - b

        # Residual sum of squares
        rss = np.sum(residuals**2)

        # Root mean square error
        rmse = np.sqrt(rss / len(b))

        # R-squared
        ss_tot = np.sum((b - np.mean(b)) ** 2)
        r_squared = 1 - (rss / ss_tot) if ss_tot > 0 else 0

        # Condition number (if not already computed)
        if "condition_number" not in self.solver_info:
            try:
                self.solver_info["condition_number"] = np.linalg.cond(A)
            except np.linalg.LinAlgError:
                self.solver_info["condition_number"] = np.inf

        self.solver_info.update(
            {
                "residual_norm": np.linalg.norm(residuals),
                "rss": rss,
                "rmse": rmse,
                "r_squared": r_squared,
                "n_samples": A.shape[0],
                "n_features": A.shape[1],
            }
        )

    def _print_solution_info(self):
        """Print solution information."""
        print("")
        print("=" * 60)
        print(f"Linear Solver: {self.method}")
        print("=" * 60)

        if "condition_number" in self.solver_info:
            cond = self.solver_info["condition_number"]
            print(f"Condition number: {cond:.2e}")

        if "rank" in self.solver_info:
            print(f"Matrix rank: {self.solver_info['rank']}")

        print(f"RMSE: {self.solver_info['rmse']:.6f}")
        print(f"R²: {self.solver_info['r_squared']:.6f}")

        if "n_iter" in self.solver_info:
            print(f"Iterations: {self.solver_info['n_iter']}")

        if "regularization" in self.solver_info:
            print(f"Regularization: {self.solver_info['regularization']}")
            print(f"Alpha: {self.solver_info['alpha']:.2e}")

        print("=" * 60)


def solve_linear_system(A, b, method="lstsq", **kwargs):
    """
    Convenience function to solve linear system Ax = b.

    Args:
        A (ndarray): Design matrix (n_samples, n_features)
        b (ndarray): Target vector (n_samples,)
        method (str): Solving method
        **kwargs: Additional arguments for LinearSolver

    Returns:
        tuple: (solution x, solver_info dict)

    Example:
        >>> A = np.random.randn(1000, 50)  # 1000 samples, 50 features
        >>> b = np.random.randn(1000)
        >>> x, info = solve_linear_system(A, b, method='ridge', alpha=0.1)
    """
    solver = LinearSolver(method=method, **kwargs)
    x = solver.solve(A, b)
    return x, solver.solver_info


# Export public API
__all__ = [
    "LinearSolver",
    "solve_linear_system",
]
