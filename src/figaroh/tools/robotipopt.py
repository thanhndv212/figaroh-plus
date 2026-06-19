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
General IPOPT-based optimization framework for robotics applications.

This module provides a comprehensive, flexible framework for setting up and
solving nonlinear optimization problems using IPOPT (Interior Point OPTimizer),
with specialized support for robotics applications including trajectory
optimization, parameter identification, and general robot optimization
problems.

Key Features:
    - Unified IPOPT interface with robotics-specific configurations
    - Automatic differentiation support for gradients and Jacobians
    - Built-in problem validation and result analysis
    - Specialized classes for trajectory optimization problems
    - Factory functions for rapid problem setup
    - Comprehensive logging and iteration tracking

Main Classes:
    - IPOPTConfig: Configuration management for IPOPT solver settings
    - BaseOptimizationProblem: Abstract base class for optimization problems
    - RobotIPOPTSolver: High-level solver interface with result analysis
    - TrajectoryOptimizationProblem: Specialized class for trajectory problems

Usage Examples:
    Basic trajectory optimization:
        ```python
        from figaroh.tools.robotipopt import IPOPTConfig, RobotIPOPTSolver

        # Create custom problem class
        class MyProblem(BaseOptimizationProblem):
            def objective(self, x):
                return np.sum(x**2)  # Minimize sum of squares

            def constraints(self, x):
                return np.array([x[0] + x[1] - 1.0])  # x[0] + x[1] = 1

            # ... implement other required methods

        # Solve the problem
        problem = MyProblem()
        config = IPOPTConfig.for_trajectory_optimization()
        solver = RobotIPOPTSolver(problem, config)
        success, results = solver.solve()
        ```

    Using the factory function:
        ```python
        from figaroh.tools.robotipopt import create_trajectory_solver

        def my_objective(x):
            return np.sum(x**2)

        def my_constraints(x):
            return np.array([x[0] + x[1] - 1.0])

        def get_bounds():
            var_bounds = ([-10, -10], [10, 10])
            cons_bounds = ([0], [0])
            return var_bounds, cons_bounds

        def initial_guess():
            return [0.5, 0.5]

        solver = create_trajectory_solver(
            my_objective, my_constraints, get_bounds, initial_guess
        )
        success, results = solver.solve()
        ```

    Complete robotics example:
        ```python
        import numpy as np
        from figaroh.tools.robotipopt import (
            BaseOptimizationProblem, RobotIPOPTSolver, IPOPTConfig
        )

        class RobotTrajectoryProblem(BaseOptimizationProblem):
            def __init__(self, robot, waypoints, active_joints):
                super().__init__("RobotTrajectory")
                self.robot = robot
                self.waypoints = waypoints
                self.active_joints = active_joints
                self.n_joints = len(active_joints)
                self.n_waypoints = len(waypoints)
                self.n_vars = self.n_joints * self.n_waypoints

            def get_variable_bounds(self):
                # Joint limits for all waypoints
                lb = [-np.pi] * self.n_vars
                ub = [np.pi] * self.n_vars
                return lb, ub

            def get_constraint_bounds(self):
                # Boundary conditions (start/end positions)
                n_constraints = 2 * self.n_joints
                return [0] * n_constraints, [0] * n_constraints

            def get_initial_guess(self):
                # Linear interpolation between start and end
                return np.linspace(
                    self.waypoints[0], self.waypoints[-1],
                    self.n_waypoints * self.n_joints
                )

            def objective(self, x):
                # Minimize trajectory smoothness (squared accelerations)
                q = x.reshape(self.n_waypoints, self.n_joints)
                acc = np.diff(q, n=2, axis=0)  # Second differences
                return np.sum(acc**2)

            def constraints(self, x):
                # Enforce boundary conditions
                q = x.reshape(self.n_waypoints, self.n_joints)
                constraints = np.concatenate([
                    q[0] - self.waypoints[0],   # Start position
                    q[-1] - self.waypoints[-1]  # End position
                ])
                return constraints

        # Usage
        problem = RobotTrajectoryProblem(robot, waypoints, active_joints)
        config = IPOPTConfig.for_trajectory_optimization()
        solver = RobotIPOPTSolver(problem, config)

        if solver.validate_problem_setup():
            success, results = solver.solve()
            if success:
                optimal_trajectory = results['x_opt'].reshape(
                    problem.n_waypoints, problem.n_joints
                )
                print("Optimization successful!")
                print(f"Final objective: {results['obj_val']:.6f}")
                print(solver.get_solution_summary())
        ```

Dependencies:
    - cyipopt: Python interface to IPOPT solver
    - numpy: Numerical computations
    - numdifftools: Automatic differentiation (fallback)

References:
    - IPOPT documentation: https://coin-or.github.io/Ipopt/
    - cyipopt: https://github.com/mechmotum/cyipopt
"""

import time
import logging
import numpy as np
import numdifftools as nd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class IPOPTConfig:
    """
    Configuration parameters for IPOPT solver.

    This class provides a convenient way to set IPOPT solver options
    with sensible defaults for robotics applications. It includes
    predefined configurations for common robotics optimization scenarios.

    The configuration is designed to work well with:
    - Trajectory optimization problems (smooth, continuous trajectories)
    - Parameter identification (high-precision parameter estimation)
    - General robotics optimization (balanced performance/accuracy)

    Attributes:
        tolerance (float): Primary convergence tolerance for optimization.
            Default: 1e-6. Smaller values increase precision but runtime.
        acceptable_tolerance (float): Fallback tolerance if primary fails.
            Default: 1e-4. Used when primary tolerance cannot be achieved.
        max_iterations (int): Maximum number of optimization iterations.
            Default: 3000. Increase for complex problems.
        max_cpu_time (float): Maximum CPU time in seconds.
            Default: 1e6. Prevents infinite loops in difficult problems.
        print_level (int): IPOPT output verbosity (0-12).
            Default: 5. Higher values provide more detailed output.
        output_file (Optional[str]): File to save IPOPT output.
            Default: None. Useful for debugging and analysis.
        hessian_approximation (str): Hessian computation method.
            Default: "limited-memory". Options: "exact", "limited-memory".
        warm_start (bool): Use previous solution as starting point.
            Default: True. Speeds up subsequent optimizations.
        check_derivatives (bool): Enable derivative checking.
            Default: True. Helps detect implementation errors.
        linear_solver (str): Linear algebra solver backend.
            Default: "mumps". Options: "mumps", "ma27", "ma57", "ma77".
        custom_options (Dict[str, Any]): Additional IPOPT options.
            Default: {}. For advanced users to set specialized options.

    Examples:
        Basic usage:
            ```python
            # Use default configuration
            config = IPOPTConfig()

            # Customize specific parameters
            config = IPOPTConfig(
                tolerance=1e-8,
                max_iterations=5000,
                print_level=3
            )
            ```

        Predefined configurations:
            ```python
            # For trajectory optimization (speed-focused)
            config = IPOPTConfig.for_trajectory_optimization()

            # For parameter identification (precision-focused)
            config = IPOPTConfig.for_parameter_identification()
            ```

        Custom IPOPT options:
            ```python
            config = IPOPTConfig(
                custom_options={
                    "mu_strategy": "adaptive",
                    "nlp_scaling_method": "gradient-based"
                }
            )
            ```

    Note:
        All string options in custom_options should be provided as strings.
        They will be automatically encoded to bytes for IPOPT compatibility.
    """

    # Convergence tolerances
    tolerance: float = 1e-6
    acceptable_tolerance: float = 1e-4

    # Iteration limits
    max_iterations: int = 3000
    max_cpu_time: float = 1e6

    # Output control
    print_level: int = 5
    output_file: Optional[str] = None

    # Algorithm options
    hessian_approximation: str = "limited-memory"
    warm_start: bool = True
    check_derivatives: bool = True

    # Linear solver options
    linear_solver: str = "mumps"

    # Custom options
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def to_ipopt_options(self) -> Dict[str, Any]:
        """
        Convert configuration to IPOPT option dictionary.

        Transforms the configuration parameters into the format expected
        by IPOPT, including proper encoding of string options to bytes.

        Returns:
            Dict[str, Any]: Dictionary of IPOPT options ready for use
                           with cyipopt.Problem.add_option()

        Note:
            All string values are automatically encoded to bytes as required
            by the IPOPT C interface via cyipopt.
        """
        options = {
            # Basic convergence settings
            b"tol": self.tolerance,
            b"acceptable_tol": self.acceptable_tolerance,
            b"max_iter": self.max_iterations,
            b"max_cpu_time": self.max_cpu_time,
            # Output settings
            b"print_level": self.print_level,
            # Algorithm settings
            b"hessian_approximation": self.hessian_approximation.encode(),
            b"linear_solver": self.linear_solver.encode(),
            # Derivative checking
            b"check_derivatives_for_naninf": (
                b"yes" if self.check_derivatives else b"no"
            ),
            # Warm start
            b"warm_start_init_point": b"yes" if self.warm_start else b"no",
            b"acceptable_obj_change_tol": self.acceptable_tolerance,
        }

        # Add output file if specified
        if self.output_file:
            options[b"output_file"] = self.output_file.encode()

        # Add custom options
        for key, value in self.custom_options.items():
            if isinstance(key, str):
                key = key.encode()
            if isinstance(value, str):
                value = value.encode()
            options[key] = value

        return options

    @classmethod
    def for_trajectory_optimization(cls) -> "IPOPTConfig":
        """
        Create configuration optimized for trajectory optimization problems.

        This configuration prioritizes convergence speed over extreme
        precision, making it suitable for real-time or interactive
        trajectory planning. Uses adaptive barrier parameter strategy
        for better convergence.

        Returns:
            IPOPTConfig: Optimized configuration for trajectory problems.
        """
        return cls(
            tolerance=1e-4,
            acceptable_tolerance=1e-3,
            max_iterations=1000,
            print_level=4,
            hessian_approximation="limited-memory",
            custom_options={
                b"mu_strategy": b"adaptive",
                b"adaptive_mu_globalization": b"obj-constr-filter",
            },
        )

    @classmethod
    def for_parameter_identification(cls) -> "IPOPTConfig":
        """
        Create configuration optimized for parameter identification problems.

        This configuration prioritizes high precision over speed, making it
        suitable for accurate parameter estimation in robotics applications.
        Uses exact Hessian computation and monotone barrier strategy for
        numerical stability.

        Returns:
            IPOPTConfig: Optimized configuration for parameter identification.
        """
        return cls(
            tolerance=1e-8,
            acceptable_tolerance=1e-6,
            max_iterations=5000,
            print_level=3,
            hessian_approximation="exact",
            custom_options={
                b"mu_strategy": b"monotone",
                b"fixed_variable_treatment": b"make_parameter",
            },
        )


class BaseOptimizationProblem(ABC):
    """
    Abstract base class for IPOPT optimization problems.

    This class defines the interface that all IPOPT problems must implement.
    Subclasses should implement the objective function, constraints, and their
    derivatives (or use automatic differentiation).

    The class provides default implementations for gradient and Jacobian
    computation using automatic differentiation via numdifftools. For better
    performance, subclasses can override these methods with analytical
    derivatives.

    Attributes:
        name (str): Human-readable name for the optimization problem.
        logger (logging.Logger): Logger instance for this problem.
        iteration_data (Dict): Storage for tracking optimization iterations.
        callback_data (Dict): Storage for passing data between methods.

    Required Methods (must be implemented by subclasses):
        - get_variable_bounds(): Return variable bounds as (lower, upper)
        - get_constraint_bounds(): Return constraint bounds as (lower, upper)
        - get_initial_guess(): Return initial guess for optimization variables
        - objective(x): Evaluate objective function at point x
        - constraints(x): Evaluate constraint functions at point x

    Optional Methods (have default implementations):
        - gradient(x): Compute objective gradient (uses auto-diff by default)
        - jacobian(x): Compute constraint Jacobian (uses auto-diff by default)
        - hessian(x, lagrange, obj_factor): Compute Hessian
          (IPOPT approx by default)
        - intermediate(...): Callback for iteration tracking

    Examples:
        Basic implementation:
            ```python
            class QuadraticProblem(BaseOptimizationProblem):
                def get_variable_bounds(self):
                    return ([-10, -10], [10, 10])

                def get_constraint_bounds(self):
                    return ([0], [0])  # Equality constraint

                def get_initial_guess(self):
                    return [1.0, 1.0]

                def objective(self, x):
                    return x[0]**2 + x[1]**2  # Minimize sum of squares

                def constraints(self, x):
                    return np.array([x[0] + x[1] - 1.0])  # x[0] + x[1] = 1
            ```

        With custom derivatives:
            ```python
            class CustomProblem(BaseOptimizationProblem):
                # ... implement required methods ...

                def gradient(self, x):
                    # Custom analytical gradient
                    return np.array([2*x[0], 2*x[1]])

                def jacobian(self, x):
                    # Custom analytical Jacobian
                    return np.array([[1.0, 1.0]])
            ```
    """

    def __init__(self, name: str = "OptimizationProblem"):
        """Initialize the optimization problem."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Storage for iteration tracking
        self.iteration_data = {
            "iterations": [],
            "obj_values": [],
            "constraint_violations": [],
            "solve_times": [],
        }

        # Callback storage for passing data between methods
        self.callback_data: Dict[str, Any] = {}

    @abstractmethod
    def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get bounds for optimization variables.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        pass

    @abstractmethod
    def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get bounds for constraints.

        Returns:
            Tuple of (constraint_lower_bounds, constraint_upper_bounds)
        """
        pass

    @abstractmethod
    def get_initial_guess(self) -> List[float]:
        """
        Get initial guess for optimization variables.

        Returns:
            Initial guess as list of floats
        """
        pass

    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate objective function.

        Args:
            x: Optimization variables

        Returns:
            Objective function value
        """
        pass

    @abstractmethod
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate constraint functions.

        Args:
            x: Optimization variables

        Returns:
            Constraint function values
        """
        pass

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function.

        Default implementation uses automatic differentiation.
        Override for custom implementations.

        Args:
            x: Optimization variables

        Returns:
            Gradient vector
        """
        try:
            return nd.Gradient(self.objective)(x)
        except Exception as e:
            self.logger.warning(f"Error computing gradient: {e}")
            return np.zeros_like(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of constraint functions.

        Default implementation uses automatic differentiation.
        Override for custom implementations.

        Args:
            x: Optimization variables

        Returns:
            Jacobian matrix
        """
        try:
            constraints = self.constraints(x)
            if len(constraints) == 0:
                # No constraints
                return np.zeros((0, len(x)))

            jac = nd.Jacobian(self.constraints)(x)

            # Handle case where constraint returns scalar
            if constraints.ndim == 0 or len(constraints) == 1:
                jac = jac.reshape(1, -1)

            self.logger.debug(f"Constraint jacobian shape: {jac.shape}")
            return jac
        except Exception as e:
            self.logger.warning(f"Error computing jacobian: {e}")
            # Return identity matrix as fallback
            constraints = self.constraints(x)
            n_constraints = len(constraints) if hasattr(constraints, "__len__") else 1
            return np.eye(n_constraints, len(x))

    def hessian(
        self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
    ) -> Union[bool, np.ndarray]:
        """
        Compute Hessian of Lagrangian.

        Default implementation returns False to use IPOPT's approximation.
        Override for custom implementations.

        Args:
            x: Optimization variables
            lagrange: Lagrange multipliers
            obj_factor: Objective scaling factor

        Returns:
            Hessian matrix or False to use approximation
        """
        return False

    def intermediate(
        self,
        alg_mod: int,
        iter_count: int,
        obj_value: float,
        inf_pr: float,
        inf_du: float,
        mu: float,
        d_norm: float,
        regularization_size: float,
        alpha_du: float,
        alpha_pr: float,
        ls_trials: int,
    ) -> bool:
        """
        Intermediate callback for iteration tracking.

        Args:
            alg_mod: Algorithm mode
            iter_count: Iteration count
            obj_value: Current objective value
            inf_pr: Primal infeasibility
            inf_du: Dual infeasibility
            mu: Barrier parameter
            d_norm: Step size
            regularization_size: Regularization parameter
            alpha_du: Dual step size
            alpha_pr: Primal step size
            ls_trials: Line search trials

        Returns:
            True to continue optimization, False to stop
        """
        self.iteration_data["iterations"].append(iter_count)
        self.iteration_data["obj_values"].append(obj_value)
        self.iteration_data["constraint_violations"].append(max(inf_pr, inf_du))

        return True


class RobotIPOPTSolver:
    """
    General IPOPT solver for robotics optimization problems.

    This class provides a high-level interface for solving optimization
    problems using IPOPT, with built-in support for result analysis,
    error handling, and problem validation specifically designed for
    robotics applications.

    Features:
        - Automatic problem setup and configuration
        - Built-in problem validation before solving
        - Comprehensive result analysis and logging
        - Solution history tracking for multiple solves
        - Robotics-specific error handling and diagnostics
        - Support for iterative solving and warm starts

    Attributes:
        problem (BaseOptimizationProblem): The optimization problem to solve.
        config (IPOPTConfig): IPOPT solver configuration.
        logger (logging.Logger): Logger for solver operations.
        last_solution (np.ndarray): Most recent optimization solution.
        last_info (Dict): Most recent IPOPT solver information.
        solve_history (List[Dict]): History of all solve attempts.

    Examples:
        Basic usage:
            ```python
            # Create problem and solver
            problem = MyOptimizationProblem()
            solver = RobotIPOPTSolver(problem)

            # Solve the problem
            success, results = solver.solve()

            if success:
                print(f"Optimal solution: {results['x_opt']}")
                print(f"Objective value: {results['obj_val']}")
            ```

        With custom configuration:
            ```python
            # Create custom configuration
            config = IPOPTConfig(
                tolerance=1e-8,
                max_iterations=5000,
                print_level=3
            )

            # Create solver with custom config
            solver = RobotIPOPTSolver(problem, config)

            # Validate problem before solving
            if solver.validate_problem_setup():
                success, results = solver.solve()
            ```

        Multiple solves with warm start:
            ```python
            solver = RobotIPOPTSolver(problem)

            # First solve
            success1, results1 = solver.solve()

            # Modify problem parameters...
            problem.update_parameters(new_params)

            # Second solve (uses warm start automatically)
            success2, results2 = solver.solve()

            # Access solve history
            print(f"Solved {len(solver.solve_history)} problems")
            ```
    """

    def __init__(
        self, problem: BaseOptimizationProblem, config: Optional[IPOPTConfig] = None
    ):
        """
        Initialize the IPOPT solver.

        Args:
            problem: Optimization problem to solve
            config: IPOPT configuration (default: trajectory optimization
                   config)
        """
        self.problem = problem
        self.config = config or IPOPTConfig.for_trajectory_optimization()
        self.logger = logging.getLogger(f"{__name__}.RobotIPOPTSolver")

        # Results storage
        self.last_solution = None
        self.last_info = None
        self.solve_history = []

    def solve(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Solve the optimization problem.

        Performs the complete optimization process including problem setup,
        IPOPT configuration, solving, and result analysis. The method
        handles all IPOPT interactions and provides comprehensive error
        handling and logging.

        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple containing:
                - success (bool): True if optimization succeeded,
                  False otherwise
                - results (Dict[str, Any]): Dictionary containing:
                    * 'x_opt': Optimal solution vector
                    * 'obj_val': Final objective function value
                    * 'status': IPOPT exit status code
                    * 'status_msg': Human-readable status message
                    * 'solve_time': Total optimization time in seconds
                    * 'iterations': Number of iterations performed
                    * 'iteration_data': Detailed iteration history
                    * 'callback_data': Custom data from problem callbacks
                    * 'ipopt_info': Complete IPOPT solver information
                    * 'success': Copy of success flag for convenience

        Raises:
            Exception: If critical errors occur during problem setup
                      or solving. Non-critical errors are caught and
                      returned in results.

        Note:
            IPOPT status codes for success: -1 (solved to acceptable level),
            0 (solved), 1 (solved to acceptable level). All other codes
            indicate various types of failures or early termination.
        """
        try:
            # Import cyipopt only when needed
            try:
                import cyipopt
            except ImportError:
                raise ImportError(
                    "cyipopt is required for IPOPT optimization. "
                    "Install with: pip install cyipopt"
                )

            self.logger.info(f"Setting up IPOPT problem: {self.problem.name}")

            # Get problem dimensions and bounds
            x0 = self.problem.get_initial_guess()
            lb, ub = self.problem.get_variable_bounds()
            cl, cu = self.problem.get_constraint_bounds()

            self.logger.info(
                f"Problem dimensions: {len(x0)} variables, " f"{len(cl)} constraints"
            )

            # Create IPOPT problem
            nlp = cyipopt.Problem(
                n=len(x0),
                m=len(cl),
                problem_obj=self.problem,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu,
            )

            # Apply configuration
            for key, value in self.config.to_ipopt_options().items():
                nlp.add_option(key, value)

            # Solve optimization
            self.logger.info("Starting IPOPT optimization...")
            start_time = time.time()
            x_opt, info = nlp.solve(x0)
            solve_time = time.time() - start_time

            # Store results
            self.last_solution = x_opt
            self.last_info = info

            # Analyze results
            # Acceptable IPOPT exit codes
            success = info["status"] in [-1, 0, 1]

            self.logger.info(f"Optimization completed in {solve_time:.2f} seconds")
            self.logger.info(f"Status: {info['status']} - {info['status_msg']}")
            self.logger.info(f"Final objective: {info['obj_val']:.6e}")

            # Prepare results dictionary
            results = {
                "success": success,
                "x_opt": x_opt,
                "obj_val": info["obj_val"],
                "status": info["status"],
                "status_msg": info["status_msg"],
                "solve_time": solve_time,
                "iterations": len(self.problem.iteration_data["iterations"]),
                "iteration_data": self.problem.iteration_data.copy(),
                "callback_data": self.problem.callback_data.copy(),
                "ipopt_info": info,
            }

            # Store in history
            self.solve_history.append(results)

            return success, results

        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            return False, {"error": str(e)}

    def get_solution_summary(self) -> str:
        """Get a summary of the last solution."""
        if self.last_solution is None:
            return "No solution available"

        info = self.last_info
        return (
            f"IPOPT Solution Summary:\n"
            f"  Status: {info['status']} - {info['status_msg']}\n"
            f"  Objective: {info['obj_val']:.6e}\n"
            f"  Iterations: {len(self.problem.iteration_data['iterations'])}\n"
            f"  Variables: {len(self.last_solution)}\n"
            f"  Constraints: {len(self.problem.get_constraint_bounds()[0])}\n"
        )

    def validate_problem_setup(self) -> bool:
        """
        Validate that the optimization problem is properly set up.

        Returns:
            True if problem appears valid, False otherwise
        """
        try:
            self.logger.info("Validating problem setup...")

            # Check dimensions
            x0 = self.problem.get_initial_guess()
            lb, ub = self.problem.get_variable_bounds()
            cl, cu = self.problem.get_constraint_bounds()

            if len(lb) != len(ub) or len(lb) != len(x0):
                self.logger.error("Variable bounds dimension mismatch")
                return False

            if len(cl) != len(cu):
                self.logger.error("Constraint bounds dimension mismatch")
                return False

            # Test function evaluations
            obj_val = self.problem.objective(np.array(x0))
            if not np.isfinite(obj_val):
                self.logger.error(
                    f"Objective function returns non-finite value: {obj_val}"
                )
                return False

            constraints = self.problem.constraints(np.array(x0))
            if len(constraints) != len(cl):
                self.logger.error("Constraint function dimension mismatch")
                return False

            if not np.all(np.isfinite(constraints)):
                self.logger.error("Constraint function returns non-finite values")
                return False

            self.logger.info("Problem validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Problem validation failed: {e}")
            return False


class TrajectoryOptimizationProblem(BaseOptimizationProblem):
    """
    Specialized optimization problem for trajectory optimization.

    This class provides a framework for trajectory optimization problems
    with common patterns like waypoint parameterization and constraint
    handling.
    """

    def __init__(self, name: str = "TrajectoryOptimization"):
        """Initialize trajectory optimization problem."""
        super().__init__(name)

        # Trajectory-specific data
        self.trajectory_data = {
            "times": None,
            "positions": None,
            "velocities": None,
            "accelerations": None,
        }

    def set_objective_function(self, obj_func: Callable[[np.ndarray], float]):
        """Set custom objective function."""
        self._custom_objective = obj_func

    def set_constraint_function(self, cons_func: Callable[[np.ndarray], np.ndarray]):
        """Set custom constraint function."""
        self._custom_constraints = cons_func

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective function."""
        if hasattr(self, "_custom_objective"):
            return self._custom_objective(x)
        else:
            raise NotImplementedError(
                "Must implement objective function or set custom function"
            )

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate constraint functions."""
        if hasattr(self, "_custom_constraints"):
            return self._custom_constraints(x)
        else:
            raise NotImplementedError(
                "Must implement constraints function or set custom function"
            )


def create_trajectory_solver(
    objective_func: Callable[[np.ndarray], float],
    constraint_func: Callable[[np.ndarray], np.ndarray],
    get_bounds_func: Callable[[], Tuple[Tuple[List, List], Tuple[List, List]]],
    initial_guess_func: Callable[[], List[float]],
    name: str = "CustomTrajectory",
) -> RobotIPOPTSolver:
    """
    Factory function to create a trajectory optimization solver.

    Args:
        objective_func: Function to evaluate objective
        constraint_func: Function to evaluate constraints
        get_bounds_func: Function that returns ((var_lb, var_ub),
                        (cons_lb, cons_ub))
        initial_guess_func: Function that returns initial guess
        name: Problem name

    Returns:
        Configured IPOPT solver
    """

    class CustomProblem(TrajectoryOptimizationProblem):
        def __init__(self):
            super().__init__(name)
            self.set_objective_function(objective_func)
            self.set_constraint_function(constraint_func)
            self._get_bounds = get_bounds_func
            self._get_initial = initial_guess_func

        def get_variable_bounds(self):
            return self._get_bounds()[0]

        def get_constraint_bounds(self):
            return self._get_bounds()[1]

        def get_initial_guess(self):
            return self._get_initial()

    problem = CustomProblem()
    config = IPOPTConfig.for_trajectory_optimization()
    return RobotIPOPTSolver(problem, config)
