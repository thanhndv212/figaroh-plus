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
Base Optimal Trajectory Generation Framework

This module provides base classes for optimal trajectory generation with
configuration management, parameter computation, constraint handling, and
IPOPT-based optimization. This framework can be extended for different robots.
"""

import logging
from abc import abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple, Any

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
)
from figaroh.tools.qrdecomposition import build_baseRegressor
from figaroh.utils.cubic_spline import (
    CubicSpline,
    WaypointsGeneration,
    calc_torque,
)
from figaroh.tools.robotipopt import (
    IPOPTConfig,
    BaseOptimizationProblem,
    RobotIPOPTSolver,
)
from figaroh.optimal.config import load_param
from figaroh.optimal.base_parameter import BaseParameterComputer
from figaroh.optimal.contraints import TrajectoryConstraintManager


class BaseOptimalTrajectory:
    """
    Base class for IPOPT-based optimal trajectory generation.

    Features:
    - Modular design with separated concerns
    - Better error handling and logging
    - Configuration validation
    - Cleaner interfaces

    This base class can be extended for specific robots by implementing
    robot-specific configuration loading and constraint handling.
    """

    def __init__(
        self,
        robot,
        active_joints: List[str],
        config_file: str = "config/robot_config.yaml",
    ):
        """Initialize the optimal trajectory generator."""
        self.robot = robot
        self.model = self.robot.model
        self.active_joints = active_joints

        # Set up logger (configuration should be done by application, not library)
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.trajectory_config, self.identif_config = load_param(
            self.robot, config_file
        )

        # # Initialize components
        # self.initialize()

        # Results storage
        self.results = {
            "T_F": [],
            "P_F": [],
            "V_F": [],
            "A_F": [],
            "iteration_data": [],
            "final_regressor_shape": None,
        }

    def initialize(self):
        """Initialize trajectory generation components."""
        # Create soft limit pool
        n_active_joints = len(self.active_joints)
        self.soft_lim_pool = np.full(
            (3, n_active_joints), self.trajectory_config["soft_lim"]
        )

        # Initialize cubic spline and waypoint generation
        self.CB = CubicSpline(
            self.robot,
            self.trajectory_config["n_wps"],
            self.active_joints,
            self.trajectory_config["soft_lim"],
        )
        self.WP = WaypointsGeneration(
            self.robot,
            self.trajectory_config["n_wps"],
            self.active_joints,
            self.trajectory_config["soft_lim"],
        )

        # Initialize specialized components
        self.base_computer = BaseParameterComputer(
            self.robot, self.identif_config, self.active_joints, self.soft_lim_pool
        )
        self.constraint_manager = TrajectoryConstraintManager(
            self.robot, self.CB, self.trajectory_config, self.identif_config
        )

        # Compute base parameters
        self.idx_e, self.idx_b = self.base_computer.compute_base_indices()

        self.logger.info(
            f"BaseOptimalTrajectory initialized with {len(self.idx_b)} base parameters"
        )

    def solve(self, stack_reps: int = 2) -> Dict[str, Any]:
        """
        Solve the optimal trajectory generation problem.

        Args:
            stack_reps: Number of trajectory segments to stack

        Returns:
            Dict containing trajectories and optimization info
        """
        self.logger.info(
            f"Starting optimal trajectory generation with {stack_reps} segments..."
        )

        try:
            # Initialize
            self.WP.gen_rand_pool(self.soft_lim_pool)
            wp_init = np.zeros(len(self.CB.act_idxq))
            vel_wp_init = np.zeros(len(self.CB.act_idxv))
            acc_wp_init = np.zeros(len(self.CB.act_idxv))

            # Random initial position
            for idx in range(len(self.CB.act_idxq)):
                wp_init[idx] = np.random.choice(self.WP.pool_q[:, idx], 1)[0]

            W_stack = None

            for s_rep in range(stack_reps):
                self.logger.info(f"Optimizing segment {s_rep + 1}/{stack_reps}")
                self.logger.info(f"Initial waypoint: {wp_init}")

                success = self._solve_segment(
                    s_rep, wp_init, vel_wp_init, acc_wp_init, W_stack
                )

                if not success:
                    self.logger.error(f"Failed to solve segment {s_rep + 1}")
                    break

                # Update for next segment
                if s_rep < stack_reps - 1:  # Not the last segment
                    wp_init, W_stack = self._prepare_next_segment()

            self.logger.info(
                f"Completed! Generated {len(self.results['T_F'])} trajectory segments"
            )
            self.results["final_regressor_shape"] = (
                W_stack.shape if W_stack is not None else None
            )

            return self.results

        except Exception as e:
            self.logger.error(f"Error in solve: {e}")
            raise

    def objective_function(
        self, X, opt_cb, tps, vel_wps, acc_wps, wp_init, W_stack=None
    ):
        """Objective function: condition number of base regressor matrix."""
        try:
            # Reshape and arrange waypoints
            X = np.array(X)
            wps_X = np.reshape(
                X,
                (self.trajectory_config["n_wps"] - 1, len(self.active_joints)),
            )
            wps = np.vstack((wp_init, wps_X))
            wps = wps.transpose()

            # Generate full trajectory configuration
            t_f, p_f, v_f, a_f = self.CB.get_full_config(
                self.trajectory_config["freq"], tps, wps, vel_wps, acc_wps
            )

            # Store in callback dictionary
            opt_cb.update({"t_f": t_f, "p_f": p_f, "v_f": v_f, "a_f": a_f})

            # Build stacked base regressor and return condition number
            W_b = self._stack_base_regressors(p_f, v_f, a_f, W_stack=W_stack)
            return np.linalg.cond(W_b)

        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return 1e10  # Return large penalty value

    @abstractmethod
    def create_ipopt_problem(
        self,
        n_joints,
        n_wps,
        Ns,
        tps,
        vel_wps,
        acc_wps,
        wp_init,
        vel_wp_init,
        acc_wp_init,
        W_stack,
    ):
        """Create IPOPT problem instance. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_ipopt_problem")

    def _stack_base_regressors(self, q, v, a, W_stack=None) -> np.ndarray:
        """Build base regressor matrix."""
        try:
            W = build_regressor_basic(self.robot, q, v, a, self.identif_config)
            W_e_ = build_regressor_reduced(W, self.idx_e)
            W_b_ = build_baseRegressor(W_e_, self.idx_b)

            if isinstance(W_stack, np.ndarray):
                W_b_ = np.vstack((W_stack, W_b_))

            return W_b_
        except Exception as e:
            self.logger.error(f"Error building base regressor: {e}")
            raise

    def _generate_feasible_initial_guess(self, wp_init, vel_wp_init, acc_wp_init):
        """Generate a feasible initial guess for optimization."""
        self.logger.info("Generating feasible initial trajectory...")

        count = 0
        is_constr_violated = True

        while is_constr_violated and count < self.trajectory_config["max_attempts"]:
            count += 1
            if count % 100 == 0:
                self.logger.info(
                    f"Attempt {count} to find feasible initial trajectory..."
                )

            try:
                # Generate random waypoints
                wps, vel_wps, acc_wps = self.WP.gen_rand_wp(
                    wp_init, vel_wp_init, acc_wp_init
                )

                # Generate time points
                tps = np.matrix(
                    [
                        self.trajectory_config["t_s"] * i_wp
                        for i_wp in range(self.trajectory_config["n_wps"])
                    ]
                ).transpose()

                # Get full configuration
                t_i, p_i, v_i, a_i = self.CB.get_full_config(
                    self.trajectory_config["freq"], tps, wps, vel_wps, acc_wps
                )

                # Compute torques and check constraints
                tau_i = calc_torque(p_i.shape[0], self.robot, p_i, v_i, a_i)
                tau_i = np.reshape(tau_i, (v_i.shape[1], v_i.shape[0])).transpose()
                is_constr_violated = self.CB.check_cfg_constraints(p_i, v_i, tau_i)

            except Exception as e:
                self.logger.warning(f"Error in attempt {count}: {e}")
                continue

        if count >= self.trajectory_config["max_attempts"]:
            self.logger.warning(
                f"Could not find feasible initial trajectory after {self.trajectory_config['max_attempts']} attempts"
            )
        else:
            self.logger.info(
                f"Found feasible initial trajectory after {count} attempts"
            )

        return wps, vel_wps, acc_wps, tps, t_i, p_i, v_i, a_i

    def _solve_segment(self, s_rep, wp_init, vel_wp_init, acc_wp_init, W_stack) -> bool:
        """Solve a single trajectory segment."""
        try:
            # Generate feasible initial guess
            wps, vel_wps, acc_wps, tps, t_i, p_i, v_i, a_i = (
                self._generate_feasible_initial_guess(wp_init, vel_wp_init, acc_wp_init)
            )

            # Adjust time points for stacking
            tps = self.trajectory_config["t_s"] * s_rep + tps

            # Create and solve IPOPT problem - This should be implemented by subclasses
            problem = self.create_ipopt_problem(
                len(self.active_joints),
                self.trajectory_config["n_wps"],
                p_i.shape[0],
                tps,
                vel_wps,
                acc_wps,
                wp_init,
                vel_wp_init,
                acc_wp_init,
                W_stack,
            )

            success, result_data = problem.solve_with_waypoints(wps)

            if success:
                self.results["T_F"].append(result_data["t_f"])
                self.results["P_F"].append(result_data["p_f"])
                self.results["V_F"].append(result_data["v_f"])
                self.results["A_F"].append(result_data["a_f"])
                self.results["iteration_data"].append(result_data["iter_data"])
                self.logger.info(f"Segment {s_rep + 1} completed successfully!")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error solving segment {s_rep + 1}: {e}")
            return False

    def _prepare_next_segment(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare initial conditions for next segment."""
        # Get last result
        last_result = self.results["iteration_data"][-1]
        wp_init = last_result["final_waypoint"]

        # Stack regressor
        last_p_f = self.results["P_F"][-1]
        last_v_f = self.results["V_F"][-1]
        last_a_f = self.results["A_F"][-1]

        # Convert back to full configuration for regressor building
        # This is a simplified version - in practice you'd need to handle this more carefully
        W_stack = self._stack_base_regressors(last_p_f, last_v_f, last_a_f)

        return wp_init, W_stack

    def plot_results(self):
        """Plot optimal trajectory results using unified results manager."""
        if not self.results["T_F"]:
            self.logger.warning("No trajectory data to plot")
            return

        try:
            from .results_manager import ResultsManager

            # Initialize results manager
            robot_name = getattr(self, "robot_name", self.robot.model.name)
            results_manager = ResultsManager("optimal_trajectory", robot_name)

            # Calculate overall condition number
            condition_number = getattr(self, "final_condition_number", 0.0)
            if (
                condition_number == 0.0
                and hasattr(self, "results")
                and "condition_numbers" in self.results
            ):
                condition_number = (
                    self.results["condition_numbers"][-1]
                    if self.results["condition_numbers"]
                    else 0.0
                )

            # Plot using unified manager
            results_manager.plot_optimal_trajectory_results(
                trajectories=self.results,
                condition_number=condition_number,
                joint_names=[f"Joint {i+1}" for i in range(len(self.CB.act_Jid))],
                title="Optimal Trajectory Generation Results",
            )

        except ImportError:
            # Fallback to existing plotting
            try:
                # Create subplots
                n_joints = len(self.CB.act_Jid)
                fig, axes = plt.subplots(
                    n_joints, 3, sharex=True, figsize=(15, 2 * n_joints)
                )
                if n_joints == 1:
                    axes = axes.reshape(1, -1)

                fig.suptitle("Optimal Trajectory Results", fontsize=16)

                # Plot each segment
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.results["T_F"])))

                for seg_idx, (T, P, V, A) in enumerate(
                    zip(
                        self.results["T_F"],
                        self.results["P_F"],
                        self.results["V_F"],
                        self.results["A_F"],
                    )
                ):
                    color = colors[seg_idx]
                    label = f"Segment {seg_idx + 1}"

                    for joint_idx in range(n_joints):
                        axes[joint_idx, 0].plot(
                            T, P[:, joint_idx], color=color, label=label
                        )
                        axes[joint_idx, 1].plot(
                            T, V[:, joint_idx], color=color, label=label
                        )
                        axes[joint_idx, 2].plot(
                            T, A[:, joint_idx], color=color, label=label
                        )

                # Set labels and formatting
                for joint_idx in range(n_joints):
                    axes[joint_idx, 0].set_ylabel(
                        f"Joint {joint_idx+1}\nPosition (rad)"
                    )
                    axes[joint_idx, 1].set_ylabel(
                        f"Joint {joint_idx+1}\nVelocity (rad/s)"
                    )
                    axes[joint_idx, 2].set_ylabel(
                        f"Joint {joint_idx+1}\nAcceleration (rad/s²)"
                    )

                    if joint_idx == 0:
                        for col in range(3):
                            axes[joint_idx, col].legend()

                    for col in range(3):
                        axes[joint_idx, col].grid(True, alpha=0.3)

                axes[-1, 0].set_xlabel("Time (s)")
                axes[-1, 1].set_xlabel("Time (s)")
                axes[-1, 2].set_xlabel("Time (s)")

                plt.tight_layout()
                plt.show()

            except Exception as e:
                self.logger.error(f"Error plotting results: {e}")

    def save_results(self, output_dir="results"):
        """Save optimal trajectory results using unified results manager."""
        if not self.results["T_F"]:
            self.logger.warning("No trajectory data to save")
            return

        try:
            from .results_manager import ResultsManager

            # Initialize results manager
            robot_name = getattr(self, "robot_name", self.robot.model.name)
            results_manager = ResultsManager("optimal_trajectory", robot_name)

            # Calculate overall condition number
            condition_number = getattr(self, "final_condition_number", 0.0)
            if (
                condition_number == 0.0
                and hasattr(self, "results")
                and "condition_numbers" in self.results
            ):
                condition_number = (
                    self.results["condition_numbers"][-1]
                    if self.results["condition_numbers"]
                    else 0.0
                )

            # Prepare results dictionary
            results_dict = {
                "trajectory_segments": len(self.results["T_F"]),
                "condition_number": float(condition_number),
                "joint_names": [f"Joint {i+1}" for i in range(len(self.CB.act_Jid))],
                "configuration": self.CB.identif_config,
                "time_segments": [t.tolist() for t in self.results["T_F"]],
                "position_segments": [p.tolist() for p in self.results["P_F"]],
                "velocity_segments": [v.tolist() for v in self.results["V_F"]],
                "acceleration_segments": [a.tolist() for a in self.results["A_F"]],
            }

            # Add condition number history if available
            if "condition_numbers" in self.results:
                results_dict["condition_number_history"] = [
                    float(c) for c in self.results["condition_numbers"]
                ]

            # Save using unified manager
            saved_files = results_manager.save_results(
                results_dict, output_dir, save_formats=["yaml", "npz"]
            )

            self.logger.info(f"Trajectory results saved successfully")
            return saved_files

        except ImportError:
            # Fallback to basic saving
            import os
            import yaml

            os.makedirs(output_dir, exist_ok=True)

            # Basic results dictionary
            robot_name = getattr(self, "robot_name", self.robot.model.name)
            filename = f"{robot_name}_optimal_trajectory.yaml"

            condition_number = getattr(self, "final_condition_number", 0.0)
            results_dict = {
                "trajectory_segments": len(self.results["T_F"]),
                "condition_number": float(condition_number),
                "joint_count": len(self.CB.act_Jid),
            }

            with open(os.path.join(output_dir, filename), "w") as f:
                yaml.dump(results_dict, f, default_flow_style=False)

            self.logger.info(f"Basic results saved to {output_dir}/{filename}")
            return {"yaml": os.path.join(output_dir, filename)}


class BaseTrajectoryIPOPTProblem(BaseOptimizationProblem):
    """
    Base IPOPT problem formulation for trajectory optimization.

    This class provides a base implementation for trajectory optimization
    that can be extended for specific robots.
    """

    def __init__(
        self,
        opt_traj,
        n_joints,
        n_wps,
        Ns,
        tps,
        vel_wps,
        acc_wps,
        wp_init,
        vel_wp_init,
        acc_wp_init,
        W_stack,
        problem_name="TrajectoryOptimization",
    ):
        super().__init__(problem_name)

        self.opt_traj = opt_traj
        self.n_joints = n_joints
        self.n_wps = n_wps
        self.Ns = Ns
        self.tps = tps
        self.vel_wps = vel_wps
        self.acc_wps = acc_wps
        self.wp_init = wp_init
        self.vel_wp_init = vel_wp_init
        self.acc_wp_init = acc_wp_init
        self.W_stack = W_stack

        # Storage for optimization callback (inherits callback_data from base)
        self.opt_cb = {"t_f": None, "p_f": None, "v_f": None, "a_f": None}

    def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
        """Get variable bounds for optimization."""
        return self.opt_traj.constraint_manager.get_variable_bounds()

    def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
        """Get constraint bounds for optimization."""
        return self.opt_traj.constraint_manager.get_constraint_bounds(self.Ns)

    def get_initial_guess(self) -> List[float]:
        """Get initial guess from waypoints."""
        # This will be set when solve() is called with waypoints
        if not hasattr(self, "_initial_wps"):
            # Return zeros as fallback
            return [0.0] * (self.n_joints * (self.n_wps - 1))

        X0 = self._initial_wps[:, range(1, self.n_wps)]
        return np.reshape(X0.transpose(), (self.n_joints * (self.n_wps - 1),)).tolist()

    def objective(self, X: np.ndarray) -> float:
        """Objective function: condition number of base regressor matrix."""
        return self.opt_traj.objective_function(
            X,
            self.opt_cb,
            self.tps,
            self.vel_wps,
            self.acc_wps,
            self.wp_init,
            self.W_stack,
        )

    def constraints(self, X: np.ndarray) -> np.ndarray:
        """Constraint function for IPOPT."""
        return self.opt_traj.constraint_manager.evaluate_constraints(
            self.Ns, X, self.opt_cb, self.tps, self.vel_wps, self.acc_wps, self.wp_init
        )

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        Jacobian of constraints - Custom implementation for better performance.

        For trajectory optimization, we can use sparse finite differences
        instead of full automatic differentiation which is too slow.
        """
        try:
            # Get current constraint values
            c0 = self.constraints(X)
            n_constraints = len(c0)
            n_vars = len(X)

            # Use finite differences with smaller step size for efficiency
            eps = 1e-6
            jac = np.zeros((n_constraints, n_vars))

            # Compute Jacobian column by column (forward differences)
            for i in range(n_vars):
                X_plus = X.copy()
                X_plus[i] += eps
                c_plus = self.constraints(X_plus)
                jac[:, i] = (c_plus - c0) / eps

            self.logger.debug(f"Constraint jacobian shape: {jac.shape}")
            return jac

        except Exception as e:
            self.logger.warning(f"Error computing jacobian: {e}")
            # Return sparse identity matrix as fallback
            n_constraints = len(self.constraints(X))
            n_vars = len(X)
            # Create a sparse jacobian approximation
            jac = np.zeros((n_constraints, n_vars))
            min_dim = min(n_constraints, n_vars)
            jac[:min_dim, :min_dim] = np.eye(min_dim)
            return jac

    def solve_with_waypoints(self, wps) -> Tuple[bool, Dict[str, Any]]:
        """
        Solve the optimization problem with given initial waypoints.

        Args:
            wps: Initial waypoints

        Returns:
            Tuple of (success, results_dict)
        """
        try:
            # Store initial waypoints for get_initial_guess
            self._initial_wps = wps

            # Create solver with trajectory optimization config
            config = IPOPTConfig.for_trajectory_optimization()
            # Adjust settings for this complex problem
            config.tolerance = 1e-3
            config.acceptable_tolerance = 1e-2
            config.max_iterations = 200
            config.print_level = 3  # Reduce output
            config.custom_options = {
                b"mu_strategy": b"adaptive",
            }
            solver = RobotIPOPTSolver(self, config)

            # Solve the problem
            success, results = solver.solve()

            if success:
                # Extract final waypoint for next segment
                X_opt = results["x_opt"]
                wps_X = np.reshape(np.array(X_opt), (self.n_wps - 1, self.n_joints))
                final_waypoint = wps_X[-1, :]

                # Update results with trajectory-specific data
                results.update(
                    {
                        "t_f": self.opt_cb["t_f"],
                        "p_f": self.opt_cb["p_f"],
                        "v_f": self.opt_cb["v_f"],
                        "a_f": self.opt_cb["a_f"],
                        "iter_data": {
                            "iterations": self.iteration_data["iterations"],
                            "obj_values": self.iteration_data["obj_values"],
                            "solve_time": results["solve_time"],
                            "status": results["status"],
                            "final_waypoint": final_waypoint,
                        },
                    }
                )

                return True, results
            else:
                self.logger.error("Optimization failed")
                return False, results

        except Exception as e:
            self.logger.error(f"Error in IPOPT solve: {e}")
            return False, {"error": str(e)}
