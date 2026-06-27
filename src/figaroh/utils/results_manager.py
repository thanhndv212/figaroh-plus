"""
Unified results management for FIGAROH examples.

This module provides standardized plotting and saving functionality
across all task types: Calibration, Identification, OptimalCalibration, OptimalTrajectory.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .error_handling import (
    FigarohExampleError,
    validate_input_data,
    setup_example_logging,
)

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ResultsManager:
    """
    Unified results management for all FIGAROH task types.

    This class provides standardized plotting and saving functionality
    with consistent styling and formats across different analysis types.
    """

    # Color schemes for consistent visualization
    COLORS = {
        "measured": "#1f77b4",  # Blue
        "identified": "#ff7f0e",  # Orange
        "calibrated": "#2ca02c",  # Green
        "optimal": "#d62728",  # Red
        "reference": "#9467bd",  # Purple
        "residual": "#8c564b",  # Brown
    }

    # Plot styles for different task types
    PLOT_STYLES = {
        "calibration": {"figsize": (14, 10), "dpi": 100},
        "identification": {"figsize": (12, 8), "dpi": 100},
        "optimal_calibration": {"figsize": (10, 8), "dpi": 100},
        "optimal_trajectory": {"figsize": (15, 10), "dpi": 100},
    }

    def __init__(
        self,
        task_type: str,
        robot_name: str = "robot",
        results_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize results manager.

        Args:
            task_type: Type of task ('calibration', 'identification',
                  'optimal_calibration', 'optimal_trajectory')
            robot_name: Name of the robot for file naming
            results_data: Optional dictionary of results data
        """
        self.task_type = task_type.lower()
        self.robot_name = robot_name
        assert (
            self.task_type in self.PLOT_STYLES
        ), f"Unsupported task type: {self.task_type}"
        assert (
            self.task_type == results_data["task type"] if results_data else True
        ), "Results data must match task type"
        self.result = results_data or {}
        self.logger = setup_example_logging()

        if not HAS_MATPLOTLIB:
            self.logger.warning("Matplotlib not available. Plotting disabled.")

    def plot_calibration_results(
        self,
        measured_poses: Optional[np.ndarray] = None,
        estimated_poses: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        parameter_values: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        outlier_indices: Optional[List[int]] = None,
        title: str = "Calibration Results",
    ) -> None:
        """
        Plot calibration results with pose comparison and residual analysis.

        Args:
            measured_poses: Measured poses (optional, uses self.result)
            estimated_poses: Estimated poses (optional, uses self.result)
            residuals: Position/orientation residuals (optional, uses self.result)
            parameter_values: Calibrated parameters (optional, uses self.result)
            parameter_names: Parameter names (optional, uses self.result)
            outlier_indices: Outlier indices (optional, uses self.result)
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            self.logger.warning("Cannot plot: matplotlib not available")
            return

        try:
            # Extract data from self.result if not provided as arguments
            if measured_poses is None:
                measured_poses = self.result.get(
                    "PEE measured (2D array)", np.array([])
                )
            if estimated_poses is None:
                estimated_poses = self.result.get(
                    "PEE estimated (2D array)", np.array([])
                )
            if residuals is None:
                residuals = self.result.get("residuals", np.array([]))
            if parameter_values is None:
                param_vals = self.result.get("calibrated parameters values", [])
                parameter_values = np.array(param_vals) if param_vals else np.array([])
            if parameter_names is None:
                parameter_names = self.result.get("calibrated parameters names", [])
            if outlier_indices is None:
                outlier_indices = self.result.get("outlier indices", [])

            # Validate we have the minimum required data
            if measured_poses.size == 0:
                self.logger.warning("No measured pose data available for plotting")
                return

            fig = plt.figure(figsize=self.PLOT_STYLES["calibration"]["figsize"])
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

            # Check position + orientation (6 DOF) or position only (3 DOF)
            n_dims = measured_poses.shape[1]

            if n_dims >= 3 and estimated_poses.size > 0:
                # Position comparison
                ax1 = fig.add_subplot(gs[0, :])
                self._plot_pose_comparison(
                    ax1,
                    measured_poses[:, :3],
                    estimated_poses[:, :3],
                    "Position Comparison",
                    "Position (m)",
                    outlier_indices,
                )

            if n_dims >= 6 and estimated_poses.size > 0:
                # Orientation comparison
                ax2 = fig.add_subplot(gs[1, :])
                self._plot_pose_comparison(
                    ax2,
                    measured_poses[:, 3:6],
                    estimated_poses[:, 3:6],
                    "Orientation Comparison",
                    "Orientation (rad)",
                    outlier_indices,
                )
            elif n_dims >= 3 and residuals.size > 0:
                # Position residuals if we don't have orientation data
                ax2 = fig.add_subplot(gs[1, :])
                self._plot_position_residuals(ax2, residuals[:, :3], outlier_indices)

            # Residual analysis
            if residuals.size > 0:
                ax3 = fig.add_subplot(gs[2, 0])
                self._plot_residuals(ax3, residuals, outlier_indices)

            # Parameter values
            if parameter_values.size > 0:
                ax5 = fig.add_subplot(gs[2, 1])
                self._plot_parameters(ax5, parameter_values, parameter_names)

                # Add statistics text
                num_params = self.result.get(
                    "number of calibrated parameters", len(parameter_values)
                )
                num_samples = self.result.get("number of samples", "N/A")
                rmse = self.result.get("rmse", "N/A")
                mae = self.result.get("mae", "N/A")

                stats_text = f"Parameters: {num_params}\n" f"Samples: {num_samples}"
                if rmse != "N/A":
                    stats_text += f"\nRMSE: {rmse:.6f}"
                if mae != "N/A":
                    stats_text += f"\nMAE: {mae:.6f}"

                ax5.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax5.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            fig.suptitle(f"{self.robot_name.upper()} {title}", fontsize=16)
            # plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting calibration results: {e}")
            # Log debug information
            logger.debug("Debug info:")
            measured_shape = (
                measured_poses.shape if hasattr(measured_poses, "shape") else "N/A"
            )
            logger.debug(f"  measured_poses shape: {measured_shape}")

            estimated_shape = (
                estimated_poses.shape if hasattr(estimated_poses, "shape") else "N/A"
            )
            logger.debug(f"  estimated_poses shape: {estimated_shape}")

            residuals_shape = residuals.shape if hasattr(residuals, "shape") else "N/A"
            logger.debug(f"  residuals shape: {residuals_shape}")

            result_keys = list(self.result.keys()) if self.result else "None"
            logger.debug(f"  Available result keys: {result_keys}")

    def plot_identification_results(
        self,
        tau_measured: Optional[np.ndarray] = None,
        tau_identified: Optional[np.ndarray] = None,
        parameter_values: Optional[np.ndarray] = None,
        parameter_names: Optional[List[str]] = None,
        time_vector: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        title: str = "Identification Results",
    ) -> None:
        """
        Plot identification results with torque comparison and analysis.

        Args:
            tau_measured: Measured joint torques (optional, uses self.result)
            tau_identified: Estimated torques (optional, uses self.result)
            parameter_values: Parameter values (optional, uses self.result)
            parameter_names: Parameter names (optional, uses self.result)
            time_vector: Time vector for plots
            joint_names: Names of robot joints
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            self.logger.warning("Cannot plot: matplotlib not available")
            return

        try:
            # Use data from self.result if parameters not provided
            if tau_measured is None:
                tau_measured = self.result.get("torque processed", np.array([]))
            if tau_identified is None:
                tau_identified = self.result.get("torque estimated", np.array([]))
            if parameter_values is None:
                parameter_values = self.result.get(
                    "base parameters values", np.array([])
                )
            if parameter_names is None:
                parameter_names = self.result.get("base parameters names", [])

            # Validate data availability
            if len(tau_measured) == 0 or len(tau_identified) == 0:
                self.logger.error("No torque data available for plotting")
                return

            fig = plt.figure(figsize=self.PLOT_STYLES["identification"]["figsize"])
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

            # Determine if we need to reshape data (1D to 2D for multi-joint)
            if tau_measured.ndim == 1:
                tau_measured = tau_measured.reshape(-1, 1)
            if tau_identified.ndim == 1:
                tau_identified = tau_identified.reshape(-1, 1)

            n_samples = tau_measured.shape[0]

            if time_vector is None:
                time_vector = np.arange(n_samples)

            # Torque comparison
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_torque_comparison(
                ax1, time_vector, tau_measured, tau_identified, joint_names
            )

            # Residual analysis
            ax2 = fig.add_subplot(gs[1, :])
            residuals = tau_measured - tau_identified
            self._plot_torque_residuals(ax2, time_vector, residuals, joint_names)

            # Parameter values
            # ax3 = fig.add_subplot(gs[1, 1])
            # self._plot_parameters(ax3, parameter_values, parameter_names)

            # Add quality metrics to title if available
            condition_num = self.result.get("condition number", "N/A")
            rmse_norm = self.result.get("rmse norm (N/m)", "N/A")

            fig.suptitle(
                f"{self.robot_name.upper()} {title}\n"
                f"Condition: {condition_num:.2e} | "
                f"RMSE: {rmse_norm:.6f}",
                fontsize=16,
            )
            # plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting identification results: {e}")
            import traceback

            traceback.print_exc()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting identification results: {e}")

    def plot_optimal_calibration_results(
        self,
        configurations: Dict[str, np.ndarray],
        weights: np.ndarray,
        condition_numbers: Optional[np.ndarray] = None,
        information_matrix: Optional[np.ndarray] = None,
        title: str = "Optimal Calibration Results",
    ) -> None:
        """
        Plot optimal calibration configuration results.

        Args:
            configurations: Dictionary of configuration data
            weights: Weights assigned to each configuration
            condition_numbers: Condition numbers for analysis
            information_matrix: Fisher information matrix
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            self.logger.warning("Cannot plot: matplotlib not available")
            return

        try:
            fig = plt.figure(figsize=self.PLOT_STYLES["optimal_calibration"]["figsize"])
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

            # Configuration weights
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_configuration_weights(ax1, weights)

            # Selected configurations visualization
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_selected_configurations(ax2, configurations, weights)

            # Condition number analysis (if available)
            if condition_numbers is not None:
                ax3 = fig.add_subplot(gs[1, 0])
                self._plot_condition_analysis(ax3, condition_numbers)

            # Information matrix visualization (if available)
            if information_matrix is not None:
                ax4 = fig.add_subplot(gs[1, 1])
                self._plot_information_matrix(ax4, information_matrix)

            fig.suptitle(f"{self.robot_name.upper()} {title}", fontsize=16)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting optimal calibration results: {e}")

    def plot_optimal_trajectory_results(
        self,
        trajectories: Dict[str, Any],
        condition_number: float,
        joint_names: Optional[List[str]] = None,
        title: str = "Optimal Trajectory Results",
    ) -> None:
        """
        Plot optimal trajectory generation results.

        Args:
            trajectories: Dictionary containing trajectory data
            condition_number: Final condition number achieved
            joint_names: Names of robot joints
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            self.logger.warning("Cannot plot: matplotlib not available")
            return

        try:
            # Extract trajectory data
            if "T_F" in trajectories:
                # Multiple segments format
                times = trajectories["T_F"]
                positions = trajectories["P_F"]
                velocities = trajectories["V_F"]
                accelerations = trajectories["A_F"]
            else:
                # Single trajectory format
                times = [
                    trajectories.get("time", np.arange(trajectories["q"].shape[0]))
                ]
                positions = [trajectories["q"]]
                velocities = [trajectories.get("dq", np.zeros_like(trajectories["q"]))]
                accelerations = [
                    trajectories.get("ddq", np.zeros_like(trajectories["q"]))
                ]

            n_joints = positions[0].shape[1]

            fig = plt.figure(figsize=self.PLOT_STYLES["optimal_trajectory"]["figsize"])
            gs = GridSpec(n_joints, 3, figure=fig, hspace=0.4, wspace=0.3)

            # Plot each joint's trajectory
            colors = plt.cm.tab10(np.linspace(0, 1, len(times)))

            for joint_idx in range(n_joints):
                # Position
                ax_pos = fig.add_subplot(gs[joint_idx, 0])
                for seg_idx, (t, pos) in enumerate(zip(times, positions)):
                    ax_pos.plot(
                        t,
                        pos[:, joint_idx],
                        color=colors[seg_idx],
                        label=f"Segment {seg_idx+1}" if joint_idx == 0 else "",
                    )
                ax_pos.set_ylabel(f"Joint {joint_idx+1}\nPosition (rad)")
                ax_pos.grid(True, alpha=0.3)
                if joint_idx == 0:
                    ax_pos.legend()
                    ax_pos.set_title("Joint Positions")

                # Velocity
                ax_vel = fig.add_subplot(gs[joint_idx, 1])
                for seg_idx, (t, vel) in enumerate(zip(times, velocities)):
                    ax_vel.plot(t, vel[:, joint_idx], color=colors[seg_idx])
                ax_vel.set_ylabel(f"Joint {joint_idx+1}\nVelocity (rad/s)")
                ax_vel.grid(True, alpha=0.3)
                if joint_idx == 0:
                    ax_vel.set_title("Joint Velocities")

                # Acceleration
                ax_acc = fig.add_subplot(gs[joint_idx, 2])
                for seg_idx, (t, acc) in enumerate(zip(times, accelerations)):
                    ax_acc.plot(t, acc[:, joint_idx], color=colors[seg_idx])
                ax_acc.set_ylabel(f"Joint {joint_idx+1}\nAcceleration (rad/s²)")
                ax_acc.grid(True, alpha=0.3)
                if joint_idx == 0:
                    ax_acc.set_title("Joint Accelerations")

            # Set x-labels for bottom row
            for col in range(3):
                fig.add_subplot(gs[-1, col]).set_xlabel("Time (s)")

            fig.suptitle(
                f"{self.robot_name.upper()} {title}\nCondition Number: {condition_number:.2e}",
                fontsize=16,
            )
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting trajectory results: {e}")

    def save_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        output_dir: str = "results",
        file_prefix: Optional[str] = None,
        save_formats: List[str] = ["yaml", "csv"],
    ) -> Dict[str, str]:
        """
        Save results to files with standardized format.

        Args:
            results: Dictionary containing results to save (uses self.result if None)
            output_dir: Output directory path
            file_prefix: Prefix for output files (default: robot_name_task_type)
            save_formats: List of formats to save ('yaml', 'csv', 'json', 'npz')

        Returns:
            Dictionary mapping format to file path
        """
        try:
            # Use self.result if no results provided
            if results is None:
                results = self.result

            if not results:
                self.logger.error("No results data available to save")
                return {}

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate file prefix
            if file_prefix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_prefix = f"{self.robot_name}_{self.task_type}_{timestamp}"

            saved_files = {}

            # Save in requested formats
            for fmt in save_formats:
                if fmt.lower() == "yaml":
                    file_path = output_path / f"{file_prefix}.yaml"
                    self._save_yaml(results, file_path)
                    saved_files["yaml"] = str(file_path)

                elif fmt.lower() == "csv":
                    file_path = output_path / f"{file_prefix}.csv"
                    self._save_csv(results, file_path)
                    saved_files["csv"] = str(file_path)

                elif fmt.lower() == "json":
                    file_path = output_path / f"{file_prefix}.json"
                    self._save_json(results, file_path)
                    saved_files["json"] = str(file_path)

                elif fmt.lower() == "npz":
                    file_path = output_path / f"{file_prefix}.npz"
                    self._save_npz(results, file_path)
                    saved_files["npz"] = str(file_path)

            # Save metadata
            metadata_path = output_path / f"{file_prefix}_metadata.yaml"
            self._save_metadata(results, metadata_path)
            saved_files["metadata"] = str(metadata_path)

            self.logger.info(f"Results saved to {output_dir}")
            for fmt, path in saved_files.items():
                self.logger.info(f"  {fmt.upper()}: {path}")

            return saved_files

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise FigarohExampleError(f"Failed to save results: {e}")

    # Helper plotting methods
    def _plot_pose_comparison(
        self, ax, measured, estimated, title, ylabel, outliers=None
    ):
        """Plot pose comparison with outlier highlighting."""
        # Handle both 1D and 2D arrays
        if measured.ndim == 1:
            measured = measured.reshape(-1, 1)
        if estimated.ndim == 1:
            estimated = estimated.reshape(-1, 1)

        n_samples, n_dims = measured.shape

        for dim in range(n_dims):
            ax.plot(
                measured[:, dim],
                label=f"Measured {dim+1}",
                color=self.COLORS["measured"],
                alpha=0.7,
            )
            ax.plot(
                estimated[:, dim],
                label=f"Estimated {dim+1}",
                color=self.COLORS["identified"],
                alpha=0.7,
                linestyle="--",
            )

        if outliers is not None:
            for outlier_idx in outliers:
                ax.axvline(
                    outlier_idx, color=self.COLORS["residual"], alpha=0.5, linestyle=":"
                )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sample Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_residuals(self, ax, residuals, outliers=None):
        """Plot residual analysis."""
        rms_residuals = np.sqrt(np.mean(residuals**2, axis=1))
        ax.bar(
            np.arange(len(rms_residuals)),
            rms_residuals,
            color=self.COLORS["residual"],
            alpha=0.7,
        )

        if outliers is not None:
            ax.scatter(
                outliers,
                rms_residuals[outliers],
                color=self.COLORS["residual"],
                s=50,
                marker="x",
            )

        ax.set_title("RMS Residuals")
        ax.set_ylabel("RMS Error")
        ax.set_xlabel("Sample Index")
        ax.grid(True, alpha=0.3)

    def _plot_position_residuals(self, ax, residuals, outliers=None):
        """Plot position residuals for x, y, z components."""
        labels = ["X", "Y", "Z"]
        for i, label in enumerate(labels):
            if i < residuals.shape[1]:
                ax.plot(residuals[:, i], label=f"{label} residual", alpha=0.7)

        if outliers is not None:
            for outlier_idx in outliers:
                ax.axvline(
                    outlier_idx, color=self.COLORS["residual"], alpha=0.5, linestyle=":"
                )

        ax.set_title("Position Residuals")
        ax.set_ylabel("Error (m)")
        ax.set_xlabel("Sample Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_parameters(self, ax, values, names=None):
        """Plot parameter values with names."""
        x_pos = np.arange(len(values))
        bars = ax.bar(x_pos, values, color=self.COLORS["optimal"], alpha=0.7)

        ax.set_title("Parameter Values")
        ax.set_ylabel("Value")
        ax.set_xlabel("Parameter Index")

        if names is not None:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(names, rotation=45, ha="right")

        ax.grid(True, alpha=0.3)

    def _plot_torque_comparison(self, ax, time, measured, identified, joint_names=None):
        """Plot torque comparison for identification."""
        n_joints = measured.shape[1]

        for joint in range(n_joints):
            label_base = (
                f"Joint {joint+1}" if joint_names is None else joint_names[joint]
            )
            ax.plot(
                time, measured[:, joint], label=f"{label_base} (measured)", alpha=0.7
            )
            ax.plot(
                time,
                identified[:, joint],
                label=f"{label_base} (identified)",
                alpha=0.7,
                linestyle="--",
            )

        ax.set_title("Torque Comparison")
        ax.set_ylabel("Torque (Nm)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_torque_residuals(self, ax, time, residuals, joint_names=None):
        """Plot torque residuals."""
        n_joints = residuals.shape[1]

        for joint in range(n_joints):
            label = f"Joint {joint+1}" if joint_names is None else joint_names[joint]
            ax.plot(time, residuals[:, joint], label=label, alpha=0.7)

        ax.set_title("Torque Residuals")
        ax.set_ylabel("Torque Error (Nm)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_configuration_weights(self, ax, weights):
        """Plot configuration selection weights."""
        significant_weights = weights[weights > 1e-6]
        ax.bar(
            range(len(significant_weights)),
            significant_weights,
            color=self.COLORS["optimal"],
            alpha=0.7,
        )
        ax.set_title("Configuration Weights")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Configuration Index")
        ax.grid(True, alpha=0.3)

    def _plot_selected_configurations(self, ax, configurations, weights):
        """Plot selected configuration visualization."""
        # This is a placeholder - would need specific implementation
        # based on the configuration format
        ax.text(
            0.5,
            0.5,
            f"Selected {np.sum(weights > 1e-6)} configurations\n"
            f"from {len(weights)} candidates",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Selected Configurations")

    def _plot_condition_analysis(self, ax, condition_numbers):
        """Plot condition number analysis."""
        ax.semilogy(condition_numbers, color=self.COLORS["optimal"], alpha=0.7)
        ax.set_title("Condition Number Evolution")
        ax.set_ylabel("Condition Number")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    def _plot_information_matrix(self, ax, info_matrix):
        """Plot information matrix heatmap."""
        im = ax.imshow(info_matrix, cmap="viridis", aspect="auto")
        ax.set_title("Information Matrix")
        plt.colorbar(im, ax=ax)

    # Helper saving methods
    def _save_yaml(self, results, file_path):
        """Save results to YAML format."""
        # Convert numpy arrays to lists for YAML serialization
        yaml_data = self._convert_for_serialization(results)

        with open(file_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)

    def _save_csv(self, results, file_path):
        """Save results to CSV format."""
        # Extract tabular data for CSV
        df_data = {}

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    df_data[key] = value
                elif value.ndim == 2:
                    for i in range(value.shape[1]):
                        df_data[f"{key}_{i}"] = value[:, i]
            elif isinstance(value, (int, float)):
                df_data[key] = [value]  # Single value as list

        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(file_path, index=False)

    def _save_json(self, results, file_path):
        """Save results to JSON format."""
        import json

        json_data = self._convert_for_serialization(results)

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=2)

    def _save_npz(self, results, file_path):
        """Save results to NumPy compressed format."""
        np.savez_compressed(file_path, **results)

    def _save_metadata(self, results, file_path):
        """Save metadata about the results."""
        metadata = {
            "task_type": self.task_type,
            "robot_name": self.robot_name,
            "timestamp": datetime.now().isoformat(),
            "data_shapes": {},
            "summary_statistics": {},
        }

        # Add data shape information
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                metadata["data_shapes"][key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }

                # Add basic statistics for numeric arrays
                if np.issubdtype(value.dtype, np.number):
                    metadata["summary_statistics"][key] = {
                        "mean": float(np.mean(value)),
                        "std": float(np.std(value)),
                        "min": float(np.min(value)),
                        "max": float(np.max(value)),
                    }

        with open(file_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)

    def _convert_for_serialization(self, data):
        """Convert numpy arrays and other non-serializable types for saving."""
        if isinstance(data, dict):
            return {
                key: self._convert_for_serialization(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_for_serialization(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data


# Convenience functions for backward compatibility
def plot_calibration_results(measured_poses, estimated_poses, residuals, **kwargs):
    """Plot calibration results (convenience function)."""
    manager = ResultsManager("calibration", kwargs.get("robot_name", "robot"))
    manager.plot_calibration_results(
        measured_poses, estimated_poses, residuals, **kwargs
    )


def plot_identification_results(tau_measured, tau_identified, parameters, **kwargs):
    """Plot identification results (convenience function)."""
    manager = ResultsManager("identification", kwargs.get("robot_name", "robot"))
    manager.plot_identification_results(
        tau_measured, tau_identified, parameters, **kwargs
    )


def save_results(results, task_type, output_dir="results", **kwargs):
    """Save results (convenience function)."""
    manager = ResultsManager(task_type, kwargs.get("robot_name", "robot"))
    return manager.save_results(results, output_dir, **kwargs)
