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
Base calibration class for FIGAROH examples.

This module provides the BaseCalibration abstract class extracted from the
FIGAROH library for use in the examples. It implements a comprehensive
framework for robot kinematic calibration.
"""

import numpy as np
import yaml
from yaml.loader import SafeLoader
from os.path import abspath
import matplotlib.pyplot as plt
import logging
from scipy.optimize import least_squares
from abc import ABC
from typing import Optional, List, Dict, Any, Tuple

# FIGAROH imports
from figaroh.calibration.calibration_tools import (
    get_param_from_yaml,
    unified_to_legacy_config,
    calculate_base_kinematics_regressor,
    add_base_name,
    add_pee_name,
    load_data,
    calc_updated_fkm,
    initialize_variables,
)
from figaroh.utils.config_parser import (
    UnifiedConfigParser,
    create_task_config,
    is_unified_config,
)
import pinocchio as pin

# Import from shared modules
from figaroh.utils.error_handling import CalibrationError, handle_calibration_errors
from figaroh.utils.results_manager import plot_with_fallback

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BaseCalibration(ABC):
    """
    Abstract base class for robot kinematic calibration.

    This class provides a comprehensive framework for calibrating robot
    kinematic parameters using measurement data. It implements the Template
    Method pattern, providing common functionality while allowing
    robot-specific implementations of the cost function.

    The calibration process follows these main steps:
    1. Parameter initialization from configuration files
    2. Data loading and validation
    3. Parameter identification using base regressor analysis
    4. Robust optimization with outlier detection and removal
    5. Solution evaluation and validation
    6. Results visualization and export

    Key Features:
    - Automatic parameter identification using QR decomposition
    - Robust optimization with iterative outlier removal
    - Unit-aware measurement weighting for position/orientation data
    - Comprehensive solution evaluation and quality metrics
    - Extensible framework for different robot types

    Attributes:
        STATUS (str): Current calibration status ("NOT CALIBRATED" or
                     "CALIBRATED")
        LM_result: Optimization result from scipy.optimize.least_squares
        var_ (ndarray): Calibrated parameter values
        evaluation_metrics (dict): Solution quality metrics
        std_dev (list): Standard deviations of calibrated parameters
        std_pctg (list): Standard deviation percentages
        PEE_measured (ndarray): Measured end-effector poses/positions
        q_measured (ndarray): Measured joint configurations
        calib_config (dict): Calibration parameters and configuration
        model: Robot kinematic model (Pinocchio)
        data: Robot data structure (Pinocchio)

    Example:
        >>> # Create robot-specific calibration
        >>> class MyRobotCalibration(BaseCalibration):
        ...     def cost_function(self, var):
        ...         PEEe = calc_updated_fkm(self.model, self.data, var,
        ...                                self.q_measured, self.calib_config)
        ...         # Use body-frame position (or position_frame="world" for world frame)
        ...         residuals = self._compute_logmap_residuals(
        ...             self.PEE_measured, PEEe, position_frame="body")
        ...         return self.apply_measurement_weighting(residuals)
        ...
        >>> # Run calibration
        >>> calibrator = MyRobotCalibration(robot, "config.yaml")
        >>> calibrator.initialize()
        >>> calibrator.solve()
        >>> print(f"RMSE: {calibrator.evaluation_metrics['rmse']:.6f}")

    Notes:
        - Derived classes should implement robot-specific cost_function()
        - Default cost_function is provided but issues performance warning
        - Configuration files must follow FIGAROH parameter structure
        - Supports both "full_params" and "joint_offset" calibration models

    See Also:
        - TiagoCalibration: TIAGo robot implementation
        - UR10Calibration: Universal Robots UR10 implementation
        - calc_updated_fkm: Forward kinematics computation function
        - apply_measurement_weighting: Unit-aware weighting utility
    """

    @handle_calibration_errors
    def __init__(self, robot, config_file: str, del_list: List[int] = None):
        """Initialize robot calibration framework.

        Sets up the calibration environment by loading robot model,
        configuration parameters, and preparing internal data structures
        for optimization.

        Args:
            robot: Robot object containing kinematic model and data structures.
                  Must have 'model' and 'data' attributes compatible with
                  Pinocchio library.
            config_file (str): Path to YAML configuration file containing
                             calibration parameters, data paths, and settings.
            del_list (list, optional): Indices of bad/outlier samples to
                                     exclude from calibration data.
                                     Defaults to [].

        Raises:
            FileNotFoundError: If config_file does not exist
            KeyError: If required parameters missing from configuration
            ValueError: If configuration parameters are invalid
            CalibrationError: If robot or configuration is invalid

        Side Effects:
            - Loads and validates configuration parameters
            - Sets initial calibration status to "NOT CALIBRATED"
            - Calculates number of calibration variables
            - Resolves absolute path to measurement data file

        Example:
            >>> robot = load_robot_model("tiago.urdf")
            >>> calibrator = TiagoCalibration(robot, "tiago_config.yaml",
            ...                              del_list=[5, 12, 18])
        """
        if del_list is None:
            del_list = []

        # Validate inputs
        if not hasattr(robot, "model") or not hasattr(robot, "data"):
            raise CalibrationError("Robot must have 'model' and 'data' attributes")

        self.robot = robot
        self.model = self.robot.model
        self.data = self.robot.data
        self.del_list_ = del_list
        self.calib_config = None
        self.load_param(config_file)
        self.nvars = len(self.calib_config["param_name"])
        self._data_path = abspath(self.calib_config["data_file"])
        self.STATUS = "NOT CALIBRATED"
        self._val_available = False

    def initialize(self):
        """Initialize calibration data and parameters.

        Performs the initialization phase of calibration by:
        1. Loading measurement data from files
        2. Creating parameter list through base regressor analysis
        3. Identifying calibratable parameters using QR decomposition

        This method must be called before solve() to prepare the calibration
        problem. It handles data validation, parameter identification, and
        sets up the optimization problem structure.

        Raises:
            FileNotFoundError: If measurement data file not found
            ValueError: If data format is invalid or incompatible
            AssertionError: If required data dimensions don't match
            CalibrationError: If initialization fails

        Side Effects:
            - Populates self.PEE_measured with measurement data
            - Populates self.q_measured with joint configuration data
            - Updates self.calib_config["param_name"] with identified
              parameters
            - Validates data consistency and dimensions

        Example:
            >>> calibrator = TiagoCalibration(robot, "config.yaml")
            >>> calibrator.initialize()
            >>> print(f"Loaded {calibrator.calib_config['NbSample']} samples")
            >>> print(f"Calibrating {len(calibrator.calib_config['param_name'])} "
            ...       f"parameters")
        """
        try:
            self.load_data_set()
            self.create_param_list()
        except Exception as e:
            raise CalibrationError(f"Initialization failed: {e}")

    def solve(
        self,
        method="lm",
        max_iterations=3,
        outlier_threshold=3.0,
        enable_logging=True,
        plotting=False,
        save_results=False,
        html_report=False,
    ):
        """Execute the complete calibration process.

        This is the main entry point for calibration that:
        1. Runs the optimization algorithm via solve_optimisation()
        2. Optionally generates visualization plots if enabled
        3. Optionally saves results to files if enabled

        The method serves as a high-level orchestrator for the calibration
        workflow, delegating the actual optimization to solve_optimisation()
        and handling visualization based on user preferences.

        Args:
            html_report: If True, also export an HTML diagnostic report
                (see :meth:`export_html_report`) after the terminal
                quality report is printed.

        Side Effects:
            - Updates calibration parameters through optimization
            - Sets self.STATUS to "CALIBRATED" on successful completion
            - May display plots if self.calib_config["PLOT"] is True

        See Also:
            solve_optimisation: Core optimization implementation
            plot: Visualization and analysis plotting
            export_html_report: Visual counterpart of the terminal report
        """
        result, outlier_indices = self.solve_optimisation(
            method=method,
            max_iterations=max_iterations,
            outlier_threshold=outlier_threshold,
            enable_logging=enable_logging,
        )

        # Evaluate solution
        evaluation = self._evaluate_solution(result, outlier_indices)

        # Log final results
        if enable_logging:
            logger.info("=" * 30)
            logger.info("FINAL CALIBRATION RESULTS")
            logger.info("=" * 30)
            self._log_iteration_results("FINAL", result, evaluation)

            if len(outlier_indices) > 0:
                logger.info(f"Outlier samples: {outlier_indices}")
            logger.info("Calibration completed successfully!")

        # Store results
        self._store_optimization_results(result, evaluation, outlier_indices)

        # Print quality report
        self.print_quality_report()

        # Generate plots if required
        if plotting:
            self.plot_results()
        if save_results:
            self.save_results()
        if html_report:
            self.export_html_report()
        return result

    def plot_results(self):
        """Generate comprehensive visualization plots for calibration results.

        Creates multiple visualization plots to analyze calibration quality:
        1. Error distribution plots showing residual patterns
        2. 3D pose visualizations comparing measured vs predicted poses
        3. Joint configuration analysis (currently commented)

        This method provides essential visual feedback for calibration
        assessment, helping users understand solution quality and identify
        potential issues with the calibration process.

        Prerequisites:
            - Calibration must be completed (solve() called)
            - Measurement data must be loaded
            - Matplotlib backend must be configured

        Side Effects:
            - Displays plots using plt.show()
            - May block execution until plots are closed

        See Also:
            plot_errors_distribution: Individual error analysis plots
            plot_3d_poses: 3D pose comparison visualization
        """

        def _basic_plots():
            try:
                self.plot_errors_distribution()
                self.plot_3d_poses()
                # self.plot_joint_configurations()
                plt.show()
            except Exception as e:
                logger.warning(f"Plotting failed: {e}")

        # Use pre-initialized results manager if available, else go straight
        # to the basic-plotting fallback.
        if hasattr(self, "results_manager") and self.results_manager is not None:
            plot_with_fallback(
                lambda: self.results_manager.plot_calibration_results(),
                _basic_plots,
                logger,
                "calibration",
            )
        else:
            _basic_plots()

    def load_param(self, config_file: str, setting_type: str = "calibration"):
        """Load calibration parameters from YAML configuration file.

        This method supports both legacy YAML format and the new unified
        configuration format. It automatically detects the format type
        and applies the appropriate parser.

        Args:
            config_file (str): Path to configuration file (legacy or unified)
            setting_type (str): Configuration section to load
        """
        self._config_file_path = config_file
        try:
            logger.info(f"Loading config from {config_file}")

            # Check if this is a unified configuration format
            if is_unified_config(config_file):
                logger.info("Detected unified configuration format")
                # Use unified parser
                parser = UnifiedConfigParser(config_file)
                unified_config = parser.parse()
                unified_calib_config = create_task_config(
                    self.robot, unified_config, setting_type
                )
                # Convert unified format to legacy calib_config format
                self.calib_config = unified_to_legacy_config(
                    self.robot, unified_calib_config
                )
            else:
                logger.info("Detected legacy configuration format")
                # Use legacy format parsing
                with open(config_file, "r") as f:
                    config = yaml.load(f, Loader=SafeLoader)

                if setting_type not in config:
                    raise KeyError(f"Setting type '{setting_type}' not found in config")

                calib_data = config[setting_type]
                self.calib_config = get_param_from_yaml(self.robot, calib_data)

        except FileNotFoundError:
            raise CalibrationError(f"Configuration file not found: {config_file}")
        except Exception as e:
            raise CalibrationError(f"Failed to load configuration: {e}")

    def create_param_list(self, q: Optional[np.ndarray] = None):
        """Initialize calibration parameter structure and validate setup.

        This method sets up the fundamental parameter structure for calibration
        by computing kinematic regressors and ensuring proper frame naming
        conventions. It serves as a critical initialization step that must be
        called before optimization begins.

        The method performs several key operations:
        1. Computes base kinematic regressors for parameter identification
        2. Adds default names for unknown base and tip frames
        3. Validates the parameter structure for calibration readiness

        Args:
            q (array_like, optional): Joint configuration for regressor
                                    computation. If None, uses empty list
                                    which may limit regressor accuracy

        Returns:
            bool: Always returns True to indicate successful completion

        Side Effects:
            - Updates self.calib_config with frame names if not known
            - Computes and caches kinematic regressors
            - May modify parameter structure for calibration compatibility

        Raises:
            ValueError: If robot model is not properly initialized
            AttributeError: If required calibration parameters are missing
            CalibrationError: If parameter creation fails

        Example:
            >>> calibrator = BaseCalibration(robot)
            >>> calibrator.load_param("config.yaml")
            >>> calibrator.create_param_list()  # Basic setup
            >>> # Or with specific joint configuration
            >>> q_nominal = np.zeros(robot.nq)
            >>> calibrator.create_param_list(q_nominal)

        See Also:
            calculate_base_kinematics_regressor: Core regressor computation
            add_base_name: Base frame naming utilities
            add_pee_name: End-effector frame naming utilities
        """
        if q is None:
            q_ = []
        else:
            q_ = q

        try:
            (
                Rrand_b,
                R_b,
                R_e,
                paramsrand_base,
                paramsrand_e,
            ) = calculate_base_kinematics_regressor(
                q_, self.model, self.data, self.calib_config, tol_qr=1e-6
            )

            if self.calib_config["known_baseframe"] is False:
                add_base_name(self.calib_config)
            if self.calib_config["known_tipframe"] is False:
                add_pee_name(self.calib_config)

            return True

        except Exception as e:
            raise CalibrationError(f"Parameter list creation failed: {e}")

    def load_data_set(self):
        """Load experimental measurement data for calibration.

        Reads measurement data from the specified data path and processes it
        for calibration use. This includes both pose measurements and
        corresponding joint configurations, with optional data filtering
        based on the deletion list.

        The method handles data preprocessing, validation, and formatting
        to ensure compatibility with the calibration algorithms. It serves
        as the primary data ingestion point for the calibration process.

        Side Effects:
            - Sets self.PEE_measured with processed pose measurements
            - Sets self.q_measured with corresponding joint configurations
            - Applies data filtering if self.del_list_ is specified

        Prerequisites:
            - self._data_path must be set to valid measurement data location
            - Robot model must be initialized
            - Calibration parameters must be loaded

        Raises:
            FileNotFoundError: If data files are not found at _data_path
            ValueError: If data format is incompatible or corrupted
            AttributeError: If required attributes are not initialized
            CalibrationError: If data loading fails

        See Also:
            load_data: Core data loading and processing function
        """
        try:
            self.PEE_measured, self.q_measured = load_data(
                self._data_path, self.model, self.calib_config, self.del_list_
            )
        except Exception as e:
            raise CalibrationError(f"Data loading failed: {e}")

        # If validation data path is specified in config, load it
        val_data_path = self.calib_config.get("validation_data_file")
        if val_data_path:
            try:
                self._load_validation_data(val_data_path)
            except Exception:
                pass  # Don't fail calibration if validation data unavailable

    def _load_validation_data(self, path: str):
        """Load separate validation measurement data.

        Args:
            path: Path to a CSV file with validation measurements,
                  in the same format as calibration data.

        Side Effects:
            - Sets self._q_val with validation joint configurations
            - Sets self._PEE_val with validation measured poses
            - Sets self._val_available = True
        """
        try:
            orig_path = self._data_path
            self._data_path = abspath(path)
            self._q_val, self._PEE_val = load_data(
                self._data_path, self.model, self.calib_config, []
            )
            self._data_path = orig_path
            self._val_available = True
        except Exception as e:
            raise CalibrationError(f"Validation data loading failed: {e}")

    def _compute_validation_metrics(self) -> Optional[Dict[str, Any]]:
        """Compute FK validation metrics on held-out data.

        Computes FK with both nominal (zero params) and calibrated
        parameters on the validation set, then compares each against
        the ground-truth measured poses.

        Returns:
            Dict with validation metrics, or None if no validation data.
        """
        if not getattr(self, "_val_available", False):
            return None

        result = self.LM_result
        zeros = np.zeros_like(result.x)

        # FK for nominal and calibrated on validation set
        PEE_nom = calc_updated_fkm(
            self.model, self.data, zeros, self._q_val, self.calib_config
        )
        PEE_cal = calc_updated_fkm(
            self.model, self.data, result.x, self._q_val, self.calib_config
        )

        # Log-map residuals
        resid_nom = self._compute_logmap_residuals(self._PEE_val, PEE_nom)
        resid_cal = self._compute_logmap_residuals(self._PEE_val, PEE_cal)

        n_dofs = self.calib_config["calibration_index"]
        n_val = len(self._q_val)

        # Reshape to (n_dofs, n_val) — DOF-major
        resid_nom_2d = resid_nom.reshape((n_dofs, n_val))
        resid_cal_2d = resid_cal.reshape((n_dofs, n_val))

        # Position DOFs (first 3), Orientation DOFs (last 3)
        pos_nom = resid_nom_2d[:3, :]
        pos_cal = resid_cal_2d[:3, :]
        orient_nom = resid_nom_2d[3:6, :]
        orient_cal = resid_cal_2d[3:6, :]

        def _error_stats(arr_2d):
            """arr_2d: (n_dof_group, n_samples) → per-sample norm → stats."""
            per_sample = np.sqrt(np.sum(arr_2d ** 2, axis=0))
            return {
                "rmse": float(np.sqrt(np.mean(np.sum(arr_2d ** 2, axis=0)))),
                "max": float(np.max(per_sample)),
                "mean": float(np.mean(per_sample)),
            }

        pos_nom_stats = _error_stats(pos_nom)
        pos_cal_stats = _error_stats(pos_cal)
        orient_nom_stats = _error_stats(orient_nom)
        orient_cal_stats = _error_stats(orient_cal)

        def _improvement(before, after):
            if before > 0:
                return (before - after) / before * 100
            return 0.0

        return {
            "n_val_samples": n_val,
            "pos_rmse_nominal_mm": pos_nom_stats["rmse"] * 1000,
            "pos_rmse_calibrated_mm": pos_cal_stats["rmse"] * 1000,
            "pos_max_nominal_mm": pos_nom_stats["max"] * 1000,
            "pos_max_calibrated_mm": pos_cal_stats["max"] * 1000,
            "pos_improvement_pct": _improvement(
                pos_nom_stats["rmse"], pos_cal_stats["rmse"]
            ),
            "orient_rmse_nominal_deg": (
                orient_nom_stats["rmse"] * 180 / np.pi
            ),
            "orient_rmse_calibrated_deg": (
                orient_cal_stats["rmse"] * 180 / np.pi
            ),
            "orient_max_nominal_deg": (
                orient_nom_stats["max"] * 180 / np.pi
            ),
            "orient_max_calibrated_deg": (
                orient_cal_stats["max"] * 180 / np.pi
            ),
            "orient_improvement_pct": _improvement(
                orient_nom_stats["rmse"], orient_cal_stats["rmse"]
            ),
            "residuals_nominal": resid_nom,
            "residuals_calibrated": resid_cal,
        }

    def get_pose_from_measure(self, res_: np.ndarray) -> np.ndarray:
        """Calculate forward kinematics with calibrated parameters.

        Computes robot end-effector poses using the updated kinematic model
        with calibrated parameters. This method applies the calibration
        results to predict poses for the measured joint configurations.

        Args:
            res_ (ndarray): Calibrated parameter vector containing kinematic
                          corrections (geometric parameters, base transform,
                          tool transform, etc.)

        Returns:
            ndarray: Predicted end-effector poses corresponding to the
                    measured joint configurations. Shape depends on the
                    number of measurements and pose representation format.

        Prerequisites:
            - Joint configurations must be loaded (q_measured available)
            - Calibration parameters must be initialized
            - Robot model must be properly configured

        Example:
            >>> # After calibration
            >>> calibrated_params = calibrator.LM_result.x
            >>> predicted_poses = calibrator.get_pose_from_measure(
            ...     calibrated_params)
            >>> # Compare with measured poses
            >>> errors = predicted_poses - calibrator.PEE_measured

        See Also:
            calc_updated_fkm: Core forward kinematics computation function
        """
        return calc_updated_fkm(
            self.model, self.data, res_, self.q_measured, self.calib_config
        )

    def _compute_logmap_residuals(
        self,
        measured_flat: np.ndarray,
        estimated_flat: np.ndarray,
        *,
        position_frame: str = "body",
    ) -> np.ndarray:
        """Compute pose residuals using the SE3 log map for geometric correctness.

        Replaces the element-wise ``measured - estimated`` subtraction (which
        treats roll-pitch-yaw angles as a vector space — incorrect) with the
        proper SE3 error ``log(M_meas⁻¹ · M_est)`` for orientation, and either
        body-frame or world-frame position error.

        The orientation error is always the **angle-axis vector** from
        ``log(R_meas^T · R_est)`` — the geodesic on SO(3).  Unlike element-wise
        RPY subtraction, it has no singularity issues and is a proper metric.

        The position error can be expressed in two frames:

        ``position_frame="body"`` (default)
            Position error in the **end-effector body frame**::

                v_body = R_meas^T · (p_est − p_meas)

            This is the right-invariant SE3 error from the log map.  It has the
            advantage that the same geometric defect produces the same residual
            regardless of the robot's orientation in the world.  Suitable for
            full 6D calibration.

        ``position_frame="world"``
            Position error in the **world (inertial) frame**::

                v_world = p_est − p_meas

            More interpretable ("the EE is 5 mm too far in world X").  The
            orientation error remains the correct angle-axis metric (not RPY),
            so the main deficiency of the original RPY subtraction is still
            fixed.  Suitable when world-frame residuals are preferred.

        For **unmeasured DOFs** (e.g., orientation in position-only calibration),
        the estimate's values are used to reconstruct the full SE3 transform.
        This ensures the log map produces a meaningful geometric error for the
        measured DOFs without requiring the unmeasured data to exist.
        Unmeasured DOF components are excluded from the output.

        Args:
            measured_flat: Measured PEE array, flat DOF-major order
                ``(n_meas * n_samples,)``.
            estimated_flat: Estimated PEE array, same format.
            position_frame: ``"body"`` (default) for body-frame position error
                from the SE3 log map, or ``"world"`` for world-frame position
                error.

        Returns:
            Flat residual array in the same DOF-major order as the input,
            with geometrically correct SE3 errors.  Can be passed directly
            to :meth:`apply_measurement_weighting`.
        """
        if position_frame not in ("body", "world"):
            raise ValueError(
                f"position_frame must be 'body' or 'world', got '{position_frame}'"
            )

        measurability = np.array(self.calib_config["measurability"], dtype=bool)
        measured_dofs = np.where(measurability)[0]
        unmeasured_dofs = np.where(~measurability)[0]
        n_meas = len(measured_dofs)
        n_samples = self.calib_config["NbSample"]
        n_markers = self.calib_config.get("NbMarkers", 1)

        # Reshape to (n_markers, n_meas, n_samples) — DOF-major
        meas_3d = measured_flat.reshape((n_markers, n_meas, n_samples))
        est_3d = estimated_flat.reshape((n_markers, n_meas, n_samples))

        # Output SE3 errors: (n_markers, 6, n_samples)
        se3_errors = np.zeros((n_markers, 6, n_samples))

        for marker in range(n_markers):
            for s in range(n_samples):
                # Full 6D vectors — fill measured DOFs from data
                meas_6d = np.zeros(6)
                est_6d = np.zeros(6)
                for i, dof in enumerate(measured_dofs):
                    meas_6d[dof] = meas_3d[marker, i, s]
                    est_6d[dof] = est_3d[marker, i, s]

                # Fill unmeasured DOFs from estimate (best available guess)
                for dof in unmeasured_dofs:
                    meas_6d[dof] = est_6d[dof]

                # Convert to SE3
                M_meas = pin.SE3(
                    pin.rpy.rpyToMatrix(meas_6d[3:6]), meas_6d[0:3]
                )
                M_est = pin.SE3(
                    pin.rpy.rpyToMatrix(est_6d[3:6]), est_6d[0:3]
                )

                # Orientation error — always angle-axis from log map
                delta = M_meas.inverse() * M_est
                motion = pin.log(delta)
                se3_errors[marker, 3:, s] = motion.angular  # ω: angle-axis orientation

                # Position error — body-frame or world-frame
                if position_frame == "body":
                    se3_errors[marker, :3, s] = motion.linear   # v: body-frame
                else:
                    se3_errors[marker, :3, s] = est_6d[:3] - meas_6d[:3]  # world-frame

        # Select only measured DOF rows, flatten to DOF-major → same shape as input
        return se3_errors[:, measured_dofs, :].flatten("C")

    def cost_function(self, var: np.ndarray) -> np.ndarray:
        """Calculate cost function for optimization.

        This method provides a default implementation but should be overridden
        by derived classes to define robot-specific cost computation with
        appropriate weighting and regularization.

        Args:
            var (ndarray): Parameter vector to evaluate

        Returns:
            ndarray: Residual vector

        Warning:
            Using default cost function. Consider implementing robot-specific
            cost function for optimal performance.

        Example implementations:

            Body-frame position (default, geometrically correct):
                >>> raw_residuals = self._compute_logmap_residuals(
                ...     self.PEE_measured, PEEe, position_frame="body")

            World-frame position (more interpretable):
                >>> raw_residuals = self._compute_logmap_residuals(
                ...     self.PEE_measured, PEEe, position_frame="world")

            Then apply weighting and regularization:
                >>> weighted_residuals = self.apply_measurement_weighting(
                ...     raw_residuals, pos_weight=1000.0, orient_weight=100.0)
        """
        import warnings

        # Issue warning about using default implementation
        warnings.warn(
            f"Using default cost function for {self.__class__.__name__}. "
            "Consider implementing a robot-specific cost function with "
            "appropriate weighting and regularization for optimal "
            "performance.",
            UserWarning,
            stacklevel=2,
        )

        # Default implementation: basic residual calculation using SE3 log map
        PEEe = calc_updated_fkm(
            self.model, self.data, var, self.q_measured, self.calib_config
        )
        raw_residuals = self._compute_logmap_residuals(self.PEE_measured, PEEe)

        # Apply basic measurement weighting if configuration is available
        try:
            weighted_residuals = self.apply_measurement_weighting(raw_residuals)
            return weighted_residuals
        except (KeyError, AttributeError):
            # Fallback to unweighted residuals if weighting config unavailable
            return raw_residuals

    def apply_measurement_weighting(
        self,
        residuals: np.ndarray,
        pos_weight: Optional[float] = None,
        orient_weight: Optional[float] = None,
    ) -> np.ndarray:
        """Apply measurement weighting to handle position/orientation units.

        This utility method can be used by derived classes to properly weight
        position (meter) and orientation (radian) measurements for equivalent
        influence in the cost function.

        Args:
            residuals (ndarray): Raw residual vector
            pos_weight (float, optional): Weight for position residuals.
                                        If None, uses 1/position_std
            orient_weight (float, optional): Weight for orientation residuals.
                                           If None, uses 1/orientation_std

        Returns:
            ndarray: Weighted residual vector

        Example:
            >>> # In derived class cost_function:
            >>> raw_residuals = self._compute_logmap_residuals(
            ...     self.PEE_measured, PEEe,
            ...     position_frame="body")  # or "world" for world-frame position
            >>> weighted_residuals = self.apply_measurement_weighting(
            ...     raw_residuals, pos_weight=1000.0, orient_weight=100.0)
        """
        # Get weights from parameters or use provided values
        if pos_weight is None:
            pos_std = self.calib_config.get("measurement_std", {}).get(
                "position", 0.001
            )
            pos_weight = 1.0 / pos_std

        if orient_weight is None:
            orient_std = self.calib_config.get("measurement_std", {}).get(
                "orientation", 0.01
            )
            orient_weight = 1.0 / orient_std

        weighted_residuals = []
        residual_idx = 0

        # Process each sample for each marker
        for marker in range(self.calib_config["NbMarkers"]):
            for dof, is_measured in enumerate(self.calib_config["measurability"]):
                if is_measured:
                    for sample in range(self.calib_config["NbSample"]):
                        res = residuals[residual_idx]
                        if dof < 3:  # Position components (x,y,z)
                            weighted_residuals.append(res * pos_weight)
                        else:  # Orientation components (rx,ry,rz)
                            weighted_residuals.append(res * orient_weight)
                            # print(f"Residual index: {residual_idx}")
                        residual_idx += 1
        return np.array(weighted_residuals)

    def _setup_logging(self):
        """Setup logging configuration for terminal output."""
        # Create logger
        logger = logging.getLogger("calibration")
        logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        return logger

    def _optimize_with_outlier_removal(
        self,
        var_init: np.ndarray,
        method: str = "lm",
        max_iterations: int = 3,
        outlier_threshold: float = 1.0,
    ) -> Tuple:
        """Optimize with iterative outlier removal.

        Args:
            var_init (ndarray): Initial parameter guess
            max_iterations (int): Maximum outlier removal iterations
            outlier_threshold (float): Threshold for outlier detection in
                                  standard deviations

        Returns:
            tuple: (result, outlier_indices, final_residuals)
        """
        logger = logging.getLogger("calibration")
        current_var = var_init.copy()
        outlier_indices = []

        for iteration in range(max_iterations):
            logger.info(f"Outlier removal iteration {iteration + 1}")

            # Run optimization
            result = least_squares(
                self.cost_function, current_var, method=method, max_nfev=1000
            )

            if not result.success:
                logger.warning(f"Optimization failed at iteration {iteration + 1}")
                break

            # Calculate residuals using SE3 log map for geometrically
            # correct error and detect outliers
            PEE_est = self.get_pose_from_measure(result.x)
            residuals = self._compute_logmap_residuals(self.PEE_measured, PEE_est)
            new_outliers = self._detect_outliers(residuals, outlier_threshold)

            if len(new_outliers) == 0:
                logger.info("No outliers detected, optimization converged")
                break

            outlier_indices.extend(new_outliers)
            outlier_indices = list(set(outlier_indices))  # Remove duplicates

            logger.info(
                f"Detected {len(new_outliers)} new outliers, "
                f"total outliers: {len(outlier_indices)}"
            )

            # Update for next iteration
            current_var = result.x

        return result, outlier_indices, residuals

    def _detect_outliers(self, residuals: np.ndarray, threshold: float) -> List[int]:
        """Detect outliers using statistical threshold.

        Args:
            residuals (ndarray): Residual vector
            threshold (float): Threshold in standard deviations

        Returns:
            list: Indices of detected outliers
        """
        # Reshape residuals to per-sample format
        n_dofs = self.calib_config["calibration_index"]
        n_samples = self.calib_config["NbSample"]

        if len(residuals) != n_dofs * n_samples:
            return []

        residuals_2d = residuals.reshape((n_dofs, n_samples))

        # Calculate RMS error per sample
        rms_errors = np.sqrt(np.mean(residuals_2d**2, axis=0))

        # Detect outliers
        mean_error = np.mean(rms_errors)
        std_error = np.std(rms_errors)
        threshold_value = mean_error + threshold * std_error

        outliers = np.where(rms_errors > threshold_value)[0].tolist()
        return outliers

    def _evaluate_solution(self, result, outlier_indices: List[int]) -> Dict[str, Any]:
        """Evaluate optimization solution quality.

        Args:
            result: Optimization result from scipy.optimize.least_squares
            outlier_indices (list): Indices of detected outliers

        Returns:
            dict: Solution evaluation metrics
        """
        PEE_est = self.get_pose_from_measure(result.x)
        residuals = self._compute_logmap_residuals(self.PEE_measured, PEE_est)
        n_dofs = self.calib_config["calibration_index"]
        n_samples = self.calib_config["NbSample"]

        # Calculate metrics
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))

        # Per-sample metrics
        if len(residuals) == n_dofs * n_samples:
            residuals_2d = residuals.reshape((n_dofs, n_samples))
            sample_rms = np.sqrt(np.mean(residuals_2d**2, axis=0))
            mean_sample_rms = np.mean(sample_rms)
            std_sample_rms = np.std(sample_rms)
        else:
            mean_sample_rms = rmse
            std_sample_rms = 0.0

        # ── Per-DOF breakdown ──
        per_dof_stats = self._compute_per_dof_stats(residuals, n_dofs, n_samples)

        # ── Condition number ──
        cond_num, cond_label = self._compute_condition_number(result)

        # ── Parameter correlation ──
        correlated_pairs = self._compute_parameter_correlation()

        # Calculate standard deviation of estimated parameters
        self.calc_stddev(result)

        return {
            "rmse": rmse,
            "mae": mae,
            "max_error": max_error,
            "mean_sample_rms": mean_sample_rms,
            "std_sample_rms": std_sample_rms,
            "param_stdev": self.std_dev,
            "param_stddev_percentage": self.std_pctg,
            "n_outliers": len(outlier_indices),
            "outlier_percentage": len(outlier_indices) / n_samples * 100,
            "optimization_success": result.success,
            "cost": result.cost,
            "n_iterations": getattr(result, "nit", 0),
            "n_function_evals": getattr(result, "nfev", 0),
            # ── New quality fields ──
            "per_dof_stats": per_dof_stats,
            "condition_number": cond_num,
            "condition_label": cond_label,
            "correlated_pairs": correlated_pairs,
        }

    def _compute_per_dof_stats(
        self, residuals: np.ndarray, n_dofs: int, n_samples: int
    ) -> Dict[str, Any]:
        """Compute per-DOF residual statistics.

        Returns dict with keys: 'dof_names', 'mean', 'std', 'rmse',
        'max_abs', 'r_squared' — each a list of length n_dofs.
        Units: position DOFs=mm, orientation DOFs=deg.
        """
        dof_names = [
            "X (mm)", "Y (mm)", "Z (mm)",
            "rx (deg)", "ry (deg)", "rz (deg)",
        ]
        dof_names = dof_names[:n_dofs]

        if len(residuals) != n_dofs * n_samples:
            return {
                "dof_names": dof_names, "mean": [], "std": [],
                "rmse": [], "max_abs": [], "r_squared": [],
            }

        residuals_2d = residuals.reshape((n_dofs, n_samples))
        PEE_meas_2d = self.PEE_measured.reshape((n_dofs, n_samples))

        means, stds, rmses, max_abs, r_squareds = [], [], [], [], []

        for i in range(n_dofs):
            row = residuals_2d[i, :]
            meas_row = PEE_meas_2d[i, :]

            # Scale: position → mm, orientation → deg
            scale = 1000.0 if i < 3 else 180.0 / np.pi
            scaled = row * scale

            means.append(float(np.mean(scaled)))
            stds.append(float(np.std(scaled)))
            rmses.append(float(np.sqrt(np.mean(scaled ** 2))))
            max_abs.append(float(np.max(np.abs(scaled))))

            # R² = 1 - SS_res / SS_tot
            ss_res = np.sum(row ** 2)
            ss_tot = np.sum((meas_row - np.mean(meas_row)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
            r_squareds.append(float(r2))

        # Overall position/orientation aggregates
        pos_rows = (
            residuals_2d[:3, :] if n_dofs >= 3 else residuals_2d
        )
        orient_rows = (
            residuals_2d[3:6, :] if n_dofs >= 6
            else np.zeros((3, n_samples))
        )
        pos_rmse = float(
            np.sqrt(np.mean(np.sum(pos_rows ** 2, axis=0)))
        ) * 1000
        orient_rmse = float(
            np.sqrt(np.mean(np.sum(orient_rows ** 2, axis=0)))
        ) * 180 / np.pi
        pos_max = float(
            np.max(np.sqrt(np.sum(pos_rows ** 2, axis=0)))
        ) * 1000
        orient_max = float(
            np.max(np.sqrt(np.sum(orient_rows ** 2, axis=0)))
        ) * 180 / np.pi

        return {
            "dof_names": dof_names,
            "mean": means,
            "std": stds,
            "rmse": rmses,
            "max_abs": max_abs,
            "r_squared": r_squareds,
            "overall": {
                "pos_rmse_mm": pos_rmse,
                "orient_rmse_deg": orient_rmse,
                "pos_max_mm": pos_max,
                "orient_max_deg": orient_max,
            },
        }

    def _compute_condition_number(self, result) -> Tuple[float, str]:
        """Compute condition number of the Jacobian matrix.

        Returns (cond_num, label) where label is one of:
        'well-conditioned' (<100), 'moderately conditioned' (100-1000),
        or 'ill-conditioned' (>1000).
        """
        try:
            J = result.jac
            if J is None:
                return float("nan"), "unavailable (no Jacobian)"
            cond_num = float(np.linalg.cond(J))
            if cond_num < 100:
                label = "well-conditioned"
            elif cond_num < 1000:
                label = "moderately conditioned"
            else:
                label = "ill-conditioned"
            return cond_num, label
        except Exception:
            return float("nan"), "unavailable (computation failed)"

    def _compute_parameter_correlation(self) -> List[Dict[str, Any]]:
        """Compute parameter correlation matrix, flag strongly correlated pairs.

        Returns:
            List of dicts with keys 'param_i', 'param_j', 'correlation'
            for pairs where |ρ| > 0.8.
        """
        try:
            C_param = getattr(self, "_C_param", None)
            if C_param is None:
                return []
            D = np.sqrt(np.diag(C_param))
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where(
                    np.outer(D, D) > 1e-15,
                    C_param / np.outer(D, D),
                    0.0,
                )
            param_names = self.calib_config.get("param_name", [])
            pairs = []
            n = len(D)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr[i, j]) > 0.8:
                        pairs.append({
                            "param_i": (
                                param_names[i]
                                if i < len(param_names)
                                else f"param_{i}"
                            ),
                            "param_j": (
                                param_names[j]
                                if j < len(param_names)
                                else f"param_{j}"
                            ),
                            "correlation": float(corr[i, j]),
                        })
            return pairs
        except Exception:
            return []

    def _prepare_next_iteration(self, result, iteration: int) -> np.ndarray:
        """Prepare for next optimization iteration.

        Args:
            result: Current optimization result
            iteration (int): Current iteration number

        Returns:
            ndarray: Initial guess for next iteration
        """
        if result.success:
            return result.x
        else:
            # If optimization failed, add small random perturbation
            perturbation = np.random.normal(0, 0.001, len(result.x))
            return result.x + perturbation

    def _log_iteration_results(self, iteration, result, evaluation: Dict[str, Any]):
        """Log results for current iteration.

        Args:
            iteration (int): Current iteration number
            result: Optimization result
            evaluation (dict): Solution evaluation metrics
        """
        logger = logging.getLogger("calibration")

        logger.info(f"Iteration {iteration} Results:")
        logger.info(f"  Success: {evaluation['optimization_success']}")
        logger.info(f"  RMSE: {evaluation['rmse']:.6f}")
        logger.info(f"  MAE: {evaluation['mae']:.6f}")
        logger.info(f"  Max Error: {evaluation['max_error']:.6f}")
        logger.info(f"  Cost: {evaluation['cost']:.6f}")
        logger.info(f"  Function Evaluations: {evaluation['n_function_evals']}")
        logger.info(
            f"  Outliers: {evaluation['n_outliers']} "
            f"({evaluation['outlier_percentage']:.1f}%)"
        )

    def _store_optimization_results(
        self, result, evaluation: Dict[str, Any], outlier_indices: List[int]
    ):
        """Store optimization results in instance variables.

        Args:
            result: Final optimization result
            evaluation (dict): Solution evaluation metrics
            outlier_indices (list): Detected outlier indices
        """
        # Store main results
        self.LM_result = result
        self.var_ = result.x
        self.uncalib_values = np.zeros_like(result.x)  # Store initial guess

        # Store evaluation metrics
        self.evaluation_metrics = evaluation
        self.outlier_indices = outlier_indices

        # Calculate per-sample error distribution for plotting (SE3 log map)
        PEE_est = self.get_pose_from_measure(result.x)
        residuals = self._compute_logmap_residuals(self.PEE_measured, PEE_est)
        n_dofs = self.calib_config["calibration_index"]
        n_samples = self.calib_config["NbSample"]
        n_markers = self.calib_config["NbMarkers"]

        # if len(residuals) == n_dofs * n_samples * n_markers:
        #     residuals_3d = residuals.reshape((n_markers, n_dofs, n_samples))
        #     self._PEE_dist = np.sqrt(np.mean(residuals_3d**2, axis=1))
        if len(residuals) == n_dofs * n_samples:
            residuals_2d = residuals.reshape((n_dofs, n_samples))
            sample_rms = np.sqrt(np.mean(residuals_2d**2, axis=0))
            self._PEE_dist = sample_rms.reshape((1, n_samples))
        else:
            # Fallback for unexpected residual shapes
            self._PEE_dist = np.ones((n_markers, n_samples)) * evaluation["rmse"]

        # Reshape PEE measured for consistency
        PEEm_LM2d = self.PEE_measured.reshape((n_dofs, n_samples))
        PEEe_LM2d = PEE_est.reshape((n_dofs, n_samples))
        # Store results
        self.results_data = {}
        self.results_data["number of calibrated parameters"] = len(result.x)
        self.results_data["calibrated parameters names"] = self.calib_config[
            "param_name"
        ]
        self.results_data["calibrated parameters values"] = result.x.tolist()
        self.results_data.update(evaluation)
        self.results_data["number of samples"] = n_samples
        self.results_data["rms residuals by samples"] = self._PEE_dist
        self.results_data["residuals"] = residuals_2d.T
        self.results_data["PEE measured (2D array)"] = PEEm_LM2d.T
        self.results_data["PEE estimated (2D array)"] = PEEe_LM2d.T
        self.results_data["outlier indices"] = outlier_indices
        self.results_data["calibration config"] = self.calib_config
        self.results_data["task type"] = "calibration"
        self.results_data["condition_number"] = evaluation.get(
            "condition_number", float("nan")
        )
        self.results_data["correlated_pairs"] = evaluation.get(
            "correlated_pairs", []
        )

        # Compute validation metrics and store if available
        val_metrics = self._compute_validation_metrics()
        if val_metrics is not None:
            self.results_data["validation_metrics"] = val_metrics

        # Initialize ResultsManager for calibration task
        try:
            from figaroh.utils.results_manager import ResultsManager

            # Get robot name from class or model
            robot_name = getattr(
                self,
                "robot_name",
                getattr(
                    self.model,
                    "name",
                    self.__class__.__name__.lower().replace("calibration", ""),
                ),
            )
            # Initialize results manager for calibration task
            self.results_manager = ResultsManager(
                "calibration", robot_name, self.results_data
            )

        except ImportError as e:
            logger.warning(f"ResultsManager not available: {e}")
            self.results_manager = None

        # Update status
        self.STATUS = "CALIBRATED"

    def solve_optimisation(
        self,
        var_init: Optional[np.ndarray] = None,
        method: str = "lm",
        max_iterations: int = 3,
        outlier_threshold: float = 3.0,
        enable_logging: bool = False,
    ):
        """Solve calibration optimization with robust outlier handling.

        This method implements a comprehensive optimization strategy:
        1. Sets up logging for progress tracking
        2. Iteratively removes outliers and re-optimizes
        3. Evaluates solution quality with detailed metrics
        4. Stores results for further analysis

        Args:
            var_init (ndarray, optional): Initial parameter guess. If None,
                                        uses zero initialization.
            max_iterations (int): Maximum outlier removal iterations
            outlier_threshold (float): Outlier detection threshold (std devs)
            enable_logging (bool): Whether to enable terminal logging

        Raises:
            ValueError: If optimization fails completely
            AssertionError: If required data is not loaded
            CalibrationError: If optimization fails

        Side Effects:
            - Updates self.LM_result with optimization results
            - Updates self.STATUS to "CALIBRATED" on success
            - Creates self.evaluation_metrics with quality metrics
            - Sets up logging if enabled
        """
        # Verify prerequisites
        if not hasattr(self, "PEE_measured"):
            raise CalibrationError("Call load_data_set() first")
        if not hasattr(self, "q_measured"):
            raise CalibrationError("Call load_data_set() first")

        # Setup logging
        if enable_logging:
            logger = self._setup_logging()
            logger.info("Starting calibration optimization")
            logger.info(f"Parameters: {len(self.calib_config['param_name'])}")
            logger.info(f"Parameter names: {self.calib_config['param_name']}")
            logger.info(f"Markers: {self.calib_config['NbMarkers']}")
            logger.info(f"Samples: {self.calib_config['NbSample']}")
            logger.info(f"DOFs: {self.calib_config['calibration_index']}")

        # Initialize parameters
        if var_init is None:
            var_init, _ = initialize_variables(self.calib_config, mode=0)

        try:
            # Run optimization with outlier removal
            result, outlier_indices, final_residuals = (
                self._optimize_with_outlier_removal(
                    var_init, method, max_iterations, outlier_threshold
                )
            )

            return result, outlier_indices

        except Exception as e:
            if enable_logging:
                logger = logging.getLogger("calibration")
                logger.error(f"Calibration failed: {str(e)}")
            raise CalibrationError(f"Optimization failed: {str(e)}")

    def calc_stddev(self, result):
        """Calculate parameter uncertainty statistics from optimization results.

        Computes standard deviation and percentage uncertainty for each
        calibrated parameter using the covariance matrix derived from the
        Jacobian at the optimal solution. This provides confidence intervals
        and parameter reliability metrics.

        The calculation uses the linearized uncertainty propagation:
        σ²(θ) = σ²(residuals) * (J^T J)^-1

        Where J is the Jacobian matrix and σ²(residuals) is the residual
        variance estimate.

        Prerequisites:
            - Calibration optimization must be completed
            - Jacobian matrix must be available from optimization

        Side Effects:
            - Sets self.std_dev with parameter standard deviations
            - Sets self.std_pctg with percentage uncertainties

        Raises:
            CalibrationError: If calibration has not been performed
            np.linalg.LinAlgError: If Jacobian matrix is singular or ill-conditioned

        Example:
            >>> calibrator.solve()
            >>> calibrator.calc_stddev()
            >>> print(f"Parameter uncertainties: {calibrator.std_dev}")
            >>> print(f"Percentage errors: {calibrator.std_pctg}")
        """
        try:
            sigma_ro_sq = (result.cost**2) / (
                self.calib_config["NbSample"] * self.calib_config["calibration_index"]
                - self.nvars
            )
            J = result.jac
            C_param = sigma_ro_sq * np.linalg.pinv(np.dot(J.T, J))
            self._C_param = C_param
            std_dev = []
            std_pctg = []
            for i_ in range(self.nvars):
                std_dev.append(np.sqrt(C_param[i_, i_]))
                if result.x[i_] != 0:
                    std_pctg.append(abs(np.sqrt(C_param[i_, i_]) / result.x[i_]))
                else:
                    std_pctg.append(0.0)
            self.std_dev = std_dev
            self.std_pctg = std_pctg
        except Exception as e:
            raise CalibrationError(f"Standard deviation calculation failed: {e}")

    def plot_errors_distribution(self):
        """Plot error distribution analysis for calibration assessment.

        Creates bar plots showing pose error magnitudes across all samples
        and markers. This visualization helps identify problematic
        measurements, assess calibration quality, and detect outliers in
        the dataset.

        The plots display error magnitudes (in meters) for each sample,
        with separate subplots for each marker when multiple markers are used.

        Prerequisites:
            - Calibration must be completed (STATUS == "CALIBRATED")
            - Error analysis must be computed (self._PEE_dist available)

        Side Effects:
            - Creates matplotlib figure with error distribution plots
            - Figure remains open until explicitly closed or plt.show() called

        Raises:
            CalibrationError: If calibration has not been performed
            AttributeError: If error analysis data is not available

        See Also:
            plot_3d_poses: 3D visualization of pose comparisons
            calc_stddev: Error statistics computation
        """
        if self.STATUS != "CALIBRATED":
            raise CalibrationError("Calibration not performed yet")

        fig1, ax1 = plt.subplots(self.calib_config["NbMarkers"], 1)
        colors = ["blue", "red", "yellow", "purple"]

        if self.calib_config["NbMarkers"] == 1:
            ax1.bar(np.arange(self.calib_config["NbSample"]), self._PEE_dist[0, :])
            ax1.set_xlabel("Sample", fontsize=25)
            ax1.set_ylabel("Error (meter)", fontsize=30)
            ax1.tick_params(axis="both", labelsize=30)
            ax1.grid()
        else:
            for i in range(self.calib_config["NbMarkers"]):
                ax1[i].bar(
                    np.arange(self.calib_config["NbSample"]),
                    self._PEE_dist[i, :],
                    color=colors[i],
                )
                ax1[i].set_xlabel("Sample", fontsize=25)
                ax1[i].set_ylabel("Error of marker %s (meter)" % (i + 1), fontsize=25)
                ax1[i].tick_params(axis="both", labelsize=30)
                ax1[i].grid()

    def plot_3d_poses(self, INCLUDE_UNCALIB: bool = False):
        """Plot 3D poses comparing measured vs estimated poses.

        Args:
            INCLUDE_UNCALIB (bool): Whether to include uncalibrated poses
        """
        if self.STATUS != "CALIBRATED":
            raise CalibrationError("Calibration not performed yet")

        fig2 = plt.figure()
        fig2.suptitle("Visualization of estimated poses and measured pose in Cartesian")
        ax2 = fig2.add_subplot(111, projection="3d")
        PEEm_LM2d = self.PEE_measured.reshape(
            (
                self.calib_config["NbMarkers"] * self.calib_config["calibration_index"],
                self.calib_config["NbSample"],
            )
        )
        PEEe_sol = self.get_pose_from_measure(self.LM_result.x)
        PEEe_sol2d = PEEe_sol.reshape(
            (
                self.calib_config["NbMarkers"] * self.calib_config["calibration_index"],
                self.calib_config["NbSample"],
            )
        )
        PEEe_uncalib = self.get_pose_from_measure(self.uncalib_values)
        PEEe_uncalib2d = PEEe_uncalib.reshape(
            (
                self.calib_config["NbMarkers"] * self.calib_config["calibration_index"],
                self.calib_config["NbSample"],
            )
        )
        for i in range(self.calib_config["NbMarkers"]):
            ax2.scatter3D(
                PEEm_LM2d[i * 3, :],
                PEEm_LM2d[i * 3 + 1, :],
                PEEm_LM2d[i * 3 + 2, :],
                marker="^",
                color="blue",
                label="Measured",
            )
            ax2.scatter3D(
                PEEe_sol2d[i * 3, :],
                PEEe_sol2d[i * 3 + 1, :],
                PEEe_sol2d[i * 3 + 2, :],
                marker="o",
                color="red",
                label="Estimated",
            )
            if INCLUDE_UNCALIB:
                ax2.scatter3D(
                    PEEe_uncalib2d[i * 3, :],
                    PEEe_uncalib2d[i * 3 + 1, :],
                    PEEe_uncalib2d[i * 3 + 2, :],
                    marker="x",
                    color="green",
                    label="Uncalibrated",
                )
            for j in range(self.calib_config["NbSample"]):
                ax2.plot3D(
                    [PEEm_LM2d[i * 3, j], PEEe_sol2d[i * 3, j]],
                    [PEEm_LM2d[i * 3 + 1, j], PEEe_sol2d[i * 3 + 1, j]],
                    [PEEm_LM2d[i * 3 + 2, j], PEEe_sol2d[i * 3 + 2, j]],
                    color="red",
                )
                if INCLUDE_UNCALIB:
                    ax2.plot3D(
                        [PEEm_LM2d[i * 3, j], PEEe_uncalib2d[i * 3, j]],
                        [
                            PEEm_LM2d[i * 3 + 1, j],
                            PEEe_uncalib2d[i * 3 + 1, j],
                        ],
                        [
                            PEEm_LM2d[i * 3 + 2, j],
                            PEEe_uncalib2d[i * 3 + 2, j],
                        ],
                        color="green",
                    )
        ax2.set_xlabel("X - front (meter)")
        ax2.set_ylabel("Y - side (meter)")
        ax2.set_zlabel("Z - height (meter)")
        ax2.grid()
        ax2.legend()

    def plot_joint_configurations(self):
        """Plot joint configurations within range bounds."""
        fig4 = plt.figure()
        fig4.suptitle("Joint configurations with joint bounds")
        ax4 = fig4.add_subplot(111, projection="3d")
        lb = ub = []
        for j in self.calib_config["config_idx"]:
            lb = np.append(lb, self.model.lowerPositionLimit[j])
            ub = np.append(ub, self.model.upperPositionLimit[j])
        q_actJoint = self.q_measured[:, self.calib_config["config_idx"]]
        sample_range = np.arange(self.calib_config["NbSample"])
        for i in range(len(self.calib_config["actJoint_idx"])):
            ax4.scatter3D(q_actJoint[:, i], sample_range, i)
        for i in range(len(self.calib_config["actJoint_idx"])):
            ax4.plot([lb[i], ub[i]], [sample_range[0], sample_range[0]], [i, i])
            ax4.set_xlabel("Angle (rad)")
            ax4.set_ylabel("Sample")
            ax4.set_zlabel("Joint")
            ax4.grid()

    def save_results(self, output_dir="results"):
        """Save calibration results using unified results manager."""
        if not hasattr(self, "result") or self.results_data is None:
            logger.warning("No calibration results to save. Run solve() first.")
            return

        # Use pre-initialized results manager if available
        if hasattr(self, "results_manager") and self.results_manager is not None:
            try:
                # Save using unified manager with self.result data
                saved_files = self.results_manager.save_results(
                    output_dir=output_dir, save_formats=["yaml", "csv", "npz"]
                )

                logger.info("Calibration results saved using ResultsManager")
                for fmt, path in saved_files.items():
                    logger.info(f"  {fmt}: {path}")

                return saved_files

            except Exception as e:
                logger.error(f"Error saving with ResultsManager: {e}")
                logger.info("Falling back to basic saving...")

    def export_html_report(
        self, output_path: str = None, output_dir: str = "results"
    ) -> str:
        """Export the calibration quality report as a self-contained HTML
        file — the visual counterpart of :meth:`print_quality_report`.

        Renders the same metrics (convergence, per-DOF residuals,
        parameter uncertainty, correlation, validation) already computed
        during :meth:`solve`, plus an auto-generated "insights" section
        flagging ill-conditioning, poorly identified parameters, and
        strongly correlated pairs.

        Args:
            output_path: Explicit file path for the report. If omitted,
                defaults to ``{output_dir}/calibration_report.html``.
            output_dir: Directory used when ``output_path`` is omitted.

        Returns:
            str: The path the report was written to.

        Raises:
            AttributeError: If called before :meth:`solve`.
        """
        if not hasattr(self, "evaluation_metrics"):
            raise AttributeError(
                "No calibration results available. Run solve() first."
            )

        from os import makedirs
        from os.path import join
        from figaroh.tools.report import generate_calibration_report

        if output_path is None:
            makedirs(output_dir, exist_ok=True)
            output_path = join(output_dir, "calibration_report.html")

        generate_calibration_report(self, output_path=output_path)
        logger.info(f"HTML quality report written to {output_path}")
        return output_path

    def verify(self, thresholds: Optional[Dict[str, Dict[str, Any]]] = None):
        """Check this calibration's metrics against pass/fail thresholds.

        Unlike :meth:`print_quality_report`/:meth:`export_html_report`
        (for a human to read), this returns a machine-checkable
        :class:`~figaroh.tools._report_common.VerificationVerdict` a CI
        script can branch on. Computed entirely from data already
        gathered during :meth:`solve` — never raises after a successful
        solve, and never gates ``solve()`` itself (opt-in, called
        whenever the caller wants a verdict).

        Args:
            thresholds: Per-metric ``{"threshold": float, "comparison":
                "max"|"min"}`` overrides. Defaults to
                ``CALIBRATION_DEFAULT_THRESHOLDS`` (a 6-DOF arm and a
                30-DOF humanoid don't share the same bar — override per
                robot as needed).

        Returns:
            VerificationVerdict: ``passed``, per-metric ``checks``, the
            raw ``metrics`` dict, human-readable ``insights`` (the same
            text used by :meth:`export_html_report`), and ``metadata``
            (git commit, config file hash, timestamp, robot name).

        Raises:
            AttributeError: If called before :meth:`solve`.
        """
        if not hasattr(self, "evaluation_metrics"):
            raise AttributeError(
                "No calibration results available. Run solve() first."
            )

        from figaroh.tools._report_common import (
            CALIBRATION_DEFAULT_THRESHOLDS,
            build_provenance_metadata,
            evaluate_thresholds,
        )
        from figaroh.tools.report import _build_insights

        thresholds = (
            thresholds if thresholds is not None
            else CALIBRATION_DEFAULT_THRESHOLDS
        )

        eval_ = self.evaluation_metrics
        n_samples = self.calib_config.get("NbSample", 0)
        param_names = self.calib_config.get("param_name", [])
        results_data = getattr(self, "results_data", None) or {}
        validation = results_data.get("validation_metrics")

        metrics: Dict[str, float] = {
            "condition_number": eval_.get("condition_number", float("nan")),
            "rmse": eval_.get("rmse", float("nan")),
            "outlier_percentage": eval_.get(
                "outlier_percentage", float("nan")
            ),
        }
        if validation is not None:
            metrics["position_rmse_mm"] = validation.get(
                "pos_rmse_calibrated_mm", float("nan")
            )
            metrics["orientation_rmse_deg"] = validation.get(
                "orient_rmse_calibrated_deg", float("nan")
            )

        verdict = evaluate_thresholds(metrics, thresholds)
        verdict.insights = [
            i["text"]
            for i in _build_insights(eval_, n_samples, param_names, validation)
        ]
        robot_name = getattr(
            self,
            "robot_name",
            getattr(
                self.model,
                "name",
                self.__class__.__name__.lower().replace("calibration", ""),
            ),
        )
        verdict.metadata = build_provenance_metadata(
            getattr(self, "_config_file_path", None), robot_name
        )
        return verdict

    def export_verification_report(
        self,
        output_path: str = None,
        output_dir: str = "results",
        thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Write this calibration's :meth:`verify` verdict as JSON.

        Args:
            output_path: Explicit file path. If omitted, defaults to
                ``{output_dir}/calibration_verification.json``.
            output_dir: Directory used when ``output_path`` is omitted.
            thresholds: Forwarded to :meth:`verify`.

        Returns:
            str: The path the JSON verdict was written to.
        """
        import dataclasses
        import json
        from os import makedirs
        from os.path import join

        verdict = self.verify(thresholds=thresholds)
        verdict_dict = dataclasses.asdict(verdict)

        results_manager = getattr(self, "results_manager", None)
        if results_manager is not None:
            verdict_dict = results_manager._convert_for_serialization(
                verdict_dict
            )

        if output_path is None:
            makedirs(output_dir, exist_ok=True)
            output_path = join(output_dir, "calibration_verification.json")

        with open(output_path, "w") as f:
            json.dump(verdict_dict, f, indent=2)

        logger.info(f"Verification report written to {output_path}")
        return output_path

    def print_quality_report(self):
        """Print a formatted calibration quality report to the terminal.

        Reports convergence, per-DOF residual statistics, validation
        metrics (if available), parameter uncertainty, and correlations.
        """
        eval_ = self.evaluation_metrics
        val = (
            self._compute_validation_metrics()
            if hasattr(self, "_compute_validation_metrics")
            else None
        )

        print()
        print("=" * 70)
        print("  CALIBRATION QUALITY REPORT")
        print("=" * 70)

        # ── Convergence ──
        status = (
            "\u2713 converged"
            if eval_["optimization_success"]
            else "\u2717 failed"
        )
        print(
            f"  Convergence:  {status}    "
            f"Iterations: {eval_['n_iterations']}    "
            f"Cost: {eval_['cost']:.6f}"
        )
        print(
            f"  Outliers:     {eval_['n_outliers']} "
            f"/ {self.calib_config['NbSample']} "
            f"({eval_['outlier_percentage']:.1f}%)"
        )

        cond_label = eval_.get("condition_label", "unavailable")
        cond_num = eval_.get("condition_number", float("nan"))
        if not np.isnan(cond_num):
            print(f"  Condition:    {cond_num:.1f} ({cond_label})")
        else:
            print(f"  Condition:    {cond_label}")

        # ── Per-DOF residuals ──
        per_dof = eval_.get("per_dof_stats", {})
        if per_dof and per_dof.get("dof_names"):
            print("-" * 70)
            n = self.calib_config["NbSample"]
            print(f"  Per-DOF Residuals (training set, n={n})")
            names = per_dof["dof_names"]
            means = per_dof.get("mean", [])
            stds = per_dof.get("std", [])
            rmses = per_dof.get("rmse", [])
            maxes = per_dof.get("max_abs", [])
            r2s = per_dof.get("r_squared", [])
            print(
                f"  {'DOF':<12s} {'Mean':>10s} {'Std':>10s} "
                f"{'RMSE':>10s} {'Max':>10s} {'R²':>10s}"
            )
            print(
                f"  {'-'*12} {'-'*10} {'-'*10} "
                f"{'-'*10} {'-'*10} {'-'*10}"
            )
            for i in range(len(names)):
                m = f"{means[i]:10.4f}" if i < len(means) else "         -"
                s = f"{stds[i]:10.4f}" if i < len(stds) else "         -"
                r = f"{rmses[i]:10.4f}" if i < len(rmses) else "         -"
                x = f"{maxes[i]:10.4f}" if i < len(maxes) else "         -"
                q = f"{r2s[i]:10.4f}" if i < len(r2s) else "         -"
                print(f"  {names[i]:<12s} {m} {s} {r} {x} {q}")

        # ── Overall ──
        overall = per_dof.get("overall", {}) if per_dof else {}
        if overall:
            print("-" * 70)
            print("  Overall")
            print(
                f"    Position RMSE:    {overall['pos_rmse_mm']:.2f} mm    "
                f"Orientation RMSE:  {overall['orient_rmse_deg']:.4f} deg"
            )
            print(
                f"    Position max:     {overall['pos_max_mm']:.2f} mm    "
                f"Orientation max:   {overall['orient_max_deg']:.4f} deg"
            )

        # ── Validation ──
        print("-" * 70)
        if val is not None:
            print(f"  Validation (separate set, n={val['n_val_samples']})")
            print(
                f"  {'Metric':<20s} {'Nominal':>10s} "
                f"{'Calibrated':>12s} {'Improvement':>14s}"
            )
            print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*14}")
            arrow_pos = (
                "\u2193" if val["pos_improvement_pct"] > 0 else "\u2191"
            )
            arrow_orient = (
                "\u2193" if val["orient_improvement_pct"] > 0 else "\u2191"
            )
            print(
                f"  {'Position RMSE':<20s} "
                f"{val['pos_rmse_nominal_mm']:10.2f} mm"
                f"{val['pos_rmse_calibrated_mm']:12.2f} mm"
                f"{val['pos_improvement_pct']:13.1f}%  {arrow_pos}"
            )
            print(
                f"  {'Orientation RMSE':<20s} "
                f"{val['orient_rmse_nominal_deg']:10.4f} deg"
                f"{val['orient_rmse_calibrated_deg']:12.4f} deg"
                f"{val['orient_improvement_pct']:13.1f}%  {arrow_orient}"
            )
            print(
                f"  {'Position max':<20s} "
                f"{val['pos_max_nominal_mm']:10.2f} mm"
                f"{val['pos_max_calibrated_mm']:12.2f} mm"
                f"{val['pos_improvement_pct']:13.1f}%  {arrow_pos}"
            )
            print(
                f"  {'Orientation max':<20s} "
                f"{val['orient_max_nominal_deg']:10.4f} deg"
                f"{val['orient_max_calibrated_deg']:12.4f} deg"
                f"{val['orient_improvement_pct']:13.1f}%  {arrow_orient}"
            )
        else:
            print("  Validation: no separate validation data provided.")
            print(
                "    Collect measurements with random configurations "
                "for FK testing."
            )

        # ── Parameter uncertainty (top 5) ──
        std_pctg = eval_.get("param_stddev_percentage", [])
        std_dev = eval_.get("param_stdev", [])
        param_names = self.calib_config.get("param_name", [])
        if std_pctg and param_names:
            print("-" * 70)
            ranked = sorted(
                zip(param_names, std_dev, std_pctg),
                key=lambda x: x[2],
                reverse=True,
            )[:5]
            n_show = min(5, len(ranked))
            print(
                f"  Parameter Uncertainty (top {n_show} most uncertain)"
            )
            print(
                f"  {'Parameter':<30s} {'Value':>12s} "
                f"{'±σ':>12s} {'σ/|val|':>10s}"
            )
            print(
                f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}"
            )
            for name, sd, sp in ranked:
                print(
                    f"  {name:<30s} {'':>12s} "
                    f"{sd:12.6f} {sp:9.1f}%"
                )

        # ── Correlated pairs ──
        corr_pairs = eval_.get("correlated_pairs", [])
        print("-" * 70)
        if corr_pairs:
            print("  Correlated pairs (|\u03c1| > 0.8):")
            for cp in corr_pairs:
                print(
                    f"    {cp['param_i']:<30s} \u2194 "
                    f"{cp['param_j']:<30s} "
                    f"\u03c1 = {cp['correlation']:+.3f}"
                )
        else:
            print(
                "  Parameter correlations: none exceed |\u03c1| > 0.8"
            )

        print("=" * 70)
        print()
