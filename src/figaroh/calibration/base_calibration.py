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
from datetime import datetime, timezone
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
    calc_updated_fkm,
    initialize_variables,
    estimate_frames_closed_form,
    measurement_jacobian,
    select_identifiable_parameters,
    drop_calibration_parameters,
)
from figaroh.calibration import estimation
from figaroh.calibration.parameter import BASE_TPL, EE_TPL
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

_COMPONENT_NAMES = ("X", "Y", "Z", "rx", "ry", "rz")


def _measured_components(calib_config: Dict[str, Any]) -> Dict[str, Any]:
    """Label, scale and kind of each residual row (#100).

    Residual rows are the *measured* pose components in order (see
    ``_compute_logmap_residuals``), not always x, y, z, rx, ry, rz: a
    table-contact calibration measuring ``[z, roll, pitch]`` has a
    position row followed by two orientation rows.

    Returns ``names`` ("Z (mm)", "rx (deg)", ...), ``scales`` (m → mm,
    rad → deg) and the row indices of the measured position
    (``pos_rows``) and orientation (``orient_rows``) components.
    """
    n_dofs = calib_config.get("calibration_index", 3)
    measurability = calib_config.get("measurability")
    if measurability is None or sum(bool(m) for m in measurability) != n_dofs:
        dofs = list(range(n_dofs))  # legacy configs: first n_dofs components
    else:
        dofs = [i for i, m in enumerate(measurability) if m]
    return {
        "names": [f"{_COMPONENT_NAMES[d]} ({'mm' if d < 3 else 'deg'})" for d in dofs],
        "scales": np.array([1000.0 if d < 3 else 180.0 / np.pi for d in dofs]),
        "pos_rows": [i for i, d in enumerate(dofs) if d < 3],
        "orient_rows": [i for i, d in enumerate(dofs) if d >= 3],
    }


def _by_component(flat, calib_config: Dict[str, Any], n_samples: int):
    """(component, point x sample) view of a flat measurement vector.

    Flat vectors are ordered marker, component, sample (``load_data``).
    Columns run over the samples of marker 1, then marker 2, ...; with one
    marker this is the usual (component, sample) array. Every per-component
    statistic and per-sample norm then treats each point of each sample as
    one error (figaroh-plus#119). None if the size does not match.
    """
    n_markers = int(calib_config.get("NbMarkers", 1))
    n_dofs = int(calib_config["calibration_index"])
    arr = np.asarray(flat)
    if arr.size != n_markers * n_dofs * n_samples:
        return None
    arr = arr.reshape(n_markers, n_dofs, n_samples).transpose(1, 0, 2)
    return arr.reshape(n_dofs, n_markers * n_samples)


def _per_point_rows(validation: Dict[str, Any]) -> List[tuple]:
    """(nominal, calibrated) position RMSE in mm per point; [] for one."""
    if validation.get("n_markers", 1) < 2:
        return []
    return list(
        zip(
            validation.get("pos_rmse_nominal_per_point_mm", []),
            validation.get("pos_rmse_calibrated_per_point_mm", []),
        )
    )


def _point_columns(samples, n_samples: int, n_markers: int) -> np.ndarray:
    """Columns of :func:`_by_component` that belong to sample indices."""
    samples = np.asarray(samples, dtype=int)
    return np.concatenate([samples + k * n_samples for k in range(n_markers)])


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
        # Data contract (#55): the observations read, and stage records
        self.observations = None
        self.stages = []

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
        outlier_threshold=None,
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
            max_iterations: Maximum number of fits; between fits, samples
                whose position error exceeds ``outlier_threshold`` are
                excluded and the rest refitted.
            outlier_threshold: Position error [m] above which a sample is
                excluded; ``None`` (default) uses the config's
                ``parameters.outlier_threshold`` (``outlier_eps``).
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
        from figaroh.tools.stages import record_stage

        self._run_started_at = datetime.now(timezone.utc).isoformat()
        try:
            result, outlier_indices = self.solve_optimisation(
                method=method,
                max_iterations=max_iterations,
                outlier_threshold=outlier_threshold,
                enable_logging=enable_logging,
            )
        except Exception as e:
            record_stage(self, "fit", "failed", f"{type(e).__name__}: {e}")
            raise

        # Evaluate solution
        evaluation = self._evaluate_solution(result, outlier_indices)
        record_stage(
            self,
            "fit",
            "ok" if getattr(result, "success", True) else "failed",
            str(getattr(result, "message", "")),
            {
                "rmse": (float(evaluation.get("rmse", float("nan"))), "m"),
                "parameters": len(result.x),
                "excluded_outliers": len(outlier_indices),
            },
        )
        self._run_finished_at = datetime.now(timezone.utc).isoformat()

        # Log final results
        if enable_logging:
            logger.info("=" * 30)
            logger.info("FINAL CALIBRATION RESULTS")
            logger.info("=" * 30)
            self._log_iteration_results("FINAL", result, evaluation)

            if len(outlier_indices) > 0:
                logger.info(f"Samples excluded as outliers: {outlier_indices}")
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

        if estimation.settings(self.calib_config)["method"] != "structural":
            try:
                estimation.configure(self)
                return True
            except Exception as e:
                raise CalibrationError(f"Parameter list creation failed: {e}")

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
            self._record_full_mapping()

            if hasattr(self, "q_measured") and hasattr(self, "PEE_measured"):
                self.eliminate_absorbed_parameters()

            return True

        except Exception as e:
            raise CalibrationError(f"Parameter list creation failed: {e}")

    def _record_full_mapping(self) -> None:
        """Keep the base mapping before rows are dropped, and its frame rows.

        ``eliminate_absorbed_parameters`` deletes rows of
        ``base_mapping_matrix``; the lift (:meth:`redistribute_parameters`,
        figaroh-plus#111) needs every row, with the dropped ones held at 0.
        At ``full_params`` with an unknown base frame, ``add_base_name``
        renames the leading base parameters to ``base_*``: those rows carry
        the base frame, not joint corrections.
        """
        cfg = self.calib_config
        if cfg.get("base_mapping_matrix") is None:
            return
        rows = list(cfg["base_mapping_row_names"])
        cfg["base_mapping_matrix_full"] = np.array(cfg["base_mapping_matrix"])
        cfg["base_mapping_row_names_full"] = rows
        start, _ = cfg["base_mapping_slice"]
        frame_rows = []
        if cfg["calib_model"] == "full_params" and not cfg["known_baseframe"]:
            frame_rows = rows[: max(0, len(BASE_TPL) - start)]
        cfg["base_frame_row_names"] = frame_rows

    def _frame_param_names(self) -> List[str]:
        """Base and tip frame parameters present in ``param_name``."""
        tip = {
            f"{e}_{k + 1}"
            for e in EE_TPL
            for k in range(self.calib_config["NbMarkers"])
        }
        return [n for n in self.calib_config["param_name"] if n in BASE_TPL or n in tip]

    def initial_frame_guess(self) -> Dict[str, float]:
        """Closed-form guess for the unknown base and tip frames.

        See :func:`estimate_frames_closed_form`. Empty when the frames are
        known or the measurement does not determine them (partial position
        measurability, several markers, camera anchor).
        """
        return estimate_frames_closed_form(
            self.model,
            self.data,
            self.q_measured,
            self.PEE_measured,
            self.calib_config,
        )

    def eliminate_absorbed_parameters(self, tol: float = 1e-4) -> List[str]:
        """Drop joint parameters that the base/tip frames absorb on this data.

        The structural selection in :func:`calculate_base_kinematics_regressor`
        uses the joint regressor alone. When the base and tip frames are also
        estimated, some retained joint parameters are combinations of frame
        parameters (e.g. a vertical prismatic or revolute first joint against
        the base's z translation and yaw), so the problem is rank deficient.
        This builds the full measurement Jacobian at the measured
        configurations, with the configured measurability, evaluated at the
        closed-form frame guess, and drops every joint parameter whose
        unit-normalised column is a combination of the frame columns and the
        joint columns kept before it (:func:`select_identifiable_parameters`).
        Frame parameters are always kept.

        Set ``calib_config["eliminate_absorbed_parameters"] = False`` to skip.

        Returns:
            list: Names of the dropped parameters (also stored in
            ``calib_config["absorbed_param_name"]``).
        """
        cfg = self.calib_config
        cfg["absorbed_param_name"] = []
        if not cfg.get("eliminate_absorbed_parameters", True):
            return []
        names = list(cfg["param_name"])
        frames = self._frame_param_names()
        guess = self.initial_frame_guess()
        var0 = np.array([guess.get(n, 0.0) for n in names])
        J = measurement_jacobian(self.model, self.data, var0, self.q_measured, cfg)
        _, dropped = select_identifiable_parameters(J, names, frames, tol=tol)
        if dropped:
            drop_calibration_parameters(cfg, dropped)
            logger.info(
                "Dropped %d parameter(s) absorbed by the base/tip frames: %s",
                len(dropped),
                dropped,
            )
        cfg["absorbed_param_name"] = dropped
        return dropped

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
        from figaroh.data.observations import PoseObservations
        from figaroh.tools.stages import record_stage

        # Same CSV layout and arrays as load_data(); del_list rows are
        # masked rather than deleted, and the config is not written by the
        # loader (#55, #105)
        try:
            self.observations = PoseObservations.from_csv(
                self._data_path,
                self.model,
                self.calib_config,
                del_list=self.del_list_ or (),
            )
            self.PEE_measured, self.q_measured = self.observations.to_legacy(
                self.model, self.calib_config
            )
        except Exception as e:
            record_stage(self, "data", "failed", f"{type(e).__name__}: {e}")
            raise CalibrationError(f"Data loading failed: {e}")
        self.calib_config["NbSample"] = len(self.q_measured)
        record_stage(
            self,
            "data",
            "ok",
            "PoseObservations from CSV",
            {
                "samples": len(self.q_measured),
                "masked_samples": int((~self.observations.mask).sum()),
            },
        )

        # If validation data path is specified in config, load it
        val_data_path = self.calib_config.get("validation_data_file")
        if val_data_path:
            try:
                self._load_validation_data(val_data_path)
            except Exception as e:
                # Don't fail calibration if validation data is unavailable
                logger.warning("Validation data %s not loaded: %s", val_data_path, e)

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
        from figaroh.data.observations import PoseObservations

        # PoseObservations does not write calib_config, so the training
        # sample count is untouched (#105)
        try:
            self._val_observations = PoseObservations.from_csv(
                abspath(path), self.model, self.calib_config
            )
            self._PEE_val, self._q_val = self._val_observations.to_legacy(
                self.model, self.calib_config
            )
            self._val_available = True
        except Exception as e:
            raise CalibrationError(f"Validation data loading failed: {e}")

    def _compute_validation_metrics(self) -> Optional[Dict[str, Any]]:
        """Compute FK validation metrics on held-out data.

        Computes FK with both nominal (zero params) and calibrated
        parameters on the validation set, then compares each against
        the ground-truth measured poses.

        Returns:
            Dict with validation metrics, or None if neither validation
            nor calibration data is available. When no separate
            validation set was configured (or it failed to load), this
            falls back to evaluating against the calibration data
            itself so a V&V report can still be produced — a warning
            is logged and ``"validation_source"`` in the returned dict
            is set to ``"calibration_data_fallback"`` so callers/report
            renderers can flag it as not an independent test.
        """
        if getattr(self, "_val_available", False):
            q_val, PEE_val = self._q_val, self._PEE_val
            validation_source = "validation_data"
        elif hasattr(self, "q_measured") and hasattr(self, "PEE_measured"):
            logger.warning(
                "No separate validation data available "
                "(validation_data_file not configured or failed to "
                "load); falling back to calibration data for "
                "validation metrics. These results are NOT an "
                "independent generalization test."
            )
            q_val, PEE_val = self.q_measured, self.PEE_measured
            validation_source = "calibration_data_fallback"
        else:
            return None

        result = self.LM_result
        zeros = np.zeros_like(result.x)
        # calc_updated_fkm evaluates calib_config["NbSample"] samples
        val_config = dict(self.calib_config, NbSample=len(q_val))

        # FK for nominal and calibrated on validation set
        PEE_nom = calc_updated_fkm(self.model, self.data, zeros, q_val, val_config)
        PEE_cal = calc_updated_fkm(self.model, self.data, result.x, q_val, val_config)

        # Log-map residuals
        n_val = len(q_val)
        resid_nom = self._compute_logmap_residuals(PEE_val, PEE_nom, n_samples=n_val)
        resid_cal = self._compute_logmap_residuals(PEE_val, PEE_cal, n_samples=n_val)

        n_dofs = self.calib_config["calibration_index"]
        n_markers = self.calib_config.get("NbMarkers", 1)

        # (n_dofs, n_markers * n_val): each point of each sample is one error
        resid_nom_2d = _by_component(resid_nom, self.calib_config, n_val)
        resid_cal_2d = _by_component(resid_cal, self.calib_config, n_val)

        # Rows are the measured components, not always x..rz (#100)
        comps = _measured_components(self.calib_config)
        pos_nom = resid_nom_2d[comps["pos_rows"], :]
        pos_cal = resid_cal_2d[comps["pos_rows"], :]
        orient_nom = resid_nom_2d[comps["orient_rows"], :]
        orient_cal = resid_cal_2d[comps["orient_rows"], :]

        def _error_stats(arr_2d):
            """arr_2d: (n_dof_group, n_samples) → per-sample norm → stats.
            NaN when no component of the group is measured."""
            if arr_2d.shape[0] == 0:
                nan = float("nan")
                return {"rmse": nan, "max": nan, "mean": nan}
            per_sample = np.sqrt(np.sum(arr_2d**2, axis=0))
            return {
                "rmse": float(np.sqrt(np.mean(np.sum(arr_2d**2, axis=0)))),
                "max": float(np.max(per_sample)),
                "mean": float(np.mean(per_sample)),
            }

        pos_nom_stats = _error_stats(pos_nom)
        pos_cal_stats = _error_stats(pos_cal)
        orient_nom_stats = _error_stats(orient_nom)
        orient_cal_stats = _error_stats(orient_cal)

        def _improvement(before, after):
            if np.isnan(before):
                return float("nan")
            if before > 0:
                return (before - after) / before * 100
            return 0.0

        # Per-DOF scaled error series (nominal vs. calibrated, in the
        # same mm/deg units as the summary stats above) — feeds
        # verify()'s before/after `series` export (Step 3, Feature 6
        # Phase A). "measured" has no separate error curve of its own
        # (a measured pose's error against itself is zero by
        # construction), so it is exposed as the zero reference line
        # nominal/fitted are being compared against.
        dof_names = comps["names"]
        scales = comps["scales"]
        nom_scaled = resid_nom_2d * scales[:, None]
        cal_scaled = resid_cal_2d * scales[:, None]

        def _per_dof(arr_2d):
            return {dof_names[i]: arr_2d[i].tolist() for i in range(n_dofs)}

        def _per_point(arr_2d):
            """Position RMSE (mm) of each marker; [] without position."""
            if not comps["pos_rows"]:
                return []
            pos = arr_2d[comps["pos_rows"]].reshape(-1, n_markers, n_val)
            return [
                float(_error_stats(pos[:, k])["rmse"] * 1000) for k in range(n_markers)
            ]

        from figaroh.tools.stages import record_stage

        metrics = {"position_rmse": (pos_cal_stats["rmse"], "m"), "samples": n_val}
        if validation_source == "validation_data":
            record_stage(self, "validation", "ok", "held-out postures", metrics)
        else:
            record_stage(
                self,
                "validation",
                "fallback",
                "no held-out data: evaluated on the calibration data, "
                "not an independent test",
                metrics,
            )
        return {
            "n_val_samples": n_val,
            "validation_source": validation_source,
            "dof_names": dof_names,
            "error_nominal_per_dof": _per_dof(nom_scaled),
            "error_fitted_per_dof": _per_dof(cal_scaled),
            "n_markers": n_markers,
            "pos_rmse_nominal_per_point_mm": _per_point(resid_nom_2d),
            "pos_rmse_calibrated_per_point_mm": _per_point(resid_cal_2d),
            "pos_rmse_nominal_mm": pos_nom_stats["rmse"] * 1000,
            "pos_rmse_calibrated_mm": pos_cal_stats["rmse"] * 1000,
            "pos_max_nominal_mm": pos_nom_stats["max"] * 1000,
            "pos_max_calibrated_mm": pos_cal_stats["max"] * 1000,
            "pos_improvement_pct": _improvement(
                pos_nom_stats["rmse"], pos_cal_stats["rmse"]
            ),
            "orient_rmse_nominal_deg": (orient_nom_stats["rmse"] * 180 / np.pi),
            "orient_rmse_calibrated_deg": (orient_cal_stats["rmse"] * 180 / np.pi),
            "orient_max_nominal_deg": (orient_nom_stats["max"] * 180 / np.pi),
            "orient_max_calibrated_deg": (orient_cal_stats["max"] * 180 / np.pi),
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
        n_samples: Optional[int] = None,
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
            n_samples: Number of samples in the arrays. Defaults to the
                training count ``calib_config["NbSample"]``.

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
        if n_samples is None:
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
                M_meas = pin.SE3(pin.rpy.rpyToMatrix(meas_6d[3:6]), meas_6d[0:3])
                M_est = pin.SE3(pin.rpy.rpyToMatrix(est_6d[3:6]), est_6d[0:3])

                # Orientation error — always angle-axis from log map
                delta = M_meas.inverse() * M_est
                motion = pin.log(delta)
                se3_errors[marker, 3:, s] = motion.angular  # ω: angle-axis orientation

                # Position error — body-frame or world-frame
                if position_frame == "body":
                    se3_errors[marker, :3, s] = motion.linear  # v: body-frame
                else:
                    se3_errors[marker, :3, s] = est_6d[:3] - meas_6d[:3]  # world-frame

        # Select only measured DOF rows, flatten to DOF-major → same shape as input
        return se3_errors[:, measured_dofs, :].flatten("C")

    def cost_function(self, var: np.ndarray) -> np.ndarray:
        """Calculate cost function for optimization.

        This method provides a default implementation but should be overridden
        by derived classes to define robot-specific cost computation with
        appropriate weighting. Regularise with priors
        (``estimation.method: map``, #120), not rows appended here.

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

            Then apply weighting:
                >>> weighted_residuals = self.apply_measurement_weighting(
                ...     raw_residuals, pos_weight=1000.0, orient_weight=100.0)
        """
        import warnings

        # Issue warning about using default implementation
        warnings.warn(
            f"Using default cost function for {self.__class__.__name__}. "
            "Consider implementing a robot-specific cost function with "
            "appropriate measurement weighting.",
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

    def _objective(self, var: np.ndarray) -> np.ndarray:
        """Residuals minimised by the solver.

        The robot's :meth:`cost_function`, plus ``w * var`` when the
        estimation method sets prior weights (``map``, ``map_cv``,
        figaroh-plus#113); ``w`` is 0 for frame parameters.
        """
        residuals = self.cost_function(var)
        excluded = getattr(self, "_excluded_rows", None)
        if excluded is not None and len(excluded):
            # excluded outlier samples (figaroh-plus#98): constant zero rows,
            # so they contribute neither cost nor Jacobian
            residuals = np.array(residuals, dtype=float)
            residuals[excluded] = 0.0
        weights = self.calib_config.get("prior_weights")
        if weights is None:
            return residuals
        return np.append(residuals, np.asarray(weights) * var)

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
        outlier_threshold: Optional[float] = None,
    ) -> Tuple:
        """Optimize, excluding samples whose position error exceeds a threshold.

        Each round fits the kept samples, then excludes every sample whose
        position error (the norm of its measured x/y/z residuals, worst
        marker) is above ``outlier_threshold`` metres, and refits. Excluded
        samples are never re-admitted. Rotational components do not take
        part, so the threshold has one unit; with no measured position
        component nothing is excluded (figaroh-plus#98).

        Args:
            var_init (ndarray): Initial parameter guess
            method (str): ``least_squares`` method
            max_iterations (int): Maximum number of fits, so at most
                ``max_iterations - 1`` exclusion rounds
            outlier_threshold (float, optional): Position error [m] above
                which a sample is excluded; ``None`` uses
                ``calib_config["outlier_eps"]`` (``parameters.
                outlier_threshold``), and no threshold excludes nothing

        Returns:
            tuple: (result, excluded sample indices, final residuals)
        """
        logger = logging.getLogger("calibration")
        if outlier_threshold is None:
            outlier_threshold = self.calib_config.get("outlier_eps")
        self._outlier_threshold = outlier_threshold
        excluded: List[int] = []
        self._excluded_rows = None
        n_vars = len(var_init)
        current_var = var_init.copy()

        for iteration in range(max(1, max_iterations)):
            result = least_squares(
                self._objective, current_var, method=method, max_nfev=1000
            )
            if not result.success:
                logger.warning(f"Optimization failed at fit {iteration + 1}")
                break

            PEE_est = self.get_pose_from_measure(result.x)
            residuals = self._compute_logmap_residuals(self.PEE_measured, PEE_est)
            errors = self._sample_position_errors(residuals)
            if outlier_threshold is None or errors is None:
                break
            new = [
                int(i)
                for i in np.flatnonzero(errors > outlier_threshold)
                if i not in excluded
            ]
            if not new:
                break
            if iteration == max(1, max_iterations) - 1:
                logger.warning(
                    f"Samples {new} exceed the outlier threshold "
                    f"({outlier_threshold} m) but max_iterations is reached: "
                    "kept in the fit"
                )
                break
            rows = self._sample_rows(excluded + new)
            if len(self.PEE_measured) - len(rows) <= n_vars:
                logger.warning(
                    f"Excluding samples {new} would leave no more observations "
                    f"than the {n_vars} parameters: kept in the fit"
                )
                break
            excluded = sorted(excluded + new)
            self._excluded_rows = rows
            logger.info(
                f"Excluding samples {new} (position error > "
                f"{outlier_threshold} m), refitting without {len(excluded)} "
                "sample(s)"
            )
            current_var = result.x

        return result, excluded, residuals

    def _sample_rows(self, samples: List[int]) -> np.ndarray:
        """Indices of the measurement residual rows of ``samples``.

        Measurement rows are marker-, then DOF-, then sample-major
        (``PEE_measured`` layout), so a row's sample is its index modulo
        ``NbSample``.
        """
        rows = np.arange(len(self.PEE_measured))
        return rows[np.isin(rows % self.calib_config["NbSample"], samples)]

    def _sample_position_errors(self, residuals: np.ndarray) -> Optional[np.ndarray]:
        """Per-sample position error [m], worst marker; None if not available.

        None when no position component is measured or the residuals do not
        have the ``PEE_measured`` layout.
        """
        n_samples = self.calib_config["NbSample"]
        measured = [
            dof for dof, m in enumerate(self.calib_config["measurability"]) if m
        ]
        position_rows = [k for k, dof in enumerate(measured) if dof < 3]
        if not position_rows or len(residuals) % (len(measured) * n_samples):
            return None
        per_marker = np.asarray(residuals).reshape(-1, len(measured), n_samples)
        errors = np.linalg.norm(per_marker[:, position_rows, :], axis=1)
        return errors.max(axis=0)

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

        # Metrics are computed on the per-sample Euclidean-norm error (all
        # DOFs combined into one physical distance per sample), not on each
        # x/y/z/... component as an independent scalar -- this is the same
        # convention _compute_per_dof_stats()'s "overall" block and
        # _error_stats() (validation table) use, so "RMSE"/"MAE" mean the
        # same thing everywhere in the report instead of differing by
        # sqrt(n_dofs) depending on which number you're looking at.
        #
        # Samples excluded as outliers (figaroh-plus#98) are not part of the
        # fit: the metrics cover the kept samples, and the excluded samples'
        # errors are reported separately.
        kept = np.setdiff1d(np.arange(n_samples), outlier_indices)
        sample_errors = self._sample_position_errors(residuals)
        residuals_2d = _by_component(residuals, self.calib_config, n_samples)
        if residuals_2d is not None:
            n_markers = self.calib_config.get("NbMarkers", 1)
            residuals_2d = residuals_2d[:, _point_columns(kept, n_samples, n_markers)]
            per_sample_error = np.sqrt(np.sum(residuals_2d**2, axis=0))
        else:
            residuals_2d = None
            per_sample_error = np.abs(residuals)

        rmse = np.sqrt(np.mean(per_sample_error**2))
        mae = np.mean(per_sample_error)
        max_error = np.max(per_sample_error)

        # Per-sample metrics
        if residuals_2d is not None:
            mean_sample_rms = np.mean(per_sample_error)
            std_sample_rms = np.std(per_sample_error)
        else:
            mean_sample_rms = rmse
            std_sample_rms = 0.0

        # ── Per-DOF breakdown ──
        per_dof_stats = self._compute_per_dof_stats(
            residuals, n_dofs, n_samples, samples=kept
        )

        # ── Condition number ──
        cond_num, cond_label = self._compute_condition_number(result)

        # Calculate standard deviation of estimated parameters -- must run
        # before _compute_parameter_correlation(), which reads self._C_param
        # and silently returns [] if it isn't set yet (e.g. first solve() on
        # a fresh instance).
        self.calc_stddev(result)

        # ── Parameter correlation ──
        correlated_pairs = self._compute_parameter_correlation()

        return {
            "rmse": rmse,
            "mae": mae,
            "max_error": max_error,
            "mean_sample_rms": mean_sample_rms,
            "std_sample_rms": std_sample_rms,
            "param_values": list(result.x),
            "param_stdev": self.std_dev,
            "param_stddev_percentage": self.std_pctg,
            "residual_dof": getattr(self, "residual_dof", None),
            "n_outliers": len(outlier_indices),
            "outlier_percentage": len(outlier_indices)
            / self.calib_config["NbSample"]
            * 100,
            "excluded_samples": list(outlier_indices),
            "excluded_sample_errors": (
                [float(sample_errors[i]) for i in outlier_indices]
                if sample_errors is not None
                else []
            ),
            "outlier_threshold": getattr(self, "_outlier_threshold", None),
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
        self,
        residuals: np.ndarray,
        n_dofs: int,
        n_samples: int,
        samples: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute per-DOF residual statistics.

        Returns dict with keys: 'dof_names', 'mean', 'std', 'rmse',
        'max_abs', 'r_squared' — each a list of length n_dofs — plus
        'overall': {pos_rmse_mm, orient_rmse_deg, pos_mae_mm,
        orient_mae_deg, pos_max_mm, orient_max_deg}, each the per-sample
        Euclidean-norm error for that DOF group (the measured position
        and orientation components, NaN for a group with none measured)
        aggregated across samples -- same convention
        _evaluate_solution()'s rmse/mae use, so a "Position RMSE"/"Position
        MAE" here always means the same thing as anywhere else in the
        report. Units: position DOFs=mm, orientation DOFs=deg.
        ``samples`` restricts the statistics to those sample indices.
        """
        comps = _measured_components(self.calib_config)
        dof_names = comps["names"]

        residuals_2d = _by_component(residuals, self.calib_config, n_samples)
        if residuals_2d is None:
            return {
                "dof_names": dof_names,
                "mean": [],
                "std": [],
                "rmse": [],
                "max_abs": [],
                "r_squared": [],
            }

        PEE_meas_2d = _by_component(self.PEE_measured, self.calib_config, n_samples)
        if samples is not None:  # only these samples (excluded outliers, #98)
            cols = _point_columns(
                samples, n_samples, self.calib_config.get("NbMarkers", 1)
            )
            residuals_2d = residuals_2d[:, cols]
            PEE_meas_2d = PEE_meas_2d[:, cols]

        means, stds, rmses, max_abs, r_squareds = [], [], [], [], []

        for i in range(n_dofs):
            row = residuals_2d[i, :]
            meas_row = PEE_meas_2d[i, :]

            # Scale: position → mm, orientation → deg
            scale = comps["scales"][i]
            scaled = row * scale

            means.append(float(np.mean(scaled)))
            stds.append(float(np.std(scaled)))
            rmses.append(float(np.sqrt(np.mean(scaled**2))))
            max_abs.append(float(np.max(np.abs(scaled))))

            # R² = 1 - SS_res / SS_tot
            ss_res = np.sum(row**2)
            ss_tot = np.sum((meas_row - np.mean(meas_row)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
            r_squareds.append(float(r2))

        # Overall position/orientation aggregates over the measured
        # components of each kind only; NaN when none is measured (#100)
        def _norm_stats(rows, scale):
            if not rows:
                return float("nan"), float("nan"), float("nan")
            norm = np.sqrt(np.sum(residuals_2d[rows, :] ** 2, axis=0))
            return (
                float(np.sqrt(np.mean(norm**2))) * scale,
                float(np.mean(norm)) * scale,
                float(np.max(norm)) * scale,
            )

        pos_rmse, pos_mae, pos_max = _norm_stats(comps["pos_rows"], 1000.0)
        orient_rmse, orient_mae, orient_max = _norm_stats(
            comps["orient_rows"], 180.0 / np.pi
        )

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
                "pos_mae_mm": pos_mae,
                "orient_mae_deg": orient_mae,
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
                        pairs.append(
                            {
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
                            }
                        )
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

        residuals_2d = _by_component(residuals, self.calib_config, n_samples)
        if residuals_2d is not None:
            # Per-sample Euclidean-norm error of each marker -- same
            # convention as _evaluate_solution()'s rmse/mae, see comment there.
            residuals_3d = residuals.reshape((n_markers, n_dofs, n_samples))
            self._PEE_dist = np.sqrt(np.sum(residuals_3d**2, axis=1))
        else:
            # Fallback for unexpected residual shapes
            self._PEE_dist = np.ones((n_markers, n_samples)) * evaluation["rmse"]

        # Reshape PEE measured for consistency
        PEEm_LM2d = _by_component(self.PEE_measured, self.calib_config, n_samples)
        PEEe_LM2d = _by_component(PEE_est, self.calib_config, n_samples)
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
        self.results_data["correlated_pairs"] = evaluation.get("correlated_pairs", [])

        # Compute validation metrics and store if available
        val_metrics = self._compute_validation_metrics()
        if val_metrics is not None:
            self.results_data["validation_metrics"] = val_metrics
        from figaroh.tools.stages import stages_as_dicts

        self.results_data["stages"] = stages_as_dicts(self)

        # Provenance snapshot — nominal model, config, software, data,
        # timestamps — consumed identically by print_quality_report,
        # export_html_report, verify(), and archive_run() so they can
        # never disagree about what produced this result.
        from figaroh.tools.provenance import collect_run_provenance

        self._run_provenance = collect_run_provenance(self, "calibration")

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
        outlier_threshold: Optional[float] = None,
        enable_logging: bool = False,
    ):
        """Solve calibration optimization with robust outlier handling.

        This method implements a comprehensive optimization strategy:
        1. Sets up logging for progress tracking
        2. Excludes samples above the outlier threshold and refits
        3. Evaluates solution quality with detailed metrics
        4. Stores results for further analysis

        Args:
            var_init (ndarray, optional): Initial parameter guess. If None,
                                        uses zero initialization.
            max_iterations (int): Maximum number of fits
            outlier_threshold (float, optional): Position error [m] above
                which a sample is excluded; ``None`` uses
                ``calib_config["outlier_eps"]``
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

        # Initialize parameters: zero, except the base/tip frames, which start
        # at their closed-form estimate when it is available
        if var_init is None:
            var_init, _ = initialize_variables(self.calib_config, mode=0)
            guess = self.initial_frame_guess()
            for i, name in enumerate(self.calib_config["param_name"]):
                if name in guess:
                    var_init[i] = guess[name]

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
            # self.nvars is set once in __init__, from calib_config["param_name"]
            # *before* initialize()/create_param_list() finishes populating it
            # (e.g. add_base_name/add_pee_name append entries afterward), so it
            # can under-count the actual calibrated parameters. result.x is the
            # solved parameter vector itself — always the true count.
            nvars = len(result.x)
            self.nvars = nvars
            # Residual variance from the measurement residuals only: a
            # subclass cost_function may append regularisation rows after
            # them. (least_squares' result.cost is 0.5 * sum(fun**2), so it
            # must not be squared, #107.)
            n_meas = len(self.PEE_measured)
            r_meas = np.asarray(result.fun)[:n_meas]
            # rows of samples excluded as outliers are zero, not observations
            excluded = getattr(self, "_excluded_rows", None)
            if excluded is not None:
                n_meas -= len(excluded)
            prior_noise = self.calib_config.get("prior_noise")
            self.residual_dof = n_meas - nvars
            if prior_noise is not None:
                # MAP: the prior rows in result.jac make this the posterior
                # covariance, with the noise the priors were scaled by
                sigma_ro_sq = prior_noise**2
            elif self.residual_dof <= 0:
                # as many parameters as observations: the residual
                # variance, hence the uncertainty, is not estimable (#100)
                logger.warning(
                    f"{n_meas} observations for {nvars} parameters: "
                    f"{self.residual_dof} residual degrees of freedom, "
                    "parameter uncertainty not estimable"
                )
                self._C_param = None
                self.std_dev = [float("nan")] * nvars
                self.std_pctg = [float("nan")] * nvars
                return
            else:
                sigma_ro_sq = np.sum(r_meas**2) / self.residual_dof
            # Covariance from the full Jacobian, so regularisation rows act
            # as prior information on the parameters they constrain.
            J = result.jac
            C_param = sigma_ro_sq * np.linalg.pinv(np.dot(J.T, J))
            self._C_param = C_param
            std_dev = []
            std_pctg = []
            for i_ in range(nvars):
                std_dev.append(np.sqrt(C_param[i_, i_]))
                if result.x[i_] != 0:
                    std_pctg.append(abs(np.sqrt(C_param[i_, i_]) / result.x[i_]))
                else:
                    std_pctg.append(0.0)
            self.std_dev = std_dev
            self.std_pctg = std_pctg
        except Exception as e:
            raise CalibrationError(f"Standard deviation calculation failed: {e}")

    def redistribute_parameters(self) -> dict:
        """Joint corrections with standard deviations, for export.

        - ``structural`` method: the fitted base parameters are lifted onto
          every joint parameter by a weighted minimum-norm lift
          (:func:`figaroh.calibration.estimation.lift_structural`,
          figaroh-plus#111): among all joint corrections that reproduce the
          fit, the most plausible under the expected error sizes
          (``calib_config["estimation"]["priors"]``, defaults in
          ``estimation.DEFAULT_PRIORS``). Rows the base frame carries and
          rows the fit dropped are held at 0, so the lifted model predicts
          what the fit predicts, and the result does not depend on which
          representative the QR chose. ``std_dev`` is conditional on that
          choice of lift: it propagates the fitted uncertainty and is 0 in
          directions the data does not see.
        - other methods (figaroh-plus#113): the fitted joint parameters
          directly; candidates left out of the fit are 0.

        Tool-point and base-frame parameters are not included.

        Returns:
            dict: ``{name: {"value": float, "std_dev": float}}`` for every
            joint parameter of the calibration level.

        Raises:
            CalibrationError: If `solve()` hasn't run yet (no `_C_param`/
                `self.var_`), or `create_param_list()` didn't populate the
                base-mapping keys in `calib_config`.
        """
        C_param = getattr(self, "_C_param", None)
        var_ = getattr(self, "var_", None)
        if C_param is None or var_ is None:
            raise CalibrationError(
                "redistribute_parameters requires solve() to have run first"
            )
        report = self.calib_config.get("estimation_report")
        if report is not None:
            # non-structural methods (#113) estimate the full joint
            # parameters directly: nothing to redistribute. Candidates
            # left out of the fit are at nominal (0, std 0).
            names = list(self.calib_config["param_name"])
            std = np.sqrt(np.abs(np.diag(C_param)))
            fitted = {n: (float(v), float(s)) for n, v, s in zip(names, var_, std)}
            return {
                n: {
                    "value": fitted.get(n, (0.0, 0.0))[0],
                    "std_dev": fitted.get(n, (0.0, 0.0))[1],
                }
                for n in report["candidates"]
            }
        if self.calib_config.get("base_mapping_matrix") is None or (
            self.calib_config.get("base_mapping_param_names") is None
        ):
            raise CalibrationError(
                "base mapping matrix not available in calib_config -- "
                "was create_param_list() run?"
            )
        names, theta, cov = estimation.lift_structural(self)
        std = np.sqrt(np.abs(np.diag(cov)))
        return {
            name: {"value": float(theta[i]), "std_dev": float(std[i])}
            for i, name in enumerate(names)
        }

    def joint_corrections(
        self, lift: bool = True, drop_unsupported: bool = False
    ) -> Dict[str, float]:
        """Joint parameter values to write into a URDF (``export_urdf``).

        ``lift=True``: :meth:`redistribute_parameters` (the weighted lift
        for ``structural``; the fitted values otherwise), so the URDF and
        the PAL export carry the same corrections. ``lift=False``: the
        fitted joint parameters as estimated (for ``structural``, one
        representative per dependent group and the rest at 0). Either way
        the reloaded URDF, with :meth:`metrology_frames` applied outside
        it, reproduces the calibrated forward kinematics (figaroh-plus#62).

        Frame parameters are never included; see :meth:`metrology_frames`.

        Args:
            lift: see above.
            drop_unsupported: leave out fitted parameters a URDF cannot
                carry (elastic ``k_*``, contact planes, ...). By default
                they raise, because the reloaded model would then not
                reproduce the calibration.

        Raises:
            CalibrationError: if the fit has parameters that are neither
                frames nor kinematic joint corrections, and
                ``drop_unsupported`` is False.
        """
        from figaroh.tools.urdf_exporter import is_kinematic_correction

        if lift:
            values = {n: v["value"] for n, v in self.redistribute_parameters().items()}
        else:
            frames = set(self._frame_param_names())
            values = {
                n: float(v)
                for n, v in zip(self.calib_config["param_name"], self.var_)
                if n not in frames
            }
        unsupported = [n for n in values if not is_kinematic_correction(n)]
        if unsupported and not drop_unsupported:
            from figaroh.tools.stages import record_stage

            record_stage(
                self,
                "export",
                "failed",
                f"not representable in a URDF: {unsupported[:6]}",
            )
            raise CalibrationError(
                f"{len(unsupported)} fitted parameter(s) cannot be written to "
                f"a URDF, so the exported model would not reproduce this "
                f"calibration: {unsupported[:6]}"
                f"{' ...' if len(unsupported) > 6 else ''}. Pass "
                f"drop_unsupported=True to export the kinematic corrections "
                f"only, and keep the others from calibrator.var_."
            )
        return {n: v for n, v in values.items() if n not in unsupported}

    def metrology_frames(self) -> Dict[str, float]:
        """Fitted base frame and tool point: the measurement setup.

        ``base_*`` places the robot base in the measurement frame (mocap
        world, camera); ``pEE*``/``phiEE*`` place the measured point on the
        tool. They describe this setup, not the robot, so they are never in
        :meth:`joint_corrections` and ``export_urdf`` does not write them.
        To reproduce the calibrated measurements from an exported URDF,
        apply them outside it, e.g. ``calc_updated_fkm(reloaded_model, ...,
        values, q, dict(calib_config, param_name=list(frames)))``.
        Frames given as known (``known_baseframe``/``known_tipframe``) are
        not fitted and not returned.

        Raises:
            CalibrationError: If `solve()` hasn't run yet.
        """
        if getattr(self, "var_", None) is None:
            raise CalibrationError("metrology_frames requires solve() to have run")
        frames = set(self._frame_param_names())
        return {
            n: float(v)
            for n, v in zip(self.calib_config["param_name"], self.var_)
            if n in frames
        }

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
            raise AttributeError("No calibration results available. Run solve() first.")

        from os import makedirs
        from os.path import join
        from figaroh.tools.report import generate_calibration_report

        if output_path is None:
            makedirs(output_dir, exist_ok=True)
            output_path = join(output_dir, "calibration_report.html")

        generate_calibration_report(self, output_path=output_path)
        logger.info(f"HTML quality report written to {output_path}")
        return output_path

    def verify(
        self,
        thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
        scope: str = "prediction",
    ):
        """Return scoped acceptance evidence with explicit incomplete states.

        ``scope="execution"`` checks finite numerical fit outputs;
        it does not certify prediction, physical parameters or export.
        ``scope="prediction"`` (default) additionally requires independent validation
        and explicit application error limits. No universal improvement,
        correlation, conditioning or prediction-error gate is imposed.

        ``thresholds`` maps metric names to ``threshold``, ``comparison``
        (min/max), and optional ``required`` (default True). Missing required
        evidence produces not_evaluated, not PASS. Nonfinite evidence fails.
        ``passed`` is True only when all required checks in scope pass.
        """
        if not hasattr(self, "evaluation_metrics"):
            raise AttributeError("No calibration results available. Run solve() first.")

        from figaroh.tools._report_common import (
            CALIBRATION_DEFAULT_THRESHOLDS,
            scoped_verification,
        )
        from figaroh.tools.report import _build_insights
        from figaroh.tools.provenance import collect_run_provenance

        thresholds = (
            thresholds if thresholds is not None else CALIBRATION_DEFAULT_THRESHOLDS
        )

        eval_ = self.evaluation_metrics
        n_samples = self.calib_config.get("NbSample", 0)
        param_names = self.calib_config.get("param_name", [])
        results_data = getattr(self, "results_data", None) or {}
        validation = results_data.get("validation_metrics")

        metrics: Dict[str, float] = {
            "condition_number": eval_.get("condition_number", float("nan")),
            "rmse": eval_.get("rmse", float("nan")),
            "outlier_percentage": eval_.get("outlier_percentage", float("nan")),
        }
        independent = (
            bool(getattr(self, "_val_available", False)) and validation is not None
        )
        # Position/orientation metrics only for the kinds actually
        # measured, not inferred from the component count (#100)
        comps = _measured_components(self.calib_config)
        prediction_keys = []
        if comps["pos_rows"]:
            prediction_keys.append("position_rmse_mm")
        if comps["orient_rows"]:
            prediction_keys.append("orientation_rmse_deg")
        if validation is not None:
            # Training-data fallback stays labelled as training evidence, so
            # it can never satisfy a validation_* prediction limit.
            prefix = "" if independent else "training_"
            if comps["pos_rows"]:
                metrics[f"{prefix}position_rmse_mm"] = validation.get(
                    "pos_rmse_calibrated_mm", float("nan")
                )
            if comps["orient_rows"]:
                metrics[f"{prefix}orientation_rmse_deg"] = validation.get(
                    "orient_rmse_calibrated_deg", float("nan")
                )
        verdict = scoped_verification(
            metrics,
            thresholds,
            scope,
            {
                "finite_fit_rmse": eval_.get("rmse"),
                "finite_parameters": results_data.get("calibrated parameters values"),
                "finite_residuals": results_data.get("residuals"),
            },
            independent,
            prediction_keys,
            solver_success=eval_.get("optimization_success"),
            facts={
                "requested_validation_loaded": (
                    not self.calib_config.get("validation_data_file")
                    or bool(getattr(self, "_val_available", False))
                )
            },
        )
        verdict.insights = [
            i["text"]
            for i in _build_insights(eval_, n_samples, param_names, validation)
        ]
        verdict.metadata = getattr(
            self, "_run_provenance", None
        ) or collect_run_provenance(self, "calibration")

        dof_names = comps["names"]
        if validation is not None and "error_nominal_per_dof" in validation:
            # one entry per point of each sample (marker-major, #119)
            n_val = validation.get("n_val_samples", 0) * validation.get("n_markers", 1)
            dof_names = validation.get("dof_names", dof_names)
            verdict.series = {
                "time": list(range(n_val)),
                "dof_names": dof_names,
                "nominal": validation["error_nominal_per_dof"],
                "fitted": validation["error_fitted_per_dof"],
                "measured": {name: [0.0] * n_val for name in dof_names},
            }
        verdict.compat = {
            "dof_names": dof_names,
            "sample_count": n_samples,
            "config_sha256": verdict.metadata.get("config", {}).get("sha256"),
        }
        from figaroh.tools.stages import apply_to_verdict

        # the reported parameters come from the fit (phi_base / var_)
        apply_to_verdict(verdict, self, selected_stage="fit")
        return verdict

    def export_verification_report(
        self,
        output_path: str = None,
        output_dir: str = "results",
        thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
        scope: str = "prediction",
    ) -> str:
        """Write this calibration's :meth:`verify` verdict as JSON.

        Args:
            output_path: Explicit file path. If omitted, defaults to
                ``{output_dir}/calibration_verification.json``.
            output_dir: Directory used when ``output_path`` is omitted.
            thresholds: Forwarded to :meth:`verify`.
            scope: Forwarded to :meth:`verify`; default prediction acceptance.

        Returns:
            str: The path the JSON verdict was written to.
        """
        import dataclasses
        import json
        from os import makedirs
        from os.path import join

        verdict = self.verify(thresholds=thresholds, scope=scope)
        verdict_dict = dataclasses.asdict(verdict)

        results_manager = getattr(self, "results_manager", None)
        if results_manager is not None:
            verdict_dict = results_manager._convert_for_serialization(verdict_dict)

        if output_path is None:
            makedirs(output_dir, exist_ok=True)
            output_path = join(output_dir, "calibration_verification.json")

        from figaroh.tools._report_common import verification_json_data
        from figaroh.tools.stages import with_schema as _with_schema

        with open(output_path, "w") as f:
            json.dump(
                verification_json_data(_with_schema(self, verdict_dict)),
                f,
                indent=2,
                allow_nan=False,
            )

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
        from figaroh.tools.stages import stages_line

        print(f"  Stages:       {stages_line(self)}")

        # ── Convergence ──
        status = (
            "\u2713 converged" if eval_["optimization_success"] else "\u2717 failed"
        )
        print(
            f"  Convergence:  {status}    "
            f"Iterations: {eval_['n_iterations']}    "
            f"Cost: {eval_['cost']:.6f}"
        )
        threshold = eval_.get("outlier_threshold")
        print(
            f"  Excluded:     {eval_['n_outliers']} "
            f"/ {self.calib_config['NbSample']} samples "
            f"({eval_['outlier_percentage']:.1f}%), position error > "
            + (f"{threshold * 1e3:.1f} mm" if threshold is not None else "no threshold")
        )
        excluded = eval_.get("excluded_samples") or []
        if excluded:
            errors = eval_.get("excluded_sample_errors") or [float("nan")] * len(
                excluded
            )
            print(
                "                "
                + ", ".join(
                    f"#{i} ({e * 1e3:.1f} mm)" for i, e in zip(excluded, errors)
                )
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
            print(f"  {'-'*12} {'-'*10} {'-'*10} " f"{'-'*10} {'-'*10} {'-'*10}")
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
            print("  Overall (measured components of each kind)")

            def _fmt(value, spec, unit):
                if value is None or np.isnan(value):
                    return "not measured"
                return f"{value:{spec}} {unit}"

            for stat, key in (("RMSE", "rmse"), ("MAE", "mae"), ("max", "max")):
                if f"pos_{key}_mm" not in overall:
                    continue
                pos = _fmt(overall[f"pos_{key}_mm"], ".2f", "mm")
                orient = _fmt(overall[f"orient_{key}_deg"], ".4f", "deg")
                print(
                    f"    {'Position ' + stat + ':':<18s}{pos}    "
                    f"{'Orientation ' + stat + ':':<19s}{orient}"
                )

        # ── Validation ──
        print("-" * 70)
        if val is not None:
            if val.get("validation_source") == "calibration_data_fallback":
                # a held-out set evaluated outside BaseCalibration is not
                # seen here (#100)
                print(
                    "  ⚠ WARNING: no validation_data_file loaded by this "
                    "calibration — falling back to calibration data. "
                    "These are NOT an independent generalization test; "
                    "a held-out evaluation done elsewhere is not shown."
                )
                print(f"  Validation (calibration set, n={val['n_val_samples']})")
            else:
                print(f"  Validation (separate set, n={val['n_val_samples']})")
            print(
                f"  {'Metric':<20s} {'Nominal':>10s} "
                f"{'Calibrated':>12s} {'Improvement':>14s}"
            )
            print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*14}")
            for label, kind, unit, digits in (
                ("Position RMSE", "pos_rmse", "mm", 2),
                ("Orientation RMSE", "orient_rmse", "deg", 4),
                ("Position max", "pos_max", "mm", 2),
                ("Orientation max", "orient_max", "deg", 4),
            ):
                nominal = val[f"{kind}_nominal_{unit}"]
                if np.isnan(nominal):
                    continue  # no component of this kind measured (#100)
                calibrated = val[f"{kind}_calibrated_{unit}"]
                gain = val[f"{kind.split('_')[0]}_improvement_pct"]
                arrow = "\u2193" if gain > 0 else "\u2191"
                print(
                    f"  {label:<20s} "
                    f"{nominal:10.{digits}f} {unit}"
                    f"{calibrated:12.{digits}f} {unit}"
                    f"{gain:13.1f}%  {arrow}"
                )
            for k, (nominal, calibrated) in enumerate(_per_point_rows(val)):
                gain = (nominal - calibrated) / nominal * 100 if nominal else 0.0
                arrow = "\u2193" if gain > 0 else "\u2191"
                print(
                    f"  {f'  point {k + 1} RMSE':<20s} "
                    f"{nominal:10.2f} mm{calibrated:12.2f} mm"
                    f"{gain:13.1f}%  {arrow}"
                )
        else:
            # None also when a subclass evaluates its held-out set
            # itself, so do not claim there is none (#100)
            print("  Validation: not computed by this calibration.")
            print(
                "    Set validation_data_file, or report the held-out "
                "evaluation done outside BaseCalibration."
            )

        # ── Parameter uncertainty (top 5) ──
        std_pctg = eval_.get("param_stddev_percentage", [])
        std_dev = eval_.get("param_stdev", [])
        param_names = self.calib_config.get("param_name", [])
        residual_dof = eval_.get("residual_dof")
        if (
            residual_dof is not None
            and residual_dof <= 0
            and self.calib_config.get("prior_noise") is None
        ):
            print("-" * 70)
            print(
                f"  Parameter Uncertainty: {residual_dof} residual degrees "
                "of freedom — uncertainty not estimable"
            )
        elif std_pctg and param_names:
            print("-" * 70)
            ranked = sorted(
                zip(param_names, std_dev, std_pctg),
                key=lambda x: x[2],
                reverse=True,
            )[:5]
            n_show = min(5, len(ranked))
            print(f"  Parameter Uncertainty (top {n_show} most uncertain)")
            print(
                f"  {'Parameter':<30s} {'Value':>12s} " f"{'±σ':>12s} {'σ/|val|':>10s}"
            )
            print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
            for name, sd, sp in ranked:
                print(f"  {name:<30s} {'':>12s} " f"{sd:12.6f} {sp:9.1f}%")

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
            print("  Parameter correlations: none exceed |\u03c1| > 0.8")

        print("=" * 70)
        print()
