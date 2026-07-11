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
Base class for robot dynamic parameter identification.
This module provides a generalized framework for dynamic parameter identification
that can be inherited by any robot type (TIAGo, UR10, MATE, etc.).
"""

import logging
import yaml
import numpy as np
from abc import ABC, abstractmethod
import dataclasses
from typing import Any, Dict, Optional

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# FIGAROH imports
from figaroh.identification.identification_tools import (
    get_param_from_yaml as get_identification_param_from_yaml,
    unified_to_legacy_identif_config,
)
from figaroh.utils.config_parser import (
    UnifiedConfigParser,
    create_task_config,
    is_unified_config,
)
from figaroh.tools.regressor import (
    build_regressor_basic,
    get_index_eliminate,
    build_regressor_reduced,
)
from figaroh.identification.identification_tools import get_standard_parameters
from figaroh.identification.parameter import (
    add_standard_additional_parameters,
    add_custom_parameters,
)
from figaroh.tools.solver import LinearSolver
from figaroh.utils.results_manager import plot_with_fallback


class BaseIdentification(ABC):
    """
    Base class for robot dynamic parameter identification.

    Provides common functionality for all robots while allowing
    robot-specific implementations of key methods.
    """

    def __init__(self, robot, config_file="config/robot_config.yaml"):
        """Initialize base identification with robot model and configuration.

        Args:
            robot: Robot model loaded with FIGAROH
            config_file: Path to robot configuration YAML file
        """
        self.robot = robot
        self.model = self.robot.model
        self.data = self.robot.data

        # Load configuration initiating self.identif_config
        self.load_param(config_file)

        # Initialize attributes for identification results
        self.dynamic_regressor = None
        self.standard_parameter = None
        self.additional_parameters = None
        self.custom_parameters = None
        self.params_base = None
        self.dynamic_regressor_base = None
        self.phi_base = None
        self.rms_error = None
        self.correlation = None
        self.processed_data = None
        self.result = None
        self.num_samples = None
        self.tau_ref = None
        self.tau_identif = None
        self.tau_noised = None

        # Held-out validation dataset (separate file, never a split of the
        # training data). Populated by _load_validation_data() if
        # identif_config["validation_data_file"] is set.
        self._val_available = False
        self._val_processed_data = None
        self._val_num_samples = None

        # Diagnostics captured during solve() for validation / reporting:
        # column indices eliminated by _eliminate_zero_columns() and the
        # base-column selection from the QR decomposition, both needed to
        # evaluate phi_base against a regressor built from new (held-out)
        # trajectory data.
        self._idx_eliminated = None
        self._base_indices = None
        self._decimate_used = False

        # Set default filter configuration, can be overridden in subclasses
        self.filter_config = self.identif_config.get(
            "filter_config",
            {
                "differentiation_method": "gradient",
                "filter_params": {
                    "nbutter": 4,
                    "f_butter": 2,
                    "med_fil": 5,
                    "f_sample": 100,
                },
            },
        )
        logger.info(f"{self.__class__.__name__} initialized")

    def initialize(self, truncate=None):
        self.process_data(truncate=truncate)
        self.calculate_full_regressor()
        self.initialize_standard_parameters()
        self.compute_reference_torque()

        # If a held-out validation dataset is configured, load and process
        # it now (mirrors BaseCalibration.load_data_set()). Never fails
        # initialization if validation data is missing/unavailable.
        val_data_source = self.identif_config.get("validation_data_file")
        if val_data_source:
            try:
                self._load_validation_data(val_data_source)
            except Exception as e:
                logger.warning(f"Validation data unavailable: {e}")

    def solve(
        self,
        decimate=True,
        decimation_factor=10,
        zero_tolerance=0.001,
        plotting=True,
        save_results=False,
        html_report=False,
    ):
        """Main solving method for dynamic parameter identification.

        This method implements the complete base parameter identification
        workflow including column elimination, optional decimation, QR
        decomposition, and quality metric computation.

        Args:
            decimate (bool): Whether to apply decimation to reduce data size
            decimation_factor (int): Factor for signal decimation (default: 10)
            zero_tolerance (float): Tolerance for eliminating zero columns
            plotting (bool): Whether to generate plots
            save_results (bool): Whether to save parameters to file
            html_report (bool): If True, also export an HTML diagnostic
                report (see :meth:`export_html_report`) after the terminal
                quality report is printed.

        Returns:
            ndarray: Base parameters phi_base

        Raises:
            AssertionError: If prerequisites not met (dynamic_regressor, standard_parameter)
            ValueError: If data shapes are incompatible
            np.linalg.LinAlgError: If QR decomposition fails
        """
        logger.info(
            f"Starting {self.__class__.__name__} dynamic parameter identification..."
        )

        self._decimate_used = decimate

        # Validate prerequisites
        self._validate_prerequisites()

        # Step 1: Eliminate zero columns
        regressor_reduced, active_params = self._eliminate_zero_columns(zero_tolerance)

        # Step 2: Apply decimation if requested
        if decimate:
            tau_processed, W_processed = self._apply_decimation(
                regressor_reduced, decimation_factor
            )
        else:
            tau_processed, W_processed = self._prepare_undecimated_data(
                regressor_reduced
            )

        # Step 3: Calculate base parameters
        results = self._calculate_base_parameters(
            tau_processed, W_processed, active_params
        )

        # Step 4: Store results and compute quality metrics
        self._compute_quality_metrics()
        self._store_results(results)

        # Print quality report
        self.print_quality_report()

        # Step 5: Optional plotting
        if plotting:
            self.plot_results()

        # Step 6: Optional parameter saving
        if save_results:
            self.save_results()

        # Step 7: Optional HTML diagnostic report
        if html_report:
            self.export_html_report()

        return self.phi_base

    def solve_with_custom_solver(
        self,
        method="lstsq",
        regularization=None,
        alpha=0.0,
        constraints=None,
        bounds=None,
        decimate=False,
        decimation_factor=10,
        zero_tolerance=0.001,
        plotting=False,
        save_results=False,
        **solver_kwargs,
    ):
        """
        Alternative solving method using advanced linear solver.

        This method provides more flexibility than the default QR-based
        solve(), offering multiple solving methods, regularization, and
        constraints.

        Args:
            method (str): Solving method ('lstsq', 'ridge', 'lasso',
                'constrained', etc.)
            regularization (str): Regularization type ('l1', 'l2',
                'elastic_net')
            alpha (float): Regularization strength
            constraints (dict): Linear constraints
            bounds (tuple): Box constraints on parameters
            decimate (bool): Whether to apply decimation
            decimation_factor (int): Decimation factor if decimate=True
            zero_tolerance (float): Tolerance for eliminating zero columns
            plotting (bool): Whether to generate plots
            save_results (bool): Whether to save parameters to file
            **solver_kwargs: Additional arguments for LinearSolver

        Returns:
            ndarray: Identified base parameters

        Example:
            >>> # Ridge regression with L2 regularization
            >>> phi = identification.solve_with_custom_solver(
            ...     method='ridge', alpha=0.01)

            >>> # Constrained optimization with physical bounds
            >>> bounds = [(0, 100) for _ in range(n_params)]
            >>> phi = identification.solve_with_custom_solver(
            ...     method='constrained', bounds=bounds)
        """
        logger.info(
            f"Starting {self.__class__.__name__} identification "
            f"with custom solver..."
        )

        # Validate prerequisites
        self._validate_prerequisites()

        # Step 1: Eliminate zero columns
        regressor_reduced, active_params = self._eliminate_zero_columns(zero_tolerance)

        # Step 2: Apply decimation if requested
        if decimate:
            tau_processed, W_processed = self._apply_decimation(
                regressor_reduced, decimation_factor
            )
        else:
            tau_processed, W_processed = self._prepare_undecimated_data(
                regressor_reduced
            )

        # Step 3: Solve using custom solver
        solver = LinearSolver(
            method=method,
            regularization=regularization,
            alpha=alpha,
            constraints=constraints,
            bounds=bounds,
            verbose=True,
            **solver_kwargs,
        )

        # # Solve for reduced parameters
        # phi_reduced = solver.solve(W_processed, tau_processed)

        # # Map back to full parameter space
        # phi_full = np.zeros(len(self.standard_parameter))
        # active_indices = [
        #     i for i, active in enumerate(active_params.values()) if active
        # ]
        # phi_full[active_indices] = phi_reduced

        # Step 4: Compute base parameters using QR decomposition
        from figaroh.tools.qrdecomposition import double_QR

        W_base, _, base_parameters, _, phi_std = double_QR(
            tau_processed, W_processed, active_params, self.standard_parameter
        )
        phi_base = solver.solve(W_base, tau_processed)
        base_param_dict = {
            param: phi_base[i] for i, param in enumerate(base_parameters)
        }

        # Store results
        self.dynamic_regressor_base = W_base
        self.phi_base = phi_base
        self.params_base = list(base_param_dict.keys())
        self.tau_identif = W_base @ phi_base
        self.tau_noised = tau_processed

        # Step 5: Compute quality metrics and store
        self._compute_quality_metrics()

        results = {
            "base_regressor": W_base,
            "base_param_dict": base_param_dict,
            "base_parameters": base_parameters,
            "phi_base": phi_base,
            "tau_estimated": self.tau_identif,
            "tau_processed": tau_processed,
            "solver_info": solver.solver_info,
            "solver_method": method,
            "regularization": regularization,
            "alpha": alpha,
        }

        self._store_results(results)

        # Step 6: Optional plotting
        if plotting:
            self.plot_results()

        # Step 7: Optional parameter saving
        if save_results:
            self.save_results()

        logger.info(f"  RMSE: {self.rms_error:.6f}")
        logger.info(f"  Correlation: {self.correlation:.6f}")

        return self.phi_base

    def load_param(self, config_file, setting_type="identification"):
        """Load the identification parameters from the yaml file.

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
                unified_identif_config = create_task_config(
                    self.robot, unified_config, setting_type
                )
                # Convert unified format to identif_config format
                self.identif_config = unified_to_legacy_identif_config(
                    self.robot, unified_identif_config
                )
            else:
                logger.info("Detected legacy configuration format")
                # Use legacy format parsing
                with open(config_file, "r") as f:
                    config = yaml.load(f, Loader=yaml.SafeLoader)
                self.identif_config = get_identification_param_from_yaml(
                    self.robot, config[setting_type]
                )
        except Exception as e:
            logger.error(f"Error loading config {config_file}: {e}")
            raise

    @abstractmethod
    def load_trajectory_data(self, data_source: str = None):
        """Load and process CSV data.

        This method must be implemented by robot-specific subclasses
        to handle their specific data formats and file structures.

        Args:
            data_source: Optional override identifying an alternate dataset
                to load instead of the class's normal training data (e.g.
                a directory holding a held-out validation trajectory with
                the same file layout/naming as the training data). When
                ``None`` (default), the subclass loads its usual training
                data exactly as before. Passed through from
                ``identif_config["validation_data_file"]`` by
                :meth:`_load_validation_data`.

        Returns:
            dict: Dictionary with keys 'timestamps', 'positions',
            'velocities', 'accelerations', 'torques' (numpy arrays;
            'velocities'/'accelerations' may be None to be derived by
            differentiation).
        """
        pass

    def process_data(self, truncate=None):
        """Load and process data"""
        # Set default filter configuration
        filter_config = self.filter_config

        # load raw data
        self.raw_data = self.load_trajectory_data()

        # Truncate data if truncation indices are provided
        self.raw_data = self._truncate_data(self.raw_data, truncate)

        # Apply filtering and differentiation kinematics data
        self.process_kinematics_data(filter_config)

        # Process joint torque data
        self.processed_data["torques"] = self.process_torque_data()

        # Update sample count to ensure consistency
        self.num_samples = self.processed_data["positions"].shape[0]

        # Build full configuration
        self._build_full_configuration()

    def calculate_full_regressor(self):
        """Build regressor matrix, compute pre-identified values of standard
        parameters, compute joint torques based on pre-identified standard
        parameters."""
        # Build full regressor matrix
        self.dynamic_regressor = build_regressor_basic(
            self.robot,
            self.processed_data["positions"],
            self.processed_data["velocities"],
            self.processed_data["accelerations"],
            self.identif_config,
        )

    def initialize_standard_parameters(
        self,
    ):
        """Initialize standard parameters for the robot."""

        # Compute standard parameters
        self.standard_parameter = get_standard_parameters(
            self.model, self.identif_config
        )

        # additional parameters can be added in robot-specific subclass
        if (
            self.identif_config.get("has_friction", False)
            or self.identif_config.get("has_actuator_inertia", False)
            or self.identif_config.get("has_joint_offset", False)
        ):
            self.additional_parameters = add_standard_additional_parameters(
                self.model, self.identif_config
            )
            self.standard_parameter.update(self.additional_parameters)

        # Add custom parameters specific to the robot
        if self.identif_config.get("has_custom_parameters", False):
            self.custom_parameters = add_custom_parameters(
                self.model, self.identif_config.get("custom_parameters", {})
            )
            self.standard_parameter.update(self.custom_parameters)

        # Convert all string values to floats in the standard_parameter dict
        for key, value in self.standard_parameter.items():
            if isinstance(value, str):
                self.standard_parameter[key] = float(value)

    def compute_reference_torque(self):
        """Compute reference joint torques based on standard parameters and dynamic regressor."""

        # joint torque estimated from p,v,a with std params
        phi_ref = np.array(list(self.standard_parameter.values()))
        tau_ref = np.dot(self.dynamic_regressor, phi_ref)

        # filter only active joints
        self.tau_ref = tau_ref[
            range(len(self.identif_config["act_idxv"]) * self.num_samples)
        ]

    def _load_validation_data(self, data_source: str):
        """Load and process a held-out validation trajectory.

        Unlike a train/validation split of one dataset, this loads a
        genuinely separate dataset — mirrors
        :meth:`BaseCalibration._load_validation_data`. Reuses the exact
        same per-robot loading/filtering/torque-processing pipeline as
        training data (via polymorphic dispatch to
        ``process_kinematics_data`` / ``process_torque_data``), so any
        robot-specific overrides (motor-to-joint conversion, torque
        scaling, etc.) apply identically to the validation set.

        Args:
            data_source: Passed through to :meth:`load_trajectory_data` as
                its ``data_source`` override — e.g. a directory holding
                validation CSVs with the same naming convention as the
                training data.

        Side Effects:
            - Sets self._val_processed_data / self._val_num_samples
            - Sets self._val_available = True
        """
        # Save training state so validation loading can reuse the same
        # instance methods without disturbing the training data already
        # held in self.raw_data / self.processed_data / self.num_samples.
        orig_raw_data = self.raw_data
        orig_processed_data = self.processed_data
        orig_num_samples = self.num_samples

        try:
            self.raw_data = self.load_trajectory_data(data_source=data_source)
            self.raw_data = self._truncate_data(self.raw_data, None)
            self.process_kinematics_data(self.filter_config)
            self.processed_data["torques"] = self.process_torque_data()
            self.num_samples = self.processed_data["positions"].shape[0]
            self._build_full_configuration()

            self._val_processed_data = self.processed_data
            self._val_num_samples = self.num_samples
            self._val_available = True
        finally:
            self.raw_data = orig_raw_data
            self.processed_data = orig_processed_data
            self.num_samples = orig_num_samples

    def _compute_validation_metrics(self):
        """Compute torque-prediction validation metrics on held-out data.

        Evaluates both the pre-identification standard/CAD parameters
        ("nominal") and the identified base parameters ("identified")
        against measured torques on a dataset never used for fitting —
        mirrors :meth:`BaseCalibration._compute_validation_metrics`.

        Returns:
            Dict with validation metrics, or None if no validation data
            (or if solve() has not been run yet).
        """
        if not getattr(self, "_val_available", False):
            return None
        if self._idx_eliminated is None or self._base_indices is None:
            return None

        q_val = self._val_processed_data["positions"]
        dq_val = self._val_processed_data["velocities"]
        ddq_val = self._val_processed_data["accelerations"]

        W_val_full = build_regressor_basic(
            self.robot, q_val, dq_val, ddq_val, self.identif_config
        )
        W_val_reduced = build_regressor_reduced(W_val_full, self._idx_eliminated)
        W_val_base = W_val_reduced[:, self._base_indices]

        n_active = len(self.identif_config["act_idxv"])
        n_val = self._val_num_samples
        n_rows = n_active * n_val

        phi_std_vec = np.array(list(self.standard_parameter.values()))
        tau_val_nominal = (W_val_full @ phi_std_vec)[:n_rows]
        tau_val_identif = (W_val_base @ self.phi_base)[:n_rows]

        # Joint-major, matching the regressor's row convention (block j
        # occupies rows [j*n_val:(j+1)*n_val]).
        tau_val_measured = (
            np.asarray(self._val_processed_data["torques"]).T.flatten()[:n_rows]
        )

        def _stats(estimated):
            residuals = tau_val_measured - estimated
            return {
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "max": float(np.max(np.abs(residuals))),
                "mean": float(np.mean(np.abs(residuals))),
            }

        nominal_stats = _stats(tau_val_nominal)
        identif_stats = _stats(tau_val_identif)

        def _improvement(before, after):
            if before > 0:
                return (before - after) / before * 100
            return 0.0

        correlation = 1.0
        if n_rows > 1:
            try:
                correlation = float(
                    np.corrcoef(tau_val_measured, tau_val_identif)[0, 1]
                )
            except (np.linalg.LinAlgError, ValueError):
                correlation = 1.0

        return {
            "n_val_samples": n_val,
            "rmse_nominal": nominal_stats["rmse"],
            "rmse_identified": identif_stats["rmse"],
            "max_nominal": nominal_stats["max"],
            "max_identified": identif_stats["max"],
            "improvement_pct": _improvement(
                nominal_stats["rmse"], identif_stats["rmse"]
            ),
            "correlation": correlation,
        }

    def _apply_filters(self, *signals, nbutter=4, f_butter=2, med_fil=5, f_sample=100):
        """Apply median and lowpass filters to any number of signals.

        Args:
            *signals: Variable number of signal arrays to filter
            nbutter (int): Butterworth filter order (default: 4)
            f_butter (float): Cutoff frequency in Hz (default: 2)
            med_fil (int): Median filter window size (default: 5)
            f_sample (float): Sampling frequency in Hz (default: 100)

        Returns:
            tuple: Filtered signals in the same order as input
        """
        from scipy import signal

        # Design Butterworth filter coefficients
        b1, b2 = signal.butter(nbutter, f_butter / (f_sample / 2), "low")

        filtered_signals = []

        for sig in signals:
            if sig is None:
                filtered_signals.append(None)
                continue

            # Ensure signal is 2D array
            if sig.ndim == 1:
                sig = sig.reshape(-1, 1)

            sig_filtered = np.zeros(sig.shape)

            # Apply filters to each column (joint/channel)
            for j in range(sig.shape[1]):
                # Apply median filter first
                sig_med = signal.medfilt(sig[:, j], med_fil)

                # Apply Butterworth lowpass filter
                sig_filtered[:, j] = signal.filtfilt(
                    b1,
                    b2,
                    sig_med,
                    padtype="odd",
                    padlen=3 * (max(len(b1), len(b2)) - 1),
                )

            filtered_signals.append(sig_filtered)

        # Return single array if only one signal, otherwise tuple
        if len(filtered_signals) == 1:
            return filtered_signals[0]
        return tuple(filtered_signals)

    def _differentiate_signal(self, time_vector, signal, method="gradient"):
        """Compute first derivative of a time series signal.

        Args:
            time_vector (ndarray): Time stamps corresponding to signal samples
            signal (ndarray): Signal to differentiate (1D or 2D array)
            method (str): Differentiation method ('gradient', 'forward', 'backward', 'central')

        Returns:
            ndarray: First derivative of the signal with same shape as input

        Raises:
            ValueError: If time_vector and signal have incompatible shapes
            ValueError: If method is not supported
        """
        # Validate inputs
        if signal.shape[0] != time_vector.shape[0]:
            raise ValueError(
                f"Time vector length {time_vector.shape[0]} "
                f"doesn't match signal length {signal.shape[0]}"
            )

        # Ensure signal is 2D
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Handle time vector shape (extract first column if 2D)
        if time_vector.ndim == 2:
            t = time_vector[:, 0]
        else:
            t = time_vector

        # Initialize output array
        derivative = np.zeros_like(signal)

        # Apply differentiation method to each column
        for j in range(signal.shape[1]):
            sig_col = signal[:, j]

            if method == "gradient":
                # Use numpy gradient (handles edge cases automatically)
                derivative[:, j] = np.gradient(sig_col, t)

            elif method == "forward":
                # Forward difference: df/dt ≈ (f[i+1] - f[i]) / (t[i+1] - t[i])
                derivative[:-1, j] = np.diff(sig_col) / np.diff(t)
                # Extrapolate last point
                derivative[-1, j] = derivative[-2, j]

            elif method == "backward":
                # Backward difference: df/dt ≈ (f[i] - f[i-1]) / (t[i] - t[i-1])
                derivative[1:, j] = np.diff(sig_col) / np.diff(t)
                # Extrapolate first point
                derivative[0, j] = derivative[1, j]

            elif method == "central":
                # Central difference: df/dt ≈ (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
                derivative[1:-1, j] = (sig_col[2:] - sig_col[:-2]) / (t[2:] - t[:-2])
                # Handle boundary conditions
                derivative[0, j] = (sig_col[1] - sig_col[0]) / (t[1] - t[0])
                derivative[-1, j] = (sig_col[-1] - sig_col[-2]) / (t[-1] - t[-2])

            else:
                raise ValueError(
                    f"Unsupported differentiation method: {method}. "
                    f"Use 'gradient', 'forward', 'backward', or 'central'"
                )

        # Return with original dimensionality
        if squeeze_output:
            return derivative.squeeze()
        return derivative

    def _build_full_configuration(self):
        """Build full configuration arrays for position, velocity, acceleration.

        This method expands the active joint data to full robot configuration
        by filling in default values for inactive joints. Uses vectorized
        operations for optimal performance.
        """
        # Validate required data
        required_keys = ["positions", "velocities", "accelerations"]
        for key in required_keys:
            if key not in self.processed_data or self.processed_data[key] is None:
                raise ValueError(f"Missing required data: {key}")

        # Get active joint data
        q_active = self.processed_data["positions"]
        dq_active = self.processed_data["velocities"]
        ddq_active = self.processed_data["accelerations"]

        # Create full configuration arrays efficiently
        config_data = [
            (q_active, np.zeros_like(self.robot.q0), self.identif_config["act_idxq"]),
            (dq_active, np.zeros_like(self.robot.v0), self.identif_config["act_idxv"]),
            (ddq_active, np.zeros_like(self.robot.v0), self.identif_config["act_idxv"]),
        ]

        full_configs = []
        for active_data, default_config, active_indices in config_data:
            # Initialize with defaults
            full_config = np.tile(default_config, (self.num_samples, 1))
            # Fill active joints
            full_config[:, active_indices] = active_data
            full_configs.append(full_config)

        # Update processed data efficiently
        config_keys = ["positions", "velocities", "accelerations"]
        self.processed_data.update(dict(zip(config_keys, full_configs)))

    def _truncate_data(self, data_dict, truncate=None):
        """Truncate data arrays based on provided indices.

        Args:
            data_dict (dict): Dictionary containing data arrays to truncate
            truncate (tuple/list): Truncation indices (start, end) or None for no truncation

        Returns:
            dict: Dictionary with truncated data arrays
        """
        if truncate is None:
            return data_dict.copy()

        if not isinstance(truncate, (list, tuple)) or len(truncate) != 2:
            raise ValueError(
                "Truncate parameter must be a tuple/list of length 2 (start, end)"
            )

        n_i, n_f = truncate
        truncated_data = {}

        for key, array in data_dict.items():
            if array is not None:
                truncated_data[key] = array[n_i:n_f]
            else:
                truncated_data[key] = None

        return truncated_data

    def process_kinematics_data(self, filter_config=None):
        """Process kinematics data (positions, velocities, accelerations) with filtering."""
        self.filter_kinematics_data(filter_config)

    def filter_kinematics_data(self, filter_config=None):
        """Apply filtering to data with configurable parameters.

        Args:
            filter_config (dict, optional): Filter configuration with keys:
                - differentiation_method: Method for derivative estimation
                - filter_params: Parameters for signal filtering

        Raises:
            ValueError: If required data is missing
        """
        # Validate required data
        if self.raw_data.get("timestamps") is None:
            raise ValueError("Timestamps are required for data processing")
        if self.raw_data.get("positions") is None:
            raise ValueError("Position data is required for processing")

        # Create processed data copy to avoid modifying raw data
        self.processed_data = {}

        # Process timestamps (no filtering needed)
        self.processed_data["timestamps"] = self.raw_data["timestamps"]

        # Define signal processing pipeline
        signal_pipeline = [
            ("positions", self.raw_data["positions"], None),
            ("velocities", self.raw_data.get("velocities"), "positions"),
            ("accelerations", self.raw_data.get("accelerations"), "velocities"),
        ]

        # Process signals through pipeline
        for signal_name, signal_data, dependency in signal_pipeline:
            if signal_data is not None:
                # Apply filtering to existing data
                self.processed_data[signal_name] = self._apply_filters(
                    signal_data, **filter_config["filter_params"]
                )
            else:
                # Estimate missing signal from dependency
                if dependency:
                    dependency_data = self.processed_data[dependency]
                    self.processed_data[signal_name] = self._differentiate_signal(
                        self.processed_data["timestamps"],
                        dependency_data,
                        method=filter_config["differentiation_method"],
                    )
                else:
                    raise ValueError(
                        f"Cannot process {signal_name}: no data or dependency"
                    )

    def process_torque_data(self, **kwargs):
        """Process torque data (generic implementation, should be overridden for robot-specific processing)."""
        # Generic torque processing - robots should override this method
        if self.raw_data["torques"] is not None:
            self.processed_data["torques"] = self.raw_data["torques"]
            return self.processed_data["torques"]
        else:
            raise ValueError("Torque data is required for processing")

    def _validate_prerequisites(self):
        """Validate that required data is available for calculation.

        Raises:
            AssertionError: If required attributes are not set
        """
        assert (
            hasattr(self, "dynamic_regressor") and self.dynamic_regressor is not None
        ), (
            "Regressor matrix not calculated. " "Call calculate_full_regressor() first."
        )
        assert (
            hasattr(self, "standard_parameter") and self.standard_parameter is not None
        ), ("Standard parameters not loaded. " "Call calculate_full_regressor() first.")
        assert (
            hasattr(self, "processed_data") and self.processed_data is not None
        ), "Data not processed. Call process_data() first."

    def _eliminate_zero_columns(self, zero_tolerance):
        """Eliminate columns with near-zero values from regressor matrix.

        Args:
            zero_tolerance (float): Tolerance for considering columns as zero

        Returns:
            tuple: (regressor_reduced, active_parameters)
        """
        idx_eliminated, active_parameters = get_index_eliminate(
            self.dynamic_regressor, self.standard_parameter, tol_e=zero_tolerance
        )
        regressor_reduced = build_regressor_reduced(
            self.dynamic_regressor, idx_eliminated
        )
        self.regressor_reduced = regressor_reduced
        self.active_parameters = active_parameters
        self._idx_eliminated = idx_eliminated
        return self.regressor_reduced, self.active_parameters

    def _apply_decimation(self, regressor_reduced, decimation_factor):
        """Apply signal decimation to reduce data size.

        Args:
            regressor_reduced (ndarray): Reduced regressor matrix
            decimation_factor (int): Factor for decimation

        Returns:
            tuple: (tau_decimated, regressor_decimated)
        """
        from scipy import signal

        # Decimate torque data
        tau_decimated_list = []
        num_joints = len(self.identif_config["act_idxv"])

        for i in range(num_joints):
            tau_joint = self.processed_data["torques"][:, i]
            tau_dec = signal.decimate(tau_joint, q=decimation_factor, zero_phase=True)
            tau_decimated_list.append(tau_dec)

        # Concatenate decimated torque data
        tau_decimated = tau_decimated_list[0]
        for i in range(1, len(tau_decimated_list)):
            tau_decimated = np.append(tau_decimated, tau_decimated_list[i])

        # Decimate regressor matrix
        regressor_decimated = self._decimate_regressor_matrix(
            regressor_reduced, decimation_factor
        )

        # Validate that decimated data is properly aligned
        if tau_decimated.shape[0] != regressor_decimated.shape[0]:
            raise ValueError(
                f"Decimated data size mismatch: "
                f"tau_decimated has {tau_decimated.shape[0]} samples, "
                f"regressor_decimated has {regressor_decimated.shape[0]} rows"
            )

        self.tau_decimated = tau_decimated
        self.regressor_decimated = regressor_decimated
        return self.tau_decimated, self.regressor_decimated

    def _decimate_regressor_matrix(self, regressor_reduced, decimation_factor):
        """Decimate the regressor matrix by joints.

        Args:
            regressor_reduced (ndarray): Reduced regressor matrix
            decimation_factor (int): Decimation factor

        Returns:
            ndarray: Decimated regressor matrix
        """
        from scipy import signal

        num_joints = len(self.identif_config["act_idxv"])
        regressor_list = []

        for i in range(num_joints):
            # Extract rows corresponding to joint i
            start_idx = self.identif_config["act_idxv"][i] * self.num_samples
            end_idx = (self.identif_config["act_idxv"][i] + 1) * self.num_samples

            joint_regressor_decimated = []
            for j in range(regressor_reduced.shape[1]):
                column_data = regressor_reduced[start_idx:end_idx, j]
                decimated_column = signal.decimate(
                    column_data, q=decimation_factor, zero_phase=True
                )
                joint_regressor_decimated.append(decimated_column)

            # Reconstruct matrix for this joint
            joint_matrix = np.zeros(
                (len(joint_regressor_decimated[0]), len(joint_regressor_decimated))
            )
            for k, column in enumerate(joint_regressor_decimated):
                joint_matrix[:, k] = column
            regressor_list.append(joint_matrix)

        # Concatenate all joint matrices
        total_rows = sum(matrix.shape[0] for matrix in regressor_list)
        regressor_decimated = np.zeros((total_rows, regressor_list[0].shape[1]))

        current_row = 0
        for matrix in regressor_list:
            next_row = current_row + matrix.shape[0]
            regressor_decimated[current_row:next_row, :] = matrix
            current_row = next_row

        return regressor_decimated

    def _prepare_undecimated_data(self, regressor_reduced):
        """Prepare data without decimation.

        Args:
            regressor_reduced (ndarray): Reduced regressor matrix

        Returns:
            tuple: (tau_flattened, regressor_reduced)

        Note:
            ``self.processed_data["torques"]`` has shape ``(num_samples,
            num_active_joints)`` — sample-major. The regressor rows built
            by :mod:`figaroh.tools.regressor` (and ``_apply_decimation``'s
            output) are joint-major: all samples of joint 0, then all
            samples of joint 1, etc. A plain ``.flatten()`` here would
            flatten sample-major, misaligning every row against the
            regressor/estimated-torque order. Transposing first makes the
            flatten joint-major, matching the regressor convention.
        """
        tau_data = self.processed_data["torques"]
        if hasattr(tau_data, "flatten"):
            tau_flattened = tau_data.T.flatten()
        else:
            tau_flattened = tau_data
        return tau_flattened, regressor_reduced

    def _calculate_base_parameters(
        self, tau_processed, regressor_processed, active_parameters
    ):
        """Calculate base parameters using QR decomposition.

        Args:
            tau_processed (ndarray): Processed torque data
            regressor_processed (ndarray): Processed regressor matrix
            active_parameters (dict): Active parameter dictionary

        Returns:
            dict: Results from QR decomposition
        """
        from figaroh.tools.qrdecomposition import QRDecomposer

        # Perform QR decomposition using explicit decomposer to access M matrix
        decomposer = QRDecomposer(tolerance=getattr(self, "tol_qr", 1e-6))
        W_base, base_param_dict, base_parameters, phi_base, phi_std = (
            decomposer.double_decomposition(
                tau_processed,
                regressor_processed,
                active_parameters,
                self.standard_parameter,
            )
        )

        # Store M matrix and params_r for optional reconstruction
        self._M_matrix = decomposer.get_M()
        self._params_r_for_recon = list(
            decomposer.get_M_labels()[1] or active_parameters.keys()
        )
        # Base-column selection, needed to evaluate phi_base against a
        # regressor built from held-out validation trajectory data (see
        # _compute_validation_metrics()).
        self._base_indices = decomposer.get_base_indices()

        # Calculate torque estimation (avoid redundant computation)
        tau_estimated = np.dot(W_base, phi_base)

        # Store key results for backward compatibility
        self.dynamic_regressor_base = W_base
        self.phi_base = phi_base
        self.params_base = list(base_param_dict.keys())
        self.tau_identif = tau_estimated
        self.tau_noised = tau_processed

        return {
            "base_regressor": W_base,
            "base_param_dict": base_param_dict,
            "base_parameters": base_parameters,
            "phi_base": phi_base,
            "tau_estimated": tau_estimated,
            "tau_processed": tau_processed,
            "M": self._M_matrix,
            "params_r": self._params_r_for_recon,
        }

    def _compute_quality_metrics(self):
        """Compute quality metrics for the identification.

        Side Effects:
            - Updates self.rms_error
            - Updates self.correlation
        """
        from figaroh.identification.identification_tools import relative_stdev

        # Calculate quality metrics
        self.std_relative = relative_stdev(
            self.dynamic_regressor_base, self.phi_base, self.tau_noised
        )
        residuals = self.tau_noised - self.tau_identif
        self.rms_error = np.sqrt(np.mean(residuals**2))

        if len(self.tau_noised) > 1 and len(self.tau_identif) > 1:
            try:
                correlation_matrix = np.corrcoef(self.tau_noised, self.tau_identif)
                self.correlation = correlation_matrix[0, 1]
            except (np.linalg.LinAlgError, ValueError):
                self.correlation = 1.0
        else:
            self.correlation = 1.0

    def _store_results(self, identif_results):
        """Store calculation results in instance attributes.

        Args:
            identif_results (dict): Results from base parameter calculation
        """

        # Store results in instance attribute
        self.result = {
            "base regressor": identif_results["base_regressor"],
            "base parameters": identif_results["base_param_dict"],
            "base parameters values": identif_results["phi_base"],
            "base parameters names": list(identif_results["base_param_dict"].keys()),
            "condition number": np.linalg.cond(identif_results["base_regressor"]),
            "torque estimated": identif_results["tau_estimated"],
            "torque processed": identif_results["tau_processed"],
            "std dev of estimated param": self.std_relative,
            "rmse norm (N/m)": self.rms_error,
            "num samples": self.num_samples,
            "identification config": getattr(self, "identif_config", {}),
            "task type": "identification",
        }

        # Optional physical-consistency post-processing (default-off)
        self._apply_physical_consistency_if_enabled(identif_results)

        # Optional full-parameter reconstruction (default-off, v0.4.2)
        self._apply_reconstruction_if_enabled(identif_results)

        # Held-out validation metrics (only present if validation_data_file
        # was configured and successfully loaded)
        val_metrics = self._compute_validation_metrics()
        if val_metrics is not None:
            self.result["validation_metrics"] = val_metrics

        # Initialize ResultsManager for identification task
        try:
            from figaroh.utils.results_manager import ResultsManager

            # Get robot name from class or model
            robot_name = getattr(
                self,
                "robot_name",
                getattr(
                    self.model,
                    "name",
                    self.__class__.__name__.lower().replace("identification", ""),
                ),
            )

            # Initialize results manager for identification task
            self.results_manager = ResultsManager(
                "identification", robot_name, self.result
            )

        except ImportError as e:
            logger.warning(f"ResultsManager not available: {e}")
            self.results_manager = None

    def _apply_physical_consistency_if_enabled(self, identif_results):
        pc_cfg = {}
        raw_cfg = getattr(self, "identif_config", {}).get("physical_consistency", {})
        if isinstance(raw_cfg, dict):
            pc_cfg.update(raw_cfg)

        # Backward-compatible flat keys (if user sets them)
        legacy_enabled = getattr(self, "identif_config", {}).get(
            "physical_consistency_enabled", None
        )
        if legacy_enabled is not None:
            pc_cfg["enabled"] = bool(legacy_enabled)

        enabled = bool(pc_cfg.get("enabled", False))
        if not enabled:
            return

        mass_min = float(pc_cfg.get("mass_min", 1e-6))
        psd_eig_tol = float(pc_cfg.get("psd_eig_tol", -1e-10))
        solver = str(pc_cfg.get("solver", "cvxopt"))
        verbose = bool(pc_cfg.get("verbose", False))
        max_seconds = pc_cfg.get("max_seconds", None)
        if max_seconds is not None:
            max_seconds = float(max_seconds)
        skip_if_feasible = bool(pc_cfg.get("skip_if_feasible", True))

        # Parse weights config (YAML: weights.mode / weights.manual)
        weights_cfg = pc_cfg.get("weights", {})
        if not isinstance(weights_cfg, dict):
            weights_cfg = {}
        weights_mode = str(weights_cfg.get("mode", "auto")).strip().lower()
        proj_weights: Optional[np.ndarray] = None
        if weights_mode == "manual":
            manual = weights_cfg.get("manual", {})
            if not isinstance(manual, dict):
                manual = {}
            w_m = float(manual.get("m", 1.0))
            w_h = float(manual.get("h", 1.0))
            w_sigma = float(manual.get("Sigma", 1.0))
            proj_weights = np.array(
                [
                    w_m,
                    w_h,
                    w_h,
                    w_h,
                    w_sigma,
                    w_sigma,
                    w_sigma,
                    w_sigma,
                    w_sigma,
                    w_sigma,
                ],
                dtype=float,
            )

        # Prefer explicitly provided parameter dicts from the solver output,
        # otherwise fall back to the model's standard parameters.
        source_label = "standard_parameter"
        parameter_dict = getattr(self, "standard_parameter", None)
        if isinstance(identif_results, dict):
            if isinstance(identif_results.get("parameter_dict"), dict):
                source_label = "identif_results.parameter_dict"
                parameter_dict = identif_results["parameter_dict"]
            elif isinstance(identif_results.get("standard_parameter_dict"), dict):
                source_label = "identif_results.standard_parameter_dict"
                parameter_dict = identif_results["standard_parameter_dict"]

        if not isinstance(parameter_dict, dict):
            self.result["physical consistency"] = {
                "enabled": True,
                "status": "skipped",
                "reason": "no parameter_dict available",
            }
            return

        joint_names = pc_cfg.get("joints", None)
        if joint_names is None:
            joint_names = list(self.model.names[1:])

        try:
            from figaroh.identification.physical_consistency import (
                check_p10_feasibility,
                param_dict_with_p10_by_joint,
                p10_by_joint_from_param_dict,
                project_robot_p10_lmi,
            )
        except Exception as e:
            self.result["physical consistency"] = {
                "enabled": True,
                "status": "unavailable",
                "reason": f"import failed: {e}",
            }
            return

        # Build per-joint p10 vectors
        try:
            p10_by_joint = p10_by_joint_from_param_dict(
                parameter_dict=parameter_dict,
                joint_names=joint_names,
            )
        except KeyError as e:
            self.result["physical consistency"] = {
                "enabled": True,
                "status": "skipped",
                "reason": f"missing inertial keys: {e}",
                "source": source_label,
            }
            return

        # Always compute feasibility diagnostics (cheap)
        feasibility = {
            joint: dataclasses.asdict(
                check_p10_feasibility(
                    p10,
                    mass_min=mass_min,
                    psd_eig_tol=psd_eig_tol,
                )
            )
            for joint, p10 in p10_by_joint.items()
        }

        if skip_if_feasible and all(
            rep.get("status") == "feasible" for rep in feasibility.values()
        ):
            self.result["physical consistency"] = {
                "enabled": True,
                "status": "already_feasible",
                "source": source_label,
                "solver": solver,
                "mass_min": mass_min,
                "psd_eig_tol": psd_eig_tol,
                "feasibility": feasibility,
                "raw_parameters": dict(parameter_dict),
                "projected_parameters": dict(parameter_dict),
            }
            return

        # Project using SDP (requires picos backend)
        try:
            from figaroh.identification.cad_constraints import (
                build_cad_constraints_from_config,
            )

            pc_cad_cfg = pc_cfg.get("cad_constraints", {})
            if not isinstance(pc_cad_cfg, dict):
                pc_cad_cfg = {}
            pc_cad_cst = build_cad_constraints_from_config(
                pc_cad_cfg, model=getattr(self, "model", None)
            )
        except Exception:
            pc_cad_cst = None

        try:
            projected_p10_by_joint, robot_report = project_robot_p10_lmi(
                p10_by_link=p10_by_joint,
                mass_min=mass_min,
                psd_eig_tol=psd_eig_tol,
                weights=proj_weights,
                solver=solver,
                verbose=verbose,
                max_seconds=max_seconds,
                cad_constraints=pc_cad_cst,
            )
        except ImportError as e:
            self.result["physical consistency"] = {
                "enabled": True,
                "status": "unavailable",
                "source": source_label,
                "reason": str(e),
                "mass_min": mass_min,
                "psd_eig_tol": psd_eig_tol,
                "feasibility": feasibility,
            }
            return
        except Exception as e:
            self.result["physical consistency"] = {
                "enabled": True,
                "status": "error",
                "source": source_label,
                "reason": str(e),
                "mass_min": mass_min,
                "psd_eig_tol": psd_eig_tol,
                "feasibility": feasibility,
            }
            return

        projected_parameter_dict = param_dict_with_p10_by_joint(
            parameter_dict=dict(parameter_dict),
            p10_by_joint=projected_p10_by_joint,
        )

        self.result["physical consistency"] = {
            "enabled": True,
            "status": robot_report.status,
            "source": source_label,
            "solver": solver,
            "mass_min": mass_min,
            "psd_eig_tol": psd_eig_tol,
            "feasibility": feasibility,
            "projection": dataclasses.asdict(robot_report),
            "raw_parameters": dict(parameter_dict),
            "projected_parameters": projected_parameter_dict,
        }

    def _apply_reconstruction_if_enabled(self, identif_results):
        """Reconstruct full parameter vector from base parameters (v0.4.2).

        Reads ``identif_config["reconstruction"]`` and, if ``enabled`` is
        truthy, runs :func:`~figaroh.identification.reconstruction.reconstruct_full_parameters`.
        Results are stored under ``self.result["reconstruction"]``.

        Args:
            identif_results (dict): Results dict from _calculate_base_parameters;
                expected to contain ``"M"``, ``"phi_base"``, and ``"params_r"``
                keys populated by the QRDecomposer.
        """
        recon_cfg = getattr(self, "identif_config", {}).get("reconstruction", {})
        if not isinstance(recon_cfg, dict):
            recon_cfg = {}
        if not bool(recon_cfg.get("enabled", False)):
            return

        M = identif_results.get("M", getattr(self, "_M_matrix", None))
        phi_base = identif_results.get("phi_base", getattr(self, "phi_base", None))
        params_r = identif_results.get(
            "params_r", getattr(self, "_params_r_for_recon", None)
        )

        if M is None or phi_base is None or params_r is None:
            self.result["reconstruction"] = {
                "enabled": True,
                "status": "skipped",
                "reason": "M matrix not available; run solve() first",
            }
            return

        method = str(recon_cfg.get("method", "nullspace"))
        prior_cfg = recon_cfg.get("prior", {})
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        prior_source = str(prior_cfg.get("source", "dict"))
        prior_yaml_path = prior_cfg.get("yaml_path", None)
        weights = recon_cfg.get("weights", None)
        joint_names = list(self.model.names[1:])
        mass_min = float(recon_cfg.get("mass_min", 1e-6))
        psd_eig_tol = float(recon_cfg.get("psd_eig_tol", -1e-10))
        sdp_cfg = recon_cfg.get("sdp", {})
        if not isinstance(sdp_cfg, dict):
            sdp_cfg = {}
        solver = str(sdp_cfg.get("solver", "cvxopt"))
        max_seconds = sdp_cfg.get("max_seconds", None)

        try:
            from figaroh.identification.reconstruction import (
                BaseResult,
                reconstruct_full_parameters,
            )
        except Exception as exc:
            self.result["reconstruction"] = {
                "enabled": True,
                "status": "unavailable",
                "reason": str(exc),
            }
            return

        base_res = BaseResult(
            M=np.asarray(M, dtype=float),
            phi_base=np.asarray(phi_base, dtype=float).reshape(-1),
            params_r=list(params_r),
        )

        # Parse optional CAD constraints from config
        from figaroh.identification.cad_constraints import (
            build_cad_constraints_from_config,
        )

        cad_cfg = recon_cfg.get("cad_constraints", {})
        if not isinstance(cad_cfg, dict):
            cad_cfg = {}
        cad_cst = build_cad_constraints_from_config(
            cad_cfg, model=getattr(self, "model", None)
        )

        result = reconstruct_full_parameters(
            base_res,
            method=method,
            params_std_prior=getattr(self, "standard_parameter", None),
            prior_source=prior_source,
            prior_yaml_path=prior_yaml_path,
            model=getattr(self, "model", None),
            weights=weights,
            joint_names=joint_names,
            mass_min=mass_min,
            psd_eig_tol=psd_eig_tol,
            solver=solver,
            max_seconds=max_seconds,
            cad_constraints=cad_cst,
        )

        self.result["reconstruction"] = {
            "enabled": True,
            "status": result.status,
            "method": method,
            "base_residual_norm": result.base_residual_norm,
            "objective": result.objective,
            "theta_r_dict": result.as_dict(),
            "params_r": result.params_r,
        }

    def _compute_per_joint_stats(self):
        """Per-joint torque residual statistics (mean/std/RMSE/max), the
        identification analogue of BaseCalibration's per-DOF stats.

        Both ``solve(decimate=True)`` (explicit per-joint loop) and
        ``solve(decimate=False)`` (``_prepare_undecimated_data``, which
        transposes before flattening) produce torque/regressor rows in the
        same joint-major block order, so slicing by joint is safe either
        way.

        Returns:
            Dict with joint_names/mean/std/rmse/max_abs lists, or None if
            unavailable.
        """
        if self.result is None:
            return None

        tau_measured = np.asarray(self.result["torque processed"]).flatten()
        tau_estimated = np.asarray(self.result["torque estimated"]).flatten()
        n_active = len(self.identif_config["act_idxv"])
        if n_active == 0 or tau_measured.size % n_active != 0:
            return None

        n_per_joint = tau_measured.size // n_active
        active_joints = self.identif_config.get("active_joints", [])

        stats = {"joint_names": [], "mean": [], "std": [], "rmse": [], "max_abs": []}
        for i in range(n_active):
            sl = slice(i * n_per_joint, (i + 1) * n_per_joint)
            residual = tau_measured[sl] - tau_estimated[sl]
            name = active_joints[i] if i < len(active_joints) else f"joint_{i}"
            stats["joint_names"].append(name)
            stats["mean"].append(float(np.mean(residual)))
            stats["std"].append(float(np.std(residual)))
            stats["rmse"].append(float(np.sqrt(np.mean(residual**2))))
            stats["max_abs"].append(float(np.max(np.abs(residual))))
        return stats

    def print_quality_report(self):
        """Print a formatted identification quality report to the terminal.

        Reports condition number, overall torque residual statistics,
        per-joint residuals (when available), base-parameter uncertainty,
        held-out validation metrics (if configured), and optional
        physical-consistency / reconstruction status.
        """
        if self.result is None:
            logger.warning("No identification results to report. Run solve() first.")
            return

        result = self.result
        per_joint = self._compute_per_joint_stats()

        print()
        print("=" * 70)
        print("  IDENTIFICATION QUALITY REPORT")
        print("=" * 70)

        cond_num = result.get("condition number", float("nan"))
        n_base = len(result.get("base parameters names", []))
        print(
            f"  Base parameters: {n_base}    "
            f"Samples: {result.get('num samples', 0)}"
        )
        if not np.isnan(cond_num):
            cond_label = (
                "well-conditioned"
                if cond_num < 100
                else "moderately-conditioned"
                if cond_num < 1000
                else "ill-conditioned"
            )
            print(f"  Condition:    {cond_num:.1f} ({cond_label})")
        else:
            print("  Condition:    unavailable")
        print(
            f"  RMSE:         {result.get('rmse norm (N/m)', float('nan')):.4f}    "
            f"Correlation: {self.correlation:.4f}"
        )

        if per_joint is not None:
            print("-" * 70)
            print("  Per-Joint Torque Residuals (training set)")
            names = per_joint["joint_names"]
            print(
                f"  {'Joint':<22s} {'Mean':>10s} {'Std':>10s} "
                f"{'RMSE':>10s} {'Max':>10s}"
            )
            print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for i in range(len(names)):
                m = f"{per_joint['mean'][i]:10.4f}"
                s = f"{per_joint['std'][i]:10.4f}"
                r = f"{per_joint['rmse'][i]:10.4f}"
                x = f"{per_joint['max_abs'][i]:10.4f}"
                print(f"  {names[i]:<22s} {m} {s} {r} {x}")
        else:
            print("-" * 70)
            print("  Per-joint residuals: unavailable")

        # ── Base-parameter uncertainty ──
        print("-" * 70)
        std_relative = getattr(self, "std_relative", None)
        base_names = result.get("base parameters names", [])
        if std_relative is not None and len(std_relative) == len(base_names):
            order = sorted(
                range(len(base_names)), key=lambda i: -abs(std_relative[i])
            )
            print("  Base-Parameter Uncertainty (top 5 by relative std-dev)")
            for i in order[:5]:
                print(f"    {base_names[i]:<50s} {std_relative[i]:8.1f}%")
        else:
            print("  Base-parameter uncertainty: unavailable")

        # ── Validation ──
        print("-" * 70)
        val = result.get("validation_metrics")
        if val is not None:
            print(f"  Validation (separate set, n={val['n_val_samples']})")
            print(
                f"    RMSE nominal:    {val['rmse_nominal']:.4f}    "
                f"RMSE identified: {val['rmse_identified']:.4f}"
            )
            print(
                f"    Improvement:     {val['improvement_pct']:.1f}%    "
                f"Correlation:     {val['correlation']:.4f}"
            )
        else:
            print("  Validation: no separate validation data provided.")
            print(
                "    Set validation_data_file in the identification config "
                "to enable held-out FK/torque validation."
            )

        # ── Optional physical consistency / reconstruction ──
        pc = result.get("physical consistency")
        if pc is not None:
            print("-" * 70)
            print(f"  Physical consistency: {pc.get('status', 'unknown')}")
        recon = result.get("reconstruction")
        if recon is not None:
            print("-" * 70)
            print(f"  Full-parameter reconstruction: {recon.get('status', 'unknown')}")

        print("=" * 70)

    def export_html_report(
        self, output_path: str = None, output_dir: str = "results"
    ) -> str:
        """Export the identification quality report as a self-contained
        HTML file — the visual counterpart of :meth:`print_quality_report`.

        Args:
            output_path: Explicit output file path. If None, writes to
                ``<output_dir>/identification_report.html``.
            output_dir: Directory used when output_path is not given
                (created if missing).

        Returns:
            The path the report was written to.

        Raises:
            AttributeError: If solve() has not been run yet.
        """
        if self.result is None:
            raise AttributeError(
                "No identification results available. Run solve() first."
            )

        from os import makedirs
        from os.path import join

        from figaroh.tools.identification_report import (
            generate_identification_report,
        )

        if output_path is None:
            makedirs(output_dir, exist_ok=True)
            output_path = join(output_dir, "identification_report.html")

        generate_identification_report(self, output_path=output_path)
        logger.info(f"HTML quality report written to {output_path}")
        return output_path

    def verify(self, thresholds: Optional[Dict[str, Dict[str, Any]]] = None):
        """Check this identification's metrics against pass/fail thresholds.

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
                ``IDENTIFICATION_DEFAULT_THRESHOLDS`` (a 6-DOF arm and a
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
        if self.result is None:
            raise AttributeError(
                "No identification results available. Run solve() first."
            )

        from figaroh.tools._report_common import (
            IDENTIFICATION_DEFAULT_THRESHOLDS,
            build_provenance_metadata,
            evaluate_thresholds,
        )
        from figaroh.tools.identification_report import _build_insights

        thresholds = (
            thresholds if thresholds is not None
            else IDENTIFICATION_DEFAULT_THRESHOLDS
        )

        result = self.result
        base_names = result.get("base parameters names", [])
        std_relative_raw = getattr(self, "std_relative", None)
        std_relative = (
            list(std_relative_raw) if std_relative_raw is not None else []
        )
        validation = result.get("validation_metrics")

        metrics: Dict[str, float] = {
            "condition_number": result.get(
                "condition number", float("nan")
            ),
            "rmse": result.get("rmse norm (N/m)", float("nan")),
        }
        if validation is not None:
            metrics["validation_correlation"] = validation.get(
                "correlation", float("nan")
            )
            metrics["validation_improvement_pct"] = validation.get(
                "improvement_pct", float("nan")
            )

        verdict = evaluate_thresholds(metrics, thresholds)
        verdict.insights = [
            i["text"]
            for i in _build_insights(
                result, std_relative, base_names, validation
            )
        ]
        robot_name = getattr(
            self,
            "robot_name",
            self.__class__.__name__.lower().replace("identification", ""),
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
        """Write this identification's :meth:`verify` verdict as JSON.

        Args:
            output_path: Explicit file path. If omitted, defaults to
                ``{output_dir}/identification_verification.json``.
            output_dir: Directory used when ``output_path`` is omitted.
            thresholds: Forwarded to :meth:`verify`.

        Returns:
            str: The path the JSON verdict was written to.
        """
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
            output_path = join(output_dir, "identification_verification.json")

        with open(output_path, "w") as f:
            json.dump(verdict_dict, f, indent=2)

        logger.info(f"Verification report written to {output_path}")
        return output_path

    def plot_results(self):
        """Plot identification results using unified results manager."""
        if not hasattr(self, "result") or self.result is None:
            logger.warning("No identification results to plot. Run solve() first.")
            return

        def _basic_plots():
            try:
                import matplotlib.pyplot as plt

                # Extract data from self.result dictionary
                tau_measured = self.result.get("torque processed", np.array([]))
                tau_identified = self.result.get("torque estimated", np.array([]))
                parameter_values = self.result.get(
                    "base parameters values", np.array([])
                )

                if len(tau_measured) == 0 or len(tau_identified) == 0:
                    logger.warning("No torque data available for plotting")
                    return

                plt.figure(figsize=(12, 8))

                plt.subplot(2, 1, 1)
                plt.plot(tau_measured, label="Measured (with noise)", alpha=0.7)
                plt.plot(tau_identified, label="Identified", alpha=0.7)
                plt.xlabel("Sample")
                plt.ylabel("Torque (Nm)")
                plt.title(f"{self.__class__.__name__} Torque Comparison")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(2, 1, 2)
                if len(parameter_values) > 0:
                    plt.bar(
                        range(len(parameter_values)),
                        parameter_values,
                        alpha=0.7,
                        label="Base Parameters",
                    )
                    plt.xlabel("Parameter Index")
                    plt.ylabel("Parameter Value")
                    plt.title("Identified Base Parameters")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            except ImportError:
                logger.warning("matplotlib not available for plotting")
            except Exception as e:
                logger.warning(f"Plotting failed: {e}")

        # Use pre-initialized results manager if available, else go straight
        # to the basic-plotting fallback.
        if hasattr(self, "results_manager") and self.results_manager is not None:
            plot_with_fallback(
                lambda: self.results_manager.plot_identification_results(
                    n_joints=len(self.identif_config["act_idxv"]),
                    joint_names=self.identif_config.get("active_joints"),
                ),
                _basic_plots,
                logger,
                "identification",
            )
        else:
            _basic_plots()

    def save_results(self, output_dir="results"):
        """Save identification results using unified results manager."""
        if not hasattr(self, "result") or self.result is None:
            logger.warning("No identification results to save. Run solve() first.")
            return

        # Use pre-initialized results manager if available
        if hasattr(self, "results_manager") and self.results_manager is not None:
            try:
                # Save using unified manager with self.result data
                saved_files = self.results_manager.save_results(
                    output_dir=output_dir, save_formats=["yaml", "csv", "npz"]
                )

                logger.info("Identification results saved using ResultsManager")
                for fmt, path in saved_files.items():
                    logger.info(f"  {fmt}: {path}")

                return saved_files

            except Exception as e:
                logger.error(f"Error saving with ResultsManager: {e}")
                logger.info("Falling back to basic saving...")

        # Fallback to basic saving if ResultsManager not available
        try:
            import os
            import yaml
            import datetime

            os.makedirs(output_dir, exist_ok=True)

            # Extract data from self.result dictionary
            parameter_values = self.result.get("base parameters values", np.array([]))
            parameter_names = self.result.get("base parameters names", [])
            condition_number = self.result.get("condition number", 0)
            rmse_norm = self.result.get("rmse norm (N/m)", 0)
            std_dev_param = self.result.get("std dev of estimated param", np.array([]))

            results_dict = {
                "base_parameters": (
                    parameter_values.tolist()
                    if hasattr(parameter_values, "tolist")
                    else parameter_values
                ),
                "parameter_names": [str(p) for p in parameter_names],
                "condition_number": float(condition_number),
                "rmse_norm": float(rmse_norm),
                "standard_deviation": (
                    std_dev_param.tolist()
                    if hasattr(std_dev_param, "tolist")
                    else std_dev_param
                ),
            }

            robot_name = self.__class__.__name__.lower().replace("identification", "")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{robot_name}_identification_results_{timestamp}.yaml"

            with open(os.path.join(output_dir, filename), "w") as f:
                yaml.dump(results_dict, f, default_flow_style=False)

            logger.info(f"Results saved to {output_dir}/{filename}")
            return {filename: os.path.join(output_dir, filename)}

        except Exception as e:
            logger.error(f"Error in fallback saving: {e}")
            return None
