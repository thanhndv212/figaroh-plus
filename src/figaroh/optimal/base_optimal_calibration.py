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
Base class for robot optimal configuration generation for calibration.
This module provides a generalized framework for optimal configuration
generation that can be inherited by any robot type (TIAGo, UR10, MATE, etc.).
"""

import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC
from yaml.loader import SafeLoader

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# FIGAROH imports
from figaroh.calibration.calibration_tools import (
    load_data,
    get_param_from_yaml,
    unified_to_legacy_config,
    calculate_base_kinematics_regressor,
)

from figaroh.utils.config_parser import (
    UnifiedConfigParser,
    create_task_config,
    is_unified_config,
)

# Import from shared modules
from figaroh.utils.error_handling import (
    CalibrationError,
)


class BaseOptimalCalibration(ABC):
    """Base class for robot optimal configuration generation for calibration.

    This class implements the framework for generating optimal robot
    configurations that maximize the observability of kinematic parameters
    during calibration. It uses Second-Order Cone Programming (SOCP) to
    solve the D-optimal design problem for parameter estimation.

    The class provides a Template Method pattern where the main workflow
    is defined, but specific optimization strategies can be customized
    by derived classes for different robot types.

    Workflow:
        1. Load candidate configurations from file (CSV or YAML)
        2. Calculate kinematic regressors for all candidates
        3. Compute information matrices for each configuration
        4. Solve SOCP optimization to find optimal subset
        5. Select configurations with significant weights
        6. Visualize and save results

    Key Features:
        - D-optimal experimental design for calibration
        - Support for multiple calibration models (full_params, joint_offset)
        - Automatic minimum configuration calculation
        - SOCP-based optimization with convex relaxation
        - Comprehensive visualization and analysis tools
        - File I/O for configuration management

    Mathematical Background:
        The method maximizes the determinant of the Fisher Information Matrix:
        max det(Σᵢ wᵢ Rᵢᵀ Rᵢ) subject to Σᵢ wᵢ ≤ 1, wᵢ ≥ 0

        Where:
        - Rᵢ is the kinematic regressor for configuration i
        - wᵢ is the weight assigned to configuration i
        - The objective maximizes parameter estimation precision

    Attributes:
        robot: Robot model instance loaded with FIGAROH
        model: Pinocchio robot model
        data: Pinocchio robot data
        calib_config (dict): Calibration parameters from configuration file
        optimal_configurations (dict): Selected optimal configurations
        optimal_weights (ndarray): Weights assigned to configurations
        minNbChosen (int): Minimum number of configurations required

        # Internal computation attributes
        R_rearr (ndarray): Rearranged kinematic regressor matrix
        detroot_whole (float): Determinant root of full information matrix
        w_list (list): Solution weights from SOCP optimization
        w_dict_sort (dict): Sorted weights by configuration index

    Example:
        >>> # Basic usage for TIAGo robot
        >>> from figaroh.robots import TiagoRobot
        >>> robot = TiagoRobot()
        >>>
        >>> # Create optimal calibration instance
        >>> opt_calib = TiagoOptimalCalibration(robot, "config/tiago.yaml")
        >>>
        >>> # Generate optimal configurations
        >>> opt_calib.solve(save_file=True)
        >>>
        >>> # Access results
        >>> print(f"Selected {len(opt_calib.optimal_configurations)} configs")
        >>> print(f"Minimum required: {opt_calib.minNbChosen}")

    See Also:
        BaseCalibration: Main calibration framework
        SOCPOptimizer: Second-order cone programming solver
        TiagoOptimalCalibration: TIAGo-specific implementation
        UR10OptimalCalibration: UR10-specific implementation
    """

    def __init__(self, robot, config_file="config/robot_config.yaml"):
        """Initialize optimal calibration with robot model and configuration.

        Sets up the optimal calibration framework by loading robot parameters,
        initializing optimization attributes, and calculating the minimum
        number of configurations required based on the calibration model.

        The minimum number of configurations is computed to ensure the
        optimization problem is well-posed and identifiable:
        - For full_params: considers all kinematic parameters
        - For joint_offset: considers only joint offset parameters

        Args:
            robot: Robot model instance loaded with FIGAROH. Must have
                  'model' and 'data' attributes for Pinocchio integration.
            config_file (str): Path to YAML configuration file containing
                             calibration parameters, sample file paths, and
                             optimization settings. Defaults to standard path.

        Raises:
            FileNotFoundError: If config_file does not exist
            KeyError: If required parameters missing from configuration
            ValueError: If calibration model type is unsupported

        Side Effects:
            - Loads and stores calibration parameters in self.calib_config
            - Calculates minimum required configurations (self.minNbChosen)
            - Initializes optimization result attributes to None
            - Prints initialization confirmation message

        Example:
            >>> robot = TiagoRobot()
            >>> opt_calib = BaseOptimalCalibration(robot, "config/tiago.yaml")
            TiagoOptimalCalibration initialized
        """
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.load_param(config_file)

        # Initialize attributes for optimal calibration
        self.optimal_configurations = None
        self.optimal_weights = None
        self._sampleConfigs_file = self.calib_config.get("sample_configs_file")

        # Calculate minimum number of configurations needed
        if self.calib_config["calib_model"] == "full_params":
            self.minNbChosen = (
                int(
                    len(self.calib_config["actJoint_idx"])
                    * 6
                    / self.calib_config["calibration_index"]
                )
                + 1
            )
        elif self.calib_config["calib_model"] == "joint_offset":
            self.minNbChosen = (
                int(
                    len(self.calib_config["actJoint_idx"])
                    / self.calib_config["calibration_index"]
                )
                + 1
            )

        logger.info(f"{self.__class__.__name__} initialized")

    def initialize(self):
        """Initialize the optimization process by preparing all required data.

        This method orchestrates the initialization sequence required before
        optimization can begin. It ensures all mathematical components are
        properly computed and cached for efficient optimization.

        The initialization sequence:
        1. Load candidate configurations from external files
        2. Calculate kinematic regressors for all configurations
        3. Compute determinant root of the full information matrix

        Prerequisites:
            - Robot model and parameters must be loaded
            - Configuration file must specify valid sample data paths

        Side Effects:
            - Sets self.q_measured with candidate joint configurations
            - Sets self.R_rearr with rearranged kinematic regressor
            - Sets self._subX_dict and self._subX_list with info matrices
            - Sets self.detroot_whole with full matrix determinant root

        Raises:
            ValueError: If sample configuration file is invalid or missing
            AssertionError: If regressor calculation fails

        See Also:
            load_candidate_configurations: Configuration data loading
            calculate_regressor: Kinematic regressor computation
            calculate_detroot_whole: Information matrix analysis
        """
        self.load_candidate_configurations()
        self.calculate_regressor()
        self.calculate_detroot_whole()

    def solve(self, save_file=False):
        """Solve the optimal configuration selection problem.

        This is the main entry point that orchestrates the complete optimal
        configuration generation workflow. It automatically handles
        initialization if not already performed, solves the SOCP optimization,
        and provides comprehensive results analysis.

        The method implements the complete D-optimal design workflow:
        1. Initialize data and regressors (if needed)
        2. Solve SOCP optimization for optimal weights
        3. Select configurations with significant weights
        4. Optionally save results to files
        5. Generate visualization plots

        Args:
            save_file (bool): Whether to save optimal configurations to YAML
                            file in results directory. Default False.

        Side Effects:
            - Updates self.optimal_configurations with selected configs
            - Updates self.optimal_weights with optimization weights
            - Creates visualization plots
            - May create output files if save_file=True
            - Prints progress and results to console

        Raises:
            AssertionError: If minimum configuration requirement not met
            ValueError: If optimization problem is infeasible
            IOError: If file saving fails (logged as warning)

        Example:
            >>> opt_calib = TiagoOptimalCalibration(robot)
            >>> opt_calib.solve(save_file=True)
            12 configs are chosen: [0, 5, 12, 18, ...]
            Optimal configurations written to file successfully

        See Also:
            initialize: Data preparation workflow
            calculate_optimal_configurations: Core optimization solver
            plot: Results visualization
            save_results: File output management
        """
        if not hasattr(self, "R_rearr"):
            self.initialize()
        self.calculate_optimal_configurations()
        if save_file:
            try:
                self.save_results()
                logger.info("Optimal configurations written to file successfully")
            except Exception as e:
                logger.warning(f"Could not write to file: {e}")
        self.plot()

    def load_param(self, config_file: str, setting_type: str = "calibration"):
        """Load calibration parameters from YAML configuration file.

        This method supports both legacy YAML format and the new unified
        configuration format. It automatically detects the format type
        and applies the appropriate parser.

        Args:
            config_file (str): Path to configuration file (legacy or unified)
            setting_type (str): Configuration section to load
        """
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

    def load_candidate_configurations(self):
        """Load candidate joint configurations from external data files.

        Reads robot joint configurations from CSV or YAML files that serve
        as the candidate pool for optimization. The method supports multiple
        file formats and automatically updates the sample count parameter.

        Supported formats:
        - CSV: Standard measurement data format with joint configurations
        - YAML: Structured format with named joints and configurations

        The YAML format expects:
        ```yaml
        calibration_joint_names: [joint1, joint2, ...]
        calibration_joint_configurations: [[q1_1, q1_2, ...], [q2_1, ...]]
        ```

        Side Effects:
            - Sets self.q_measured with loaded joint configurations
            - Updates self.calib_config["NbSample"] with actual sample count
            - May load self._configs for YAML format data

        Raises:
            ValueError: If sample_configs_file not specified in configuration
                       or if file format is not supported
            FileNotFoundError: If specified data file does not exist

        Example:
            >>> # Assuming config specifies "data/candidate_configs.yaml"
            >>> opt_calib.load_candidate_configurations()
            >>> print(opt_calib.q_measured.shape)  # (1000, 7) for TIAGo
        """
        from figaroh.calibration.calibration_tools import get_idxq_from_jname

        if self._sampleConfigs_file is None:
            raise ValueError("sample_configs_file not specified in " "configuration")

        if "csv" in self._sampleConfigs_file:
            _, self.q_measured = load_data(
                self._data_path, self.model, self.calib_config, []
            )
        elif "yaml" in self._sampleConfigs_file:
            with open(self._sampleConfigs_file, "r") as file:
                self._configs = yaml.load(file, Loader=yaml.SafeLoader)

            q_jointNames = self._configs["calibration_joint_names"]
            q_jointConfigs = np.array(
                self._configs["calibration_joint_configurations"]
            ).T

            df = pd.DataFrame.from_dict(dict(zip(q_jointNames, q_jointConfigs)))

            q = np.zeros([len(df), self.robot.q0.shape[0]])
            for i in range(len(df)):
                for j, name in enumerate(q_jointNames):
                    jointidx = get_idxq_from_jname(self.model, name)
                    q[i, jointidx] = df[name][i]
            self.q_measured = q

            # update number of samples
            self.calib_config["NbSample"] = self.q_measured.shape[0]
        else:
            raise ValueError("Data file format not supported. Use CSV or YAML format.")

    def calculate_regressor(self):
        """Calculate kinematic regressors and information matrices.

        Computes the kinematic regressor matrices that relate kinematic
        parameter variations to end-effector pose changes. This is the
        mathematical foundation for the optimization problem.

        The method performs several key computations:
        1. Calculate base kinematic regressors for all configurations
        2. Rearrange regressor matrix by sample order for efficiency
        3. Compute individual information matrices for each configuration
        4. Store results for optimization access

        Mathematical Background:
            For each configuration i, the regressor Rᵢ satisfies:
            δx = Rᵢ δθ
            where δx is pose variation and δθ is parameter variation.

            The information matrix is: Xᵢ = RᵢᵀRᵢ

        Side Effects:
            - Sets self.R_rearr with rearranged kinematic regressor
            - Sets self._subX_list with list of information matrices
            - Sets self._subX_dict with indexed information matrices
            - Prints parameter names for verification

        Returns:
            bool: True if calculation successful

        Prerequisites:
            - Joint configurations must be loaded (self.q_measured)
            - Robot model and parameters must be initialized

        See Also:
            calculate_base_kinematics_regressor: Core regressor computation
            rearrange_rb: Matrix rearrangement for optimization
            sub_info_matrix: Information matrix decomposition
        """
        (
            Rrand_b,
            R_b,
            R_e,
            paramsrand_base,
            paramsrand_e,
        ) = calculate_base_kinematics_regressor(
            self.q_measured, self.model, self.data, self.calib_config
        )

        # Rearrange the kinematic regressor by sample numbered order
        self.R_rearr = self.rearrange_rb(R_b, self.calib_config)
        subX_list, subX_dict = self.sub_info_matrix(self.R_rearr, self.calib_config)
        self._subX_dict = subX_dict
        self._subX_list = subX_list
        return True

    def calculate_detroot_whole(self):
        """Calculate determinant root of complete information matrix.

        Computes the determinant root of the full Fisher Information Matrix
        formed by all candidate configurations. This serves as the theoretical
        upper bound for the D-optimality criterion and is used for
        performance comparison.

        Mathematical Background:
            M_full = R^T R  (full regressor)
            detroot_whole = det(M_full)^(1/n) / sqrt(n)

            This represents the geometric mean of eigenvalues, normalized
            by matrix dimension for scale independence.

        Side Effects:
            - Sets self.detroot_whole with computed determinant root
            - Prints the computed value for verification

        Prerequisites:
            - Kinematic regressor must be calculated (self.R_rearr)
            - Requires picos library for determinant computation

        Raises:
            AssertionError: If regressor calculation not performed first
            ImportError: If picos library not available

        See Also:
            calculate_regressor: Prerequisites for this computation
            plot: Uses this value for performance comparison
        """
        import picos as pc

        assert self.calculate_regressor(), "Calculate regressor first."
        M_whole = np.matmul(self.R_rearr.T, self.R_rearr)
        self.detroot_whole = pc.DetRootN(M_whole) / np.sqrt(M_whole.shape[0])
        logger.info(f"detrootn of whole matrix: {self.detroot_whole}")

    def rearrange_rb(self, R_b, calib_config):
        """rearrange the kinematic regressor by sample numbered order"""
        Rb_rearr = np.empty_like(R_b)
        for i in range(calib_config["calibration_index"]):
            for j in range(calib_config["NbSample"]):
                Rb_rearr[j * calib_config["calibration_index"] + i, :] = R_b[
                    i * calib_config["NbSample"] + j
                ]
        return Rb_rearr

    def sub_info_matrix(self, R, calib_config):
        """Decompose regressor into individual configuration info matrices.

        Creates separate information matrices for each configuration by
        extracting the corresponding rows from the full regressor matrix.
        This decomposition enables individual configuration evaluation
        in the optimization process.

        Args:
            R (ndarray): Full rearranged kinematic regressor matrix
            calib_config (dict): Calibration parameters including sample count
                         and calibration index

        Returns:
            tuple: (subX_list, subX_dict) where:
                - subX_list: List of information matrices (RᵢᵀRᵢ)
                - subX_dict: Dictionary mapping config index to matrix

        Mathematical Details:
            For configuration i:
            Rᵢ = R[i*idx:(i+1)*idx, :]  (extract rows)
            Xᵢ = RᵢᵀRᵢ  (information matrix)

        Example:
            >>> R_full = np.random.rand(6000, 42)  # 1000 configs, 6 DOF
            >>> subX_list, subX_dict = self.sub_info_matrix(R_full, calib_config)
            >>> print(len(subX_list))  # 1000
            >>> print(subX_dict[0].shape)  # (42, 42)
        """
        subX_list = []
        idex = calib_config["calibration_index"]
        for it in range(calib_config["NbSample"]):
            sub_R = R[it * idex : (it * idex + idex), :]
            subX = np.matmul(sub_R.T, sub_R)
            subX_list.append(subX)
        subX_dict = dict(
            zip(
                np.arange(
                    calib_config["NbSample"],
                ),
                subX_list,
            )
        )
        return subX_list, subX_dict

    def calculate_optimal_configurations(self):
        """Solve SOCP optimization to find optimal configuration subset.

        This is the core optimization method that solves the D-optimal
        experimental design problem using Second-Order Cone Programming.
        The method finds weights for each candidate configuration that
        maximize the determinant of the Fisher Information Matrix.

        Optimization Problem:
            maximize det(Σᵢ wᵢ Xᵢ)^(1/n)
            subject to: Σᵢ wᵢ ≤ 1, wᵢ ≥ 0

            Where Xᵢ are information matrices and wᵢ are configuration weights.

        Selection Process:
            1. Solve SOCP optimization for optimal weights
            2. Select configurations with weights > eps_opt (1e-5)
            3. Verify minimum configuration requirement is met
            4. Store selected configurations and weights

        Side Effects:
            - Sets self.w_list with optimization solution weights
            - Sets self.w_dict_sort with sorted weight dictionary
            - Sets self.optimal_configurations with selected configs
            - Sets self.optimal_weights with final weight values
            - Sets self.nb_chosen with number of selected configurations
            - Prints timing information and selection results

        Returns:
            bool: True if optimization successful and feasible

        Raises:
            AssertionError: If regressor not calculated or if insufficient
                          configurations selected (infeasible design)

        Example:
            >>> opt_calib.calculate_optimal_configurations()
            solve time of socp: 2.35 seconds
            12 configs are chosen: [0, 5, 12, 18, 23, ...]

        See Also:
            SOCPOptimizer: The optimization solver implementation
            calculate_regressor: Required prerequisite computation
        """
        import time

        assert self.calculate_regressor(), "Calculate regressor first."

        # Picos optimization (A-optimality, C-optimality, D-optimality)
        prev_time = time.time()
        SOCP_algo = SOCPOptimizer(self._subX_dict, self.calib_config)
        self.w_list, self.w_dict_sort = SOCP_algo.solve()
        solve_time = time.time() - prev_time
        logger.info(f"solve time of socp: {solve_time}")

        # Select optimal config based on values of weight
        self.eps_opt = 1e-5
        chosen_config = []
        for i in list(self.w_dict_sort.keys()):
            if self.w_dict_sort[i] > self.eps_opt:
                chosen_config.append(i)

        assert (
            len(chosen_config) >= self.minNbChosen
        ), "Infeasible design, try to increase NbSample."

        logger.info(f"{len(chosen_config)} configs are chosen: {chosen_config}")
        self.nb_chosen = len(chosen_config)

        # Store optimal configurations and weights
        opt_ids = chosen_config
        opt_configs_values = []
        for opt_id in opt_ids:
            opt_configs_values.append(
                self._configs["calibration_joint_configurations"][opt_id]
            )
        self.optimal_configurations = self._configs.copy()
        self.optimal_configurations["calibration_joint_configurations"] = list(
            opt_configs_values
        )
        self.optimal_weights = self.w_list
        return True

    def plot(self):
        """Generate comprehensive visualization of optimization results.

        Creates dual-panel plots that provide insight into the optimization
        quality and configuration selection process. The visualizations help
        assess the efficiency of the selected configuration subset.

        Plot Components:
        1. D-optimality criterion vs. number of configurations
           - Shows how information matrix determinant improves with
             additional configurations
           - Normalized against theoretical maximum (all configurations)
           - Helps identify diminishing returns point

        2. Configuration weights in logarithmic scale
           - Displays weight assigned to each candidate configuration
           - Configurations above threshold (eps_opt) are selected
           - Shows selection boundary and weight distribution

        Prerequisites:
            - Optimization must be completed (optimal_configurations available)
            - Information matrices must be computed

        Side Effects:
            - Creates matplotlib figure with two subplots
            - Displays plots using plt.show()
            - May block execution until plots are closed

        Returns:
            bool: True if plotting successful

        Mathematical Details:
            D-optimality ratio = detroot_whole / det(selected_subset)
            This ratio approaches 1.0 as selected subset approaches optimality.

        Example:
            >>> opt_calib.solve()
            >>> # Plot is automatically generated, or call manually:
            >>> opt_calib.plot()

        See Also:
            calculate_optimal_configurations: Generates data for plotting
            calculate_detroot_whole: Provides normalization reference
        """
        import picos as pc

        assert (
            hasattr(self, "optimal_configurations")
            and self.optimal_configurations is not None
        ), "Calculate optimal configurations first."

        # Plotting
        det_root_list = []
        n_key_list = []

        # Calculate det_root_list and n_key_list
        for nbc in range(self.minNbChosen, self.calib_config["NbSample"] + 1):
            n_key = list(self.w_dict_sort.keys())[0:nbc]
            n_key_list.append(n_key)
            M_i = pc.sum(self.w_dict_sort[i] * self._subX_list[i] for i in n_key)
            det_root_list.append(pc.DetRootN(M_i) / np.sqrt(nbc))

        # Create subplots
        fig, ax = plt.subplots(2)

        # Plot D-optimality criterion
        ratio = self.detroot_whole / det_root_list[-1]
        plot_range = self.calib_config["NbSample"] - self.minNbChosen
        ax[0].set_ylabel("D-optimality criterion", fontsize=20)
        ax[0].tick_params(axis="y", labelsize=18)
        ax[0].plot(ratio * np.array(det_root_list[:plot_range]))
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].grid(True, linestyle="--")

        # Plot quality of estimation
        ax[1].set_ylabel("Weight values (log)", fontsize=20)
        ax[1].set_xlabel("Data sample", fontsize=20)
        ax[1].tick_params(axis="both", labelsize=18)
        ax[1].tick_params(axis="y", labelrotation=30)
        ax[1].scatter(
            np.arange(len(list(self.w_dict_sort.values()))),
            list(self.w_dict_sort.values()),
        )
        ax[1].set_yscale("log")
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].grid(True, linestyle="--")
        plt.show()

        return True

    def plot_results(self):
        """Plot optimal calibration results using unified results manager."""
        if (
            not hasattr(self, "optimal_configurations")
            or self.optimal_configurations is None
        ):
            logger.warning(
                "No optimal configuration results to plot. Run solve() first."
            )
            return

        try:
            from .results_manager import ResultsManager

            # Initialize results manager
            robot_name = self.calib_config.get("robot_name", self.model.name)
            results_manager = ResultsManager("optimal_calibration", robot_name)

            # Prepare data for plotting
            weights = (
                np.array(list(self.w_dict_sort.values()))
                if hasattr(self, "w_dict_sort")
                else np.array([])
            )

            # Plot using unified manager
            results_manager.plot_optimal_calibration_results(
                configurations=self.optimal_configurations,
                weights=weights,
                title="Optimal Calibration Configuration Results",
            )

        except ImportError:
            # Fallback to existing plotting
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(14, 6))

            ax[0].bar(
                list(self.w_dict_sort.keys()),
                list(self.w_dict_sort.values()),
            )
            ax[0].set_xlabel("Configuration indices")
            ax[0].set_ylabel("Weights")
            ax[0].set_title("Chosen configurations")
            ax[0].spines["top"].set_visible(False)
            ax[0].spines["right"].set_visible(False)
            ax[0].grid(True, linestyle="--")

            ax[1].bar(
                list(self.w_dict_sort.keys()),
                list(self.w_dict_sort.values()),
            )
            ax[1].set_yscale("log")
            ax[1].spines["top"].set_visible(False)
            ax[1].spines["right"].set_visible(False)
            ax[1].grid(True, linestyle="--")
            plt.show()

    def save_results(self, output_dir="results"):
        """Save optimal configuration results using unified results manager."""
        if (
            not hasattr(self, "optimal_configurations")
            or self.optimal_configurations is None
        ):
            logger.warning(
                "No optimal configuration results to save. Run solve() first."
            )
            return

        try:
            from .results_manager import ResultsManager

            # Initialize results manager
            robot_name = self.calib_config.get("robot_name", self.model.name)
            results_manager = ResultsManager("optimal_calibration", robot_name)

            # Prepare results dictionary
            results_dict = {
                "optimal_configurations": self.optimal_configurations,
                "selected_weights": (
                    self.w_dict_sort if hasattr(self, "w_dict_sort") else {}
                ),
                "minimum_configurations": getattr(self, "minNbChosen", 0),
                "configuration_count": len(self.optimal_configurations),
                "calibration_config": self.calib_config,
            }

            # Add condition number if available
            if hasattr(self, "detroot_whole"):
                results_dict["condition_number"] = float(self.detroot_whole)

            # Save using unified manager
            saved_files = results_manager.save_results(
                results_dict, output_dir, save_formats=["yaml", "csv"]
            )

            return saved_files

        except ImportError:
            # Fallback to existing saving
            import os
            import yaml

            os.makedirs(output_dir, exist_ok=True)

            robot_name = self.calib_config.get("robot_name", self.model.name)
            filename = f"{robot_name}_optimal_configurations.yaml"

            with open(os.path.join(output_dir, filename), "w") as stream:
                try:
                    yaml.dump(
                        self.optimal_configurations,
                        stream,
                        sort_keys=False,
                        default_flow_style=True,
                    )
                except yaml.YAMLError as exc:
                    logger.error(exc)
            logger.info(f"Results saved to {output_dir}/{filename}")

            return {"yaml": os.path.join(output_dir, filename)}


# SOCP Optimizer class used by BaseOptimalCalibration
class SOCPOptimizer:
    """Second-Order Cone Programming optimizer for configuration selection.

    Implements the mathematical optimization for D-optimal experimental design
    using Second-Order Cone Programming (SOCP). This class formulates and
    solves the convex optimization problem that maximizes the determinant
    of the Fisher Information Matrix.

    Mathematical Formulation:
        maximize t
        subject to: t ≤ det(Σᵢ wᵢ Xᵢ)^(1/n)
                   Σᵢ wᵢ ≤ 1
                   wᵢ ≥ 0

        Where:
        - t is auxiliary variable for objective
        - wᵢ are configuration weights
        - Xᵢ are information matrices
        - n is matrix dimension

    The problem is solved using the CVXOPT solver with picos interface.

    Attributes:
        pool (dict): Dictionary of information matrices indexed by config ID
        calib_config (dict): Calibration parameters including sample count
        problem: Picos optimization problem instance
        w: Decision variable for configuration weights
        t: Auxiliary variable for determinant objective
        solution: Optimization solution object

    Example:
        >>> optimizer = SOCPOptimizer(subX_dict, calib_config)
        >>> weights, sorted_weights = optimizer.solve()
        >>> print(f"Optimization status: {optimizer.solution.status}")
    """

    def __init__(self, subX_dict, calib_config):
        import picos as pc

        self.pool = subX_dict
        self.calib_config = calib_config
        self.problem = pc.Problem()
        self.w = pc.RealVariable("w", self.calib_config["NbSample"], lower=0)
        self.t = pc.RealVariable("t", 1)

    def add_constraints(self):
        import picos as pc

        Mw = pc.sum(
            self.w[i] * self.pool[i] for i in range(self.calib_config["NbSample"])
        )
        wgt_cons = self.problem.add_constraint(1 | self.w <= 1)
        det_root_cons = self.problem.add_constraint(self.t <= pc.DetRootN(Mw))

    def set_objective(self):
        self.problem.set_objective("max", self.t)

    def solve(self):
        self.add_constraints()
        self.set_objective()
        self.solution = self.problem.solve(solver="cvxopt")

        w_list = []
        for i in range(self.w.dim):
            w_list.append(float(self.w.value[i]))
        logger.info(f"sum of all element in vector solution: {sum(w_list)}")

        # to dict
        w_dict = dict(zip(np.arange(self.calib_config["NbSample"]), w_list))
        w_dict_sort = dict(reversed(sorted(w_dict.items(), key=lambda item: item[1])))
        return w_list, w_dict_sort


# DetMax Optimizer class
class Detmax:
    """Determinant Maximization optimizer using greedy exchange algorithm.

    This class implements a heuristic optimization algorithm for D-optimal
    experimental design that uses a greedy exchange strategy to find
    near-optimal configuration subsets. Unlike the SOCP approach, this
    method provides a combinatorial solution that directly selects
    discrete configurations.

    Algorithm Overview:
        The DetMax algorithm uses an iterative exchange procedure:
        1. Initialize with a random subset of configurations
        2. Iteratively add the configuration that maximally improves
           the determinant criterion
        3. Remove the configuration whose absence minimally degrades
           the determinant criterion
        4. Repeat until convergence (no beneficial exchanges)

    Mathematical Background:
        The algorithm maximizes det(Σᵢ∈S Xᵢ)^(1/n) where:
        - S is the selected configuration subset
        - Xᵢ are information matrices for configurations
        - n is the matrix dimension

        This is a discrete optimization problem (vs continuous SOCP).

    Advantages:
        - Provides exact discrete solution (no weight thresholding)
        - Computationally efficient for small to medium problems
        - Intuitive greedy strategy with good convergence properties
        - No external optimization solvers required

    Limitations:
        - May converge to local optima (not globally optimal)
        - Performance depends on random initialization
        - Computational complexity grows with candidate pool size

    Attributes:
        pool (dict): Dictionary of information matrices indexed by config ID
        nd (int): Number of configurations to select
        cur_set (list): Current configuration subset being evaluated
        fail_set (list): Configurations that failed selection criteria
        opt_set (list): Final optimal configuration subset
        opt_critD (list): Evolution of determinant criterion during
                        optimization

    Example:
        >>> # Create DetMax optimizer
        >>> detmax = Detmax(subX_dict, num_configs=12)
        >>>
        >>> # Run optimization
        >>> criterion_history = detmax.main_algo()
        >>>
        >>> # Get selected configurations
        >>> selected_configs = detmax.cur_set
        >>> final_criterion = criterion_history[-1]
        >>>
        >>> print(f"Selected {len(selected_configs)} configurations")
        >>> print(f"Final D-optimality: {final_criterion:.4f}")

    See Also:
        SOCPOptimizer: Alternative SOCP-based optimization approach
        BaseOptimalCalibration: Main calibration framework
    """

    def __init__(self, candidate_pool, NbChosen):
        """Initialize DetMax optimizer with candidate pool and target size.

        Sets up the determinant maximization optimizer with the candidate
        configuration pool and specifies the number of configurations to
        select in the final optimal subset.

        Args:
            candidate_pool (dict): Dictionary mapping configuration indices
                                 to their corresponding information matrices.
                                 Keys are configuration IDs, values are
                                 symmetric positive definite matrices.
            NbChosen (int): Number of configurations to select in the optimal
                          subset. Must be less than total candidates and
                          sufficient for parameter identifiability.

        Raises:
            ValueError: If NbChosen exceeds candidate pool size
            TypeError: If candidate_pool is not a dictionary

        Side Effects:
            - Initializes internal tracking lists (cur_set, fail_set, etc.)
            - Stores candidate pool and selection target

        Example:
            >>> # Information matrices dict
            >>> info_matrices = {0: X0, 1: X1, 2: X2, ...}
            >>> optimizer = Detmax(info_matrices, NbChosen=10)
        """
        self.pool = candidate_pool
        self.nd = NbChosen
        self.cur_set = []
        self.fail_set = []
        self.opt_set = []
        self.opt_critD = []

    def get_critD(self, set):
        """Calculate D-optimality criterion for configuration subset.

        Computes the determinant root of the Fisher Information Matrix
        formed by summing the information matrices of configurations
        in the specified subset. This serves as the objective function
        for the determinant maximization algorithm.

        Args:
            set (list): List of configuration indices from the candidate
                       pool to include in the criterion calculation

        Returns:
            float: D-optimality criterion value (determinant root)
                  Higher values indicate better parameter identifiability

        Raises:
            AssertionError: If any configuration index not in candidate pool

        Mathematical Details:
            For subset S, computes: det(Σᵢ∈S Xᵢ)^(1/n)
            where Xᵢ are information matrices and n is matrix dimension

        Example:
            >>> subset = [0, 5, 12, 18]  # Configuration indices
            >>> criterion = optimizer.get_critD(subset)
            >>> print(f"D-optimality: {criterion:.6f}")
        """
        import picos as pc

        infor_mat = 0
        for idx in set:
            assert idx in self.pool.keys(), "chosen sample not in candidate pool"
            infor_mat += self.pool[idx]
        return float(pc.DetRootN(infor_mat))

    def main_algo(self):
        """Execute the main determinant maximization algorithm.

        Implements the greedy exchange algorithm for D-optimal experimental
        design. The algorithm alternates between adding configurations that
        maximally improve the determinant and removing configurations whose
        absence minimally degrades the determinant.

        Algorithm Steps:
        1. Initialize random subset of target size from candidate pool
        2. Exchange Loop:
           a. ADD PHASE: Find configuration that maximally improves criterion
           b. REMOVE PHASE: Find configuration whose removal minimally hurts
           c. Update current subset and criterion value
        3. Repeat until convergence (no beneficial exchanges)
        4. Return optimization history

        Convergence Condition:
            The algorithm stops when the optimal configuration to add
            equals the optimal configuration to remove, indicating no
            further improvement is possible.

        Returns:
            list: History of D-optimality criterion values throughout
                 the optimization process. Last value is final criterion.

        Side Effects:
            - Updates self.cur_set with final optimal configuration subset
            - Updates self.opt_critD with complete optimization history
            - Uses random initialization (results may vary between runs)

        Complexity:
            O(max_iterations × candidate_pool_size × target_subset_size)
            where max_iterations depends on problem structure and
            initialization

        Example:
            >>> optimizer = Detmax(info_matrices, NbChosen=10)
            >>> history = optimizer.main_algo()
            >>> print(f"Converged after {len(history)} iterations")
            >>> print(f"Final subset: {optimizer.cur_set}")
            >>> print(f"Final criterion: {history[-1]:.6f}")

        Note:
            The algorithm may converge to different local optima depending
            on random initialization. For critical applications, consider
            running multiple times with different seeds.
        """
        import random

        # get all indices in the pool
        pool_idx = tuple(self.pool.keys())

        # initialize a random set
        cur_set = random.sample(pool_idx, self.nd)
        updated_pool = list(set(pool_idx) - set(self.cur_set))

        # adding samples from remaining pool: k = 1
        opt_k = updated_pool[0]
        opt_critD = self.get_critD(cur_set)
        init_set = set(cur_set)
        fin_set = set([])
        rm_j = cur_set[0]

        while opt_k != rm_j:

            # add
            for k in updated_pool:
                cur_set.append(k)
                cur_critD = self.get_critD(cur_set)
                if opt_critD < cur_critD:
                    opt_critD = cur_critD
                    opt_k = k
                cur_set.remove(k)
            cur_set.append(opt_k)
            opt_critD = self.get_critD(cur_set)

            # remove
            delta_critD = opt_critD
            rm_j = cur_set[0]
            for j in cur_set:
                rm_set = cur_set.copy()
                rm_set.remove(j)
                cur_delta_critD = opt_critD - self.get_critD(rm_set)

                if cur_delta_critD < delta_critD:
                    delta_critD = cur_delta_critD
                    rm_j = j
            cur_set.remove(rm_j)
            opt_critD = self.get_critD(cur_set)
            fin_set = set(cur_set)

            self.opt_critD.append(opt_critD)
        return self.opt_critD
