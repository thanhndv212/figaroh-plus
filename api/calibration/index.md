# Calibration

The calibration module provides tools for geometric calibration of robots, including the [reporting & verification suite](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md) (`print_quality_report()`, `export_html_report()`, `verify()`, `export_verification_report()`) attached directly to `BaseCalibration`.

## base_calibration

Base calibration class for FIGAROH examples.

This module provides the BaseCalibration abstract class extracted from the FIGAROH library for use in the examples. It implements a comprehensive framework for robot kinematic calibration.

## `BaseCalibration(robot, config_file, del_list=None)`

Bases: `ABC`

Abstract base class for robot kinematic calibration.

This class provides a comprehensive framework for calibrating robot kinematic parameters using measurement data. It implements the Template Method pattern, providing common functionality while allowing robot-specific implementations of the cost function.

The calibration process follows these main steps:

1. Parameter initialization from configuration files
1. Data loading and validation
1. Parameter identification using base regressor analysis
1. Robust optimization with outlier detection and removal
1. Solution evaluation and validation
1. Results visualization and export

Key Features:

- Automatic parameter identification using QR decomposition
- Robust optimization with iterative outlier removal
- Unit-aware measurement weighting for position/orientation data
- Comprehensive solution evaluation and quality metrics
- Extensible framework for different robot types

Attributes:

| Name                 | Type      | Description                                                   |
| -------------------- | --------- | ------------------------------------------------------------- |
| `STATUS`             | `str`     | Current calibration status ("NOT CALIBRATED" or "CALIBRATED") |
| `LM_result`          | `str`     | Optimization result from scipy.optimize.least_squares         |
| `var_`               | `ndarray` | Calibrated parameter values                                   |
| `evaluation_metrics` | `dict`    | Solution quality metrics                                      |
| `std_dev`            | `list`    | Standard deviations of calibrated parameters                  |
| `std_pctg`           | `list`    | Standard deviation percentages                                |
| `PEE_measured`       | `ndarray` | Measured end-effector poses/positions                         |
| `q_measured`         | `ndarray` | Measured joint configurations                                 |
| `calib_config`       | `dict`    | Calibration parameters and configuration                      |
| `model`              |           | Robot kinematic model (Pinocchio)                             |
| `data`               |           | Robot data structure (Pinocchio)                              |

Example

> > > ### Create robot-specific calibration
> > >
> > > class MyRobotCalibration(BaseCalibration): ... def cost_function(self, var): ... PEEe = calc_updated_fkm(self.model, self.data, var, ... self.q_measured, self.calib_config) ... # Use body-frame position (or position_frame="world" for world frame) ... residuals = self.\_compute_logmap_residuals( ... self.PEE_measured, PEEe, position_frame="body") ... return self.apply_measurement_weighting(residuals) ...
> > >
> > > ### Run calibration
> > >
> > > calibrator = MyRobotCalibration(robot, "config.yaml") calibrator.initialize() calibrator.solve() print(f"RMSE: {calibrator.evaluation_metrics\['rmse'\]:.6f}")

Notes

- Derived classes should implement robot-specific cost_function()
- Default cost_function is provided but issues performance warning
- Configuration files must follow FIGAROH parameter structure
- Supports both "full_params" and "joint_offset" calibration models

See Also

- TiagoCalibration: TIAGo robot implementation
- UR10Calibration: Universal Robots UR10 implementation
- calc_updated_fkm: Forward kinematics computation function
- apply_measurement_weighting: Unit-aware weighting utility

Initialize robot calibration framework.

Sets up the calibration environment by loading robot model, configuration parameters, and preparing internal data structures for optimization.

Parameters:

| Name          | Type   | Description                                                                                                                             | Default    |
| ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `robot`       |        | Robot object containing kinematic model and data structures. Must have 'model' and 'data' attributes compatible with Pinocchio library. | *required* |
| `config_file` | `str`  | Path to YAML configuration file containing calibration parameters, data paths, and settings.                                            | *required* |
| `del_list`    | `list` | Indices of bad/outlier samples to exclude from calibration data. Defaults to [].                                                        | `None`     |

Raises:

| Type                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `FileNotFoundError` | If config_file does not exist                     |
| `KeyError`          | If required parameters missing from configuration |
| `ValueError`        | If configuration parameters are invalid           |
| `CalibrationError`  | If robot or configuration is invalid              |

Side Effects

- Loads and validates configuration parameters
- Sets initial calibration status to "NOT CALIBRATED"
- Calculates number of calibration variables
- Resolves absolute path to measurement data file

Example

> > > robot = load_robot_model("tiago.urdf") calibrator = TiagoCalibration(robot, "tiago_config.yaml", ... del_list=[5, 12, 18])

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `initialize()`

Initialize calibration data and parameters.

Performs the initialization phase of calibration by:

1. Loading measurement data from files
1. Creating parameter list through base regressor analysis
1. Identifying calibratable parameters using QR decomposition

This method must be called before solve() to prepare the calibration problem. It handles data validation, parameter identification, and sets up the optimization problem structure.

Raises:

| Type                | Description                               |
| ------------------- | ----------------------------------------- |
| `FileNotFoundError` | If measurement data file not found        |
| `ValueError`        | If data format is invalid or incompatible |
| `AssertionError`    | If required data dimensions don't match   |
| `CalibrationError`  | If initialization fails                   |

Side Effects

- Populates self.PEE_measured with measurement data
- Populates self.q_measured with joint configuration data
- Updates self.calib_config["param_name"] with identified parameters
- Validates data consistency and dimensions

Example

> > > calibrator = TiagoCalibration(robot, "config.yaml") calibrator.initialize() print(f"Loaded {calibrator.calib_config['NbSample']} samples") print(f"Calibrating {len(calibrator.calib_config['param_name'])} " ... f"parameters")

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `solve(method='lm', max_iterations=3, outlier_threshold=None, enable_logging=True, plotting=False, save_results=False, html_report=False)`

Execute the complete calibration process.

This is the main entry point for calibration that:

1. Runs the optimization algorithm via solve_optimisation()
1. Optionally generates visualization plots if enabled
1. Optionally saves results to files if enabled

The method serves as a high-level orchestrator for the calibration workflow, delegating the actual optimization to solve_optimisation() and handling visualization based on user preferences.

Parameters:

| Name                | Type | Description                                                                                                                       | Default |
| ------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `max_iterations`    |      | Maximum number of fits; between fits, samples whose position error exceeds outlier_threshold are excluded and the rest refitted.  | `3`     |
| `outlier_threshold` |      | Position error [m] above which a sample is excluded; None (default) uses the config's parameters.outlier_threshold (outlier_eps). | `None`  |
| `html_report`       |      | If True, also export an HTML diagnostic report (see :meth:export_html_report) after the terminal quality report is printed.       | `False` |

Side Effects

- Updates calibration parameters through optimization
- Sets self.STATUS to "CALIBRATED" when the solver converged and to "FAILED" when it did not (the fit stage is then `failed` and the export paths refuse, see :meth:`require_exportable`)
- May display plots if self.calib_config["PLOT"] is True

See Also

solve_optimisation: Core optimization implementation plot: Visualization and analysis plotting export_html_report: Visual counterpart of the terminal report

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
        - Sets self.STATUS to "CALIBRATED" when the solver converged
          and to "FAILED" when it did not (the fit stage is then
          ``failed`` and the export paths refuse, see
          :meth:`require_exportable`)
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
```

### `plot_results()`

Generate comprehensive visualization plots for calibration results.

Creates multiple visualization plots to analyze calibration quality:

1. Error distribution plots showing residual patterns
1. 3D pose visualizations comparing measured vs predicted poses
1. Joint configuration analysis (currently commented)

This method provides essential visual feedback for calibration assessment, helping users understand solution quality and identify potential issues with the calibration process.

Prerequisites

- Calibration must be completed (solve() called)
- Measurement data must be loaded
- Matplotlib backend must be configured

Side Effects

- Displays plots using plt.show()
- May block execution until plots are closed

See Also

plot_errors_distribution: Individual error analysis plots plot_3d_poses: 3D pose comparison visualization

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `load_param(config_file, setting_type='calibration')`

Load calibration parameters from YAML configuration file.

Only the unified configuration format is supported; a legacy flat config raises an error naming `figaroh.utils.config_migration`.

Parameters:

| Name           | Type  | Description                          | Default         |
| -------------- | ----- | ------------------------------------ | --------------- |
| `config_file`  | `str` | Path to a unified configuration file | *required*      |
| `setting_type` | `str` | Configuration section to load        | `'calibration'` |

Source code in `src/figaroh/calibration/base_calibration.py`

```
def load_param(self, config_file: str, setting_type: str = "calibration"):
    """Load calibration parameters from YAML configuration file.

    Only the unified configuration format is supported; a legacy flat
    config raises an error naming ``figaroh.utils.config_migration``.

    Args:
        config_file (str): Path to a unified configuration file
        setting_type (str): Configuration section to load
    """
    self._config_file_path = config_file
    try:
        logger.info(f"Loading config from {config_file}")

        # The legacy flat format was removed in 1.0
        require_unified_config(config_file)
        parser = UnifiedConfigParser(config_file)
        unified_config = parser.parse()
        unified_calib_config = create_task_config(
            self.robot, unified_config, setting_type
        )
        # Convert unified format to the calib_config dict
        self.calib_config = unified_to_legacy_config(
            self.robot, unified_calib_config
        )

    except FileNotFoundError:
        raise CalibrationError(f"Configuration file not found: {config_file}")
    except Exception as e:
        raise CalibrationError(f"Failed to load configuration: {e}")
```

### `create_param_list(q=None)`

Initialize calibration parameter structure and validate setup.

This method sets up the fundamental parameter structure for calibration by computing kinematic regressors and ensuring proper frame naming conventions. It serves as a critical initialization step that must be called before optimization begins.

The method performs several key operations:

1. Computes base kinematic regressors for parameter identification
1. Adds default names for unknown base and tip frames
1. Validates the parameter structure for calibration readiness

Parameters:

| Name | Type         | Description                                                                                                | Default |
| ---- | ------------ | ---------------------------------------------------------------------------------------------------------- | ------- |
| `q`  | `array_like` | Joint configuration for regressor computation. If None, uses empty list which may limit regressor accuracy | `None`  |

Returns:

| Name   | Type | Description                                           |
| ------ | ---- | ----------------------------------------------------- |
| `bool` |      | Always returns True to indicate successful completion |

Side Effects

- Updates self.calib_config with frame names if not known
- Computes and caches kinematic regressors
- May modify parameter structure for calibration compatibility

Raises:

| Type               | Description                                    |
| ------------------ | ---------------------------------------------- |
| `ValueError`       | If robot model is not properly initialized     |
| `AttributeError`   | If required calibration parameters are missing |
| `CalibrationError` | If parameter creation fails                    |

Example

> > > calibrator = BaseCalibration(robot) calibrator.load_param("config.yaml") calibrator.create_param_list() # Basic setup
> > >
> > > #### Or with specific joint configuration
> > >
> > > q_nominal = np.zeros(robot.nq) calibrator.create_param_list(q_nominal)

See Also

calculate_base_kinematics_regressor: Core regressor computation add_base_name: Base frame naming utilities add_pee_name: End-effector frame naming utilities

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `initial_frame_guess()`

Closed-form guess for the unknown base and tip frames.

See :func:`estimate_frames_closed_form`. Empty when the frames are known or the measurement does not determine them (partial position measurability, several markers, camera anchor).

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `eliminate_absorbed_parameters(tol=0.0001)`

Drop joint parameters that the base/tip frames absorb on this data.

The structural selection in :func:`calculate_base_kinematics_regressor` uses the joint regressor alone. When the base and tip frames are also estimated, some retained joint parameters are combinations of frame parameters (e.g. a vertical prismatic or revolute first joint against the base's z translation and yaw), so the problem is rank deficient. This builds the full measurement Jacobian at the measured configurations, with the configured measurability, evaluated at the closed-form frame guess, and drops every joint parameter whose unit-normalised column is a combination of the frame columns and the joint columns kept before it (:func:`select_identifiable_parameters`). Frame parameters are always kept.

Set `calib_config["eliminate_absorbed_parameters"] = False` to skip.

Returns:

| Name   | Type        | Description                                     |
| ------ | ----------- | ----------------------------------------------- |
| `list` | `List[str]` | Names of the dropped parameters (also stored in |
|        | `List[str]` | calib_config["absorbed_param_name"]).           |

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `load_data_set()`

Load experimental measurement data for calibration.

Reads measurement data from the specified data path and processes it for calibration use. This includes both pose measurements and corresponding joint configurations, with optional data filtering based on the deletion list.

The method handles data preprocessing, validation, and formatting to ensure compatibility with the calibration algorithms. It serves as the primary data ingestion point for the calibration process.

Side Effects

- Sets self.PEE_measured with processed pose measurements
- Sets self.q_measured with corresponding joint configurations
- Applies data filtering if self.del_list\_ is specified

Prerequisites

- self.\_data_path must be set to valid measurement data location
- Robot model must be initialized
- Calibration parameters must be loaded

Raises:

| Type                | Description                                 |
| ------------------- | ------------------------------------------- |
| `FileNotFoundError` | If data files are not found at \_data_path  |
| `ValueError`        | If data format is incompatible or corrupted |
| `AttributeError`    | If required attributes are not initialized  |
| `CalibrationError`  | If data loading fails                       |

See Also

load_data: Core data loading and processing function

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `get_pose_from_measure(res_)`

Calculate forward kinematics with calibrated parameters.

Computes robot end-effector poses using the updated kinematic model with calibrated parameters. This method applies the calibration results to predict poses for the measured joint configurations.

Parameters:

| Name   | Type      | Description                                                                                                               | Default    |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `res_` | `ndarray` | Calibrated parameter vector containing kinematic corrections (geometric parameters, base transform, tool transform, etc.) | *required* |

Returns:

| Name      | Type      | Description                                                                                                                                                  |
| --------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ndarray` | `ndarray` | Predicted end-effector poses corresponding to the measured joint configurations. Shape depends on the number of measurements and pose representation format. |

Prerequisites

- Joint configurations must be loaded (q_measured available)
- Calibration parameters must be initialized
- Robot model must be properly configured

Example

> > > #### After calibration
> > >
> > > calibrated_params = calibrator.LM_result.x predicted_poses = calibrator.get_pose_from_measure( ... calibrated_params)
> > >
> > > #### Compare with measured poses
> > >
> > > errors = predicted_poses - calibrator.PEE_measured

See Also

calc_updated_fkm: Core forward kinematics computation function

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `cost_function(var)`

Calculate cost function for optimization.

This method provides a default implementation but should be overridden by derived classes to define robot-specific cost computation with appropriate weighting. Regularise with priors (`estimation.method: map`, #120), not rows appended here.

Parameters:

| Name  | Type      | Description                  | Default    |
| ----- | --------- | ---------------------------- | ---------- |
| `var` | `ndarray` | Parameter vector to evaluate | *required* |

Returns:

| Name      | Type      | Description     |
| --------- | --------- | --------------- |
| `ndarray` | `ndarray` | Residual vector |

Warning

Using default cost function. Consider implementing robot-specific cost function for optimal performance.

Example implementations:

```
Body-frame position (default, geometrically correct):
    >>> raw_residuals = self._compute_logmap_residuals(
    ...     self.PEE_measured, PEEe, position_frame="body")

World-frame position (more interpretable):
    >>> raw_residuals = self._compute_logmap_residuals(
    ...     self.PEE_measured, PEEe, position_frame="world")

Then apply weighting:
    >>> weighted_residuals = self.apply_measurement_weighting(
    ...     raw_residuals, pos_weight=1000.0, orient_weight=100.0)
```

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `apply_measurement_weighting(residuals, pos_weight=None, orient_weight=None)`

Apply measurement weighting to handle position/orientation units.

This utility method can be used by derived classes to properly weight position (meter) and orientation (radian) measurements for equivalent influence in the cost function.

Parameters:

| Name            | Type      | Description                                                       | Default    |
| --------------- | --------- | ----------------------------------------------------------------- | ---------- |
| `residuals`     | `ndarray` | Raw residual vector                                               | *required* |
| `pos_weight`    | `float`   | Weight for position residuals. If None, uses 1/position_std       | `None`     |
| `orient_weight` | `float`   | Weight for orientation residuals. If None, uses 1/orientation_std | `None`     |

Returns:

| Name      | Type      | Description              |
| --------- | --------- | ------------------------ |
| `ndarray` | `ndarray` | Weighted residual vector |

Example

> > > #### In derived class cost_function:
> > >
> > > raw_residuals = self.\_compute_logmap_residuals( ... self.PEE_measured, PEEe, ... position_frame="body") # or "world" for world-frame position weighted_residuals = self.apply_measurement_weighting( ... raw_residuals, pos_weight=1000.0, orient_weight=100.0)

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `solve_optimisation(var_init=None, method='lm', max_iterations=3, outlier_threshold=None, enable_logging=False)`

Solve calibration optimization with robust outlier handling.

This method implements a comprehensive optimization strategy:

1. Sets up logging for progress tracking
1. Excludes samples above the outlier threshold and refits
1. Evaluates solution quality with detailed metrics
1. Stores results for further analysis

Parameters:

| Name                | Type      | Description                                                                                | Default |
| ------------------- | --------- | ------------------------------------------------------------------------------------------ | ------- |
| `var_init`          | `ndarray` | Initial parameter guess. If None, uses zero initialization.                                | `None`  |
| `max_iterations`    | `int`     | Maximum number of fits                                                                     | `3`     |
| `outlier_threshold` | `float`   | Position error [m] above which a sample is excluded; None uses calib_config["outlier_eps"] | `None`  |
| `enable_logging`    | `bool`    | Whether to enable terminal logging                                                         | `False` |

Raises:

| Type               | Description                      |
| ------------------ | -------------------------------- |
| `ValueError`       | If optimization fails completely |
| `AssertionError`   | If required data is not loaded   |
| `CalibrationError` | If optimization fails            |

Side Effects

- Updates self.LM_result with optimization results
- Updates self.STATUS to "CALIBRATED" on success
- Creates self.evaluation_metrics with quality metrics
- Sets up logging if enabled

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `calc_stddev(result)`

Calculate parameter uncertainty statistics from optimization results.

Computes standard deviation and percentage uncertainty for each calibrated parameter using the covariance matrix derived from the Jacobian at the optimal solution. This provides confidence intervals and parameter reliability metrics.

The calculation uses the linearized uncertainty propagation: σ²(θ) = σ²(residuals) * (J^T J)^-1

Where J is the Jacobian matrix and σ²(residuals) is the residual variance estimate.

Prerequisites

- Calibration optimization must be completed
- Jacobian matrix must be available from optimization

Side Effects

- Sets self.std_dev with parameter standard deviations
- Sets self.std_pctg with percentage uncertainties

Raises:

| Type               | Description                                       |
| ------------------ | ------------------------------------------------- |
| `CalibrationError` | If calibration has not been performed             |
| `LinAlgError`      | If Jacobian matrix is singular or ill-conditioned |

Example

> > > calibrator.solve() calibrator.calc_stddev() print(f"Parameter uncertainties: {calibrator.std_dev}") print(f"Percentage errors: {calibrator.std_pctg}")

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `redistribute_parameters()`

Joint corrections with standard deviations, for export.

- `structural` method: the fitted base parameters are lifted onto every joint parameter by a weighted minimum-norm lift (:func:`figaroh.calibration.estimation.lift_structural`, figaroh-plus#111): among all joint corrections that reproduce the fit, the most plausible under the expected error sizes (`calib_config["estimation"]["priors"]`, defaults in `estimation.DEFAULT_PRIORS`). Rows the base frame carries and rows the fit dropped are held at 0, so the lifted model predicts what the fit predicts, and the result does not depend on which representative the QR chose. `std_dev` is conditional on that choice of lift: it propagates the fitted uncertainty and is 0 in directions the data does not see.
- other methods (figaroh-plus#113): the fitted joint parameters directly; candidates left out of the fit are 0.

Tool-point and base-frame parameters are not included.

Returns:

| Name   | Type   | Description                                          |
| ------ | ------ | ---------------------------------------------------- |
| `dict` | `dict` | {name: {"value": float, "std_dev": float}} for every |
|        | `dict` | joint parameter of the calibration level.            |

Raises:

| Type               | Description                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationError` | If solve() hasn't run yet (no _C_param/ self.var_), or create_param_list() didn't populate the base-mapping keys in calib_config. |

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `require_exportable(allow_failed_fit=False)`

Refuse to export a calibration whose `fit` stage `failed`.

The gateway of every calibration export (:meth:`joint_corrections`, hence the URDF export, and `export_geometric_calibration_yaml`). A failed fit records stage `export` as failed and raises :class:`CalibrationError`. `allow_failed_fit=True` is the explicit override for a diagnostic export: it logs a warning, records stage `export` as `fallback` and notes the override in the run provenance. A successful fit is unaffected.

Source code in `src/figaroh/calibration/base_calibration.py`

```
def require_exportable(self, allow_failed_fit: bool = False) -> None:
    """Refuse to export a calibration whose ``fit`` stage ``failed``.

    The gateway of every calibration export (:meth:`joint_corrections`,
    hence the URDF export, and ``export_geometric_calibration_yaml``).
    A failed fit records stage ``export`` as failed and raises
    :class:`CalibrationError`. ``allow_failed_fit=True`` is the explicit
    override for a diagnostic export: it logs a warning, records stage
    ``export`` as ``fallback`` and notes the override in the run
    provenance. A successful fit is unaffected.
    """
    from figaroh.tools.stages import record_stage

    fit = next(
        (s for s in getattr(self, "stages", None) or [] if s.stage == "fit"), None
    )
    if fit is None or fit.status != "failed":
        return
    reason = f"the fit failed ({fit.reason or 'solver did not converge'})"
    if allow_failed_fit:
        logger.warning(
            "Exporting from a failed fit (allow_failed_fit=True): %s", reason
        )
        record_stage(
            self,
            "export",
            "fallback",
            f"exported from a failed fit, diagnostic only: {reason}",
        )
        prov = getattr(self, "_run_provenance", None)
        if isinstance(prov, dict):
            prov["export_override"] = {
                "allow_failed_fit": True,
                "fit_reason": fit.reason,
            }
        return
    record_stage(self, "export", "failed", reason)
    raise CalibrationError(
        f"export failed: {reason}; nothing to export. Pass "
        f"allow_failed_fit=True for a diagnostic export only."
    )
```

### `joint_corrections(lift=True, drop_unsupported=False, allow_failed_fit=False)`

Joint parameter values to write into a URDF (`export_urdf`).

`lift=True`: :meth:`redistribute_parameters` (the weighted lift for `structural`; the fitted values otherwise), so the URDF and the PAL export carry the same corrections. `lift=False`: the fitted joint parameters as estimated (for `structural`, one representative per dependent group and the rest at 0). Either way the reloaded URDF, with :meth:`metrology_frames` applied outside it, reproduces the calibrated forward kinematics (figaroh-plus#62).

Frame parameters are never included; see :meth:`metrology_frames`.

Parameters:

| Name               | Type   | Description                                                                                                                                                                       | Default |
| ------------------ | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `lift`             | `bool` | see above.                                                                                                                                                                        | `True`  |
| `drop_unsupported` | `bool` | leave out fitted parameters a URDF cannot carry (elastic k\_\*, contact planes, ...). By default they raise, because the reloaded model would then not reproduce the calibration. | `False` |
| `allow_failed_fit` | `bool` | export although the solver failed, for diagnosis only; see :meth:require_exportable.                                                                                              | `False` |

Raises:

| Type               | Description                                                                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationError` | if the fit stage failed (stage export is recorded as failed), or if the fit has parameters that are neither frames nor kinematic joint corrections, and drop_unsupported is False. |

Source code in `src/figaroh/calibration/base_calibration.py`

```
def joint_corrections(
    self,
    lift: bool = True,
    drop_unsupported: bool = False,
    allow_failed_fit: bool = False,
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
        allow_failed_fit: export although the solver failed, for
            diagnosis only; see :meth:`require_exportable`.

    Raises:
        CalibrationError: if the ``fit`` stage ``failed`` (stage
            ``export`` is recorded as failed), or if the fit has parameters that are neither
            frames nor kinematic joint corrections, and
            ``drop_unsupported`` is False.
    """
    from figaroh.tools.urdf_exporter import is_kinematic_correction

    self.require_exportable(allow_failed_fit)
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
```

### `metrology_frames()`

Fitted base frame and tool point: the measurement setup.

`base_*` places the robot base in the measurement frame (mocap world, camera); `pEE*`/`phiEE*` place the measured point on the tool. They describe this setup, not the robot, so they are never in :meth:`joint_corrections` and `export_urdf` does not write them. To reproduce the calibrated measurements from an exported URDF, apply them outside it, e.g. `calc_updated_fkm(reloaded_model, ..., values, q, dict(calib_config, param_name=list(frames)))`. Frames given as known (`known_baseframe`/`known_tipframe`) are not fitted and not returned.

Raises:

| Type               | Description                |
| ------------------ | -------------------------- |
| `CalibrationError` | If solve() hasn't run yet. |

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `plot_errors_distribution()`

Plot error distribution analysis for calibration assessment.

Creates bar plots showing pose error magnitudes across all samples and markers. This visualization helps identify problematic measurements, assess calibration quality, and detect outliers in the dataset.

The plots display error magnitudes (in meters) for each sample, with separate subplots for each marker when multiple markers are used.

Prerequisites

- Calibration must have run (STATUS "CALIBRATED", or "FAILED" to inspect a fit that did not converge)
- Error analysis must be computed (self.\_PEE_dist available)

Side Effects

- Creates matplotlib figure with error distribution plots
- Figure remains open until explicitly closed or plt.show() called

Raises:

| Type               | Description                             |
| ------------------ | --------------------------------------- |
| `CalibrationError` | If calibration has not been performed   |
| `AttributeError`   | If error analysis data is not available |

See Also

plot_3d_poses: 3D visualization of pose comparisons calc_stddev: Error statistics computation

Source code in `src/figaroh/calibration/base_calibration.py`

```
def plot_errors_distribution(self):
    """Plot error distribution analysis for calibration assessment.

    Creates bar plots showing pose error magnitudes across all samples
    and markers. This visualization helps identify problematic
    measurements, assess calibration quality, and detect outliers in
    the dataset.

    The plots display error magnitudes (in meters) for each sample,
    with separate subplots for each marker when multiple markers are used.

    Prerequisites:
        - Calibration must have run (STATUS "CALIBRATED", or "FAILED" to
          inspect a fit that did not converge)
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
    if self.STATUS == "NOT CALIBRATED":
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
```

### `plot_3d_poses(INCLUDE_UNCALIB=False)`

Plot 3D poses comparing measured vs estimated poses.

Parameters:

| Name              | Type   | Description                           | Default |
| ----------------- | ------ | ------------------------------------- | ------- |
| `INCLUDE_UNCALIB` | `bool` | Whether to include uncalibrated poses | `False` |

Source code in `src/figaroh/calibration/base_calibration.py`

```
def plot_3d_poses(self, INCLUDE_UNCALIB: bool = False):
    """Plot 3D poses comparing measured vs estimated poses.

    Args:
        INCLUDE_UNCALIB (bool): Whether to include uncalibrated poses
    """
    if self.STATUS == "NOT CALIBRATED":
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
```

### `plot_joint_configurations()`

Plot joint configurations within range bounds.

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `save_results(output_dir='results')`

Save calibration results using unified results manager.

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `export_html_report(output_path=None, output_dir='results')`

Export the calibration quality report as a self-contained HTML file — the visual counterpart of :meth:`print_quality_report`.

Renders the same metrics (convergence, per-DOF residuals, parameter uncertainty, correlation, validation) already computed during :meth:`solve`, plus an auto-generated "insights" section flagging ill-conditioning, poorly identified parameters, and strongly correlated pairs.

Parameters:

| Name          | Type  | Description                                                                                      | Default     |
| ------------- | ----- | ------------------------------------------------------------------------------------------------ | ----------- |
| `output_path` | `str` | Explicit file path for the report. If omitted, defaults to {output_dir}/calibration_report.html. | `None`      |
| `output_dir`  | `str` | Directory used when output_path is omitted.                                                      | `'results'` |

Returns:

| Name  | Type  | Description                         |
| ----- | ----- | ----------------------------------- |
| `str` | `str` | The path the report was written to. |

Raises:

| Type             | Description                   |
| ---------------- | ----------------------------- |
| `AttributeError` | If called before :meth:solve. |

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `verify(thresholds=None, scope='prediction')`

Return scoped acceptance evidence with explicit incomplete states.

`scope="execution"` checks finite numerical fit outputs; it does not certify prediction, physical parameters or export. `scope="prediction"` (default) additionally requires independent validation and explicit application error limits. No universal improvement, correlation, conditioning or prediction-error gate is imposed.

`thresholds` maps metric names to `threshold`, `comparison` (min/max), and optional `required` (default True). Missing required evidence produces not_evaluated, not PASS. Nonfinite evidence fails. `passed` is True only when all required checks in scope pass.

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `export_verification_report(output_path=None, output_dir='results', thresholds=None, scope='prediction')`

Write this calibration's :meth:`verify` verdict as JSON.

Parameters:

| Name          | Type                                  | Description                                                                             | Default        |
| ------------- | ------------------------------------- | --------------------------------------------------------------------------------------- | -------------- |
| `output_path` | `str`                                 | Explicit file path. If omitted, defaults to {output_dir}/calibration_verification.json. | `None`         |
| `output_dir`  | `str`                                 | Directory used when output_path is omitted.                                             | `'results'`    |
| `thresholds`  | `Optional[Dict[str, Dict[str, Any]]]` | Forwarded to :meth:verify.                                                              | `None`         |
| `scope`       | `str`                                 | Forwarded to :meth:verify; default prediction acceptance.                               | `'prediction'` |

Returns:

| Name  | Type  | Description                               |
| ----- | ----- | ----------------------------------------- |
| `str` | `str` | The path the JSON verdict was written to. |

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

### `print_quality_report()`

Print a formatted calibration quality report to the terminal.

Reports convergence, per-DOF residual statistics, validation metrics (if available), parameter uncertainty, and correlations.

Source code in `src/figaroh/calibration/base_calibration.py`

```
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
```

## calibration_tools

Calibration tools and algorithms for robot kinematic calibration.

This module contains the implementation of calibration algorithms including:

- Forward kinematics update functions
- Levenberg-Marquardt optimization
- Base regressor calculation
- Data loading and processing utilities

## `unified_to_legacy_config(robot, unified_calib_config)`

Convert unified configuration format to legacy calib_config format.

Maps the unified configuration structure to the `calib_config` dict the calibration classes consume.

Parameters:

| Name                   | Type           | Description                              | Default    |
| ---------------------- | -------------- | ---------------------------------------- | ---------- |
| `robot`                | `RobotWrapper` | Robot instance containing model and data | *required* |
| `unified_calib_config` | `dict`         | Configuration from create_task_config    | *required* |

Returns:

| Name   | Type   | Description                                   |
| ------ | ------ | --------------------------------------------- |
| `dict` | `dict` | Calibration configuration dict (calib_config) |

Raises:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `KeyError`       | If required fields are missing from unified config |
| `AssertionError` | If frame validation fails                          |

Example

> > > unified_config = create_task_config(robot, parsed_config, ... "calibration") legacy_config = unified_to_legacy_config(robot, unified_config)

Source code in `src/figaroh/calibration/config.py`

```
def unified_to_legacy_config(robot, unified_calib_config) -> dict:
    """Convert unified configuration format to legacy calib_config format.

    Maps the unified configuration structure to the ``calib_config`` dict the
    calibration classes consume.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_calib_config (dict): Configuration from create_task_config

    Returns:
        dict: Calibration configuration dict (``calib_config``)

    Raises:
        KeyError: If required fields are missing from unified config
        AssertionError: If frame validation fails

    Example:
        >>> unified_config = create_task_config(robot, parsed_config,
        ...                                    "calibration")
        >>> legacy_config = unified_to_legacy_config(robot, unified_config)
    """
    # Initialize output configuration
    calib_config = {}

    # Extract unified config sections
    joints = unified_calib_config.get("joints", {})
    kinematics = unified_calib_config.get("kinematics", {})
    parameters = unified_calib_config.get("parameters", {})
    measurements = unified_calib_config.get("measurements", {})
    data = unified_calib_config.get("data", {})

    # 1. Extract basic robot information
    calib_config["robot_name"] = robot.model.name
    calib_config["q0"] = robot.q0
    calib_config["instance"] = unified_calib_config.get("instance", {})

    # 2. Extract and validate markers/measurements
    _extract_marker_info(calib_config, measurements)

    # 3. Extract and validate kinematic frames
    _extract_frame_info(calib_config, robot, kinematics)

    # 4. Extract tool frame information
    _extract_tool_info(calib_config, robot, calib_config["end_frame"])

    # 5. Determine active joints
    _determine_active_joints(
        calib_config,
        robot,
        joints,
        calib_config["start_frame"],
        calib_config["end_frame"],
    )

    # 6. Extract poses
    _extract_poses(calib_config, measurements)

    # 7. Extract calibration parameters
    _extract_calibration_params(calib_config, robot, parameters)

    # 7b. Extract eye-hand (camera) calibration frames, if enabled
    _extract_eye_hand_params(calib_config, unified_calib_config)

    # 8. Extract data configuration
    calib_config["NbSample"] = data.get("number_of_samples", 500)
    calib_config["data_file"] = data.get("source_file")
    # held-out measurements in the same format; "" means none
    calib_config["validation_data_file"] = data.get("validation_data_file") or None
    calib_config["sample_configs_file"] = data.get("sample_configurations_file")

    return calib_config
```

## `get_sup_joints(model, start_frame, end_frame)`

Get list of supporting joints between two frames in kinematic chain.

Finds all joints that contribute to relative motion between start_frame and end_frame by analyzing their support branches in the kinematic tree.

Parameters:

| Name          | Type    | Description            | Default    |
| ------------- | ------- | ---------------------- | ---------- |
| `model`       | `Model` | Robot model            | *required* |
| `start_frame` | `str`   | Name of starting frame | *required* |
| `end_frame`   | `str`   | Name of ending frame   | *required* |

Returns:

| Type | Description                                                                                                                                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      | list\[int\]: Joint IDs ordered from start to end frame, handling cases: 1. Branch entirely contained in another branch 2. Disjoint branches with fixed root 3. Partially overlapping branches |

Raises:

| Type             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `AssertionError` | If end frame appears before start frame in chain |

Note

Excludes "universe" joints from returned list since they don't contribute to relative motion.

Source code in `src/figaroh/calibration/config.py`

```
def get_sup_joints(model, start_frame, end_frame):
    """Get list of supporting joints between two frames in kinematic chain.

    Finds all joints that contribute to relative motion between start_frame and
    end_frame by analyzing their support branches in the kinematic tree.

    Args:
        model (pin.Model): Robot model
        start_frame (str): Name of starting frame
        end_frame (str): Name of ending frame

    Returns:
        list[int]: Joint IDs ordered from start to end frame, handling cases:
            1. Branch entirely contained in another branch
            2. Disjoint branches with fixed root
            3. Partially overlapping branches

    Raises:
        AssertionError: If end frame appears before start frame in chain

    Note:
        Excludes "universe" joints from returned list since they don't
        contribute to relative motion.
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)
    start_par = model.frames[start_frameId].parentJoint
    end_par = model.frames[end_frameId].parentJoint
    branch_s = model.supports[start_par].tolist()
    branch_e = model.supports[end_par].tolist()
    # remove 'universe' joint from branches
    if model.names[branch_s[0]] == "universe":
        branch_s.remove(branch_s[0])
    if model.names[branch_e[0]] == "universe":
        branch_e.remove(branch_e[0])

    # find over-lapping joints in two branches
    shared_joints = list(set(branch_s) & set(branch_e))
    # create a list of supporting joints between two frames
    list_1 = [x for x in branch_s if x not in branch_e]
    list_1.reverse()
    list_2 = [y for y in branch_e if y not in branch_s]
    # case 2: root_joint is fixed; branch_s and branch_e are separate
    if shared_joints == []:
        sup_joints = list_1 + list_2
    else:
        # case 1: branch_s is part of branch_e
        if shared_joints == branch_s:
            sup_joints = [branch_s[-1]] + list_2
        else:
            assert shared_joints != branch_e, "End frame should be before start frame."
            # case 3: there are overlapping joints between two branches
            sup_joints = list_1 + [shared_joints[-1]] + list_2
    return sup_joints
```

## `get_joint_offset(model, joint_names)`

Get dictionary of joint offset parameters.

Maps joint names to their offset parameters, handling special cases for different joint types and multiple DOF joints.

Parameters:

| Name          | Type | Description                          | Default    |
| ------------- | ---- | ------------------------------------ | ---------- |
| `model`       |      | Pinocchio robot model                | *required* |
| `joint_names` |      | List of joint names from model.names | *required* |

Returns:

| Name   | Type | Description                                                                                                     |
| ------ | ---- | --------------------------------------------------------------------------------------------------------------- |
| `dict` |      | Mapping of joint offset parameter names to initial zero values. Keys have format: "{offset_type}\_{joint_name}" |

Example

> > > offsets = get_joint_offset(robot.model, robot.model.names[1:]) print(offsets["offsetRZ_joint1"]) 0.0

Source code in `src/figaroh/calibration/parameter.py`

```
def get_joint_offset(model, joint_names):
    """Get dictionary of joint offset parameters.

    Maps joint names to their offset parameters, handling special cases for
    different joint types and multiple DOF joints.

    Args:
        model: Pinocchio robot model
        joint_names: List of joint names from model.names

    Returns:
        dict: Mapping of joint offset parameter names to initial zero values.
            Keys have format: "{offset_type}_{joint_name}"

    Example:
        >>> offsets = get_joint_offset(robot.model, robot.model.names[1:])
        >>> print(offsets["offsetRZ_joint1"])
        0.0
    """
    joint_off = []
    joint_names = list(model.names[1:])
    joints = list(model.joints[1:])
    assert len(joint_names) == len(
        joints
    ), "Number of jointnames does not match number of joints! Please check\
        imported model."
    for id, joint in enumerate(joints):
        name = joint_names[id]
        shortname = joint.shortname()
        if model.name == "canopies":
            if "RevoluteUnaligned" in shortname:
                shortname = shortname.replace("RevoluteUnaligned", "RZ")
        for i in range(joint.nv):
            if i > 0:
                offset_param = (
                    shortname.replace("JointModel", "offset")
                    + "{}".format(i + 1)
                    + "_"
                    + name
                )
            else:
                offset_param = shortname.replace("JointModel", "offset") + "_" + name
            joint_off.append(offset_param)

    phi_jo = [0] * len(joint_off)  # default zero values
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off
```

## `get_fullparam_offset(joint_names)`

Get dictionary of geometric parameter variations.

Creates mapping of geometric offset parameters for each joint's position and orientation.

Parameters:

| Name          | Type | Description                          | Default    |
| ------------- | ---- | ------------------------------------ | ---------- |
| `joint_names` |      | List of joint names from robot model | *required* |

Returns:

| Name   | Type | Description                                                                                                                                                                                      |
| ------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dict` |      | Mapping of geometric parameter names to initial zero values. Keys have format: "d\_{param}\_{joint_name}" where param is: - px, py, pz: Position offsets - phix, phiy, phiz: Orientation offsets |

Example

> > > geo_params = get_fullparam_offset(robot.model.names[1:]) print(geo_params["d_px_joint1"]) 0.0

Source code in `src/figaroh/calibration/parameter.py`

```
def get_fullparam_offset(joint_names):
    """Get dictionary of geometric parameter variations.

    Creates mapping of geometric offset parameters for each joint's
    position and orientation.

    Args:
        joint_names: List of joint names from robot model

    Returns:
        dict: Mapping of geometric parameter names to initial zero values.
            Keys have format: "d_{param}_{joint_name}" where param is:
            - px, py, pz: Position offsets
            - phix, phiy, phiz: Orientation offsets

    Example:
        >>> geo_params = get_fullparam_offset(robot.model.names[1:])
        >>> print(geo_params["d_px_joint1"])
        0.0
    """
    geo_params = []

    for i in range(len(joint_names)):
        for j in FULL_PARAMTPL:
            # geo_params.append(j + ("_%d" % i))
            geo_params.append(j + "_" + joint_names[i])

    phi_gp = [0] * len(geo_params)  # default zero values
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params
```

## `add_base_name(calib_config)`

Add base frame parameters to parameter list.

Updates calib_config["param_name"] with base frame parameters depending on calibration model type.

Parameters:

| Name           | Type | Description                                                                                                                     | Default    |
| -------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `calib_config` |      | Parameter dictionary containing: - calib_model: "full_params" or "joint_offset" - param_name: List of parameter names to update | *required* |

Side Effects

Modifies calib_config["param_name"] in place by:

- For full_params: Replaces first 6 entries with base parameters
- For joint_offset: Prepends base parameters to list

Source code in `src/figaroh/calibration/parameter.py`

```
def add_base_name(calib_config):
    """Add base frame parameters to parameter list.

    Updates calib_config["param_name"] with base frame parameters depending on
    calibration model type.

    Args:
        calib_config: Parameter dictionary containing:
            - calib_model: "full_params" or "joint_offset"
            - param_name: List of parameter names to update

    Side Effects:
        Modifies calib_config["param_name"] in place by:
        - For full_params: Replaces first 6 entries with base parameters
        - For joint_offset: Prepends base parameters to list
    """
    if calib_config["calib_model"] == "full_params":
        calib_config["param_name"][0:6] = BASE_TPL
    elif calib_config["calib_model"] == "joint_offset":
        calib_config["param_name"] = BASE_TPL + calib_config["param_name"]
        # base_mapping_slice is positional: prepending shifts it
        if "base_mapping_slice" in calib_config:
            start, end = calib_config["base_mapping_slice"]
            calib_config["base_mapping_slice"] = (
                start + len(BASE_TPL),
                end + len(BASE_TPL),
            )
```

## `add_pee_name(calib_config)`

Add end-effector marker parameters to parameter list.

Adds parameters for each active measurement DOF of each marker.

Parameters:

| Name           | Type | Description                                                                                                                                                       | Default    |
| -------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `calib_config` |      | Parameter dictionary containing: - NbMarkers: Number of markers - measurability: List of booleans for active DOFs - param_name: List of parameter names to update | *required* |

Side Effects

Modifies calib_config["param_name"] in place by appending marker parameters in format: "{param_type}\_{marker_num}"

Source code in `src/figaroh/calibration/parameter.py`

```
def add_pee_name(calib_config):
    """Add end-effector marker parameters to parameter list.

    Adds parameters for each active measurement DOF of each marker.

    Args:
        calib_config: Parameter dictionary containing:
            - NbMarkers: Number of markers
            - measurability: List of booleans for active DOFs
            - param_name: List of parameter names to update

    Side Effects:
        Modifies calib_config["param_name"] in place by appending marker
        parameters in format: "{param_type}_{marker_num}"
    """
    PEE_names = []
    for i in range(calib_config["NbMarkers"]):
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                PEE_names.extend(["{}_{}".format(EE_TPL[j], i + 1)])
    calib_config["param_name"] = calib_config["param_name"] + PEE_names
```

## `add_eemarker_frame(frame_name, p, rpy, model, data)`

Add a new frame attached to the end-effector.

Creates and adds a fixed frame to the robot model at the end-effector location, typically used for marker or tool frames.

Parameters:

| Name         | Type      | Description                                 | Default    |
| ------------ | --------- | ------------------------------------------- | ---------- |
| `frame_name` | `str`     | Name for the new frame                      | *required* |
| `p`          | `ndarray` | 3D position offset from parent frame        | *required* |
| `rpy`        | `ndarray` | Roll-pitch-yaw angles for frame orientation | *required* |
| `model`      | `Model`   | Robot model to add frame to                 | *required* |
| `data`       | `Data`    | Robot data structure                        | *required* |

Returns:

| Name  | Type | Description               |
| ----- | ---- | ------------------------- |
| `int` |      | ID of newly created frame |

Note

Currently hardcoded to attach to "arm_7_joint". This should be made configurable in future versions.

Source code in `src/figaroh/calibration/parameter.py`

```
def add_eemarker_frame(frame_name, p, rpy, model, data):
    """Add a new frame attached to the end-effector.

    Creates and adds a fixed frame to the robot model at the end-effector
    location, typically used for marker or tool frames.

    Args:
        frame_name (str): Name for the new frame
        p (ndarray): 3D position offset from parent frame
        rpy (ndarray): Roll-pitch-yaw angles for frame orientation
        model (pin.Model): Robot model to add frame to
        data (pin.Data): Robot data structure

    Returns:
        int: ID of newly created frame

    Note:
        Currently hardcoded to attach to "arm_7_joint". This should be made
        configurable in future versions.
    """
    p = np.array([0.1, 0.1, 0.1])
    R = pin.rpy.rpyToMatrix(rpy)
    frame_placement = pin.SE3(R, p)

    parent_jointId = model.getJointId("arm_7_joint")
    prev_frameId = model.getFrameId("arm_7_joint")
    ee_frame_id = model.addFrame(
        pin.Frame(
            frame_name,
            parent_jointId,
            prev_frameId,
            frame_placement,
            pin.FrameType(0),
            pin.Inertia.Zero(),
        ),
        False,
    )
    return ee_frame_id
```

## `read_config_data(model, path_to_file)`

Read joint configurations from CSV file.

Parameters:

| Name           | Type    | Description                                      | Default    |
| -------------- | ------- | ------------------------------------------------ | ---------- |
| `model`        | `Model` | Robot model containing joint information         | *required* |
| `path_to_file` | `str`   | Path to CSV file containing joint configurations | *required* |

Returns:

| Name      | Type | Description                                                        |
| --------- | ---- | ------------------------------------------------------------------ |
| `ndarray` |      | Matrix of shape (n_samples, n_joints-1) containing joint positions |

Source code in `src/figaroh/calibration/data_loader.py`

```
def read_config_data(model, path_to_file):
    """Read joint configurations from CSV file.

    Args:
        model (pin.Model): Robot model containing joint information
        path_to_file (str): Path to CSV file containing joint configurations

    Returns:
        ndarray: Matrix of shape (n_samples, n_joints-1) containing joint
            positions
    """
    df = pd.read_csv(path_to_file)
    q = np.zeros([len(df), model.njoints - 1])
    for i in range(len(df)):
        for j, name in enumerate(model.names[1:].tolist()):
            jointidx = get_idxq_from_jname(model, name)
            q[i, jointidx] = df[name][i]
    return q
```

## `load_data(path_to_file, model, calib_config, del_list=[])`

Load joint configuration and marker data from CSV file.

Reads marker positions/orientations and joint configurations from a CSV file. Handles data validation, bad sample removal, and conversion to numpy arrays.

Parameters:

| Name           | Type    | Description                                                                                                                                                                                                                                                 | Default    |
| -------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `path_to_file` | `str`   | Path to CSV file containing recorded data                                                                                                                                                                                                                   | *required* |
| `model`        | `Model` | Robot model containing joint information                                                                                                                                                                                                                    | *required* |
| `calib_config` | `dict`  | Parameter dictionary containing: - NbMarkers: Number of markers to load - measurability: List indicating which DOFs are measured - actJoint_idx: List of active joint indices - config_idx: Configuration vector indices - q0: Default configuration vector | *required* |
| `del_list`     | `list`  | Indices of bad samples to remove. Defaults to [].                                                                                                                                                                                                           | `[]`       |

Returns:

| Name    | Type | Description                                                                                                                                      |
| ------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tuple` |      | PEEm_exp (ndarray): Flattened marker measurements of shape (n_active_dofs,) q_exp (ndarray): Joint configurations of shape (n_samples, n_joints) |

Note

CSV file must contain columns:

- For each marker i: [xi, yi, zi, phixi, phiyi, phizi]
- Joint names matching model.names for active joints

Raises:

| Type       | Description                              |
| ---------- | ---------------------------------------- |
| `KeyError` | If required columns are missing from CSV |

Side Effects

- Prints joint headers
- Updates calib_config["NbSample"] with number of valid samples

Source code in `src/figaroh/calibration/data_loader.py`

```
def load_data(path_to_file, model, calib_config, del_list=[]):
    """Load joint configuration and marker data from CSV file.

    Reads marker positions/orientations and joint configurations from a CSV
    file. Handles data validation, bad sample removal, and conversion to
    numpy arrays.

    Args:
        path_to_file (str): Path to CSV file containing recorded data
        model (pin.Model): Robot model containing joint information
        calib_config (dict): Parameter dictionary containing:
            - NbMarkers: Number of markers to load
            - measurability: List indicating which DOFs are measured
            - actJoint_idx: List of active joint indices
            - config_idx: Configuration vector indices
            - q0: Default configuration vector
        del_list (list, optional): Indices of bad samples to remove.
            Defaults to [].

    Returns:
        tuple:
            - PEEm_exp (ndarray): Flattened marker measurements of shape
                (n_active_dofs,)
            - q_exp (ndarray): Joint configurations of shape
                (n_samples, n_joints)

    Note:
        CSV file must contain columns:
        - For each marker i: [xi, yi, zi, phixi, phiyi, phizi]
        - Joint names matching model.names for active joints

    Raises:
        KeyError: If required columns are missing from CSV

    Side Effects:
        - Prints joint headers
        - Updates calib_config["NbSample"] with number of valid samples
    """
    # read_csv
    df = pd.read_csv(path_to_file)

    # create headers for marker position
    PEE_headers = []
    pee_tpl = ["x", "y", "z", "phix", "phiy", "phiz"]
    for i in range(calib_config["NbMarkers"]):
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                PEE_headers.extend(["{}{}".format(pee_tpl[j], i + 1)])

    # create headers for joint configurations
    joint_headers = [model.names[i] for i in calib_config["actJoint_idx"]]

    # check if all created headers present in csv file
    csv_headers = list(df.columns)
    for header in PEE_headers + joint_headers:
        if header not in csv_headers:
            logger.warning("%s does not exist in the file.", header)
            break

    # Extract marker position/location
    pose_ee = df[PEE_headers].to_numpy()

    # Extract joint configurations
    q_act = df[joint_headers].to_numpy()

    # remove bad data
    if del_list:
        pose_ee = np.delete(pose_ee, del_list, axis=0)
        q_act = np.delete(q_act, del_list, axis=0)

    # a missing (occluded) point would make every residual NaN: name the rows
    bad = ~np.isfinite(pose_ee).all(axis=1)
    if bad.any():
        rows = np.delete(np.arange(len(df)), del_list or [])[bad]
        raise ValueError(
            f"{path_to_file}: non-finite marker measurements in CSV rows "
            f"{rows.tolist()} (e.g. an occluded point); exclude them with "
            "del_list or clean the file"
        )

    # update number of data points
    calib_config["NbSample"] = q_act.shape[0]

    PEEm_exp = pose_ee.T
    PEEm_exp = PEEm_exp.flatten("C")

    q_exp = np.empty((calib_config["NbSample"], calib_config["q0"].shape[0]))
    for i in range(calib_config["NbSample"]):
        # a copy: q0 is robot.q0, which must not end up as the last sample (#125)
        config = calib_config["q0"].copy()
        config[calib_config["config_idx"]] = q_act[i, :]
        q_exp[i, :] = config

    return PEEm_exp, q_exp
```

## `get_idxq_from_jname(model, joint_name)`

Get index of joint in configuration vector.

Parameters:

| Name         | Type    | Description                     | Default    |
| ------------ | ------- | ------------------------------- | ---------- |
| `model`      | `Model` | Robot model                     | *required* |
| `joint_name` | `str`   | Name of joint to find index for | *required* |

Returns:

| Name  | Type | Description                              |
| ----- | ---- | ---------------------------------------- |
| `int` |      | Index of joint in configuration vector q |

Raises:

| Type             | Description                           |
| ---------------- | ------------------------------------- |
| `AssertionError` | If joint name does not exist in model |

Source code in `src/figaroh/calibration/data_loader.py`

```
def get_idxq_from_jname(model, joint_name):
    """Get index of joint in configuration vector.

    Args:
        model (pin.Model): Robot model
        joint_name (str): Name of joint to find index for

    Returns:
        int: Index of joint in configuration vector q

    Raises:
        AssertionError: If joint name does not exist in model
    """
    assert joint_name in model.names, "Given joint name does not exist."
    jointId = model.getJointId(joint_name)
    joint_idx = model.joints[jointId].idx_q
    return joint_idx
```

## `cartesian_to_SE3(X)`

Convert cartesian coordinates to SE3 transformation.

Parameters:

| Name | Type      | Description                                  | Default    |
| ---- | --------- | -------------------------------------------- | ---------- |
| `X`  | `ndarray` | (6,) array with [x,y,z,rx,ry,rz] coordinates | *required* |

Returns:

| Type | Description                                                                                          |
| ---- | ---------------------------------------------------------------------------------------------------- |
|      | pin.SE3: SE3 transformation with: - translation from X[0:3] - rotation matrix from RPY angles X[3:6] |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def cartesian_to_SE3(X):
    """Convert cartesian coordinates to SE3 transformation.

    Args:
        X (ndarray): (6,) array with [x,y,z,rx,ry,rz] coordinates

    Returns:
        pin.SE3: SE3 transformation with:
            - translation from X[0:3]
            - rotation matrix from RPY angles X[3:6]
    """
    X = np.array(X)
    X = X.flatten("C")
    translation = X[0:3]
    rot_matrix = pin.rpy.rpyToMatrix(X[3:6])
    placement = pin.SE3(rot_matrix, translation)
    return placement
```

## `xyzquat_to_SE3(xyzquat)`

Convert XYZ position and quaternion orientation to SE3 transformation.

Takes a 7D vector containing XYZ position and WXYZ quaternion and creates an SE3 transformation matrix.

Parameters:

| Name      | Type      | Description                                                                                                     | Default    |
| --------- | --------- | --------------------------------------------------------------------------------------------------------------- | ---------- |
| `xyzquat` | `ndarray` | (7,) array containing: - xyzquat\[0:3\]: XYZ position coordinates - xyzquat\[3:7\]: WXYZ quaternion orientation | *required* |

Returns:

| Type | Description                                                                                                           |
| ---- | --------------------------------------------------------------------------------------------------------------------- |
|      | pin.SE3: Rigid body transformation with: - Translation from XYZ position - Rotation matrix from normalized quaternion |

Example

> > > pos_quat = np.array([0.1, 0.2, 0.3, 1.0, 0, 0, 0]) transform = xyzquat_to_SE3(pos_quat)

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def xyzquat_to_SE3(xyzquat):
    """Convert XYZ position and quaternion orientation to SE3 transformation.

    Takes a 7D vector containing XYZ position and WXYZ quaternion and creates
    an SE3 transformation matrix.

    Args:
        xyzquat (ndarray): (7,) array containing:
            - xyzquat[0:3]: XYZ position coordinates
            - xyzquat[3:7]: WXYZ quaternion orientation

    Returns:
        pin.SE3: Rigid body transformation with:
            - Translation from XYZ position
            - Rotation matrix from normalized quaternion

    Example:
        >>> pos_quat = np.array([0.1, 0.2, 0.3, 1.0, 0, 0, 0])
        >>> transform = xyzquat_to_SE3(pos_quat)
    """
    xyzquat = np.array(xyzquat)
    xyzquat = xyzquat.flatten("C")
    translation = xyzquat[0:3]
    rot_matrix = pin.Quaternion(xyzquat[3:7]).normalize().toRotationMatrix()
    placement = pin.SE3(rot_matrix, translation)
    return placement
```

## `get_rel_transform(model, data, start_frame, end_frame)`

Get relative transformation between two frames.

Calculates the transform from start_frame to end_frame in the kinematic chain. Assumes forward kinematics has been updated.

Parameters:

| Name          | Type    | Description         | Default    |
| ------------- | ------- | ------------------- | ---------- |
| `model`       | `Model` | Robot model         | *required* |
| `data`        | `Data`  | Robot data          | *required* |
| `start_frame` | `str`   | Starting frame name | *required* |
| `end_frame`   | `str`   | Target frame name   | *required* |

Returns:

| Type | Description                                                     |
| ---- | --------------------------------------------------------------- |
|      | pin.SE3: Relative transformation sMt from start to target frame |

Raises:

| Type             | Description                         |
| ---------------- | ----------------------------------- |
| `AssertionError` | If frame names don't exist in model |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def get_rel_transform(model, data, start_frame, end_frame):
    """Get relative transformation between two frames.

    Calculates the transform from start_frame to end_frame in the kinematic chain.
    Assumes forward kinematics has been updated.

    Args:
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        start_frame (str): Starting frame name
        end_frame (str): Target frame name

    Returns:
        pin.SE3: Relative transformation sMt from start to target frame

    Raises:
        AssertionError: If frame names don't exist in model
    """
    frames = [f.name for f in model.frames]
    assert start_frame in frames, "{} does not exist.".format(start_frame)
    assert end_frame in frames, "{} does not exist.".format(end_frame)
    start_frameId = model.getFrameId(start_frame)
    oMsf = data.oMf[start_frameId]
    end_frameId = model.getFrameId(end_frame)
    oMef = data.oMf[end_frameId]
    sMef = oMsf.actInv(oMef)
    return sMef
```

## `get_rel_kinreg(model, data, start_frame, end_frame, q, backend=None)`

Calculate relative kinematic regressor between frames.

Computes frame Jacobian-based regressor matrix mapping small joint displacements to spatial velocities.

Parameters:

| Name          | Type              | Description                                                                   | Default    |
| ------------- | ----------------- | ----------------------------------------------------------------------------- | ---------- |
| `model`       | `Model`           | Robot model                                                                   | *required* |
| `data`        | `Data`            | Robot data                                                                    | *required* |
| `start_frame` | `str`             | Starting frame name                                                           | *required* |
| `end_frame`   | `str`             | Target frame name                                                             | *required* |
| `q`           | `ndarray`         | Joint configuration vector                                                    | *required* |
| `backend`     | `DynamicsBackend` | If provided, routes forward kinematics calls through the backend abstraction. | `None`     |

Returns:

| Name      | Type | Description                           |
| --------- | ---- | ------------------------------------- |
| `ndarray` |      | (6, 6n) regressor matrix for n joints |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def get_rel_kinreg(model, data, start_frame, end_frame, q, backend=None):
    """Calculate relative kinematic regressor between frames.

    Computes frame Jacobian-based regressor matrix mapping small joint displacements
    to spatial velocities.

    Args:
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        start_frame (str): Starting frame name
        end_frame (str): Target frame name
        q (ndarray): Joint configuration vector
        backend (DynamicsBackend, optional): If provided, routes forward kinematics
            calls through the backend abstraction.

    Returns:
        ndarray: (6, 6n) regressor matrix for n joints
    """
    sup_joints = get_sup_joints(model, start_frame, end_frame)
    if backend is not None:
        backend.compute_forward_kinematics(q)
    else:
        pin.framesForwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
    kinreg = np.zeros((6, 6 * (model.njoints - 1)))
    frame = model.frames[model.getFrameId(end_frame)]
    oMf = data.oMi[frame.parentJoint] * frame.placement
    for p in sup_joints:
        oMp = data.oMi[model.parents[p]] * model.jointPlacements[p]
        fMp = oMf.actInv(oMp)
        fXp = fMp.toActionMatrix()
        kinreg[:, 6 * (p - 1) : 6 * p] = fXp
    return kinreg
```

## `get_rel_jac(model, data, start_frame, end_frame, q, backend=None)`

Calculate relative Jacobian matrix between two frames.

Computes the difference between Jacobians of end_frame and start_frame, giving the differential mapping from joint velocities to relative spatial velocity.

Parameters:

| Name          | Type              | Description                                                                                | Default    |
| ------------- | ----------------- | ------------------------------------------------------------------------------------------ | ---------- |
| `model`       | `Model`           | Robot model                                                                                | *required* |
| `data`        | `Data`            | Robot data                                                                                 | *required* |
| `start_frame` | `str`             | Starting frame name                                                                        | *required* |
| `end_frame`   | `str`             | Target frame name                                                                          | *required* |
| `q`           | `ndarray`         | Joint configuration vector                                                                 | *required* |
| `backend`     | `DynamicsBackend` | If provided, routes forward kinematics and Jacobian calls through the backend abstraction. | `None`     |

Returns:

| Name      | Type | Description                                                                                                                                                 |
| --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ndarray` |      | (6, n) relative Jacobian matrix where: - Rows represent [dx,dy,dz,wx,wy,wz] spatial velocities - Columns represent joint velocities - n is number of joints |

Note

Updates forward kinematics before computing Jacobians

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def get_rel_jac(model, data, start_frame, end_frame, q, backend=None):
    """Calculate relative Jacobian matrix between two frames.

    Computes the difference between Jacobians of end_frame and start_frame,
    giving the differential mapping from joint velocities to relative spatial velocity.

    Args:
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        start_frame (str): Starting frame name
        end_frame (str): Target frame name
        q (ndarray): Joint configuration vector
        backend (DynamicsBackend, optional): If provided, routes forward kinematics
            and Jacobian calls through the backend abstraction.

    Returns:
        ndarray: (6, n) relative Jacobian matrix where:
            - Rows represent [dx,dy,dz,wx,wy,wz] spatial velocities
            - Columns represent joint velocities
            - n is number of joints

    Note:
        Updates forward kinematics before computing Jacobians
    """
    if backend is not None:
        # compute_jacobian updates FK internally
        J_start = backend.compute_jacobian(q, start_frame)
        J_end = backend.compute_jacobian(q, end_frame)
    else:
        start_frameId = model.getFrameId(start_frame)
        end_frameId = model.getFrameId(end_frame)

        # update frameForwardKinematics and updateFramePlacements
        pin.framesForwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # relative Jacobian
        J_start = pin.computeFrameJacobian(model, data, q, start_frameId, pin.LOCAL)
        J_end = pin.computeFrameJacobian(model, data, q, end_frameId, pin.LOCAL)
    J_rel = J_end - J_start
    return J_rel
```

## `initialize_variables(calib_config, mode=0, seed=0)`

Initialize variables for Levenberg-Marquardt optimization.

Creates initial parameter vector either as zeros or random values within bounds.

Parameters:

| Name           | Type    | Description                                                                                      | Default    |
| -------------- | ------- | ------------------------------------------------------------------------------------------------ | ---------- |
| `calib_config` | `dict`  | Parameter dictionary containing: - param_name: List of parameter names to initialize             | *required* |
| `mode`         | `int`   | Initialization mode: - 0: Zero initialization - 1: Random uniform initialization. Defaults to 0. | `0`        |
| `seed`         | `float` | Range [-seed,seed] for random init. Defaults to 0.                                               | `0`        |

Returns:

| Name    | Type | Description                                                              |
| ------- | ---- | ------------------------------------------------------------------------ |
| `tuple` |      | var (ndarray): Initial parameter vector nvar (int): Number of parameters |

Example

> > > var, n = initialize_variables(params, mode=1, seed=0.1) print(var.shape) (42,)

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def initialize_variables(calib_config, mode=0, seed=0):
    """Initialize variables for Levenberg-Marquardt optimization.

    Creates initial parameter vector either as zeros or random values within bounds.

    Args:
        calib_config (dict): Parameter dictionary containing:
            - param_name: List of parameter names to initialize
        mode (int, optional): Initialization mode:
            - 0: Zero initialization
            - 1: Random uniform initialization. Defaults to 0.
        seed (float, optional): Range [-seed,seed] for random init. Defaults to 0.

    Returns:
        tuple:
            - var (ndarray): Initial parameter vector
            - nvar (int): Number of parameters

    Example:
        >>> var, n = initialize_variables(params, mode=1, seed=0.1)
        >>> print(var.shape)
        (42,)
    """
    # initialize all variables at zeros
    nvar = len(calib_config["param_name"])
    if mode == 0:
        var = np.zeros(nvar)
    elif mode == 1:
        var = np.random.uniform(-seed, seed, nvar)
    return var, nvar
```

## `calc_updated_fkm(model, data, var, q, calib_config, verbose=0, backend=None)`

Update forward kinematics with world frame transformations.

Single, unified FK-update function for calibration: composes the full chain of transformations::

```
wMf = wMo * oMee * eeMf
```

where

- `wMo`: world (measurement) frame to the kinematic chain's start frame. Estimated directly when `BASE_TPL` params are present in `param_name` (unknown base frame, e.g. `known_baseframe=False`), estimated via a known camera/ref-frame anchor when `calib_config["base_to_ref_frame"]`/`"ref_frame"` are set (e.g. eye-hand calibration), or identity otherwise.
- `oMee`: start frame to end frame, through the updated kinematic chain (`full_params`/`joint_offset` geometric error parameters), optionally including joint elasticity when `calib_config["non_geom"]` is set — a per-joint compliance parameter (`ELAS_TPL`) that adds a gravity-torque-proportional deflection about that joint's own motion axis, then reverts it, on every sample.
- `eeMf`: end frame to the measured marker frame (`EE_TPL` params), or identity if not estimated. With several markers (`NbMarkers` > 1, e.g. the points of one rigid body), each marker `k` has its own `eeMf_k` (`pEEx_k` ... `phiEEz_k`) on the same tool frame, and its rows follow marker `k - 1`'s (figaroh-plus#119).

Parameters:

| Name           | Type              | Description                                                                                                                                                                                                                                                                                                                                                                                               | Default    |
| -------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `model`        | `Model`           | Robot model to update                                                                                                                                                                                                                                                                                                                                                                                     | *required* |
| `data`         | `Data`            | Robot data                                                                                                                                                                                                                                                                                                                                                                                                | *required* |
| `var`          | `ndarray`         | Parameter vector matching calib_config["param_name"]                                                                                                                                                                                                                                                                                                                                                      | *required* |
| `q`            | `ndarray`         | Joint configurations matrix (n_samples, n_joints)                                                                                                                                                                                                                                                                                                                                                         | *required* |
| `calib_config` | `dict`            | Calibration parameters containing: - calib_model: "full_params" or "joint_offset" - start_frame, end_frame: Frame names - base_to_ref_frame, ref_frame: Optional camera-style known chain anchor (eye-hand calibration); None to disable - non_geom: Whether to apply joint elasticity - actJoint_idx: Active joint indices - measurability: Active DOFs - NbMarkers: Number of markers on the tool frame | *required* |
| `verbose`      | `int`             | Print update info. Defaults to 0.                                                                                                                                                                                                                                                                                                                                                                         | `0`        |
| `backend`      | `DynamicsBackend` | If provided, routes forward kinematics and gravity calls through the backend abstraction.                                                                                                                                                                                                                                                                                                                 | `None`     |

Returns:

| Name      | Type | Description                                                 |
| --------- | ---- | ----------------------------------------------------------- |
| `ndarray` |      | Flattened marker measurements in world frame, ordered       |
|           |      | marker, then measured component, then sample (as load_data) |

Notes

- Requires base or end-effector parameters in param_name to estimate wMo / eeMf; otherwise they default to identity.
- Validates all parameters in param_name are consumed exactly once.

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def calc_updated_fkm(model, data, var, q, calib_config, verbose=0, backend=None):
    """Update forward kinematics with world frame transformations.

    Single, unified FK-update function for calibration: composes the full
    chain of transformations::

        wMf = wMo * oMee * eeMf

    where:
        - ``wMo``: world (measurement) frame to the kinematic chain's start
          frame. Estimated directly when ``BASE_TPL`` params are present in
          ``param_name`` (unknown base frame, e.g. ``known_baseframe=False``),
          estimated via a known camera/ref-frame anchor when
          ``calib_config["base_to_ref_frame"]``/``"ref_frame"`` are set (e.g.
          eye-hand calibration), or identity otherwise.
        - ``oMee``: start frame to end frame, through the updated kinematic
          chain (``full_params``/``joint_offset`` geometric error
          parameters), optionally including joint elasticity when
          ``calib_config["non_geom"]`` is set — a per-joint compliance
          parameter (``ELAS_TPL``) that adds a gravity-torque-proportional
          deflection about that joint's own motion axis, then reverts it,
          on every sample.
        - ``eeMf``: end frame to the measured marker frame (``EE_TPL``
          params), or identity if not estimated. With several markers
          (``NbMarkers`` > 1, e.g. the points of one rigid body), each
          marker ``k`` has its own ``eeMf_k`` (``pEEx_k`` ... ``phiEEz_k``)
          on the same tool frame, and its rows follow marker ``k - 1``'s
          (figaroh-plus#119).

    Args:
        model (pin.Model): Robot model to update
        data (pin.Data): Robot data
        var (ndarray): Parameter vector matching calib_config["param_name"]
        q (ndarray): Joint configurations matrix (n_samples, n_joints)
        calib_config (dict): Calibration parameters containing:
            - calib_model: "full_params" or "joint_offset"
            - start_frame, end_frame: Frame names
            - base_to_ref_frame, ref_frame: Optional camera-style known
              chain anchor (eye-hand calibration); None to disable
            - non_geom: Whether to apply joint elasticity
            - actJoint_idx: Active joint indices
            - measurability: Active DOFs
            - NbMarkers: Number of markers on the tool frame
        verbose (int, optional): Print update info. Defaults to 0.
        backend (DynamicsBackend, optional): If provided, routes forward
            kinematics and gravity calls through the backend abstraction.

    Returns:
        ndarray: Flattened marker measurements in world frame, ordered
        marker, then measured component, then sample (as ``load_data``)

    Notes:
        - Requires base or end-effector parameters in param_name to
          estimate wMo / eeMf; otherwise they default to identity.
        - Validates all parameters in param_name are consumed exactly once.
    """

    # name reference of calibration parameters
    if calib_config["calib_model"] == "full_params":
        axis_tpl = FULL_PARAMTPL

    elif calib_config["calib_model"] == "joint_offset":
        axis_tpl = JOINT_OFFSETTPL

    # order of joint in variables are arranged as in calib_config['actJoint_idx']
    assert len(var) == len(
        calib_config["param_name"]
    ), "Length of variables != length of params"
    param_dict = dict(zip(calib_config["param_name"], var))

    # store parameter updated to the model
    updated_params = []

    # check if baseframe and end--effector frame are known; with no
    # parameters at all (nominal FK) neither is (#129)
    base_param_incl = any("base" in key for key in param_dict)
    ee_param_incl = any("EE" in key for key in param_dict)

    # kinematic chain
    start_f = calib_config["start_frame"]
    end_f = calib_config["end_frame"]

    # 1/ calc transformation from the world frame to start frame: wMo
    base_to_ref_frame = calib_config.get("base_to_ref_frame")
    if base_to_ref_frame is not None:
        # known chain anchor + unknown camera/ref pose (e.g. eye-hand calib)
        start_f = calib_config["ref_frame"]
        base_tf = np.zeros(6)
        for key in param_dict.keys():
            for base_id, base_ax in enumerate(BASE_TPL):
                if base_ax in key:
                    base_tf[base_id] = param_dict[key]
                    updated_params.append(key)
        b_to_cam = get_rel_transform(
            model, data, calib_config["start_frame"], base_to_ref_frame
        )
        ref_to_cam = cartesian_to_SE3(base_tf)
        cam_to_ref = ref_to_cam.actInv(pin.SE3.Identity())
        wMo = b_to_cam * cam_to_ref
    elif base_param_incl:
        # fully unknown world (measurement) frame to start frame transform
        base_tf = np.zeros(6)
        for key in param_dict.keys():
            for base_id, base_ax in enumerate(BASE_TPL):
                if base_ax in key:
                    base_tf[base_id] = param_dict[key]
                    updated_params.append(key)

        wMo = cartesian_to_SE3(base_tf)
    else:
        wMo = pin.SE3.Identity()

    # 2/ calculate transformation from the end frame to each marker frame,
    # if not known: eeMf (one per marker, identity when not estimated)
    n_markers = calib_config["NbMarkers"]
    eeMf = [pin.SE3.Identity()] * n_markers
    if ee_param_incl:
        for marker_idx in range(1, n_markers + 1):
            pee = np.zeros(6)
            for axis_pee_id, axis_pee in enumerate(EE_TPL):
                key = "{}_{}".format(axis_pee, marker_idx)
                if key in param_dict:
                    if verbose == 1:
                        logger.debug("Updating marker frame with [{}]".format(key))
                    pee[axis_pee_id] += param_dict[key]
                    updated_params.append(key)
            eeMf[marker_idx - 1] = cartesian_to_SE3(pee)

    # 3/ calculate transformation from start frame to end frame of kinematic chain using updated model: oMee

    # placements are restored from this copy once the samples are evaluated
    saved_placements = {
        j_id: model.jointPlacements[j_id].copy()
        for j_id in calib_config["actJoint_idx"]
    }

    # update model.jointPlacements with kinematic error parameter
    for j_id in calib_config["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]

        # check joint name in param dict
        for key in param_dict.keys():
            if j_name in key:

                # update xyz_rpy with kinematic errors based on identifiable axis
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose == 1:
                            logger.debug(
                                "Updating [{}] joint placement at axis {} with [{}]".format(
                                    j_name, axis, key
                                )
                            )
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)

        # update joint placement: full_params perturb the placement in the
        # parent frame; joint offsets act about the joint's own axes (q + offset)
        if calib_config["calib_model"] == "joint_offset":
            model = apply_joint_offset(model, j_id, xyz_rpy)
        else:
            model = update_joint_placement(model, j_id, xyz_rpy)

    # joint elasticity: one compliance parameter per active joint (ELAS_TPL,
    # see _build_elastic_param_names), matched once here since the mapping
    # joint -> param is static; the deflection itself is gravity-torque
    # dependent and recomputed per sample below.
    elastic_map = {}
    if calib_config.get("non_geom"):
        for j_id in calib_config["actJoint_idx"]:
            j_name = model.names[j_id]
            for key in param_dict.keys():
                if j_name in key:
                    for elas_id, elas in enumerate(ELAS_TPL):
                        if elas in key:
                            if verbose == 1:
                                logger.debug(
                                    "Joint [{}] elastic gain [{}] on axis {}".format(
                                        j_name, key, elas
                                    )
                                )
                            elastic_map[j_id] = (key, elas_id)
                            updated_params.append(key)

    # check if all parameters are updated to the model
    assert len(updated_params) == len(
        list(param_dict.keys())
    ), "Not all parameters are updated {} and {}".format(
        updated_params, list(param_dict.keys())
    )

    # pose vector of the markers: rows are marker-major, then component
    n_dofs = calib_config["calibration_index"]
    PEE = np.zeros((n_markers * n_dofs, calib_config["NbSample"]))

    q_ = np.copy(q)
    for i in range(calib_config["NbSample"]):

        if backend is not None:
            backend.compute_forward_kinematics(q_[i, :])
        else:
            pin.framesForwardKinematics(model, data, q_[i, :])
            pin.updateFramePlacements(model, data)

        if elastic_map:
            if backend is not None:
                tau = backend.compute_gravity_vector(q_[i, :])
            else:
                tau = pin.computeGeneralizedGravity(model, data, q_[i, :])

            geometric_placements = {
                j_id: model.jointPlacements[j_id].copy() for j_id in elastic_map
            }
            for j_id, (key, elas_id) in elastic_map.items():
                xyz_rpy = np.zeros(6)
                xyz_rpy[elas_id] = param_dict[key] * tau[j_id - 1]
                model = apply_joint_offset(model, j_id, xyz_rpy)

            # jointPlacements changed: data.oMf is stale until FK is redone
            if backend is not None:
                backend.compute_forward_kinematics(q_[i, :])
            else:
                pin.framesForwardKinematics(model, data, q_[i, :])
                pin.updateFramePlacements(model, data)

            oMee = get_rel_transform(model, data, start_f, end_f)

            # revert model back to origin from the added joint elastic error
            for j_id, placement in geometric_placements.items():
                model.jointPlacements[j_id] = placement
        else:
            oMee = get_rel_transform(model, data, start_f, end_f)

        # calculate transformation from world frame to each marker frame
        wMee = wMo * oMee
        for k, eeMf_k in enumerate(eeMf):
            wMf = wMee * eeMf_k
            trans = wMf.translation.tolist()
            orient = pin.rpy.matrixToRpy(wMf.rotation).tolist()
            loc = trans + orient
            measure = [
                loc[mea_id]
                for mea_id, mea in enumerate(calib_config["measurability"])
                if mea
            ]
            PEE[k * n_dofs : (k + 1) * n_dofs, i] = np.array(measure)

    # final result of updated fkm
    PEE = PEE.flatten("C")

    # revert model back to original
    for j_id, placement in saved_placements.items():
        model.jointPlacements[j_id] = placement

    return PEE
```

## `update_joint_placement(model, joint_idx, xyz_rpy)`

Apply a `full_params` placement error in the joint frame.

`M <- M * SE3(exp3(xyz_rpy[3:6]), xyz_rpy[0:3])`: the translation is expressed in the joint frame and the rotation is a rotation vector. This is the convention of Pinocchio's kinematic regressor (`computeFrameKinematicRegressor(..., LOCAL)`), from which the base parameters and the base-mapping matrix are derived, so the fitted model has exactly the dependencies that selection assumed (figaroh-plus#110). Earlier versions added the translation in the parent frame and the rotation to the placement's RPY angles, which disagrees with the regressor wherever the nominal placement is rotated.

Parameters:

| Name        | Type      | Description                                                                                     | Default    |
| ----------- | --------- | ----------------------------------------------------------------------------------------------- | ---------- |
| `model`     | `Model`   | Robot model to modify                                                                           | *required* |
| `joint_idx` | `int`     | Index of joint to update                                                                        | *required* |
| `xyz_rpy`   | `ndarray` | (6,) d_px, d_py, d_pz (m) and d_phix, d_phiy, d_phiz (rad, rotation vector), in the joint frame | *required* |

Returns:

| Type | Description                    |
| ---- | ------------------------------ |
|      | pin.Model: Updated robot model |

Side Effects

Modifies model.jointPlacements[joint_idx] in place

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def update_joint_placement(model, joint_idx, xyz_rpy):
    """Apply a ``full_params`` placement error in the joint frame.

    ``M <- M * SE3(exp3(xyz_rpy[3:6]), xyz_rpy[0:3])``: the translation is
    expressed in the joint frame and the rotation is a rotation vector. This
    is the convention of Pinocchio's kinematic regressor
    (``computeFrameKinematicRegressor(..., LOCAL)``), from which the base
    parameters and the base-mapping matrix are derived, so the fitted model
    has exactly the dependencies that selection assumed (figaroh-plus#110).
    Earlier versions added the translation in the parent frame and the
    rotation to the placement's RPY angles, which disagrees with the
    regressor wherever the nominal placement is rotated.

    Args:
        model (pin.Model): Robot model to modify
        joint_idx (int): Index of joint to update
        xyz_rpy (ndarray): (6,) ``d_px, d_py, d_pz`` (m) and ``d_phix,
            d_phiy, d_phiz`` (rad, rotation vector), in the joint frame

    Returns:
        pin.Model: Updated robot model

    Side Effects:
        Modifies model.jointPlacements[joint_idx] in place
    """
    xyz_rpy = np.asarray(xyz_rpy, dtype=float)
    delta = pin.SE3(pin.exp3(xyz_rpy[3:6]), xyz_rpy[0:3])
    model.jointPlacements[joint_idx] = model.jointPlacements[joint_idx] * delta
    return model
```

## `apply_joint_offset(model, joint_idx, offset)`

Offset a joint about its own axes (joint-angle / joint-position offset).

Composes the offset on the child side of the joint placement, `M <- M * SE3(R(offset[3:6]), offset[0:3])`, so that it is expressed in the joint frame. For a revolute joint about z, `offset[5] = d` is then exactly the configuration `q + d`; for a prismatic joint along x, `offset[0] = d` is `q + d`. This is the meaning of the `offset{PX,PY,PZ,RX,RY,RZ}_<joint>` parameters (`JOINT_OFFSETTPL`, axis taken from the joint's `shortname()`) and of the elastic deflections (`ELAS_TPL`).

:func:`update_joint_placement` composes on the same side for the `full_params` (`d_p*`, `d_phi*`) placement errors, with all three rotation components as one rotation vector.

Parameters:

| Name        | Type      | Description                                                                                      | Default    |
| ----------- | --------- | ------------------------------------------------------------------------------------------------ | ---------- |
| `model`     | `Model`   | Robot model to modify                                                                            | *required* |
| `joint_idx` | `int`     | Index of joint to update                                                                         | *required* |
| `offset`    | `ndarray` | (6,) translation (x, y, z) and rotation (roll, pitch, yaw) offsets, expressed in the joint frame | *required* |

Returns:

| Type | Description                    |
| ---- | ------------------------------ |
|      | pin.Model: Updated robot model |

Side Effects

Modifies model.jointPlacements[joint_idx] in place

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def apply_joint_offset(model, joint_idx, offset):
    """Offset a joint about its own axes (joint-angle / joint-position offset).

    Composes the offset on the child side of the joint placement,
    ``M <- M * SE3(R(offset[3:6]), offset[0:3])``, so that it is expressed in
    the joint frame. For a revolute joint about z, ``offset[5] = d`` is then
    exactly the configuration ``q + d``; for a prismatic joint along x,
    ``offset[0] = d`` is ``q + d``. This is the meaning of the
    ``offset{PX,PY,PZ,RX,RY,RZ}_<joint>`` parameters (``JOINT_OFFSETTPL``,
    axis taken from the joint's ``shortname()``) and of the elastic
    deflections (``ELAS_TPL``).

    :func:`update_joint_placement` composes on the same side for the
    ``full_params`` (``d_p*``, ``d_phi*``) placement errors, with all three
    rotation components as one rotation vector.

    Args:
        model (pin.Model): Robot model to modify
        joint_idx (int): Index of joint to update
        offset (ndarray): (6,) translation (x, y, z) and rotation (roll,
            pitch, yaw) offsets, expressed in the joint frame

    Returns:
        pin.Model: Updated robot model

    Side Effects:
        Modifies model.jointPlacements[joint_idx] in place
    """
    offset = np.asarray(offset, dtype=float)
    delta = pin.SE3(pin.rpy.rpyToMatrix(offset[3:6]), offset[0:3])
    model.jointPlacements[joint_idx] = model.jointPlacements[joint_idx] * delta
    return model
```

## `calculate_kinematics_model(q_i, model, data, calib_config, backend=None)`

Calculate Jacobian and kinematic regressor for single configuration.

Computes frame Jacobian and kinematic regressor matrices for tool frame at given joint configuration.

Parameters:

| Name           | Type              | Description                                                                                | Default    |
| -------------- | ----------------- | ------------------------------------------------------------------------------------------ | ---------- |
| `q_i`          | `ndarray`         | Joint configuration vector                                                                 | *required* |
| `model`        | `Model`           | Robot model                                                                                | *required* |
| `data`         | `Data`            | Robot data                                                                                 | *required* |
| `calib_config` | `dict`            | Parameters containing "IDX_TOOL" frame index                                               | *required* |
| `backend`      | `DynamicsBackend` | If provided, routes forward kinematics and Jacobian calls through the backend abstraction. | `None`     |

Returns:

| Name    | Type | Description                                                                                                                                            |
| ------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tuple` |      | model (pin.Model): Updated model data (pin.Data): Updated data R (ndarray): (6,6n) Kinematic regressor matrix J (ndarray): (6,n) Frame Jacobian matrix |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def calculate_kinematics_model(q_i, model, data, calib_config, backend=None):
    """Calculate Jacobian and kinematic regressor for single configuration.

    Computes frame Jacobian and kinematic regressor matrices for tool frame
    at given joint configuration.

    Args:
        q_i (ndarray): Joint configuration vector
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        calib_config (dict): Parameters containing "IDX_TOOL" frame index
        backend (DynamicsBackend, optional): If provided, routes forward kinematics
            and Jacobian calls through the backend abstraction.

    Returns:
        tuple:
            - model (pin.Model): Updated model
            - data (pin.Data): Updated data
            - R (ndarray): (6,6n) Kinematic regressor matrix
            - J (ndarray): (6,n) Frame Jacobian matrix
    """
    if backend is not None:
        # compute_forward_kinematics updates FK internally
        backend.compute_forward_kinematics(q_i)
        # Convert frame ID to name for backend Jacobian
        frame_id = calib_config["IDX_TOOL"]
        frame_name = model.frames[frame_id].name
        J = backend.compute_jacobian(q_i, frame_name)
    else:
        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)
        J = pin.computeFrameJacobian(
            model, data, q_i, calib_config["IDX_TOOL"], pin.LOCAL
        )

    # computeFrameKinematicRegressor has no backend equivalent — use escape hatch
    R = pin.computeFrameKinematicRegressor(
        model, data, calib_config["IDX_TOOL"], pin.LOCAL
    )
    return model, data, R, J
```

## `calculate_identifiable_kinematics_model(q, model, data, calib_config, backend=None)`

Calculate identifiable Jacobian and regressor matrices.

Builds aggregated Jacobian and regressor matrices from either:

1. Given set of configurations, or
1. Random configurations if none provided

Parameters:

| Name           | Type              | Description                                                                                                                                                            | Default    |
| -------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `q`            | `ndarray`         | Joint configurations matrix. If empty, uses random configs.                                                                                                            | *required* |
| `model`        | `Model`           | Robot model                                                                                                                                                            | *required* |
| `data`         | `Data`            | Robot data                                                                                                                                                             | *required* |
| `calib_config` | `dict`            | Parameters containing: - NbSample: Number of configurations - calibration_index: Number of active DOFs - start_frame, end_frame: Frame names - calib_model: Model type | *required* |
| `backend`      | `DynamicsBackend` | If provided, routes random configuration and forwards backend to called functions.                                                                                     | `None`     |

Returns:

| Name      | Type | Description                                                                                       |
| --------- | ---- | ------------------------------------------------------------------------------------------------- |
| `ndarray` |      | Either: - Joint offset case: Frame Jacobian matrix - Full params case: Kinematic regressor matrix |

Note

Removes rows corresponding to inactive DOFs and zero elements

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def calculate_identifiable_kinematics_model(q, model, data, calib_config, backend=None):
    """Calculate identifiable Jacobian and regressor matrices.

    Builds aggregated Jacobian and regressor matrices from either:
    1. Given set of configurations, or
    2. Random configurations if none provided

    Args:
        q (ndarray, optional): Joint configurations matrix. If empty, uses random configs.
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        calib_config (dict): Parameters containing:
            - NbSample: Number of configurations
            - calibration_index: Number of active DOFs
            - start_frame, end_frame: Frame names
            - calib_model: Model type
        backend (DynamicsBackend, optional): If provided, routes random configuration
            and forwards backend to called functions.

    Returns:
        ndarray: Either:
            - Joint offset case: Frame Jacobian matrix
            - Full params case: Kinematic regressor matrix

    Note:
        Removes rows corresponding to inactive DOFs and zero elements
    """
    q_temp = np.copy(q)
    # Note if no q id given then use random generation of q to determine the
    # minimal kinematics model
    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain aggreated Jacobian matrix J and kinematic regressor R
    R = np.zeros([6 * calib_config["NbSample"], 6 * (model.njoints - 1)])
    J = np.zeros([6 * calib_config["NbSample"], model.njoints - 1])
    # seeded, so the selected parameter set does not depend on global RNG
    # state (figaroh-plus#99)
    rng = np.random.default_rng(calib_config.get("random_seed", 0))
    for i in range(calib_config["NbSample"]):
        if MIN_MODEL == 1:
            q_rand = random_joint_configuration(model, rng)
            # a copy: q0 is robot.q0 and must not be overwritten (#125)
            q_i = calib_config["q0"].copy()
            q_i[calib_config["config_idx"]] = q_rand[calib_config["config_idx"]]
        else:
            q_i = q_temp[i, :]
        if calib_config["start_frame"] == "universe":
            model, data, Ri, Ji = calculate_kinematics_model(
                q_i, model, data, calib_config, backend=backend
            )
        else:
            Ri = get_rel_kinreg(
                model,
                data,
                calib_config["start_frame"],
                calib_config["end_frame"],
                q_i,
                backend=backend,
            )
            # Ji = np.zeros([6, model.njoints-1]) ## TODO: get_rel_jac
            Ji = get_rel_jac(
                model,
                data,
                calib_config["start_frame"],
                calib_config["end_frame"],
                q_i,
                backend=backend,
            )
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                R[calib_config["NbSample"] * j + i, :] = Ri[j, :]
                J[calib_config["NbSample"] * j + i, :] = Ji[j, :]
    # remove zero rows
    zero_rows = []
    for r_idx in range(R.shape[0]):
        if np.linalg.norm(R[r_idx, :]) < 1e-6:
            zero_rows.append(r_idx)
    R = np.delete(R, zero_rows, axis=0)
    zero_rows = []
    for r_idx in range(J.shape[0]):
        if np.linalg.norm(J[r_idx, :]) < 1e-6:
            zero_rows.append(r_idx)
    J = np.delete(J, zero_rows, axis=0)

    # select regressor matrix based on calibration model
    if calib_config["calib_model"] == "joint_offset":
        return J
    elif calib_config["calib_model"] == "full_params":
        return R
```

## `calculate_base_kinematics_regressor(q, model, data, calib_config, tol_qr=TOL_QR, backend=None)`

Calculate base regressor matrix for calibration parameters.

Identifies base (identifiable) parameters by:

1. Computing regressors with random/given configurations
1. Eliminating unidentifiable parameters
1. Finding independent regressor columns

Parameters:

| Name           | Type              | Description                                                                                                                 | Default    |
| -------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `q`            | `ndarray`         | Joint configurations matrix                                                                                                 | *required* |
| `model`        | `Model`           | Robot model                                                                                                                 | *required* |
| `data`         | `Data`            | Robot data                                                                                                                  | *required* |
| `calib_config` | `dict`            | Contains calibration settings: - free_flyer: Whether base is floating - calib_model: Either "joint_offset" or "full_params" | *required* |
| `tol_qr`       | `float`           | QR decomposition tolerance. Defaults to TOL_QR.                                                                             | `TOL_QR`   |
| `backend`      | `DynamicsBackend` | If provided, forwards backend to called functions for backend-aware computation.                                            | `None`     |

Returns:

| Name    | Type | Description                                                                                                                                                                                                                                                                                             |
| ------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tuple` |      | Rrand_b (ndarray): Base regressor from random configs R_b (ndarray): Base regressor from given configs R_e (ndarray): Full regressor after eliminating unidentifiable params paramsrand_base (list): Names of base parameters from random configs paramsrand_e (list): Names of identifiable parameters |

Side Effects

- Updates calib_config["param_name"] with identified base parameters
- Prints regressor matrix shapes

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def calculate_base_kinematics_regressor(
    q, model, data, calib_config, tol_qr=TOL_QR, backend=None
):
    """Calculate base regressor matrix for calibration parameters.

    Identifies base (identifiable) parameters by:
    1. Computing regressors with random/given configurations
    2. Eliminating unidentifiable parameters
    3. Finding independent regressor columns

    Args:
        q (ndarray): Joint configurations matrix
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        calib_config (dict): Contains calibration settings:
            - free_flyer: Whether base is floating
            - calib_model: Either "joint_offset" or "full_params"
        tol_qr (float, optional): QR decomposition tolerance. Defaults to TOL_QR.
        backend (DynamicsBackend, optional): If provided, forwards backend to
            called functions for backend-aware computation.

    Returns:
        tuple:
            - Rrand_b (ndarray): Base regressor from random configs
            - R_b (ndarray): Base regressor from given configs
            - R_e (ndarray): Full regressor after eliminating unidentifiable params
            - paramsrand_base (list): Names of base parameters from random configs
            - paramsrand_e (list): Names of identifiable parameters

    Side Effects:
        - Updates calib_config["param_name"] with identified base parameters
        - Prints regressor matrix shapes
    """
    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names[1:])]
    geo_params = get_fullparam_offset(joint_names)
    joint_offsets = get_joint_offset(model, joint_names)

    # Several markers of one body (figaroh-plus#119) observe the tool
    # frame's orientation, even when each marker's position alone is
    # measured: the structural regressor then uses every component.
    # Dependencies the actual points leave are removed at the data level.
    reg_config = calib_config
    if calib_config.get("NbMarkers", 1) > 1:
        reg_config = dict(calib_config, measurability=[True] * 6, calibration_index=6)

    # calculate kinematic regressor with random configs
    if not calib_config["free_flyer"]:
        Rrand = calculate_identifiable_kinematics_model(
            [], model, data, reg_config, backend=backend
        )
    else:
        Rrand = calculate_identifiable_kinematics_model(
            q, model, data, reg_config, backend=backend
        )
    # calculate kinematic regressor with input configs
    if np.any(np.array(q)):
        R = calculate_identifiable_kinematics_model(
            q, model, data, reg_config, backend=backend
        )
    else:
        R = Rrand

    # only joint offset parameters
    if calib_config["calib_model"] == "joint_offset":
        geo_params_sel = joint_offsets

        # select columns corresponding to joint_idx
        Rrand_sel = Rrand

        # select columns corresponding to joint_idx
        R_sel = R

    # full 6 parameters
    elif calib_config["calib_model"] == "full_params":
        geo_params_sel = geo_params
        Rrand_sel = Rrand
        R_sel = R

    # remove non affect columns from random data => reduced regressor
    Rrand_e, paramsrand_e = eliminate_non_dynaffect(
        Rrand_sel, geo_params_sel, tol_e=1e-6
    )

    # indices of independent columns (base param) w.r.t the reduced
    # regressor, and the base-mapping matrix M s.t. phi_base = M @ theta_r
    # (theta_r ordered as paramsrand_e) -- one QR pass on the structural
    # (random-config) regressor covers both, since the base/dependent
    # column split is a property of the kinematic chain, not of any
    # particular dataset.
    # LAPACK pivoting: calibration selection is made deterministic in #113,
    # not here, so calibration results stay as they were.
    decomposer = QRDecomposer(tolerance=tol_qr, deterministic=False)
    M, paramsrand_base, idx_base, _ = decomposer.get_base_mapping_matrix_double(
        Rrand_e, paramsrand_e
    )

    # get base regressor from random data
    Rrand_b = build_baseRegressor(Rrand_e, idx_base)

    # remove non affect columns from GIVEN data
    R_e, _ = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

    # get base regressor from GIVEN data
    R_b = build_baseRegressor(R_e, idx_base)

    # update calibrating calib_config['param_name']/calibrating parameters
    # -- record the slice these base entries land at (param_name may
    # already carry earlier entries, e.g. elastic-gain params) so callers
    # can locate the corresponding values in a solved variable vector
    # (which is built one-to-one, in order, from param_name) regardless of
    # any later in-place renaming (add_base_name) or appending (add_pee_name).
    _base_slice_start = len(calib_config["param_name"])
    for j in idx_base:
        calib_config["param_name"].append(paramsrand_e[j])
    _base_slice_end = len(calib_config["param_name"])

    # Structural base-mapping, stashed for optional downstream
    # redistribution (see figaroh.tools.qrdecomposition.redistribute_min_norm
    # and BaseCalibration.redistribute_parameters). base_mapping_row_names
    # is a snapshot of the base-parameter name order *before*
    # add_base_name/add_pee_name may rename/prepend entries in
    # calib_config["param_name"], so callers should locate fitted values by
    # base_mapping_slice (positional), not by re-looking-up these names.
    calib_config["base_mapping_matrix"] = M
    calib_config["base_mapping_param_names"] = list(paramsrand_e)
    calib_config["base_mapping_row_names"] = [paramsrand_e[j] for j in idx_base]
    calib_config["base_mapping_slice"] = (_base_slice_start, _base_slice_end)

    return Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e
```

## `random_joint_configuration(model, rng)`

Draw a configuration uniformly within joint limits from `rng`.

One-DoF joints are drawn within their position limits, or within [-pi, pi] when the limits are missing or wider than a turn; other joints stay at the neutral configuration. Unlike `pin.randomConfiguration`, the draw depends only on `rng`, not on Pinocchio's global generator.

Parameters:

| Name    | Type        | Description      | Default    |
| ------- | ----------- | ---------------- | ---------- |
| `model` | `Model`     | Robot model      | *required* |
| `rng`   | `Generator` | Random generator | *required* |

Returns:

| Name      | Type | Description         |
| --------- | ---- | ------------------- |
| `ndarray` |      | (nq,) configuration |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def random_joint_configuration(model, rng):
    """Draw a configuration uniformly within joint limits from ``rng``.

    One-DoF joints are drawn within their position limits, or within
    [-pi, pi] when the limits are missing or wider than a turn; other joints
    stay at the neutral configuration. Unlike ``pin.randomConfiguration``,
    the draw depends only on ``rng``, not on Pinocchio's global generator.

    Args:
        model (pin.Model): Robot model
        rng (np.random.Generator): Random generator

    Returns:
        ndarray: (nq,) configuration
    """
    q = pin.neutral(model)
    for joint in model.joints[1:]:
        if joint.nq != 1:
            continue
        lo = model.lowerPositionLimit[joint.idx_q]
        hi = model.upperPositionLimit[joint.idx_q]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo > 2 * np.pi:
            lo, hi = -np.pi, np.pi
        q[joint.idx_q] = rng.uniform(lo, hi)
    return q
```

## `estimate_frames_closed_form(model, data, q, PEE, calib_config, n_iter=50)`

Closed-form initial guess for the unknown base and tip frames.

With nominal joint parameters, the measured marker satisfies `P_i = R_b (p_i + R_i t_tip) + t_b` (and `R_meas_i = R_b R_i R_tip` when orientation is measured), where `(R_i, p_i)` is the nominal start-to-end frame transform. Rotations are estimated by Kabsch (positions) or chordal averaging (orientations), alternated with a linear least-squares solve for `t_b` and `t_tip`.

With several markers (`NbMarkers` > 1, points of one rigid body, figaroh-plus#119), each has its own `t_tip_k` and the base frame is fitted to all of them; orientations, if measured, are not used then and tip rotations are guessed as zero.

Only applies to markers whose position is fully measured, with the base frame estimated directly (`base_*` parameters, no camera `base_to_ref_frame` anchor). Otherwise returns an empty dict.

Parameters:

| Name           | Type      | Description                                         | Default    |
| -------------- | --------- | --------------------------------------------------- | ---------- |
| `model`        | `Model`   | Robot model (nominal joint placements)              | *required* |
| `data`         | `Data`    | Robot data                                          | *required* |
| `q`            | `ndarray` | (NbSample, nq) joint configurations                 | *required* |
| `PEE`          | `ndarray` | Flattened DOF-major measurements, as from load_data | *required* |
| `calib_config` | `dict`    | Calibration configuration                           | *required* |
| `n_iter`       | `int`     | Alternation iterations                              | `50`       |

Returns:

| Name   | Type | Description                                         |
| ------ | ---- | --------------------------------------------------- |
| `dict` |      | Initial values keyed by the base\_\*, pEE\*/phiEE\* |
|        |      | names present in calib_config["param_name"].        |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def estimate_frames_closed_form(model, data, q, PEE, calib_config, n_iter=50):
    """Closed-form initial guess for the unknown base and tip frames.

    With nominal joint parameters, the measured marker satisfies
    ``P_i = R_b (p_i + R_i t_tip) + t_b`` (and ``R_meas_i = R_b R_i R_tip``
    when orientation is measured), where ``(R_i, p_i)`` is the nominal
    start-to-end frame transform. Rotations are estimated by Kabsch
    (positions) or chordal averaging (orientations), alternated with a linear
    least-squares solve for ``t_b`` and ``t_tip``.

    With several markers (``NbMarkers`` > 1, points of one rigid body,
    figaroh-plus#119), each has its own ``t_tip_k`` and the base frame is
    fitted to all of them; orientations, if measured, are not used then and
    tip rotations are guessed as zero.

    Only applies to markers whose position is fully measured, with the
    base frame estimated directly (``base_*`` parameters, no camera
    ``base_to_ref_frame`` anchor). Otherwise returns an empty dict.

    Args:
        model (pin.Model): Robot model (nominal joint placements)
        data (pin.Data): Robot data
        q (ndarray): (NbSample, nq) joint configurations
        PEE (ndarray): Flattened DOF-major measurements, as from ``load_data``
        calib_config (dict): Calibration configuration
        n_iter (int): Alternation iterations

    Returns:
        dict: Initial values keyed by the ``base_*``, ``pEE*``/``phiEE*``
        names present in ``calib_config["param_name"]``.
    """
    names = list(calib_config["param_name"])
    meas = list(calib_config["measurability"])
    n_markers = calib_config.get("NbMarkers", 1)
    if (
        not any(n in names for n in BASE_TPL)
        or calib_config.get("base_to_ref_frame") is not None
        or not all(meas[:3])
    ):
        return {}
    n = len(q)
    M = np.asarray(PEE, dtype=float).reshape(n_markers, sum(meas), n)
    P = np.transpose(M[:, :3], (0, 2, 1))  # (n_markers, n, 3)
    orient = n_markers == 1 and len(meas) == 6 and all(meas[3:6])
    if orient:
        R_meas = np.array([pin.rpy.rpyToMatrix(M[0, 3:6, i]) for i in range(n)])

    R, p = np.empty((n, 3, 3)), np.empty((n, 3))
    for i in range(n):
        pin.framesForwardKinematics(model, data, q[i])
        T = get_rel_transform(
            model, data, calib_config["start_frame"], calib_config["end_frame"]
        )
        R[i], p[i] = T.rotation, T.translation

    tip_pos = [
        any(f"{e}_{k + 1}" in names for e in EE_TPL[:3]) for k in range(n_markers)
    ]
    tip_rot = orient and any(f"{e}_1" in names for e in EE_TPL[3:])
    R_b, R_tip = np.eye(3), np.eye(3)
    t_b, t_tip = np.zeros(3), np.zeros((n_markers, 3))
    free = [k for k in range(n_markers) if tip_pos[k]]
    for _ in range(n_iter):
        if orient:
            R_b = _chordal_mean(R_meas @ np.transpose(R @ R_tip, (0, 2, 1)))
            if tip_rot:
                R_tip = _chordal_mean(np.transpose(R_b @ R, (0, 2, 1)) @ R_meas)
        else:
            source = np.concatenate([p + R @ t_tip[k] for k in range(n_markers)])
            R_b = _kabsch(source, P.reshape(-1, 3))
        # P_k,i - R_b p_i = R_b R_i t_tip_k + t_b, linear in (t_tip_k, t_b);
        # tips not estimated stay at 0
        rhs = (P - (p @ R_b.T)[None]).reshape(-1)
        A = np.zeros((n_markers, n, 3, 3 * len(free) + 3))
        for col, k in enumerate(free):
            A[k, :, :, 3 * col : 3 * col + 3] = R_b @ R
        A[:, :, :, -3:] = np.eye(3)
        sol = np.linalg.lstsq(A.reshape(-1, A.shape[-1]), rhs, rcond=None)[0]
        for col, k in enumerate(free):
            t_tip[k] = sol[3 * col : 3 * col + 3]
        t_b = sol[-3:]

    values = np.concatenate([t_b, pin.rpy.matrixToRpy(R_b)])
    guess = {n_: v for n_, v in zip(BASE_TPL, values) if n_ in names}
    for k in range(n_markers):
        tip = np.concatenate([t_tip[k], pin.rpy.matrixToRpy(R_tip)])
        for e, v in zip(EE_TPL, tip):
            if f"{e}_{k + 1}" in names:
                guess[f"{e}_{k + 1}"] = v
    return guess
```

## `measurement_jacobian(model, data, var, q, calib_config, step=1e-06)`

Central-difference Jacobian of `calc_updated_fkm` w.r.t. `var`.

Rows follow the flattened, DOF-major measurement vector; columns follow `calib_config["param_name"]`.

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def measurement_jacobian(model, data, var, q, calib_config, step=1e-6):
    """Central-difference Jacobian of ``calc_updated_fkm`` w.r.t. ``var``.

    Rows follow the flattened, DOF-major measurement vector; columns follow
    ``calib_config["param_name"]``.
    """
    var = np.asarray(var, dtype=float)
    cfg = dict(calib_config, NbSample=len(q))
    cols = []
    for j in range(len(var)):
        dv = np.zeros_like(var)
        dv[j] = step
        hi = calc_updated_fkm(model, data, var + dv, q, cfg)
        lo = calc_updated_fkm(model, data, var - dv, q, cfg)
        cols.append((hi - lo) / (2 * step))
    return np.column_stack(cols)
```

## `select_identifiable_parameters(jacobian, names, always_keep=(), tol=0.0001)`

Split parameters into identifiable and absorbed ones, deterministically.

Columns are normalised to unit length (so units do not matter) and taken in order: first `always_keep` (e.g. base and tip frames), then the rest of `names` in order. A column is kept when its component orthogonal to the columns already kept exceeds `tol`; otherwise it is a combination of them and is reported as absorbed. `always_keep` columns are never dropped.

Parameters:

| Name          | Type       | Description                                                                                                                                                                                                                                                      | Default    |
| ------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `jacobian`    | `ndarray`  | (n_meas, n_params) measurement Jacobian                                                                                                                                                                                                                          | *required* |
| `names`       | `list`     | Parameter names, one per column                                                                                                                                                                                                                                  | *required* |
| `always_keep` | `iterable` | Names kept unconditionally and tested first                                                                                                                                                                                                                      | `()`       |
| `tol`         | `float`    | Threshold on the orthogonal residual of a unit column. Exact dependencies give ~1e-10 (finite-difference noise); the default 1e-4 also drops near-dependencies whose column alone would have a condition number above 1e4 (a direction the data barely excites). | `0.0001`   |

Returns:

| Name    | Type | Description                                                      |
| ------- | ---- | ---------------------------------------------------------------- |
| `tuple` |      | (kept names in original order, absorbed names in original order) |

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def select_identifiable_parameters(jacobian, names, always_keep=(), tol=1e-4):
    """Split parameters into identifiable and absorbed ones, deterministically.

    Columns are normalised to unit length (so units do not matter) and taken
    in order: first ``always_keep`` (e.g. base and tip frames), then the rest
    of ``names`` in order. A column is kept when its component orthogonal to
    the columns already kept exceeds ``tol``; otherwise it is a combination of
    them and is reported as absorbed. ``always_keep`` columns are never
    dropped.

    Args:
        jacobian (ndarray): (n_meas, n_params) measurement Jacobian
        names (list): Parameter names, one per column
        always_keep (iterable): Names kept unconditionally and tested first
        tol (float): Threshold on the orthogonal residual of a unit column.
            Exact dependencies give ~1e-10 (finite-difference noise); the
            default 1e-4 also drops near-dependencies whose column alone
            would have a condition number above 1e4 (a direction the data
            barely excites).

    Returns:
        tuple: (kept names in original order, absorbed names in original order)
    """
    names = list(names)
    always_keep = [n for n in names if n in set(always_keep)]
    order = always_keep + [n for n in names if n not in set(always_keep)]
    basis = np.zeros((jacobian.shape[0], 0))
    kept = set()
    for name in order:
        col = jacobian[:, names.index(name)]
        norm = np.linalg.norm(col)
        if norm > 0:
            r = col / norm
            for _ in range(2):  # re-orthogonalise for numerical stability
                r = r - basis @ (basis.T @ r)
            res = np.linalg.norm(r)
        else:
            res = 0.0
        if name in always_keep or res > tol:
            kept.add(name)
            if res > tol:
                basis = np.column_stack([basis, r / res])
    return [n for n in names if n in kept], [n for n in names if n not in kept]
```

## `drop_calibration_parameters(calib_config, dropped)`

Remove parameters from `param_name`, keeping the base mapping aligned.

A dropped name inside `base_mapping_slice` also loses its row of `base_mapping_matrix` / `base_mapping_row_names`; names before the slice shift it left.

Source code in `src/figaroh/calibration/calibration_tools.py`

```
def drop_calibration_parameters(calib_config, dropped):
    """Remove parameters from ``param_name``, keeping the base mapping aligned.

    A dropped name inside ``base_mapping_slice`` also loses its row of
    ``base_mapping_matrix`` / ``base_mapping_row_names``; names before the
    slice shift it left.
    """
    names = list(calib_config["param_name"])
    for idx in sorted((names.index(n) for n in dropped), reverse=True):
        if "base_mapping_slice" in calib_config:
            start, end = calib_config["base_mapping_slice"]
            if start <= idx < end:
                row = idx - start
                calib_config["base_mapping_matrix"] = np.delete(
                    calib_config["base_mapping_matrix"], row, axis=0
                )
                rows = list(calib_config["base_mapping_row_names"])
                del rows[row]
                calib_config["base_mapping_row_names"] = rows
                calib_config["base_mapping_slice"] = (start, end - 1)
            elif idx < start:
                calib_config["base_mapping_slice"] = (start - 1, end - 1)
        del names[idx]
    calib_config["param_name"] = names
```

## estimation

Parameter selection and estimation methods; see [Calibration estimation methods](https://thanhndv212.github.io/figaroh-plus/concepts/calibration_estimation/index.md).

Choosing which calibration parameters to estimate, and how (#113).

`calib_config["estimation"]["method"]` selects one of:

`structural` (default) The structural base parameters from random configurations, then the data-level elimination of frame-absorbed parameters (#102). Unchanged behaviour. `excitation` Start from every joint parameter of the calibration level, drop exact dependencies on the measured postures, then remove, one at a time, the parameter whose predicted standard error is largest relative to its expected size, while that ratio exceeds `excitation_k`. Kept parameters are estimated freely. `map` Estimate every joint parameter with a zero-mean Gaussian prior of the given expected sizes (`priors`). Weakly excited directions stay near nominal; nothing is dropped. `map_cv` `map` with the prior sizes scaled by a factor chosen by k-fold cross-validation over the training postures (optionally one factor per parameter group). Needs no robot-specific sizes. `cv_subset` Nested parameter sets in the `excitation` removal order; the set size is chosen by k-fold cross-validation over the training postures.

The base and tool frames are always estimated freely (no prior). Noise is `noise_std` in the units of the cost function's measurement residuals (metres for position unless the robot's cost function weights them), or, when `None`, estimated from a free fit of the identifiable parameters. Selection runs once, in :meth:`BaseCalibration.create_param_list`; the choices and diagnostics are stored in `calib_config["estimation_report"]`. See the guide `docs/source/concepts/calibration_estimation.md`.

## `EstimationError`

Bases: `ValueError`

Invalid estimation settings or data for the chosen method.

## `settings(calib_config)`

`calib_config["estimation"]` merged over :data:`DEFAULTS`.

Source code in `src/figaroh/calibration/estimation.py`

```
def settings(calib_config: dict) -> dict:
    """``calib_config["estimation"]`` merged over :data:`DEFAULTS`."""
    user = dict(calib_config.get("estimation") or {})
    out = dict(DEFAULTS, **user)
    out["priors"] = dict(DEFAULT_PRIORS, **(user.get("priors") or {}))
    if out["method"] not in METHODS:
        raise EstimationError(
            f"estimation.method '{out['method']}' not in {', '.join(METHODS)}"
        )
    unknown = set(out["priors"]) - set(GROUPS)
    if unknown:
        raise EstimationError(f"unknown prior groups {sorted(unknown)}")
    if out["cv_rule"] not in ("min", "one_se"):
        raise EstimationError("estimation.cv_rule must be 'min' or 'one_se'")
    return out
```

## `joint_candidates(model, calib_config)`

Every joint parameter of the calibration level, in joint order.

Source code in `src/figaroh/calibration/estimation.py`

```
def joint_candidates(model, calib_config: dict) -> List[str]:
    """Every joint parameter of the calibration level, in joint order."""
    act = [model.names[j] for j in calib_config["actJoint_idx"]]
    if calib_config["calib_model"] == "full_params":
        return list(get_fullparam_offset(act))
    names = list(get_joint_offset(model, act))  # covers every model joint
    by_joint = {n.split("_", 1)[1]: n for n in names}
    return [by_joint[j] for j in act if j in by_joint]
```

## `prior_group(name, model)`

Prior group of a parameter name; `None` for frame parameters.

Source code in `src/figaroh/calibration/estimation.py`

```
def prior_group(name: str, model) -> Optional[str]:
    """Prior group of a parameter name; ``None`` for frame parameters."""
    if name.startswith("base_") or "EE" in name:
        return None
    if name.startswith("d_"):
        _, kind, joint = name.split("_", 2)
        axis = _joint_axis(model, joint)
        if axis == ("R", kind[-1]) and kind.startswith("phi"):
            return "joint_offset"
        if axis == ("P", kind[-1]) and kind.startswith("p"):
            return "prismatic_offset"
        return "rotation" if kind.startswith("phi") else "translation"
    if name.startswith("offsetP"):
        return "prismatic_offset"
    return "joint_offset"
```

## `prior_std(names, model, priors)`

Expected size per parameter; `inf` for frame parameters.

Source code in `src/figaroh/calibration/estimation.py`

```
def prior_std(names: Sequence[str], model, priors: Dict[str, float]) -> np.ndarray:
    """Expected size per parameter; ``inf`` for frame parameters."""
    out = []
    for n in names:
        group = prior_group(n, model)
        out.append(np.inf if group is None else float(priors[group]))
    return np.array(out)
```

## `identifiable(calib, names, frames, tol)`

Drop exact dependencies on the measured postures (frames kept).

Source code in `src/figaroh/calibration/estimation.py`

```
def identifiable(calib, names, frames, tol) -> List[str]:
    """Drop exact dependencies on the measured postures (frames kept)."""
    cfg = calib.calib_config
    x0 = _start(calib, names)
    J = measurement_jacobian(
        calib.model, calib.data, x0, calib.q_measured, dict(cfg, param_name=names)
    )
    kept, _ = select_identifiable_parameters(J, names, frames, tol=tol)
    return kept
```

## `excitation_order(J, names, frames, prior, noise, k=None)`

Backward elimination by predicted standard error / expected size.

Returns (kept, removed in order, ratio at removal). With `k` None the elimination runs until only frames remain.

Source code in `src/figaroh/calibration/estimation.py`

```
def excitation_order(J, names, frames, prior, noise, k=None):
    """Backward elimination by predicted standard error / expected size.

    Returns (kept, removed in order, ratio at removal). With ``k`` None the
    elimination runs until only frames remain.
    """
    active = list(names)
    removed, ratios = [], []
    while True:
        cols = [names.index(n) for n in active]
        Ja = J[:, cols]
        cov = noise**2 * np.linalg.inv(Ja.T @ Ja)
        ratio = np.sqrt(np.abs(np.diag(cov))) / prior[cols]
        joint = [i for i, n in enumerate(active) if n not in frames]
        if not joint:
            break
        worst = max(joint, key=lambda i: (ratio[i], -i))
        if k is not None and ratio[worst] <= k:
            break
        removed.append(active[worst])
        ratios.append(float(ratio[worst]))
        active.pop(worst)
    return active, removed, ratios
```

## `configure(calib)`

Set `param_name` and prior weights for a non-structural method.

Source code in `src/figaroh/calibration/estimation.py`

```
def configure(calib) -> dict:
    """Set ``param_name`` and prior weights for a non-structural method."""
    cfg = calib.calib_config
    est = settings(cfg)
    if cfg.get("non_geom"):
        raise EstimationError(
            f"estimation.method '{est['method']}' does not support "
            "include_non_geometric; use 'structural'"
        )
    if not (hasattr(calib, "q_measured") and hasattr(calib, "PEE_measured")):
        raise EstimationError(
            f"estimation.method '{est['method']}' selects on the measured "
            "postures: load the data before create_param_list()"
        )
    model = calib.model
    candidates = joint_candidates(model, cfg)
    base = [] if cfg["known_baseframe"] else list(BASE_TPL)
    cfg["param_name"] = base + candidates
    if not cfg["known_tipframe"]:
        add_pee_name(cfg)
    all_names = list(cfg["param_name"])
    for key in _BASE_MAPPING_KEYS:
        cfg.pop(key, None)
    frames = calib._frame_param_names()
    prior = prior_std(all_names, model, est["priors"])
    prior_of = dict(zip(all_names, prior))

    ident = identifiable(calib, all_names, frames, est["dependency_tol"])
    prelim = _fit(calib, ident)
    n_meas = len(calib.PEE_measured)
    noise, noise_source = est["noise_std"], "given"
    if noise is None:
        dof = n_meas - len(ident)
        if dof <= 0:
            raise EstimationError(
                "too few measurements to estimate the noise; set estimation.noise_std"
            )
        noise = float(np.sqrt(np.sum(prelim.fun[:n_meas] ** 2) / dof))
        noise_source = "estimated (free fit of the identifiable parameters)"
    report = {
        "method": est["method"],
        "candidates": candidates,
        "dependent": [n for n in all_names if n not in ident],
        "noise_std": noise,
        "noise_source": noise_source,
        "priors": est["priors"],
    }

    def weights(names, scale=None):
        scale = scale or {}
        s = np.array(
            [prior_of[n] * scale.get(prior_group(n, model), 1.0) for n in names]
        )
        return np.where(np.isfinite(s), noise / s, 0.0)

    method = est["method"]
    final, w = None, None
    if method in ("excitation", "cv_subset"):
        J = prelim.jac[:n_meas]
        prior_ident = np.array([prior_of[n] for n in ident])
        k = est["excitation_k"] if method == "excitation" else None
        kept, removed, ratios = excitation_order(
            J, ident, frames, prior_ident, noise, k
        )
        report["removal_order"] = list(zip(removed, ratios))
        if method == "excitation":
            report["excitation_k"] = est["excitation_k"]
            final = kept
        else:
            final = _choose_subset(calib, ident, removed, est, report)
    elif method == "map":
        final, w = all_names, weights(all_names)
    elif method == "map_cv":
        scale = _choose_scale(calib, all_names, weights, est, report)
        report["prior_scale"] = scale
        final, w = all_names, weights(all_names, scale)

    cfg["param_name"] = _canonical(all_names, final)
    cfg["prior_weights"] = None if w is None else w
    cfg["prior_noise"] = None if w is None else noise
    cfg["absorbed_param_name"] = report["dependent"]
    report["param_name"] = list(cfg["param_name"])
    cfg["estimation_report"] = report
    calib.estimation_report = report
    logger.info(
        "estimation '%s': %d parameters (noise %.3g, %s)",
        method,
        len(cfg["param_name"]),
        noise,
        noise_source,
    )
    return report
```

## `lift_structural(calib, priors=None, refine_iterations=3)`

Weighted minimum-norm lift of fitted base parameters to joint corrections.

The `structural` method estimates one representative per group of dependent joint parameters; which one is a numerical accident. Among all joint corrections that reproduce the fit, return the most plausible under the expected sizes `priors` (default: the estimation priors):

```
theta = argmin theta' P^-1 theta  subject to  M_full theta = t
```

`M_full` is the base mapping before any row was dropped; `t` holds the fitted value of every kept joint row, and 0 on the rows the base frame carries (`base_frame_row_names`) and on the rows the fit dropped (held at 0). The result does not depend on which representatives were chosen.

`M` is a first-order (structural) map, so with corrections of tens of mrad the lifted model drifts from the fit by up to ~0.1 mm. When the measured postures are available, `refine_iterations` Gauss-Newton steps then make the lifted model reproduce the fitted predictions at those postures, each step taking the smallest correction in the same prior-weighted norm.

Returns:

| Type                  | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `(names, theta, cov)` | joint parameter names                                           |
|                       | (base_mapping_param_names), lifted values, and their covariance |
|                       | conditional on this lift (propagated from the fitted base       |
|                       | parameters; directions the data does not see have zero variance |
|                       | here, not their prior).                                         |

Source code in `src/figaroh/calibration/estimation.py`

```
def lift_structural(
    calib, priors: Optional[Dict[str, float]] = None, refine_iterations: int = 3
):
    """Weighted minimum-norm lift of fitted base parameters to joint corrections.

    The ``structural`` method estimates one representative per group of
    dependent joint parameters; which one is a numerical accident. Among all
    joint corrections that reproduce the fit, return the most plausible
    under the expected sizes ``priors`` (default: the estimation priors):

        theta = argmin theta' P^-1 theta  subject to  M_full theta = t

    ``M_full`` is the base mapping before any row was dropped; ``t`` holds
    the fitted value of every kept joint row, and 0 on the rows the base
    frame carries (``base_frame_row_names``) and on the rows the fit
    dropped (held at 0). The result does not depend on which
    representatives were chosen.

    ``M`` is a first-order (structural) map, so with corrections of tens of
    mrad the lifted model drifts from the fit by up to ~0.1 mm. When the
    measured postures are available, ``refine_iterations`` Gauss-Newton
    steps then make the lifted model reproduce the fitted predictions at
    those postures, each step taking the smallest correction in the same
    prior-weighted norm.

    Returns:
        (names, theta, cov): joint parameter names
        (``base_mapping_param_names``), lifted values, and their covariance
        conditional on this lift (propagated from the fitted base
        parameters; directions the data does not see have zero variance
        here, not their prior).
    """
    cfg = calib.calib_config
    M = np.asarray(cfg.get("base_mapping_matrix_full", cfg["base_mapping_matrix"]))
    rows_full = list(
        cfg.get("base_mapping_row_names_full", cfg["base_mapping_row_names"])
    )
    kept = list(cfg["base_mapping_row_names"])
    frame_rows = set(cfg.get("base_frame_row_names", []))
    names = list(cfg["base_mapping_param_names"])
    start, _ = cfg["base_mapping_slice"]
    x = np.asarray(calib.var_)
    C = np.asarray(calib._C_param)

    free = [
        (i, start + kept.index(r))
        for i, r in enumerate(rows_full)
        if r in kept and r not in frame_rows
    ]
    t = np.zeros(len(rows_full))
    for i, k in free:
        t[i] = x[k]

    sizes = (
        settings(cfg)["priors"] if priors is None else dict(DEFAULT_PRIORS, **priors)
    )
    P = prior_std(names, getattr(calib, "model", None), sizes) ** 2
    G = (P[:, None] * M.T) @ np.linalg.inv((M * P) @ M.T)
    theta = G @ t
    if refine_iterations and hasattr(calib, "q_measured"):
        frames = set(calib._frame_param_names())
        other = [
            n
            for n in cfg["param_name"]
            if n not in frames and not n.startswith(("d_", "offset"))
        ]
        if other:
            # e.g. contact-plane parameters: core's FK cannot evaluate the fit
            logger.info(
                "lift not refined (first order only): parameters %s are not "
                "joint or frame parameters",
                other[:3],
            )
        else:
            theta = _refine_lift(calib, names, theta, np.sqrt(P), refine_iterations)
    rows_i = [i for i, _ in free]
    var_k = [k for _, k in free]
    Gs = G[:, rows_i]
    cov = Gs @ C[np.ix_(var_k, var_k)] @ Gs.T
    return names, theta, cov
```

## config

Configuration parsing and parameter management for robot calibration.

This module handles all configuration-related functionality including:

- YAML configuration file parsing
- Unified to legacy config format conversion
- Parameter extraction and validation
- Frame and joint configuration management

## `regularization_coefficient(value)`

`coeff_regularize` from a config value; a non-zero value is an error.

The coefficient weighted `sqrt(c) * theta` rows that robot cost functions appended for every joint parameter, so a metre and a radian were penalised alike (#120). It was deprecated in 0.6.0 and removed in 1.0: `estimation.method: map` replaces it with priors of physical size per parameter group. Zero or absent is accepted and passed through.

Raises:

| Type                 | Description        |
| -------------------- | ------------------ |
| `ConfigurationError` | value is non-zero. |

Source code in `src/figaroh/calibration/config.py`

```
def regularization_coefficient(value):
    """``coeff_regularize`` from a config value; a non-zero value is an error.

    The coefficient weighted ``sqrt(c) * theta`` rows that robot cost
    functions appended for every joint parameter, so a metre and a radian
    were penalised alike (#120). It was deprecated in 0.6.0 and removed in
    1.0: ``estimation.method: map`` replaces it with priors of physical size
    per parameter group. Zero or absent is accepted and passed through.

    Raises:
        ConfigurationError: ``value`` is non-zero.
    """
    if value:
        raise ConfigurationError(
            f"regularization_coefficient ({value}) was removed in 1.0: it "
            "penalised metres and radians alike. Remove it (or set it to 0) "
            "and use estimation.method: map (priors per parameter group, "
            "docs/source/concepts/calibration_estimation.md; figaroh-plus#120)."
        )
    return value
```

## `get_sup_joints(model, start_frame, end_frame)`

Get list of supporting joints between two frames in kinematic chain.

Finds all joints that contribute to relative motion between start_frame and end_frame by analyzing their support branches in the kinematic tree.

Parameters:

| Name          | Type    | Description            | Default    |
| ------------- | ------- | ---------------------- | ---------- |
| `model`       | `Model` | Robot model            | *required* |
| `start_frame` | `str`   | Name of starting frame | *required* |
| `end_frame`   | `str`   | Name of ending frame   | *required* |

Returns:

| Type | Description                                                                                                                                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      | list\[int\]: Joint IDs ordered from start to end frame, handling cases: 1. Branch entirely contained in another branch 2. Disjoint branches with fixed root 3. Partially overlapping branches |

Raises:

| Type             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `AssertionError` | If end frame appears before start frame in chain |

Note

Excludes "universe" joints from returned list since they don't contribute to relative motion.

Source code in `src/figaroh/calibration/config.py`

```
def get_sup_joints(model, start_frame, end_frame):
    """Get list of supporting joints between two frames in kinematic chain.

    Finds all joints that contribute to relative motion between start_frame and
    end_frame by analyzing their support branches in the kinematic tree.

    Args:
        model (pin.Model): Robot model
        start_frame (str): Name of starting frame
        end_frame (str): Name of ending frame

    Returns:
        list[int]: Joint IDs ordered from start to end frame, handling cases:
            1. Branch entirely contained in another branch
            2. Disjoint branches with fixed root
            3. Partially overlapping branches

    Raises:
        AssertionError: If end frame appears before start frame in chain

    Note:
        Excludes "universe" joints from returned list since they don't
        contribute to relative motion.
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)
    start_par = model.frames[start_frameId].parentJoint
    end_par = model.frames[end_frameId].parentJoint
    branch_s = model.supports[start_par].tolist()
    branch_e = model.supports[end_par].tolist()
    # remove 'universe' joint from branches
    if model.names[branch_s[0]] == "universe":
        branch_s.remove(branch_s[0])
    if model.names[branch_e[0]] == "universe":
        branch_e.remove(branch_e[0])

    # find over-lapping joints in two branches
    shared_joints = list(set(branch_s) & set(branch_e))
    # create a list of supporting joints between two frames
    list_1 = [x for x in branch_s if x not in branch_e]
    list_1.reverse()
    list_2 = [y for y in branch_e if y not in branch_s]
    # case 2: root_joint is fixed; branch_s and branch_e are separate
    if shared_joints == []:
        sup_joints = list_1 + list_2
    else:
        # case 1: branch_s is part of branch_e
        if shared_joints == branch_s:
            sup_joints = [branch_s[-1]] + list_2
        else:
            assert shared_joints != branch_e, "End frame should be before start frame."
            # case 3: there are overlapping joints between two branches
            sup_joints = list_1 + [shared_joints[-1]] + list_2
    return sup_joints
```

## `unified_to_legacy_config(robot, unified_calib_config)`

Convert unified configuration format to legacy calib_config format.

Maps the unified configuration structure to the `calib_config` dict the calibration classes consume.

Parameters:

| Name                   | Type           | Description                              | Default    |
| ---------------------- | -------------- | ---------------------------------------- | ---------- |
| `robot`                | `RobotWrapper` | Robot instance containing model and data | *required* |
| `unified_calib_config` | `dict`         | Configuration from create_task_config    | *required* |

Returns:

| Name   | Type   | Description                                   |
| ------ | ------ | --------------------------------------------- |
| `dict` | `dict` | Calibration configuration dict (calib_config) |

Raises:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `KeyError`       | If required fields are missing from unified config |
| `AssertionError` | If frame validation fails                          |

Example

> > > unified_config = create_task_config(robot, parsed_config, ... "calibration") legacy_config = unified_to_legacy_config(robot, unified_config)

Source code in `src/figaroh/calibration/config.py`

```
def unified_to_legacy_config(robot, unified_calib_config) -> dict:
    """Convert unified configuration format to legacy calib_config format.

    Maps the unified configuration structure to the ``calib_config`` dict the
    calibration classes consume.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_calib_config (dict): Configuration from create_task_config

    Returns:
        dict: Calibration configuration dict (``calib_config``)

    Raises:
        KeyError: If required fields are missing from unified config
        AssertionError: If frame validation fails

    Example:
        >>> unified_config = create_task_config(robot, parsed_config,
        ...                                    "calibration")
        >>> legacy_config = unified_to_legacy_config(robot, unified_config)
    """
    # Initialize output configuration
    calib_config = {}

    # Extract unified config sections
    joints = unified_calib_config.get("joints", {})
    kinematics = unified_calib_config.get("kinematics", {})
    parameters = unified_calib_config.get("parameters", {})
    measurements = unified_calib_config.get("measurements", {})
    data = unified_calib_config.get("data", {})

    # 1. Extract basic robot information
    calib_config["robot_name"] = robot.model.name
    calib_config["q0"] = robot.q0
    calib_config["instance"] = unified_calib_config.get("instance", {})

    # 2. Extract and validate markers/measurements
    _extract_marker_info(calib_config, measurements)

    # 3. Extract and validate kinematic frames
    _extract_frame_info(calib_config, robot, kinematics)

    # 4. Extract tool frame information
    _extract_tool_info(calib_config, robot, calib_config["end_frame"])

    # 5. Determine active joints
    _determine_active_joints(
        calib_config,
        robot,
        joints,
        calib_config["start_frame"],
        calib_config["end_frame"],
    )

    # 6. Extract poses
    _extract_poses(calib_config, measurements)

    # 7. Extract calibration parameters
    _extract_calibration_params(calib_config, robot, parameters)

    # 7b. Extract eye-hand (camera) calibration frames, if enabled
    _extract_eye_hand_params(calib_config, unified_calib_config)

    # 8. Extract data configuration
    calib_config["NbSample"] = data.get("number_of_samples", 500)
    calib_config["data_file"] = data.get("source_file")
    # held-out measurements in the same format; "" means none
    calib_config["validation_data_file"] = data.get("validation_data_file") or None
    calib_config["sample_configs_file"] = data.get("sample_configurations_file")

    return calib_config
```

## parameter

Parameter management utilities for robot calibration.

This module contains functions for creating and managing calibration parameter dictionaries, including:

- Joint offset parameters
- Geometric parameter offsets
- Base frame parameters
- End-effector marker parameters
- Frame management utilities

## `get_joint_offset(model, joint_names)`

Get dictionary of joint offset parameters.

Maps joint names to their offset parameters, handling special cases for different joint types and multiple DOF joints.

Parameters:

| Name          | Type | Description                          | Default    |
| ------------- | ---- | ------------------------------------ | ---------- |
| `model`       |      | Pinocchio robot model                | *required* |
| `joint_names` |      | List of joint names from model.names | *required* |

Returns:

| Name   | Type | Description                                                                                                     |
| ------ | ---- | --------------------------------------------------------------------------------------------------------------- |
| `dict` |      | Mapping of joint offset parameter names to initial zero values. Keys have format: "{offset_type}\_{joint_name}" |

Example

> > > offsets = get_joint_offset(robot.model, robot.model.names[1:]) print(offsets["offsetRZ_joint1"]) 0.0

Source code in `src/figaroh/calibration/parameter.py`

```
def get_joint_offset(model, joint_names):
    """Get dictionary of joint offset parameters.

    Maps joint names to their offset parameters, handling special cases for
    different joint types and multiple DOF joints.

    Args:
        model: Pinocchio robot model
        joint_names: List of joint names from model.names

    Returns:
        dict: Mapping of joint offset parameter names to initial zero values.
            Keys have format: "{offset_type}_{joint_name}"

    Example:
        >>> offsets = get_joint_offset(robot.model, robot.model.names[1:])
        >>> print(offsets["offsetRZ_joint1"])
        0.0
    """
    joint_off = []
    joint_names = list(model.names[1:])
    joints = list(model.joints[1:])
    assert len(joint_names) == len(
        joints
    ), "Number of jointnames does not match number of joints! Please check\
        imported model."
    for id, joint in enumerate(joints):
        name = joint_names[id]
        shortname = joint.shortname()
        if model.name == "canopies":
            if "RevoluteUnaligned" in shortname:
                shortname = shortname.replace("RevoluteUnaligned", "RZ")
        for i in range(joint.nv):
            if i > 0:
                offset_param = (
                    shortname.replace("JointModel", "offset")
                    + "{}".format(i + 1)
                    + "_"
                    + name
                )
            else:
                offset_param = shortname.replace("JointModel", "offset") + "_" + name
            joint_off.append(offset_param)

    phi_jo = [0] * len(joint_off)  # default zero values
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off
```

## `get_fullparam_offset(joint_names)`

Get dictionary of geometric parameter variations.

Creates mapping of geometric offset parameters for each joint's position and orientation.

Parameters:

| Name          | Type | Description                          | Default    |
| ------------- | ---- | ------------------------------------ | ---------- |
| `joint_names` |      | List of joint names from robot model | *required* |

Returns:

| Name   | Type | Description                                                                                                                                                                                      |
| ------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dict` |      | Mapping of geometric parameter names to initial zero values. Keys have format: "d\_{param}\_{joint_name}" where param is: - px, py, pz: Position offsets - phix, phiy, phiz: Orientation offsets |

Example

> > > geo_params = get_fullparam_offset(robot.model.names[1:]) print(geo_params["d_px_joint1"]) 0.0

Source code in `src/figaroh/calibration/parameter.py`

```
def get_fullparam_offset(joint_names):
    """Get dictionary of geometric parameter variations.

    Creates mapping of geometric offset parameters for each joint's
    position and orientation.

    Args:
        joint_names: List of joint names from robot model

    Returns:
        dict: Mapping of geometric parameter names to initial zero values.
            Keys have format: "d_{param}_{joint_name}" where param is:
            - px, py, pz: Position offsets
            - phix, phiy, phiz: Orientation offsets

    Example:
        >>> geo_params = get_fullparam_offset(robot.model.names[1:])
        >>> print(geo_params["d_px_joint1"])
        0.0
    """
    geo_params = []

    for i in range(len(joint_names)):
        for j in FULL_PARAMTPL:
            # geo_params.append(j + ("_%d" % i))
            geo_params.append(j + "_" + joint_names[i])

    phi_gp = [0] * len(geo_params)  # default zero values
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params
```

## `add_base_name(calib_config)`

Add base frame parameters to parameter list.

Updates calib_config["param_name"] with base frame parameters depending on calibration model type.

Parameters:

| Name           | Type | Description                                                                                                                     | Default    |
| -------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `calib_config` |      | Parameter dictionary containing: - calib_model: "full_params" or "joint_offset" - param_name: List of parameter names to update | *required* |

Side Effects

Modifies calib_config["param_name"] in place by:

- For full_params: Replaces first 6 entries with base parameters
- For joint_offset: Prepends base parameters to list

Source code in `src/figaroh/calibration/parameter.py`

```
def add_base_name(calib_config):
    """Add base frame parameters to parameter list.

    Updates calib_config["param_name"] with base frame parameters depending on
    calibration model type.

    Args:
        calib_config: Parameter dictionary containing:
            - calib_model: "full_params" or "joint_offset"
            - param_name: List of parameter names to update

    Side Effects:
        Modifies calib_config["param_name"] in place by:
        - For full_params: Replaces first 6 entries with base parameters
        - For joint_offset: Prepends base parameters to list
    """
    if calib_config["calib_model"] == "full_params":
        calib_config["param_name"][0:6] = BASE_TPL
    elif calib_config["calib_model"] == "joint_offset":
        calib_config["param_name"] = BASE_TPL + calib_config["param_name"]
        # base_mapping_slice is positional: prepending shifts it
        if "base_mapping_slice" in calib_config:
            start, end = calib_config["base_mapping_slice"]
            calib_config["base_mapping_slice"] = (
                start + len(BASE_TPL),
                end + len(BASE_TPL),
            )
```

## `add_pee_name(calib_config)`

Add end-effector marker parameters to parameter list.

Adds parameters for each active measurement DOF of each marker.

Parameters:

| Name           | Type | Description                                                                                                                                                       | Default    |
| -------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `calib_config` |      | Parameter dictionary containing: - NbMarkers: Number of markers - measurability: List of booleans for active DOFs - param_name: List of parameter names to update | *required* |

Side Effects

Modifies calib_config["param_name"] in place by appending marker parameters in format: "{param_type}\_{marker_num}"

Source code in `src/figaroh/calibration/parameter.py`

```
def add_pee_name(calib_config):
    """Add end-effector marker parameters to parameter list.

    Adds parameters for each active measurement DOF of each marker.

    Args:
        calib_config: Parameter dictionary containing:
            - NbMarkers: Number of markers
            - measurability: List of booleans for active DOFs
            - param_name: List of parameter names to update

    Side Effects:
        Modifies calib_config["param_name"] in place by appending marker
        parameters in format: "{param_type}_{marker_num}"
    """
    PEE_names = []
    for i in range(calib_config["NbMarkers"]):
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                PEE_names.extend(["{}_{}".format(EE_TPL[j], i + 1)])
    calib_config["param_name"] = calib_config["param_name"] + PEE_names
```

## `add_eemarker_frame(frame_name, p, rpy, model, data)`

Add a new frame attached to the end-effector.

Creates and adds a fixed frame to the robot model at the end-effector location, typically used for marker or tool frames.

Parameters:

| Name         | Type      | Description                                 | Default    |
| ------------ | --------- | ------------------------------------------- | ---------- |
| `frame_name` | `str`     | Name for the new frame                      | *required* |
| `p`          | `ndarray` | 3D position offset from parent frame        | *required* |
| `rpy`        | `ndarray` | Roll-pitch-yaw angles for frame orientation | *required* |
| `model`      | `Model`   | Robot model to add frame to                 | *required* |
| `data`       | `Data`    | Robot data structure                        | *required* |

Returns:

| Name  | Type | Description               |
| ----- | ---- | ------------------------- |
| `int` |      | ID of newly created frame |

Note

Currently hardcoded to attach to "arm_7_joint". This should be made configurable in future versions.

Source code in `src/figaroh/calibration/parameter.py`

```
def add_eemarker_frame(frame_name, p, rpy, model, data):
    """Add a new frame attached to the end-effector.

    Creates and adds a fixed frame to the robot model at the end-effector
    location, typically used for marker or tool frames.

    Args:
        frame_name (str): Name for the new frame
        p (ndarray): 3D position offset from parent frame
        rpy (ndarray): Roll-pitch-yaw angles for frame orientation
        model (pin.Model): Robot model to add frame to
        data (pin.Data): Robot data structure

    Returns:
        int: ID of newly created frame

    Note:
        Currently hardcoded to attach to "arm_7_joint". This should be made
        configurable in future versions.
    """
    p = np.array([0.1, 0.1, 0.1])
    R = pin.rpy.rpyToMatrix(rpy)
    frame_placement = pin.SE3(R, p)

    parent_jointId = model.getJointId("arm_7_joint")
    prev_frameId = model.getFrameId("arm_7_joint")
    ee_frame_id = model.addFrame(
        pin.Frame(
            frame_name,
            parent_jointId,
            prev_frameId,
            frame_placement,
            pin.FrameType(0),
            pin.Inertia.Zero(),
        ),
        False,
    )
    return ee_frame_id
```

## data_loader

Data loading and processing utilities for robot calibration.

This module provides functions for loading and processing calibration data from various file formats, including:

- CSV file reading for joint configurations
- Marker position/orientation data loading
- Data validation and cleanup
- Configuration vector management

## `get_idxq_from_jname(model, joint_name)`

Get index of joint in configuration vector.

Parameters:

| Name         | Type    | Description                     | Default    |
| ------------ | ------- | ------------------------------- | ---------- |
| `model`      | `Model` | Robot model                     | *required* |
| `joint_name` | `str`   | Name of joint to find index for | *required* |

Returns:

| Name  | Type | Description                              |
| ----- | ---- | ---------------------------------------- |
| `int` |      | Index of joint in configuration vector q |

Raises:

| Type             | Description                           |
| ---------------- | ------------------------------------- |
| `AssertionError` | If joint name does not exist in model |

Source code in `src/figaroh/calibration/data_loader.py`

```
def get_idxq_from_jname(model, joint_name):
    """Get index of joint in configuration vector.

    Args:
        model (pin.Model): Robot model
        joint_name (str): Name of joint to find index for

    Returns:
        int: Index of joint in configuration vector q

    Raises:
        AssertionError: If joint name does not exist in model
    """
    assert joint_name in model.names, "Given joint name does not exist."
    jointId = model.getJointId(joint_name)
    joint_idx = model.joints[jointId].idx_q
    return joint_idx
```

## `read_config_data(model, path_to_file)`

Read joint configurations from CSV file.

Parameters:

| Name           | Type    | Description                                      | Default    |
| -------------- | ------- | ------------------------------------------------ | ---------- |
| `model`        | `Model` | Robot model containing joint information         | *required* |
| `path_to_file` | `str`   | Path to CSV file containing joint configurations | *required* |

Returns:

| Name      | Type | Description                                                        |
| --------- | ---- | ------------------------------------------------------------------ |
| `ndarray` |      | Matrix of shape (n_samples, n_joints-1) containing joint positions |

Source code in `src/figaroh/calibration/data_loader.py`

```
def read_config_data(model, path_to_file):
    """Read joint configurations from CSV file.

    Args:
        model (pin.Model): Robot model containing joint information
        path_to_file (str): Path to CSV file containing joint configurations

    Returns:
        ndarray: Matrix of shape (n_samples, n_joints-1) containing joint
            positions
    """
    df = pd.read_csv(path_to_file)
    q = np.zeros([len(df), model.njoints - 1])
    for i in range(len(df)):
        for j, name in enumerate(model.names[1:].tolist()):
            jointidx = get_idxq_from_jname(model, name)
            q[i, jointidx] = df[name][i]
    return q
```

## `load_data(path_to_file, model, calib_config, del_list=[])`

Load joint configuration and marker data from CSV file.

Reads marker positions/orientations and joint configurations from a CSV file. Handles data validation, bad sample removal, and conversion to numpy arrays.

Parameters:

| Name           | Type    | Description                                                                                                                                                                                                                                                 | Default    |
| -------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `path_to_file` | `str`   | Path to CSV file containing recorded data                                                                                                                                                                                                                   | *required* |
| `model`        | `Model` | Robot model containing joint information                                                                                                                                                                                                                    | *required* |
| `calib_config` | `dict`  | Parameter dictionary containing: - NbMarkers: Number of markers to load - measurability: List indicating which DOFs are measured - actJoint_idx: List of active joint indices - config_idx: Configuration vector indices - q0: Default configuration vector | *required* |
| `del_list`     | `list`  | Indices of bad samples to remove. Defaults to [].                                                                                                                                                                                                           | `[]`       |

Returns:

| Name    | Type | Description                                                                                                                                      |
| ------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tuple` |      | PEEm_exp (ndarray): Flattened marker measurements of shape (n_active_dofs,) q_exp (ndarray): Joint configurations of shape (n_samples, n_joints) |

Note

CSV file must contain columns:

- For each marker i: [xi, yi, zi, phixi, phiyi, phizi]
- Joint names matching model.names for active joints

Raises:

| Type       | Description                              |
| ---------- | ---------------------------------------- |
| `KeyError` | If required columns are missing from CSV |

Side Effects

- Prints joint headers
- Updates calib_config["NbSample"] with number of valid samples

Source code in `src/figaroh/calibration/data_loader.py`

```
def load_data(path_to_file, model, calib_config, del_list=[]):
    """Load joint configuration and marker data from CSV file.

    Reads marker positions/orientations and joint configurations from a CSV
    file. Handles data validation, bad sample removal, and conversion to
    numpy arrays.

    Args:
        path_to_file (str): Path to CSV file containing recorded data
        model (pin.Model): Robot model containing joint information
        calib_config (dict): Parameter dictionary containing:
            - NbMarkers: Number of markers to load
            - measurability: List indicating which DOFs are measured
            - actJoint_idx: List of active joint indices
            - config_idx: Configuration vector indices
            - q0: Default configuration vector
        del_list (list, optional): Indices of bad samples to remove.
            Defaults to [].

    Returns:
        tuple:
            - PEEm_exp (ndarray): Flattened marker measurements of shape
                (n_active_dofs,)
            - q_exp (ndarray): Joint configurations of shape
                (n_samples, n_joints)

    Note:
        CSV file must contain columns:
        - For each marker i: [xi, yi, zi, phixi, phiyi, phizi]
        - Joint names matching model.names for active joints

    Raises:
        KeyError: If required columns are missing from CSV

    Side Effects:
        - Prints joint headers
        - Updates calib_config["NbSample"] with number of valid samples
    """
    # read_csv
    df = pd.read_csv(path_to_file)

    # create headers for marker position
    PEE_headers = []
    pee_tpl = ["x", "y", "z", "phix", "phiy", "phiz"]
    for i in range(calib_config["NbMarkers"]):
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                PEE_headers.extend(["{}{}".format(pee_tpl[j], i + 1)])

    # create headers for joint configurations
    joint_headers = [model.names[i] for i in calib_config["actJoint_idx"]]

    # check if all created headers present in csv file
    csv_headers = list(df.columns)
    for header in PEE_headers + joint_headers:
        if header not in csv_headers:
            logger.warning("%s does not exist in the file.", header)
            break

    # Extract marker position/location
    pose_ee = df[PEE_headers].to_numpy()

    # Extract joint configurations
    q_act = df[joint_headers].to_numpy()

    # remove bad data
    if del_list:
        pose_ee = np.delete(pose_ee, del_list, axis=0)
        q_act = np.delete(q_act, del_list, axis=0)

    # a missing (occluded) point would make every residual NaN: name the rows
    bad = ~np.isfinite(pose_ee).all(axis=1)
    if bad.any():
        rows = np.delete(np.arange(len(df)), del_list or [])[bad]
        raise ValueError(
            f"{path_to_file}: non-finite marker measurements in CSV rows "
            f"{rows.tolist()} (e.g. an occluded point); exclude them with "
            "del_list or clean the file"
        )

    # update number of data points
    calib_config["NbSample"] = q_act.shape[0]

    PEEm_exp = pose_ee.T
    PEEm_exp = PEEm_exp.flatten("C")

    q_exp = np.empty((calib_config["NbSample"], calib_config["q0"].shape[0]))
    for i in range(calib_config["NbSample"]):
        # a copy: q0 is robot.q0, which must not end up as the last sample (#125)
        config = calib_config["q0"].copy()
        config[calib_config["config_idx"]] = q_act[i, :]
        q_exp[i, :] = config

    return PEEm_exp, q_exp
```
