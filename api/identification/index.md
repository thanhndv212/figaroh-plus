# Identification

The identification module provides tools for dynamic parameter identification of robots, including the [reporting & verification suite](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md) (`print_quality_report()`, `export_html_report()`, `verify()`, `export_verification_report()`) attached directly to `BaseIdentification`.

## base_identification

Base class for robot dynamic parameter identification. This module provides a generalized framework for dynamic parameter identification that can be inherited by any robot type (TIAGo, UR10, MATE, etc.).

## `BaseIdentification(robot, config_file='config/robot_config.yaml')`

Bases: `ABC`

Base class for robot dynamic parameter identification.

Provides common functionality for all robots while allowing robot-specific implementations of key methods.

Initialize base identification with robot model and configuration.

Parameters:

| Name          | Type | Description                           | Default                      |
| ------------- | ---- | ------------------------------------- | ---------------------------- |
| `robot`       |      | Robot model loaded with FIGAROH       | *required*                   |
| `config_file` |      | Path to robot configuration YAML file | `'config/robot_config.yaml'` |

Source code in `src/figaroh/identification/base_identification.py`

```
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

    # Data contract (#55): the TrajectoryData an adapter returned, if
    # any, its valid-sample masks, and the stage records of this run
    self.trajectory = None
    self._sample_mask = None
    self._val_trajectory = None
    self._val_sample_mask = None
    self.stages = []

    # Diagnostics captured during solve() for validation / reporting:
    # column indices eliminated by _eliminate_zero_columns() and the
    # base-column selection from the QR decomposition, both needed to
    # evaluate phi_base against a regressor built from new (held-out)
    # trajectory data.
    self._idx_eliminated = None
    self._base_indices = None
    self.selected = None
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
```

### `solve(decimate=True, decimation_factor=10, zero_tolerance=0.001, plotting=True, save_results=False, html_report=False, wls=False)`

Main solving method for dynamic parameter identification.

This method implements the complete base parameter identification workflow including column elimination, optional decimation, QR decomposition, and quality metric computation.

Parameters:

| Name                | Type    | Description                                                                                                                                                                                                                                                                            | Default |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `decimate`          | `bool`  | Whether to apply decimation to reduce data size                                                                                                                                                                                                                                        | `True`  |
| `decimation_factor` | `int`   | Factor for signal decimation (default: 10)                                                                                                                                                                                                                                             | `10`    |
| `zero_tolerance`    | `float` | Tolerance for eliminating zero columns                                                                                                                                                                                                                                                 | `0.001` |
| `plotting`          | `bool`  | Whether to generate plots                                                                                                                                                                                                                                                              | `True`  |
| `save_results`      | `bool`  | Whether to save parameters to file                                                                                                                                                                                                                                                     | `False` |
| `html_report`       | `bool`  | If True, also export an HTML diagnostic report (see :meth:export_html_report) after the terminal quality report is printed.                                                                                                                                                            | `False` |
| `wls`               | `bool`  | If True, refine the OLS base-parameter estimate with iteratively-weighted least squares (Gautier, 1997) before computing quality metrics — see :meth:\_apply_weighted_least_squares. Relative standard deviations from the WLS fit are stored under self.result["wls_std_deviations"]. | `False` |

Returns:

| Name      | Type | Description              |
| --------- | ---- | ------------------------ |
| `ndarray` |      | Base parameters phi_base |

Raises:

| Type             | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| `AssertionError` | If prerequisites not met (dynamic_regressor, standard_parameter) |
| `ValueError`     | If data shapes are incompatible                                  |
| `LinAlgError`    | If QR decomposition fails                                        |

Source code in `src/figaroh/identification/base_identification.py`

```
def solve(
    self,
    decimate=True,
    decimation_factor=10,
    zero_tolerance=0.001,
    plotting=True,
    save_results=False,
    html_report=False,
    wls=False,
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
        wls (bool): If True, refine the OLS base-parameter estimate with
            iteratively-weighted least squares (Gautier, 1997) before
            computing quality metrics — see
            :meth:`_apply_weighted_least_squares`. Relative standard
            deviations from the WLS fit are stored under
            ``self.result["wls_std_deviations"]``.

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

    self._run_started_at = datetime.now(timezone.utc).isoformat()
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

    # Step 2b: leave out masked samples (after filtering/decimation)
    tau_processed, W_processed = self._mask_rows(
        tau_processed, W_processed, decimation_factor if decimate else 1
    )

    # Step 2c: keep only the joints whose effort enters the fit
    tau_processed, W_processed = self._select_fit_rows(tau_processed, W_processed)

    self._reset_solve_state(tau_processed, W_processed, active_params)

    # Step 3: Calculate base parameters
    from figaroh.tools.stages import record_stage

    try:
        results = self._calculate_base_parameters(
            tau_processed, W_processed, active_params
        )
    except Exception as e:
        record_stage(self, "fit", "failed", f"{type(e).__name__}: {e}")
        raise

    # Step 3b: Optional weighted least squares refinement
    wls_std = None
    if wls:
        phi_wls, wls_std = self._apply_weighted_least_squares()
        self.phi_base = phi_wls
        results["phi_base"] = phi_wls
        results["tau_estimated"] = self.dynamic_regressor_base @ phi_wls
        self.tau_identif = results["tau_estimated"]

    # Step 4: Store results and compute quality metrics
    self._run_finished_at = datetime.now(timezone.utc).isoformat()
    self._compute_quality_metrics()
    record_stage(
        self,
        "fit",
        "ok",
        "weighted least squares" if wls else "least squares",
        {
            "effort_rmse": (float(self.rms_error), "N·m or N (stacked joints)"),
            "rows": len(self.tau_noised),
            "base_parameters": len(self.phi_base),
        },
    )
    self._store_results(results)
    if wls_std is not None:
        self.result["wls_std_deviations"] = wls_std

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
```

### `solve_with_custom_solver(method='lstsq', regularization=None, alpha=0.0, constraints=None, bounds=None, decimate=False, decimation_factor=10, zero_tolerance=0.001, plotting=False, save_results=False, **solver_kwargs)`

Alternative solving method using advanced linear solver.

This method provides more flexibility than the default QR-based solve(), offering multiple solving methods, regularization, and constraints.

Parameters:

| Name                | Type    | Description                                                     | Default   |
| ------------------- | ------- | --------------------------------------------------------------- | --------- |
| `method`            | `str`   | Solving method ('lstsq', 'ridge', 'lasso', 'constrained', etc.) | `'lstsq'` |
| `regularization`    | `str`   | Regularization type ('l1', 'l2', 'elastic_net')                 | `None`    |
| `alpha`             | `float` | Regularization strength                                         | `0.0`     |
| `constraints`       | `dict`  | Linear constraints                                              | `None`    |
| `bounds`            | `tuple` | Box constraints on parameters                                   | `None`    |
| `decimate`          | `bool`  | Whether to apply decimation                                     | `False`   |
| `decimation_factor` | `int`   | Decimation factor if decimate=True                              | `10`      |
| `zero_tolerance`    | `float` | Tolerance for eliminating zero columns                          | `0.001`   |
| `plotting`          | `bool`  | Whether to generate plots                                       | `False`   |
| `save_results`      | `bool`  | Whether to save parameters to file                              | `False`   |
| `**solver_kwargs`   |         | Additional arguments for LinearSolver                           | `{}`      |

Returns:

| Name      | Type | Description                |
| --------- | ---- | -------------------------- |
| `ndarray` |      | Identified base parameters |

Example

> > > #### Ridge regression with L2 regularization
> > >
> > > phi = identification.solve_with_custom_solver( ... method='ridge', alpha=0.01)
> > >
> > > #### Constrained optimization with physical bounds
> > >
> > > bounds = [(0, 100) for _ in range(n_params)] phi = identification.solve_with_custom_solver( ... method='constrained', bounds=bounds)

Source code in `src/figaroh/identification/base_identification.py`

```
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
    tau_processed, W_processed = self._select_fit_rows(tau_processed, W_processed)

    self._reset_solve_state(tau_processed, W_processed, active_params)

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
    from figaroh.tools.qrdecomposition import QRDecomposer

    W_base, _, base_parameters, _, phi_std = QRDecomposer().double_decomposition(
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
```

### `load_param(config_file, setting_type='identification')`

Load the identification parameters from the yaml file.

Only the unified configuration format is supported; a legacy flat config raises an error naming `figaroh.utils.config_migration`.

Parameters:

| Name           | Type  | Description                          | Default            |
| -------------- | ----- | ------------------------------------ | ------------------ |
| `config_file`  | `str` | Path to a unified configuration file | *required*         |
| `setting_type` | `str` | Configuration section to load        | `'identification'` |

Source code in `src/figaroh/identification/base_identification.py`

```
def load_param(self, config_file, setting_type="identification"):
    """Load the identification parameters from the yaml file.

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
        unified_identif_config = create_task_config(
            self.robot, unified_config, setting_type
        )
        # Convert unified format to the identif_config dict
        self.identif_config = unified_to_legacy_identif_config(
            self.robot, unified_identif_config
        )
    except Exception as e:
        logger.error(f"Error loading config {config_file}: {e}")
        raise
```

### `load_trajectory_data(data_source=None)`

Load and process CSV data.

This method must be implemented by robot-specific subclasses to handle their specific data formats and file structures.

Parameters:

| Name          | Type  | Description                                                                                                                                                                                                                                                                                                                                                                                              | Default |
| ------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `data_source` | `str` | Optional override identifying an alternate dataset to load instead of the class's normal training data (e.g. a directory holding a held-out validation trajectory with the same file layout/naming as the training data). When None (default), the subclass loads its usual training data exactly as before. Passed through from identif_config["validation_data_file"] by :meth:\_load_validation_data. | `None`  |

Returns:

| Name   | Type | Description                                               |
| ------ | ---- | --------------------------------------------------------- |
| `dict` |      | Dictionary with keys 'timestamps', 'positions',           |
|        |      | 'velocities', 'accelerations', 'torques' (numpy arrays;   |
|        |      | 'velocities'/'accelerations' may be None to be derived by |
|        |      | differentiation).                                         |

Source code in `src/figaroh/identification/base_identification.py`

```
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
```

### `process_data(truncate=None)`

Load and process data

Source code in `src/figaroh/identification/base_identification.py`

```
def process_data(self, truncate=None):
    """Load and process data"""
    # Set default filter configuration
    filter_config = self.filter_config

    # load raw data: the legacy dict, or a TrajectoryData (#55)
    self.raw_data = self._adopt_trajectory(self.load_trajectory_data())

    # Truncate data if truncation indices are provided
    self.raw_data = self._truncate_data(self.raw_data, truncate)
    self._sample_mask = self._trajectory_mask(self.trajectory, truncate)

    # Apply filtering and differentiation kinematics data
    self.process_kinematics_data(filter_config)

    # Process joint torque data
    self.processed_data["torques"] = self.process_torque_data()

    # Update sample count to ensure consistency
    self.num_samples = self.processed_data["positions"].shape[0]

    # Build full configuration
    self._build_full_configuration()
    self._record_data_stage()
```

### `fit_joints()`

Active joints whose effort rows enter the fit, in active order.

`identif_config["torque_fit_joints"]` names them; by default every active joint. The other active joints keep their kinematics in the regressor (their motion still loads the fitted joints) but their measured effort is neither fitted nor scored: use it for joints whose effort signal is not trusted (unknown force constant, quantised).

Source code in `src/figaroh/identification/base_identification.py`

```
def fit_joints(self):
    """Active joints whose effort rows enter the fit, in active order.

    ``identif_config["torque_fit_joints"]`` names them; by default every
    active joint. The other active joints keep their kinematics in the
    regressor (their motion still loads the fitted joints) but their
    measured effort is neither fitted nor scored: use it for joints whose
    effort signal is not trusted (unknown force constant, quantised).
    """
    active = list(self.identif_config.get("active_joints", []))
    chosen = self.identif_config.get("torque_fit_joints")
    if chosen is None:
        return active
    unknown = [j for j in chosen if j not in active]
    if unknown:
        raise ValueError(
            f"torque_fit_joints {unknown} are not active joints {active}"
        )
    if not chosen:
        raise ValueError("torque_fit_joints is empty: nothing to fit")
    return [j for j in active if j in chosen]
```

### `calculate_full_regressor()`

Build regressor matrix, compute pre-identified values of standard parameters, compute joint torques based on pre-identified standard parameters.

Source code in `src/figaroh/identification/base_identification.py`

```
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
```

### `initialize_standard_parameters()`

Initialize standard parameters for the robot.

Source code in `src/figaroh/identification/base_identification.py`

```
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
```

### `compute_reference_torque()`

Compute reference joint torques based on standard parameters and dynamic regressor.

Source code in `src/figaroh/identification/base_identification.py`

```
def compute_reference_torque(self):
    """Compute reference joint torques based on standard parameters and dynamic regressor."""

    # joint torque estimated from p,v,a with std params
    phi_ref = np.array(list(self.standard_parameter.values()))
    tau_ref = np.dot(self.dynamic_regressor, phi_ref)

    # filter only active joints
    self.tau_ref = tau_ref[
        range(len(self.identif_config["act_idxv"]) * self.num_samples)
    ]
```

### `process_kinematics_data(filter_config=None)`

Process kinematics data (positions, velocities, accelerations) with filtering.

Source code in `src/figaroh/identification/base_identification.py`

```
def process_kinematics_data(self, filter_config=None):
    """Process kinematics data (positions, velocities, accelerations) with filtering."""
    self.filter_kinematics_data(filter_config)
```

### `filter_kinematics_data(filter_config=None)`

Apply filtering to data with configurable parameters.

Parameters:

| Name            | Type   | Description                                                                                                                                 | Default |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `filter_config` | `dict` | Filter configuration with keys: - differentiation_method: Method for derivative estimation - filter_params: Parameters for signal filtering | `None`  |

Raises:

| Type         | Description                 |
| ------------ | --------------------------- |
| `ValueError` | If required data is missing |

Source code in `src/figaroh/identification/base_identification.py`

```
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
```

### `process_torque_data(**kwargs)`

Process torque data (generic implementation, should be overridden for robot-specific processing).

Source code in `src/figaroh/identification/base_identification.py`

```
def process_torque_data(self, **kwargs):
    """Process torque data (generic implementation, should be overridden for robot-specific processing)."""
    # Generic torque processing - robots should override this method
    if self.raw_data["torques"] is not None:
        self.processed_data["torques"] = self.raw_data["torques"]
        return self.processed_data["torques"]
    else:
        raise ValueError("Torque data is required for processing")
```

### `print_quality_report()`

Print a formatted identification quality report to the terminal.

Reports condition number, overall torque residual statistics, per-joint residuals (when available), base-parameter uncertainty, held-out validation metrics (if configured), and optional physical-consistency / reconstruction status.

Source code in `src/figaroh/identification/base_identification.py`

```
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
    from figaroh.tools.stages import stages_line

    print(f"  Stages:          {stages_line(self)}")
    sel = getattr(self, "selected", None)
    if sel is not None:
        print(
            f"  Reported estimate: "
            f"{sel.stage if sel.accepted else 'none'}"
            f" (requested {sel.requested}: {sel.status}; {sel.reason})"
        )

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
            else "moderately-conditioned" if cond_num < 1000 else "ill-conditioned"
        )
        print(f"  Condition:    {cond_num:.1f} ({cond_label})")
    else:
        print("  Condition:    unavailable")
    rmse_label = "RMSE:" if sel is None else "RMSE (base fit):"
    print(
        f"  {rmse_label:<13s} {result.get('rmse norm (N/m)', float('nan')):.4f}    "
        f"Correlation: {self.correlation:.4f}"
    )
    selected_info = result.get("selected")
    if isinstance(selected_info, dict) and "effort_rmse_fit" in selected_info:
        print(
            f"  RMSE ({selected_info['stage']} estimate): "
            f"{selected_info['effort_rmse_fit']:.4f}"
        )

    if per_joint is not None:
        print("-" * 70)
        heading = "Per-Joint Torque Residuals (training set)"
        if sel is not None:
            heading = "Base fit residuals, per joint (training set)"
        print(f"  {heading}")
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
        per_joint_sel = (selected_info or {}).get("effort_rmse_fit_per_joint")
        if per_joint_sel:
            print(f"  Selected estimate ({selected_info['stage']}) RMSE per joint")
            for name, value in per_joint_sel.items():
                print(f"  {name:<22s} {value:10.4f}")
    else:
        print("-" * 70)
        print("  Per-joint residuals: unavailable")

    # ── Base-parameter uncertainty ──
    print("-" * 70)
    std_relative = getattr(self, "std_relative", None)
    base_names = result.get("base parameters names", [])
    if std_relative is not None and len(std_relative) == len(base_names):
        order = sorted(range(len(base_names)), key=lambda i: -abs(std_relative[i]))
        print("  Base-Parameter Uncertainty (top 5 by relative std-dev)")
        for i in order[:5]:
            print(f"    {base_names[i]:<50s} {std_relative[i]:8.1f}%")
    else:
        print("  Base-parameter uncertainty: unavailable")

    # ── Validation ──
    print("-" * 70)
    val = result.get("validation_metrics")
    if val is not None:
        if val.get("validation_source") == "identification_data_fallback":
            print(
                "  ⚠ WARNING: no separate validation data "
                "provided — falling back to identification data. "
                "These are NOT an independent generalization test."
            )
            print(
                f"  Validation (identification set, " f"n={val['n_val_samples']})"
            )
        else:
            print(f"  Validation (separate set, n={val['n_val_samples']})")
        print(
            f"    RMSE nominal:    {val['rmse_nominal']:.4f}    "
            f"RMSE identified: {val['rmse_identified']:.4f}"
        )
        print(
            f"    Improvement:     {val['improvement_pct']:.1f}%    "
            "Correlation (per-joint normalised): "
            f"{val.get('correlation_normalised', float('nan')):.4f}"
        )
        for name, m in val.get("per_joint", {}).items():
            flag = "" if m["predictive"] else "  <- not better than a constant"
            print(
                f"    {name:24s} RMSE {m['rmse_identified']:.4g} {m['unit']}"
                f"  std {m['std_measured']:.4g}  R² {m['r2']:.3f}{flag}"
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
```

### `export_html_report(output_path=None, output_dir='results')`

Export the identification quality report as a self-contained HTML file — the visual counterpart of :meth:`print_quality_report`.

Parameters:

| Name          | Type  | Description                                                                             | Default     |
| ------------- | ----- | --------------------------------------------------------------------------------------- | ----------- |
| `output_path` | `str` | Explicit output file path. If None, writes to \<output_dir>/identification_report.html. | `None`      |
| `output_dir`  | `str` | Directory used when output_path is not given (created if missing).                      | `'results'` |

Returns:

| Type  | Description                         |
| ----- | ----------------------------------- |
| `str` | The path the report was written to. |

Raises:

| Type             | Description                      |
| ---------------- | -------------------------------- |
| `AttributeError` | If solve() has not been run yet. |

Source code in `src/figaroh/identification/base_identification.py`

```
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
```

### `verify(thresholds=None, scope='prediction')`

Return scoped acceptance evidence with explicit incomplete states.

`scope="execution"` checks finite numerical fit outputs; it does not certify prediction, physical parameters or export. `scope="prediction"` (default) additionally requires independent validation and explicit application error limits. No universal improvement, correlation, conditioning or prediction-error gate is imposed.

`thresholds` maps metric names to `threshold`, `comparison` (min/max), and optional `required` (default True). Missing required evidence produces not_evaluated, not PASS. Nonfinite evidence fails. `passed` is True only when all required checks in scope pass.

Source code in `src/figaroh/identification/base_identification.py`

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
    if self.result is None:
        raise AttributeError(
            "No identification results available. Run solve() first."
        )

    from figaroh.tools._report_common import (
        IDENTIFICATION_DEFAULT_THRESHOLDS,
        scoped_verification,
    )
    from figaroh.tools.identification_report import _build_insights
    from figaroh.tools.provenance import collect_run_provenance

    thresholds = (
        thresholds if thresholds is not None else IDENTIFICATION_DEFAULT_THRESHOLDS
    )

    result = self.result
    base_names = result.get("base parameters names", [])
    std_relative_raw = getattr(self, "std_relative", None)
    std_relative = list(std_relative_raw) if std_relative_raw is not None else []
    validation = result.get("validation_metrics")

    # with a non-default select_stage the thresholds judge the selected
    # estimate, not the base fit it was derived from (#61)
    fit_rmse = result.get("rmse norm (N/m)", float("nan"))
    selected_info = result.get("selected")
    if (
        isinstance(selected_info, dict)
        and selected_info.get("effort_rmse_fit") is not None
    ):
        fit_rmse = selected_info["effort_rmse_fit"]
    metrics: Dict[str, float] = {
        "condition_number": result.get("condition number", float("nan")),
        "rmse": fit_rmse,
    }
    if validation is not None:
        # per-joint normalised: a pooled correlation is dominated by the
        # largest (gravity) torques and mixes N with N·m (#103)
        metrics["validation_correlation"] = validation.get(
            "correlation_normalised", validation.get("correlation", float("nan"))
        )
        metrics["validation_unpredictable_joints"] = float(
            len(validation.get("unpredictable_joints", []))
        )
        metrics["validation_improvement_pct"] = validation.get(
            "improvement_pct", float("nan")
        )

    joint_names = (validation or {}).get("joint_names", [])
    independent = (validation or {}).get("validation_source") == "validation_data"
    for joint in joint_names:
        observed = np.asarray(
            validation.get("tau_measured_per_joint", {}).get(joint, [])
        )
        predicted = np.asarray(
            validation.get("tau_identified_per_joint", {}).get(joint, [])
        )
        if observed.size and observed.shape == predicted.shape:
            prefix = "validation" if independent else "training"
            metrics[f"{prefix}_rmse:{joint}"] = float(
                np.sqrt(np.mean((observed - predicted) ** 2))
            )
            metrics[f"{prefix}_abs_bias:{joint}"] = float(
                abs(np.mean(observed - predicted))
            )
            metrics[f"{prefix}_peak_error:{joint}"] = float(
                np.max(np.abs(observed - predicted))
            )
    sel = getattr(self, "selected", None)
    finite_parameters = result.get("base parameters values")
    finite_prediction = result.get("torque estimated")
    facts_extra = {}
    if sel is not None:
        # the candidate that was asked for, not the fit (#61)
        if sel.values is not None:
            finite_parameters = sel.values
            if sel.space == "standard":
                eliminated = set(self._idx_eliminated or [])
                keep = [i for i in range(len(sel.names)) if i not in eliminated]
                finite_prediction = self._solve_W @ np.asarray(sel.values)[keep]
        facts_extra["selected_stage_accepted"] = (
            bool(sel.accepted),
            f"The requested {sel.requested} estimate was rejected: {sel.reason}",
        )
    verdict = scoped_verification(
        metrics,
        thresholds,
        scope,
        {
            "finite_parameters": finite_parameters,
            "finite_prediction": finite_prediction,
            "finite_measurements": result.get("torque processed"),
            "finite_fit_rmse": (
                fit_rmse
                if isinstance(selected_info, dict)
                else result.get("rmse norm (N/m)")
            ),
        },
        independent,
        [f"validation_rmse:{j}" for j in joint_names],
        facts={
            "requested_validation_loaded": (
                not self.identif_config.get("validation_data_file")
                or bool(getattr(self, "_val_available", False))
            ),
            "matching_torque_shapes": (
                np.shape(result["torque estimated"])
                == np.shape(result["torque processed"])
                if result.get("torque estimated") is not None
                and result.get("torque processed") is not None
                else None
            ),
            **facts_extra,
        },
    )
    verdict.insights = [
        i["text"]
        for i in _build_insights(result, std_relative, base_names, validation)
    ]
    verdict.metadata = getattr(
        self, "_run_provenance", None
    ) or collect_run_provenance(self, "identification")

    active_joints = self.identif_config.get("active_joints", [])
    if validation is not None and "tau_nominal_per_joint" in validation:
        verdict.series = {
            # every sample (masked ones too), like the per-joint series
            "time": list(
                range(
                    len(next(iter(validation["tau_measured_per_joint"].values())))
                )
            ),
            "joint_names": validation.get("joint_names", active_joints),
            "nominal": validation["tau_nominal_per_joint"],
            "fitted": validation["tau_identified_per_joint"],
            "measured": validation["tau_measured_per_joint"],
        }
    verdict.compat = {
        "active_joints": active_joints,
        "decimate": bool(getattr(self, "_decimate_used", False)),
        "sample_count": result.get("num samples", 0),
        "config_sha256": verdict.metadata.get("config", {}).get("sha256"),
    }
    from figaroh.tools.stages import apply_to_verdict

    # the reported parameters come from the fit (phi_base / var_) unless
    # select_stage asked for another estimate (#61)
    apply_to_verdict(
        verdict,
        self,
        selected_stage=(
            "fit" if sel is None else (sel.stage if sel.accepted else "none")
        ),
    )
    return verdict
```

### `export_urdf(nominal_urdf, output_path=None, merged_bodies='refuse')`

Write the selected estimate's link inertials into a URDF.

Needs `select_stage` `reconstruction` or `physical_fit` and an *accepted* estimate: the base-parameter fit has no per-link inertials, and a rejected estimate is not exported. Either case records stage `export` as failed and raises. On success the URDF is written, reloaded with Pinocchio as a check and `export` is recorded `ok` with the file as artifact.

Parameters:

| Name            | Type            | Description                                                                                                                                           | Default    |
| --------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `nominal_urdf`  | `str`           | Path of the nominal URDF the run was based on.                                                                                                        | *required* |
| `output_path`   | `Optional[str]` | Where to write; default <stem>\_modified.urdf beside the nominal one.                                                                                 | `None`     |
| `merged_bodies` | `str`           | "refuse" (default) or "subtract_fixed" for joints whose Pinocchio body holds fixed-attached links, see :func:figaroh.tools.urdf_exporter.export_urdf. | `'refuse'` |

Returns:

| Name  | Type  | Description                        |
| ----- | ----- | ---------------------------------- |
| `str` | `str` | Absolute path of the written URDF. |

Raises:

| Type         | Description                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| `ValueError` | no accepted standard-space estimate, or the exporter refused the parameters. |

Source code in `src/figaroh/identification/base_identification.py`

```
def export_urdf(
    self,
    nominal_urdf: str,
    output_path: Optional[str] = None,
    merged_bodies: str = "refuse",
) -> str:
    """Write the selected estimate's link inertials into a URDF.

    Needs ``select_stage`` ``reconstruction`` or ``physical_fit`` and an
    *accepted* estimate: the base-parameter fit has no per-link
    inertials, and a rejected estimate is not exported. Either case
    records stage ``export`` as failed and raises. On success the URDF is
    written, reloaded with Pinocchio as a check and ``export`` is
    recorded ``ok`` with the file as artifact.

    Args:
        nominal_urdf: Path of the nominal URDF the run was based on.
        output_path: Where to write; default ``<stem>_modified.urdf``
            beside the nominal one.
        merged_bodies: ``"refuse"`` (default) or ``"subtract_fixed"`` for
            joints whose Pinocchio body holds fixed-attached links, see
            :func:`figaroh.tools.urdf_exporter.export_urdf`.

    Returns:
        str: Absolute path of the written URDF.

    Raises:
        ValueError: no accepted standard-space estimate, or the exporter
            refused the parameters.
    """
    from figaroh.tools.stages import record_stage
    from figaroh.tools.urdf_exporter import export_urdf

    sel = getattr(self, "selected", None)
    try:
        if sel is None or sel.space != "standard":
            raise ValueError(
                "the base-parameter fit has no link inertials to export; "
                "set select_stage to 'reconstruction' or 'physical_fit'"
            )
        if not sel.accepted:
            raise ValueError(
                f"the requested {sel.requested} estimate was rejected "
                f"({sel.reason}); nothing to export"
            )
        params = {}
        # only joints whose inertials were identified: the free-flyer
        # body has no URDF joint and non-identified joints keep their
        # nominal values (also when a fixed link is merged into them)
        for joint, p10 in sel.link_p10(self._identified_joints()).items():
            for key, value in zip(_INERTIAL_KEYS, p10):
                params[f"{key}_{joint}"] = float(value)
        out = export_urdf(
            nominal_urdf,
            params,
            output_path=output_path,
            merged_bodies=merged_bodies,
            psd_eig_tol=sel.psd_eig_tol,
        )
        import pinocchio as pin

        try:
            pin.buildModelFromUrdf(out)  # the file must load
        except Exception:
            os.remove(out)
            raise
    except (ValueError, KeyError) as exc:
        record_stage(self, "export", "failed", str(exc))
        self._refresh_result_stages()
        raise ValueError(f"export failed: {exc}") from exc
    except Exception as exc:  # I/O, Pinocchio reload, ...
        record_stage(self, "export", "failed", f"{type(exc).__name__}: {exc}")
        self._refresh_result_stages()
        raise
    record_stage(
        self,
        "export",
        "ok",
        f"{sel.stage} link inertials written",
        artifacts=[out],
    )
    self._refresh_result_stages()
    return out
```

### `export_verification_report(output_path=None, output_dir='results', thresholds=None, scope='prediction')`

Write this identification's :meth:`verify` verdict as JSON.

Parameters:

| Name          | Type                                  | Description                                                                                | Default        |
| ------------- | ------------------------------------- | ------------------------------------------------------------------------------------------ | -------------- |
| `output_path` | `str`                                 | Explicit file path. If omitted, defaults to {output_dir}/identification_verification.json. | `None`         |
| `output_dir`  | `str`                                 | Directory used when output_path is omitted.                                                | `'results'`    |
| `thresholds`  | `Optional[Dict[str, Dict[str, Any]]]` | Forwarded to :meth:verify.                                                                 | `None`         |
| `scope`       | `str`                                 | Forwarded to :meth:verify; default prediction acceptance.                                  | `'prediction'` |

Returns:

| Name  | Type  | Description                               |
| ----- | ----- | ----------------------------------------- |
| `str` | `str` | The path the JSON verdict was written to. |

Source code in `src/figaroh/identification/base_identification.py`

```
def export_verification_report(
    self,
    output_path: str = None,
    output_dir: str = "results",
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    scope: str = "prediction",
) -> str:
    """Write this identification's :meth:`verify` verdict as JSON.

    Args:
        output_path: Explicit file path. If omitted, defaults to
            ``{output_dir}/identification_verification.json``.
        output_dir: Directory used when ``output_path`` is omitted.
        thresholds: Forwarded to :meth:`verify`.
        scope: Forwarded to :meth:`verify`; default prediction acceptance.

    Returns:
        str: The path the JSON verdict was written to.
    """
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
        output_path = join(output_dir, "identification_verification.json")

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

### `plot_results()`

Plot identification results using unified results manager.

Source code in `src/figaroh/identification/base_identification.py`

```
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
            plt.title(
                f"{self.__class__.__name__} Torque Comparison"
                + ("" if self.selected is None else " (base fit)")
            )
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
                n_joints=len(self._fit_blocks()),
                joint_names=self.fit_joints(),
            ),
            _basic_plots,
            logger,
            "identification",
        )
    else:
        _basic_plots()
```

### `save_results(output_dir='results')`

Save identification results using unified results manager.

Source code in `src/figaroh/identification/base_identification.py`

```
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
```

## identification_tools

## `base_param_from_standard(phi_standard, params_base)`

Convert standard parameters to base parameters.

Takes standard dynamic parameters and calculates the corresponding base parameters using analytical relationships between them.

Parameters:

| Name           | Type   | Description                         | Default    |
| -------------- | ------ | ----------------------------------- | ---------- |
| `phi_standard` | `dict` | Standard parameters from model/URDF | *required* |
| `params_base`  | `list` | Analytical parameter relationships  | *required* |

Returns:

| Name   | Type | Description                                               |
| ------ | ---- | --------------------------------------------------------- |
| `list` |      | Base parameter values calculated from standard parameters |

Source code in `src/figaroh/identification/identification_tools.py`

```
def base_param_from_standard(phi_standard, params_base):
    """Convert standard parameters to base parameters.

    Takes standard dynamic parameters and calculates the corresponding base
    parameters using analytical relationships between them.

    Args:
        phi_standard (dict): Standard parameters from model/URDF
        params_base (list): Analytical parameter relationships

    Returns:
        list: Base parameter values calculated from standard parameters
    """
    phi_base = []
    ops = {"+": operator.add, "-": operator.sub}
    for ii in range(len(params_base)):
        param_base_i = params_base[ii].split(" ")
        values = []
        list_ops = []
        for jj in range(len(param_base_i)):
            param_base_j = param_base_i[jj].split("*")
            if len(param_base_j) == 2:
                value = float(param_base_j[0]) * phi_standard[param_base_j[1]]
                values.append(value)
            elif param_base_j[0] != "+" and param_base_j[0] != "-":
                value = phi_standard[param_base_j[0]]
                values.append(value)
            else:
                list_ops.append(ops[param_base_j[0]])
        value_phi_base = values[0]
        for kk in range(len(list_ops)):
            value_phi_base = list_ops[kk](value_phi_base, values[kk + 1])
        phi_base.append(value_phi_base)
    return phi_base
```

## `relative_stdev(W_b, phi_b, tau)`

Calculate relative standard deviation of identified parameters.

Implements the residual error method from [Pressé & Gautier 1991] to estimate parameter uncertainty.

Parameters:

| Name    | Type      | Description                   | Default    |
| ------- | --------- | ----------------------------- | ---------- |
| `W_b`   | `ndarray` | Base regressor matrix         | *required* |
| `phi_b` | `list`    | Base parameter values         | *required* |
| `tau`   | `ndarray` | Measured joint torques/forces | *required* |

Returns:

| Name      | Type | Description                                             |
| --------- | ---- | ------------------------------------------------------- |
| `ndarray` |      | Relative standard deviation (%) for each base parameter |

Source code in `src/figaroh/identification/identification_tools.py`

```
def relative_stdev(W_b, phi_b, tau):
    """Calculate relative standard deviation of identified parameters.

    Implements the residual error method from [Pressé & Gautier 1991] to
    estimate parameter uncertainty.

    Args:
        W_b (ndarray): Base regressor matrix
        phi_b (list): Base parameter values
        tau (ndarray): Measured joint torques/forces

    Returns:
        ndarray: Relative standard deviation (%) for each base parameter
    """
    # stdev of residual error ro
    sig_ro_sqr = np.linalg.norm((tau - np.dot(W_b, phi_b))) ** 2 / (
        W_b.shape[0] - phi_b.shape[0]
    )

    # covariance matrix of estimated parameters
    C_x = sig_ro_sqr * np.linalg.inv(np.dot(W_b.T, W_b))

    # relative stdev of estimated parameters
    std_x_sqr = np.diag(C_x)
    std_xr = np.zeros(std_x_sqr.shape[0])
    for i in range(std_x_sqr.shape[0]):
        std_xr[i] = np.round(100 * np.sqrt(std_x_sqr[i]) / np.abs(phi_b[i]), 2)

    return std_xr
```

## `index_in_base_params(params, id_segments)`

Map segment IDs to their base parameters.

For each segment ID, finds which base parameters contain inertial parameters from that segment.

Parameters:

| Name          | Type   | Description                | Default    |
| ------------- | ------ | -------------------------- | ---------- |
| `params`      | `list` | Base parameter expressions | *required* |
| `id_segments` | `list` | Segment IDs to map         | *required* |

Returns:

| Name   | Type | Description                                         |
| ------ | ---- | --------------------------------------------------- |
| `dict` |      | Maps segment IDs to lists of base parameter indices |

Source code in `src/figaroh/identification/identification_tools.py`

```
def index_in_base_params(params, id_segments):
    """Map segment IDs to their base parameters.

    For each segment ID, finds which base parameters contain inertial
    parameters from that segment.

    Args:
        params (list): Base parameter expressions
        id_segments (list): Segment IDs to map

    Returns:
        dict: Maps segment IDs to lists of base parameter indices
    """
    base_index = []
    params_name = [
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
        "mx",
        "my",
        "mz",
        "m",
    ]

    id_segments_new = [i for i in range(len(id_segments))]

    for id in id_segments:
        for ii in range(len(params)):
            param_base_i = params[ii].split(" ")
            for jj in range(len(param_base_i)):
                param_base_j = param_base_i[jj].split("*")
                for ll in range(len(param_base_j)):
                    for kk in params_name:
                        if kk + str(id) == param_base_j[ll]:
                            base_index.append((id, ii))

    base_index[:] = list(set(base_index))
    base_index = sorted(base_index)

    dictio = {}

    for i in base_index:
        dictio.setdefault(i[0], []).append(i[1])

    values = []
    for ii in dictio:
        values.append(dictio[ii])

    return dict(zip(id_segments_new, values))
```

## `weigthed_least_squares(robot, phi_b, W_b, tau_meas, tau_est, identif_config)`

Compute weighted least squares solution for parameter identification.

Implements iteratively reweighted least squares method from [Gautier, 1997]. Accounts for heteroscedastic noise.

Parameters:

| Name       | Type      | Description                     | Default    |
| ---------- | --------- | ------------------------------- | ---------- |
| `robot`    | `Robot`   | Robot model                     | *required* |
| `phi_b`    | `ndarray` | Initial base parameters         | *required* |
| `W_b`      | `ndarray` | Base regressor matrix           | *required* |
| `tau_meas` | `ndarray` | Measured joint torques          | *required* |
| `tau_est`  | `ndarray` | Estimated joint torques         | *required* |
| `param`    | `dict`    | Settings including idx_tau_stop | *required* |

Returns:

| Name      | Type | Description                |
| --------- | ---- | -------------------------- |
| `ndarray` |      | Identified base parameters |

Source code in `src/figaroh/identification/identification_tools.py`

```
def weigthed_least_squares(robot, phi_b, W_b, tau_meas, tau_est, identif_config):
    """Compute weighted least squares solution for parameter identification.

    Implements iteratively reweighted least squares method from
    [Gautier, 1997]. Accounts for heteroscedastic noise.

    Args:
        robot (pin.Robot): Robot model
        phi_b (ndarray): Initial base parameters
        W_b (ndarray): Base regressor matrix
        tau_meas (ndarray): Measured joint torques
        tau_est (ndarray): Estimated joint torques
        param (dict): Settings including idx_tau_stop

    Returns:
        ndarray: Identified base parameters
    """
    sigma = np.zeros(robot.model.nq)  # For ground reaction force model
    P = np.zeros((len(tau_meas), len(tau_meas)))
    nb_samples = int(identif_config["idx_tau_stop"][0])
    start_idx = int(0)
    for ii in range(robot.model.nq):
        tau_slice = slice(int(start_idx), int(identif_config["idx_tau_stop"][ii]))
        diff = tau_meas[tau_slice] - tau_est[tau_slice]
        denom = len(tau_meas[tau_slice]) - len(phi_b)
        sigma[ii] = np.linalg.norm(diff) / denom

        start_idx = identif_config["idx_tau_stop"][ii]

        for jj in range(nb_samples):
            idx = jj + ii * nb_samples
            P[idx, idx] = 1 / sigma[ii]

        phi_b = np.matmul(np.linalg.pinv(np.matmul(P, W_b)), np.matmul(P, tau_meas))

    return phi_b  # full precision (#142)
```

## `calculate_first_second_order_differentiation(model, q, identif_config, dt=None, backend=None)`

Estimate interval tangent velocities and their coordinate derivatives.

Configuration differences use the model/backend Lie-group operation, so positions have width `nq` and velocities/accelerations have width `nv`. Effort-selection flags do not determine these dimensions.

Parameters:

| Name             | Type               | Description                                                                                                                                                              | Default    |
| ---------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `model`          | `Model`            | Robot model; unused when backend is supplied.                                                                                                                            | *required* |
| `q`              | `ndarray`          | Finite configurations of shape (n_samples, nq), with at least three samples. Manifold configurations must be valid for the selected model (e.g. normalized quaternions). | *required* |
| `identif_config` | `dict`             | Contains positive finite ts when dt is None.                                                                                                                             | *required* |
| `dt`             | `float or ndarray` | Positive finite sample interval, or one interval per consecutive configuration pair (n_samples - 1). Pass timestamp differences, not absolute timestamps.                | `None`     |
| `backend`        | `DynamicsBackend`  | Supplies nq, nv and compute_difference instead of the Pinocchio model.                                                                                                   | `None`     |

Returns:

| Name    | Type | Description                                                      |
| ------- | ---- | ---------------------------------------------------------------- |
| `tuple` |      | Positions (n_samples - 2, nq), velocities and accelerations      |
|         |      | (n_samples - 2, nv). All tangent coordinates are differentiated. |

Raises:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `ValueError` | Invalid configurations, sample count or timesteps. |

Note

Preserves the historical alignment: q[:-2] and the first n_samples-2 forward interval velocities. Velocities represent interval midpoints, not the returned position timestamps. Acceleration is the gradient of those tangent components at the midpoint times (one-sided at endpoints). No interpolation to position timestamps or moving-frame transport is performed. Both discarded position samples are at the end.

Source code in `src/figaroh/identification/identification_tools.py`

```
def calculate_first_second_order_differentiation(
    model, q, identif_config, dt=None, backend=None
):
    """Estimate interval tangent velocities and their coordinate derivatives.

    Configuration differences use the model/backend Lie-group operation, so
    positions have width ``nq`` and velocities/accelerations have width ``nv``.
    Effort-selection flags do not determine these dimensions.

    Args:
        model (pin.Model): Robot model; unused when backend is supplied.
        q (ndarray): Finite configurations of shape (n_samples, nq), with at
            least three samples. Manifold configurations must be valid for
            the selected model (e.g. normalized quaternions).
        identif_config (dict): Contains positive finite ``ts`` when dt is None.
        dt (float or ndarray, optional): Positive finite sample interval, or
            one interval per consecutive configuration pair (n_samples - 1).
            Pass timestamp differences, not absolute timestamps.
        backend (DynamicsBackend, optional): Supplies nq, nv and
            compute_difference instead of the Pinocchio model.

    Returns:
        tuple: Positions (n_samples - 2, nq), velocities and accelerations
        (n_samples - 2, nv). All tangent coordinates are differentiated.

    Raises:
        ValueError: Invalid configurations, sample count or timesteps.

    Note:
        Preserves the historical alignment: q[:-2] and the first n_samples-2
        forward interval velocities. Velocities represent interval midpoints,
        not the returned position timestamps. Acceleration is the gradient of
        those tangent components at the midpoint times (one-sided at endpoints).
        No interpolation to position timestamps or moving-frame transport is
        performed. Both discarded position samples are at the end.
    """
    nq = backend.nq if backend is not None else model.nq
    nv = backend.nv if backend is not None else model.nv
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != nq:
        raise ValueError(f"q must have shape (n_samples, {nq})")
    if q.shape[0] < 3:
        raise ValueError("q must contain at least three samples")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must contain only finite configurations")

    intervals = np.asarray(identif_config["ts"] if dt is None else dt, dtype=float)
    constant_dt = intervals.ndim == 0
    if constant_dt:
        intervals = np.full(q.shape[0] - 1, float(intervals))
    if intervals.shape != (q.shape[0] - 1,):
        raise ValueError("dt must be scalar or have one timestep per sample pair")
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise ValueError("Every timestep must be finite and strictly positive")

    dq = np.empty((q.shape[0] - 1, nv))
    for ii, timestep in enumerate(intervals):
        if backend is not None:
            difference = backend.compute_difference(q[ii], q[ii + 1])
        else:
            difference = pin.difference(model, q[ii], q[ii + 1])
        dq[ii] = difference / timestep

    if constant_dt:
        ddq = np.gradient(dq, axis=0, edge_order=1) / intervals[0]
    else:
        midpoint_times = np.cumsum(intervals) - 0.5 * intervals
        if not np.all(np.isfinite(midpoint_times)) or np.any(
            np.diff(midpoint_times) <= 0
        ):
            raise ValueError("dt produces invalid interval midpoint times")
        ddq = np.gradient(dq, midpoint_times, axis=0, edge_order=1)

    return q[:-2].copy(), dq[:-1], ddq[:-1]
```

## `low_pass_filter_data(data, identif_config, nbutter=5)`

Apply zero-phase Butterworth low-pass filter to measurement data.

Uses scipy's filtfilt for zero-phase digital filtering. Removes high frequency noise while preserving signal phase. Handles border effects by trimming filtered data.

Parameters:

| Name      | Type      | Description                                                                                             | Default    |
| --------- | --------- | ------------------------------------------------------------------------------------------------------- | ---------- |
| `data`    | `ndarray` | Raw measurement data to filter                                                                          | *required* |
| `param`   | `dict`    | Filter parameters containing: - ts: Sample time - cut_off_frequency_butterworth: Cutoff frequency in Hz | *required* |
| `nbutter` | `int`     | Filter order. Higher order gives sharper frequency cutoff. Defaults to 5.                               | `5`        |

Returns:

| Name      | Type | Description                               |
| --------- | ---- | ----------------------------------------- |
| `ndarray` |      | Filtered data with border regions removed |

Note

Border effects are handled by removing nborder = 5\*nbutter samples from start and end of filtered signal.

Source code in `src/figaroh/identification/identification_tools.py`

```
def low_pass_filter_data(data, identif_config, nbutter=5):
    """Apply zero-phase Butterworth low-pass filter to measurement data.

    Uses scipy's filtfilt for zero-phase digital filtering. Removes high
    frequency noise while preserving signal phase. Handles border effects by
    trimming filtered data.

    Args:
        data (ndarray): Raw measurement data to filter
        param (dict): Filter parameters containing:
            - ts: Sample time
            - cut_off_frequency_butterworth: Cutoff frequency in Hz
        nbutter (int, optional): Filter order. Higher order gives sharper
            frequency cutoff. Defaults to 5.

    Returns:
        ndarray: Filtered data with border regions removed

    Note:
        Border effects are handled by removing nborder = 5*nbutter samples
        from start and end of filtered signal.
    """
    cutoff = identif_config["ts"] * identif_config["cut_off_frequency_butterworth"] / 2
    b, a = signal.butter(nbutter, cutoff, "low")

    padlen = 3 * (max(len(b), len(a)) - 1)
    data = signal.filtfilt(b, a, data, axis=0, padtype="odd", padlen=padlen)

    # Remove border effects
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    end_slice = slice(data.shape[0] - nbord, data.shape[0])
    data = np.delete(data, end_slice, axis=0)

    return data
```

## config

Configuration parsing and parameter management for robot identification.

This module handles all configuration-related functionality including:

- YAML configuration file parsing
- Unified to legacy config format conversion
- Parameter extraction and validation
- Signal processing and mechanical parameter management

## `unified_to_legacy_identif_config(robot, unified_identif_config)`

Convert unified identification format to legacy identif_config format.

Maps the unified identification configuration structure to the `identif_config` dict the identification classes consume.

Parameters:

| Name                     | Type           | Description                              | Default    |
| ------------------------ | -------------- | ---------------------------------------- | ---------- |
| `robot`                  | `RobotWrapper` | Robot instance containing model and data | *required* |
| `unified_identif_config` | `dict`         | Configuration from create_task_config    | *required* |

Returns:

| Name   | Type   | Description                                   |
| ------ | ------ | --------------------------------------------- |
| `dict` | `dict` | Identification configuration (identif_config) |

Example

> > > unified_config = create_task_config(robot, parsed_config, ... "identification") legacy_config = unified_to_legacy_identif_config(robot, ... unified_config)
> > >
> > > ### legacy_config has the identif_config keys

Source code in `src/figaroh/identification/config.py`

```
def unified_to_legacy_identif_config(robot, unified_identif_config) -> dict:
    """Convert unified identification format to legacy identif_config format.

    Maps the unified identification configuration structure to the
    ``identif_config`` dict the identification classes consume.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_identif_config (dict): Configuration from create_task_config

    Returns:
        dict: Identification configuration (``identif_config``)

    Example:
        >>> unified_config = create_task_config(robot, parsed_config,
        ...                                    "identification")
        >>> legacy_config = unified_to_legacy_identif_config(robot,
        ...                                                  unified_config)
        >>> # legacy_config has the identif_config keys
    """
    # Initialize output configuration
    identif_config = {}

    # Extract unified config sections
    mechanics = unified_identif_config.get("mechanics", {})
    joints = unified_identif_config.get("joints", {})
    problem = unified_identif_config.get("problem", {})
    coupling = unified_identif_config.get("coupling", {})
    signal_processing = unified_identif_config.get("signal_processing", {})

    # 1. Extract basic robot information
    identif_config["robot_name"] = robot.model.name
    identif_config["instance"] = unified_identif_config.get("instance", {})

    # 2. Extract signal processing parameters
    _extract_signal_processing_params(identif_config, signal_processing)

    # 3. Extract joint limits
    _extract_joint_limits(identif_config, joints)

    # 4. Extract problem configuration
    _extract_problem_config(identif_config, problem)

    # 5. Extract mechanical parameters
    _extract_mechanical_params(identif_config, mechanics)

    # 6. Extract coupling parameters
    _extract_coupling_params(identif_config, coupling)

    # 7. Extract load parameters (defaults)
    _extract_load_params(identif_config)

    # 8. Optional physical consistency configuration (v0.4.1, default-off)
    physical_consistency = unified_identif_config.get("physical_consistency", {})
    if isinstance(physical_consistency, dict) and physical_consistency:
        identif_config["physical_consistency"] = physical_consistency

    # 9. Optional reconstruction configuration (v0.4.2, default-off)
    reconstruction = unified_identif_config.get("reconstruction", {})
    if isinstance(reconstruction, dict) and reconstruction:
        identif_config["reconstruction"] = reconstruction

    _extract_selection_config(identif_config, unified_identif_config)

    # 10. Optional held-out validation dataset (separate file/directory,
    # never a split of the training data). Mirrors calibration's
    # tasks.calibration.data.validation_data_file.
    data_section = unified_identif_config.get("data", {})
    if not isinstance(data_section, dict):
        data_section = {}
    identif_config["validation_data_file"] = data_section.get(
        "validation_data_file", ""
    )

    return identif_config
```

## parameter

Parameter management utilities for robot identification.

This module handles parameter extraction, reordering, and management including:

- Inertial parameter extraction and reordering
- Standard additional parameters (friction, actuator inertia, offsets)
- Custom parameter support
- Parameter information queries

## `reorder_inertial_parameters(pinocchio_params)`

Reorder inertial parameters from Pinocchio format to desired format.

Parameters:

| Name               | Type | Description                                                                 | Default    |
| ------------------ | ---- | --------------------------------------------------------------------------- | ---------- |
| `pinocchio_params` |      | Parameters in Pinocchio order [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] | *required* |

Returns:

| Name   | Type | Description                                                               |
| ------ | ---- | ------------------------------------------------------------------------- |
| `list` |      | Parameters in desired order [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m] |

Source code in `src/figaroh/identification/parameter.py`

```
def reorder_inertial_parameters(pinocchio_params):
    """Reorder inertial parameters from Pinocchio format to desired format.

    Args:
        pinocchio_params: Parameters in Pinocchio order
            [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

    Returns:
        list: Parameters in desired order
            [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m]
    """
    if len(pinocchio_params) != 10:
        raise ValueError(
            f"Expected 10 inertial parameters, got {len(pinocchio_params)}"
        )

    # Mapping from Pinocchio indices to desired indices
    reordered = np.zeros_like(pinocchio_params)
    reordered[0] = pinocchio_params[4]  # Ixx
    reordered[1] = pinocchio_params[5]  # Ixy
    reordered[2] = pinocchio_params[7]  # Ixz
    reordered[3] = pinocchio_params[6]  # Iyy
    reordered[4] = pinocchio_params[8]  # Iyz
    reordered[5] = pinocchio_params[9]  # Izz
    reordered[6] = pinocchio_params[1]  # mx
    reordered[7] = pinocchio_params[2]  # my
    reordered[8] = pinocchio_params[3]  # mz
    reordered[9] = pinocchio_params[0]  # m

    return reordered.tolist()
```

## `add_standard_additional_parameters(model, identif_config)`

Add standard additional parameters (actuator inertia, friction, offsets).

Parameters:

| Name             | Type   | Description                  | Default    |
| ---------------- | ------ | ---------------------------- | ---------- |
| `model`          |        | Robot model                  | *required* |
| `identif_config` | `dict` | Identification configuration | *required* |

Returns:

| Name   | Type | Description                             |
| ------ | ---- | --------------------------------------- |
| `dict` |      | Additional parameters with their values |

Source code in `src/figaroh/identification/parameter.py`

```
def add_standard_additional_parameters(model, identif_config):
    """Add standard additional parameters (actuator inertia, friction,
    offsets).

    Args:
        model: Robot model
        identif_config (dict): Identification configuration

    Returns:
        dict: Additional parameters with their values
    """
    phi = []
    params = []

    # Standard additional parameters configuration
    additional_params = [
        {
            "name": "fv",
            "enabled_key": "has_friction",
            "values_key": "fv",
            "default": 0.0,
            "description": "viscous friction",
        },
        {
            "name": "fs",
            "enabled_key": "has_friction",
            "values_key": "fs",
            "default": 0.0,
            "description": "static friction",
        },
        {
            "name": "Ia",
            "enabled_key": "has_actuator_inertia",
            "values_key": "Ia",
            "default": 0.0,
            "description": "actuator inertia",
        },
        {
            "name": "off",
            "enabled_key": "has_joint_offset",
            "values_key": "off",
            "default": 0.0,
            "description": "joint offset",
        },
    ]

    for param_def in additional_params:
        # Disabled blocks get no columns in the regressor, so they get no
        # keys here either; keys and regressor columns must line up.
        if not identif_config.get(param_def["enabled_key"], False):
            continue
        for link_idx, jname in enumerate(model.names[1:]):  # Skip world link
            param_name = f"{param_def['name']}_{jname}"
            params.append(param_name)

            # Get parameter value
            try:
                values_list = identif_config.get(param_def["values_key"], [])
                if len(values_list) >= link_idx + 1:
                    value = values_list[link_idx]
                else:
                    value = param_def["default"]
                    logger.warning(
                        f"Missing {param_def['description']} "
                        f"for joint {jname}, using default: {value}"
                    )
            except (KeyError, IndexError, TypeError) as e:
                value = param_def["default"]
                logger.warning(
                    f"Error getting {param_def['description']} "
                    f"for joint {jname}: {e}, using default: {value}"
                )
            phi.append(value)

    return dict(zip(params, phi))
```

## `add_custom_parameters(model, custom_params)`

Add custom user-defined parameters.

Parameters:

| Name            | Type   | Description                                                                                        | Default    |
| --------------- | ------ | -------------------------------------------------------------------------------------------------- | ---------- |
| `model`         |        | Robot model                                                                                        | *required* |
| `custom_params` | `dict` | Custom parameter definitions Format: {param_name: {values: list, per_joint: bool, default: float}} | *required* |

Returns:

| Name   | Type | Description                         |
| ------ | ---- | ----------------------------------- |
| `dict` |      | Custom parameters with their values |

Source code in `src/figaroh/identification/parameter.py`

```
def add_custom_parameters(model, custom_params):
    """Add custom user-defined parameters.

    Args:
        model: Robot model
        custom_params (dict): Custom parameter definitions
            Format: {param_name: {values: list, per_joint: bool,
            default: float}}

    Returns:
        dict: Custom parameters with their values
    """
    phi = []
    params = []

    for param_name, param_def in custom_params.items():
        if not isinstance(param_def, dict):
            logger.warning(
                f"Invalid custom parameter definition for " f"'{param_name}', skipping"
            )
            continue

        values = param_def.get("values", [])
        per_joint = param_def.get("per_joint", True)
        default_value = param_def.get("default", 0.0)

        if per_joint:
            # Add parameter for each joint
            for link_idx, jname in enumerate(model.names[1:]):
                param_full_name = f"{param_name}_{jname}"
                params.append(param_full_name)

                try:
                    if len(values) >= link_idx + 1:
                        value = values[link_idx]
                    else:
                        value = default_value
                        # Only warn if values were provided but insufficient
                        if values:
                            logger.warning(
                                f"Missing value for custom "
                                f"parameter '{param_name}' joint "
                                f"{jname}, using default: {value}"
                            )
                except (IndexError, TypeError):
                    value = default_value
                    logger.warning(
                        f"Error accessing custom parameter "
                        f"'{param_name}' for joint {jname}, "
                        f"using default: {value}"
                    )

                phi.append(value)
        else:
            # Global parameter (not per joint)
            params.append(param_name)
            try:
                value = values[0] if values else default_value
            except (IndexError, TypeError):
                value = default_value
                logger.warning(
                    f"Error accessing global custom parameter "
                    f"'{param_name}', using default: {value}"
                )

            phi.append(value)

    return dict(zip(params, phi))
```

## `get_standard_parameters(model, identif_config=None)`

Get standard inertial parameters from robot model with extensible parameter support.

Parameters:

| Name             | Type   | Description                      | Default    |
| ---------------- | ------ | -------------------------------- | ---------- |
| `model`          |        | Robot model (Pinocchio model)    | *required* |
| `identif_config` | `dict` | Dictionary of parameter settings | `None`     |

Returns:

| Name   | Type | Description                            |
| ------ | ---- | -------------------------------------- |
| `dict` |      | Parameter names mapped to their values |

Source code in `src/figaroh/identification/parameter.py`

```
def get_standard_parameters(model, identif_config=None):
    """Get standard inertial parameters from robot model with extensible
    parameter support.

    Args:
        model: Robot model (Pinocchio model)
        identif_config (dict, optional): Dictionary of parameter settings

    Returns:
        dict: Parameter names mapped to their values
    """
    if identif_config is None:
        identif_config = {}

    phi = []
    params = []

    # Standard inertial parameter names in desired order
    # inertial_params = [
    #     "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz",
    #     "mx", "my", "mz", "m"
    # ]
    inertial_params = [
        "m",
        "mx",
        "my",
        "mz",
        "Ixx",
        "Ixy",
        "Iyy",
        "Ixz",
        "Iyz",
        "Izz",
    ]

    # Extract and rearrange inertial parameters for each link
    assert len(model.inertias) == model.njoints, "Inertia count mismatch with joints"
    for link_idx, jname in enumerate(model.names[1:]):
        # Get dynamic parameters from Pinocchio (in Pinocchio order)
        # Returns the representation of the matrix as a vector of dynamic
        # parameters. The parameters are given as 𝑣=[𝑚,𝑚𝑐𝑥,𝑚𝑐𝑦,𝑚𝑐𝑧,
        # 𝐼𝑥𝑥,𝐼𝑥𝑦,𝐼𝑦𝑦,𝐼𝑥𝑧,𝐼𝑦𝑧,𝐼𝑧𝑧]^𝑇 where 𝑐 is the center
        # of mass, 𝐼=𝐼𝐶+𝑚𝑆𝑇(𝑐)𝑆(𝑐) and 𝐼𝐶 has its origin at the
        # barycenter and 𝑆(𝑐) is the the skew matrix representation of the
        # cross product operator from Vector of spatial inertias supported by
        # each joint.
        pinocchio_params = model.inertias[link_idx + 1].toDynamicParameters()

        # Rearrange from Pinocchio order [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz,
        # Iyz, Izz] to desired order [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my,
        # mz, m]
        # reordered_params = reorder_inertial_parameters(pinocchio_params)
        reordered_params = pinocchio_params
        # Add parameter names and values
        for param_name in inertial_params:
            params.append(f"{param_name}_{jname}")
        phi.extend(reordered_params)

    return dict(zip(params, phi))
```

## `get_parameter_info()`

Get information about available parameter types.

Returns:

| Name   | Type | Description                                           |
| ------ | ---- | ----------------------------------------------------- |
| `dict` |      | Information about standard and custom parameter types |

Source code in `src/figaroh/identification/parameter.py`

```
def get_parameter_info():
    """Get information about available parameter types.

    Returns:
        dict: Information about standard and custom parameter types
    """
    return {
        "inertial_parameters": [
            # "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz",
            # "mx", "my", "mz", "m"
            "m",
            "mx",
            "my",
            "mz",
            "Ixx",
            "Ixy",
            "Iyy",
            "Ixz",
            "Iyz",
            "Izz",
        ],
        "standard_additional": {
            "viscous_friction": {
                "name": "fv",
                "enabled_key": "has_friction",
                "values_key": "fv",
                "description": "Viscous friction coefficient",
            },
            "static_friction": {
                "name": "fs",
                "enabled_key": "has_friction",
                "values_key": "fs",
                "description": "Static friction coefficient",
            },
            "actuator_inertia": {
                "name": "Ia",
                "enabled_key": "has_actuator_inertia",
                "values_key": "Ia",
                "description": "Actuator/rotor inertia",
            },
            "joint_offset": {
                "name": "off",
                "enabled_key": "has_joint_offset",
                "values_key": "off",
                "description": "Joint position offset",
            },
        },
        "custom_parameters_format": {
            "parameter_name": {
                "values": "list of values",
                "per_joint": "boolean - if True, creates param for each joint",
                "default": "default value if not enough values provided",
            }
        },
    }
```

## physical_fit

Direct LMI effort fit over all standard parameters (public since #61; `figaroh.identification._physical_comparator` remains as an alias).

Physically consistent parameter recovery by convex (LMI) effort fits.

Three *different* objectives are available on one shared problem; none of them silently falls back to another:

1. :func:`solve_exact_reconstruction` - closest-to-prior theta that reproduces the OLS base parameters exactly (`M_full theta = phi_ols`) with a pseudo-inertia LMI per link.
1. :func:`solve_direct_effort_fit` - regularised effort fit with the same LMIs and bounds, no base equality. This is the estimator that `select_stage: physical_fit` selects in :class:`BaseIdentification`.
1. :func:`solve_per_link_projection` - nullspace representative of `phi_ols` followed by an independent LMI projection per link.

:func:`diagnose_exact` runs staged diagnostics (D0..D3) to explain why objective 1 fails on a given problem.

Typical use::

```
problem = build_problem(Y, tau, params_std, joint_names, prior=prior)
record = solve_direct_effort_fit(problem, solver="cvxopt")
if record.accepted:
    theta = record.theta  # aligned to ``problem.params_std``
```

Every solve returns a :class:`SolveRecord` that states the solver status, the base residual and per-link feasibility, so a caller can decide whether the estimate is usable. The module was promoted from the private `_physical_comparator`; that name remains as a re-export shim.

## `FixedExtras(names=(), values=(), source='zero')`

Parameters held fixed (e.g. friction); their effort is subtracted.

`names` must be column names of `Y_std`; `source` records where `values` came from (`truth`, `zero` or `declared`).

## `PhysicalPolicy(mass_min=1e-06, mass_ratio_bounds=None, feas_tol=-1e-08, base_rel_tol=1e-06, prior_weight=1e-06)`

Bounds and tolerances shared by all objectives.

`mass_min` lower-bounds every link mass, `feas_tol` is the pseudo-inertia eigenvalue tolerance of the acceptance test, `base_rel_tol` the relative base residual allowed for the exact objective and `prior_weight` the Tikhonov weight towards the prior.

## `ComparatorProblem(params_std, joint_names, Y, tau, row_weight, theta_prior, coord_scale, M_full, phi_ols, R, Qt_tau, extras, policy, theta_truth=None, base_indices=(), removed_columns=())`

Shared problem data built by :func:`build_problem`.

`Y`/`tau` are the (unweighted) regressor and effort over `params_std`; `row_weight` the per-row weights; `M_full`/`phi_ols` the base mapping and OLS base parameters used by the exact objective.

## `SolveRecord(objective, solver, solver_status, picos_status, exception, runtime_s, objective_value, theta, fallback_used, fallback_candidate, base_residual, effort_rmse_fit_per_joint, feasibility, accepted, truth_error, inputs_hash, notes=dict())`

Outcome of one solve: status, estimate, residuals and acceptance.

`accepted` is true only for an optimal solve whose every link passes the pseudo-inertia feasibility test (and, for the exact objective, whose relative base residual is within tolerance). `theta` is `None` for a non-optimal solve; no estimate is substituted.

## `build_problem(Y_std, tau, params_std, joint_names, *, prior, extras=None, policy=None, row_weight=None, qr_tol=1e-06, theta_truth=None)`

Assemble the shared problem.

`prior`/`theta_truth` are given over `params_std`. Columns named in `extras` are removed from the unknowns and their effort is subtracted from `tau`. Columns removed by the QR step stay in `M_full` as zeros so every link keeps its LMI.

Source code in `src/figaroh/identification/physical_fit.py`

```
def build_problem(
    Y_std: np.ndarray,
    tau: np.ndarray,
    params_std: Sequence[str],
    joint_names: Sequence[str],
    *,
    prior: np.ndarray,
    extras: Optional[FixedExtras] = None,
    policy: Optional[PhysicalPolicy] = None,
    row_weight: Optional[np.ndarray] = None,
    qr_tol: float = 1e-6,
    theta_truth: Optional[np.ndarray] = None,
) -> ComparatorProblem:
    """Assemble the shared problem.

    ``prior``/``theta_truth`` are given over ``params_std``. Columns named in
    ``extras`` are removed from the unknowns and their effort is subtracted
    from ``tau``. Columns removed by the QR step stay in ``M_full`` as zeros
    so every link keeps its LMI.
    """
    extras = extras or FixedExtras()
    policy = policy or PhysicalPolicy()
    Y_std = np.asarray(Y_std, dtype=float)
    tau = np.asarray(tau, dtype=float).reshape(-1)
    names = list(params_std)
    prior = np.asarray(prior, dtype=float).reshape(len(names))
    truth = (
        None
        if theta_truth is None
        else np.asarray(theta_truth, dtype=float).reshape(len(names))
    )
    if Y_std.shape != (tau.size, len(names)):
        raise ValueError(
            f"Y_std shape {Y_std.shape} inconsistent with tau {tau.shape} "
            f"and {len(names)} parameters"
        )

    # Freeze extras
    tau = tau.copy()
    keep = np.ones(len(names), dtype=bool)
    for nm, val in zip(extras.names, extras.values):
        j = names.index(nm)
        tau = tau - Y_std[:, j] * float(val)
        keep[j] = False
    names = [n for n, k in zip(names, keep) if k]
    Y = Y_std[:, keep]
    prior = prior[keep]
    truth = None if truth is None else truth[keep]

    w = (
        np.ones(tau.size)
        if row_weight is None
        else np.asarray(row_weight, dtype=float).reshape(tau.size)
    )
    Yw = w[:, None] * Y
    tw = w * tau
    n = len(names)

    # Core QR decomposer on the non-zero columns; removed columns stay zero.
    colnorm = np.linalg.norm(Yw, axis=0)
    thresh = qr_tol * max(float(colnorm.max()), 1e-300)
    kept = [i for i in range(n) if colnorm[i] > thresh]
    removed = tuple(i for i in range(n) if i not in kept)
    params_r = [names[i] for i in kept]
    dec = QRDecomposer(tolerance=qr_tol)
    M_r, _, base_r, _ = dec.get_base_mapping_matrix_double(Yw[:, kept], params_r)
    M_full = QRDecomposer.expand_mapping_matrix_to_full(M_r, params_r, names)
    base_full = tuple(kept[i] for i in base_r)
    phi_ols, *_ = np.linalg.lstsq(Yw[:, list(base_full)], tw, rcond=None)

    Q, R = np.linalg.qr(Yw, mode="reduced")
    return ComparatorProblem(
        params_std=tuple(names),
        joint_names=tuple(joint_names),
        Y=Y,
        tau=tau,
        row_weight=w,
        theta_prior=prior,
        coord_scale=_coord_scale(names, joint_names, prior),
        M_full=M_full,
        phi_ols=phi_ols,
        R=R,
        Qt_tau=Q.T @ tw,
        extras=extras,
        policy=policy,
        theta_truth=truth,
        base_indices=base_full,
        removed_columns=removed,
    )
```

## `solve_exact_reconstruction(p, *, solver='cvxopt', **opts)`

Objective 1: min ||D(theta-theta0)|| s.t. M_full theta = phi_ols, LMIs.

Source code in `src/figaroh/identification/physical_fit.py`

```
def solve_exact_reconstruction(
    p: ComparatorProblem, *, solver: str = "cvxopt", **opts: Any
) -> SolveRecord:
    """Objective 1: min ||D(theta-theta0)|| s.t. M_full theta = phi_ols, LMIs."""
    raw = _solve_core(p, "exact", solver=solver, **opts)
    return _make_record(p, "exact_reconstruction", solver=solver, raw=raw, exact=True)
```

## `solve_direct_effort_fit(p, *, solver='cvxopt', **opts)`

Objective 2: regularised effort fit with the same LMIs, no equality.

Source code in `src/figaroh/identification/physical_fit.py`

```
def solve_direct_effort_fit(
    p: ComparatorProblem, *, solver: str = "cvxopt", **opts: Any
) -> SolveRecord:
    """Objective 2: regularised effort fit with the same LMIs, no equality."""
    raw = _solve_core(p, "direct", solver=solver, **opts)
    return _make_record(p, "direct_effort_fit", solver=solver, raw=raw)
```

## `solve_per_link_projection(p, *, solver='cvxopt')`

Objective 3: nullspace representative, then per-link LMI projection.

Source code in `src/figaroh/identification/physical_fit.py`

```
def solve_per_link_projection(
    p: ComparatorProblem, *, solver: str = "cvxopt"
) -> SolveRecord:
    """Objective 3: nullspace representative, then per-link LMI projection."""
    t0 = time.perf_counter()
    raw: Dict[str, Any] = {
        "status": "error",
        "picos_status": None,
        "exception": None,
        "objective_value": None,
        "theta": None,
    }
    try:
        D = p.coord_scale
        A = p.M_full * D[None, :]
        z = np.linalg.pinv(A) @ (p.phi_ols - p.M_full @ p.theta_prior)
        theta = p.theta_prior + D * z
        total = 0.0
        for j, idx in _link_indices(p).items():
            cols = [idx[k] for k in _INERTIAL_KEYS]
            p10, rep = project_p10_lmi(
                theta[cols],
                mass_min=p.policy.mass_min,
                psd_eig_tol=p.policy.feas_tol,
                solver=solver,
                mass_bounds=_mass_bounds(p, idx),
            )
            if rep.status == "error":
                raise RuntimeError(f"link {j}: {rep.message}")
            theta[cols] = p10
            total += float(rep.objective or 0.0)
        raw.update(status="optimal", picos_status="optimal", theta=theta)
        raw["objective_value"] = total
    except Exception as exc:
        raw["exception"] = f"{type(exc).__name__}: {exc}"
        raw["theta"] = None
    raw["runtime_s"] = time.perf_counter() - t0
    return _make_record(p, "per_link_projection", solver=solver, raw=raw)
```

## `diagnose_exact(p, *, solver='cvxopt')`

Staged diagnostics for objective 1 (D0 reproduce, D1 phase-I, D2 one-change variants, D3 relaxed equality).

Source code in `src/figaroh/identification/physical_fit.py`

```
def diagnose_exact(
    p: ComparatorProblem, *, solver: str = "cvxopt"
) -> List[SolveRecord]:
    """Staged diagnostics for objective 1 (D0 reproduce, D1 phase-I,
    D2 one-change variants, D3 relaxed equality)."""
    records: List[SolveRecord] = []

    # D0: reproduce with the production reconstruction entry point.
    t0 = time.perf_counter()
    raw: Dict[str, Any] = {
        "status": "error",
        "picos_status": None,
        "exception": None,
        "theta": None,
    }
    try:
        res = reconstruct_full_parameters(
            (p.M_full, p.phi_ols, list(p.params_std)),
            method="sdp",
            theta0=p.theta_prior,
            weights=1.0 / p.coord_scale,
            joint_names=list(p.joint_names),
            mass_min=p.policy.mass_min,
            solver=solver,
        )
        raw["objective_value"] = res.objective
        if res.status == "ok":
            raw.update(status="optimal", theta=res.theta_r)
        else:
            # Legacy code silently fell back; keep it only as a candidate.
            raw["status"] = "error"
            raw["exception"] = f"reconstruct_full_parameters status={res.status}"
            raw["fallback_candidate"] = res.theta_r
    except Exception as exc:
        raw["exception"] = f"{type(exc).__name__}: {exc}"
    raw["runtime_s"] = time.perf_counter() - t0
    records.append(
        _make_record(
            p, "D0:reconstruct_full_parameters_sdp", solver=solver, raw=raw, exact=True
        )
    )

    # D1: phase-I feasibility.
    raw = _solve_core(p, "phase1", solver=solver, bounds=False)
    s_star = raw.get("s_star")
    notes: Dict[str, Any] = {"s_star": s_star}
    if raw["status"] == "optimal" and s_star is not None:
        notes["label"] = "infeasible (certificate s*<0)" if s_star < 0 else "feasible"
        if s_star < 0:
            raw["theta"] = None  # a strictly infeasible point is not a solution
    else:
        notes["label"] = "solver failure"
    rec = _make_record(p, "D1:phase1", solver=solver, raw=raw, exact=True, notes=notes)
    rec.objective_value = s_star
    rec.accepted = False
    records.append(rec)

    # D2: one change at a time.
    variants = {
        "schur_norm": {"norm": "schur"},
        "scaling_off": {"scaling": False},
        "params_r_only": {"variables": "params_r"},
        "bounds_off": {"bounds": False},
    }
    for name, opts in variants.items():
        raw = _solve_core(p, "exact", solver=solver, **opts)
        records.append(
            _make_record(p, f"D2:{name}", solver=solver, raw=raw, exact=True)
        )

    # D3: relaxed equality.
    raw = _solve_core(p, "relaxed", solver=solver)
    records.append(_make_record(p, "D3:relaxed_equality", solver=solver, raw=raw))
    return records
```

## selection

Which estimate an identification run reports (#61).

`tasks.identification.select_stage` picks the estimate that validation, `verify()`, the reports, the run archive and the URDF export use:

- `fit` (default): the base-parameter least-squares fit, unchanged.
- `reconstruction`: the full parameter vector reconstructed from the fit, accepted only when it is physically consistent and was produced by the requested method.
- `physical_fit`: the direct LMI effort fit (:func:`figaroh.identification.physical_fit.solve_direct_effort_fit`).

A requested stage that does not meet its acceptance conditions is *rejected*: nothing else is substituted, the selected stage is `none` and the verdict fails. The fit numbers stay under their usual result keys.

## `SelectedEstimate(stage, requested, status, reason, space, names=list(), values=None, phi_base_equivalent=None, base_residual_rel=None, feasibility=dict(), effective_method=None, solvers=list(), psd_eig_tol=-1e-10, extra=dict())`

The estimate chosen by `select_stage` and the verdict on it.

Attributes:

| Name                  | Type                        | Description                                                                                              |
| --------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------- |
| `stage`               | `str`                       | Stage whose numbers are reported: fit, reconstruction or physical_fit when accepted, none when rejected. |
| `requested`           | `str`                       | The stage that was asked for.                                                                            |
| `status`              | `str`                       | accepted or rejected.                                                                                    |
| `reason`              | `str`                       | Why, in one line (names the solver(s) and the failed test).                                              |
| `space`               | `str`                       | standard (values are the full standard parameter vector).                                                |
| `names`               | `List[str]`                 | Parameter names aligned with values.                                                                     |
| `values`              | `Optional[ndarray]`         | The candidate vector; kept for diagnosis when rejected.                                                  |
| `phi_base_equivalent` | `Optional[ndarray]`         | M @ theta for a standard candidate (diagnostic only, never used for prediction).                         |
| `base_residual_rel`   | `Optional[float]`           | Relative distance of phi_base_equivalent from the fitted base parameters.                                |
| `feasibility`         | `Dict[str, Dict[str, Any]]` | Per link {mass, min_eig, ok} for standard candidates.                                                    |
| `effective_method`    | `Optional[str]`             | Method that produced a reconstruction.                                                                   |
| `solvers`             | `List[str]`                 | Solver names tried for a physical fit.                                                                   |
| `psd_eig_tol`         | `float`                     | Pseudo-inertia eigenvalue tolerance the links were judged feasible with; export uses the same one.       |

### `predict(W_full)`

Effort predicted by this estimate: the full regressor times the standard parameter vector (never `phi_base_equivalent`).

Source code in `src/figaroh/identification/selection.py`

```
def predict(self, W_full: np.ndarray) -> np.ndarray:
    """Effort predicted by this estimate: the full regressor times the
    standard parameter vector (never ``phi_base_equivalent``)."""
    if self.values is None:
        raise ValueError("no estimate values to predict with")
    return W_full @ self.values
```

### `parameter_dict()`

`{name: value}` of the candidate (empty when no values).

Source code in `src/figaroh/identification/selection.py`

```
def parameter_dict(self) -> Dict[str, float]:
    """``{name: value}`` of the candidate (empty when no values)."""
    if self.values is None:
        return {}
    return {n: float(v) for n, v in zip(self.names, self.values)}
```

### `link_p10(joint_names)`

Per-link 10-vectors of a standard-space estimate.

Source code in `src/figaroh/identification/selection.py`

```
def link_p10(self, joint_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per-link 10-vectors of a standard-space estimate."""
    if self.space != "standard":
        raise ValueError("link inertias need a standard-space estimate")
    from figaroh.identification.physical_consistency import (
        p10_by_joint_from_param_dict,
    )

    return p10_by_joint_from_param_dict(
        parameter_dict=self.parameter_dict(), joint_names=joint_names
    )
```

## `requested_stage(identif)`

The configured `select_stage` (`fit` when absent).

Source code in `src/figaroh/identification/selection.py`

```
def requested_stage(identif) -> str:
    """The configured ``select_stage`` (``fit`` when absent)."""
    cfg = getattr(identif, "identif_config", None) or {}
    return str(cfg.get("select_stage") or DEFAULT_STAGE)
```

## `physical_fit_config(identif)`

The `physical_fit` settings with every key present.

The config parser already merges the defaults; a programmatic config may hold a partial mapping (or none), which is completed here.

Source code in `src/figaroh/identification/selection.py`

```
def physical_fit_config(identif) -> Dict[str, Any]:
    """The ``physical_fit`` settings with every key present.

    The config parser already merges the defaults; a programmatic config
    may hold a partial mapping (or none), which is completed here.
    """
    from figaroh.identification.config import PHYSICAL_FIT_DEFAULTS

    raw = (getattr(identif, "identif_config", None) or {}).get("physical_fit")
    if isinstance(raw, dict) and PHYSICAL_FIT_DEFAULTS.keys() <= raw.keys():
        return dict(raw)
    cfg = dict(PHYSICAL_FIT_DEFAULTS)
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg
```

## `run_physical_fit(identif)`

Solve the direct LMI effort fit on the data the fit used.

Returns an outcome dict (never raises): `record` (a :class:`~figaroh.identification.physical_fit.SolveRecord` or `None`), `solvers` (names tried), `reason` and `accepted`. The processed regressor is scattered into all standard columns (zeros where columns were eliminated); friction and offsets are free parameters and the prior is the nominal standard parameter vector.

Source code in `src/figaroh/identification/selection.py`

```
def run_physical_fit(identif) -> Dict[str, Any]:
    """Solve the direct LMI effort fit on the data the fit used.

    Returns an outcome dict (never raises): ``record`` (a
    :class:`~figaroh.identification.physical_fit.SolveRecord` or ``None``),
    ``solvers`` (names tried), ``reason`` and ``accepted``. The processed
    regressor is scattered into all standard columns (zeros where columns
    were eliminated); friction and offsets are free parameters and the
    prior is the nominal standard parameter vector.
    """
    cfg = physical_fit_config(identif)
    solvers = [str(cfg["solver"])]
    outcome: Dict[str, Any] = {
        "record": None,
        "solvers": solvers,
        "reason": "",
        "accepted": False,
        "config": cfg,
    }
    try:
        from figaroh.identification import physical_fit as pf

        names = list(identif.standard_parameter.keys())
        n = len(names)
        W = np.asarray(identif._solve_W, dtype=float)
        tau = np.asarray(identif._solve_tau, dtype=float).reshape(-1)
        eliminated = set(int(i) for i in (identif._idx_eliminated or []))
        keep = [i for i in range(n) if i not in eliminated]
        if W.shape[1] != len(keep):
            raise ValueError(
                f"processed regressor has {W.shape[1]} columns, expected {len(keep)}"
            )
        Y = np.zeros((W.shape[0], n))
        Y[:, keep] = W
        problem = pf.build_problem(
            Y,
            tau,
            names,
            list(identif.model.names[1:]),
            prior=_nominal_vector(identif),
            policy=pf.PhysicalPolicy(
                mass_min=float(cfg["mass_min"]),
                prior_weight=float(cfg["prior_weight"]),
            ),
            row_weight=getattr(identif, "_wls_row_weight", None),
        )
        opts = {}
        if cfg["max_seconds"] is not None:
            opts["max_seconds"] = float(cfg["max_seconds"])
        record = pf.solve_direct_effort_fit(problem, solver=solvers[0], **opts)
        steps = [f"{solvers[0]}: {record.solver_status}"]
        if record.solver_status != "optimal" and cfg["second_solver"]:
            second = str(cfg["second_solver"])
            solvers.append(second)
            record = pf.solve_direct_effort_fit(problem, solver=second, **opts)
            steps.append(f"{second}: {record.solver_status}")
        outcome["record"] = record
        outcome["psd_eig_tol"] = problem.policy.feas_tol
        outcome["accepted"] = bool(
            record.solver_status == "optimal" and record.accepted
        )
        verdict = (
            "all links feasible"
            if outcome["accepted"]
            else (
                "a link is not physically feasible"
                if record.solver_status == "optimal"
                else "no optimal solution"
            )
        )
        outcome["reason"] = "; ".join(steps) + f"; {verdict}"
    except Exception as exc:  # solver stack missing, shape problems, ...
        logger.warning("physical fit failed: %s", exc)
        outcome["reason"] = f"{type(exc).__name__}: {exc}"
    return outcome
```

## `select_estimate(identif)`

Build the :class:`SelectedEstimate` for the configured stage.

Source code in `src/figaroh/identification/selection.py`

```
def select_estimate(identif) -> SelectedEstimate:
    """Build the :class:`SelectedEstimate` for the configured stage."""
    stage = requested_stage(identif)
    if stage == DEFAULT_STAGE:
        raise ValueError("select_stage='fit' reports the base fit; nothing to select")
    if stage == "reconstruction":
        return _select_reconstruction(identif)
    if stage == "physical_fit":
        return _select_physical_fit(identif)
    raise ValueError(f"select_stage={stage!r} is not selectable")
```

## Physical inertial conventions

Physical-consistency utilities use Pinocchio dynamic-parameter order: `[m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`. The first moments are `h = m*c`; the six tensor entries describe rotational inertia `I_O` about the link-frame origin. They are distinct from inertia `I_C` about the centre of mass: `I_O = I_C + m * ((c.T*c)*eye(3) - c*c.T)`.

The physical constraint is positive semidefiniteness of the pseudo-inertia `P = [[Sigma, h], [h.T, m]]`, with `Sigma = 0.5*trace(I_O)*eye(3) - I_O`. Projection uses this second-moment block as its decision variable and converts back using `I_O = trace(Sigma)*eye(3) - Sigma`. Its objective and weights apply to the ten dynamic parameters. SDP reconstruction imposes the same pseudo-inertia constraint while preserving its base-parameter equalities.

The existing CAD `com_bounds` option bounds first moments `h`, in kg\*m, rather than CoM coordinates in metres. Physical feasibility does not establish identifiability or complete inertial URDF export.
