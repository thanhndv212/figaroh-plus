# Optimal

Optimal calibration and trajectory optimization module.

This module provides base classes for optimal calibration and trajectory optimization for robotic systems.

## `BaseOptimalCalibration(robot, config_file='config/robot_config.yaml')`

Bases: `ABC`

Base class for robot optimal configuration generation for calibration.

This class implements the framework for generating optimal robot configurations that maximize the observability of kinematic parameters during calibration. It uses Second-Order Cone Programming (SOCP) to solve the D-optimal design problem for parameter estimation.

The class provides a Template Method pattern where the main workflow is defined, but specific optimization strategies can be customized by derived classes for different robot types.

Workflow

1. Load candidate configurations from file (CSV or YAML)
1. Calculate kinematic regressors for all candidates
1. Compute information matrices for each configuration
1. Solve SOCP optimization to find optimal subset
1. Select configurations with significant weights
1. Visualize and save results

Key Features

- D-optimal experimental design for calibration
- Support for multiple calibration models (full_params, joint_offset)
- Automatic minimum configuration calculation
- SOCP-based optimization with convex relaxation
- Comprehensive visualization and analysis tools
- File I/O for configuration management

Mathematical Background

The method maximizes the determinant of the Fisher Information Matrix: max det(Σᵢ wᵢ Rᵢᵀ Rᵢ) subject to Σᵢ wᵢ ≤ 1, wᵢ ≥ 0

Where:

- Rᵢ is the kinematic regressor for configuration i
- wᵢ is the weight assigned to configuration i
- The objective maximizes parameter estimation precision

Attributes:

| Name                     | Type      | Description                                    |
| ------------------------ | --------- | ---------------------------------------------- |
| `robot`                  |           | Robot model instance loaded with FIGAROH       |
| `model`                  |           | Pinocchio robot model                          |
| `data`                   |           | Pinocchio robot data                           |
| `calib_config`           | `dict`    | Calibration parameters from configuration file |
| `optimal_configurations` | `dict`    | Selected optimal configurations                |
| `optimal_weights`        | `ndarray` | Weights assigned to configurations             |
| `minNbChosen`            | `int`     | Minimum number of configurations required      |
| `R_rearr`                | `ndarray` | Rearranged kinematic regressor matrix          |
| `detroot_whole`          | `float`   | Determinant root of full information matrix    |
| `w_list`                 | `list`    | Solution weights from SOCP optimization        |
| `w_dict_sort`            | `dict`    | Sorted weights by configuration index          |

Example

> > > ### Basic usage for TIAGo robot
> > >
> > > from figaroh.robots import TiagoRobot robot = TiagoRobot()
> > >
> > > ### Create optimal calibration instance
> > >
> > > opt_calib = TiagoOptimalCalibration(robot, "config/tiago.yaml")
> > >
> > > ### Generate optimal configurations
> > >
> > > opt_calib.solve(save_file=True)
> > >
> > > ### Access results
> > >
> > > print(f"Selected {len(opt_calib.optimal_configurations)} configs") print(f"Minimum required: {opt_calib.minNbChosen}")

See Also

BaseCalibration: Main calibration framework SOCPOptimizer: Second-order cone programming solver TiagoOptimalCalibration: TIAGo-specific implementation UR10OptimalCalibration: UR10-specific implementation

Initialize optimal calibration with robot model and configuration.

Sets up the optimal calibration framework by loading robot parameters, initializing optimization attributes, and calculating the minimum number of configurations required based on the calibration model.

The minimum number of configurations is computed to ensure the optimization problem is well-posed and identifiable:

- For full_params: considers all kinematic parameters
- For joint_offset: considers only joint offset parameters

Parameters:

| Name          | Type  | Description                                                                                                                                 | Default                      |
| ------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `robot`       |       | Robot model instance loaded with FIGAROH. Must have 'model' and 'data' attributes for Pinocchio integration.                                | *required*                   |
| `config_file` | `str` | Path to YAML configuration file containing calibration parameters, sample file paths, and optimization settings. Defaults to standard path. | `'config/robot_config.yaml'` |

Raises:

| Type                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `FileNotFoundError` | If config_file does not exist                     |
| `KeyError`          | If required parameters missing from configuration |
| `ValueError`        | If calibration model type is unsupported          |

Side Effects

- Loads and stores calibration parameters in self.calib_config
- Calculates minimum required configurations (self.minNbChosen)
- Initializes optimization result attributes to None
- Prints initialization confirmation message

Example

> > > robot = TiagoRobot() opt_calib = BaseOptimalCalibration(robot, "config/tiago.yaml") TiagoOptimalCalibration initialized

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
    self._config_file = config_file
    self.load_param(config_file)

    # Initialize attributes for optimal calibration
    self.optimal_configurations = None
    self.optimal_weights = None
    self._sampleConfigs_file = self.calib_config.get("sample_configs_file")
    # load_param() only reads the "calibration" task, so the optimal task's
    # own output settings are picked up separately here.
    self._optimal_output_file = self._load_optimal_output_file(config_file)

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
```

### `initialize()`

Initialize the optimization process by preparing all required data.

This method orchestrates the initialization sequence required before optimization can begin. It ensures all mathematical components are properly computed and cached for efficient optimization.

The initialization sequence:

1. Load candidate configurations from external files
1. Calculate kinematic regressors for all configurations
1. Compute determinant root of the full information matrix

Prerequisites

- Robot model and parameters must be loaded
- Configuration file must specify valid sample data paths

Side Effects

- Sets self.q_measured with candidate joint configurations
- Sets self.R_rearr with rearranged kinematic regressor
- Sets self.\_subX_dict and self.\_subX_list with info matrices
- Sets self.detroot_whole with full matrix determinant root

Raises:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `ValueError`     | If sample configuration file is invalid or missing |
| `AssertionError` | If regressor calculation fails                     |

See Also

load_candidate_configurations: Configuration data loading calculate_regressor: Kinematic regressor computation calculate_detroot_whole: Information matrix analysis

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `solve(save_file=False, selection_method='threshold', **selection_kwargs)`

Solve the optimal configuration selection problem.

This is the main entry point that orchestrates the complete optimal configuration generation workflow. It automatically handles initialization if not already performed, solves the SOCP optimization, and provides comprehensive results analysis.

The method implements the complete D-optimal design workflow:

1. Initialize data and regressors (if needed)
1. Solve SOCP optimization for optimal weights
1. Select configurations with significant weights
1. Optionally save results to files
1. Generate visualization plots

Parameters:

| Name                 | Type   | Description                                                                                                    | Default       |
| -------------------- | ------ | -------------------------------------------------------------------------------------------------------------- | ------------- |
| `save_file`          | `bool` | Whether to save optimal configurations to YAML file in results directory. Default False.                       | `False`       |
| `selection_method`   | `str`  | "threshold" (default) or "iroc" -- see calculate_optimal_configurations for what each one does.                | `'threshold'` |
| `**selection_kwargs` |        | Forwarded to calculate_optimal_configurations (e.g. iroc_rel_tol, iroc_patience when selection_method="iroc"). | `{}`          |

Side Effects

- Updates self.optimal_configurations with selected configs
- Updates self.optimal_weights with optimization weights
- Creates visualization plots
- May create output files if save_file=True
- Prints progress and results to console

Raises:

| Type             | Description                                  |
| ---------------- | -------------------------------------------- |
| `AssertionError` | If minimum configuration requirement not met |
| `ValueError`     | If optimization problem is infeasible        |
| `IOError`        | If file saving fails (logged as warning)     |

Example

> > > opt_calib = TiagoOptimalCalibration(robot) opt_calib.solve(save_file=True) 12 configs are chosen: [0, 5, 12, 18, ...] Optimal configurations written to file successfully

See Also

initialize: Data preparation workflow calculate_optimal_configurations: Core optimization solver plot: Results visualization save_results: File output management

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def solve(self, save_file=False, selection_method="threshold", **selection_kwargs):
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
        selection_method (str): "threshold" (default) or "iroc" -- see
                        calculate_optimal_configurations for what each
                        one does.
        **selection_kwargs: Forwarded to calculate_optimal_configurations
                        (e.g. iroc_rel_tol, iroc_patience when
                        selection_method="iroc").

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
    self.calculate_optimal_configurations(
        selection_method=selection_method, **selection_kwargs
    )
    if save_file:
        try:
            self.save_results()
            logger.info("Optimal configurations written to file successfully")
        except Exception as e:
            logger.warning(f"Could not write to file: {e}")
    self.plot()
```

### `load_param(config_file, setting_type='calibration')`

Load calibration parameters from YAML configuration file.

Only the unified configuration format is supported; a legacy flat config raises an error naming `figaroh.utils.config_migration`.

Parameters:

| Name           | Type  | Description                          | Default         |
| -------------- | ----- | ------------------------------------ | --------------- |
| `config_file`  | `str` | Path to a unified configuration file | *required*      |
| `setting_type` | `str` | Configuration section to load        | `'calibration'` |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def load_param(self, config_file: str, setting_type: str = "calibration"):
    """Load calibration parameters from YAML configuration file.

    Only the unified configuration format is supported; a legacy flat
    config raises an error naming ``figaroh.utils.config_migration``.

    Args:
        config_file (str): Path to a unified configuration file
        setting_type (str): Configuration section to load
    """
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

### `load_candidate_configurations()`

Load candidate joint configurations from external data files.

Reads robot joint configurations from CSV or YAML files that serve as the candidate pool for optimization. The method supports multiple file formats and automatically updates the sample count parameter.

Supported formats:

- CSV: Standard measurement data format with joint configurations
- YAML: Structured format with named joints and configurations

The YAML format expects:

```
calibration_joint_names: [joint1, joint2, ...]
calibration_joint_configurations: [[q1_1, q1_2, ...], [q2_1, ...]]
```

Side Effects

- Sets self.q_measured with loaded joint configurations
- Updates self.calib_config["NbSample"] with actual sample count
- May load self.\_configs for YAML format data

Raises:

| Type                | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `ValueError`        | If sample_configs_file not specified in configuration or if file format is not supported |
| `FileNotFoundError` | If specified data file does not exist                                                    |

Example

> > > #### Assuming config specifies "data/candidate_configs.yaml"
> > >
> > > opt_calib.load_candidate_configurations() print(opt_calib.q_measured.shape) # (1000, 7) for TIAGo

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

````
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
````

### `calculate_regressor()`

Calculate kinematic regressors and information matrices.

Computes the kinematic regressor matrices that relate kinematic parameter variations to end-effector pose changes. This is the mathematical foundation for the optimization problem.

The method performs several key computations:

1. Calculate base kinematic regressors for all configurations
1. Rearrange regressor matrix by sample order for efficiency
1. Compute individual information matrices for each configuration
1. Store results for optimization access

Mathematical Background

For each configuration i, the regressor Rᵢ satisfies: δx = Rᵢ δθ where δx is pose variation and δθ is parameter variation.

The information matrix is: Xᵢ = RᵢᵀRᵢ

Side Effects

- Sets self.R_rearr with rearranged kinematic regressor
- Sets self.\_subX_list with list of information matrices
- Sets self.\_subX_dict with indexed information matrices
- Prints parameter names for verification

Returns:

| Name   | Type | Description                    |
| ------ | ---- | ------------------------------ |
| `bool` |      | True if calculation successful |

Prerequisites

- Joint configurations must be loaded (self.q_measured)
- Robot model and parameters must be initialized

See Also

calculate_base_kinematics_regressor: Core regressor computation rearrange_rb: Matrix rearrangement for optimization sub_info_matrix: Information matrix decomposition

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `calculate_detroot_whole()`

Calculate determinant root of complete information matrix.

Computes the determinant root of the full Fisher Information Matrix formed by all candidate configurations. This serves as the theoretical upper bound for the D-optimality criterion and is used for performance comparison.

Mathematical Background

M_full = R^T R (full regressor) detroot_whole = det(M_full)^(1/n) / sqrt(n)

This represents the geometric mean of eigenvalues, normalized by matrix dimension for scale independence.

Side Effects

- Sets self.detroot_whole with computed determinant root
- Prints the computed value for verification

Prerequisites

- Kinematic regressor must be calculated (self.R_rearr)
- Requires picos library for determinant computation

Raises:

| Type             | Description                                  |
| ---------------- | -------------------------------------------- |
| `AssertionError` | If regressor calculation not performed first |
| `ImportError`    | If picos library not available               |

See Also

calculate_regressor: Prerequisites for this computation plot: Uses this value for performance comparison

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `rearrange_rb(R_b, calib_config)`

rearrange the kinematic regressor by sample numbered order

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def rearrange_rb(self, R_b, calib_config):
    """rearrange the kinematic regressor by sample numbered order"""
    Rb_rearr = np.empty_like(R_b)
    for i in range(calib_config["calibration_index"]):
        for j in range(calib_config["NbSample"]):
            Rb_rearr[j * calib_config["calibration_index"] + i, :] = R_b[
                i * calib_config["NbSample"] + j
            ]
    return Rb_rearr
```

### `sub_info_matrix(R, calib_config)`

Decompose regressor into individual configuration info matrices.

Creates separate information matrices for each configuration by extracting the corresponding rows from the full regressor matrix. This decomposition enables individual configuration evaluation in the optimization process.

Parameters:

| Name           | Type      | Description                                                         | Default    |
| -------------- | --------- | ------------------------------------------------------------------- | ---------- |
| `R`            | `ndarray` | Full rearranged kinematic regressor matrix                          | *required* |
| `calib_config` | `dict`    | Calibration parameters including sample count and calibration index | *required* |

Returns:

| Name    | Type | Description                                                                                                                            |
| ------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `tuple` |      | (subX_list, subX_dict) where: - subX_list: List of information matrices (RᵢᵀRᵢ) - subX_dict: Dictionary mapping config index to matrix |

Mathematical Details

For configuration i: Rᵢ = R[i\*idx:(i+1)\*idx, :] (extract rows) Xᵢ = RᵢᵀRᵢ (information matrix)

Example

> > > R_full = np.random.rand(6000, 42) # 1000 configs, 6 DOF subX_list, subX_dict = self.sub_info_matrix(R_full, calib_config) print(len(subX_list)) # 1000 print(subX_dict[0].shape) # (42, 42)

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `calculate_optimal_configurations(selection_method='threshold', iroc_rel_tol=0.001, iroc_patience=3)`

Solve SOCP optimization to find optimal configuration subset.

This is the core optimization method that solves the D-optimal experimental design problem using Second-Order Cone Programming. The method finds weights for each candidate configuration that maximize the determinant of the Fisher Information Matrix.

Optimization Problem

maximize det(Σᵢ wᵢ Xᵢ)^(1/n) subject to: Σᵢ wᵢ ≤ 1, wᵢ ≥ 0

Where Xᵢ are information matrices and wᵢ are configuration weights.

Selection Process

1. Solve SOCP optimization for optimal (continuous-relaxation) weights
1. Select a discrete subset from the weight ranking, via one of two interchangeable strategies (`selection_method`):
1. `"threshold"` (default, original behavior): keep every configuration whose weight exceeds `eps_opt` (1e-5). Simple, but says nothing about *how many* configurations are actually enough -- it only discards the ones the SOCP relaxation assigned essentially zero weight.
1. `"iroc"`: rank-and-incrementally-test against the normalized D-optimality criterion (the same quantity :meth:`plot` already computes for visualization, here used as a stopping rule instead): grow the selected subset one configuration at a time in weight-rank order and stop at the first point where the criterion's relative gain stays below `iroc_rel_tol` for `iroc_patience` consecutive configurations in a row (a plateau) -- i.e. an automatic, data-driven minimal configuration count, rather than a fixed threshold on individual weights. See :meth:`_select_by_iroc`.
1. Verify minimum configuration requirement is met
1. Store selected configurations and weights

Parameters:

| Name               | Type    | Description                                                                                                      | Default       |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------------------- | ------------- |
| `selection_method` | `str`   | "threshold" or "iroc" (see above).                                                                               | `'threshold'` |
| `iroc_rel_tol`     | `float` | Relative-gain plateau tolerance, only used when selection_method="iroc".                                         | `0.001`       |
| `iroc_patience`    | `int`   | Number of consecutive sub-tolerance gains required to declare a plateau, only used when selection_method="iroc". | `3`           |

Side Effects

- Sets self.w_list with optimization solution weights
- Sets self.w_dict_sort with sorted weight dictionary
- Sets self.o1_history with the incremental D-optimality criterion trace when selection_method="iroc" (for inspection/plotting)
- Sets self.optimal_configurations with selected configs
- Sets self.optimal_weights with final weight values
- Sets self.nb_chosen with number of selected configurations
- Prints timing information and selection results

Returns:

| Name   | Type | Description                                  |
| ------ | ---- | -------------------------------------------- |
| `bool` |      | True if optimization successful and feasible |

Raises:

| Type             | Description                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `AssertionError` | If regressor not calculated or if insufficient configurations selected (infeasible design) |
| `ValueError`     | If selection_method is not "threshold" or "iroc"                                           |

Example

> > > opt_calib.calculate_optimal_configurations() solve time of socp: 2.35 seconds 12 configs are chosen: [0, 5, 12, 18, 23, ...] opt_calib.calculate_optimal_configurations( ... selection_method="iroc" ... ) 9 configs are chosen: [0, 5, 12, 18, 23, 7, 14, 2, 9]

See Also

SOCPOptimizer: The optimization solver implementation calculate_regressor: Required prerequisite computation \_select_by_iroc: The O1-plateau automatic count selection

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def calculate_optimal_configurations(
    self,
    selection_method="threshold",
    iroc_rel_tol=1e-3,
    iroc_patience=3,
):
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
        1. Solve SOCP optimization for optimal (continuous-relaxation)
           weights
        2. Select a discrete subset from the weight ranking, via one of
           two interchangeable strategies (``selection_method``):

           - ``"threshold"`` (default, original behavior): keep every
             configuration whose weight exceeds ``eps_opt`` (1e-5).
             Simple, but says nothing about *how many* configurations
             are actually enough -- it only discards the ones the SOCP
             relaxation assigned essentially zero weight.
           - ``"iroc"``: rank-and-incrementally-test against the
             normalized D-optimality criterion (the same quantity
             :meth:`plot` already computes for visualization, here used
             as a stopping rule instead): grow the selected subset one
             configuration at a time in weight-rank order and stop at
             the first point where the criterion's relative gain stays
             below ``iroc_rel_tol`` for ``iroc_patience`` consecutive
             configurations in a row (a plateau) -- i.e. an automatic,
             data-driven minimal configuration count, rather than a
             fixed threshold on individual weights. See
             :meth:`_select_by_iroc`.
        3. Verify minimum configuration requirement is met
        4. Store selected configurations and weights

    Args:
        selection_method (str): "threshold" or "iroc" (see above).
        iroc_rel_tol (float): Relative-gain plateau tolerance, only
            used when ``selection_method="iroc"``.
        iroc_patience (int): Number of consecutive sub-tolerance gains
            required to declare a plateau, only used when
            ``selection_method="iroc"``.

    Side Effects:
        - Sets self.w_list with optimization solution weights
        - Sets self.w_dict_sort with sorted weight dictionary
        - Sets self.o1_history with the incremental D-optimality
          criterion trace when selection_method="iroc" (for
          inspection/plotting)
        - Sets self.optimal_configurations with selected configs
        - Sets self.optimal_weights with final weight values
        - Sets self.nb_chosen with number of selected configurations
        - Prints timing information and selection results

    Returns:
        bool: True if optimization successful and feasible

    Raises:
        AssertionError: If regressor not calculated or if insufficient
                      configurations selected (infeasible design)
        ValueError: If selection_method is not "threshold" or "iroc"

    Example:
        >>> opt_calib.calculate_optimal_configurations()
        solve time of socp: 2.35 seconds
        12 configs are chosen: [0, 5, 12, 18, 23, ...]
        >>> opt_calib.calculate_optimal_configurations(
        ...     selection_method="iroc"
        ... )
        9 configs are chosen: [0, 5, 12, 18, 23, 7, 14, 2, 9]

    See Also:
        SOCPOptimizer: The optimization solver implementation
        calculate_regressor: Required prerequisite computation
        _select_by_iroc: The O1-plateau automatic count selection
    """
    import time

    assert self.calculate_regressor(), "Calculate regressor first."

    # Picos optimization (A-optimality, C-optimality, D-optimality)
    prev_time = time.time()
    SOCP_algo = SOCPOptimizer(self._subX_dict, self.calib_config)
    self.w_list, self.w_dict_sort = SOCP_algo.solve()
    solve_time = time.time() - prev_time
    logger.info(f"solve time of socp: {solve_time}")

    if selection_method == "threshold":
        chosen_config = self._select_by_threshold()
    elif selection_method == "iroc":
        chosen_config = self._select_by_iroc(iroc_rel_tol, iroc_patience)
    else:
        raise ValueError(
            "selection_method must be 'threshold' or 'iroc', got "
            f"{selection_method!r}."
        )

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
```

### `plot()`

Generate comprehensive visualization of optimization results.

Creates dual-panel plots that provide insight into the optimization quality and configuration selection process. The visualizations help assess the efficiency of the selected configuration subset.

Plot Components:

1. D-optimality criterion vs. number of configurations

- Shows how information matrix determinant improves with additional configurations
- Normalized against theoretical maximum (all configurations)
- Helps identify diminishing returns point

1. Configuration weights in logarithmic scale
1. Displays weight assigned to each candidate configuration
1. Configurations above threshold (eps_opt) are selected
1. Shows selection boundary and weight distribution

Prerequisites

- Optimization must be completed (optimal_configurations available)
- Information matrices must be computed

Side Effects

- Creates matplotlib figure with two subplots
- Displays plots using plt.show()
- May block execution until plots are closed

Returns:

| Name   | Type | Description                 |
| ------ | ---- | --------------------------- |
| `bool` |      | True if plotting successful |

Mathematical Details

D-optimality ratio = detroot_whole / det(selected_subset) This ratio approaches 1.0 as selected subset approaches optimality.

Example

> > > opt_calib.solve()
> > >
> > > #### Plot is automatically generated, or call manually:
> > >
> > > opt_calib.plot()

See Also

calculate_optimal_configurations: Generates data for plotting calculate_detroot_whole: Provides normalization reference

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `plot_results()`

Plot optimal calibration results using unified results manager.

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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

    def _basic_plots():
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

    def _managed_plot():
        robot_name = self.calib_config.get("robot_name", self.model.name)
        results_manager = ResultsManager("optimal_calibration", robot_name)
        weights = (
            np.array(list(self.w_dict_sort.values()))
            if hasattr(self, "w_dict_sort")
            else np.array([])
        )
        results_manager.plot_optimal_calibration_results(
            configurations=self.optimal_configurations,
            weights=weights,
            title="Optimal Calibration Configuration Results",
        )

    plot_with_fallback(_managed_plot, _basic_plots, logger, "optimal_calibration")
```

### `get_optimal_output_dir(default='results')`

Return the directory optimal-configuration results are written to.

Honours the directory component of the configured `tasks.optimal_configuration.output.output_file`. The *filename* is managed by :class:`ResultsManager` (timestamped, one file per format), so only the directory part of that setting is used.

Parameters:

| Name      | Type  | Description                                  | Default     |
| --------- | ----- | -------------------------------------------- | ----------- |
| `default` | `str` | Directory to use when nothing is configured. | `'results'` |

Returns:

| Name  | Type | Description            |
| ----- | ---- | ---------------------- |
| `str` |      | Output directory path. |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def get_optimal_output_dir(self, default="results"):
    """Return the directory optimal-configuration results are written to.

    Honours the directory component of the configured
    ``tasks.optimal_configuration.output.output_file``. The *filename* is
    managed by :class:`ResultsManager` (timestamped, one file per format),
    so only the directory part of that setting is used.

    Args:
        default (str): Directory to use when nothing is configured.

    Returns:
        str: Output directory path.
    """
    configured = getattr(self, "_optimal_output_file", None)
    if not configured:
        return default
    return os.path.dirname(configured) or default
```

### `count_optimal_configurations()`

Return the number of selected configurations.

`optimal_configurations` is a *dict* holding the joint names alongside the selected joint configurations, so `len()` on it counts keys (2), not configurations. Always go through this helper.

Returns:

| Name  | Type | Description                                         |
| ----- | ---- | --------------------------------------------------- |
| `int` |      | Number of selected joint configurations, 0 if none. |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def count_optimal_configurations(self):
    """Return the number of selected configurations.

    ``optimal_configurations`` is a *dict* holding the joint names alongside
    the selected joint configurations, so ``len()`` on it counts keys (2),
    not configurations. Always go through this helper.

    Returns:
        int: Number of selected joint configurations, 0 if none.
    """
    configs = getattr(self, "optimal_configurations", None)
    if not configs:
        return 0
    return len(configs.get("calibration_joint_configurations", []))
```

### `save_results(output_dir=None)`

Save optimal configuration results using unified results manager.

Parameters:

| Name         | Type  | Description                                                                                                                                                                            | Default |
| ------------ | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `output_dir` | `str` | Directory to write results into. When omitted, the directory configured under the unified config's tasks.optimal_configuration.output.output_file is used, and failing that "results". | `None`  |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def save_results(self, output_dir=None):
    """Save optimal configuration results using unified results manager.

    Args:
        output_dir (str, optional): Directory to write results into. When
            omitted, the directory configured under the unified config's
            ``tasks.optimal_configuration.output.output_file`` is used, and
            failing that ``"results"``.
    """
    if (
        not hasattr(self, "optimal_configurations")
        or self.optimal_configurations is None
    ):
        logger.warning(
            "No optimal configuration results to save. Run solve() first."
        )
        return

    if output_dir is None:
        output_dir = self.get_optimal_output_dir()

    try:
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
            "configuration_count": self.count_optimal_configurations(),
            "calibration_config": self.calib_config,
        }

        # D-optimality determinant root of the full information matrix.
        # Named after what it is: this is NOT a condition number.
        if hasattr(self, "detroot_whole"):
            results_dict["d_optimality_detroot"] = float(self.detroot_whole)

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
```

## `BaseOptimalTrajectory(robot, active_joints, config_file='config/robot_config.yaml')`

Base class for IPOPT-based optimal trajectory generation.

Features:

- Modular design with separated concerns
- Better error handling and logging
- Configuration validation
- Cleaner interfaces

This base class can be extended for specific robots by implementing robot-specific configuration loading and constraint handling.

Initialize the optimal trajectory generator.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `initialize()`

Initialize trajectory generation components.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `solve(stack_reps=2)`

Solve the optimal trajectory generation problem.

Parameters:

| Name         | Type  | Description                            | Default |
| ------------ | ----- | -------------------------------------- | ------- |
| `stack_reps` | `int` | Number of trajectory segments to stack | `2`     |

Returns:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `Dict[str, Any]` | Dict containing trajectories and optimization info |

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `objective_function(X, opt_cb, tps, vel_wps, acc_wps, wp_init, W_stack=None)`

Objective function: condition number of base regressor matrix.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `create_ipopt_problem(n_joints, n_wps, Ns, tps, vel_wps, acc_wps, wp_init, vel_wp_init, acc_wp_init, W_stack)`

Create IPOPT problem instance. Should be implemented by subclasses.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `plot_results()`

Plot optimal trajectory results using unified results manager.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def plot_results(self):
    """Plot optimal trajectory results using unified results manager."""
    if not self.results["T_F"]:
        self.logger.warning("No trajectory data to plot")
        return

    def _basic_plots():
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

                # P spans nq columns, V and A span nv: index each by
                # the active joint's own idx_q / idx_v (#149).
                for joint_idx, (iq, iv) in enumerate(
                    zip(self.CB.act_idxq, self.CB.act_idxv)
                ):
                    axes[joint_idx, 0].plot(T, P[:, iq], color=color, label=label)
                    axes[joint_idx, 1].plot(T, V[:, iv], color=color, label=label)
                    axes[joint_idx, 2].plot(T, A[:, iv], color=color, label=label)

            # Set labels and formatting
            for joint_idx, name in enumerate(self.CB.act_Jname):
                axes[joint_idx, 0].set_ylabel(f"{name}\nPosition (rad)")
                axes[joint_idx, 1].set_ylabel(f"{name}\nVelocity (rad/s)")
                axes[joint_idx, 2].set_ylabel(f"{name}\nAcceleration (rad/s²)")

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

    def _managed_plot():
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

        results_manager.plot_optimal_trajectory_results(
            trajectories=self.results,
            condition_number=condition_number,
            joint_names=self.CB.act_Jname,
            title="Optimal Trajectory Generation Results",
            q_indices=self.CB.act_idxq,
            v_indices=self.CB.act_idxv,
        )

    plot_with_fallback(
        _managed_plot, _basic_plots, self.logger, "optimal_trajectory"
    )
```

### `save_results(output_dir='results')`

Save optimal trajectory results using unified results manager.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def save_results(self, output_dir="results"):
    """Save optimal trajectory results using unified results manager."""
    if not self.results["T_F"]:
        self.logger.warning("No trajectory data to save")
        return

    try:
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

        self.logger.info("Trajectory results saved successfully")
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
```

## `BaseTrajectoryIPOPTProblem(opt_traj, n_joints, n_wps, Ns, tps, vel_wps, acc_wps, wp_init, vel_wp_init, acc_wp_init, W_stack, problem_name='TrajectoryOptimization')`

Bases: `BaseOptimizationProblem`

Base IPOPT problem formulation for trajectory optimization.

This class provides a base implementation for trajectory optimization that can be extended for specific robots.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `get_variable_bounds()`

Get variable bounds for optimization.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
    """Get variable bounds for optimization."""
    return self.opt_traj.constraint_manager.get_variable_bounds()
```

### `get_constraint_bounds()`

Get constraint bounds for optimization.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
    """Get constraint bounds for optimization."""
    return self.opt_traj.constraint_manager.get_constraint_bounds(self.Ns)
```

### `get_initial_guess()`

Get initial guess from waypoints.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def get_initial_guess(self) -> List[float]:
    """Get initial guess from waypoints."""
    # This will be set when solve() is called with waypoints
    if not hasattr(self, "_initial_wps"):
        # Return zeros as fallback
        return [0.0] * (self.n_joints * (self.n_wps - 1))

    X0 = self._initial_wps[:, range(1, self.n_wps)]
    return np.reshape(X0.transpose(), (self.n_joints * (self.n_wps - 1),)).tolist()
```

### `objective(X)`

Objective function: condition number of base regressor matrix.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `constraints(X)`

Constraint function for IPOPT.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def constraints(self, X: np.ndarray) -> np.ndarray:
    """Constraint function for IPOPT."""
    return self.opt_traj.constraint_manager.evaluate_constraints(
        self.Ns, X, self.opt_cb, self.tps, self.vel_wps, self.acc_wps, self.wp_init
    )
```

### `gradient(X)`

Gradient of the objective by forward finite differences.

The default numdifftools gradient uses Richardson extrapolation over many step sizes, which costs tens of seconds per call here because every objective evaluation rebuilds the full regressor.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def gradient(self, X: np.ndarray) -> np.ndarray:
    """
    Gradient of the objective by forward finite differences.

    The default numdifftools gradient uses Richardson extrapolation over
    many step sizes, which costs tens of seconds per call here because
    every objective evaluation rebuilds the full regressor.
    """
    X = np.asarray(X, dtype=float)
    f0 = self.objective(X)
    eps = 1e-6
    grad = np.zeros_like(X)
    for i in range(len(X)):
        X_plus = X.copy()
        X_plus[i] += eps
        grad[i] = (self.objective(X_plus) - f0) / eps
    return grad
```

### `jacobian(X)`

Jacobian of constraints - Custom implementation for better performance.

For trajectory optimization, we can use sparse finite differences instead of full automatic differentiation which is too slow.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `solve_with_waypoints(wps)`

Solve the optimization problem with given initial waypoints.

Parameters:

| Name  | Type | Description       | Default    |
| ----- | ---- | ----------------- | ---------- |
| `wps` |      | Initial waypoints | *required* |

Returns:

| Type                          | Description                      |
| ----------------------------- | -------------------------------- |
| `Tuple[bool, Dict[str, Any]]` | Tuple of (success, results_dict) |

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
        config.max_iterations = self.opt_traj.trajectory_config.get(
            "max_iterations", 200
        )
        config.print_level = 3  # Reduce output
        config.custom_options = {
            b"mu_strategy": b"adaptive",
        }
        solver = RobotIPOPTSolver(self, config)

        # Solve the problem
        success, results = solver.solve()
        converged = success

        # The condition-number objective is non-smooth, so IPOPT often
        # stalls on dual infeasibility after reaching a feasible plateau.
        # Keep such an iterate, but only if it satisfies every constraint.
        if not success and results.get("status") == -1:
            success = self._is_feasible(results["x_opt"])
            if success:
                self.logger.warning(
                    "IPOPT reached the iteration limit (%d) without "
                    "converging; keeping the feasible final iterate "
                    "(objective %.4g)",
                    config.max_iterations,
                    results["obj_val"],
                )

        # Collision is constrained at check points only; accept the
        # segment only if the whole spline is clear (#143).
        if success and not self._is_collision_free(results["x_opt"]):
            self.logger.error(
                "Optimised segment collides between collision check points"
            )
            return False, results

        if success:
            # Extract final waypoint for next segment
            X_opt = results["x_opt"]
            wps_X = np.reshape(np.array(X_opt), (self.n_wps - 1, self.n_joints))
            final_waypoint = wps_X[-1, :]
            # The last callback may have been a finite-difference probe;
            # rebuild the stored trajectory at the returned solution.
            self.objective(np.asarray(X_opt))

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
                        "converged": converged,
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
```

## base_optimal_calibration

Base class for robot optimal configuration generation for calibration. This module provides a generalized framework for optimal configuration generation that can be inherited by any robot type (TIAGo, UR10, MATE, etc.).

## `BaseOptimalCalibration(robot, config_file='config/robot_config.yaml')`

Bases: `ABC`

Base class for robot optimal configuration generation for calibration.

This class implements the framework for generating optimal robot configurations that maximize the observability of kinematic parameters during calibration. It uses Second-Order Cone Programming (SOCP) to solve the D-optimal design problem for parameter estimation.

The class provides a Template Method pattern where the main workflow is defined, but specific optimization strategies can be customized by derived classes for different robot types.

Workflow

1. Load candidate configurations from file (CSV or YAML)
1. Calculate kinematic regressors for all candidates
1. Compute information matrices for each configuration
1. Solve SOCP optimization to find optimal subset
1. Select configurations with significant weights
1. Visualize and save results

Key Features

- D-optimal experimental design for calibration
- Support for multiple calibration models (full_params, joint_offset)
- Automatic minimum configuration calculation
- SOCP-based optimization with convex relaxation
- Comprehensive visualization and analysis tools
- File I/O for configuration management

Mathematical Background

The method maximizes the determinant of the Fisher Information Matrix: max det(Σᵢ wᵢ Rᵢᵀ Rᵢ) subject to Σᵢ wᵢ ≤ 1, wᵢ ≥ 0

Where:

- Rᵢ is the kinematic regressor for configuration i
- wᵢ is the weight assigned to configuration i
- The objective maximizes parameter estimation precision

Attributes:

| Name                     | Type      | Description                                    |
| ------------------------ | --------- | ---------------------------------------------- |
| `robot`                  |           | Robot model instance loaded with FIGAROH       |
| `model`                  |           | Pinocchio robot model                          |
| `data`                   |           | Pinocchio robot data                           |
| `calib_config`           | `dict`    | Calibration parameters from configuration file |
| `optimal_configurations` | `dict`    | Selected optimal configurations                |
| `optimal_weights`        | `ndarray` | Weights assigned to configurations             |
| `minNbChosen`            | `int`     | Minimum number of configurations required      |
| `R_rearr`                | `ndarray` | Rearranged kinematic regressor matrix          |
| `detroot_whole`          | `float`   | Determinant root of full information matrix    |
| `w_list`                 | `list`    | Solution weights from SOCP optimization        |
| `w_dict_sort`            | `dict`    | Sorted weights by configuration index          |

Example

> > > ### Basic usage for TIAGo robot
> > >
> > > from figaroh.robots import TiagoRobot robot = TiagoRobot()
> > >
> > > ### Create optimal calibration instance
> > >
> > > opt_calib = TiagoOptimalCalibration(robot, "config/tiago.yaml")
> > >
> > > ### Generate optimal configurations
> > >
> > > opt_calib.solve(save_file=True)
> > >
> > > ### Access results
> > >
> > > print(f"Selected {len(opt_calib.optimal_configurations)} configs") print(f"Minimum required: {opt_calib.minNbChosen}")

See Also

BaseCalibration: Main calibration framework SOCPOptimizer: Second-order cone programming solver TiagoOptimalCalibration: TIAGo-specific implementation UR10OptimalCalibration: UR10-specific implementation

Initialize optimal calibration with robot model and configuration.

Sets up the optimal calibration framework by loading robot parameters, initializing optimization attributes, and calculating the minimum number of configurations required based on the calibration model.

The minimum number of configurations is computed to ensure the optimization problem is well-posed and identifiable:

- For full_params: considers all kinematic parameters
- For joint_offset: considers only joint offset parameters

Parameters:

| Name          | Type  | Description                                                                                                                                 | Default                      |
| ------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `robot`       |       | Robot model instance loaded with FIGAROH. Must have 'model' and 'data' attributes for Pinocchio integration.                                | *required*                   |
| `config_file` | `str` | Path to YAML configuration file containing calibration parameters, sample file paths, and optimization settings. Defaults to standard path. | `'config/robot_config.yaml'` |

Raises:

| Type                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `FileNotFoundError` | If config_file does not exist                     |
| `KeyError`          | If required parameters missing from configuration |
| `ValueError`        | If calibration model type is unsupported          |

Side Effects

- Loads and stores calibration parameters in self.calib_config
- Calculates minimum required configurations (self.minNbChosen)
- Initializes optimization result attributes to None
- Prints initialization confirmation message

Example

> > > robot = TiagoRobot() opt_calib = BaseOptimalCalibration(robot, "config/tiago.yaml") TiagoOptimalCalibration initialized

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
    self._config_file = config_file
    self.load_param(config_file)

    # Initialize attributes for optimal calibration
    self.optimal_configurations = None
    self.optimal_weights = None
    self._sampleConfigs_file = self.calib_config.get("sample_configs_file")
    # load_param() only reads the "calibration" task, so the optimal task's
    # own output settings are picked up separately here.
    self._optimal_output_file = self._load_optimal_output_file(config_file)

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
```

### `initialize()`

Initialize the optimization process by preparing all required data.

This method orchestrates the initialization sequence required before optimization can begin. It ensures all mathematical components are properly computed and cached for efficient optimization.

The initialization sequence:

1. Load candidate configurations from external files
1. Calculate kinematic regressors for all configurations
1. Compute determinant root of the full information matrix

Prerequisites

- Robot model and parameters must be loaded
- Configuration file must specify valid sample data paths

Side Effects

- Sets self.q_measured with candidate joint configurations
- Sets self.R_rearr with rearranged kinematic regressor
- Sets self.\_subX_dict and self.\_subX_list with info matrices
- Sets self.detroot_whole with full matrix determinant root

Raises:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `ValueError`     | If sample configuration file is invalid or missing |
| `AssertionError` | If regressor calculation fails                     |

See Also

load_candidate_configurations: Configuration data loading calculate_regressor: Kinematic regressor computation calculate_detroot_whole: Information matrix analysis

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `solve(save_file=False, selection_method='threshold', **selection_kwargs)`

Solve the optimal configuration selection problem.

This is the main entry point that orchestrates the complete optimal configuration generation workflow. It automatically handles initialization if not already performed, solves the SOCP optimization, and provides comprehensive results analysis.

The method implements the complete D-optimal design workflow:

1. Initialize data and regressors (if needed)
1. Solve SOCP optimization for optimal weights
1. Select configurations with significant weights
1. Optionally save results to files
1. Generate visualization plots

Parameters:

| Name                 | Type   | Description                                                                                                    | Default       |
| -------------------- | ------ | -------------------------------------------------------------------------------------------------------------- | ------------- |
| `save_file`          | `bool` | Whether to save optimal configurations to YAML file in results directory. Default False.                       | `False`       |
| `selection_method`   | `str`  | "threshold" (default) or "iroc" -- see calculate_optimal_configurations for what each one does.                | `'threshold'` |
| `**selection_kwargs` |        | Forwarded to calculate_optimal_configurations (e.g. iroc_rel_tol, iroc_patience when selection_method="iroc"). | `{}`          |

Side Effects

- Updates self.optimal_configurations with selected configs
- Updates self.optimal_weights with optimization weights
- Creates visualization plots
- May create output files if save_file=True
- Prints progress and results to console

Raises:

| Type             | Description                                  |
| ---------------- | -------------------------------------------- |
| `AssertionError` | If minimum configuration requirement not met |
| `ValueError`     | If optimization problem is infeasible        |
| `IOError`        | If file saving fails (logged as warning)     |

Example

> > > opt_calib = TiagoOptimalCalibration(robot) opt_calib.solve(save_file=True) 12 configs are chosen: [0, 5, 12, 18, ...] Optimal configurations written to file successfully

See Also

initialize: Data preparation workflow calculate_optimal_configurations: Core optimization solver plot: Results visualization save_results: File output management

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def solve(self, save_file=False, selection_method="threshold", **selection_kwargs):
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
        selection_method (str): "threshold" (default) or "iroc" -- see
                        calculate_optimal_configurations for what each
                        one does.
        **selection_kwargs: Forwarded to calculate_optimal_configurations
                        (e.g. iroc_rel_tol, iroc_patience when
                        selection_method="iroc").

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
    self.calculate_optimal_configurations(
        selection_method=selection_method, **selection_kwargs
    )
    if save_file:
        try:
            self.save_results()
            logger.info("Optimal configurations written to file successfully")
        except Exception as e:
            logger.warning(f"Could not write to file: {e}")
    self.plot()
```

### `load_param(config_file, setting_type='calibration')`

Load calibration parameters from YAML configuration file.

Only the unified configuration format is supported; a legacy flat config raises an error naming `figaroh.utils.config_migration`.

Parameters:

| Name           | Type  | Description                          | Default         |
| -------------- | ----- | ------------------------------------ | --------------- |
| `config_file`  | `str` | Path to a unified configuration file | *required*      |
| `setting_type` | `str` | Configuration section to load        | `'calibration'` |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def load_param(self, config_file: str, setting_type: str = "calibration"):
    """Load calibration parameters from YAML configuration file.

    Only the unified configuration format is supported; a legacy flat
    config raises an error naming ``figaroh.utils.config_migration``.

    Args:
        config_file (str): Path to a unified configuration file
        setting_type (str): Configuration section to load
    """
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

### `load_candidate_configurations()`

Load candidate joint configurations from external data files.

Reads robot joint configurations from CSV or YAML files that serve as the candidate pool for optimization. The method supports multiple file formats and automatically updates the sample count parameter.

Supported formats:

- CSV: Standard measurement data format with joint configurations
- YAML: Structured format with named joints and configurations

The YAML format expects:

```
calibration_joint_names: [joint1, joint2, ...]
calibration_joint_configurations: [[q1_1, q1_2, ...], [q2_1, ...]]
```

Side Effects

- Sets self.q_measured with loaded joint configurations
- Updates self.calib_config["NbSample"] with actual sample count
- May load self.\_configs for YAML format data

Raises:

| Type                | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `ValueError`        | If sample_configs_file not specified in configuration or if file format is not supported |
| `FileNotFoundError` | If specified data file does not exist                                                    |

Example

> > > #### Assuming config specifies "data/candidate_configs.yaml"
> > >
> > > opt_calib.load_candidate_configurations() print(opt_calib.q_measured.shape) # (1000, 7) for TIAGo

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

````
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
````

### `calculate_regressor()`

Calculate kinematic regressors and information matrices.

Computes the kinematic regressor matrices that relate kinematic parameter variations to end-effector pose changes. This is the mathematical foundation for the optimization problem.

The method performs several key computations:

1. Calculate base kinematic regressors for all configurations
1. Rearrange regressor matrix by sample order for efficiency
1. Compute individual information matrices for each configuration
1. Store results for optimization access

Mathematical Background

For each configuration i, the regressor Rᵢ satisfies: δx = Rᵢ δθ where δx is pose variation and δθ is parameter variation.

The information matrix is: Xᵢ = RᵢᵀRᵢ

Side Effects

- Sets self.R_rearr with rearranged kinematic regressor
- Sets self.\_subX_list with list of information matrices
- Sets self.\_subX_dict with indexed information matrices
- Prints parameter names for verification

Returns:

| Name   | Type | Description                    |
| ------ | ---- | ------------------------------ |
| `bool` |      | True if calculation successful |

Prerequisites

- Joint configurations must be loaded (self.q_measured)
- Robot model and parameters must be initialized

See Also

calculate_base_kinematics_regressor: Core regressor computation rearrange_rb: Matrix rearrangement for optimization sub_info_matrix: Information matrix decomposition

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `calculate_detroot_whole()`

Calculate determinant root of complete information matrix.

Computes the determinant root of the full Fisher Information Matrix formed by all candidate configurations. This serves as the theoretical upper bound for the D-optimality criterion and is used for performance comparison.

Mathematical Background

M_full = R^T R (full regressor) detroot_whole = det(M_full)^(1/n) / sqrt(n)

This represents the geometric mean of eigenvalues, normalized by matrix dimension for scale independence.

Side Effects

- Sets self.detroot_whole with computed determinant root
- Prints the computed value for verification

Prerequisites

- Kinematic regressor must be calculated (self.R_rearr)
- Requires picos library for determinant computation

Raises:

| Type             | Description                                  |
| ---------------- | -------------------------------------------- |
| `AssertionError` | If regressor calculation not performed first |
| `ImportError`    | If picos library not available               |

See Also

calculate_regressor: Prerequisites for this computation plot: Uses this value for performance comparison

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `rearrange_rb(R_b, calib_config)`

rearrange the kinematic regressor by sample numbered order

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def rearrange_rb(self, R_b, calib_config):
    """rearrange the kinematic regressor by sample numbered order"""
    Rb_rearr = np.empty_like(R_b)
    for i in range(calib_config["calibration_index"]):
        for j in range(calib_config["NbSample"]):
            Rb_rearr[j * calib_config["calibration_index"] + i, :] = R_b[
                i * calib_config["NbSample"] + j
            ]
    return Rb_rearr
```

### `sub_info_matrix(R, calib_config)`

Decompose regressor into individual configuration info matrices.

Creates separate information matrices for each configuration by extracting the corresponding rows from the full regressor matrix. This decomposition enables individual configuration evaluation in the optimization process.

Parameters:

| Name           | Type      | Description                                                         | Default    |
| -------------- | --------- | ------------------------------------------------------------------- | ---------- |
| `R`            | `ndarray` | Full rearranged kinematic regressor matrix                          | *required* |
| `calib_config` | `dict`    | Calibration parameters including sample count and calibration index | *required* |

Returns:

| Name    | Type | Description                                                                                                                            |
| ------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `tuple` |      | (subX_list, subX_dict) where: - subX_list: List of information matrices (RᵢᵀRᵢ) - subX_dict: Dictionary mapping config index to matrix |

Mathematical Details

For configuration i: Rᵢ = R[i\*idx:(i+1)\*idx, :] (extract rows) Xᵢ = RᵢᵀRᵢ (information matrix)

Example

> > > R_full = np.random.rand(6000, 42) # 1000 configs, 6 DOF subX_list, subX_dict = self.sub_info_matrix(R_full, calib_config) print(len(subX_list)) # 1000 print(subX_dict[0].shape) # (42, 42)

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `calculate_optimal_configurations(selection_method='threshold', iroc_rel_tol=0.001, iroc_patience=3)`

Solve SOCP optimization to find optimal configuration subset.

This is the core optimization method that solves the D-optimal experimental design problem using Second-Order Cone Programming. The method finds weights for each candidate configuration that maximize the determinant of the Fisher Information Matrix.

Optimization Problem

maximize det(Σᵢ wᵢ Xᵢ)^(1/n) subject to: Σᵢ wᵢ ≤ 1, wᵢ ≥ 0

Where Xᵢ are information matrices and wᵢ are configuration weights.

Selection Process

1. Solve SOCP optimization for optimal (continuous-relaxation) weights
1. Select a discrete subset from the weight ranking, via one of two interchangeable strategies (`selection_method`):
1. `"threshold"` (default, original behavior): keep every configuration whose weight exceeds `eps_opt` (1e-5). Simple, but says nothing about *how many* configurations are actually enough -- it only discards the ones the SOCP relaxation assigned essentially zero weight.
1. `"iroc"`: rank-and-incrementally-test against the normalized D-optimality criterion (the same quantity :meth:`plot` already computes for visualization, here used as a stopping rule instead): grow the selected subset one configuration at a time in weight-rank order and stop at the first point where the criterion's relative gain stays below `iroc_rel_tol` for `iroc_patience` consecutive configurations in a row (a plateau) -- i.e. an automatic, data-driven minimal configuration count, rather than a fixed threshold on individual weights. See :meth:`_select_by_iroc`.
1. Verify minimum configuration requirement is met
1. Store selected configurations and weights

Parameters:

| Name               | Type    | Description                                                                                                      | Default       |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------------------- | ------------- |
| `selection_method` | `str`   | "threshold" or "iroc" (see above).                                                                               | `'threshold'` |
| `iroc_rel_tol`     | `float` | Relative-gain plateau tolerance, only used when selection_method="iroc".                                         | `0.001`       |
| `iroc_patience`    | `int`   | Number of consecutive sub-tolerance gains required to declare a plateau, only used when selection_method="iroc". | `3`           |

Side Effects

- Sets self.w_list with optimization solution weights
- Sets self.w_dict_sort with sorted weight dictionary
- Sets self.o1_history with the incremental D-optimality criterion trace when selection_method="iroc" (for inspection/plotting)
- Sets self.optimal_configurations with selected configs
- Sets self.optimal_weights with final weight values
- Sets self.nb_chosen with number of selected configurations
- Prints timing information and selection results

Returns:

| Name   | Type | Description                                  |
| ------ | ---- | -------------------------------------------- |
| `bool` |      | True if optimization successful and feasible |

Raises:

| Type             | Description                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `AssertionError` | If regressor not calculated or if insufficient configurations selected (infeasible design) |
| `ValueError`     | If selection_method is not "threshold" or "iroc"                                           |

Example

> > > opt_calib.calculate_optimal_configurations() solve time of socp: 2.35 seconds 12 configs are chosen: [0, 5, 12, 18, 23, ...] opt_calib.calculate_optimal_configurations( ... selection_method="iroc" ... ) 9 configs are chosen: [0, 5, 12, 18, 23, 7, 14, 2, 9]

See Also

SOCPOptimizer: The optimization solver implementation calculate_regressor: Required prerequisite computation \_select_by_iroc: The O1-plateau automatic count selection

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def calculate_optimal_configurations(
    self,
    selection_method="threshold",
    iroc_rel_tol=1e-3,
    iroc_patience=3,
):
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
        1. Solve SOCP optimization for optimal (continuous-relaxation)
           weights
        2. Select a discrete subset from the weight ranking, via one of
           two interchangeable strategies (``selection_method``):

           - ``"threshold"`` (default, original behavior): keep every
             configuration whose weight exceeds ``eps_opt`` (1e-5).
             Simple, but says nothing about *how many* configurations
             are actually enough -- it only discards the ones the SOCP
             relaxation assigned essentially zero weight.
           - ``"iroc"``: rank-and-incrementally-test against the
             normalized D-optimality criterion (the same quantity
             :meth:`plot` already computes for visualization, here used
             as a stopping rule instead): grow the selected subset one
             configuration at a time in weight-rank order and stop at
             the first point where the criterion's relative gain stays
             below ``iroc_rel_tol`` for ``iroc_patience`` consecutive
             configurations in a row (a plateau) -- i.e. an automatic,
             data-driven minimal configuration count, rather than a
             fixed threshold on individual weights. See
             :meth:`_select_by_iroc`.
        3. Verify minimum configuration requirement is met
        4. Store selected configurations and weights

    Args:
        selection_method (str): "threshold" or "iroc" (see above).
        iroc_rel_tol (float): Relative-gain plateau tolerance, only
            used when ``selection_method="iroc"``.
        iroc_patience (int): Number of consecutive sub-tolerance gains
            required to declare a plateau, only used when
            ``selection_method="iroc"``.

    Side Effects:
        - Sets self.w_list with optimization solution weights
        - Sets self.w_dict_sort with sorted weight dictionary
        - Sets self.o1_history with the incremental D-optimality
          criterion trace when selection_method="iroc" (for
          inspection/plotting)
        - Sets self.optimal_configurations with selected configs
        - Sets self.optimal_weights with final weight values
        - Sets self.nb_chosen with number of selected configurations
        - Prints timing information and selection results

    Returns:
        bool: True if optimization successful and feasible

    Raises:
        AssertionError: If regressor not calculated or if insufficient
                      configurations selected (infeasible design)
        ValueError: If selection_method is not "threshold" or "iroc"

    Example:
        >>> opt_calib.calculate_optimal_configurations()
        solve time of socp: 2.35 seconds
        12 configs are chosen: [0, 5, 12, 18, 23, ...]
        >>> opt_calib.calculate_optimal_configurations(
        ...     selection_method="iroc"
        ... )
        9 configs are chosen: [0, 5, 12, 18, 23, 7, 14, 2, 9]

    See Also:
        SOCPOptimizer: The optimization solver implementation
        calculate_regressor: Required prerequisite computation
        _select_by_iroc: The O1-plateau automatic count selection
    """
    import time

    assert self.calculate_regressor(), "Calculate regressor first."

    # Picos optimization (A-optimality, C-optimality, D-optimality)
    prev_time = time.time()
    SOCP_algo = SOCPOptimizer(self._subX_dict, self.calib_config)
    self.w_list, self.w_dict_sort = SOCP_algo.solve()
    solve_time = time.time() - prev_time
    logger.info(f"solve time of socp: {solve_time}")

    if selection_method == "threshold":
        chosen_config = self._select_by_threshold()
    elif selection_method == "iroc":
        chosen_config = self._select_by_iroc(iroc_rel_tol, iroc_patience)
    else:
        raise ValueError(
            "selection_method must be 'threshold' or 'iroc', got "
            f"{selection_method!r}."
        )

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
```

### `plot()`

Generate comprehensive visualization of optimization results.

Creates dual-panel plots that provide insight into the optimization quality and configuration selection process. The visualizations help assess the efficiency of the selected configuration subset.

Plot Components:

1. D-optimality criterion vs. number of configurations

- Shows how information matrix determinant improves with additional configurations
- Normalized against theoretical maximum (all configurations)
- Helps identify diminishing returns point

1. Configuration weights in logarithmic scale
1. Displays weight assigned to each candidate configuration
1. Configurations above threshold (eps_opt) are selected
1. Shows selection boundary and weight distribution

Prerequisites

- Optimization must be completed (optimal_configurations available)
- Information matrices must be computed

Side Effects

- Creates matplotlib figure with two subplots
- Displays plots using plt.show()
- May block execution until plots are closed

Returns:

| Name   | Type | Description                 |
| ------ | ---- | --------------------------- |
| `bool` |      | True if plotting successful |

Mathematical Details

D-optimality ratio = detroot_whole / det(selected_subset) This ratio approaches 1.0 as selected subset approaches optimality.

Example

> > > opt_calib.solve()
> > >
> > > #### Plot is automatically generated, or call manually:
> > >
> > > opt_calib.plot()

See Also

calculate_optimal_configurations: Generates data for plotting calculate_detroot_whole: Provides normalization reference

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `plot_results()`

Plot optimal calibration results using unified results manager.

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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

    def _basic_plots():
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

    def _managed_plot():
        robot_name = self.calib_config.get("robot_name", self.model.name)
        results_manager = ResultsManager("optimal_calibration", robot_name)
        weights = (
            np.array(list(self.w_dict_sort.values()))
            if hasattr(self, "w_dict_sort")
            else np.array([])
        )
        results_manager.plot_optimal_calibration_results(
            configurations=self.optimal_configurations,
            weights=weights,
            title="Optimal Calibration Configuration Results",
        )

    plot_with_fallback(_managed_plot, _basic_plots, logger, "optimal_calibration")
```

### `get_optimal_output_dir(default='results')`

Return the directory optimal-configuration results are written to.

Honours the directory component of the configured `tasks.optimal_configuration.output.output_file`. The *filename* is managed by :class:`ResultsManager` (timestamped, one file per format), so only the directory part of that setting is used.

Parameters:

| Name      | Type  | Description                                  | Default     |
| --------- | ----- | -------------------------------------------- | ----------- |
| `default` | `str` | Directory to use when nothing is configured. | `'results'` |

Returns:

| Name  | Type | Description            |
| ----- | ---- | ---------------------- |
| `str` |      | Output directory path. |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def get_optimal_output_dir(self, default="results"):
    """Return the directory optimal-configuration results are written to.

    Honours the directory component of the configured
    ``tasks.optimal_configuration.output.output_file``. The *filename* is
    managed by :class:`ResultsManager` (timestamped, one file per format),
    so only the directory part of that setting is used.

    Args:
        default (str): Directory to use when nothing is configured.

    Returns:
        str: Output directory path.
    """
    configured = getattr(self, "_optimal_output_file", None)
    if not configured:
        return default
    return os.path.dirname(configured) or default
```

### `count_optimal_configurations()`

Return the number of selected configurations.

`optimal_configurations` is a *dict* holding the joint names alongside the selected joint configurations, so `len()` on it counts keys (2), not configurations. Always go through this helper.

Returns:

| Name  | Type | Description                                         |
| ----- | ---- | --------------------------------------------------- |
| `int` |      | Number of selected joint configurations, 0 if none. |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def count_optimal_configurations(self):
    """Return the number of selected configurations.

    ``optimal_configurations`` is a *dict* holding the joint names alongside
    the selected joint configurations, so ``len()`` on it counts keys (2),
    not configurations. Always go through this helper.

    Returns:
        int: Number of selected joint configurations, 0 if none.
    """
    configs = getattr(self, "optimal_configurations", None)
    if not configs:
        return 0
    return len(configs.get("calibration_joint_configurations", []))
```

### `save_results(output_dir=None)`

Save optimal configuration results using unified results manager.

Parameters:

| Name         | Type  | Description                                                                                                                                                                            | Default |
| ------------ | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `output_dir` | `str` | Directory to write results into. When omitted, the directory configured under the unified config's tasks.optimal_configuration.output.output_file is used, and failing that "results". | `None`  |

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def save_results(self, output_dir=None):
    """Save optimal configuration results using unified results manager.

    Args:
        output_dir (str, optional): Directory to write results into. When
            omitted, the directory configured under the unified config's
            ``tasks.optimal_configuration.output.output_file`` is used, and
            failing that ``"results"``.
    """
    if (
        not hasattr(self, "optimal_configurations")
        or self.optimal_configurations is None
    ):
        logger.warning(
            "No optimal configuration results to save. Run solve() first."
        )
        return

    if output_dir is None:
        output_dir = self.get_optimal_output_dir()

    try:
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
            "configuration_count": self.count_optimal_configurations(),
            "calibration_config": self.calib_config,
        }

        # D-optimality determinant root of the full information matrix.
        # Named after what it is: this is NOT a condition number.
        if hasattr(self, "detroot_whole"):
            results_dict["d_optimality_detroot"] = float(self.detroot_whole)

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
```

## `SOCPOptimizer(subX_dict, calib_config)`

Second-Order Cone Programming optimizer for configuration selection.

Implements the mathematical optimization for D-optimal experimental design using Second-Order Cone Programming (SOCP). This class formulates and solves the convex optimization problem that maximizes the determinant of the Fisher Information Matrix.

Mathematical Formulation

maximize t subject to: t ≤ det(Σᵢ wᵢ Xᵢ)^(1/n) Σᵢ wᵢ ≤ 1 wᵢ ≥ 0

Where:

- t is auxiliary variable for objective
- wᵢ are configuration weights
- Xᵢ are information matrices
- n is matrix dimension

The problem is solved using the CVXOPT solver with picos interface.

Attributes:

| Name           | Type   | Description                                             |
| -------------- | ------ | ------------------------------------------------------- |
| `pool`         | `dict` | Dictionary of information matrices indexed by config ID |
| `calib_config` | `dict` | Calibration parameters including sample count           |
| `problem`      |        | Picos optimization problem instance                     |
| `w`            |        | Decision variable for configuration weights             |
| `t`            |        | Auxiliary variable for determinant objective            |
| `solution`     |        | Optimization solution object                            |

Example

> > > optimizer = SOCPOptimizer(subX_dict, calib_config) weights, sorted_weights = optimizer.solve() print(f"Optimization status: {optimizer.solution.status}")

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
def __init__(self, subX_dict, calib_config):
    import picos as pc

    self.pool = subX_dict
    self.calib_config = calib_config
    self.problem = pc.Problem()
    self.w = pc.RealVariable("w", self.calib_config["NbSample"], lower=0)
    self.t = pc.RealVariable("t", 1)
```

## `Detmax(candidate_pool, NbChosen)`

Determinant Maximization optimizer using greedy exchange algorithm.

This class implements a heuristic optimization algorithm for D-optimal experimental design that uses a greedy exchange strategy to find near-optimal configuration subsets. Unlike the SOCP approach, this method provides a combinatorial solution that directly selects discrete configurations.

Algorithm Overview

The DetMax algorithm uses an iterative exchange procedure:

1. Initialize with a random subset of configurations
1. Iteratively add the configuration that maximally improves the determinant criterion
1. Remove the configuration whose absence minimally degrades the determinant criterion
1. Repeat until convergence (no beneficial exchanges)

Mathematical Background

The algorithm maximizes det(Σᵢ∈S Xᵢ)^(1/n) where:

- S is the selected configuration subset
- Xᵢ are information matrices for configurations
- n is the matrix dimension

This is a discrete optimization problem (vs continuous SOCP).

Advantages

- Provides exact discrete solution (no weight thresholding)
- Computationally efficient for small to medium problems
- Intuitive greedy strategy with good convergence properties
- No external optimization solvers required

Limitations

- May converge to local optima (not globally optimal)
- Performance depends on random initialization
- Computational complexity grows with candidate pool size

Attributes:

| Name        | Type   | Description                                             |
| ----------- | ------ | ------------------------------------------------------- |
| `pool`      | `dict` | Dictionary of information matrices indexed by config ID |
| `nd`        | `int`  | Number of configurations to select                      |
| `cur_set`   | `list` | Current configuration subset being evaluated            |
| `fail_set`  | `list` | Configurations that failed selection criteria           |
| `opt_set`   | `list` | Final optimal configuration subset                      |
| `opt_critD` | `list` | Evolution of determinant criterion during optimization  |

Example

> > > ### Create DetMax optimizer
> > >
> > > detmax = Detmax(subX_dict, num_configs=12)
> > >
> > > ### Run optimization
> > >
> > > criterion_history = detmax.main_algo()
> > >
> > > ### Get selected configurations
> > >
> > > selected_configs = detmax.cur_set final_criterion = criterion_history[-1]
> > >
> > > print(f"Selected {len(selected_configs)} configurations") print(f"Final D-optimality: {final_criterion:.4f}")

See Also

SOCPOptimizer: Alternative SOCP-based optimization approach BaseOptimalCalibration: Main calibration framework

Initialize DetMax optimizer with candidate pool and target size.

Sets up the determinant maximization optimizer with the candidate configuration pool and specifies the number of configurations to select in the final optimal subset.

Parameters:

| Name             | Type   | Description                                                                                                                                                        | Default    |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `candidate_pool` | `dict` | Dictionary mapping configuration indices to their corresponding information matrices. Keys are configuration IDs, values are symmetric positive definite matrices. | *required* |
| `NbChosen`       | `int`  | Number of configurations to select in the optimal subset. Must be less than total candidates and sufficient for parameter identifiability.                         | *required* |

Raises:

| Type         | Description                             |
| ------------ | --------------------------------------- |
| `ValueError` | If NbChosen exceeds candidate pool size |
| `TypeError`  | If candidate_pool is not a dictionary   |

Side Effects

- Initializes internal tracking lists (cur_set, fail_set, etc.)
- Stores candidate pool and selection target

Example

> > > ### Information matrices dict
> > >
> > > info_matrices = {0: X0, 1: X1, 2: X2, ...} optimizer = Detmax(info_matrices, NbChosen=10)

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `get_critD(set)`

Calculate D-optimality criterion for configuration subset.

Computes the determinant root of the Fisher Information Matrix formed by summing the information matrices of configurations in the specified subset. This serves as the objective function for the determinant maximization algorithm.

Parameters:

| Name  | Type   | Description                                                                                   | Default    |
| ----- | ------ | --------------------------------------------------------------------------------------------- | ---------- |
| `set` | `list` | List of configuration indices from the candidate pool to include in the criterion calculation | *required* |

Returns:

| Name    | Type | Description                                                                                             |
| ------- | ---- | ------------------------------------------------------------------------------------------------------- |
| `float` |      | D-optimality criterion value (determinant root) Higher values indicate better parameter identifiability |

Raises:

| Type             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `AssertionError` | If any configuration index not in candidate pool |

Mathematical Details

For subset S, computes: det(Σᵢ∈S Xᵢ)^(1/n) where Xᵢ are information matrices and n is matrix dimension

Example

> > > subset = [0, 5, 12, 18] # Configuration indices criterion = optimizer.get_critD(subset) print(f"D-optimality: {criterion:.6f}")

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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
```

### `main_algo()`

Execute the main determinant maximization algorithm.

Implements the greedy exchange algorithm for D-optimal experimental design. The algorithm alternates between adding configurations that maximally improve the determinant and removing configurations whose absence minimally degrades the determinant.

Algorithm Steps:

1. Initialize random subset of target size from candidate pool
1. Exchange Loop: a. ADD PHASE: Find configuration that maximally improves criterion b. REMOVE PHASE: Find configuration whose removal minimally hurts c. Update current subset and criterion value
1. Repeat until convergence (no beneficial exchanges)
1. Return optimization history

Convergence Condition

The algorithm stops when the optimal configuration to add equals the optimal configuration to remove, indicating no further improvement is possible.

Returns:

| Name   | Type | Description                                                                                                  |
| ------ | ---- | ------------------------------------------------------------------------------------------------------------ |
| `list` |      | History of D-optimality criterion values throughout the optimization process. Last value is final criterion. |

Side Effects

- Updates self.cur_set with final optimal configuration subset
- Updates self.opt_critD with complete optimization history
- Uses random initialization (results may vary between runs)

Complexity

O(max_iterations × candidate_pool_size × target_subset_size) where max_iterations depends on problem structure and initialization

Example

> > > optimizer = Detmax(info_matrices, NbChosen=10) history = optimizer.main_algo() print(f"Converged after {len(history)} iterations") print(f"Final subset: {optimizer.cur_set}") print(f"Final criterion: {history\[-1\]:.6f}")

Note

The algorithm may converge to different local optima depending on random initialization. For critical applications, consider running multiple times with different seeds.

Source code in `src/figaroh/optimal/base_optimal_calibration.py`

```
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

        self.opt_critD.append(opt_critD)
    return self.opt_critD
```

## base_optimal_trajectory

Base Optimal Trajectory Generation Framework

This module provides base classes for optimal trajectory generation with configuration management, parameter computation, constraint handling, and IPOPT-based optimization. This framework can be extended for different robots.

## `BaseOptimalTrajectory(robot, active_joints, config_file='config/robot_config.yaml')`

Base class for IPOPT-based optimal trajectory generation.

Features:

- Modular design with separated concerns
- Better error handling and logging
- Configuration validation
- Cleaner interfaces

This base class can be extended for specific robots by implementing robot-specific configuration loading and constraint handling.

Initialize the optimal trajectory generator.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `initialize()`

Initialize trajectory generation components.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `solve(stack_reps=2)`

Solve the optimal trajectory generation problem.

Parameters:

| Name         | Type  | Description                            | Default |
| ------------ | ----- | -------------------------------------- | ------- |
| `stack_reps` | `int` | Number of trajectory segments to stack | `2`     |

Returns:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `Dict[str, Any]` | Dict containing trajectories and optimization info |

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `objective_function(X, opt_cb, tps, vel_wps, acc_wps, wp_init, W_stack=None)`

Objective function: condition number of base regressor matrix.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `create_ipopt_problem(n_joints, n_wps, Ns, tps, vel_wps, acc_wps, wp_init, vel_wp_init, acc_wp_init, W_stack)`

Create IPOPT problem instance. Should be implemented by subclasses.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `plot_results()`

Plot optimal trajectory results using unified results manager.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def plot_results(self):
    """Plot optimal trajectory results using unified results manager."""
    if not self.results["T_F"]:
        self.logger.warning("No trajectory data to plot")
        return

    def _basic_plots():
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

                # P spans nq columns, V and A span nv: index each by
                # the active joint's own idx_q / idx_v (#149).
                for joint_idx, (iq, iv) in enumerate(
                    zip(self.CB.act_idxq, self.CB.act_idxv)
                ):
                    axes[joint_idx, 0].plot(T, P[:, iq], color=color, label=label)
                    axes[joint_idx, 1].plot(T, V[:, iv], color=color, label=label)
                    axes[joint_idx, 2].plot(T, A[:, iv], color=color, label=label)

            # Set labels and formatting
            for joint_idx, name in enumerate(self.CB.act_Jname):
                axes[joint_idx, 0].set_ylabel(f"{name}\nPosition (rad)")
                axes[joint_idx, 1].set_ylabel(f"{name}\nVelocity (rad/s)")
                axes[joint_idx, 2].set_ylabel(f"{name}\nAcceleration (rad/s²)")

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

    def _managed_plot():
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

        results_manager.plot_optimal_trajectory_results(
            trajectories=self.results,
            condition_number=condition_number,
            joint_names=self.CB.act_Jname,
            title="Optimal Trajectory Generation Results",
            q_indices=self.CB.act_idxq,
            v_indices=self.CB.act_idxv,
        )

    plot_with_fallback(
        _managed_plot, _basic_plots, self.logger, "optimal_trajectory"
    )
```

### `save_results(output_dir='results')`

Save optimal trajectory results using unified results manager.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def save_results(self, output_dir="results"):
    """Save optimal trajectory results using unified results manager."""
    if not self.results["T_F"]:
        self.logger.warning("No trajectory data to save")
        return

    try:
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

        self.logger.info("Trajectory results saved successfully")
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
```

## `BaseTrajectoryIPOPTProblem(opt_traj, n_joints, n_wps, Ns, tps, vel_wps, acc_wps, wp_init, vel_wp_init, acc_wp_init, W_stack, problem_name='TrajectoryOptimization')`

Bases: `BaseOptimizationProblem`

Base IPOPT problem formulation for trajectory optimization.

This class provides a base implementation for trajectory optimization that can be extended for specific robots.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `get_variable_bounds()`

Get variable bounds for optimization.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
    """Get variable bounds for optimization."""
    return self.opt_traj.constraint_manager.get_variable_bounds()
```

### `get_constraint_bounds()`

Get constraint bounds for optimization.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
    """Get constraint bounds for optimization."""
    return self.opt_traj.constraint_manager.get_constraint_bounds(self.Ns)
```

### `get_initial_guess()`

Get initial guess from waypoints.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def get_initial_guess(self) -> List[float]:
    """Get initial guess from waypoints."""
    # This will be set when solve() is called with waypoints
    if not hasattr(self, "_initial_wps"):
        # Return zeros as fallback
        return [0.0] * (self.n_joints * (self.n_wps - 1))

    X0 = self._initial_wps[:, range(1, self.n_wps)]
    return np.reshape(X0.transpose(), (self.n_joints * (self.n_wps - 1),)).tolist()
```

### `objective(X)`

Objective function: condition number of base regressor matrix.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `constraints(X)`

Constraint function for IPOPT.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def constraints(self, X: np.ndarray) -> np.ndarray:
    """Constraint function for IPOPT."""
    return self.opt_traj.constraint_manager.evaluate_constraints(
        self.Ns, X, self.opt_cb, self.tps, self.vel_wps, self.acc_wps, self.wp_init
    )
```

### `gradient(X)`

Gradient of the objective by forward finite differences.

The default numdifftools gradient uses Richardson extrapolation over many step sizes, which costs tens of seconds per call here because every objective evaluation rebuilds the full regressor.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
def gradient(self, X: np.ndarray) -> np.ndarray:
    """
    Gradient of the objective by forward finite differences.

    The default numdifftools gradient uses Richardson extrapolation over
    many step sizes, which costs tens of seconds per call here because
    every objective evaluation rebuilds the full regressor.
    """
    X = np.asarray(X, dtype=float)
    f0 = self.objective(X)
    eps = 1e-6
    grad = np.zeros_like(X)
    for i in range(len(X)):
        X_plus = X.copy()
        X_plus[i] += eps
        grad[i] = (self.objective(X_plus) - f0) / eps
    return grad
```

### `jacobian(X)`

Jacobian of constraints - Custom implementation for better performance.

For trajectory optimization, we can use sparse finite differences instead of full automatic differentiation which is too slow.

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
```

### `solve_with_waypoints(wps)`

Solve the optimization problem with given initial waypoints.

Parameters:

| Name  | Type | Description       | Default    |
| ----- | ---- | ----------------- | ---------- |
| `wps` |      | Initial waypoints | *required* |

Returns:

| Type                          | Description                      |
| ----------------------------- | -------------------------------- |
| `Tuple[bool, Dict[str, Any]]` | Tuple of (success, results_dict) |

Source code in `src/figaroh/optimal/base_optimal_trajectory.py`

```
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
        config.max_iterations = self.opt_traj.trajectory_config.get(
            "max_iterations", 200
        )
        config.print_level = 3  # Reduce output
        config.custom_options = {
            b"mu_strategy": b"adaptive",
        }
        solver = RobotIPOPTSolver(self, config)

        # Solve the problem
        success, results = solver.solve()
        converged = success

        # The condition-number objective is non-smooth, so IPOPT often
        # stalls on dual infeasibility after reaching a feasible plateau.
        # Keep such an iterate, but only if it satisfies every constraint.
        if not success and results.get("status") == -1:
            success = self._is_feasible(results["x_opt"])
            if success:
                self.logger.warning(
                    "IPOPT reached the iteration limit (%d) without "
                    "converging; keeping the feasible final iterate "
                    "(objective %.4g)",
                    config.max_iterations,
                    results["obj_val"],
                )

        # Collision is constrained at check points only; accept the
        # segment only if the whole spline is clear (#143).
        if success and not self._is_collision_free(results["x_opt"]):
            self.logger.error(
                "Optimised segment collides between collision check points"
            )
            return False, results

        if success:
            # Extract final waypoint for next segment
            X_opt = results["x_opt"]
            wps_X = np.reshape(np.array(X_opt), (self.n_wps - 1, self.n_joints))
            final_waypoint = wps_X[-1, :]
            # The last callback may have been a finite-difference probe;
            # rebuild the stored trajectory at the returned solution.
            self.objective(np.asarray(X_opt))

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
                        "converged": converged,
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
```
