# Utilities

## Configuration Parser

Unified Configuration Parser for FIGAROH

This module provides a unified configuration parsing system that supports:

- Template inheritance and extension
- Variant configurations
- Variable expansion
- Schema validation

## `ConfigurationError`

Bases: `FigarohExampleError`

Exception raised for configuration-related issues.

## `ConfigMetadata(schema_version='2.0', config_type='robot_configuration', source_file=None, variant=None, resolved_at=None)`

Configuration metadata container.

## `UnifiedConfigParser(config_path, variant=None)`

Unified configuration parser supporting inheritance, templates, and validation.

Initialize parser with configuration file path.

Parameters:

| Name          | Type               | Description                                     | Default    |
| ------------- | ------------------ | ----------------------------------------------- | ---------- |
| `config_path` | `Union[str, Path]` | Path to configuration file                      | *required* |
| `variant`     | `Optional[str]`    | Optional variant name to use from configuration | `None`     |

Source code in `src/figaroh/utils/config_parser.py`

```
def __init__(self, config_path: Union[str, Path], variant: Optional[str] = None):
    """Initialize parser with configuration file path.

    Args:
        config_path: Path to configuration file
        variant: Optional variant name to use from configuration
    """
    self.config_path = Path(config_path).resolve()
    self.variant = variant
    self.logger = logging.getLogger(self.__class__.__name__)
    self._template_cache = {}
    self._variable_cache = {}

    # Validate config file exists
    if not self.config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {self.config_path}"
        )
```

### `parse()`

Parse configuration with inheritance and validation.

Returns:

| Type             | Description                                     |
| ---------------- | ----------------------------------------------- |
| `Dict[str, Any]` | Complete configuration dictionary with metadata |

Raises:

| Type                 | Description                    |
| -------------------- | ------------------------------ |
| `ConfigurationError` | If parsing or validation fails |

Source code in `src/figaroh/utils/config_parser.py`

```
def parse(self) -> Dict[str, Any]:
    """Parse configuration with inheritance and validation.

    Returns:
        Complete configuration dictionary with metadata

    Raises:
        ConfigurationError: If parsing or validation fails
    """
    try:
        self.logger.info(f"Parsing configuration: {self.config_path}")

        # Load base configuration
        config = self._load_config_file(self.config_path)

        # Handle inheritance/extensions
        if "extends" in config:
            config = self._resolve_inheritance(config)

        # Apply variant if specified
        if self.variant:
            config = self._apply_variant(config, self.variant)

        # Resolve task inheritance
        config = self._resolve_task_inheritance(config)

        # Expand variables
        config = self._expand_variables(config)

        # Validate configuration
        self._validate_configuration(config)

        # Add metadata
        config["_metadata"] = ConfigMetadata(
            source_file=str(self.config_path),
            variant=self.variant,
            resolved_at=str(Path.cwd()),
        ).__dict__

        self.logger.info("Configuration parsing completed successfully")
        return config

    except Exception as e:
        self.logger.error(f"Failed to parse configuration: {e}")
        raise ConfigurationError(
            f"Failed to parse configuration {self.config_path}: {e}"
        ) from e
```

## `create_task_config(robot, unified_config, task_name)`

Create task-specific configuration from unified config.

Parameters:

| Name             | Type             | Description                               | Default    |
| ---------------- | ---------------- | ----------------------------------------- | ---------- |
| `robot`          |                  | Robot instance                            | *required* |
| `unified_config` | `Dict[str, Any]` | Parsed unified configuration              | *required* |
| `task_name`      | `str`            | Name of task to extract configuration for | *required* |

Returns:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `Dict[str, Any]` | Task-specific configuration dictionary |

Raises:

| Type                 | Description                  |
| -------------------- | ---------------------------- |
| `ConfigurationError` | If task not found or invalid |

Source code in `src/figaroh/utils/config_parser.py`

```
def create_task_config(
    robot, unified_config: Dict[str, Any], task_name: str
) -> Dict[str, Any]:
    """Create task-specific configuration from unified config.

    Args:
        robot: Robot instance
        unified_config: Parsed unified configuration
        task_name: Name of task to extract configuration for

    Returns:
        Task-specific configuration dictionary

    Raises:
        ConfigurationError: If task not found or invalid
    """
    if "tasks" not in unified_config or task_name not in unified_config["tasks"]:
        available_tasks = list(unified_config.get("tasks", {}).keys())
        raise ConfigurationError(
            f"Task '{task_name}' not found in configuration. Available tasks: {available_tasks}"
        )

    task_config = unified_config["tasks"][task_name]
    robot_config = unified_config.get("robot", {})

    # Check if task is enabled
    if not task_config.get("enabled", True):
        raise ConfigurationError(f"Task '{task_name}' is disabled in configuration")

    # Combine robot properties with task configuration
    combined_config = {
        "robot_name": robot_config.get("name", robot.model.name),
        "task_type": task_config.get("type", task_name),
    }

    # Add robot properties
    if "properties" in robot_config:
        combined_config.update(robot_config["properties"])

    # Physical-asset identity, distinct from the model type above (e.g.
    # "ur10" is shared by every UR10; robot.instance.asset_id identifies
    # one physical unit). Optional — omitted configs simply carry no
    # instance block, and provenance capture renders that as
    # "unspecified unit" rather than guessing.
    if "instance" in robot_config:
        combined_config["instance"] = robot_config["instance"]

    # Add task configuration (excluding metadata)
    task_data = {
        k: v
        for k, v in task_config.items()
        if not k.startswith("_") and k not in ["enabled", "type"]
    }
    combined_config.update(task_data)

    # Add custom parameters
    if "custom" in unified_config:
        combined_config["custom"] = unified_config["custom"]

    # Add environment settings that might be relevant
    if "environment" in unified_config:
        env = unified_config["environment"]
        combined_config.update(
            {
                "working_directory": env.get("working_directory", "."),
                "data_directory": env.get("data_directory", "data"),
                "results_directory": env.get("results_directory", "results"),
            }
        )

    return combined_config
```

## `parse_configuration(config_path, variant=None)`

Convenience function to parse a configuration file.

Parameters:

| Name          | Type               | Description                    | Default    |
| ------------- | ------------------ | ------------------------------ | ---------- |
| `config_path` | `Union[str, Path]` | Path to configuration file     | *required* |
| `variant`     | `Optional[str]`    | Optional variant name to apply | `None`     |

Returns:

| Type             | Description                     |
| ---------------- | ------------------------------- |
| `Dict[str, Any]` | Parsed configuration dictionary |

Source code in `src/figaroh/utils/config_parser.py`

```
def parse_configuration(
    config_path: Union[str, Path], variant: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to parse a configuration file.

    Args:
        config_path: Path to configuration file
        variant: Optional variant name to apply

    Returns:
        Parsed configuration dictionary
    """
    parser = UnifiedConfigParser(config_path, variant)
    return parser.parse()
```

## `get_param_from_yaml(robot, config_data, task_type='auto', variant=None)`

Unified parameter parser.

Parameters:

| Name          | Type                               | Description                                                                     | Default    |
| ------------- | ---------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| `robot`       |                                    | Robot instance                                                                  | *required* |
| `config_data` | `Union[Dict[str, Any], str, Path]` | Configuration data (dict), or path to config file                               | *required* |
| `task_type`   | `str`                              | Type of task configuration to extract ("calibration", "identification", "auto") | `'auto'`   |
| `variant`     | `Optional[str]`                    | Optional variant to apply                                                       | `None`     |

Returns:

| Type             | Description                        |
| ---------------- | ---------------------------------- |
| `Dict[str, Any]` | Task-specific parameter dictionary |

Raises:

| Type                 | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `ConfigurationError` | If parsing fails, the task is not found, or the data is a legacy flat config (removed in 1.0) |

Source code in `src/figaroh/utils/config_parser.py`

```
def get_param_from_yaml(
    robot,
    config_data: Union[Dict[str, Any], str, Path],
    task_type: str = "auto",
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified parameter parser.

    Args:
        robot: Robot instance
        config_data: Configuration data (dict), or path to config file
        task_type: Type of task configuration to extract ("calibration", "identification", "auto")
        variant: Optional variant to apply

    Returns:
        Task-specific parameter dictionary

    Raises:
        ConfigurationError: If parsing fails, the task is not found, or the
            data is a legacy flat config (removed in 1.0)
    """
    # Handle file path input
    if isinstance(config_data, (str, Path)):
        unified_config = parse_configuration(config_data, variant)

        # Auto-detect task type if needed
        if task_type == "auto":
            tasks = unified_config.get("tasks", {})
            enabled_tasks = [
                name for name, config in tasks.items() if config.get("enabled", True)
            ]
            if len(enabled_tasks) == 1:
                task_type = enabled_tasks[0]
            else:
                raise ConfigurationError(
                    f"Cannot auto-detect task type. Available tasks: {enabled_tasks}"
                )

        return create_task_config(robot, unified_config, task_type)

    # Handle dictionary input (must be unified)
    if not isinstance(config_data, dict):
        raise ConfigurationError("config_data must be a dictionary or file path")

    # Check if it's already a unified configuration
    if "tasks" in config_data:
        # Unified format
        if task_type == "auto":
            tasks = config_data.get("tasks", {})
            enabled_tasks = [
                name for name, config in tasks.items() if config.get("enabled", True)
            ]
            if len(enabled_tasks) == 1:
                task_type = enabled_tasks[0]
            else:
                raise ConfigurationError(
                    f"Cannot auto-detect task type. Available tasks: {enabled_tasks}"
                )

        return create_task_config(robot, config_data, task_type)

    # Legacy flat format (removed in 1.0)
    raise legacy_config_error()
```

## `legacy_config_error(config_file=None)`

Build the error raised when a legacy flat config is loaded.

Parameters:

| Name          | Type                         | Description                                      | Default |
| ------------- | ---------------------------- | ------------------------------------------------ | ------- |
| `config_file` | `Optional[Union[str, Path]]` | Offending file, named in the message when given. | `None`  |

Returns:

| Type                 | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `ConfigurationError` | An error naming the migration CLI. The caller raises it. |

Source code in `src/figaroh/utils/config_parser.py`

```
def legacy_config_error(
    config_file: Optional[Union[str, Path]] = None,
) -> ConfigurationError:
    """Build the error raised when a legacy flat config is loaded.

    Args:
        config_file: Offending file, named in the message when given.

    Returns:
        An error naming the migration CLI. The caller raises it.
    """
    where = f" ({config_file})" if config_file is not None else ""
    return ConfigurationError(f"{LEGACY_CONFIG_REMOVED_MESSAGE}{where}")
```

## `require_unified_config(config_file)`

Raise unless `config_file` is an existing unified-format config.

Raises:

| Type                 | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `FileNotFoundError`  | The file does not exist.                                    |
| `ConfigurationError` | The file is a legacy flat config (names the migration CLI). |

Source code in `src/figaroh/utils/config_parser.py`

```
def require_unified_config(config_file: Union[str, Path]) -> None:
    """Raise unless ``config_file`` is an existing unified-format config.

    Raises:
        FileNotFoundError: The file does not exist.
        ConfigurationError: The file is a legacy flat config (names the
            migration CLI).
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    if not is_unified_config(config_file):
        raise legacy_config_error(config_file)
```

## `is_unified_config(config_file)`

Check if a configuration file is using the unified format.

Parameters:

| Name          | Type               | Description                | Default    |
| ------------- | ------------------ | -------------------------- | ---------- |
| `config_file` | `Union[str, Path]` | Path to configuration file | *required* |

Returns:

| Type   | Description                                    |
| ------ | ---------------------------------------------- |
| `bool` | True if unified format, False if legacy format |

Source code in `src/figaroh/utils/config_parser.py`

```
def is_unified_config(config_file: Union[str, Path]) -> bool:
    """Check if a configuration file is using the unified format.

    Args:
        config_file: Path to configuration file

    Returns:
        True if unified format, False if legacy format
    """
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return False

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            return False

        # Check for unified format indicators
        unified_indicators = [
            "_metadata",
            "schema_version",
            "extends",
            "variants",
            "tasks",
            "common",
            "variables",
        ]

        # If any unified indicators are present, consider it unified
        for indicator in unified_indicators:
            if indicator in config:
                logger.debug(f"Detected unified config indicator: {indicator}")
                return True

        # Check if the file has task inheritance patterns
        for key, value in config.items():
            if isinstance(value, dict) and "inherits_from" in value:
                logger.debug(f"Detected task inheritance in {key}")
                return True

        # Default to legacy format
        logger.debug("No unified format indicators found, treating as legacy")
        return False

    except Exception as e:
        logger.warning(f"Error checking config format for {config_file}: {e}")
        return False
```

## Cubic Spline

Cubic Spline Trajectory Generation for Robotics Applications.

This module provides comprehensive tools for generating smooth, continuous trajectories using cubic splines for robotic systems. It supports constraint handling, waypoint generation, and trajectory validation with joint limits.

The module contains two main classes:

- CubicSpline: Core trajectory generation with cubic splines
- WaypointsGeneration: Automated waypoint generation for trajectory planning

Key Features

- C2 continuous cubic spline trajectories
- Joint position, velocity, and acceleration constraints
- Waypoint-based trajectory planning
- Collision and constraint checking
- Visualization capabilities
- Random and systematic waypoint generation

Dependencies

- ndcurves: Curve generation and manipulation
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- pinocchio: Robot kinematics and dynamics

Examples:

Basic trajectory generation:

```
from figaroh.utils.cubic_spline import CubicSpline

# Initialize spline generator
spline = CubicSpline(robot, num_waypoints=5,
                   active_joints=['joint1', 'joint2'])

# Generate trajectory
time_points = np.array([[0], [1], [2], [3], [4]])
waypoints = np.random.rand(2, 5)  # 2 joints, 5 waypoints

t, pos, vel, acc = spline.get_full_config(
    freq=100, time_points=time_points, waypoints=waypoints
)

# Check constraints
spline.check_cfg_constraints(pos, vel)
```

Automated waypoint generation:

```
from figaroh.utils.cubic_spline import WaypointsGeneration

# Initialize waypoint generator
wp_gen = WaypointsGeneration(robot, num_waypoints=5,
                           active_joints=['joint1', 'joint2'])

# Generate waypoint pool
wp_gen.gen_rand_pool()

# Generate random waypoints
pos_wp, vel_wp, acc_wp = wp_gen.gen_rand_wp()

# Generate trajectory from waypoints
t, pos, vel, acc = wp_gen.get_full_config(
    freq=100, time_points=time_points, waypoints=pos_wp
)
```

References

- Spline theory: "A Practical Guide to Splines" by Carl de Boor
- Robot trajectory planning: "Introduction to Robotics" by John Craig
- ndcurves library: https://github.com/humanoid-path-planner/ndcurves

## `CubicSpline(robot, num_waypoints, active_joints, soft_lim=0)`

Cubic spline trajectory generator for robotic systems.

This class generates smooth, C2-continuous trajectories using cubic splines for robotic systems. It supports both position-only and full kinematic constraint specification (position, velocity, acceleration) at waypoints.

The class handles active joint selection, joint limit constraints, and provides methods for trajectory validation and visualization.

Attributes:

| Name            | Type                       | Description                        |
| --------------- | -------------------------- | ---------------------------------- |
| `robot`         |                            | Robot model instance               |
| `rmodel`        |                            | Robot kinematic model              |
| `num_waypoints` | `int`                      | Number of waypoints in trajectory  |
| `act_Jid`       | `List[int]`                | Active joint IDs                   |
| `act_Jname`     | `List[str]`                | Active joint names                 |
| `act_J`         | `List`                     | Active joint objects               |
| `act_idxq`      | `List[int]`                | Position indices for active joints |
| `act_idxv`      | `List[int]`                | Velocity indices for active joints |
| `dim_q`         | `Tuple[int, int]`          | Position vector dimensions         |
| `dim_v`         | `Tuple[int, int]`          | Velocity vector dimensions         |
| `upper_q,`      | `lower_q (np.ndarray`      | Position limits for active joints  |
| `upper_dq,`     | `lower_dq (np.ndarray`     | Velocity limits for active joints  |
| `upper_effort,` | `lower_effort (np.ndarray` | Effort limits for active joints    |

Examples:

Basic usage:

```
# Initialize for 2 joints, 5 waypoints
spline = CubicSpline(robot, num_waypoints=5,
                   active_joints=['joint1', 'joint2'])

# Define waypoints and time stamps
waypoints = np.array([[0, 1, 2, 1, 0],      # joint1 positions
                    [0, 0.5, 1, 0.5, 0]])   # joint2 positions
time_points = np.array([[0], [1], [2], [3], [4]])

# Generate trajectory at 100 Hz
t, pos, vel, acc = spline.get_full_config(
    freq=100, time_points=time_points, waypoints=waypoints
)

# Validate constraints
is_violated = spline.check_cfg_constraints(pos, vel)
```

With velocity and acceleration constraints:

```
# Define waypoint constraints
vel_waypoints = np.zeros((2, 5))  # Zero velocity at waypoints
acc_waypoints = np.zeros((2, 5))  # Zero acceleration at waypoints

# Generate constrained trajectory
t, pos, vel, acc = spline.get_full_config(
    freq=100, time_points=time_points, waypoints=waypoints,
    vel_waypoints=vel_waypoints, acc_waypoints=acc_waypoints
)

# Visualize results
spline.plot_spline(t, pos, vel, acc)
```

Note

The class uses the ndcurves library for cubic spline generation. Joint limits are automatically extracted from the robot model and can be modified with soft limits for safety margins.

Initialize the cubic spline trajectory generator.

Parameters:

| Name            | Type    | Description                                                                                         | Default    |
| --------------- | ------- | --------------------------------------------------------------------------------------------------- | ---------- |
| `robot`         |         | Robot model instance containing kinematic information                                               | *required* |
| `num_waypoints` | `int`   | Number of waypoints for the trajectory                                                              | *required* |
| `active_joints` | `list`  | List of joint names to include in trajectory                                                        | *required* |
| `soft_lim`      | `float` | Soft limit reduction factor (0-1). Reduces joint limits by this fraction for safety. Defaults to 0. | `0`        |

Raises:

| Type             | Description                                 |
| ---------------- | ------------------------------------------- |
| `AssertionError` | If joint names are not found in robot model |

Examples:

```
# Basic initialization
spline = CubicSpline(robot, 5, ['joint1', 'joint2'])

# With 10% safety margin on joint limits
spline = CubicSpline(robot, 5, ['joint1', 'joint2'],
                   soft_lim=0.1)
```

Source code in `src/figaroh/utils/cubic_spline.py`

````
def __init__(self, robot, num_waypoints: int, active_joints: list, soft_lim=0):
    """
    Initialize the cubic spline trajectory generator.

    Args:
        robot: Robot model instance containing kinematic information
        num_waypoints (int): Number of waypoints for the trajectory
        active_joints (list): List of joint names to include in trajectory
        soft_lim (float, optional): Soft limit reduction factor (0-1).
            Reduces joint limits by this fraction for safety.
            Defaults to 0.

    Raises:
        AssertionError: If joint names are not found in robot model

    Examples:
        ```python
        # Basic initialization
        spline = CubicSpline(robot, 5, ['joint1', 'joint2'])

        # With 10% safety margin on joint limits
        spline = CubicSpline(robot, 5, ['joint1', 'joint2'],
                           soft_lim=0.1)
        ```
    """

    self.robot = robot
    self.rmodel = self.robot.model
    self.num_waypoints = num_waypoints
    # joint id of active joints
    self.act_Jid = [self.rmodel.getJointId(i) for i in active_joints]
    # active joint objects and their names
    self.act_Jname = [self.rmodel.names[jid] for jid in self.act_Jid]
    self.act_J = [self.rmodel.joints[jid] for jid in self.act_Jid]
    # joint config id (e.g one joint might have >1 DOF)
    self.act_idxq = [J.idx_q for J in self.act_J]
    # joint velocity id
    self.act_idxv = [J.idx_v for J in self.act_J]

    # size of waypoints vector for all active joints
    self.dim_q = (len(self.act_idxq), self.num_waypoints)
    self.dim_v = (len(self.act_idxv), self.num_waypoints)

    # joint limits on active joints
    self.upper_q = self.rmodel.upperPositionLimit[self.act_idxq]
    self.lower_q = self.rmodel.lowerPositionLimit[self.act_idxq]

    self.upper_dq = self.rmodel.velocityLimit[self.act_idxv]
    self.lower_dq = -self.rmodel.velocityLimit[self.act_idxv]

    self.upper_effort = self.rmodel.effortLimit[self.act_idxv]
    self.lower_effort = -self.rmodel.effortLimit[self.act_idxv]

    # joint limits on active joints with soft limit on both limit ends
    if soft_lim > 0:
        self.upper_q = self.upper_q - soft_lim * abs(self.upper_q - self.lower_q)
        self.lower_q = self.lower_q + soft_lim * abs(self.upper_q - self.lower_q)

        self.upper_dq = self.upper_dq - soft_lim * abs(
            self.upper_dq - self.lower_dq
        )
        self.lower_dq = self.lower_dq + soft_lim * abs(
            self.upper_dq - self.lower_dq
        )

        self.upper_effort = self.upper_effort - soft_lim * abs(
            self.upper_effort - self.lower_effort
        )
        self.lower_effort = self.lower_effort + soft_lim * abs(
            self.upper_effort - self.lower_effort
        )
````

### `get_active_config(freq, time_points, waypoints, vel_waypoints=None, acc_waypoints=None)`

Generate cubic splines on active joints

Source code in `src/figaroh/utils/cubic_spline.py`

```
def get_active_config(
    self,
    freq: int,
    time_points: np.ndarray,
    waypoints: np.ndarray,
    vel_waypoints=None,
    acc_waypoints=None,
):
    """Generate cubic splines on active joints"""
    # dimensions
    assert (
        self.dim_q == waypoints.shape
    ), "(Pos) Check size \
                                    (num_active_joints,num_waypoints)!"
    self.pc = ndcurves.piecewise()  # set piecewise object to join segments

    # C_2 continuous at waypoints
    if vel_waypoints is not None and acc_waypoints is not None:
        # dimensions
        assert (
            self.dim_v == vel_waypoints.shape
        ), "(Vel) Check size\
                                    (num_active_joints, num_waypoints)!"
        assert (
            self.dim_v == acc_waypoints.shape
        ), "(Acc) Check size\
                                    (num_active_joints, num_waypoints)!"

        # make exact cubic WITH constraints on vel and acc on both ends
        for i in range(self.num_waypoints - 1):
            self.c = ndcurves.curve_constraints()
            self.c.init_vel = np.matrix(vel_waypoints[:, i]).transpose()
            self.c.end_vel = np.matrix(vel_waypoints[:, i + 1]).transpose()
            self.c.init_acc = np.matrix(acc_waypoints[:, i]).transpose()
            self.c.end_acc = np.matrix(acc_waypoints[:, i + 1]).transpose()
            ec = ndcurves.exact_cubic(
                waypoints[:, range(i, i + 2)],
                time_points[range(i, i + 2), 0],
                self.c,
            )
            self.pc.append(ec)

    # make exact cubic WITHOUT constraints on vel and acc on both ends
    else:
        for i in range(self.num_waypoints - 1):
            ec = ndcurves.exact_cubic(
                waypoints[:, range(i, i + 2)], time_points[range(i, i + 2), 0]
            )  # Added spaces around '+'
            self.pc.append(ec)

    # time step
    self.delta_t = 1 / freq  # Added spaces around '/'
    # total travel time
    self.T = self.pc.max() - self.pc.min()
    # get number sample points from generated trajectory
    self.N = int(self.T / self.delta_t) + 1
    # create time stamps on all sample points
    self.t = (
        self.pc.min()
        + np.matrix([i * self.delta_t for i in range(self.N)]).transpose()
    )

    # compute derivatives to obtain pos/vel/acc on all samples (bad)
    self.q_act = np.array(
        [self.pc(self.t[i, 0]) for i in range(self.N)], dtype="float"
    )
    self.dq_act = np.array(
        [self.pc.derivate(self.t[i, 0], 1) for i in range(self.N)], dtype="float"
    )
    self.ddq_act = np.array(
        [self.pc.derivate(self.t[i, 0], 2) for i in range(self.N)], dtype="float"
    )
    t, p_act, v_act, a_act = self.t, self.q_act, self.dq_act, self.ddq_act

    return t, p_act, v_act, a_act
```

### `get_full_config(freq, time_points, waypoints, vel_waypoints=None, acc_waypoints=None)`

Generate complete robot configuration trajectory with cubic splines.

This method creates smooth trajectories for all robot joints by:

1. Generating cubic splines for active joints between waypoints
1. Filling inactive joints with zero values
1. Ensuring C2 continuity at waypoints

Parameters:

| Name            | Type      | Description                                                                                                                | Default    |
| --------------- | --------- | -------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `freq`          | `int`     | Sampling frequency for trajectory generation (Hz)                                                                          | *required* |
| `time_points`   | `ndarray` | Time stamps for waypoints, shape (num_waypoints, 1)                                                                        | *required* |
| `waypoints`     | `ndarray` | Position waypoints for active joints, shape (num_active_joints, num_waypoints)                                             | *required* |
| `vel_waypoints` | `ndarray` | Velocity constraints at waypoints, same shape as waypoints. If provided, enforces specific velocities at waypoints.        | `None`     |
| `acc_waypoints` | `ndarray` | Acceleration constraints at waypoints, same shape as waypoints. If provided, enforces specific accelerations at waypoints. | `None`     |

Returns:

| Name    | Type | Description                                                                                                                                                                                                                                                                                                                                                         |
| ------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tuple` |      | Four-element tuple containing: - t (np.ndarray): Time stamps, shape (N, 1) - q_full (np.ndarray): Position trajectory for all joints, shape (N, robot_nq) - dq_full (np.ndarray): Velocity trajectory for all joints, shape (N, robot_nv) - ddq_full (np.ndarray): Acceleration trajectory for all joints, shape (N, robot_nv) Where N = int(total_time * freq) + 1 |

Raises:

| Type             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `AssertionError` | If waypoint dimensions don't match active joints |

Examples:

Position-only trajectory:

```
time_pts = np.array([[0], [1], [2], [3]])
waypts = np.array([[0, 1, 2, 1],     # joint1
                  [0, 0.5, 1, 0.5]]) # joint2

t, q, dq, ddq = spline.get_full_config(
    freq=100, time_points=time_pts, waypoints=waypts
)
```

With velocity/acceleration constraints:

```
vel_waypts = np.zeros((2, 4))  # Zero velocity at waypoints
acc_waypts = np.zeros((2, 4))  # Zero acceleration at waypoints

t, q, dq, ddq = spline.get_full_config(
    freq=100, time_points=time_pts, waypoints=waypts,
    vel_waypoints=vel_waypts, acc_waypoints=acc_waypts
)
```

Note

The trajectory uses piecewise cubic curves connected at waypoints. When velocity and acceleration constraints are provided, the resulting trajectory enforces exact values at waypoints.

Source code in `src/figaroh/utils/cubic_spline.py`

````
def get_full_config(
    self,
    freq: int,
    time_points: np.ndarray,
    waypoints: np.ndarray,
    vel_waypoints=None,
    acc_waypoints=None,
):
    """
    Generate complete robot configuration trajectory with cubic splines.

    This method creates smooth trajectories for all robot joints by:
    1. Generating cubic splines for active joints between waypoints
    2. Filling inactive joints with zero values
    3. Ensuring C2 continuity at waypoints

    Args:
        freq (int): Sampling frequency for trajectory generation (Hz)
        time_points (np.ndarray): Time stamps for waypoints,
            shape (num_waypoints, 1)
        waypoints (np.ndarray): Position waypoints for active joints,
            shape (num_active_joints, num_waypoints)
        vel_waypoints (np.ndarray, optional): Velocity constraints at
            waypoints, same shape as waypoints. If provided, enforces
            specific velocities at waypoints.
        acc_waypoints (np.ndarray, optional): Acceleration constraints at
            waypoints, same shape as waypoints. If provided, enforces
            specific accelerations at waypoints.

    Returns:
        tuple: Four-element tuple containing:
            - t (np.ndarray): Time stamps, shape (N, 1)
            - q_full (np.ndarray): Position trajectory for all joints,
              shape (N, robot_nq)
            - dq_full (np.ndarray): Velocity trajectory for all joints,
              shape (N, robot_nv)
            - ddq_full (np.ndarray): Acceleration trajectory for all
              joints, shape (N, robot_nv)

            Where N = int(total_time * freq) + 1

    Raises:
        AssertionError: If waypoint dimensions don't match active joints

    Examples:
        Position-only trajectory:
            ```python
            time_pts = np.array([[0], [1], [2], [3]])
            waypts = np.array([[0, 1, 2, 1],     # joint1
                              [0, 0.5, 1, 0.5]]) # joint2

            t, q, dq, ddq = spline.get_full_config(
                freq=100, time_points=time_pts, waypoints=waypts
            )
            ```

        With velocity/acceleration constraints:
            ```python
            vel_waypts = np.zeros((2, 4))  # Zero velocity at waypoints
            acc_waypts = np.zeros((2, 4))  # Zero acceleration at waypoints

            t, q, dq, ddq = spline.get_full_config(
                freq=100, time_points=time_pts, waypoints=waypts,
                vel_waypoints=vel_waypts, acc_waypoints=acc_waypts
            )
            ```

    Note:
        The trajectory uses piecewise cubic curves connected at waypoints.
        When velocity and acceleration constraints are provided, the
        resulting trajectory enforces exact values at waypoints.
    """
    t, p_act, v_act, a_act = self.get_active_config(
        freq, time_points, waypoints, vel_waypoints, acc_waypoints
    )
    # create array of zero configuration times N samples
    self.q_full = np.array([self.robot.q0] * self.N)
    self.dq_full = np.array([np.zeros_like(self.robot.v0)] * self.N)
    self.ddq_full = np.array([np.zeros_like(self.robot.v0)] * self.N)

    # fill in trajectory with active joints values
    self.q_full[:, self.act_idxq] = p_act
    self.dq_full[:, self.act_idxv] = v_act
    self.ddq_full[:, self.act_idxv] = a_act

    p_full, v_full, a_full = self.q_full, self.dq_full, self.ddq_full
    return t, p_full, v_full, a_full
````

### `check_cfg_constraints(q, v=None, tau=None, soft_lim=0)`

Check joint constraints violation for trajectory configurations.

Validates whether the generated trajectory respects robot joint limits including position, velocity, and effort constraints. Provides detailed violation reporting for debugging.

Parameters:

| Name       | Type      | Description                                                                                               | Default    |
| ---------- | --------- | --------------------------------------------------------------------------------------------------------- | ---------- |
| `q`        | `ndarray` | Position trajectory to check, shape (N, robot_nq)                                                         | *required* |
| `v`        | `ndarray` | Velocity trajectory to check, shape (N, robot_nv). If None, velocity checks are skipped.                  | `None`     |
| `tau`      | `ndarray` | Effort trajectory to check, shape (N, robot_nv). If None, effort checks are skipped.                      | `None`     |
| `soft_lim` | `float`   | Additional safety margin factor (0-1). Adds this fraction of joint range as safety buffer. Defaults to 0. | `0`        |

Returns:

| Name   | Type | Description                                                                |
| ------ | ---- | -------------------------------------------------------------------------- |
| `bool` |      | True if any constraint is violated, False if all constraints are satisfied |

Examples:

Basic position check:

```
t, q, dq, ddq = spline.get_full_config(...)
is_violated = spline.check_cfg_constraints(q)
if is_violated:
    print("Trajectory violates position limits!")
```

Full constraint check with safety margin:

```
is_violated = spline.check_cfg_constraints(
    q, v=dq, tau=torques, soft_lim=0.1
)
```

Note

Constraint violations are logged at DEBUG level with specific joint indices and violation types; callers report the outcome.

Source code in `src/figaroh/utils/cubic_spline.py`

````
def check_cfg_constraints(self, q, v=None, tau=None, soft_lim=0):
    """
    Check joint constraints violation for trajectory configurations.

    Validates whether the generated trajectory respects robot joint
    limits including position, velocity, and effort constraints.
    Provides detailed violation reporting for debugging.

    Args:
        q (np.ndarray): Position trajectory to check,
            shape (N, robot_nq)
        v (np.ndarray, optional): Velocity trajectory to check,
            shape (N, robot_nv). If None, velocity checks are skipped.
        tau (np.ndarray, optional): Effort trajectory to check,
            shape (N, robot_nv). If None, effort checks are skipped.
        soft_lim (float, optional): Additional safety margin factor
            (0-1). Adds this fraction of joint range as safety buffer.
            Defaults to 0.

    Returns:
        bool: True if any constraint is violated, False if all
            constraints are satisfied

    Examples:
        Basic position check:
            ```python
            t, q, dq, ddq = spline.get_full_config(...)
            is_violated = spline.check_cfg_constraints(q)
            if is_violated:
                print("Trajectory violates position limits!")
            ```

        Full constraint check with safety margin:
            ```python
            is_violated = spline.check_cfg_constraints(
                q, v=dq, tau=torques, soft_lim=0.1
            )
            ```

    Note:
        Constraint violations are logged at DEBUG level with specific
        joint indices and violation types; callers report the outcome.
    """
    __isViolated = False
    for i in range(q.shape[0]):
        for j in self.act_idxq:
            delta_lim = soft_lim * abs(
                self.rmodel.upperPositionLimit[j]
                - self.rmodel.lowerPositionLimit[j]
            )
            if q[i, j] > self.rmodel.upperPositionLimit[j] - delta_lim:
                logger.debug("Joint q %d upper limit violated!", j)
                __isViolated_pos = True

            elif q[i, j] < self.rmodel.lowerPositionLimit[j] + delta_lim:
                logger.debug("Joint position idx_q %d lower limit violated!", j)
                __isViolated_pos = True
            else:
                __isViolated_pos = False
            __isViolated = __isViolated or __isViolated_pos
            # print(__isViolated)
    if v is not None:
        for i in range(v.shape[0]):
            for j in self.act_idxv:
                if abs(v[i, j]) > (1 - soft_lim) * abs(
                    self.rmodel.velocityLimit[j]
                ):
                    logger.debug("Joint vel idx_v %d limits violated!", j)
                    __isViolated_vel = True
                else:
                    __isViolated_vel = False
                __isViolated = __isViolated or __isViolated_vel
            # print(__isViolated)
    if tau is not None:
        for i in range(tau.shape[0]):
            for j in self.act_idxv:
                if abs(tau[i, j]) > (1 - soft_lim) * abs(
                    self.rmodel.effortLimit[j]
                ):
                    logger.debug("Joint effort idx_v %d limits violated!", j)
                    __isViolated_eff = True
                else:
                    __isViolated_eff = False
                __isViolated = __isViolated or __isViolated_eff
                # print(__isViolated)
    if not __isViolated:
        logger.debug(
            "SUCCEEDED to generate waypoints for a feasible initial cubic spline"
        )
    else:
        logger.debug("FAILED to generate a feasible cubic spline")
    return __isViolated
````

## `WaypointsGeneration(robot, num_waypoints, active_joints, soft_lim=0)`

Bases: `CubicSpline`

Automated waypoint generation for cubic spline trajectories.

This class extends CubicSpline to provide automated generation of feasible waypoints that respect robot joint constraints. It creates pools of valid configurations and uses random sampling to generate diverse trajectory waypoints.

The class generates waypoints for position, velocity, and acceleration that can be used as initial guesses for trajectory optimization or as standalone feasible trajectories.

Attributes:

| Name                      | Type      | Description                               |
| ------------------------- | --------- | ----------------------------------------- |
| `n_set`                   | `int`     | Size of waypoint pools (default: 10)      |
| `pool_q`                  | `ndarray` | Pool of valid position configurations     |
| `pool_dq`                 | `ndarray` | Pool of valid velocity configurations     |
| `pool_ddq`                | `ndarray` | Pool of valid acceleration configurations |
| `soft_limit_pool_default` | `ndarray` | Default soft limit values                 |

Examples:

Basic waypoint generation:

```
wp_gen = WaypointsGeneration(robot, num_waypoints=5,
                           active_joints=['joint1', 'joint2'])

# Generate random feasible waypoints
pos_wp, vel_wp, acc_wp = wp_gen.random_feasible_waypoints()

# Use with cubic spline
time_pts = np.linspace(0, 4, 5).reshape(-1, 1)
t, q, dq, ddq = wp_gen.get_full_config(
    freq=100, time_points=time_pts, waypoints=pos_wp.T,
    vel_waypoints=vel_wp.T, acc_waypoints=acc_wp.T
)
```

With custom soft limits:

```
# Different safety margins for position/velocity/acceleration
soft_lim_custom = np.array([
    [0.1, 0.15],  # Position limits (10%, 15% for joints)
    [0.2, 0.2],   # Velocity limits (20% for both joints)
    [0.3, 0.25]   # Acceleration limits (30%, 25%)
])

wp_gen.set_soft_limit_pool(soft_lim_custom)
pos_wp, vel_wp, acc_wp = wp_gen.random_feasible_waypoints()
```

Note

The class automatically ensures waypoints don't violate joint constraints and avoids repeated waypoint values that could cause numerical issues in spline generation.

Initialize waypoint generation for cubic spline trajectories.

Sets up pools for generating random feasible waypoints that respect robot joint constraints. Initializes configuration pools for positions, velocities, and accelerations.

Parameters:

| Name            | Type    | Description                                           | Default    |
| --------------- | ------- | ----------------------------------------------------- | ---------- |
| `robot`         |         | Robot model instance containing kinematic information | *required* |
| `num_waypoints` | `int`   | Number of waypoints for trajectory generation         | *required* |
| `active_joints` | `list`  | List of joint names to include in trajectory          | *required* |
| `soft_lim`      | `float` | Soft limit reduction factor (0-1). Defaults to 0.     | `0`        |

Note

The soft_lim parameter affects the base CubicSpline initialization but does not set the waypoint generation pools. Use set_soft_limit_pool() for custom pool limits.

Source code in `src/figaroh/utils/cubic_spline.py`

```
def __init__(self, robot, num_waypoints: int, active_joints: list, soft_lim=0):
    """
    Initialize waypoint generation for cubic spline trajectories.

    Sets up pools for generating random feasible waypoints that respect
    robot joint constraints. Initializes configuration pools for
    positions, velocities, and accelerations.

    Args:
        robot: Robot model instance containing kinematic information
        num_waypoints (int): Number of waypoints for trajectory generation
        active_joints (list): List of joint names to include in trajectory
        soft_lim (float, optional): Soft limit reduction factor (0-1).
            Defaults to 0.

    Note:
        The soft_lim parameter affects the base CubicSpline initialization
        but does not set the waypoint generation pools. Use
        set_soft_limit_pool() for custom pool limits.
    """
    super().__init__(robot, num_waypoints, active_joints, soft_lim=0)

    self.n_set = 10  # size of waypoints pool
    self.pool_q = np.zeros((self.n_set, len(self.act_idxq)))
    self.pool_dq = np.zeros((self.n_set, len(self.act_idxv)))
    self.pool_ddq = np.zeros((self.n_set, len(self.act_idxv)))
    self.soft_limit_pool_default = np.zeros((3, len(self.act_idxq)))
```

### `gen_rand_pool(soft_limit_pool=None)`

Generate a uniformly distributed waypoint pool of pos/vel/acc over a specific range

Source code in `src/figaroh/utils/cubic_spline.py`

```
def gen_rand_pool(self, soft_limit_pool=None):
    """Generate a uniformly distributed waypoint pool of pos/vel/acc over
    a specific range
    """
    if soft_limit_pool is None:
        soft_limit_pool = self.soft_limit_pool_default
    assert np.array(soft_limit_pool).shape == (
        3,
        len(self.act_idxq),
    ), "input a vector of soft limit pool with a shape of (3, len(activejoints)"
    lim_q = soft_limit_pool[0, :]
    lim_dq = soft_limit_pool[1, :]
    lim_ddq = soft_limit_pool[2, :]

    new_upper_q = np.zeros_like(self.upper_q)
    new_lower_q = np.zeros_like(self.lower_q)
    new_upper_dq = np.zeros_like(self.upper_dq)
    new_lower_dq = np.zeros_like(self.lower_dq)
    new_upper_ddq = np.zeros_like(self.upper_dq)
    new_lower_ddq = np.zeros_like(self.lower_dq)

    for i in range(len(self.act_idxq)):
        new_upper_q[i] = self.upper_q[i] - lim_q[i] * abs(
            self.upper_q[i] - self.lower_q[i]
        )
        new_lower_q[i] = self.lower_q[i] + lim_q[i] * abs(
            self.upper_q[i] - self.lower_q[i]
        )

        step_q = (new_upper_q[i] - new_lower_q[i]) / (self.n_set - 1)
        self.pool_q[:, i] = np.array(
            [new_lower_q[i] + j * step_q for j in range(self.n_set)]
        )

    for i in range(len(self.act_idxv)):
        new_upper_dq[i] = self.upper_dq[i] - lim_dq[i] * abs(
            self.upper_dq[i] - self.lower_dq[i]
        )
        new_lower_dq[i] = self.lower_dq[i] + lim_dq[i] * abs(
            self.upper_dq[i] - self.lower_dq[i]
        )

        new_upper_ddq[i] = k * (
            self.upper_dq[i] - lim_ddq[i] * abs(self.upper_dq[i] - self.lower_dq[i])
        )  # Fixed line break
        new_lower_ddq[i] = k * (
            self.lower_dq[i] + lim_ddq[i] * abs(self.upper_dq[i] - self.lower_dq[i])
        )  # Fixed line break

        step_dq = (new_upper_dq[i] - new_lower_dq[i]) / (self.n_set - 1)
        self.pool_dq[:, i] = np.array(
            [new_lower_dq[i] + j * step_dq for j in range(self.n_set)]
        )

        step_ddq = (new_upper_ddq[i] - new_lower_ddq[i]) / (self.n_set - 1)
        self.pool_ddq[:, i] = np.array(
            [new_lower_ddq[i] + j * step_ddq for j in range(self.n_set)]
        )
```

### `gen_rand_wp(wp_init=None, vel_wp_init=None, acc_wp_init=None, vel_set_zero=True, acc_set_zero=True)`

Generate waypoint pos/vel/acc which randomly pick from waypoint pool Or, set vel and/or acc at waypoints to be zero

Source code in `src/figaroh/utils/cubic_spline.py`

```
def gen_rand_wp(
    self,
    wp_init=None,
    vel_wp_init=None,
    acc_wp_init=None,
    vel_set_zero=True,
    acc_set_zero=True,
):
    """Generate waypoint pos/vel/acc which randomly pick from waypoint
    pool
    Or, set vel and/or acc at waypoints to be zero
    """
    wps_rand = np.zeros((self.num_waypoints, len(self.act_idxq)))
    vel_wps_rand = np.zeros((self.num_waypoints, len(self.act_idxv)))
    acc_wps_rand = np.zeros((self.num_waypoints, len(self.act_idxv)))
    if wp_init is not None:
        wps_rand[0, :] = wp_init
        for i in range(len(self.act_idxq)):
            repeat_ = True
            while repeat_:
                wps_rand[range(1, self.num_waypoints), i] = np.random.choice(
                    self.pool_q[:, i], self.num_waypoints - 1
                )
                repeat_ = self.check_repeat_wp(
                    list(wps_rand[range(1, self.num_waypoints), i])
                )
    else:
        for i in range(len(self.act_idxq)):
            repeat_ = True
            while repeat_:
                wps_rand[:, i] = np.random.choice(
                    self.pool_q[:, i], self.num_waypoints
                )
                repeat_ = self.check_repeat_wp(list(wps_rand[:, i]))
    if vel_wp_init is not None:
        vel_wps_rand[0, :] = vel_wp_init
        if not vel_set_zero:
            for i in range(len(self.act_idxv)):
                repeat_ = True
                while repeat_:
                    vel_wps_rand[range(1, self.num_waypoints), i] = (
                        np.random.choice(self.pool_dq[:, i], self.num_waypoints - 1)
                    )
                    repeat_ = self.check_repeat_wp(
                        list(vel_wps_rand[range(1, self.num_waypoints), i])
                    )
    else:
        if not vel_set_zero:
            for i in range(len(self.act_idxv)):
                repeat_ = True
                while repeat_:
                    vel_wps_rand[:, i] = np.random.choice(
                        self.pool_dq[:, i], self.num_waypoints
                    )
                    repeat_ = self.check_repeat_wp(list(vel_wps_rand[:, i]))
    if vel_wp_init is not None:
        acc_wps_rand[0, :] = vel_wp_init
        if not acc_set_zero:
            for i in range(len(self.act_idxv)):
                repeat_ = True
                while repeat_:
                    acc_wps_rand[range(1, self.num_waypoints), i] = (
                        np.random.choice(
                            self.pool_ddq[:, i], self.num_waypoints - 1
                        )
                    )
                    repeat_ = self.check_repeat_wp(
                        list(acc_wps_rand[range(1, self.num_waypoints), i])
                    )
    else:
        if not acc_set_zero:
            for i in range(len(self.act_idxv)):
                repeat_ = True
                while repeat_:
                    acc_wps_rand[:, i] = np.random.choice(
                        self.pool_ddq[:, i], self.num_waypoints
                    )
                    repeat_ = self.check_repeat_wp(list(acc_wps_rand[:, i]))
    return wps_rand.transpose(), vel_wps_rand.transpose(), acc_wps_rand.transpose()
```

### `gen_equal_wp(wp_init=None, vel_wp_init=None, acc_wp_init=None)`

Generate equal waypoints everywhere same as first waypoints with default: zero vel and zero acc

Source code in `src/figaroh/utils/cubic_spline.py`

```
def gen_equal_wp(self, wp_init=None, vel_wp_init=None, acc_wp_init=None):
    """Generate equal waypoints everywhere same as first waypoints
    with default: zero vel and zero acc
    """
    wps_equal = np.zeros((self.num_waypoints, len(self.act_idxq)))
    vel_wps_equal = np.zeros((self.num_waypoints, len(self.act_idxv)))
    acc_wps_equal = np.zeros((self.num_waypoints, len(self.act_idxv)))

    step_q = (self.upper_q - self.lower_q) / 20
    if wp_init is not None:
        wps_equal = np.tile(wp_init, (self.num_waypoints, 1))
        for jj in range(1, self.num_waypoints):
            wps_equal[jj, :] = wps_equal[jj - 1, :] + step_q

    if vel_wp_init is not None:
        vel_wps_equal = np.tile(vel_wp_init, (self.num_waypoints, 1))

    if acc_wp_init is not None:
        acc_wps_equal = np.tile(acc_wp_init, (self.num_waypoints, 1))
    return (
        wps_equal.transpose(),
        vel_wps_equal.transpose(),
        acc_wps_equal.transpose(),
    )
```

## Results Manager

Shared fallback-plotting helper and the joint-major torque-plotting fix used by both calibration and identification (`plot_with_fallback`, `plot_calibration_results`, `plot_identification_results`).

Unified results management for FIGAROH examples.

This module provides standardized plotting and saving functionality across all task types: Calibration, Identification, OptimalCalibration, OptimalTrajectory.

## `ResultsManager(task_type, robot_name='robot', results_data=None)`

Unified results management for all FIGAROH task types.

This class provides standardized plotting and saving functionality with consistent styling and formats across different analysis types.

Initialize results manager.

Parameters:

| Name           | Type                       | Description                                                                                 | Default    |
| -------------- | -------------------------- | ------------------------------------------------------------------------------------------- | ---------- |
| `task_type`    | `str`                      | Type of task ('calibration', 'identification', 'optimal_calibration', 'optimal_trajectory') | *required* |
| `robot_name`   | `str`                      | Name of the robot for file naming                                                           | `'robot'`  |
| `results_data` | `Optional[Dict[str, Any]]` | Optional dictionary of results data                                                         | `None`     |

Source code in `src/figaroh/utils/results_manager.py`

```
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
```

### `plot_calibration_results(measured_poses=None, estimated_poses=None, residuals=None, parameter_values=None, parameter_names=None, outlier_indices=None, title='Calibration Results')`

Plot calibration results with pose comparison and residual analysis.

Parameters:

| Name               | Type                  | Description                                                 | Default                 |
| ------------------ | --------------------- | ----------------------------------------------------------- | ----------------------- |
| `measured_poses`   | `Optional[ndarray]`   | Measured poses (optional, uses self.result)                 | `None`                  |
| `estimated_poses`  | `Optional[ndarray]`   | Estimated poses (optional, uses self.result)                | `None`                  |
| `residuals`        | `Optional[ndarray]`   | Position/orientation residuals (optional, uses self.result) | `None`                  |
| `parameter_values` | `Optional[ndarray]`   | Calibrated parameters (optional, uses self.result)          | `None`                  |
| `parameter_names`  | `Optional[List[str]]` | Parameter names (optional, uses self.result)                | `None`                  |
| `outlier_indices`  | `Optional[List[int]]` | Outlier indices (optional, uses self.result)                | `None`                  |
| `title`            | `str`                 | Plot title                                                  | `'Calibration Results'` |

Source code in `src/figaroh/utils/results_manager.py`

```
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
```

### `plot_identification_results(tau_measured=None, tau_identified=None, parameter_values=None, parameter_names=None, time_vector=None, joint_names=None, n_joints=None, title='Identification Results')`

Plot identification results with torque comparison and analysis.

Parameters:

| Name               | Type                  | Description                                                                                                                                                                                                       | Default                    |
| ------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `tau_measured`     | `Optional[ndarray]`   | Measured joint torques (optional, uses self.result)                                                                                                                                                               | `None`                     |
| `tau_identified`   | `Optional[ndarray]`   | Estimated torques (optional, uses self.result)                                                                                                                                                                    | `None`                     |
| `parameter_values` | `Optional[ndarray]`   | Parameter values (optional, uses self.result)                                                                                                                                                                     | `None`                     |
| `parameter_names`  | `Optional[List[str]]` | Parameter names (optional, uses self.result)                                                                                                                                                                      | `None`                     |
| `time_vector`      | `Optional[ndarray]`   | Time vector for plots                                                                                                                                                                                             | `None`                     |
| `joint_names`      | `Optional[List[str]]` | Names of robot joints                                                                                                                                                                                             | `None`                     |
| `n_joints`         | `Optional[int]`       | Number of active joints. Required when a torque array is 1D, since a flattened array is joint-major (all samples of joint 0, then joint 1, ...) and cannot be reshaped correctly without knowing the joint count. | `None`                     |
| `title`            | `str`                 | Plot title                                                                                                                                                                                                        | `'Identification Results'` |

Source code in `src/figaroh/utils/results_manager.py`

```
def plot_identification_results(
    self,
    tau_measured: Optional[np.ndarray] = None,
    tau_identified: Optional[np.ndarray] = None,
    parameter_values: Optional[np.ndarray] = None,
    parameter_names: Optional[List[str]] = None,
    time_vector: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None,
    n_joints: Optional[int] = None,
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
        n_joints: Number of active joints. Required when a torque array is
            1D, since a flattened array is joint-major (all samples of
            joint 0, then joint 1, ...) and cannot be reshaped correctly
            without knowing the joint count.
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
        if joint_names is None:
            joint_names = self.result.get("active joints", None)
        if n_joints is None:
            n_joints = self.result.get("n_active_joints", None)

        tau_measured = np.asarray(tau_measured)
        tau_identified = np.asarray(tau_identified)

        # Validate data availability
        if tau_measured.size == 0 or tau_identified.size == 0:
            self.logger.error("No torque data available for plotting")
            return

        fig = plt.figure(figsize=self.PLOT_STYLES["identification"]["figsize"])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # A 1D array is joint-major (all samples of joint 0, then joint
        # 1, ...), not sample-major — reshape(-1, 1) would silently
        # treat the whole concatenated array as a single joint. Require
        # n_joints to reshape correctly; without it, fall back to a
        # single-column plot rather than guessing.
        if tau_measured.ndim == 1:
            if n_joints:
                tau_measured = tau_measured.reshape(n_joints, -1).T
            else:
                tau_measured = tau_measured.reshape(-1, 1)
        if tau_identified.ndim == 1:
            if n_joints:
                tau_identified = tau_identified.reshape(n_joints, -1).T
            else:
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
        logger.debug("Debug info:", exc_info=True)
```

### `plot_optimal_calibration_results(configurations, weights, condition_numbers=None, information_matrix=None, title='Optimal Calibration Results')`

Plot optimal calibration configuration results.

Parameters:

| Name                 | Type                 | Description                            | Default                         |
| -------------------- | -------------------- | -------------------------------------- | ------------------------------- |
| `configurations`     | `Dict[str, ndarray]` | Dictionary of configuration data       | *required*                      |
| `weights`            | `ndarray`            | Weights assigned to each configuration | *required*                      |
| `condition_numbers`  | `Optional[ndarray]`  | Condition numbers for analysis         | `None`                          |
| `information_matrix` | `Optional[ndarray]`  | Fisher information matrix              | `None`                          |
| `title`              | `str`                | Plot title                             | `'Optimal Calibration Results'` |

Source code in `src/figaroh/utils/results_manager.py`

```
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
```

### `plot_optimal_trajectory_results(trajectories, condition_number, joint_names=None, title='Optimal Trajectory Results', q_indices=None, v_indices=None)`

Plot optimal trajectory generation results.

Positions span the model's `nq` coordinates and velocities and accelerations its `nv` coordinates, which differ for continuous, spherical or free-flyer joints. Pass `q_indices` and `v_indices` (e.g. each plotted joint's `idx_q` and `idx_v`) to plot those joints; without them every column is plotted, which requires `nq == nv`.

Parameters:

| Name               | Type                  | Description                                        | Default                        |
| ------------------ | --------------------- | -------------------------------------------------- | ------------------------------ |
| `trajectories`     | `Dict[str, Any]`      | Dictionary containing trajectory data              | *required*                     |
| `condition_number` | `float`               | Final condition number achieved                    | *required*                     |
| `joint_names`      | `Optional[List[str]]` | Names of the plotted joints, used as row labels    | `None`                         |
| `title`            | `str`                 | Plot title                                         | `'Optimal Trajectory Results'` |
| `q_indices`        | `Optional[List[int]]` | Position column of each plotted joint              | `None`                         |
| `v_indices`        | `Optional[List[int]]` | Velocity/acceleration column of each plotted joint | `None`                         |

Source code in `src/figaroh/utils/results_manager.py`

```
def plot_optimal_trajectory_results(
    self,
    trajectories: Dict[str, Any],
    condition_number: float,
    joint_names: Optional[List[str]] = None,
    title: str = "Optimal Trajectory Results",
    q_indices: Optional[List[int]] = None,
    v_indices: Optional[List[int]] = None,
) -> None:
    """
    Plot optimal trajectory generation results.

    Positions span the model's ``nq`` coordinates and velocities and
    accelerations its ``nv`` coordinates, which differ for continuous,
    spherical or free-flyer joints. Pass ``q_indices`` and ``v_indices``
    (e.g. each plotted joint's ``idx_q`` and ``idx_v``) to plot those
    joints; without them every column is plotted, which requires
    ``nq == nv``.

    Args:
        trajectories: Dictionary containing trajectory data
        condition_number: Final condition number achieved
        joint_names: Names of the plotted joints, used as row labels
        title: Plot title
        q_indices: Position column of each plotted joint
        v_indices: Velocity/acceleration column of each plotted joint
    """
    if not HAS_MATPLOTLIB:
        self.logger.warning("Cannot plot: matplotlib not available")
        return
    if (q_indices is None) != (v_indices is None) or (
        q_indices is not None and len(q_indices) != len(v_indices)
    ):
        raise ValueError(
            "q_indices and v_indices must be given together, one per joint"
        )

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

        if q_indices is None:
            nq, nv = positions[0].shape[1], velocities[0].shape[1]
            if nq != nv:
                self.logger.error(
                    "Cannot plot trajectory: positions have %d columns and "
                    "velocities %d (nq != nv); pass q_indices and v_indices",
                    nq,
                    nv,
                )
                return
            q_indices = v_indices = list(range(nq))
        n_joints = len(q_indices)
        if joint_names is None or len(joint_names) != n_joints:
            joint_names = [f"Joint {i+1}" for i in range(n_joints)]

        fig = plt.figure(figsize=self.PLOT_STYLES["optimal_trajectory"]["figsize"])
        gs = GridSpec(n_joints, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Plot each joint's trajectory
        colors = plt.cm.tab10(np.linspace(0, 1, len(times)))

        for joint_idx, (iq, iv, name) in enumerate(
            zip(q_indices, v_indices, joint_names)
        ):
            # Position
            ax_pos = fig.add_subplot(gs[joint_idx, 0])
            for seg_idx, (t, pos) in enumerate(zip(times, positions)):
                ax_pos.plot(
                    t,
                    pos[:, iq],
                    color=colors[seg_idx],
                    label=f"Segment {seg_idx+1}" if joint_idx == 0 else "",
                )
            ax_pos.set_ylabel(f"{name}\nPosition (rad)")
            ax_pos.grid(True, alpha=0.3)
            if joint_idx == 0:
                ax_pos.legend()
                ax_pos.set_title("Joint Positions")

            # Velocity
            ax_vel = fig.add_subplot(gs[joint_idx, 1])
            for seg_idx, (t, vel) in enumerate(zip(times, velocities)):
                ax_vel.plot(t, vel[:, iv], color=colors[seg_idx])
            ax_vel.set_ylabel(f"{name}\nVelocity (rad/s)")
            ax_vel.grid(True, alpha=0.3)
            if joint_idx == 0:
                ax_vel.set_title("Joint Velocities")

            # Acceleration
            ax_acc = fig.add_subplot(gs[joint_idx, 2])
            for seg_idx, (t, acc) in enumerate(zip(times, accelerations)):
                ax_acc.plot(t, acc[:, iv], color=colors[seg_idx])
            ax_acc.set_ylabel(f"{name}\nAcceleration (rad/s²)")
            ax_acc.grid(True, alpha=0.3)
            if joint_idx == 0:
                ax_acc.set_title("Joint Accelerations")

        # Label the bottom row's existing axes (a new add_subplot on the
        # same cell would stack an empty axes on top of them).
        for ax in (ax_pos, ax_vel, ax_acc):
            ax.set_xlabel("Time (s)")

        fig.suptitle(
            f"{self.robot_name.upper()} {title}\nCondition Number: {condition_number:.2e}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.show()

    except Exception as e:
        self.logger.error(f"Error plotting trajectory results: {e}")
```

### `save_results(results=None, output_dir='results', file_prefix=None, save_formats=['yaml', 'csv'])`

Save results to files with standardized format.

Parameters:

| Name           | Type                       | Description                                                      | Default           |
| -------------- | -------------------------- | ---------------------------------------------------------------- | ----------------- |
| `results`      | `Optional[Dict[str, Any]]` | Dictionary containing results to save (uses self.result if None) | `None`            |
| `output_dir`   | `str`                      | Output directory path                                            | `'results'`       |
| `file_prefix`  | `Optional[str]`            | Prefix for output files (default: robot_name_task_type)          | `None`            |
| `save_formats` | `List[str]`                | List of formats to save ('yaml', 'csv', 'json', 'npz')           | `['yaml', 'csv']` |

Returns:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `Dict[str, str]` | Dictionary mapping format to file path |

Source code in `src/figaroh/utils/results_manager.py`

```
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
```

## `plot_with_fallback(primary, fallback, logger, context)`

Run `primary()`; on any exception, log it and run `fallback()`.

Shared by every `Base*.plot_results()` method, each of which previously duplicated this same try/except pattern around a call into :class:`ResultsManager` with a raw-matplotlib fallback.

Source code in `src/figaroh/utils/results_manager.py`

```
def plot_with_fallback(
    primary: Callable[[], None],
    fallback: Callable[[], None],
    logger: logging.Logger,
    context: str,
) -> None:
    """Run ``primary()``; on any exception, log it and run ``fallback()``.

    Shared by every ``Base*.plot_results()`` method, each of which
    previously duplicated this same try/except pattern around a call into
    :class:`ResultsManager` with a raw-matplotlib fallback.
    """
    try:
        primary()
    except Exception as e:
        logger.error(f"Error plotting with ResultsManager ({context}): {e}")
        logger.info("Falling back to basic plotting...")
        fallback()
```

## `plot_calibration_results(measured_poses, estimated_poses, residuals, **kwargs)`

Plot calibration results (convenience function).

Source code in `src/figaroh/utils/results_manager.py`

```
def plot_calibration_results(measured_poses, estimated_poses, residuals, **kwargs):
    """Plot calibration results (convenience function)."""
    warnings.warn(
        "plot_calibration_results is deprecated in 1.0, will be removed in 2.0; use ResultsManager.plot_calibration_results",
        DeprecationWarning,
        stacklevel=2,
    )
    manager = ResultsManager("calibration", kwargs.get("robot_name", "robot"))
    manager.plot_calibration_results(
        measured_poses, estimated_poses, residuals, **kwargs
    )
```

## `plot_identification_results(tau_measured, tau_identified, parameters, **kwargs)`

Plot identification results (convenience function).

Source code in `src/figaroh/utils/results_manager.py`

```
def plot_identification_results(tau_measured, tau_identified, parameters, **kwargs):
    """Plot identification results (convenience function)."""
    warnings.warn(
        "plot_identification_results is deprecated in 1.0, will be removed in 2.0; use ResultsManager.plot_identification_results",
        DeprecationWarning,
        stacklevel=2,
    )
    manager = ResultsManager("identification", kwargs.get("robot_name", "robot"))
    manager.plot_identification_results(
        tau_measured, tau_identified, parameters, **kwargs
    )
```

## `save_results(results, task_type, output_dir='results', **kwargs)`

Save results (convenience function).

Source code in `src/figaroh/utils/results_manager.py`

```
def save_results(results, task_type, output_dir="results", **kwargs):
    """Save results (convenience function)."""
    warnings.warn(
        "save_results is deprecated in 1.0, will be removed in 2.0; use ResultsManager.save_results",
        DeprecationWarning,
        stacklevel=2,
    )
    manager = ResultsManager(task_type, kwargs.get("robot_name", "robot"))
    return manager.save_results(results, output_dir, **kwargs)
```

## Error Handling

Custom exceptions and error handling for FIGAROH examples.

This module provides standardized error handling and validation utilities for robot identification and calibration workflows.

## `FigarohExampleError`

Bases: `Exception`

Base exception for FIGAROH examples.

## `RobotInitializationError`

Bases: `FigarohExampleError`

Exception raised when robot initialization fails.

## `ConfigurationError`

Bases: `FigarohExampleError`

Exception raised for configuration-related issues.

## `DataProcessingError`

Bases: `FigarohExampleError`

Exception raised during data processing operations.

## `CalibrationError`

Bases: `FigarohExampleError`

Exception raised during calibration procedures.

## `IdentificationError`

Bases: `FigarohExampleError`

Exception raised during identification procedures.

## `ValidationError`

Bases: `FigarohExampleError`

Exception raised when validation fails.

## `ErrorContext(operation_name, raise_on_error=True)`

Context manager for structured error handling.

Source code in `src/figaroh/utils/error_handling.py`

```
def __init__(self, operation_name: str, raise_on_error: bool = True):
    self.operation_name = operation_name
    self.raise_on_error = raise_on_error
    self.error = None
```

## `validate_robot_config(config)`

Validate robot configuration dictionary.

Parameters:

| Name     | Type             | Description                          | Default    |
| -------- | ---------------- | ------------------------------------ | ---------- |
| `config` | `Dict[str, Any]` | Configuration dictionary to validate | *required* |

Raises:

| Type              | Description                 |
| ----------------- | --------------------------- |
| `ValidationError` | If configuration is invalid |

Source code in `src/figaroh/utils/error_handling.py`

```
def validate_robot_config(config: Dict[str, Any]) -> None:
    """
    Validate robot configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")

    # Check for required fields
    required_fields = ["robot_name"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
```

## `validate_trajectory_data(q, qd=None, qdd=None, tau=None)`

Validate trajectory data arrays.

Parameters:

| Name  | Type                | Description                    | Default    |
| ----- | ------------------- | ------------------------------ | ---------- |
| `q`   | `ndarray`           | Joint positions                | *required* |
| `qd`  | `Optional[ndarray]` | Joint velocities (optional)    | `None`     |
| `qdd` | `Optional[ndarray]` | Joint accelerations (optional) | `None`     |
| `tau` | `Optional[ndarray]` | Joint torques (optional)       | `None`     |

Raises:

| Type              | Description        |
| ----------------- | ------------------ |
| `ValidationError` | If data is invalid |

Source code in `src/figaroh/utils/error_handling.py`

```
def validate_trajectory_data(
    q: np.ndarray,
    qd: Optional[np.ndarray] = None,
    qdd: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
) -> None:
    """
    Validate trajectory data arrays.

    Args:
        q: Joint positions
        qd: Joint velocities (optional)
        qdd: Joint accelerations (optional)
        tau: Joint torques (optional)

    Raises:
        ValidationError: If data is invalid
    """
    if not isinstance(q, np.ndarray):
        raise ValidationError("Joint positions must be numpy array")

    if q.ndim != 2:
        raise ValidationError("Joint positions must be 2D array (n_samples, n_joints)")

    n_samples, n_joints = q.shape

    # Validate other arrays if provided
    arrays_to_check = [("velocities", qd), ("accelerations", qdd), ("torques", tau)]

    for name, array in arrays_to_check:
        if array is not None:
            if not isinstance(array, np.ndarray):
                raise ValidationError(f"{name} must be numpy array")

            if array.shape != (n_samples, n_joints):
                raise ValidationError(
                    f"{name} shape {array.shape} doesn't match "
                    f"positions shape {(n_samples, n_joints)}"
                )

            if np.any(np.isnan(array)):
                raise ValidationError(f"{name} contains NaN values")

            if np.any(np.isinf(array)):
                raise ValidationError(f"{name} contains infinite values")
```

## `validate_numeric_range(value, min_val=None, max_val=None, name='value')`

Validate that numeric value(s) are within specified range.

Parameters:

| Name      | Type                         | Description                          | Default    |
| --------- | ---------------------------- | ------------------------------------ | ---------- |
| `value`   | `Union[float, int, ndarray]` | Value or array to validate           | *required* |
| `min_val` | `Optional[float]`            | Minimum allowed value                | `None`     |
| `max_val` | `Optional[float]`            | Maximum allowed value                | `None`     |
| `name`    | `str`                        | Name of the value for error messages | `'value'`  |

Raises:

| Type              | Description              |
| ----------------- | ------------------------ |
| `ValidationError` | If value is out of range |

Source code in `src/figaroh/utils/error_handling.py`

```
def validate_numeric_range(
    value: Union[float, int, np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
) -> None:
    """
    Validate that numeric value(s) are within specified range.

    Args:
        value: Value or array to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is out of range
    """
    if isinstance(value, np.ndarray):
        if min_val is not None and np.any(value < min_val):
            raise ValidationError(f"{name} contains values below {min_val}")

        if max_val is not None and np.any(value > max_val):
            raise ValidationError(f"{name} contains values above {max_val}")
    else:
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} {value} is below minimum {min_val}")

        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} {value} is above maximum {max_val}")
```

## `validate_robot_initialization(func)`

Decorator to validate robot initialization.

Parameters:

| Name   | Type | Description          | Default    |
| ------ | ---- | -------------------- | ---------- |
| `func` | `F`  | Function to decorate | *required* |

Returns:

| Type | Description                              |
| ---- | ---------------------------------------- |
| `F`  | Decorated function with robot validation |

Source code in `src/figaroh/utils/error_handling.py`

```
def validate_robot_initialization(func: F) -> F:
    """
    Decorator to validate robot initialization.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with robot validation
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise RobotInitializationError(f"Robot initialization failed: {e}") from e

    return wrapper
```

## `validate_input_data(func)`

Decorator to validate input data for processing functions.

Parameters:

| Name   | Type | Description          | Default    |
| ------ | ---- | -------------------- | ---------- |
| `func` | `F`  | Function to decorate | *required* |

Returns:

| Type | Description                              |
| ---- | ---------------------------------------- |
| `F`  | Decorated function with input validation |

Source code in `src/figaroh/utils/error_handling.py`

```
def validate_input_data(func: F) -> F:
    """
    Decorator to validate input data for processing functions.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with input validation
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Basic validation - can be extended based on function signature
            for arg in args:
                if isinstance(arg, np.ndarray):
                    if np.any(np.isnan(arg)):
                        raise ValidationError("Input data contains NaN values")
                    if np.any(np.isinf(arg)):
                        raise ValidationError("Input data contains infinite values")

            return func(*args, **kwargs)
        except ValidationError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Data processing failed: {e}") from e

    return wrapper
```

## `handle_calibration_errors(func)`

Decorator to handle calibration-specific errors.

Parameters:

| Name   | Type | Description          | Default    |
| ------ | ---- | -------------------- | ---------- |
| `func` | `F`  | Function to decorate | *required* |

Returns:

| Type | Description                                        |
| ---- | -------------------------------------------------- |
| `F`  | Decorated function with calibration error handling |

Source code in `src/figaroh/utils/error_handling.py`

```
def handle_calibration_errors(func: F) -> F:
    """
    Decorator to handle calibration-specific errors.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with calibration error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise CalibrationError(f"Calibration failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in calibration: {e}")
            raise CalibrationError(f"Unexpected calibration error: {e}") from e

    return wrapper
```

## `handle_identification_errors(func)`

Decorator to handle identification-specific errors.

Parameters:

| Name   | Type | Description          | Default    |
| ------ | ---- | -------------------- | ---------- |
| `func` | `F`  | Function to decorate | *required* |

Returns:

| Type | Description                                           |
| ---- | ----------------------------------------------------- |
| `F`  | Decorated function with identification error handling |

Source code in `src/figaroh/utils/error_handling.py`

```
def handle_identification_errors(func: F) -> F:
    """
    Decorator to handle identification-specific errors.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with identification error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise IdentificationError(f"Identification failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in identification: {e}")
            raise IdentificationError(f"Unexpected identification error: {e}") from e

    return wrapper
```

## `safe_execute(func, *args, **kwargs)`

Safely execute a function and return (success, result_or_error).

Parameters:

| Name       | Type | Description                           | Default    |
| ---------- | ---- | ------------------------------------- | ---------- |
| `func`     | `F`  | Function to execute                   | *required* |
| `*args`    |      | Positional arguments for the function | `()`       |
| `**kwargs` |      | Keyword arguments for the function    | `{}`       |

Returns:

| Type    | Description                                  |
| ------- | -------------------------------------------- |
| `tuple` | Tuple of (success_flag, result_or_exception) |

Source code in `src/figaroh/utils/error_handling.py`

```
def safe_execute(func: F, *args, **kwargs) -> tuple:
    """
    Safely execute a function and return (success, result_or_error).

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (success_flag, result_or_exception)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Function {func.__name__} failed: {e}")
        return False, e
```

## `setup_example_logging(log_level='INFO', log_file=None)`

Setup logging for FIGAROH examples.

Parameters:

| Name        | Type            | Description                                 | Default  |
| ----------- | --------------- | ------------------------------------------- | -------- |
| `log_level` | `str`           | Logging level (DEBUG, INFO, WARNING, ERROR) | `'INFO'` |
| `log_file`  | `Optional[str]` | Optional log file path                      | `None`   |

Returns:

| Type     | Description                |
| -------- | -------------------------- |
| `Logger` | Configured logger instance |

Source code in `src/figaroh/utils/error_handling.py`

```
def setup_example_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for FIGAROH examples.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("figaroh_examples")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

## Config Migration

Converts a legacy flat (`calibration:`/`identification:`) config to the unified `extends:`/`tasks:` format, including an optional round-trip self-check against a URDF. See "Migrating from legacy to unified format" in [Config Parameters](https://thanhndv212.github.io/figaroh-plus/concepts/config_parameters/index.md).

Legacy -> unified config migration.

Converts a legacy flat `calibration:`/`identification:` config into the unified `extends:`/`tasks:` format, using the exact field mapping documented in `docs/source/concepts/config_parameters.md`. This is the reverse of :func:`figaroh.calibration.config.unified_to_legacy_config` and :func:`figaroh.identification.config.unified_to_legacy_identif_config`.

Fields with no unified-format consumer today (`is_static_regressor`, `is_inertia_regressor`, `embedded_forces`, `sync_joint_motion`, and the parsed-but-unused joint limit / `ratio_essential` fields — see config_parameters.md) are preserved under the `custom:` section rather than silently dropped.

CLI usage::

```
python -m figaroh.utils.config_migration \
    --input legacy_config.yaml --output unified_config.yaml \
    [--task auto|calibration|identification] \
    [--template base|manipulator|humanoid] \
    [--urdf robot.urdf]   # enables the round-trip self-check
```

## `legacy_calibration_to_unified(calib_data)`

Convert a legacy `calibration:` section to unified `tasks.calibration`.

Parameters:

| Name         | Type             | Description                                                                                          | Default    |
| ------------ | ---------------- | ---------------------------------------------------------------------------------------------------- | ---------- |
| `calib_data` | `Dict[str, Any]` | The dict under the calibration: key of a legacy config file (e.g. yaml.safe_load(f)["calibration"]). | *required* |

Returns:

| Type             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `Dict[str, Any]` | A {"tasks": {"calibration": {...}}} fragment, ready to merge |
| `Dict[str, Any]` | with the identification fragment and an extends: root.       |

Source code in `src/figaroh/utils/config_migration.py`

```
def legacy_calibration_to_unified(calib_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy ``calibration:`` section to unified ``tasks.calibration``.

    Args:
        calib_data: The dict under the ``calibration:`` key of a legacy
            config file (e.g. ``yaml.safe_load(f)["calibration"]``).

    Returns:
        A ``{"tasks": {"calibration": {...}}}`` fragment, ready to merge
        with the identification fragment and an ``extends:`` root.
    """
    markers = calib_data.get("markers") or []
    unified_markers = [
        {
            "reference_joint": m.get("ref_joint"),
            "measurable_dof": m.get("measure", [True, True, True, False, False, False]),
        }
        for m in markers
    ]

    coeff = calib_data.get("coeff_regularize", 0.0)
    if coeff:
        logger.warning(
            "coeff_regularize (%s) was removed in 1.0: the migrated config "
            "sets regularization_coefficient to 0. Use estimation.method: map "
            "(docs/source/concepts/calibration_estimation.md).",
            coeff,
        )
        coeff = 0.0

    calibration_task: Dict[str, Any] = {
        "enabled": True,
        "type": "kinematic_calibration",
        "parameters": {
            "calibration_level": calib_data.get("calib_level", "full_params"),
            "include_non_geometric": bool(calib_data.get("non_geom", False)),
            "regularization_coefficient": coeff,
            "outlier_threshold": calib_data.get("outlier_eps", 0.05),
        },
        "kinematics": {
            "base_frame": calib_data.get("base_frame", "universe"),
            "tool_frame": calib_data.get("tool_frame"),
            "free_flying_base": bool(calib_data.get("free_flyer", False)),
        },
        "measurements": {
            "markers": unified_markers,
            "poses": {
                "base_pose": calib_data.get("base_pose"),
                "tool_pose": calib_data.get("tip_pose"),
            },
        },
        "data": {
            "source_file": calib_data.get("data_file"),
            "sample_configurations_file": calib_data.get("sample_configs_file"),
            "number_of_samples": calib_data.get("nb_sample", 500),
        },
    }

    base_to_ref_frame = calib_data.get("base_to_ref_frame")
    ref_frame = calib_data.get("ref_frame")
    if base_to_ref_frame is not None or ref_frame is not None:
        calibration_task["eye_hand"] = {
            "enabled": True,
            "camera_frame": base_to_ref_frame,
            "reference_frame": ref_frame,
        }

    return {"tasks": {"calibration": calibration_task}}
```

## `legacy_identification_to_unified(identif_data)`

Convert a legacy `identification:` section to unified format.

Parameters:

| Name           | Type             | Description                                                     | Default    |
| -------------- | ---------------- | --------------------------------------------------------------- | ---------- |
| `identif_data` | `Dict[str, Any]` | The dict under the identification: key of a legacy config file. | *required* |

Returns:

| Type             | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `Dict[str, Any]` | A \`\`{"robot": {"properties": {...}}, "tasks": {"identification": |
| `Dict[str, Any]` | {...}}, "custom": {...}}fragment.robot.properties\`\` holds        |
| `Dict[str, Any]` | fields shared across tasks (mechanics, joints); custom holds       |
| `Dict[str, Any]` | fields with no current unified-format consumer.                    |

Source code in `src/figaroh/utils/config_migration.py`

```
def legacy_identification_to_unified(identif_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy ``identification:`` section to unified format.

    Args:
        identif_data: The dict under the ``identification:`` key of a
            legacy config file.

    Returns:
        A ``{"robot": {"properties": {...}}, "tasks": {"identification":
        {...}}, "custom": {...}}`` fragment. ``robot.properties`` holds
        fields shared across tasks (mechanics, joints); ``custom`` holds
        fields with no current unified-format consumer.
    """
    robot_params = (identif_data.get("robot_params") or [{}])[0]
    problem_params = (identif_data.get("problem_params") or [{}])[0]
    process_params = (identif_data.get("processing_params") or [{}])[0]
    tls_params = (identif_data.get("tls_params") or [{}])[0]

    robot_properties: Dict[str, Any] = {
        "joints": {
            "joint_limits": {
                "position": robot_params.get("q_lim_def"),
                "velocity": robot_params.get("dq_lim_def"),
                "acceleration": robot_params.get("ddq_lim_def"),
                "torque": robot_params.get("tau_lim_def"),
            }
        },
        "mechanics": {
            "friction_coefficients": {
                "viscous": robot_params.get("fv"),
                "static": robot_params.get("fs"),
            },
            "actuator_inertias": robot_params.get("Ia"),
            "joint_offsets": robot_params.get("offset"),
            "reduction_ratios": robot_params.get("reduction_ratio"),
            "ratio_essential": robot_params.get("ratio_essential"),
        },
        "coupling": {
            "has_coupled_wrist": bool(problem_params.get("has_coupled_wrist", False)),
            "Iam6": robot_params.get("Iam6", 0),
            "fvm6": robot_params.get("fvm6", 0),
            "fsm6": robot_params.get("fsm6", 0),
        },
    }
    active_joints = problem_params.get("active_joints")
    if active_joints:
        robot_properties["joints"]["active_joints"] = active_joints

    ts = process_params.get("ts")
    sampling_frequency = (1.0 / ts) if ts else 5000.0

    force_torque = problem_params.get("force_torque")
    force_torque_sensors: List[Any] = (
        list(force_torque)
        if isinstance(force_torque, list)
        else ([force_torque] if force_torque else [])
    )

    identification_task: Dict[str, Any] = {
        "enabled": True,
        "type": "dynamic_identification",
        "problem": {
            "include_external_forces": bool(
                problem_params.get("is_external_wrench", False)
            ),
            "use_joint_torques": bool(problem_params.get("is_joint_torques", True)),
            "force_torque_sensors": force_torque_sensors,
            "external_wrench_offsets": bool(
                problem_params.get("external_wrench_offsets", False)
            ),
            "model_components": {
                "friction": bool(problem_params.get("has_friction", True)),
                "actuator_inertia": bool(
                    problem_params.get("has_actuator_inertia", True)
                ),
                "joint_offset": bool(problem_params.get("has_joint_offset", True)),
            },
        },
        "signal_processing": {
            "sampling_frequency": sampling_frequency,
            "cutoff_frequency": process_params.get(
                "cut_off_frequency_butterworth", 100.0
            ),
        },
        "load_configuration": {
            "additional_mass": tls_params.get("mass_load", 0.0),
            "loaded_body_index": tls_params.get("which_body_loaded", 0),
        },
    }

    custom: Dict[str, Any] = {}
    unconsumed_problem = {
        k: problem_params[k] for k in _UNCONSUMED_PROBLEM_PARAMS if k in problem_params
    }
    unconsumed_tls = {
        k: tls_params[k] for k in _UNCONSUMED_TLS_PARAMS if k in tls_params
    }
    if unconsumed_problem:
        custom["legacy_problem_params"] = unconsumed_problem
    if unconsumed_tls:
        custom["legacy_tls_params"] = unconsumed_tls

    result: Dict[str, Any] = {
        "robot": {"properties": robot_properties},
        "tasks": {"identification": identification_task},
    }
    if custom:
        result["custom"] = custom
    return result
```

## `merge_unified_sections(*docs)`

Merge fragments produced by `legacy_*_to_unified` into one document.

A legacy file commonly has both `calibration:` and `identification:` sections; each converts independently and the results are merged here.

Source code in `src/figaroh/utils/config_migration.py`

```
def merge_unified_sections(*docs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge fragments produced by ``legacy_*_to_unified`` into one document.

    A legacy file commonly has both ``calibration:`` and ``identification:``
    sections; each converts independently and the results are merged here.
    """
    merged: Dict[str, Any] = {"tasks": {}}
    for doc in docs:
        for key, value in doc.items():
            if key == "tasks":
                merged["tasks"].update(value)
            elif key == "robot":
                merged.setdefault("robot", {"properties": {}})
                _deep_merge(
                    merged["robot"].setdefault("properties", {}),
                    value.get("properties", {}),
                )
            elif key == "custom":
                merged.setdefault("custom", {}).update(value)
            else:
                merged[key] = value
    return merged
```

## `convert_legacy_to_unified(legacy_config, template='base')`

Convert a full legacy config dict (both sections, if present).

Parameters:

| Name            | Type             | Description                                                                                                                  | Default    |
| --------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `legacy_config` | `Dict[str, Any]` | The parsed legacy YAML (top-level dict with calibration/identification keys).                                                | *required* |
| `template`      | `str`            | Which template to extends: — "base" (default, safest — no inherited defaults to reason about), "manipulator", or "humanoid". | `'base'`   |

Returns:

| Type             | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `Dict[str, Any]` | A complete unified-format config dict, ready to yaml.dump. |

Raises:

| Type         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `ValueError` | If neither calibration nor identification is present, or template is not a known key. |

Source code in `src/figaroh/utils/config_migration.py`

```
def convert_legacy_to_unified(
    legacy_config: Dict[str, Any], template: str = "base"
) -> Dict[str, Any]:
    """Convert a full legacy config dict (both sections, if present).

    Args:
        legacy_config: The parsed legacy YAML (top-level dict with
            ``calibration``/``identification`` keys).
        template: Which template to ``extends:`` — ``"base"`` (default,
            safest — no inherited defaults to reason about),
            ``"manipulator"``, or ``"humanoid"``.

    Returns:
        A complete unified-format config dict, ready to ``yaml.dump``.

    Raises:
        ValueError: If neither ``calibration`` nor ``identification`` is
            present, or ``template`` is not a known key.
    """
    if template not in TEMPLATE_PATHS:
        raise ValueError(
            f"Unknown template {template!r}; choose from {sorted(TEMPLATE_PATHS)}"
        )

    docs = []
    if "calibration" in legacy_config:
        docs.append(legacy_calibration_to_unified(legacy_config["calibration"]))
    if "identification" in legacy_config:
        docs.append(legacy_identification_to_unified(legacy_config["identification"]))
    if not docs:
        raise ValueError(
            "No 'calibration' or 'identification' section found in legacy config"
        )

    merged = merge_unified_sections(*docs)
    for task_name in ("calibration", "identification"):
        merged["tasks"].setdefault(task_name, {"enabled": False})

    result = {"extends": TEMPLATE_PATHS[template]}
    result.update(merged)
    return result
```

## `self_check(robot, legacy_config, unified_doc)`

Round-trip `unified_doc` back through the real unified parser and diff the recovered `calib_config`/`identif_config` against the original legacy sections.

This reuses the exact same code path production uses (`create_task_config` + `unified_to_legacy_*config`) rather than reimplementing the flattening logic, so it can't silently diverge from what the pipeline actually does.

Parameters:

| Name            | Type             | Description                                                                                                                                                                                                                                                                        | Default    |
| --------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `robot`         |                  | A robot instance with .model/.data (needed for frame validation inside unified_to_legacy_config).                                                                                                                                                                                  | *required* |
| `legacy_config` | `Dict[str, Any]` | The original parsed legacy YAML.                                                                                                                                                                                                                                                   | *required* |
| `unified_doc`   | `Dict[str, Any]` | The output of :func:convert_legacy_to_unified (with extends: already resolved away, i.e. pass the merged dict as if template inheritance had already happened — for a from-scratch check, resolve inheritance first via UnifiedConfigParser if the template adds required fields). | *required* |

Returns:

| Type             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `Dict[str, Any]` | \`\`{"ok": bool, "mismatches": {".": {"expected": |
| `Dict[str, Any]` | ..., "got": ...}}}\`\`                            |

Source code in `src/figaroh/utils/config_migration.py`

```
def self_check(
    robot, legacy_config: Dict[str, Any], unified_doc: Dict[str, Any]
) -> Dict[str, Any]:
    """Round-trip ``unified_doc`` back through the real unified parser and
    diff the recovered ``calib_config``/``identif_config`` against the
    original legacy sections.

    This reuses the exact same code path production uses
    (``create_task_config`` + ``unified_to_legacy_*config``) rather than
    reimplementing the flattening logic, so it can't silently diverge from
    what the pipeline actually does.

    Args:
        robot: A robot instance with ``.model``/``.data`` (needed for frame
            validation inside ``unified_to_legacy_config``).
        legacy_config: The original parsed legacy YAML.
        unified_doc: The output of :func:`convert_legacy_to_unified`
            (with ``extends:`` already resolved away, i.e. pass the merged
            dict as if template inheritance had already happened — for a
            from-scratch check, resolve inheritance first via
            ``UnifiedConfigParser`` if the template adds required fields).

    Returns:
        ``{"ok": bool, "mismatches": {"<section>.<field>": {"expected":
        ..., "got": ...}}}``
    """
    from figaroh.utils.config_parser import create_task_config
    from figaroh.calibration.config import unified_to_legacy_config
    from figaroh.identification.config import unified_to_legacy_identif_config

    mismatches: Dict[str, Dict[str, Any]] = {}

    if "calibration" in legacy_config:
        original = legacy_config["calibration"]
        task_config = create_task_config(robot, unified_doc, "calibration")
        recovered = unified_to_legacy_config(robot, task_config)
        checks = {
            "calib_level": ("calib_model", "full_params"),
            "non_geom": ("non_geom", False),
            "base_frame": ("start_frame", "universe"),
            "tool_frame": ("end_frame", None),
            "free_flyer": ("free_flyer", False),
            "base_to_ref_frame": ("base_to_ref_frame", None),
            "ref_frame": ("ref_frame", None),
            "coeff_regularize": ("coeff_regularize", 0.0),
            "outlier_eps": ("outlier_eps", 0.05),
            "data_file": ("data_file", None),
            "nb_sample": ("NbSample", 500),
        }
        for legacy_key, (recovered_key, default) in checks.items():
            expected = original.get(legacy_key, default)
            if legacy_key == "coeff_regularize":
                expected = 0.0  # removed in 1.0: migrated as 0
            got = recovered.get(recovered_key, default)
            if expected != got:
                mismatches[f"calibration.{legacy_key}"] = {
                    "expected": expected,
                    "got": got,
                }

    if "identification" in legacy_config:
        original = legacy_config["identification"]
        task_config = create_task_config(robot, unified_doc, "identification")
        recovered = unified_to_legacy_identif_config(robot, task_config)
        robot_params = (original.get("robot_params") or [{}])[0]
        problem_params = (original.get("problem_params") or [{}])[0]
        checks = {
            "fv": (robot_params.get("fv"), recovered.get("fv")),
            "fs": (robot_params.get("fs"), recovered.get("fs")),
            "Ia": (robot_params.get("Ia"), recovered.get("Ia")),
            "has_friction": (
                problem_params.get("has_friction", True),
                recovered.get("has_friction"),
            ),
            "has_coupled_wrist": (
                problem_params.get("has_coupled_wrist", True),
                recovered.get("has_coupled_wrist"),
            ),
        }
        for field, (expected, got) in checks.items():
            if expected != got:
                mismatches[f"identification.{field}"] = {
                    "expected": expected,
                    "got": got,
                }

    return {"ok": not mismatches, "mismatches": mismatches}
```
