# Integration

High-level, one-line API for identification workflows. See [Concepts → Integration API](https://thanhndv212.github.io/figaroh-plus/concepts/integration/index.md) for usage and when to reach for this instead of subclassing `BaseIdentification` directly.

## api

Integration API for FIGAROH identification workflows.

Provides the high-level RobotIdentificationSystem class that wraps the backend abstraction and BaseIdentification workflow into a simple, one-line API.

## `IdentificationResult(phi_base=None, params_base=None, phi_standard=None, rms_error=None, correlation=None, backend='', model_path='', config=None, raw=None)`

Container for identification results.

Attributes:

| Name           | Type                | Description                                                   |
| -------------- | ------------------- | ------------------------------------------------------------- |
| `phi_base`     | `Optional[ndarray]` | Identified base parameters                                    |
| `params_base`  | `Optional[list]`    | Base parameter names                                          |
| `phi_standard` | `Optional[ndarray]` | Full standard parameters (if reconstruction enabled)          |
| `rms_error`    | `Optional[float]`   | RMS error of the identification                               |
| `correlation`  | `Optional[float]`   | Correlation between measured and estimated torques            |
| `backend`      | `str`               | Name of the backend used                                      |
| `model_path`   | `str`               | Path to the robot model                                       |
| `config`       | `Optional[dict]`    | Configuration used                                            |
| `raw`          | `Optional[dict]`    | Raw result dict from BaseIdentification (for advanced access) |

## `RobotIdentificationSystem(robot, backend_name='pinocchio', **kwargs)`

High-level API for robot dynamic parameter identification.

Provides a simple interface that wraps the backend abstraction and BaseIdentification workflow. Supports switching between dynamics backends (Pinocchio, MuJoCo) with minimal code changes.

Example

> > > ### Create system from URDF
> > >
> > > system = RobotIdentificationSystem.from_urdf("robot.urdf")
> > >
> > > ### Or with specific backend
> > >
> > > system = RobotIdentificationSystem.from_urdf( ... "robot.urdf", backend="mujoco" ... )
> > >
> > > ### Run identification
> > >
> > > results = system.identify_parameters( ... config="config/robot_config.yaml", ... data_dir="data/", ... plotting=False ... ) print(f"Identified {len(results.phi_base)} base parameters") print(f"RMS error: {results.rms_error}")

Initialize identification system.

Parameters:

| Name           | Type | Description                                             | Default       |
| -------------- | ---- | ------------------------------------------------------- | ------------- |
| `robot`        |      | Robot model (figaroh.tools.robot.Robot or RobotWrapper) | *required*    |
| `backend_name` |      | Name of the dynamics backend ('pinocchio', 'mujoco')    | `'pinocchio'` |
| `**kwargs`     |      | Additional configuration                                | `{}`          |

Source code in `src/figaroh/integration/api.py`

```
def __init__(self, robot, backend_name="pinocchio", **kwargs):
    """Initialize identification system.

    Args:
        robot: Robot model (figaroh.tools.robot.Robot or RobotWrapper)
        backend_name: Name of the dynamics backend ('pinocchio', 'mujoco')
        **kwargs: Additional configuration
    """
    self.robot = robot
    self.backend_name = backend_name
    self._identification = None
    self._config = kwargs
```

### `backend`

Access the underlying dynamics backend.

### `nq`

Number of position variables.

### `nv`

Number of velocity variables.

### `from_urdf(urdf_path, backend='pinocchio', package_dirs=None, free_flyer=False, **kwargs)`

Create identification system from URDF file.

Parameters:

| Name           | Type | Description                          | Default       |
| -------------- | ---- | ------------------------------------ | ------------- |
| `urdf_path`    |      | Path to URDF file                    | *required*    |
| `backend`      |      | Backend name ('pinocchio', 'mujoco') | `'pinocchio'` |
| `package_dirs` |      | Package directories for mesh files   | `None`        |
| `free_flyer`   |      | Whether to add free-flyer joint      | `False`       |
| `**kwargs`     |      | Additional arguments for load_robot  | `{}`          |

Returns:

| Type | Description                        |
| ---- | ---------------------------------- |
|      | RobotIdentificationSystem instance |

Source code in `src/figaroh/integration/api.py`

```
@classmethod
def from_urdf(
    cls,
    urdf_path,
    backend="pinocchio",
    package_dirs=None,
    free_flyer=False,
    **kwargs,
):
    """Create identification system from URDF file.

    Args:
        urdf_path: Path to URDF file
        backend: Backend name ('pinocchio', 'mujoco')
        package_dirs: Package directories for mesh files
        free_flyer: Whether to add free-flyer joint
        **kwargs: Additional arguments for load_robot

    Returns:
        RobotIdentificationSystem instance
    """
    from ..tools.load_robot import load_robot

    robot = load_robot(
        urdf_path,
        package_dirs=package_dirs,
        isFext=free_flyer,
        loader="figaroh",
        **kwargs,
    )
    return cls(robot, backend_name=backend, **kwargs)
```

### `from_mjcf(mjcf_path, backend='mujoco', **kwargs)`

Create identification system from MJCF file.

.. note:: MJCF support is not yet implemented. The identification workflow requires a Robot object (URDF-based). Full MJCF support requires refactoring BaseIdentification to accept a DynamicsBackend directly — planned for a future release.

Parameters:

| Name        | Type | Description                                  | Default    |
| ----------- | ---- | -------------------------------------------- | ---------- |
| `mjcf_path` |      | Path to MJCF file                            | *required* |
| `backend`   |      | Backend name (defaults to 'mujoco' for MJCF) | `'mujoco'` |
| `**kwargs`  |      | Additional arguments                         | `{}`       |

Raises:

| Type                  | Description                            |
| --------------------- | -------------------------------------- |
| `NotImplementedError` | Always raised — MJCF not yet supported |

Source code in `src/figaroh/integration/api.py`

```
@classmethod
def from_mjcf(cls, mjcf_path, backend="mujoco", **kwargs):
    """Create identification system from MJCF file.

    .. note::
        MJCF support is not yet implemented. The identification workflow
        requires a Robot object (URDF-based). Full MJCF support requires
        refactoring BaseIdentification to accept a DynamicsBackend
        directly — planned for a future release.

    Args:
        mjcf_path: Path to MJCF file
        backend: Backend name (defaults to 'mujoco' for MJCF)
        **kwargs: Additional arguments

    Raises:
        NotImplementedError: Always raised — MJCF not yet supported
    """
    raise NotImplementedError(
        "from_mjcf is not yet supported. The identification workflow "
        "requires a Robot object (URDF-based). Use from_urdf instead. "
        "Full MJCF support requires refactoring BaseIdentification to "
        "accept a DynamicsBackend directly — planned for a future release."
    )
```

### `identify_parameters(config, data_dir=None, truncate=None, decimate=True, decimation_factor=10, zero_tolerance=0.001, plotting=False, save_results=False, **kwargs)`

Run parameter identification.

This is the main entry point. It:

1. Creates a BaseIdentification subclass instance
1. Loads configuration from YAML
1. Processes trajectory data from CSV files
1. Computes the regressor
1. Solves for base parameters
1. Returns structured results

Parameters:

| Name                | Type | Description                                             | Default    |
| ------------------- | ---- | ------------------------------------------------------- | ---------- |
| `config`            |      | Path to YAML config file                                | *required* |
| `data_dir`          |      | Directory containing trajectory data (overrides config) | `None`     |
| `truncate`          |      | Optional truncation indices for data                    | `None`     |
| `decimate`          |      | Whether to decimate the regressor                       | `True`     |
| `decimation_factor` |      | Decimation factor (default: 10)                         | `10`       |
| `zero_tolerance`    |      | Tolerance for zero column elimination (default: 0.001)  | `0.001`    |
| `plotting`          |      | Whether to generate plots (default: False)              | `False`    |
| `save_results`      |      | Whether to save results to file (default: False)        | `False`    |
| `**kwargs`          |      | Additional solver arguments                             | `{}`       |

Returns:

| Type | Description                                                 |
| ---- | ----------------------------------------------------------- |
|      | IdentificationResult with identified parameters and metrics |

Raises:

| Type                | Description                       |
| ------------------- | --------------------------------- |
| `ValueError`        | If no data source is configured   |
| `FileNotFoundError` | If config or data files not found |

Source code in `src/figaroh/integration/api.py`

```
def identify_parameters(
    self,
    config,
    data_dir=None,
    truncate=None,
    decimate=True,
    decimation_factor=10,
    zero_tolerance=0.001,
    plotting=False,
    save_results=False,
    **kwargs,
):
    """Run parameter identification.

    This is the main entry point. It:
    1. Creates a BaseIdentification subclass instance
    2. Loads configuration from YAML
    3. Processes trajectory data from CSV files
    4. Computes the regressor
    5. Solves for base parameters
    6. Returns structured results

    Args:
        config: Path to YAML config file
        data_dir: Directory containing trajectory data (overrides config)
        truncate: Optional truncation indices for data
        decimate: Whether to decimate the regressor
        decimation_factor: Decimation factor (default: 10)
        zero_tolerance: Tolerance for zero column elimination (default: 0.001)
        plotting: Whether to generate plots (default: False)
        save_results: Whether to save results to file (default: False)
        **kwargs: Additional solver arguments

    Returns:
        IdentificationResult with identified parameters and metrics

    Raises:
        ValueError: If no data source is configured
        FileNotFoundError: If config or data files not found
    """
    # Override data_dir in config if provided
    _created_temp_config = False
    if data_dir is not None:
        import tempfile
        import yaml

        # Read existing config and add/override data_dir
        if isinstance(config, str):
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config

        # Inject data_dir into the config
        if "identification" in cfg:
            if isinstance(cfg["identification"], dict):
                cfg["identification"]["data_dir"] = data_dir
            elif isinstance(cfg["identification"], list):
                cfg["identification"][0]["data_dir"] = data_dir
        elif "calibration" in cfg:
            if isinstance(cfg["calibration"], dict):
                cfg["calibration"]["data_dir"] = data_dir
            elif isinstance(cfg["calibration"], list):
                cfg["calibration"][0]["data_dir"] = data_dir
        else:
            cfg["data_dir"] = data_dir

        # Write modified config to a temp file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tmp)
        tmp.close()
        config_file = tmp.name
        _created_temp_config = True
    elif isinstance(config, str):
        config_file = config
    else:
        raise ValueError(
            "config must be a file path (string). "
            "Use data_dir= to specify the trajectory data directory."
        )

    # Extract data_dir from config if not passed explicitly
    if data_dir is None and isinstance(config, str):
        import yaml

        try:
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f)
            # unified configs nest the task sections under ``tasks:``
            sections = cfg.get("tasks", cfg)
            for section in ("identification", "calibration"):
                if section in sections and isinstance(sections[section], dict):
                    data_dir = sections[section].get("data_dir", data_dir)
        except Exception:
            pass
    elif data_dir is None and not isinstance(config, str):
        if isinstance(config, dict):
            sections = config.get("tasks", config)
            for section in ("identification", "calibration"):
                if section in sections and isinstance(sections[section], dict):
                    data_dir = sections[section].get("data_dir", data_dir)

    # Create the concrete identification instance
    identification = _SimpleIdentification(
        self.robot, config_file, data_dir=data_dir
    )

    try:
        # Initialize: process data, compute regressor, compute reference torque
        identification.initialize(truncate=truncate)

        # Solve for base parameters
        phi_base = identification.solve(
            decimate=decimate,
            decimation_factor=decimation_factor,
            zero_tolerance=zero_tolerance,
            plotting=plotting,
            save_results=save_results,
        )

        # Extract results
        result_dict = identification.result or {}

        return IdentificationResult(
            phi_base=phi_base,
            params_base=result_dict.get("params_base"),
            phi_standard=result_dict.get("phi_standard"),
            rms_error=identification.rms_error,
            correlation=identification.correlation,
            backend=self.backend_name,
            model_path=getattr(self.robot, "robot_urdf", ""),
            config=identification.identif_config,
            raw=result_dict,
        )
    finally:
        # Clean up temp config file if we created one
        if _created_temp_config:
            try:
                os.unlink(config_file)
            except OSError:
                pass
```
