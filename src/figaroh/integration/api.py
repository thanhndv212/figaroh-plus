"""
Integration API for FIGAROH identification workflows.

Provides the high-level RobotIdentificationSystem class that wraps the
backend abstraction and BaseIdentification workflow into a simple,
one-line API.
"""

import os
import logging
import numpy as np
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class IdentificationResult:
    """Container for identification results.

    Attributes:
        phi_base: Identified base parameters
        params_base: Base parameter names
        phi_standard: Full standard parameters (if reconstruction enabled)
        rms_error: RMS error of the identification
        correlation: Correlation between measured and estimated torques
        backend: Name of the backend used
        model_path: Path to the robot model
        config: Configuration used
        raw: Raw result dict from BaseIdentification (for advanced access)
    """
    phi_base: Optional[np.ndarray] = None
    params_base: Optional[list] = None
    phi_standard: Optional[np.ndarray] = None
    rms_error: Optional[float] = None
    correlation: Optional[float] = None
    backend: str = ""
    model_path: str = ""
    config: Optional[dict] = None
    raw: Optional[dict] = None


class _SimpleIdentification:
    """Concrete identification class for the integration API.

    Implements the BaseIdentification interface with a simple CSV-based
    data loading strategy. This wraps BaseIdentification's workflow so
    the integration API can use it without requiring robot-specific
    subclasses.

    Note:
        This is a lightweight wrapper around BaseIdentification.
        For advanced use cases, subclass BaseIdentification directly.
    """

    def __init__(self, robot, config_file="config/robot_config.yaml",
                 data_dir=None):
        # Import here to avoid circular imports at module level
        from ..identification.base_identification import BaseIdentification

        # Store for load_trajectory_data
        self._config_file = config_file
        self._data_dir = data_dir

        # Dynamically create a concrete subclass
        class _ConcreteIdentification(BaseIdentification):
            """Concrete subclass that delegates load_trajectory_data."""

            def load_trajectory_data(self):
                return self._outer.load_trajectory_data()

        # Set the outer reference for delegation
        _ConcreteIdentification._outer = self

        # Create and initialize the identification instance
        self._identification = _ConcreteIdentification(robot, config_file)

    # Forward attribute access to the wrapped identification
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._identification, name)

    def load_trajectory_data(self):
        """Load trajectory data from CSV files.

        Reads position, velocity, acceleration, and torque data from
        CSV files specified in the identification config under 'data_dir'
        or 'data_files'.

        Expected files in data_dir:
            - q.csv:    Joint positions (N x nq)
            - dq.csv:   Joint velocities (N x nv)
            - ddq.csv:  Joint accelerations (N x nv)
            - tau.csv:  Joint torques (N x nv)

        Returns:
            tuple: (timestamps, positions, velocities, accelerations, torques)

        Raises:
            ValueError: If no data_dir in config
            FileNotFoundError: If data files not found
        """
        # Check instance data_dir first (passed directly to constructor),
        # then fall back to config data_dir
        data_dir = self._data_dir
        if data_dir is None:
            data_dir = self._identification.identif_config.get("data_dir")
        data_files = self._identification.identif_config.get("data_files", {})

        if data_dir is not None:
            # Default file paths
            default_files = {
                'q': os.path.join(data_dir, 'q.csv'),
                'dq': os.path.join(data_dir, 'dq.csv'),
                'ddq': os.path.join(data_dir, 'ddq.csv'),
                'tau': os.path.join(data_dir, 'tau.csv'),
            }
        elif data_files:
            default_files = {}
        else:
            raise ValueError(
                "No data source specified. The identification config must "
                "include a 'data_dir' field pointing to trajectory CSV files, "
                "or 'data_files' with explicit file paths."
            )

        # Merge defaults with overrides from config
        files = {**default_files, **data_files}

        data = {}
        for key in ['q', 'dq', 'ddq', 'tau']:
            path = files.get(key)
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(
                    f"Data file for '{key}' not found: {path}. "
                    f"Ensure the file exists or specify 'data_files' in config."
                )
            data[key] = np.loadtxt(path, delimiter=',')

        # Ensure 2D arrays
        for key in data:
            if data[key].ndim == 1:
                data[key] = data[key].reshape(-1, 1)

        timestamps = np.arange(len(data['q']))
        return (
            timestamps,
            data['q'],
            data['dq'],
            data['ddq'],
            data['tau'],
        )


class RobotIdentificationSystem:
    """High-level API for robot dynamic parameter identification.

    Provides a simple interface that wraps the backend abstraction and
    BaseIdentification workflow. Supports switching between dynamics
    backends (Pinocchio, MuJoCo) with minimal code changes.

    Example:
        >>> # Create system from URDF
        >>> system = RobotIdentificationSystem.from_urdf("robot.urdf")
        >>>
        >>> # Or with specific backend
        >>> system = RobotIdentificationSystem.from_urdf(
        ...     "robot.urdf", backend="mujoco"
        ... )
        >>>
        >>> # Run identification
        >>> results = system.identify_parameters(
        ...     config="config/robot_config.yaml",
        ...     data_dir="data/",
        ...     plotting=False
        ... )
        >>> print(f"Identified {len(results.phi_base)} base parameters")
        >>> print(f"RMS error: {results.rms_error}")
    """

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

    @classmethod
    def from_urdf(cls, urdf_path, backend="pinocchio", package_dirs=None,
                  free_flyer=False, **kwargs):
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
            **kwargs
        )
        return cls(robot, backend_name=backend, **kwargs)

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

    def identify_parameters(self, config, data_dir=None, truncate=None,
                            decimate=True, decimation_factor=10,
                            zero_tolerance=0.001, plotting=False,
                            save_results=False, **kwargs):
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
                with open(config, 'r') as f:
                    cfg = yaml.safe_load(f)
            else:
                cfg = config

            # Inject data_dir into the config
            if 'identification' in cfg:
                if isinstance(cfg['identification'], dict):
                    cfg['identification']['data_dir'] = data_dir
                elif isinstance(cfg['identification'], list):
                    cfg['identification'][0]['data_dir'] = data_dir
            elif 'calibration' in cfg:
                if isinstance(cfg['calibration'], dict):
                    cfg['calibration']['data_dir'] = data_dir
                elif isinstance(cfg['calibration'], list):
                    cfg['calibration'][0]['data_dir'] = data_dir
            else:
                cfg['data_dir'] = data_dir

            # Write modified config to a temp file
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False
            )
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
                with open(config_file, 'r') as f:
                    cfg = yaml.safe_load(f)
                for section in ('identification', 'calibration'):
                    if section in cfg and isinstance(cfg[section], dict):
                        data_dir = cfg[section].get('data_dir', data_dir)
            except Exception:
                pass
        elif data_dir is None and not isinstance(config, str):
            if isinstance(config, dict):
                for section in ('identification', 'calibration'):
                    if section in config and isinstance(config[section], dict):
                        data_dir = config[section].get('data_dir', data_dir)

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
                save_results=save_results
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
                model_path=getattr(self.robot, 'robot_urdf', ''),
                config=identification.identif_config,
                raw=result_dict
            )
        finally:
            # Clean up temp config file if we created one
            if _created_temp_config:
                try:
                    os.unlink(config_file)
                except OSError:
                    pass

    @property
    def backend(self):
        """Access the underlying dynamics backend."""
        return self.robot.backend

    @property
    def nq(self) -> int:
        """Number of position variables."""
        return self.robot.model.nq

    @property
    def nv(self) -> int:
        """Number of velocity variables."""
        return self.robot.model.nv

    def __repr__(self) -> str:
        return (
            f"RobotIdentificationSystem("
            f"backend='{self.backend_name}', "
            f"nq={self.nq}, nv={self.nv})"
        )
