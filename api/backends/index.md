# Backends

Pluggable dynamics computation backends. See [Concepts → Backends](https://thanhndv212.github.io/figaroh-plus/concepts/backends/index.md) for the design overview and a guide to choosing one.

## base

FIGAROH Dynamics Backend System

Abstract base class for pluggable dynamics computation backends. This enables FIGAROH to work with multiple simulators (Pinocchio, MuJoCo, Genesis, Isaac Sim) while maintaining consistent algorithms and APIs.

## `DynamicsBackend(model_path, **kwargs)`

Bases: `ABC`

Abstract base class for dynamics computation backends.

This interface defines the minimum set of dynamics computations required for FIGAROH's calibration and identification algorithms. Implementations wrap specific simulators (Pinocchio, MuJoCo, Genesis, Isaac Sim).

Design Philosophy:

- Abstract interface, concrete implementations
- Consistent algorithm behavior across backends
- Performance optimization in implementations
- No algorithm changes required when switching backends

Initialize dynamics backend with robot model.

Parameters:

| Name         | Type  | Description                                          | Default    |
| ------------ | ----- | ---------------------------------------------------- | ---------- |
| `model_path` | `str` | Path to robot model file (format depends on backend) | *required* |
| `**kwargs`   |       | Backend-specific configuration options               | `{}`       |

Source code in `src/figaroh/backends/base.py`

```
def __init__(self, model_path: str, **kwargs):
    """
    Initialize dynamics backend with robot model.

    Args:
        model_path: Path to robot model file (format depends on backend)
        **kwargs: Backend-specific configuration options
    """
    self._model_path = model_path
    self._config = kwargs
```

### `nq`

Number of position variables (configuration space dimension).

### `nv`

Number of velocity variables (tangent space dimension).

### `model_format`

Backend model format.

Returns:

| Type  | Description                              |
| ----- | ---------------------------------------- |
| `str` | Format name: 'urdf', 'mjcf', 'usd', etc. |

### `model_path`

Path to robot model file.

### `config`

Backend configuration options.

### `compute_mass_matrix(q)`

Compute mass/inertia matrix M(q).

The mass matrix appears in the equation of motion

M(q) * qdd + C(q, qd) * qd + g(q) = tau

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Name | Type      | Description                                        |
| ---- | --------- | -------------------------------------------------- |
| `M`  | `ndarray` | Mass matrix [nv x nv], symmetric positive definite |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
    """
    Compute mass/inertia matrix M(q).

    The mass matrix appears in the equation of motion:
        M(q) * qdd + C(q, qd) * qd + g(q) = tau

    Args:
        q: Joint positions [nq]

    Returns:
        M: Mass matrix [nv x nv], symmetric positive definite
    """
    pass
```

### `compute_coriolis_matrix(q, v)`

Compute Coriolis and centrifugal effects matrix C(q, qd).

Note: Some simulators compute C such that C(q, qd) * qd represents Coriolis forces, while others use different conventions.

Parameters:

| Name | Type      | Description           | Default    |
| ---- | --------- | --------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq]  | *required* |
| `v`  | `ndarray` | Joint velocities [nv] | *required* |

Returns:

| Name | Type      | Description               |
| ---- | --------- | ------------------------- |
| `C`  | `ndarray` | Coriolis matrix [nv x nv] |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_coriolis_matrix(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute Coriolis and centrifugal effects matrix C(q, qd).

    Note: Some simulators compute C such that C(q, qd) * qd represents
    Coriolis forces, while others use different conventions.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]

    Returns:
        C: Coriolis matrix [nv x nv]
    """
    pass
```

### `compute_gravity_vector(q)`

Compute gravity effects vector g(q).

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Name | Type      | Description         |
| ---- | --------- | ------------------- |
| `g`  | `ndarray` | Gravity vector [nv] |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
    """
    Compute gravity effects vector g(q).

    Args:
        q: Joint positions [nq]

    Returns:
        g: Gravity vector [nv]
    """
    pass
```

### `compute_forward_kinematics(q)`

Compute forward kinematics for all frames.

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Type             | Description                                                                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dict[str, Any]` | Dictionary mapping frame names to transformations:                                                                                             |
| `Dict[str, Any]` | { 'frame_name': { 'position': np.ndarray [3], 'orientation': np.ndarray [3, 3] or [4] (quat), 'transformation': np.ndarray [4, 4] (optional) } |
| `Dict[str, Any]` | }                                                                                                                                              |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_forward_kinematics(self, q: np.ndarray) -> Dict[str, Any]:
    """
    Compute forward kinematics for all frames.

    Args:
        q: Joint positions [nq]

    Returns:
        Dictionary mapping frame names to transformations:
        {
            'frame_name': {
                'position': np.ndarray [3],
                'orientation': np.ndarray [3, 3] or [4] (quat),
                'transformation': np.ndarray [4, 4] (optional)
            }
        }
    """
    pass
```

### `compute_jacobian(q, frame)`

Compute geometric Jacobian for a specific frame.

Parameters:

| Name    | Type      | Description          | Default    |
| ------- | --------- | -------------------- | ---------- |
| `q`     | `ndarray` | Joint positions [nq] | *required* |
| `frame` | `str`     | Name of the frame    | *required* |

Returns:

| Name | Type      | Description                                                                |
| ---- | --------- | -------------------------------------------------------------------------- |
| `J`  | `ndarray` | Geometric Jacobian [6 x nv] Stacked as [linear_velocity; angular_velocity] |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_jacobian(self, q: np.ndarray, frame: str) -> np.ndarray:
    """
    Compute geometric Jacobian for a specific frame.

    Args:
        q: Joint positions [nq]
        frame: Name of the frame

    Returns:
        J: Geometric Jacobian [6 x nv]
           Stacked as [linear_velocity; angular_velocity]
    """
    pass
```

### `compute_regressor(q, v, a)`

Compute observation regressor matrix W(q, v, a).

The regressor satisfies: tau = W(q, v, a) * theta where theta is the parameter vector.

This is the core computation for linear-in-parameters identification.

Parameters:

| Name | Type      | Description              | Default    |
| ---- | --------- | ------------------------ | ---------- |
| `q`  | `ndarray` | Joint positions [nq]     | *required* |
| `v`  | `ndarray` | Joint velocities [nv]    | *required* |
| `a`  | `ndarray` | Joint accelerations [nv] | *required* |

Returns:

| Name | Type      | Description                      |
| ---- | --------- | -------------------------------- |
| `W`  | `ndarray` | Regressor matrix [nv x n_params] |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_regressor(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """
    Compute observation regressor matrix W(q, v, a).

    The regressor satisfies: tau = W(q, v, a) * theta
    where theta is the parameter vector.

    This is the core computation for linear-in-parameters identification.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        W: Regressor matrix [nv x n_params]
    """
    pass
```

### `compute_inverse_dynamics(q, v, a)`

Compute inverse dynamics (RNEA).

Given q, qd, qdd, compute required torques tau.

Parameters:

| Name | Type      | Description              | Default    |
| ---- | --------- | ------------------------ | ---------- |
| `q`  | `ndarray` | Joint positions [nq]     | *required* |
| `v`  | `ndarray` | Joint velocities [nv]    | *required* |
| `a`  | `ndarray` | Joint accelerations [nv] | *required* |

Returns:

| Name  | Type      | Description        |
| ----- | --------- | ------------------ |
| `tau` | `ndarray` | Joint torques [nv] |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_inverse_dynamics(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """
    Compute inverse dynamics (RNEA).

    Given q, qd, qdd, compute required torques tau.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        tau: Joint torques [nv]
    """
    pass
```

### `compute_forward_dynamics(q, v, tau)`

Compute forward dynamics (ABA).

Given q, qd, tau, compute resulting accelerations qdd.

Parameters:

| Name  | Type      | Description           | Default    |
| ----- | --------- | --------------------- | ---------- |
| `q`   | `ndarray` | Joint positions [nq]  | *required* |
| `v`   | `ndarray` | Joint velocities [nv] | *required* |
| `tau` | `ndarray` | Joint torques [nv]    | *required* |

Returns:

| Name | Type      | Description              |
| ---- | --------- | ------------------------ |
| `a`  | `ndarray` | Joint accelerations [nv] |

Source code in `src/figaroh/backends/base.py`

```
@abstractmethod
def compute_forward_dynamics(
    self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """
    Compute forward dynamics (ABA).

    Given q, qd, tau, compute resulting accelerations qdd.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        tau: Joint torques [nv]

    Returns:
        a: Joint accelerations [nv]
    """
    pass
```

### `compute_dynamics_derivatives(q, v)`

Compute derivatives of dynamics (optional, for advanced algorithms).

Parameters:

| Name | Type      | Description           | Default    |
| ---- | --------- | --------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq]  | *required* |
| `v`  | `ndarray` | Joint velocities [nv] | *required* |

Returns:

| Type                 | Description                                                                                                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dict[str, ndarray]` | Dictionary with derivatives:                                                                                                                                                                        |
| `Dict[str, ndarray]` | { 'dM_dq': [nv x nv x nq], # Jacobian of M w.r.t. q 'dC_dq': [nv x nv x nq], # Jacobian of C w.r.t. q 'dC_dv': [nv x nv x nv], # Jacobian of C w.r.t. v 'dg_dq': [nv x nq] # Jacobian of g w.r.t. q |
| `Dict[str, ndarray]` | }                                                                                                                                                                                                   |

Source code in `src/figaroh/backends/base.py`

```
def compute_dynamics_derivatives(
    self, q: np.ndarray, v: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute derivatives of dynamics (optional, for advanced algorithms).

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]

    Returns:
        Dictionary with derivatives:
        {
            'dM_dq': [nv x nv x nq],  # Jacobian of M w.r.t. q
            'dC_dq': [nv x nv x nq],  # Jacobian of C w.r.t. q
            'dC_dv': [nv x nv x nv],  # Jacobian of C w.r.t. v
            'dg_dq': [nv x nq]        # Jacobian of g w.r.t. q
        }
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement dynamics derivatives"
    )
```

### `get_joint_names()`

Get list of joint names.

Returns:

| Type   | Description                  |
| ------ | ---------------------------- |
| `list` | List of joint names in order |

Source code in `src/figaroh/backends/base.py`

```
def get_joint_names(self) -> list:
    """
    Get list of joint names.

    Returns:
        List of joint names in order
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement get_joint_names"
    )
```

### `get_frame_names()`

Get list of frame names.

Returns:

| Type   | Description                                   |
| ------ | --------------------------------------------- |
| `list` | List of frame names available for FK/Jacobian |

Source code in `src/figaroh/backends/base.py`

```
def get_frame_names(self) -> list:
    """
    Get list of frame names.

    Returns:
        List of frame names available for FK/Jacobian
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement get_frame_names"
    )
```

### `get_inertias()`

Get per-body inertia objects.

Used for nonzero-inertia filtering in regressor computation.

Returns:

| Type   | Description                                      |
| ------ | ------------------------------------------------ |
| `list` | List of inertia objects (backend-specific type). |

Source code in `src/figaroh/backends/base.py`

```
def get_inertias(self) -> list:
    """
    Get per-body inertia objects.

    Used for nonzero-inertia filtering in regressor computation.

    Returns:
        List of inertia objects (backend-specific type).
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement get_inertias"
    )
```

### `get_frame_id(frame)`

Get frame ID by name.

Used for frame-based Jacobian/calibration computations.

Parameters:

| Name    | Type  | Description | Default    |
| ------- | ----- | ----------- | ---------- |
| `frame` | `str` | Frame name  | *required* |

Returns:

| Type  | Description        |
| ----- | ------------------ |
| `int` | Frame ID (integer) |

Source code in `src/figaroh/backends/base.py`

```
def get_frame_id(self, frame: str) -> int:
    """
    Get frame ID by name.

    Used for frame-based Jacobian/calibration computations.

    Args:
        frame: Frame name

    Returns:
        Frame ID (integer)
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement get_frame_id"
    )
```

### `compute_difference(q1, q2)`

Compute Lie group difference between two configurations (q2 ⊖ q1).

For revolute joints this is simple subtraction; for free-flyer it's SE3 log.

Parameters:

| Name | Type      | Description               | Default    |
| ---- | --------- | ------------------------- | ---------- |
| `q1` | `ndarray` | First configuration [nq]  | *required* |
| `q2` | `ndarray` | Second configuration [nq] | *required* |

Returns:

| Type      | Description            |
| --------- | ---------------------- |
| `ndarray` | Difference vector [nv] |

Source code in `src/figaroh/backends/base.py`

```
def compute_difference(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute Lie group difference between two configurations (q2 ⊖ q1).

    For revolute joints this is simple subtraction; for free-flyer it's SE3 log.

    Args:
        q1: First configuration [nq]
        q2: Second configuration [nq]

    Returns:
        Difference vector [nv]
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement compute_difference"
    )
```

### `compute_integrate(q, v)`

Integrate configuration by velocity (q ⊕ v).

Parameters:

| Name | Type      | Description        | Default    |
| ---- | --------- | ------------------ | ---------- |
| `q`  | `ndarray` | Configuration [nq] | *required* |
| `v`  | `ndarray` | Velocity [nv]      | *required* |

Returns:

| Type      | Description            |
| --------- | ---------------------- |
| `ndarray` | New configuration [nq] |

Source code in `src/figaroh/backends/base.py`

```
def compute_integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Integrate configuration by velocity (q ⊕ v).

    Args:
        q: Configuration [nq]
        v: Velocity [nv]

    Returns:
        New configuration [nq]
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement compute_integrate"
    )
```

### `random_configuration()`

Generate a random configuration within joint limits.

Returns:

| Type      | Description               |
| --------- | ------------------------- |
| `ndarray` | Random configuration [nq] |

Source code in `src/figaroh/backends/base.py`

```
def random_configuration(self) -> np.ndarray:
    """
    Generate a random configuration within joint limits.

    Returns:
        Random configuration [nq]
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement random_configuration"
    )
```

### `get_model_object()`

Escape hatch: return the underlying simulator model object.

For Pinocchio this is the pin.Model; for MuJoCo it's mj.MjModel. Used by advanced features (PseudoInertia, collision model, etc.) that need backend-specific types not yet abstracted.

Returns:

| Type  | Description                   |
| ----- | ----------------------------- |
| `Any` | Backend-specific model object |

Source code in `src/figaroh/backends/base.py`

```
def get_model_object(self) -> Any:
    """
    Escape hatch: return the underlying simulator model object.

    For Pinocchio this is the pin.Model; for MuJoCo it's mj.MjModel.
    Used by advanced features (PseudoInertia, collision model, etc.) that need
    backend-specific types not yet abstracted.

    Returns:
        Backend-specific model object
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not implement get_model_object"
    )
```

### `__enter__()`

Enter context manager (for resource management).

Source code in `src/figaroh/backends/base.py`

```
def __enter__(self):
    """Enter context manager (for resource management)."""
    return self
```

### `__exit__(exc_type, exc_val, exc_tb)`

Exit context manager (for cleanup).

Source code in `src/figaroh/backends/base.py`

```
def __exit__(self, exc_type, exc_val, exc_tb):
    """Exit context manager (for cleanup)."""
    pass
```

## pinocchio

Pinocchio Dynamics Backend for FIGAROH

Default backend using Pinocchio's rigid body dynamics library. Wraps existing Pinocchio usage into the DynamicsBackend interface.

Features:

- Excellent URDF support
- CPU-optimized dynamics algorithms
- Full frame and Jacobian support
- Lie group operations (difference, integrate)

## `PinocchioBackend(model_path, **kwargs)`

Bases: `DynamicsBackend`

Pinocchio dynamics backend for FIGAROH (default).

This backend leverages Pinocchio's rigid body dynamics algorithms for efficient computation of mass matrices, Coriolis effects, gravity, forward kinematics, Jacobians, and the regressor matrix.

Pinocchio is the primary dependency of FIGAROH and provides the most complete URDF support.

Example

> > > backend = PinocchioBackend(model_path="robot.urdf") M = backend.compute_mass_matrix(q)

Initialize Pinocchio backend.

Parameters:

| Name         | Type  | Description                                                                                                                                                                                                                                            | Default    |
| ------------ | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `model_path` | `str` | Path to URDF file                                                                                                                                                                                                                                      | *required* |
| `**kwargs`   |       | Additional configuration - free_flyer: Enable free-flyer root joint (default: False) - isFext: Alias for free_flyer (default: False) - package_dirs: Directories for mesh resolution (default: None) - verbose: Enable verbose output (default: False) | `{}`       |

Source code in `src/figaroh/backends/pinocchio.py`

```
def __init__(self, model_path: str, **kwargs):
    """
    Initialize Pinocchio backend.

    Args:
        model_path: Path to URDF file
        **kwargs: Additional configuration
            - free_flyer: Enable free-flyer root joint (default: False)
            - isFext: Alias for free_flyer (default: False)
            - package_dirs: Directories for mesh resolution (default: None)
            - verbose: Enable verbose output (default: False)
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError(
            "Pinocchio is not installed. Install with: pip install pin"
        )

    super().__init__(model_path, **kwargs)

    # Optional free-flyer root joint
    root_joint = None
    if kwargs.get("free_flyer", False) or kwargs.get("isFext", False):
        root_joint = pin.JointModelFreeFlyer()

    # Package dirs for mesh files
    package_dirs = kwargs.get("package_dirs", None)

    # Build model from URDF
    try:
        if package_dirs is not None:
            self.model = pin.buildModelFromUrdf(
                model_path, package_dirs, root_joint
            )
        else:
            self.model = pin.buildModelFromUrdf(model_path, root_joint)
        self.data = self.model.createData()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_path}' with Pinocchio: {e}\n"
            f"Ensure the file is a valid URDF."
        )

    self._verbose = kwargs.get("verbose", False)
```

### `nq`

Number of position variables (configuration space dimension).

### `nv`

Number of velocity variables (tangent space dimension).

### `model_format`

Model format (URDF).

### `from_model(model, data=None, **kwargs)`

Create PinocchioBackend from an existing pin.Model.

This avoids re-loading the URDF and shares the same model/data pair as an existing Robot/RobotWrapper instance.

Parameters:

| Name       | Type | Description                                          | Default    |
| ---------- | ---- | ---------------------------------------------------- | ---------- |
| `model`    |      | Existing pinocchio.Model                             | *required* |
| `data`     |      | Existing pinocchio.Data (created from model if None) | `None`     |
| `**kwargs` |      | Additional configuration (verbose, etc.)             | `{}`       |

Returns:

| Type | Description                                           |
| ---- | ----------------------------------------------------- |
|      | PinocchioBackend instance wrapping the existing model |

Raises:

| Type          | Description                   |
| ------------- | ----------------------------- |
| `ImportError` | If Pinocchio is not installed |

Source code in `src/figaroh/backends/pinocchio.py`

```
@classmethod
def from_model(cls, model, data=None, **kwargs):
    """
    Create PinocchioBackend from an existing pin.Model.

    This avoids re-loading the URDF and shares the same model/data pair
    as an existing Robot/RobotWrapper instance.

    Args:
        model: Existing pinocchio.Model
        data: Existing pinocchio.Data (created from model if None)
        **kwargs: Additional configuration (verbose, etc.)

    Returns:
        PinocchioBackend instance wrapping the existing model

    Raises:
        ImportError: If Pinocchio is not installed
    """
    if not PINOCCHIO_AVAILABLE:
        raise ImportError(
            "Pinocchio is not installed. Install with: pip install pin"
        )

    # Create instance without calling __init__ (which loads from URDF)
    backend = cls.__new__(cls)
    DynamicsBackend.__init__(backend, model_path=None, **kwargs)
    backend.model = model
    backend.data = data if data is not None else model.createData()
    backend._verbose = kwargs.get("verbose", False)
    return backend
```

### `compute_mass_matrix(q)`

Compute mass matrix using Pinocchio's CRBA (Composite Rigid Body Algorithm).

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Name | Type      | Description                                        |
| ---- | --------- | -------------------------------------------------- |
| `M`  | `ndarray` | Mass matrix [nv x nv], symmetric positive definite |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
    """
    Compute mass matrix using Pinocchio's CRBA (Composite Rigid Body Algorithm).

    Args:
        q: Joint positions [nq]

    Returns:
        M: Mass matrix [nv x nv], symmetric positive definite
    """
    return pin.crba(self.model, self.data, q).copy()
```

### `compute_coriolis_matrix(q, v)`

Compute Coriolis and centrifugal effects matrix C(q, qd).

Uses Pinocchio's computeCoriolisMatrix algorithm.

Parameters:

| Name | Type      | Description           | Default    |
| ---- | --------- | --------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq]  | *required* |
| `v`  | `ndarray` | Joint velocities [nv] | *required* |

Returns:

| Name | Type      | Description               |
| ---- | --------- | ------------------------- |
| `C`  | `ndarray` | Coriolis matrix [nv x nv] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_coriolis_matrix(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute Coriolis and centrifugal effects matrix C(q, qd).

    Uses Pinocchio's computeCoriolisMatrix algorithm.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]

    Returns:
        C: Coriolis matrix [nv x nv]
    """
    return pin.computeCoriolisMatrix(self.model, self.data, q, v).copy()
```

### `compute_gravity_vector(q)`

Compute gravity effects vector using Pinocchio's generalized gravity.

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Name | Type      | Description         |
| ---- | --------- | ------------------- |
| `g`  | `ndarray` | Gravity vector [nv] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
    """
    Compute gravity effects vector using Pinocchio's generalized gravity.

    Args:
        q: Joint positions [nq]

    Returns:
        g: Gravity vector [nv]
    """
    return pin.computeGeneralizedGravity(self.model, self.data, q).copy()
```

### `compute_forward_kinematics(q)`

Compute forward kinematics for all frames.

Computes joint placements via forwardKinematics and updates all frame placements via updateFramePlacements.

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Type             | Description                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| `Dict[str, Any]` | Dictionary mapping frame names to transformations:                                                                    |
| `Dict[str, Any]` | { 'frame_name': { 'position': np.ndarray [3], 'orientation': np.ndarray [3, 3], 'transformation': np.ndarray [4, 4] } |
| `Dict[str, Any]` | }                                                                                                                     |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_forward_kinematics(self, q: np.ndarray) -> Dict[str, Any]:
    """
    Compute forward kinematics for all frames.

    Computes joint placements via forwardKinematics and updates all
    frame placements via updateFramePlacements.

    Args:
        q: Joint positions [nq]

    Returns:
        Dictionary mapping frame names to transformations:
        {
            'frame_name': {
                'position': np.ndarray [3],
                'orientation': np.ndarray [3, 3],
                'transformation': np.ndarray [4, 4]
            }
        }
    """
    # Compute joint placements
    pin.forwardKinematics(self.model, self.data, q)

    # Update all frame placements
    pin.updateFramePlacements(self.model, self.data)

    fk_results = {}

    # Iterate over all frames (skip universe at index 0)
    for i in range(1, len(self.model.frames)):
        frame = self.model.frames[i]
        placement = self.data.oMf[i]

        fk_results[frame.name] = {
            "position": placement.translation.copy(),
            "orientation": placement.rotation.copy(),
            "transformation": placement.homogeneous.copy(),
        }

    return fk_results
```

### `compute_jacobian(q, frame)`

Compute geometric Jacobian for a specific frame.

Parameters:

| Name    | Type      | Description          | Default    |
| ------- | --------- | -------------------- | ---------- |
| `q`     | `ndarray` | Joint positions [nq] | *required* |
| `frame` | `str`     | Name of the frame    | *required* |

Returns:

| Name | Type      | Description                                                                |
| ---- | --------- | -------------------------------------------------------------------------- |
| `J`  | `ndarray` | Geometric Jacobian [6 x nv] Stacked as [linear_velocity; angular_velocity] |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If frame is not found in the model |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_jacobian(self, q: np.ndarray, frame: str) -> np.ndarray:
    """
    Compute geometric Jacobian for a specific frame.

    Args:
        q: Joint positions [nq]
        frame: Name of the frame

    Returns:
        J: Geometric Jacobian [6 x nv]
           Stacked as [linear_velocity; angular_velocity]

    Raises:
        ValueError: If frame is not found in the model
    """
    # Look up frame ID
    frame_id = self.model.getFrameId(frame)
    if frame_id >= len(self.model.frames):
        raise ValueError(
            f"Frame '{frame}' not found in model. "
            f"Available frames: {[f.name for f in self.model.frames[1:]]}"
        )

    # Compute frame Jacobian (ensure [6, nv] shape)
    J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL)

    return J.reshape(6, self.model.nv).copy()
```

### `compute_regressor(q, v, a)`

Compute observation regressor matrix W(q, v, a).

The regressor satisfies: tau = W(q, v, a) * theta where theta is the parameter vector.

Uses Pinocchio's computeJointTorqueRegressor, which computes the regressor for all 10 standard inertial parameters per body.

Parameters:

| Name | Type      | Description              | Default    |
| ---- | --------- | ------------------------ | ---------- |
| `q`  | `ndarray` | Joint positions [nq]     | *required* |
| `v`  | `ndarray` | Joint velocities [nv]    | *required* |
| `a`  | `ndarray` | Joint accelerations [nv] | *required* |

Returns:

| Name | Type      | Description                          |
| ---- | --------- | ------------------------------------ |
| `W`  | `ndarray` | Regressor matrix [nv x (10 * nbody)] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_regressor(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """
    Compute observation regressor matrix W(q, v, a).

    The regressor satisfies: tau = W(q, v, a) * theta
    where theta is the parameter vector.

    Uses Pinocchio's computeJointTorqueRegressor, which computes
    the regressor for all 10 standard inertial parameters per body.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        W: Regressor matrix [nv x (10 * nbody)]
    """
    W = pin.computeJointTorqueRegressor(self.model, self.data, q, v, a)
    # Ensure 2D shape [nv, n_params] (Pinocchio may return 1D for nv=1)
    if W.ndim == 1:
        W = W.reshape(self.model.nv, -1)
    return W.copy()
```

### `compute_inverse_dynamics(q, v, a)`

Compute inverse dynamics (RNEA) using Pinocchio's rnea.

Parameters:

| Name | Type      | Description              | Default    |
| ---- | --------- | ------------------------ | ---------- |
| `q`  | `ndarray` | Joint positions [nq]     | *required* |
| `v`  | `ndarray` | Joint velocities [nv]    | *required* |
| `a`  | `ndarray` | Joint accelerations [nv] | *required* |

Returns:

| Name  | Type      | Description        |
| ----- | --------- | ------------------ |
| `tau` | `ndarray` | Joint torques [nv] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_inverse_dynamics(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """
    Compute inverse dynamics (RNEA) using Pinocchio's rnea.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        tau: Joint torques [nv]
    """
    return pin.rnea(self.model, self.data, q, v, a).copy()
```

### `compute_forward_dynamics(q, v, tau)`

Compute forward dynamics (ABA) using Pinocchio's aba.

Parameters:

| Name  | Type      | Description           | Default    |
| ----- | --------- | --------------------- | ---------- |
| `q`   | `ndarray` | Joint positions [nq]  | *required* |
| `v`   | `ndarray` | Joint velocities [nv] | *required* |
| `tau` | `ndarray` | Joint torques [nv]    | *required* |

Returns:

| Name | Type      | Description              |
| ---- | --------- | ------------------------ |
| `a`  | `ndarray` | Joint accelerations [nv] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_forward_dynamics(
    self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """
    Compute forward dynamics (ABA) using Pinocchio's aba.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        tau: Joint torques [nv]

    Returns:
        a: Joint accelerations [nv]
    """
    return pin.aba(self.model, self.data, q, v, tau).copy()
```

### `get_joint_names()`

Get list of joint names.

Returns:

| Type   | Description                                      |
| ------ | ------------------------------------------------ |
| `list` | List of joint names in order (skipping universe) |

Source code in `src/figaroh/backends/pinocchio.py`

```
def get_joint_names(self) -> list:
    """
    Get list of joint names.

    Returns:
        List of joint names in order (skipping universe)
    """
    return list(self.model.names[1:])
```

### `get_frame_names()`

Get list of frame names.

Returns:

| Type   | Description                                                       |
| ------ | ----------------------------------------------------------------- |
| `list` | List of frame names available for FK/Jacobian (skipping universe) |

Source code in `src/figaroh/backends/pinocchio.py`

```
def get_frame_names(self) -> list:
    """
    Get list of frame names.

    Returns:
        List of frame names available for FK/Jacobian (skipping universe)
    """
    return [f.name for f in self.model.frames[1:]]
```

### `get_inertias()`

Get per-body inertia objects.

Returns:

| Type   | Description                                                              |
| ------ | ------------------------------------------------------------------------ |
| `list` | List of inertia objects (includes universe at index 0 with zero inertia) |

Source code in `src/figaroh/backends/pinocchio.py`

```
def get_inertias(self) -> list:
    """
    Get per-body inertia objects.

    Returns:
        List of inertia objects (includes universe at index 0 with zero inertia)
    """
    return list(self.model.inertias)
```

### `get_frame_id(frame)`

Get frame ID by name.

Parameters:

| Name    | Type  | Description | Default    |
| ------- | ----- | ----------- | ---------- |
| `frame` | `str` | Frame name  | *required* |

Returns:

| Type  | Description        |
| ----- | ------------------ |
| `int` | Frame ID (integer) |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If frame is not found in the model |

Source code in `src/figaroh/backends/pinocchio.py`

```
def get_frame_id(self, frame: str) -> int:
    """
    Get frame ID by name.

    Args:
        frame: Frame name

    Returns:
        Frame ID (integer)

    Raises:
        ValueError: If frame is not found in the model
    """
    frame_id = self.model.getFrameId(frame)
    if frame_id >= len(self.model.frames):
        raise ValueError(
            f"Frame '{frame}' not found in model. "
            f"Available frames: {[f.name for f in self.model.frames[1:]]}"
        )
    return frame_id
```

### `compute_difference(q1, q2)`

Compute Lie group difference between two configurations (q2 ⊖ q1).

Parameters:

| Name | Type      | Description               | Default    |
| ---- | --------- | ------------------------- | ---------- |
| `q1` | `ndarray` | First configuration [nq]  | *required* |
| `q2` | `ndarray` | Second configuration [nq] | *required* |

Returns:

| Type      | Description            |
| --------- | ---------------------- |
| `ndarray` | Difference vector [nv] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_difference(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compute Lie group difference between two configurations (q2 ⊖ q1).

    Args:
        q1: First configuration [nq]
        q2: Second configuration [nq]

    Returns:
        Difference vector [nv]
    """
    return pin.difference(self.model, q1, q2)
```

### `compute_integrate(q, v)`

Integrate configuration by velocity (q ⊕ v).

Parameters:

| Name | Type      | Description        | Default    |
| ---- | --------- | ------------------ | ---------- |
| `q`  | `ndarray` | Configuration [nq] | *required* |
| `v`  | `ndarray` | Velocity [nv]      | *required* |

Returns:

| Type      | Description            |
| --------- | ---------------------- |
| `ndarray` | New configuration [nq] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def compute_integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Integrate configuration by velocity (q ⊕ v).

    Args:
        q: Configuration [nq]
        v: Velocity [nv]

    Returns:
        New configuration [nq]
    """
    return pin.integrate(self.model, q, v)
```

### `random_configuration()`

Generate a random configuration within joint limits.

Returns:

| Type      | Description               |
| --------- | ------------------------- |
| `ndarray` | Random configuration [nq] |

Source code in `src/figaroh/backends/pinocchio.py`

```
def random_configuration(self) -> np.ndarray:
    """
    Generate a random configuration within joint limits.

    Returns:
        Random configuration [nq]
    """
    return pin.randomConfiguration(self.model)
```

### `get_model_object()`

Escape hatch: return the underlying Pinocchio model.

Returns:

| Type  | Description      |
| ----- | ---------------- |
| `Any` | pin.Model object |

Source code in `src/figaroh/backends/pinocchio.py`

```
def get_model_object(self) -> Any:
    """
    Escape hatch: return the underlying Pinocchio model.

    Returns:
        pin.Model object
    """
    return self.model
```

## mujoco

MuJoCo Dynamics Backend for FIGAROH

High-performance dynamics computation using MuJoCo's optimized algorithms. Extracted and integrated from figaroh-mujoco project.

Features:

- Sparse matrix operations
- Built-in URDF → MJCF conversion
- Efficient contact dynamics
- 2-3x faster than Pinocchio for large systems

## `MuJoCoBackend(model_path, **kwargs)`

Bases: `DynamicsBackend`

MuJoCo dynamics backend for FIGAROH.

This backend leverages MuJoCo's highly optimized sparse matrix operations and efficient dynamics algorithms. MuJoCo automatically converts URDF files to its internal MJCF format.

Performance

- Mass matrix: 2-3x faster than Pinocchio
- Regressor: 2-3x faster (sparse operations)
- Best for: Large systems, optimal control, contact dynamics

Example

> > > backend = MuJoCoBackend(model_path="robot.urdf") M = backend.compute_mass_matrix(q)

Initialize MuJoCo backend.

Parameters:

| Name         | Type  | Description                                                                | Default    |
| ------------ | ----- | -------------------------------------------------------------------------- | ---------- |
| `model_path` | `str` | Path to URDF or MJCF file                                                  | *required* |
| `**kwargs`   |       | Additional configuration - verbose: Enable verbose output (default: False) | `{}`       |

Source code in `src/figaroh/backends/mujoco.py`

```
def __init__(self, model_path: str, **kwargs):
    """
    Initialize MuJoCo backend.

    Args:
        model_path: Path to URDF or MJCF file
        **kwargs: Additional configuration
            - verbose: Enable verbose output (default: False)
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError(
            "MuJoCo is not installed. Install with: pip install mujoco>=3.0.0"
        )

    super().__init__(model_path, **kwargs)

    # Load model (MuJoCo auto-converts URDF → MJCF)
    try:
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_path}' with MuJoCo: {e}\n"
            f"Ensure the file is valid URDF or MJCF format."
        )

    # Pre-allocate matrices for performance
    self._M = np.zeros((self.model.nv, self.model.nv))
    self._temp_vec = np.zeros(self.model.nv)
    self._verbose = kwargs.get("verbose", False)

    # Lazy-loaded Pinocchio model for analytical regressor computation
    # (MuJoCo does not support runtime inertial parameter perturbation)
    self._pin_model = None
    self._pin_data = None

    if self._verbose:
        print(
            f"MuJoCo model loaded: {self.model.nq} positions, {self.model.nv} velocities"
        )
```

### `nq`

Number of position variables.

### `nv`

Number of velocity variables.

### `model_format`

Model format (MJCF, but supports URDF input).

### `compute_mass_matrix(q)`

Compute mass matrix using MuJoCo's mj_crb (Composite Rigid Body) algorithm.

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Name | Type      | Description                                        |
| ---- | --------- | -------------------------------------------------- |
| `M`  | `ndarray` | Mass matrix [nv x nv], symmetric positive definite |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
    """
    Compute mass matrix using MuJoCo's mj_crb (Composite Rigid Body) algorithm.

    Args:
        q: Joint positions [nq]

    Returns:
        M: Mass matrix [nv x nv], symmetric positive definite
    """
    # Set joint positions
    self.data.qpos[:] = q

    # mj_forward must be called to initialize FK before mj_crb
    mj.mj_forward(self.model, self.data)

    # Compute mass matrix using composite rigid body algorithm
    mj.mj_crb(self.model, self.data)

    # Extract full mass matrix from sparse representation
    # MuJoCo 3.10 changed mj_fullM from (model, dst, qM) to
    # (model, data, dst); newer releases also remove data.qM.
    try:
        mj.mj_fullM(self.model, self.data, self._M)
    except TypeError:
        # The older Python binding rejects the new argument types.
        mj.mj_fullM(self.model, self._M, self.data.qM)

    return self._M.copy()
```

### `compute_coriolis_matrix(q, v)`

Compute Coriolis matrix C(q,v) via finite differences.

Uses the property that Coriolis forces f(v) = C(q,v)*v are quadratic in v. By Euler's theorem and Christoffel symbol symmetry, the Jacobian df/dv = 2*C, so C = (1/2) * df/dv.

Parameters:

| Name | Type      | Description           | Default    |
| ---- | --------- | --------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq]  | *required* |
| `v`  | `ndarray` | Joint velocities [nv] | *required* |

Returns:

| Name | Type      | Description               |
| ---- | --------- | ------------------------- |
| `C`  | `ndarray` | Coriolis matrix [nv x nv] |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_coriolis_matrix(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute Coriolis matrix C(q,v) via finite differences.

    Uses the property that Coriolis forces f(v) = C(q,v)*v are quadratic in v.
    By Euler's theorem and Christoffel symbol symmetry, the Jacobian
    df/dv = 2*C, so C = (1/2) * df/dv.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]

    Returns:
        C: Coriolis matrix [nv x nv]
    """
    nv = self.model.nv

    # Handle zero velocity: C(q, 0) = 0
    if np.allclose(v, 0):
        return np.zeros((nv, nv))

    # Compute gravity bias
    self.data.qpos[:] = q
    self.data.qvel[:] = 0
    self.data.qacc[:] = 0
    mj.mj_inverse(self.model, self.data)
    gravity = self.data.qfrc_inverse.copy()

    # Compute bias forces at velocity v
    self.data.qvel[:] = v
    mj.mj_inverse(self.model, self.data)
    bias = self.data.qfrc_inverse.copy()

    coriolis_forces = bias - gravity  # = C(q,v) * v

    # Compute Jacobian of coriolis_forces w.r.t. v via finite differences
    # df/dv[:,j] = (f(v + eps*e_j) - f(v)) / eps
    # C = (1/2) * df/dv (because f is quadratic and Christoffel symbols are symmetric)
    eps = 1e-6
    C = np.zeros((nv, nv))
    for j in range(nv):
        v_perturbed = v.copy()
        v_perturbed[j] += eps

        self.data.qpos[:] = q
        self.data.qvel[:] = v_perturbed
        self.data.qacc[:] = 0
        mj.mj_inverse(self.model, self.data)
        bias_perturbed = self.data.qfrc_inverse.copy()

        coriolis_perturbed = bias_perturbed - gravity
        C[:, j] = (coriolis_perturbed - coriolis_forces) / (2.0 * eps)

    # Restore state
    self.data.qpos[:] = q
    self.data.qvel[:] = v
    self.data.qacc[:] = 0
    mj.mj_inverse(self.model, self.data)

    return C
```

### `compute_gravity_vector(q)`

Compute gravity vector.

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Name | Type      | Description         |
| ---- | --------- | ------------------- |
| `g`  | `ndarray` | Gravity vector [nv] |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
    """
    Compute gravity vector.

    Args:
        q: Joint positions [nq]

    Returns:
        g: Gravity vector [nv]
    """
    # Set position and zero velocity/acceleration
    self.data.qpos[:] = q
    self.data.qvel[:] = 0
    self.data.qacc[:] = 0

    # Compute inverse dynamics with zero velocity/acceleration
    mj.mj_inverse(self.model, self.data)

    return self.data.qfrc_inverse.copy()
```

### `compute_forward_kinematics(q)`

Compute forward kinematics for all bodies.

Parameters:

| Name | Type      | Description          | Default    |
| ---- | --------- | -------------------- | ---------- |
| `q`  | `ndarray` | Joint positions [nq] | *required* |

Returns:

| Type             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `Dict[str, Any]` | Dictionary mapping body names to transformations |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_forward_kinematics(self, q: np.ndarray) -> Dict[str, Any]:
    """
    Compute forward kinematics for all bodies.

    Args:
        q: Joint positions [nq]

    Returns:
        Dictionary mapping body names to transformations
    """
    self.data.qpos[:] = q
    mj.mj_forward(self.model, self.data)

    fk_results = {}

    # Iterate over all bodies (excluding world)
    for i in range(1, self.model.nbody):
        body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
        if body_name is None:
            body_name = f"body_{i}"

        # Get body position and orientation
        pos = self.data.xpos[i].copy()

        # Rotation matrix from quaternion
        quat = self.data.xquat[i]  # [w, x, y, z]
        rot_mat = np.zeros(9)
        mj.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos

        fk_results[body_name] = {
            "position": pos,
            "orientation": rot_mat,
            "quaternion": quat.copy(),
            "transformation": T,
        }

    return fk_results
```

### `compute_jacobian(q, frame)`

Compute geometric Jacobian for a specific body/site.

Parameters:

| Name    | Type      | Description              | Default    |
| ------- | --------- | ------------------------ | ---------- |
| `q`     | `ndarray` | Joint positions [nq]     | *required* |
| `frame` | `str`     | Name of the body or site | *required* |

Returns:

| Name | Type      | Description                                               |
| ---- | --------- | --------------------------------------------------------- |
| `J`  | `ndarray` | Geometric Jacobian [6 x nv], stacked as [linear; angular] |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_jacobian(self, q: np.ndarray, frame: str) -> np.ndarray:
    """
    Compute geometric Jacobian for a specific body/site.

    Args:
        q: Joint positions [nq]
        frame: Name of the body or site

    Returns:
        J: Geometric Jacobian [6 x nv], stacked as [linear; angular]
    """
    self.data.qpos[:] = q
    mj.mj_forward(self.model, self.data)

    # Try to find body by name
    body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, frame)

    if body_id < 0:
        # Try site
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, frame)
        if body_id < 0:
            raise ValueError(f"Frame '{frame}' not found in model")
        use_site = True
    else:
        use_site = False

    # Allocate Jacobian
    jacp = np.zeros(3 * self.model.nv)  # Linear part
    jacr = np.zeros(3 * self.model.nv)  # Angular part

    # Compute Jacobian
    if use_site:
        mj.mj_jacSite(self.model, self.data, jacp, jacr, body_id)
    else:
        mj.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

    # Reshape and stack
    jacp = jacp.reshape(3, self.model.nv)
    jacr = jacr.reshape(3, self.model.nv)
    J = np.vstack([jacp, jacr])

    return J
```

### `compute_regressor(q, v, a)`

Compute observation regressor matrix W(q, v, a).

The regressor satisfies: tau = W @ theta where theta is the 10D inertial parameter vector per body in Pinocchio convention: [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz].

Uses Pinocchio's analytical computeJointTorqueRegressor (MuJoCo does not support runtime inertial parameter perturbation for finite differences).

Parameters:

| Name | Type      | Description              | Default    |
| ---- | --------- | ------------------------ | ---------- |
| `q`  | `ndarray` | Joint positions [nq]     | *required* |
| `v`  | `ndarray` | Joint velocities [nv]    | *required* |
| `a`  | `ndarray` | Joint accelerations [nv] | *required* |

Returns:

| Name | Type      | Description                          |
| ---- | --------- | ------------------------------------ |
| `W`  | `ndarray` | Regressor matrix [nv, 10\*(nbody-1)] |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_regressor(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """
    Compute observation regressor matrix W(q, v, a).

    The regressor satisfies: tau = W @ theta where theta is the 10D inertial
    parameter vector per body in Pinocchio convention:
    [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz].

    Uses Pinocchio's analytical computeJointTorqueRegressor (MuJoCo does not
    support runtime inertial parameter perturbation for finite differences).

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        W: Regressor matrix [nv, 10*(nbody-1)]
    """
    pin_model, pin_data = self._get_pin_model()

    W = pin.computeJointTorqueRegressor(pin_model, pin_data, q, v, a)

    # Ensure 2D shape [nv, n_params] (Pinocchio may return 1D for nv=1)
    if W.ndim == 1:
        W = W.reshape(self.model.nv, -1)

    return W.copy()
```

### `compute_inverse_dynamics(q, v, a)`

Compute inverse dynamics (RNEA) using mj_inverse.

Parameters:

| Name | Type      | Description              | Default    |
| ---- | --------- | ------------------------ | ---------- |
| `q`  | `ndarray` | Joint positions [nq]     | *required* |
| `v`  | `ndarray` | Joint velocities [nv]    | *required* |
| `a`  | `ndarray` | Joint accelerations [nv] | *required* |

Returns:

| Name  | Type      | Description        |
| ----- | --------- | ------------------ |
| `tau` | `ndarray` | Joint torques [nv] |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_inverse_dynamics(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """
    Compute inverse dynamics (RNEA) using mj_inverse.

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        tau: Joint torques [nv]
    """
    self.data.qpos[:] = q
    self.data.qvel[:] = v
    self.data.qacc[:] = a

    mj.mj_inverse(self.model, self.data)

    return self.data.qfrc_inverse.copy()
```

### `compute_forward_dynamics(q, v, tau)`

Compute forward dynamics (ABA) using mj_forward.

Applies joint torques via qfrc_applied (applied forces) rather than ctrl (which goes through actuators, which may not exist in URDF models).

Parameters:

| Name  | Type      | Description           | Default    |
| ----- | --------- | --------------------- | ---------- |
| `q`   | `ndarray` | Joint positions [nq]  | *required* |
| `v`   | `ndarray` | Joint velocities [nv] | *required* |
| `tau` | `ndarray` | Joint torques [nv]    | *required* |

Returns:

| Name | Type      | Description              |
| ---- | --------- | ------------------------ |
| `a`  | `ndarray` | Joint accelerations [nv] |

Source code in `src/figaroh/backends/mujoco.py`

```
def compute_forward_dynamics(
    self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """
    Compute forward dynamics (ABA) using mj_forward.

    Applies joint torques via qfrc_applied (applied forces) rather than
    ctrl (which goes through actuators, which may not exist in URDF models).

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        tau: Joint torques [nv]

    Returns:
        a: Joint accelerations [nv]
    """
    self.data.qpos[:] = q
    self.data.qvel[:] = v
    # Zero out any previous applied forces
    self.data.qfrc_applied[:] = 0.0
    self.data.qfrc_applied[: len(tau)] = tau

    mj.mj_forward(self.model, self.data)

    return self.data.qacc.copy()
```

### `get_joint_names()`

Get list of joint names.

Source code in `src/figaroh/backends/mujoco.py`

```
def get_joint_names(self) -> list:
    """Get list of joint names."""
    names = []
    for i in range(self.model.njnt):
        name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
        if name is None:
            name = f"joint_{i}"
        names.append(name)
    return names
```

### `get_frame_names()`

Get list of body and site names.

Source code in `src/figaroh/backends/mujoco.py`

```
def get_frame_names(self) -> list:
    """Get list of body and site names."""
    names = []

    # Add body names
    for i in range(1, self.model.nbody):  # Skip world
        name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None:
            name = f"body_{i}"
        names.append(name)

    # Add site names
    for i in range(self.model.nsite):
        name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SITE, i)
        if name is None:
            name = f"site_{i}"
        names.append(name)

    return names
```

### `__exit__(exc_type, exc_val, exc_tb)`

Cleanup MuJoCo resources.

Source code in `src/figaroh/backends/mujoco.py`

```
def __exit__(self, exc_type, exc_val, exc_tb):
    """Cleanup MuJoCo resources."""
    # MuJoCo handles cleanup automatically
    pass
```
