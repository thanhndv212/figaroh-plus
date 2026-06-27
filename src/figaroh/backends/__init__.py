"""
FIGAROH Dynamics Backends

This module provides pluggable dynamics computation backends for FIGAROH,
enabling seamless switching between different simulators while maintaining
consistent identification and calibration algorithms.

Available Backends:
-------------------
- **Pinocchio**: Default backend, excellent URDF support, CPU-optimized
- **MuJoCo**: High-performance sparse operations, built-in URDF converter
- **Genesis**: GPU-accelerated, native URDF/MJCF/USD support (requires genesis-world)
- **IsaacSim**: USD-based, photorealistic rendering (requires Isaac Sim)

Quick Start:
-----------
>>> from figaroh.backends import get_backend
>>>
>>> # Create backend from URDF
>>> backend = get_backend("pinocchio", model_path="robot.urdf")
>>>
>>> # Or specify backend explicitly
>>> backend = get_backend("mujoco", model_path="robot.urdf")
>>>
>>> # Use backend for dynamics computation
>>> import numpy as np
>>> q = np.zeros(backend.nq)
>>> M = backend.compute_mass_matrix(q)

Backend Selection Guide:
-----------------------
- **Research & Identification**: Use Pinocchio or MuJoCo
- **Large-Scale Parallel**: Use Genesis (GPU)
- **Sim-to-Real**: Use Isaac Sim
- **Contact Dynamics**: Use MuJoCo or Genesis
- **Optimal Control**: Use Pinocchio or MuJoCo
"""

from typing import Optional, Dict, Any
from .base import DynamicsBackend

# Import available backends
_AVAILABLE_BACKENDS = {}

# Always available: Pinocchio (FIGAROH dependency)
try:
    from .pinocchio import PinocchioBackend, PINOCCHIO_AVAILABLE

    if PINOCCHIO_AVAILABLE:
        _AVAILABLE_BACKENDS["pinocchio"] = PinocchioBackend
except ImportError:
    pass

# Optional: MuJoCo
try:
    from .mujoco import MuJoCoBackend, MUJOCO_AVAILABLE

    if MUJOCO_AVAILABLE:
        _AVAILABLE_BACKENDS["mujoco"] = MuJoCoBackend
except ImportError:
    pass

# Optional: Genesis
try:
    from .genesis import GenesisBackend

    _AVAILABLE_BACKENDS["genesis"] = GenesisBackend
except ImportError:
    pass

# Optional: Isaac Sim
try:
    from .isaacsim import IsaacSimBackend

    _AVAILABLE_BACKENDS["isaacsim"] = IsaacSimBackend
except ImportError:
    pass


def get_backend(
    backend: str = "pinocchio", model_path: Optional[str] = None, **kwargs
) -> DynamicsBackend:
    """
    Get a dynamics backend instance.

    Args:
        backend: Backend name ('pinocchio', 'mujoco', 'genesis', 'isaacsim')
        model_path: Path to robot model file
        **kwargs: Backend-specific configuration

    Returns:
        Initialized backend instance

    Raises:
        ValueError: If backend is not available
        ImportError: If backend dependencies are not installed

    Example:
        >>> backend = get_backend("mujoco", model_path="robot.urdf")
        >>> M = backend.compute_mass_matrix(q)
    """
    backend_lower = backend.lower()

    if backend_lower not in _AVAILABLE_BACKENDS:
        available = ", ".join(_AVAILABLE_BACKENDS.keys())
        raise ValueError(
            f"Backend '{backend}' not available. "
            f"Available backends: {available}. "
            f"Install missing dependencies or check backend name."
        )

    backend_class = _AVAILABLE_BACKENDS[backend_lower]

    if model_path is None:
        raise ValueError("model_path is required")

    return backend_class(model_path=model_path, **kwargs)


def list_backends() -> Dict[str, bool]:
    """
    List all backends and their availability.

    Returns:
        Dictionary mapping backend names to availability status

    Example:
        >>> list_backends()
        {'pinocchio': True, 'mujoco': True, 'genesis': False, 'isaacsim': False}
    """
    all_backends = {
        "pinocchio": "pinocchio" in _AVAILABLE_BACKENDS,
        "mujoco": "mujoco" in _AVAILABLE_BACKENDS,
        "genesis": "genesis" in _AVAILABLE_BACKENDS,
        "isaacsim": "isaacsim" in _AVAILABLE_BACKENDS,
    }
    return all_backends


def get_backend_info(backend: str) -> Dict[str, Any]:
    """
    Get information about a specific backend.

    Args:
        backend: Backend name

    Returns:
        Dictionary with backend information

    Example:
        >>> info = get_backend_info("mujoco")
        >>> print(info["description"])
    """
    info_map = {
        "pinocchio": {
            "name": "Pinocchio",
            "description": "Rigid body dynamics library with excellent URDF support",
            "formats": ["urdf"],
            "license": "BSD-2-Clause",
            "performance": "High (CPU)",
            "use_cases": ["Research", "Identification", "Control"],
            "install": "pip install pin",
        },
        "mujoco": {
            "name": "MuJoCo",
            "description": "High-performance physics engine with sparse operations",
            "formats": ["urdf", "mjcf"],
            "license": "Apache-2.0",
            "performance": "Very High (CPU)",
            "use_cases": ["Identification", "Optimal Control", "Contact"],
            "install": "pip install mujoco",
        },
        "genesis": {
            "name": "Genesis",
            "description": "GPU-accelerated universal physics engine",
            "formats": ["urdf", "mjcf", "usd"],
            "license": "Apache-2.0",
            "performance": "Extreme (GPU)",
            "use_cases": ["Large-Scale", "Parallel", "Multi-Robot"],
            "install": "pip install genesis-world",
        },
        "isaacsim": {
            "name": "Isaac Sim",
            "description": "NVIDIA's photorealistic robotics simulator",
            "formats": ["usd", "urdf"],
            "license": "Proprietary",
            "performance": "High (GPU)",
            "use_cases": ["Sim-to-Real", "RL", "Synthetic Data"],
            "install": "See NVIDIA Omniverse documentation",
        },
    }

    backend_lower = backend.lower()
    if backend_lower not in info_map:
        raise ValueError(f"Unknown backend: {backend}")

    info = info_map[backend_lower].copy()
    info["available"] = backend_lower in _AVAILABLE_BACKENDS

    return info


__all__ = [
    "DynamicsBackend",
    "get_backend",
    "list_backends",
    "get_backend_info",
]
