"""
FIGAROH Dynamics Backend System

Abstract base class for pluggable dynamics computation backends.
This enables FIGAROH to work with multiple simulators (Pinocchio, MuJoCo, Genesis, Isaac Sim)
while maintaining consistent algorithms and APIs.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class DynamicsBackend(ABC):
    """
    Abstract base class for dynamics computation backends.

    This interface defines the minimum set of dynamics computations required
    for FIGAROH's calibration and identification algorithms. Implementations
    wrap specific simulators (Pinocchio, MuJoCo, Genesis, Isaac Sim).

    Design Philosophy:
    - Abstract interface, concrete implementations
    - Consistent algorithm behavior across backends
    - Performance optimization in implementations
    - No algorithm changes required when switching backends
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initialize dynamics backend with robot model.

        Args:
            model_path: Path to robot model file (format depends on backend)
            **kwargs: Backend-specific configuration options
        """
        self._model_path = model_path
        self._config = kwargs

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

    @abstractmethod
    def compute_coriolis_matrix(
        self, q: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
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

    # Properties

    @property
    @abstractmethod
    def nq(self) -> int:
        """Number of position variables (configuration space dimension)."""
        pass

    @property
    @abstractmethod
    def nv(self) -> int:
        """Number of velocity variables (tangent space dimension)."""
        pass

    @property
    @abstractmethod
    def model_format(self) -> str:
        """
        Backend model format.

        Returns:
            Format name: 'urdf', 'mjcf', 'usd', etc.
        """
        pass

    @property
    def model_path(self) -> str:
        """Path to robot model file."""
        return self._model_path

    @property
    def config(self) -> Dict[str, Any]:
        """Backend configuration options."""
        return self._config

    # Optional methods (can be overridden for better performance)

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

    def get_joint_names(self) -> list:
        """
        Get list of joint names.

        Returns:
            List of joint names in order
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_joint_names"
        )

    def get_frame_names(self) -> list:
        """
        Get list of frame names.

        Returns:
            List of frame names available for FK/Jacobian
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_frame_names"
        )

    # Context manager support

    def __enter__(self):
        """Enter context manager (for resource management)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (for cleanup)."""
        pass

    # String representation

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_path='{self.model_path}', "
            f"format='{self.model_format}', "
            f"nq={self.nq}, nv={self.nv})"
        )


__all__ = ["DynamicsBackend"]
