"""
Pinocchio Dynamics Backend for FIGAROH

Default backend using Pinocchio's rigid body dynamics library.
Wraps existing Pinocchio usage into the DynamicsBackend interface.

Features:
- Excellent URDF support
- CPU-optimized dynamics algorithms
- Full frame and Jacobian support
- Lie group operations (difference, integrate)
"""

import numpy as np
from typing import Dict, Any
from .base import DynamicsBackend

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    pin = None


class PinocchioBackend(DynamicsBackend):
    """
    Pinocchio dynamics backend for FIGAROH (default).

    This backend leverages Pinocchio's rigid body dynamics algorithms
    for efficient computation of mass matrices, Coriolis effects,
    gravity, forward kinematics, Jacobians, and the regressor matrix.

    Pinocchio is the primary dependency of FIGAROH and provides the most
    complete URDF support.

    Example:
        >>> backend = PinocchioBackend(model_path="robot.urdf")
        >>> M = backend.compute_mass_matrix(q)
    """

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

    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute mass matrix using Pinocchio's CRBA (Composite Rigid Body Algorithm).

        Args:
            q: Joint positions [nq]

        Returns:
            M: Mass matrix [nv x nv], symmetric positive definite
        """
        return pin.crba(self.model, self.data, q).copy()

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

    def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity effects vector using Pinocchio's generalized gravity.

        Args:
            q: Joint positions [nq]

        Returns:
            g: Gravity vector [nv]
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

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

    # Properties

    @property
    def nq(self) -> int:
        """Number of position variables (configuration space dimension)."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Number of velocity variables (tangent space dimension)."""
        return self.model.nv

    @property
    def model_format(self) -> str:
        """Model format (URDF)."""
        return "urdf"

    # Optional methods

    def get_joint_names(self) -> list:
        """
        Get list of joint names.

        Returns:
            List of joint names in order (skipping universe)
        """
        return list(self.model.names[1:])

    def get_frame_names(self) -> list:
        """
        Get list of frame names.

        Returns:
            List of frame names available for FK/Jacobian (skipping universe)
        """
        return [f.name for f in self.model.frames[1:]]

    def get_inertias(self) -> list:
        """
        Get per-body inertia objects.

        Returns:
            List of inertia objects (includes universe at index 0 with zero inertia)
        """
        return list(self.model.inertias)

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

    def random_configuration(self) -> np.ndarray:
        """
        Generate a random configuration within joint limits.

        Returns:
            Random configuration [nq]
        """
        return pin.randomConfiguration(self.model)

    def get_model_object(self) -> Any:
        """
        Escape hatch: return the underlying Pinocchio model.

        Returns:
            pin.Model object
        """
        return self.model


__all__ = ["PinocchioBackend"]
