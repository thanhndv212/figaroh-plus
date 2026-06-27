# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sys import argv
from typing import Optional, Tuple, Union
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer


class Robot(RobotWrapper):
    """Enhanced Robot class with advanced free-flyer support and visualization utilities.

    This class extends RobotWrapper with:
    - Sophisticated free-flyer configuration and limits
    - Integrated visualization support
    - Convenient API for common robot operations
    - Enhanced error handling and validation
    """

    DEFAULT_FREEFLYER_LIMITS = (-1.0, 1.0)

    def __init__(
        self,
        robot_urdf: str,
        package_dirs: str,
        isFext: bool = False,
        freeflyer_ori: Optional[np.ndarray] = None,
        freeflyer_limits: Optional[Tuple[float, float]] = None,
    ):
        """Initialize enhanced robot model from URDF.

        Args:
            robot_urdf: Path to URDF file
            package_dirs: Package directories for mesh files
            isFext: Whether to add floating base joint
            freeflyer_ori: Optional 3x3 rotation matrix for floating base orientation
            freeflyer_limits: Optional custom limits for free-flyer (default: (-1, 1))
        """
        self.isFext = isFext
        self.robot_urdf = robot_urdf
        self._freeflyer_limits = freeflyer_limits or self.DEFAULT_FREEFLYER_LIMITS

        # Initialize robot model
        root_joint = pin.JointModelFreeFlyer() if isFext else None
        self.initFromURDF(robot_urdf, package_dirs=package_dirs, root_joint=root_joint)

        # Configure free-flyer with advanced options
        if isFext:
            self._configure_freeflyer(freeflyer_ori)

        # Set aliases for backward compatibility
        self.geom_model = self.collision_model

        # Lazy backend for backend-aware computations
        self._backend = None

    def _configure_freeflyer(self, freeflyer_ori: Optional[np.ndarray]) -> None:
        """Configure free-flyer with orientation and limits."""
        if freeflyer_ori is not None:
            self._set_freeflyer_orientation(freeflyer_ori)
        self._set_freeflyer_limits()

    def _set_freeflyer_orientation(self, orientation: np.ndarray) -> None:
        """Set free-flyer orientation with validation."""
        if orientation.shape != (3, 3):
            raise ValueError("Orientation must be a 3x3 rotation matrix")

        joint_id = self.model.getJointId("root_joint")
        self.model.jointPlacements[joint_id].rotation = orientation

    def _set_freeflyer_limits(self) -> None:
        """Set sophisticated free-flyer limits."""
        lb, ub = self._freeflyer_limits

        # Set position limits (first 3 DOFs)
        self.model.lowerPositionLimit[:3] = lb
        self.model.upperPositionLimit[:3] = ub

        # Set quaternion limits (next 4 DOFs) - normalized quaternion bounds
        self.model.lowerPositionLimit[3:7] = -1.0
        self.model.upperPositionLimit[3:7] = 1.0

        # Recreate data structure
        self.data = self.model.createData()

    def update_freeflyer_limits(self, limits: Tuple[float, float]) -> None:
        """Update free-flyer limits dynamically."""
        if not self.isFext:
            raise ValueError("Robot does not have a free-flyer joint")

        self._freeflyer_limits = limits
        self._set_freeflyer_limits()

    def display_q0(
        self, visualizer_type: Optional[str] = None, q: Optional[np.ndarray] = None
    ) -> None:
        """Enhanced visualization with configuration options."""
        visualizer_class = self._get_visualizer_class(visualizer_type)

        if visualizer_class:
            self.setVisualizer(visualizer_class())
            self.initViewer()
            self.loadViewerModel(self.robot_urdf)

            # Display specified configuration or initial configuration
            config = q if q is not None else self.q0
            self.display(config)

    def _get_visualizer_class(self, visualizer_type: Optional[str]):
        """Smart visualizer selection with fallbacks."""
        visualizer_map = {
            "gepetto": GepettoVisualizer,
            "meshcat": MeshcatVisualizer,
            "g": GepettoVisualizer,
            "m": MeshcatVisualizer,
        }

        if visualizer_type:
            return visualizer_map.get(visualizer_type.lower())

        # Auto-detect from command line
        if len(argv) > 1:
            return visualizer_map.get(argv[1])

        return None

    # Enhanced properties with better documentation
    @property
    def num_joints(self) -> int:
        """Number of actuated joints (excluding universe and free-flyer)."""
        base_joints = 1 + (1 if self.isFext else 0)  # universe + optional free-flyer
        return self.model.njoints - base_joints

    @property
    def actuated_joint_names(self) -> Tuple[str, ...]:
        """Names of actuated joints only."""
        start_idx = 2 if self.isFext else 1  # Skip universe and optional free-flyer
        return tuple(self.model.names[start_idx:])

    @property
    def freeflyer_limits(self) -> Tuple[float, float]:
        """Current free-flyer position limits."""
        return self._freeflyer_limits

    @property
    def backend(self):
        """Lazily-created DynamicsBackend wrapping this robot's model.

        Returns a PinocchioBackend that shares the same pin.Model and pin.Data
        as this Robot instance — no URDF re-loading, no model duplication.
        Used by RegressorBuilder and other tools to route dynamics computation
        through the backend abstraction.

        Returns:
            PinocchioBackend instance
        """
        if self._backend is None:
            from ..backends.pinocchio import PinocchioBackend

            self._backend = PinocchioBackend.from_model(self.model, self.data)
        return self._backend

    def get_configuration_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get configuration space bounds with proper handling of quaternions."""
        lb, ub = (
            self.model.lowerPositionLimit.copy(),
            self.model.upperPositionLimit.copy(),
        )

        if self.isFext:
            # Ensure quaternion bounds are properly set
            lb[3:7] = -1.0
            ub[3:7] = 1.0

        return lb, ub

    def validate_configuration(self, q: np.ndarray) -> bool:
        """Validate if configuration is within bounds and constraints."""
        if len(q) != self.model.nq:
            return False

        lb, ub = self.get_configuration_bounds()

        # Check bounds
        if not np.all((q >= lb) & (q <= ub)):
            return False

        # Check quaternion normalization for free-flyer
        if self.isFext:
            quat = q[3:7]
            if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6):
                return False

        return True

    def __repr__(self) -> str:
        """Enhanced string representation."""
        return (
            f"Robot(urdf='{self.robot_urdf}', "
            f"actuated_joints={self.num_joints}, "
            f"total_dofs={self.model.nv}, "
            f"free_flyer={self.isFext})"
        )


# Simple factory function that leverages load_robot.py
def create_robot(
    robot_urdf: str,
    package_dirs: Optional[str] = None,
    loader: str = "figaroh",
    enhanced_features: bool = True,
    **kwargs,
) -> Union[Robot, RobotWrapper]:
    """Factory function to create robots with optional enhanced features.

    Args:
        robot_urdf: Robot identifier or URDF path
        package_dirs: Package directories
        loader: Loader type from load_robot.py
        enhanced_features: If True, returns Robot class with enhanced features
        **kwargs: Additional arguments for loaders

    Returns:
        Robot instance (if enhanced_features=True) or RobotWrapper
    """
    if enhanced_features and loader == "figaroh":
        # Use enhanced Robot class
        from .load_robot import _prepare_package_dirs, _validate_urdf_exists

        isFext = kwargs.get("isFext", False)
        robot_pkg = kwargs.get("robot_pkg")

        package_dirs = _prepare_package_dirs(robot_urdf, package_dirs, robot_pkg)
        _validate_urdf_exists(robot_urdf)

        return Robot(
            robot_urdf,
            package_dirs=package_dirs,
            isFext=isFext,
            freeflyer_ori=kwargs.get("freeflyer_ori"),
            freeflyer_limits=kwargs.get("freeflyer_limits"),
        )
    else:
        # Use load_robot.py for other loaders
        from .load_robot import load_robot

        return load_robot(robot_urdf, package_dirs, loader=loader, **kwargs)


# Import utilities from load_robot
from .load_robot import load_robot, get_available_loaders, list_available_robots

__all__ = [
    "Robot",
    "create_robot",
    "load_robot",
    "get_available_loaders",
    "list_available_robots",
]
