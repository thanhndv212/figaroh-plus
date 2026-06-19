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

"""Enhanced robot visualization utilities with better performance."""

from typing import List, Optional, Union
import numpy as np
import pinocchio as pin
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for robot visualization."""

    com_color: List[float] = None
    axes_color: List[float] = None
    force_color: List[float] = None
    bbox_color: List[float] = None
    scale_factor: float = 1.0

    def __post_init__(self):
        if self.com_color is None:
            self.com_color = [1.0, 0.0, 0.0, 1.0]  # Red
        if self.axes_color is None:
            self.axes_color = [1.0, 0.0, 0.0, 1.0]  # Red
        if self.force_color is None:
            self.force_color = [1.0, 1.0, 0.0, 1.0]  # Yellow
        if self.bbox_color is None:
            self.bbox_color = [0.5, 0.0, 0.5, 0.5]  # Purple, transparent


class RobotVisualizer:
    """Enhanced robot visualizer with better organization."""

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        viz,
        config: Optional[VisualizationConfig] = None,
    ):
        self.model = model
        self.data = data
        self.viz = viz
        self.config = config or VisualizationConfig()

        # Cache frequently used values
        self.joint_names = model.names
        self.nv = model.nv

    def update_kinematics(self, q: np.ndarray) -> None:
        """Update robot kinematics efficiently."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.centerOfMass(self.model, self.data, q, True)
        pin.computeSubtreeMasses(self.model, self.data)

    def display_com(self, q: np.ndarray, frame_indices: List[int]) -> None:
        """Display center of mass positions with better naming."""
        self.update_kinematics(q)

        for i, frame_idx in enumerate(frame_indices[:-1]):  # Exclude last frame
            next_frame_idx = frame_indices[i + 1]

            # Calculate link properties
            link_length = np.linalg.norm(
                self.data.oMf[next_frame_idx].translation
                - self.data.oMf[frame_idx].translation
            )

            # Mass-based radius scaling
            mass_ratio = (
                self.data.mass[i] / self.data.mass[0] if self.data.mass[0] > 0 else 0.1
            )
            radius = link_length * mass_ratio * self.config.scale_factor

            # COM placement
            placement = self.data.oMf[frame_idx].copy()
            parent_id = self.model.frames[frame_idx].parent - 1
            if parent_id >= 0:
                placement.translation = self.data.com[parent_id]

            # Create and place sphere
            sphere_name = f"world/com_sphere_{i}"
            self.viz.viewer.gui.addSphere(sphere_name, radius, self.config.com_color)
            self._apply_placement(sphere_name, placement)

    def display_axes(
        self, q: np.ndarray, axis_length: float = 0.15, axis_radius: float = 0.01
    ) -> None:
        """Display coordinate axes for joints."""
        self.update_kinematics(q)

        for i, joint_name in enumerate(self.joint_names):
            joint_id = self.model.getJointId(joint_name)
            axis_name = f"world/axis_{joint_name}_{i}"

            # Create axis visualization
            self.viz.viewer.gui.addXYZaxis(
                axis_name,
                self.config.axes_color,
                axis_radius * self.config.scale_factor,
                axis_length * self.config.scale_factor,
            )

            # Apply joint placement
            self._apply_placement(axis_name, self.data.oMi[joint_id])

    def display_force(
        self,
        force: pin.Force,
        placement: pin.SE3,
        scale: float = 1e-3,
        name: str = "force_vector",
    ) -> None:
        """Display force vector with better scaling."""
        # Transform force to world frame
        world_force = force.se3Action(placement)
        force_magnitude = np.linalg.norm(world_force.linear)

        if force_magnitude < 1e-10:  # Skip negligible forces
            return

        # Calculate arrow properties
        arrow_length = force_magnitude * scale * self.config.scale_factor
        arrow_radius = arrow_length * 0.05

        # Align arrow with force direction
        force_direction = world_force.linear / force_magnitude
        rotation = self._rotation_from_vectors([1, 0, 0], force_direction)

        arrow_placement = placement.copy()
        arrow_placement.rotation = rotation

        # Create and place arrow
        arrow_name = f"world/{name}_arrow"
        self.viz.viewer.gui.addArrow(
            arrow_name, arrow_radius, arrow_length, self.config.force_color
        )
        self._apply_placement(arrow_name, arrow_placement)

    def display_bounding_boxes(
        self,
        q: np.ndarray,
        com_min: np.ndarray,
        com_max: np.ndarray,
        frame_indices: List[int],
    ) -> None:
        """Display COM bounding boxes for optimization visualization."""
        self.update_kinematics(q)

        for i, frame_idx in enumerate(frame_indices):
            # Get COM bounds for this segment
            bounds_start = 3 * i
            min_bounds = com_min[bounds_start : bounds_start + 3]
            max_bounds = com_max[bounds_start : bounds_start + 3]

            # Calculate box dimensions
            box_size = (max_bounds - min_bounds) * self.config.scale_factor

            # Box placement at COM location
            placement = self.data.oMf[frame_idx].copy()
            parent_id = self.model.frames[frame_idx].parent - 1
            if parent_id >= 0:
                placement.translation = self.data.com[parent_id]

            # Create and place box
            box_name = f"world/com_bbox_{i}"
            self.viz.viewer.gui.addBox(
                box_name, box_size[0], box_size[1], box_size[2], self.config.bbox_color
            )
            self._apply_placement(box_name, placement)

    def display_joints(self, q: np.ndarray) -> None:
        """Display joint frames efficiently."""
        self.update_kinematics(q)

        for i in range(self.nv):
            joint_placement = self.data.oMi[i]
            joint_name = f"world/joint_frame_{i}"

            # Create joint frame visualization
            self.viz.viewer.gui.addXYZaxis(
                joint_name,
                self.config.axes_color,
                0.01 * self.config.scale_factor,
                0.15 * self.config.scale_factor,
            )

            # Apply configuration
            self._apply_placement(joint_name, joint_placement)

    def _apply_placement(self, name: str, placement: pin.SE3) -> None:
        """Apply SE3 placement to visualization object."""
        self.viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(placement))
        self.viz.viewer.gui.refresh()

    def _rotation_from_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Calculate rotation matrix aligning vec1 to vec2."""
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)

        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)

        if s < 1e-10:  # Vectors are parallel
            return np.eye(3) if c > 0 else -np.eye(3)

        # Rodrigues' rotation formula
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


# Backward compatibility functions
def place(viz, name: str, M: pin.SE3) -> None:
    """Legacy placement function."""
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(M))


def display_COM(
    model: pin.Model, data: pin.Data, viz, q: np.ndarray, IDX: list
) -> None:
    """Legacy COM display function."""
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_com(q, IDX)


def display_axes(model: pin.Model, data: pin.Data, viz, q: np.ndarray) -> None:
    """Legacy axes display function."""
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_axes(q)


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Legacy rotation function."""
    visualizer = RobotVisualizer(None, None, None)
    return visualizer._rotation_from_vectors(vec1, vec2)


def display_force(viz, phi: pin.Force, M_se3: pin.SE3) -> None:
    """Legacy force display function."""
    visualizer = RobotVisualizer(None, None, viz)
    visualizer.display_force(phi, M_se3)


def display_bounding_boxes(
    viz,
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    COM_min: np.ndarray,
    COM_max: np.ndarray,
    IDX: list,
) -> None:
    """Legacy bounding box display function."""
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_bounding_boxes(q, COM_min, COM_max, IDX)


def display_joints(viz, model: pin.Model, data: pin.Data, q: np.ndarray) -> None:
    """Legacy joint display function."""
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_joints(q)
