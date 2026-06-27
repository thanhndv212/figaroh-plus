"""Tests for robot visualization functionality."""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from figaroh.tools.robotvisualization import (
        VisualizationConfig,
        RobotVisualizer,
        place,
        display_COM,
        display_axes,
        rotation_matrix_from_vectors,
        display_force,
        display_bounding_boxes,
        display_joints,
    )
    import pinocchio as pin
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)


class TestVisualizationConfig:
    """Test the visualization configuration dataclass."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = VisualizationConfig()

        # Check default colors are set
        assert config.com_color == [1.0, 0.0, 0.0, 1.0]  # Red
        assert config.axes_color == [1.0, 0.0, 0.0, 1.0]  # Red
        assert config.force_color == [1.0, 1.0, 0.0, 1.0]  # Yellow
        assert config.bbox_color == [0.5, 0.0, 0.5, 0.5]  # Purple, transparent
        assert config.scale_factor == 1.0

    def test_custom_initialization(self):
        """Test custom configuration values."""
        custom_colors = {
            "com_color": [0.0, 1.0, 0.0, 1.0],
            "axes_color": [0.0, 0.0, 1.0, 1.0],
            "force_color": [1.0, 0.0, 1.0, 1.0],
            "bbox_color": [0.0, 1.0, 1.0, 0.3],
            "scale_factor": 2.0,
        }

        config = VisualizationConfig(**custom_colors)

        assert config.com_color == custom_colors["com_color"]
        assert config.axes_color == custom_colors["axes_color"]
        assert config.force_color == custom_colors["force_color"]
        assert config.bbox_color == custom_colors["bbox_color"]
        assert config.scale_factor == 2.0

    def test_partial_initialization(self):
        """Test initialization with some custom values."""
        config = VisualizationConfig(scale_factor=0.5)

        # Custom value should be set
        assert config.scale_factor == 0.5

        # Defaults should still be applied
        assert config.com_color == [1.0, 0.0, 0.0, 1.0]
        assert config.axes_color == [1.0, 0.0, 0.0, 1.0]


class TestRobotVisualizer:
    """Test the RobotVisualizer class."""

    @pytest.fixture
    def mock_robot_components(self):
        """Create mock robot model and data components."""
        # Mock model
        model = Mock()
        model.names = ["universe", "joint1", "joint2", "joint3"]
        model.nv = 3
        model.getJointId = Mock(side_effect=lambda name: model.names.index(name))

        # Mock frames
        frame1 = Mock()
        frame1.parent = 1
        frame2 = Mock()
        frame2.parent = 2
        model.frames = [None, frame1, frame2]  # Index 0 unused

        # Mock data
        data = Mock()
        data.oMf = [Mock() for _ in range(5)]  # Frame placements
        data.oMi = [Mock() for _ in range(5)]  # Joint placements
        data.com = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        data.mass = [1.0, 0.5, 0.3]

        # Set mock placement translations
        for i, placement in enumerate(data.oMf):
            placement.translation = np.array([i * 0.1, i * 0.2, i * 0.3])
            placement.copy.return_value = placement

        for i, placement in enumerate(data.oMi):
            placement.translation = np.array([i * 0.15, i * 0.25, i * 0.35])

        # Mock viz
        viz = Mock()
        viz.viewer.gui.addSphere = Mock()
        viz.viewer.gui.addXYZaxis = Mock()
        viz.viewer.gui.addArrow = Mock()
        viz.viewer.gui.addBox = Mock()
        viz.viewer.gui.applyConfiguration = Mock()
        viz.viewer.gui.refresh = Mock()

        return model, data, viz

    @pytest.fixture
    def robot_visualizer(self, mock_robot_components):
        """Create a RobotVisualizer instance."""
        model, data, viz = mock_robot_components
        return RobotVisualizer(model, data, viz)

    def test_initialization(self, mock_robot_components):
        """Test RobotVisualizer initialization."""
        model, data, viz = mock_robot_components
        config = VisualizationConfig(scale_factor=2.0)

        visualizer = RobotVisualizer(model, data, viz, config)

        assert visualizer.model == model
        assert visualizer.data == data
        assert visualizer.viz == viz
        assert visualizer.config.scale_factor == 2.0
        assert visualizer.joint_names == model.names
        assert visualizer.nv == model.nv

    @patch("figaroh.tools.robotvisualization.pin")
    def test_update_kinematics(self, mock_pin, robot_visualizer):
        """Test kinematic update."""
        q = np.array([0.1, 0.2, 0.3])

        robot_visualizer.update_kinematics(q)

        # Check that pinocchio functions were called
        mock_pin.forwardKinematics.assert_called_once_with(
            robot_visualizer.model, robot_visualizer.data, q
        )
        mock_pin.updateFramePlacements.assert_called_once()
        mock_pin.centerOfMass.assert_called_once()
        mock_pin.computeSubtreeMasses.assert_called_once()

    @patch("figaroh.tools.robotvisualization.pin")
    def test_display_com(self, mock_pin, robot_visualizer):
        """Test COM display functionality."""
        q = np.array([0.1, 0.2, 0.3])
        frame_indices = [1, 2, 3]

        robot_visualizer.display_com(q, frame_indices)

        # Check that spheres were added (should be len(frame_indices) - 1)
        expected_calls = len(frame_indices) - 1
        assert robot_visualizer.viz.viewer.gui.addSphere.call_count == expected_calls
        assert (
            robot_visualizer.viz.viewer.gui.applyConfiguration.call_count
            == expected_calls
        )

    @patch("figaroh.tools.robotvisualization.pin")
    def test_display_axes(self, mock_pin, robot_visualizer):
        """Test axes display functionality."""
        q = np.array([0.1, 0.2, 0.3])

        robot_visualizer.display_axes(q)

        # Should create axes for each joint name
        expected_calls = len(robot_visualizer.joint_names)
        assert robot_visualizer.viz.viewer.gui.addXYZaxis.call_count == expected_calls
        assert (
            robot_visualizer.viz.viewer.gui.applyConfiguration.call_count
            == expected_calls
        )

    @patch("figaroh.tools.robotvisualization.pin")
    def test_display_force(self, mock_pin, robot_visualizer):
        """Test force display functionality."""
        # Mock force
        force = Mock()
        force.se3Action.return_value = Mock()
        force.se3Action.return_value.linear = np.array([1.0, 0.0, 0.0])

        # Mock placement
        placement = Mock()
        placement.copy.return_value = placement
        placement.rotation = np.eye(3)

        robot_visualizer.display_force(force, placement)

        # Should create an arrow
        robot_visualizer.viz.viewer.gui.addArrow.assert_called_once()
        robot_visualizer.viz.viewer.gui.applyConfiguration.assert_called_once()

    @patch("figaroh.tools.robotvisualization.pin")
    def test_display_force_negligible(self, mock_pin, robot_visualizer):
        """Test that negligible forces are skipped."""
        # Mock negligible force
        force = Mock()
        force.se3Action.return_value = Mock()
        force.se3Action.return_value.linear = np.array([1e-12, 0.0, 0.0])

        placement = Mock()

        robot_visualizer.display_force(force, placement)

        # Should not create an arrow for negligible force
        robot_visualizer.viz.viewer.gui.addArrow.assert_not_called()

    @patch("figaroh.tools.robotvisualization.pin")
    def test_display_bounding_boxes(self, mock_pin, robot_visualizer):
        """Test bounding box display."""
        q = np.array([0.1, 0.2, 0.3])
        com_min = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
        com_max = np.array([0.2, 0.2, 0.2, 0.3, 0.3, 0.3])
        frame_indices = [1, 2]

        robot_visualizer.display_bounding_boxes(q, com_min, com_max, frame_indices)

        # Should create boxes for each frame
        expected_calls = len(frame_indices)
        assert robot_visualizer.viz.viewer.gui.addBox.call_count == expected_calls
        assert (
            robot_visualizer.viz.viewer.gui.applyConfiguration.call_count
            == expected_calls
        )

    @patch("figaroh.tools.robotvisualization.pin")
    def test_display_joints(self, mock_pin, robot_visualizer):
        """Test joint display functionality."""
        q = np.array([0.1, 0.2, 0.3])

        robot_visualizer.display_joints(q)

        # Should create axes for each joint
        expected_calls = robot_visualizer.nv
        assert robot_visualizer.viz.viewer.gui.addXYZaxis.call_count == expected_calls
        assert (
            robot_visualizer.viz.viewer.gui.applyConfiguration.call_count
            == expected_calls
        )

    def test_rotation_from_vectors(self, robot_visualizer):
        """Test rotation matrix calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        rotation = robot_visualizer._rotation_from_vectors(vec1, vec2)

        # Should return a 3x3 rotation matrix
        assert rotation.shape == (3, 3)

        # Check that it's a proper rotation matrix
        assert np.allclose(np.linalg.det(rotation), 1.0)
        assert np.allclose(rotation @ rotation.T, np.eye(3))

    def test_rotation_from_parallel_vectors(self, robot_visualizer):
        """Test rotation with parallel vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([2.0, 0.0, 0.0])  # Same direction

        rotation = robot_visualizer._rotation_from_vectors(vec1, vec2)

        # Should return identity matrix
        np.testing.assert_allclose(rotation, np.eye(3))

    def test_rotation_from_opposite_vectors(self, robot_visualizer):
        """Test rotation with opposite vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])  # Opposite direction

        rotation = robot_visualizer._rotation_from_vectors(vec1, vec2)

        # Should return negative identity
        np.testing.assert_allclose(rotation, -np.eye(3))


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions."""

    @pytest.fixture
    def mock_viz(self):
        """Create mock visualizer."""
        viz = Mock()
        viz.viewer.gui.applyConfiguration = Mock()
        return viz

    @pytest.fixture
    def mock_se3(self):
        """Create mock SE3 transform."""
        return Mock()

    def test_place_function(self, mock_viz, mock_se3):
        """Test legacy place function."""
        with patch(
            "figaroh.tools.robotvisualization.pin.SE3ToXYZQUATtuple"
        ) as mock_convert:
            mock_convert.return_value = [0, 0, 0, 0, 0, 0, 1]

            place(mock_viz, "test_object", mock_se3)

            mock_viz.viewer.gui.applyConfiguration.assert_called_once_with(
                "test_object", [0, 0, 0, 0, 0, 0, 1]
            )

    @patch("figaroh.tools.robotvisualization.RobotVisualizer")
    def test_display_com_legacy(self, mock_visualizer_class):
        """Test legacy COM display function."""
        mock_model = Mock()
        mock_data = Mock()
        mock_viz = Mock()
        q = np.array([0.1, 0.2, 0.3])
        idx = [1, 2, 3]

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        display_COM(mock_model, mock_data, mock_viz, q, idx)

        mock_visualizer_class.assert_called_once_with(mock_model, mock_data, mock_viz)
        mock_visualizer.display_com.assert_called_once_with(q, idx)

    @patch("figaroh.tools.robotvisualization.RobotVisualizer")
    def test_display_axes_legacy(self, mock_visualizer_class):
        """Test legacy axes display function."""
        mock_model = Mock()
        mock_data = Mock()
        mock_viz = Mock()
        q = np.array([0.1, 0.2, 0.3])

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        display_axes(mock_model, mock_data, mock_viz, q)

        mock_visualizer_class.assert_called_once_with(mock_model, mock_data, mock_viz)
        mock_visualizer.display_axes.assert_called_once_with(q)

    @patch("figaroh.tools.robotvisualization.RobotVisualizer")
    def test_rotation_matrix_from_vectors_legacy(self, mock_visualizer_class):
        """Test legacy rotation function."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        mock_visualizer = Mock()
        mock_visualizer._rotation_from_vectors.return_value = np.eye(3)
        mock_visualizer_class.return_value = mock_visualizer

        result = rotation_matrix_from_vectors(vec1, vec2)

        mock_visualizer_class.assert_called_once_with(None, None, None)
        mock_visualizer._rotation_from_vectors.assert_called_once_with(vec1, vec2)

    @patch("figaroh.tools.robotvisualization.RobotVisualizer")
    def test_display_force_legacy(self, mock_visualizer_class):
        """Test legacy force display function."""
        mock_viz = Mock()
        mock_force = Mock()
        mock_se3 = Mock()

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        display_force(mock_viz, mock_force, mock_se3)

        mock_visualizer_class.assert_called_once_with(None, None, mock_viz)
        mock_visualizer.display_force.assert_called_once_with(mock_force, mock_se3)

    @patch("figaroh.tools.robotvisualization.RobotVisualizer")
    def test_display_bounding_boxes_legacy(self, mock_visualizer_class):
        """Test legacy bounding box display function."""
        mock_viz = Mock()
        mock_model = Mock()
        mock_data = Mock()
        q = np.array([0.1, 0.2, 0.3])
        com_min = np.array([0.0, 0.0, 0.0])
        com_max = np.array([0.2, 0.2, 0.2])
        idx = [1, 2]

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        display_bounding_boxes(
            mock_viz, mock_model, mock_data, q, com_min, com_max, idx
        )

        mock_visualizer_class.assert_called_once_with(mock_model, mock_data, mock_viz)
        mock_visualizer.display_bounding_boxes.assert_called_once_with(
            q, com_min, com_max, idx
        )

    @patch("figaroh.tools.robotvisualization.RobotVisualizer")
    def test_display_joints_legacy(self, mock_visualizer_class):
        """Test legacy joint display function."""
        mock_viz = Mock()
        mock_model = Mock()
        mock_data = Mock()
        q = np.array([0.1, 0.2, 0.3])

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        display_joints(mock_viz, mock_model, mock_data, q)

        mock_visualizer_class.assert_called_once_with(mock_model, mock_data, mock_viz)
        mock_visualizer.display_joints.assert_called_once_with(q)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_visualizer_with_none_config(self):
        """Test visualizer with None config."""
        model = Mock()
        model.names = ["joint1"]
        model.nv = 1
        data = Mock()
        viz = Mock()

        visualizer = RobotVisualizer(model, data, viz, None)

        # Should create default config
        assert isinstance(visualizer.config, VisualizationConfig)
        assert visualizer.config.scale_factor == 1.0

    def test_config_with_none_colors(self):
        """Test config initialization with None colors."""
        config = VisualizationConfig(
            com_color=None, axes_color=None, force_color=None, bbox_color=None
        )

        # Should set defaults in __post_init__
        assert config.com_color == [1.0, 0.0, 0.0, 1.0]
        assert config.axes_color == [1.0, 0.0, 0.0, 1.0]
        assert config.force_color == [1.0, 1.0, 0.0, 1.0]
        assert config.bbox_color == [0.5, 0.0, 0.5, 0.5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
