"""Tests for robot collision detection functionality."""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from figaroh.tools.robotcollisions import CollisionManager, CollisionWrapper
    import pinocchio as pin
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)


class TestCollisionManager:
    """Test the CollisionManager class."""

    @pytest.fixture
    def mock_robot(self):
        """Create a mock robot for testing."""
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()

        # Mock geometry model
        geom_model = Mock()
        geom_model.createData.return_value = Mock()
        geom_model.addAllCollisionPairs = Mock()
        geom_model.collisionPairs = [Mock(), Mock(), Mock()]

        # Mock geometry objects
        geom_obj1 = Mock()
        geom_obj1.name = "link1_geom"
        geom_obj2 = Mock()
        geom_obj2.name = "link2_geom"
        geom_model.geometryObjects = [geom_obj1, geom_obj2]

        robot.geom_model = geom_model

        return robot

    @pytest.fixture
    def mock_viz(self):
        """Create a mock visualizer."""
        viz = Mock()
        viz.viewer.gui.addSphere = Mock()
        viz.viewer.gui.applyConfiguration = Mock()
        viz.viewer.gui.deleteNode = Mock()
        return viz

    @pytest.fixture
    def collision_manager(self, mock_robot, mock_viz):
        """Create a CollisionManager instance."""
        return CollisionManager(mock_robot, viz=mock_viz)

    def test_initialization_with_robot(self, mock_robot):
        """Test CollisionManager initialization with robot."""
        manager = CollisionManager(mock_robot)

        assert manager.robot == mock_robot
        assert manager.model == mock_robot.model
        assert manager.geom_model == mock_robot.geom_model
        assert manager.geom_data is not None
        assert manager._max_patches == 10
        assert manager._vis_cache == {}

    def test_initialization_with_explicit_geom_models(self, mock_robot):
        """Test initialization with explicit geometry models."""
        geom_model = Mock()
        geom_data = Mock()

        manager = CollisionManager(mock_robot, geom_model, geom_data)

        assert manager.geom_model == geom_model
        assert manager.geom_data == geom_data

    def test_initialization_without_geom_model(self):
        """Test initialization without geometry model."""
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()
        # Remove geom_model attribute
        if hasattr(robot, "geom_model"):
            delattr(robot, "geom_model")

        manager = CollisionManager(robot)

        assert manager.geom_model is None
        assert manager.geom_data is None

    def test_setup_collision_pairs_no_geom_model(self, collision_manager):
        """Test setup collision pairs without geometry model."""
        collision_manager.geom_model = None

        with pytest.raises(ValueError, match="Geometry model not available"):
            collision_manager.setup_collision_pairs()

    def test_setup_collision_pairs_success(self, collision_manager):
        """Test successful collision pairs setup."""
        collision_manager.setup_collision_pairs()

        collision_manager.geom_model.addAllCollisionPairs.assert_called_once()

    @patch("figaroh.tools.robotcollisions.pin.removeCollisionPairs")
    @patch("os.path.exists")
    def test_setup_collision_pairs_with_srdf(
        self, mock_exists, mock_remove, collision_manager
    ):
        """Test collision pairs setup with SRDF filtering."""
        mock_exists.return_value = True
        srdf_path = "/path/to/robot.srdf"

        collision_manager.setup_collision_pairs(srdf_path)

        collision_manager.geom_model.addAllCollisionPairs.assert_called_once()
        mock_remove.assert_called_once_with(
            collision_manager.model, collision_manager.geom_model, srdf_path
        )

    @patch("figaroh.tools.robotcollisions.pin.updateGeometryPlacements")
    @patch("figaroh.tools.robotcollisions.pin.computeCollisions")
    def test_check_collisions_with_update(
        self, mock_compute, mock_update, collision_manager
    ):
        """Test collision checking with geometry update."""
        q = np.array([0.1, 0.2, 0.3])
        mock_compute.return_value = True

        result = collision_manager.check_collisions(q, update_geometry=True)

        assert result is True
        mock_update.assert_called_once()
        mock_compute.assert_called_once()

    @patch("figaroh.tools.robotcollisions.pin.computeCollisions")
    def test_check_collisions_without_update(self, mock_compute, collision_manager):
        """Test collision checking without geometry update."""
        q = np.array([0.1, 0.2, 0.3])
        mock_compute.return_value = False

        result = collision_manager.check_collisions(q, update_geometry=False)

        assert result is False
        mock_compute.assert_called_once()

    def test_check_collisions_no_geom_model(self, collision_manager):
        """Test collision checking without geometry model."""
        collision_manager.geom_model = None

        result = collision_manager.check_collisions(np.array([0.1, 0.2, 0.3]))

        assert result is False

    def test_get_collision_details(self, collision_manager):
        """Test getting collision details."""
        # Mock collision results
        result1 = Mock()
        result1.isCollision.return_value = True
        result2 = Mock()
        result2.isCollision.return_value = False
        result3 = Mock()
        result3.isCollision.return_value = True

        collision_manager.geom_data.collisionResults = [result1, result2, result3]

        details = collision_manager.get_collision_details()

        # Should return only colliding pairs (indices 0 and 2)
        assert len(details) == 2
        assert details[0][0] == 0  # First collision index
        assert details[1][0] == 2  # Third collision index

    def test_get_collision_details_no_geom_data(self, collision_manager):
        """Test getting collision details without geometry data."""
        collision_manager.geom_data = None

        details = collision_manager.get_collision_details()

        assert details == []

    def test_get_collision_distances(self, collision_manager):
        """Test getting collision distances."""
        # Mock distance results
        dist1 = Mock()
        dist1.min_distance = 0.01
        dist2 = Mock()
        dist2.min_distance = 0.02

        collision_manager.geom_data.distanceResults = [dist1, dist2]

        # Mock collision details
        collision_details = [(0, None, None), (1, None, None)]

        distances = collision_manager.get_collision_distances(collision_details)

        expected = np.array([0.01, 0.02])
        np.testing.assert_array_equal(distances, expected)

    def test_get_collision_distances_no_details(self, collision_manager):
        """Test getting distances with no collision details."""
        distances = collision_manager.get_collision_distances([])

        assert len(distances) == 0

    @patch("figaroh.tools.robotcollisions.pin.computeDistance")
    def test_get_all_distances(self, mock_compute_distance, collision_manager):
        """Test getting all distances."""
        # Mock distance computation
        mock_result1 = Mock()
        mock_result1.min_distance = 0.1
        mock_result2 = Mock()
        mock_result2.min_distance = 0.2

        mock_compute_distance.side_effect = [mock_result1, mock_result2]

        # Set up collision pairs
        collision_manager.geom_model.collisionPairs = [Mock(), Mock()]

        distances = collision_manager.get_all_distances()

        expected = np.array([0.1, 0.2])
        np.testing.assert_array_equal(distances, expected)
        assert mock_compute_distance.call_count == 2

    def test_get_all_distances_no_geom_model(self, collision_manager):
        """Test getting all distances without geometry model."""
        collision_manager.geom_model = None

        distances = collision_manager.get_all_distances()

        assert len(distances) == 0

    def test_print_collision_pairs(self, collision_manager, capsys):
        """Test printing collision pairs."""
        # Mock collision pairs and results
        pair1 = Mock()
        pair1.first = 0
        pair1.second = 1

        result1 = Mock()
        result1.isCollision.return_value = True

        collision_manager.geom_model.collisionPairs = [pair1]
        collision_manager.geom_data.collisionResults = [result1]

        collision_manager.print_collision_pairs()

        captured = capsys.readouterr()
        assert "Total collision pairs: 1" in captured.out
        assert "link1_geom" in captured.out
        assert "link2_geom" in captured.out
        assert "COLLISION" in captured.out

    def test_print_collision_pairs_no_geom_model(self, collision_manager, capsys):
        """Test printing collision pairs without geometry model."""
        collision_manager.geom_model = None

        collision_manager.print_collision_pairs()

        captured = capsys.readouterr()
        assert "No geometry model available" in captured.out

    def test_visualize_collisions(self, collision_manager):
        """Test collision visualization."""
        # Mock contact
        contact = Mock()
        contact.pos = np.array([0.1, 0.2, 0.3])

        # Mock collision result
        result = Mock()
        result.getNbContacts.return_value = 1
        result.getContact.return_value = contact

        collision_details = [(0, Mock(), result)]

        with patch(
            "figaroh.tools.robotcollisions.pin.SE3ToXYZQUATtuple"
        ) as mock_convert:
            mock_convert.return_value = [0.1, 0.2, 0.3, 0, 0, 0, 1]

            collision_manager.visualize_collisions(collision_details)

            # Should add a sphere for the contact
            collision_manager.viz.viewer.gui.addSphere.assert_called_once()
            collision_manager.viz.viewer.gui.applyConfiguration.assert_called_once()

    def test_visualize_collisions_no_viz(self, mock_robot):
        """Test collision visualization without visualizer."""
        manager = CollisionManager(mock_robot, viz=None)

        # Should not raise an error
        manager.visualize_collisions([])

    def test_cleanup_old_contacts(self, collision_manager):
        """Test cleanup of old contact visualizations."""
        # Add some cached contacts
        collision_manager._vis_cache = {
            "world/collision_contact_0": True,
            "world/collision_contact_1": True,
            "world/collision_contact_2": True,
        }

        # Cleanup should remove contacts beyond current_count
        collision_manager._cleanup_old_contacts(1)

        # Should attempt to delete contacts 1 and 2
        assert collision_manager.viz.viewer.gui.deleteNode.call_count == 2

    def test_file_exists_utility(self, collision_manager):
        """Test file existence utility."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            result = collision_manager._file_exists("/some/path")

            assert result is True
            mock_exists.assert_called_once_with("/some/path")

    def test_file_exists_exception(self, collision_manager):
        """Test file existence utility with exception."""
        with patch("os.path.exists", side_effect=Exception("OS Error")):
            result = collision_manager._file_exists("/some/path")

            assert result is False


class TestCollisionWrapper:
    """Test the backward compatibility CollisionWrapper class."""

    @pytest.fixture
    def mock_robot(self):
        """Create a mock robot for testing."""
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()

        geom_model = Mock()
        geom_model.createData.return_value = Mock()
        geom_model.addAllCollisionPairs = Mock()
        robot.geom_model = geom_model

        return robot

    @pytest.fixture
    def collision_wrapper(self, mock_robot):
        """Create a CollisionWrapper instance."""
        return CollisionWrapper(mock_robot)

    def test_legacy_aliases(self, collision_wrapper):
        """Test that legacy aliases are set correctly."""
        assert collision_wrapper.rmodel == collision_wrapper.model
        assert collision_wrapper.rdata == collision_wrapper.data
        assert collision_wrapper.gmodel == collision_wrapper.geom_model
        assert collision_wrapper.gdata == collision_wrapper.geom_data

    def test_add_collisions_legacy(self, collision_wrapper):
        """Test legacy add_collisions method."""
        collision_wrapper.add_collisions()

        collision_wrapper.geom_model.addAllCollisionPairs.assert_called_once()

    @patch("figaroh.tools.robotcollisions.pin.removeCollisionPairs")
    def test_remove_collisions_legacy(self, mock_remove, collision_wrapper):
        """Test legacy remove_collisions method."""
        srdf_path = "/path/to/robot.srdf"

        collision_wrapper.remove_collisions(srdf_path)

        mock_remove.assert_called_once_with(
            collision_wrapper.model, collision_wrapper.geom_model, srdf_path
        )

    def test_compute_collisions_legacy(self, collision_wrapper):
        """Test legacy computeCollisions method."""
        q = np.array([0.1, 0.2, 0.3])
        custom_geom_data = Mock()

        with patch.object(collision_wrapper, "check_collisions") as mock_check:
            mock_check.return_value = True

            result = collision_wrapper.computeCollisions(q, custom_geom_data)

            assert result is True
            assert collision_wrapper.geom_data == custom_geom_data
            mock_check.assert_called_once_with(q)

    def test_get_collision_list_legacy(self, collision_wrapper):
        """Test legacy getCollisionList method."""
        with patch.object(collision_wrapper, "get_collision_details") as mock_get:
            mock_get.return_value = [(0, Mock(), Mock())]

            result = collision_wrapper.getCollisionList()

            assert len(result) == 1
            mock_get.assert_called_once()

    def test_get_collision_distances_legacy(self, collision_wrapper):
        """Test legacy getCollisionDistances method."""
        collisions = [(0, Mock(), Mock())]

        with patch.object(collision_wrapper, "get_collision_distances") as mock_get:
            mock_get.return_value = np.array([0.1])

            result = collision_wrapper.getCollisionDistances(collisions)

            np.testing.assert_array_equal(result, np.array([0.1]))
            mock_get.assert_called_once_with(collisions)

    def test_get_distances_legacy(self, collision_wrapper):
        """Test legacy getDistances method."""
        with patch.object(collision_wrapper, "get_all_distances") as mock_get:
            mock_get.return_value = np.array([0.1, 0.2])

            result = collision_wrapper.getDistances()

            np.testing.assert_array_equal(result, np.array([0.1, 0.2]))
            mock_get.assert_called_once()

    def test_get_all_pairs_legacy(self, collision_wrapper):
        """Test legacy getAllpairs method."""
        with patch.object(collision_wrapper, "print_collision_pairs") as mock_print:
            collision_wrapper.getAllpairs()

            mock_print.assert_called_once()

    def test_check_collision_legacy(self, collision_wrapper):
        """Test legacy check_collision method."""
        q = np.array([0.1, 0.2, 0.3])

        with patch.object(collision_wrapper, "check_collisions") as mock_check:
            mock_check.return_value = False

            result = collision_wrapper.check_collision(q)

            assert result is False
            mock_check.assert_called_once_with(q)

    def test_display_collisions_legacy(self, collision_wrapper):
        """Test legacy displayCollisions method."""
        collisions = [(0, Mock(), Mock())]

        with patch.object(collision_wrapper, "visualize_collisions") as mock_display:
            collision_wrapper.displayCollisions(collisions)

            mock_display.assert_called_once_with(collisions)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_collision_manager_with_no_robot_geom_model(self):
        """Test CollisionManager when robot has no geometry model."""
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()

        # Properly remove geom_model attribute
        if hasattr(robot, "geom_model"):
            delattr(robot, "geom_model")

        manager = CollisionManager(robot)

        assert manager.geom_model is None
        assert manager.geom_data is None

    def test_collision_manager_geom_data_configuration(self):
        """Test that geometry data is properly configured."""
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()

        geom_model = Mock()
        geom_data = Mock()
        geom_model.createData.return_value = geom_data
        robot.geom_model = geom_model

        manager = CollisionManager(robot)

        # Should enable contact detection
        assert geom_data.collisionRequests.enable_contact is True

    def test_visualization_with_no_contacts(self):
        """Test visualization when collision result has no contacts."""
        # Create mock robot
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()

        geom_model = Mock()
        geom_data = Mock()
        geom_model.createData.return_value = geom_data
        geom_data.collisionRequests = Mock()
        robot.geom_model = geom_model

        # Create mock viz
        viz = Mock()
        viz.viewer.gui.addSphere = Mock()
        viz.viewer.gui.applyConfiguration = Mock()
        viz.viewer.gui.deleteNode = Mock()

        manager = CollisionManager(robot, viz=viz)

        # Mock collision result with no contacts
        result = Mock()
        result.getNbContacts.return_value = 0

        collision_details = [(0, Mock(), result)]

        manager.visualize_collisions(collision_details)

        # Should not add any spheres
        manager.viz.viewer.gui.addSphere.assert_not_called()


class TestCollisionManagerAdvanced:
    """Advanced tests for CollisionManager with edge cases and performance."""

    @pytest.fixture
    def mock_robot_complete(self):
        """Create a complete mock robot with all required attributes."""
        robot = Mock()
        robot.model = Mock()
        robot.model.createData.return_value = Mock()

        # Mock geometry model with all required methods
        geom_model = Mock()
        geom_data = Mock()
        geom_model.createData.return_value = geom_data
        geom_model.addAllCollisionPairs = Mock()
        geom_model.collisionPairs = []
        geom_model.geometryObjects = []

        # Mock collision requests
        geom_data.collisionRequests = Mock()
        geom_data.collisionRequests.enable_contact = False
        geom_data.collisionResults = []
        geom_data.distanceResults = []

        robot.geom_model = geom_model
        return robot

    @pytest.fixture
    def collision_manager_complete(self, mock_robot_complete):
        """Create CollisionManager with complete mock robot."""
        return CollisionManager(mock_robot_complete)

    def test_setup_collision_pairs_with_nonexistent_srdf(
        self, collision_manager_complete
    ):
        """Test setup collision pairs with non-existent SRDF file."""
        with patch("os.path.exists", return_value=False):
            collision_manager_complete.setup_collision_pairs("/nonexistent/path.srdf")

            # Should still add collision pairs even if SRDF doesn't exist
            collision_manager_complete.geom_model.addAllCollisionPairs.assert_called_once()

    def test_check_collisions_with_none_geom_data(self, collision_manager_complete):
        """Test collision checking when geom_data is None."""
        collision_manager_complete.geom_data = None

        result = collision_manager_complete.check_collisions(np.array([0.1, 0.2, 0.3]))

        assert result is False

    @patch("figaroh.tools.robotcollisions.pin.updateGeometryPlacements")
    @patch("figaroh.tools.robotcollisions.pin.computeCollisions")
    def test_check_collisions_with_exception(
        self, mock_compute, mock_update, collision_manager_complete
    ):
        """Test collision checking when pinocchio functions raise exceptions."""
        q = np.array([0.1, 0.2, 0.3])
        mock_compute.side_effect = Exception("Pinocchio error")

        # Should handle exceptions gracefully
        with pytest.raises(Exception):
            collision_manager_complete.check_collisions(q)

    def test_get_collision_details_with_empty_results(self, collision_manager_complete):
        """Test getting collision details when no collisions exist."""
        collision_manager_complete.geom_data.collisionResults = []
        collision_manager_complete.geom_model.collisionPairs = []

        details = collision_manager_complete.get_collision_details()

        assert details == []

    def test_visualize_collisions_with_multiple_contacts(
        self, collision_manager_complete
    ):
        """Test visualization with multiple contact points."""
        viz = Mock()
        viz.viewer.gui.addSphere = Mock()
        viz.viewer.gui.applyConfiguration = Mock()
        collision_manager_complete.viz = viz

        # Create multiple collision results with contacts
        collision_details = []
        for i in range(5):
            contact = Mock()
            contact.pos = np.array([i * 0.1, i * 0.2, i * 0.3])

            result = Mock()
            result.getNbContacts.return_value = 1
            result.getContact.return_value = contact

            collision_details.append((i, Mock(), result))

        with patch(
            "figaroh.tools.robotcollisions.pin.SE3ToXYZQUATtuple"
        ) as mock_convert:
            mock_convert.return_value = [0.1, 0.2, 0.3, 0, 0, 0, 1]

            collision_manager_complete.visualize_collisions(collision_details)

            # Should add spheres for all contacts
            assert viz.viewer.gui.addSphere.call_count == 5
            assert viz.viewer.gui.applyConfiguration.call_count == 5

    def test_visualize_collisions_exceeding_max_patches(
        self, collision_manager_complete
    ):
        """Test visualization when number of collisions exceeds max patches."""
        viz = Mock()
        viz.viewer.gui.addSphere = Mock()
        viz.viewer.gui.applyConfiguration = Mock()
        collision_manager_complete.viz = viz
        collision_manager_complete._max_patches = 3

        # Create more collision details than max patches
        collision_details = []
        for i in range(5):
            contact = Mock()
            contact.pos = np.array([i * 0.1, i * 0.2, i * 0.3])

            result = Mock()
            result.getNbContacts.return_value = 1
            result.getContact.return_value = contact

            collision_details.append((i, Mock(), result))

        with patch(
            "figaroh.tools.robotcollisions.pin.SE3ToXYZQUATtuple"
        ) as mock_convert:
            mock_convert.return_value = [0.1, 0.2, 0.3, 0, 0, 0, 1]

            collision_manager_complete.visualize_collisions(collision_details)

            # Should only add spheres up to max_patches
            assert viz.viewer.gui.addSphere.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
