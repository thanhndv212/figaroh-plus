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

"""Enhanced collision detection and visualization utilities."""

import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import pinocchio as pin

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CollisionManager:
    """Enhanced collision detection with better performance and safety."""
    
    def __init__(self, robot, geom_model=None, geom_data=None, viz=None):
        self.robot = robot
        self.viz = viz
        self.model = robot.model
        self.data = robot.model.createData()
        
        # Initialize geometry models
        self.geom_model = geom_model or getattr(robot, 'geom_model', None)
        self.geom_data = geom_data or (self.geom_model.createData() if self.geom_model else None)
        
        if self.geom_data:
            self.geom_data.collisionRequests.enable_contact = True
        
        # Visualization cache
        self._vis_cache = {}
        self._max_patches = 10
    
    def setup_collision_pairs(self, srdf_model_path: Optional[str] = None) -> None:
        """Setup collision pairs with optional SRDF filtering."""
        if not self.geom_model:
            raise ValueError("Geometry model not available for collision setup")
        
        # Add all collision pairs
        self.geom_model.addAllCollisionPairs()
        
        # Remove pairs specified in SRDF
        if srdf_model_path and self._file_exists(srdf_model_path):
            pin.removeCollisionPairs(self.model, self.geom_model, srdf_model_path)
    
    def check_collisions(self, q: np.ndarray, update_geometry: bool = True) -> bool:
        """Check for collisions at given configuration."""
        if not self.geom_model or not self.geom_data:
            return False
        
        if update_geometry:
            pin.updateGeometryPlacements(
                self.model, self.data, self.geom_model, self.geom_data, q
            )
        
        return pin.computeCollisions(
            self.model, self.data, self.geom_model, self.geom_data, q, False
        )
    
    def get_collision_details(self) -> List[Tuple[int, any, any]]:
        """Get detailed collision information."""
        if not self.geom_data:
            return []
        
        return [
            (idx, self.geom_model.collisionPairs[idx], result)
            for idx, result in enumerate(self.geom_data.collisionResults)
            if result.isCollision()
        ]
    
    def get_collision_distances(self, collision_details: Optional[List] = None) -> np.ndarray:
        """Get minimum distances for collision pairs."""
        if not self.geom_data:
            return np.array([])
        
        if collision_details is None:
            collision_details = self.get_collision_details()
        
        if not collision_details:
            return np.array([])
        
        return np.array([
            self.geom_data.distanceResults[idx].min_distance
            for idx, _, _ in collision_details
        ])
    
    def get_all_distances(self) -> np.ndarray:
        """Get distances for all collision pairs."""
        if not self.geom_model or not self.geom_data:
            return np.array([])
        
        return np.array([
            pin.computeDistance(self.geom_model, self.geom_data, k).min_distance
            for k in range(len(self.geom_model.collisionPairs))
        ])
    
    def print_collision_pairs(self) -> None:
        """Print all collision pair information."""
        if not self.geom_model or not self.geom_data:
            print("No geometry model available")
            return

        print(f"Total collision pairs: {len(self.geom_model.collisionPairs)}")
        print("-" * 60)

        for k in range(len(self.geom_model.collisionPairs)):
            result = self.geom_data.collisionResults[k]
            pair = self.geom_model.collisionPairs[k]

            name1 = self.geom_model.geometryObjects[pair.first].name
            name2 = self.geom_model.geometryObjects[pair.second].name
            status = "COLLISION" if result.isCollision() else "FREE"

            print(f"Pair {k:3d}: {name1:20s} <-> {name2:20s} [{status}]")
    
    def visualize_collisions(self, collision_details: Optional[List] = None) -> None:
        """Visualize collision contacts with enhanced display."""
        if not self.viz:
            return
        
        if collision_details is None:
            collision_details = self.get_collision_details()
        
        # Clean up old visualizations
        self._cleanup_old_contacts(len(collision_details))
        
        # Display new contacts
        for i, (idx, pair, result) in enumerate(collision_details[:self._max_patches]):
            if result.getNbContacts() > 0:
                contact = result.getContact(0)  # First contact point
                self._display_contact(i, contact, pair)
    
    def _display_contact(self, patch_idx: int, contact, pair) -> None:
        """Display individual contact point."""
        contact_name = f"world/collision_contact_{patch_idx}"
        
        # Create contact visualization
        if contact_name not in self._vis_cache:
            self.viz.viewer.gui.addSphere(contact_name, 0.01, [1.0, 0.0, 0.0, 0.8])
            self._vis_cache[contact_name] = True
        
        # Position at contact point
        placement = pin.SE3.Identity()
        placement.translation = contact.pos
        
        self.viz.viewer.gui.applyConfiguration(
            contact_name, pin.SE3ToXYZQUATtuple(placement)
        )
    
    def _cleanup_old_contacts(self, current_count: int) -> None:
        """Remove old contact visualizations."""
        if not self.viz:
            return
        
        # Remove excess contact visualizations
        for i in range(current_count, self._max_patches):
            contact_name = f"world/collision_contact_{i}"
            if contact_name in self._vis_cache:
                try:
                    self.viz.viewer.gui.deleteNode(contact_name, True)
                    del self._vis_cache[contact_name]
                except:
                    pass  # Ignore deletion errors
    
    def _file_exists(self, filepath: str) -> bool:
        """Check if file exists safely."""
        try:
            import os
            return os.path.exists(filepath)
        except:
            return False


# Backward compatibility wrapper
class CollisionWrapper(CollisionManager):
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, robot, geom_model=None, geom_data=None, viz=None):
        super().__init__(robot, geom_model, geom_data, viz)
        
        # Legacy aliases
        self.rmodel = self.model
        self.rdata = self.data
        self.gmodel = self.geom_model
        self.gdata = self.geom_data
    
    def add_collisions(self):
        """Legacy method."""
        if self.geom_model:
            self.geom_model.addAllCollisionPairs()
    
    def remove_collisions(self, srdf_model_path):
        """Legacy method."""
        if srdf_model_path and self.geom_model:
            pin.removeCollisionPairs(self.model, self.geom_model, srdf_model_path)
    
    def computeCollisions(self, q, geom_data=None):
        """Legacy method."""
        if geom_data:
            self.geom_data = geom_data
        return self.check_collisions(q)
    
    def getCollisionList(self):
        """Legacy method."""
        return self.get_collision_details()
    
    def getCollisionDistances(self, collisions=None):
        """Legacy method."""
        return self.get_collision_distances(collisions)
    
    def getDistances(self):
        """Legacy method."""
        return self.get_all_distances()
    
    def getAllpairs(self):
        """Legacy method."""
        self.print_collision_pairs()
    
    def check_collision(self, q):
        """Legacy method."""
        return self.check_collisions(q)
    
    def displayCollisions(self, collisions=None):
        """Legacy method."""
        self.visualize_collisions(collisions)
