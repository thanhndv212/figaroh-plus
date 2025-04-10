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

import pinocchio as pin
import numpy as np


class CollisionWrapper:
    """Wrapper class for handling collision checking and visualization."""

    def __init__(
        self,
        robot,
        geom_model=None,
        geom_data=None,
        viz=None
    ):
        """Initialize collision wrapper.
        
        Args:
            robot: Robot model
            geom_model: Optional geometry model 
            geom_data: Optional geometry data
            viz: Optional visualizer instance
        """
        self.robot = robot
        self.viz = viz
        self.rmodel = robot.model
        self.rdata = self.rmodel.createData()
        if geom_model is None:
            self.gmodel = self.robot.geom_model
        else:
            self.gmodel = geom_model

        if geom_data is None:
            self.gdata = self.gmodel.createData()
        else:
            self.gdata = geom_data
        self.gdata.collisionRequests.enable_contact = True

    def add_collisions(self):
        self.gmodel.addAllCollisionPairs()

    def remove_collisions(self, srdf_model_path):
        if srdf_model_path is None:
            pass
        else:
            pin.removeCollisionPairs(self.rmodel, self.gmodel, srdf_model_path)

    def computeCollisions(self, q, geom_data=None):
        if geom_data is not None:
            self.gdata = geom_data

        pin.updateGeometryPlacements(
            self.rmodel, self.rdata, self.gmodel, self.gdata, q
        )
        res = pin.computeCollisions(
            self.rmodel, self.rdata, self.gmodel, self.gdata, q, False
        )
        return res

    def getCollisionList(self):
        """Get list of collision triplets.
        
        Returns:
            list: Triplets [index, collision, result] where:
                - index: Index of collision pair
                - collision: gmodel.collisionPairs[index]  
                - result: gdata.collisionResults[index]
        """
        return [
            [ir, self.gmodel.collisionPairs[ir], r]
            for ir, r in enumerate(self.gdata.collisionResults)
            if r.isCollision()
        ]

    def getCollisionDistances(self, collisions=None):
        """Get minimum distances for collision pairs.

        Args:
            collisions: Optional list of collision triplets
            
        Returns:
            ndarray: Array of minimum distances
        """
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.array([])
        dist = np.array([
            self.gdata.distanceResults[i].min_distance 
            for (i, c, r) in collisions
        ])
        return dist

    def getDistances(self):
        dist_all = np.array([
            pin.computeDistance(self.gmodel, self.gdata, k).min_distance
            for k in range(len(self.gmodel.collisionPairs))
        ])
        return dist_all

    def getAllpairs(self):
        for k in range(len(self.gmodel.collisionPairs)):
            cr = self.gdata.collisionResults[k]
            cp = self.gmodel.collisionPairs[k]
            name1 = self.gmodel.geometryObjects[cp.first].name
            name2 = self.gmodel.geometryObjects[cp.second].name
            is_collision = "Yes" if cr.isCollision() else "No"
            print(
                "collision pair:", k, " ",
                name1, ",", name2,
                "- collision:", is_collision,
            )

    def check_collision(self, q):
        for k in range(len(self.gmodel.collisionPairs)):
            cr = self.gdata.collisionResults[k]
            if cr.isCollision():
                return True
                break
        return False

    # --- DISPLAY

    def initDisplay(self, viz=None):
        if viz is not None:
            self.viz = viz
        assert self.viz is not None

        self.patchName = "world/contact_%d_%s"
        self.ncollisions = 3

    def createDisplayPatchs(self, ncollisions):

        if ncollisions == self.ncollisions:
            return
        elif ncollisions < self.ncollisions:  # Remove patches
            for i in range(ncollisions, self.ncollisions):
                self.viz[self.patchName % (i, "a")].delete()
                self.viz[self.patchName % (i, "b")].delete()
        else:
            for i in range(self.ncollisions, ncollisions):
                self.viz.addCylinder(
                    self.patchName % (i, "a"),
                    0.0005,
                    0.05,
                    "red"
                )

        self.ncollisions = ncollisions

    def displayContact(self, ipatch, contact):
        """Display contact indicator in visualization.

        Args:
            ipatch: Index for naming displayed contact
            contact: Contact object from collision results
        """
        name = self.patchName % (ipatch, "a")
        R = pin.Quaternion.FromTwoVectors(
            np.array([0, 1, 0]), 
            contact.normal
        ).matrix()
        M = pin.SE3(R, contact.pos)
        self.viz.addCylinder(
            self.patchName % (ipatch, "a"),
            0.0005,
            0.05,
            "red"
        )
        self.viz.applyConfiguration(name, M)

    def displayCollisions(self, collisions=None):
        """Display collision contacts in visualization.

        Args:
            collisions: Optional list of collision triplets
        """
        if self.viz is None:
            return
        if collisions is None:
            collisions = self.getCollisionList()
        if collisions is None:
            return
        else:
            for ic, [i, c, r] in enumerate(collisions):
                self.displayContact(ic, r.getContact(ic))
