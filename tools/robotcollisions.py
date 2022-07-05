import pinocchio as pin
import numpy as np


class CollisionWrapper:
    def __init__(self, robot, geom_model=None, geom_data=None, viz=None):
        self.robot = robot
        self.viz = viz
        self.rmodel = robot.model
        self.rdata = self.rmodel.createData()
        if geom_model is None:
            self.gmodel = self.robot.geom_model
        else:
            self.gmodel = geom_model
        # self.add_collisions()

        if geom_data is None:
            self.gdata = self.gmodel.createData()
        else:
            self.gdata = geom_data
        self.gdata.collisionRequests.enable_contact = True

    def add_collisions(self):
        self.gmodel.addAllCollisionPairs()
        # print("num collision pairs - initial:", len(self.gmodel.collisionPairs))

    def remove_collisions(self, srdf_model_path):
        if srdf_model_path is None:
            pass
        else:
            pin.removeCollisionPairs(self.rmodel, self.gmodel, srdf_model_path)
            # print(
            #     "num collision pairs - after removing useless collision pairs:",
            #     len(self.gmodel.collisionPairs),
            # )

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
        """Return a list of triplets [ index,collision,result ] where index is the
        index of the collision pair, colision is gmodel.collisionPairs[index]
        and result is gdata.collisionResults[index].
        """
        return [
            [ir, self.gmodel.collisionPairs[ir], r]
            for ir, r in enumerate(self.gdata.collisionResults)
            if r.isCollision()
        ]

    def getCollisionDistances(self, collisions=None):
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.array([])
        dist = np.array(
            [self.gdata.distanceResults[i].min_distance for (i, c, r) in collisions]
        )
        return dist

    def getDistances(self):
        dist_all = np.array(
            [
                pin.computeDistance(self.gmodel, self.gdata, k).min_distance
                for k in range(len(self.gmodel.collisionPairs))
            ]
        )
        return dist_all

    def getAllpairs(self):
        for k in range(len(self.gmodel.collisionPairs)):
            cr = self.gdata.collisionResults[k]
            cp = self.gmodel.collisionPairs[k]
            print(
                "collision pair:",
                k,
                " ",
                self.gmodel.geometryObjects[cp.first].name,
                ",",
                self.gmodel.geometryObjects[cp.second].name,
                "- collision:",
                "Yes" if cr.isCollision() else "No",
            )

    def check_collision(self, q):
        for k in range(len(self.gmodel.collisionPairs)):
            cr = self.gdata.collisionResults[k]
            if cr.isCollision():
                return True
                break
        return False

    # --- DISPLAY ---------------------------------------------------------------------
    # --- DISPLAY ---------------------------------------------------------------------
    # --- DISPLAY ---------------------------------------------------------------------

    def initDisplay(self, viz=None):
        if viz is not None:
            self.viz = viz
        assert self.viz is not None

        self.patchName = "world/contact_%d_%s"
        self.ncollisions = 3
        # self.createDisplayPatchs(0)

    def createDisplayPatchs(self, ncollisions):

        if ncollisions == self.ncollisions:
            return
        elif ncollisions < self.ncollisions:  # Remove patches
            for i in range(ncollisions, self.ncollisions):
                self.viz[self.patchName % (i, "a")].delete()
                self.viz[self.patchName % (i, "b")].delete()
        else:
            for i in range(self.ncollisions, ncollisions):
                self.viz.addCylinder(self.patchName % (i, "a"), 0.0005, 0.05, "red")
                # viz.addCylinder( self.patchName % (i,'b') , .0005,.05,"red")

        self.ncollisions = ncollisions

    def displayContact(self, ipatch, contact):
        """
        Display a small red disk at the position of the contact, perpendicular to the
        contact normal.

        @param ipatchf: use patch named "world/contact_%d" % contactRef.
        @param contact: the contact object, taken from Pinocchio (HPP-FCL) e.g.
        geomModel.collisionResults[0].getContact(0).
        """
        name = self.patchName % (ipatch, "a")
        R = pin.Quaternion.FromTwoVectors(np.array([0, 1, 0]), contact.normal).matrix()
        M = pin.SE3(R, contact.pos)
        self.viz.addCylinder(self.patchName % (ipatch, "a"), 0.0005, 0.05, "red")
        self.viz.applyConfiguration(name, M)

    def displayCollisions(self, collisions=None):
        """Display in the viewer the collision list get from getCollisionList()."""
        if self.viz is None:
            return
        if collisions is None:
            collisions = self.getCollisionList()

        # self.createDisplayPatchs(len(collisions))
        if collisions is None:
            return
        else:
            for ic, [i, c, r] in enumerate(collisions):
                self.displayContact(ic, r.getContact(ic))
