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

import numpy as np
from numpy import pi
import hppfcl
import pinocchio as pin
import time

from figaroh.tools.robot import Robot
# from figaroh.tools.regressor import *
# from figaroh.tools.qrdecomposition import *
# from figaroh.tools.randomdata import *
from figaroh.tools.robotcollisions import *
from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer

import os
from os.path import dirname, join, abspath


def Capsule(name, joint, radius, length, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    # They should be capsules ... but hppfcl current version is buggy with Capsules...
    # hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Cylinder(radius, length)
    geom = pin.GeometryObject(name, joint, hppgeom, placement)
    geom.meshColor = np.array(color)
    return geom


def Box(name, joint, x, y, z, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    # They should be capsules ... but hppfcl current version is buggy with Capsules...
    # hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Box(x, y, z)
    geom = pin.GeometryObject(name, joint, hppgeom, placement)
    geom.meshColor = np.array(color)
    return geom


def Sphere(name, joint, radius, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    hppgeom = hppfcl.Sphere(radius)
    geom = pin.GeometryObject(name, joint, hppgeom, placement)
    geom.meshColor = np.array(color)
    return geom


def build_tiago_simplified(robot):
    visual_model = robot.visual_model
    geom_model = robot.geom_model

    xyz_1 = np.array([-0.028, 0, -0.01])
    xyz_2 = np.array([0.005, 0, -0.35])
    xyz_3 = np.array([0, 0, 0.20])
    xyz_4 = np.array([0, 0, 0])
    xyz_5 = np.array([-0.03, 0.09, 0])

    geom_model.addGeometryObject(
        Box("torso_up_box", 1, 0.28, 0.35, 0.2, pin.SE3(np.eye(3), xyz_1))
    )
    visual_model.addGeometryObject(
        Box("torso_up_box", 1, 0.28, 0.35, 0.2, pin.SE3(np.eye(3), xyz_1))
    )
    geom_model.addGeometryObject(
        Box("torso_low_box", 1, 0.221, 0.26, 0.35, pin.SE3(np.eye(3), xyz_2))
    )
    visual_model.addGeometryObject(
        Box("torso_low_box", 1, 0.221, 0.26, 0.35, pin.SE3(np.eye(3), xyz_2))
    )
    geom_model.addGeometryObject(
        Capsule("base_cap", 0, 0.3, 0.25, pin.SE3(np.eye(3), xyz_3))
    )
    visual_model.addGeometryObject(
        Capsule("base_cap", 0, 0.3, 0.25, pin.SE3(np.eye(3), xyz_3))
    )
    geom_model.addGeometryObject(
        Capsule("forearm_cap", 6,  0.30, 0.10,pin.SE3(np.eye(3), xyz_4))
    )
    visual_model.addGeometryObject(
        Capsule("forearm_cap", 6,  0.10, 0.30, pin.SE3(np.eye(3), xyz_4))
    )
    geom_model.addGeometryObject(
        Capsule("head_cap", 10, 0.17, 0.25, pin.SE3(np.eye(3), xyz_5))
    )
    visual_model.addGeometryObject(
        Capsule("head_cap", 10, 0.17, 0.25, pin.SE3(np.eye(3), xyz_5))
    )

    # for k in range(len(geom_model.geometryObjects)):
    #     print("object number %d" % k, geom_model.geometryObjects[k].name)

    arm_link_names = [
        "forearm_cap"
        # "arm_4_link_0",
        # "arm_5_link_0",
        # "arm_6_link_0",
        # "wrist_ft_link_0",
        # "wrist_ft_tool_link_0",
    ]
    arm_link_ids = [geom_model.getGeometryId(k) for k in arm_link_names]
    mask_link_names = [
        "torso_up_box",
        "torso_low_box",
        "base_cap",
        "head_cap",
    ]
    mask_link_ids = [geom_model.getGeometryId(k) for k in mask_link_names]
    for i in mask_link_ids:
        for j in arm_link_ids:
            geom_model.addCollisionPair(pin.CollisionPair(i, j))
    print("number of collision pairs of simplified model is: ",
          len(geom_model.collisionPairs))

    return robot

def build_tiago_normal(robot):
    # # Remove collision pairs listed in the SRDF file
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "models")
    model_path = join(pinocchio_model_dir, "others/robots")
    srdf_filename = "tiago.srdf"
    srdf_model_path = model_path + "/tiago_description/srdf/" + srdf_filename

    geom_model = robot.geom_model
    geom_model.addAllCollisionPairs()

    collision = CollisionWrapper(robot=robot, viz=None)
    collision.remove_collisions(srdf_model_path)

    # Create data structures
    geom_data = pin.GeometryData(geom_model)


def main():
    print("Start 'meshcat-serve' in a terminal ... ")
    time.sleep(1)
    from tiago_tools import load_robot
    # 1/ Load robot model and create a dictionary containing reserved constants
    robot = load_robot("data/urdf/tiago_hey5.urdf")
    robot = build_tiago_simplified(robot)
    collision = CollisionWrapper(robot, viz=None)

    def check_collision(collision, q):
        is_collision = collision.computeCollisions(q)
        # collision.getAllpairs()
        if not is_collision:
            print("self-collision is not violated!")
        else:
            print("self-collision is violated!")
        return is_collision
    # TODO: write for checking collision model and collision data
    print(robot.model)
    viz = MeshcatVisualizer(
        model=robot.model, collision_model=robot.collision_model, visual_model=robot.visual_model, url='classical'
    )
    time.sleep(3)
    for i in range(20):
        q = pin.randomConfiguration(robot.model)
        if not check_collision(collision, q):
            viz.display(q)
            time.sleep(0.5)


if __name__ == '__main__':
    main()
