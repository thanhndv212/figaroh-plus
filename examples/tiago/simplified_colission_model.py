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
# from tools.regressor import *
# from tools.qrdecomposition import *
# from tools.randomdata import *
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

    # custom translational placement
    xyz_1 = np.array([-0.028, 0, -0.01])
    xyz_2 = np.array([0.005, 0, -0.35])
    xyz_3 = np.array([0, 0, 0.20])
    xyz_4 = np.array([0.005, 0, -0.05])
    xyz_5 = np.array([-0.03, 0.09, 0])
    xyz_6 = np.array([0., 0., 0.17])

    # joint parent ID
    base_ID = robot.model.getJointId("universe")
    torso_ID = robot.model.getJointId("torso_lift_joint")
    head_cap_ID = robot.model.getJointId("head_2_joint")
    head_box_ID = robot.model.getJointId("head_1_joint")
    hand_ID = robot.model.getJointId("arm_7_joint")

    geom_model.addGeometryObject(
        Box("torso_up_box", torso_ID, 0.28, 0.35, 0.2, pin.SE3(np.eye(3), xyz_1))
    )
    visual_model.addGeometryObject(
        Box("torso_up_box", torso_ID, 0.28, 0.35, 0.2, pin.SE3(np.eye(3), xyz_1))
    )
    geom_model.addGeometryObject(
        Box("torso_low_box", torso_ID, 0.221,
            0.26, 0.35, pin.SE3(np.eye(3), xyz_2))
    )
    visual_model.addGeometryObject(
        Box("torso_low_box", torso_ID, 0.221,
            0.26, 0.35, pin.SE3(np.eye(3), xyz_2))
    )
    geom_model.addGeometryObject(
        Capsule("base_cap", base_ID, 0.3, 0.25, pin.SE3(np.eye(3), xyz_3))
    )
    visual_model.addGeometryObject(
        Capsule("base_cap", base_ID, 0.3, 0.25, pin.SE3(np.eye(3), xyz_3))
    )
    # geom_model.addGeometryObject(
    #     Box("head_box", head_box_ID, 0.1, 0.14, 0.1, pin.SE3(np.eye(3), xyz_4))
    # )
    # visual_model.addGeometryObject(
    #     Box("head_box", head_box_ID, 0.1, 0.14, 0.1, pin.SE3(np.eye(3), xyz_4))
    # )
    geom_model.addGeometryObject(
        Capsule("head_cap", head_cap_ID, 0.17, 0.25, pin.SE3(np.eye(3), xyz_5))
    )
    visual_model.addGeometryObject(
        Capsule("head_cap", head_cap_ID, 0.17, 0.25, pin.SE3(np.eye(3), xyz_5))
    )

    geom_model.addGeometryObject(
        Capsule("hand_cap", hand_ID, 0.1, 0.2, pin.SE3(np.eye(3), xyz_6))
    )
    visual_model.addGeometryObject(
        Capsule("hand_cap", hand_ID, 0.1, 0.2, pin.SE3(np.eye(3), xyz_6))
    )
    # for k in range(len(geom_model.geometryObjects)):
    #     print("object number %d" % k, geom_model.geometryObjects[k].name)

    arm_link_names = [
        "arm_4_link_0",
        "arm_5_link_0",
        # "arm_6_link_0",
        # "wrist_ft_link_0",
        # "wrist_ft_tool_link_0",
        "hand_cap"
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


def build_canopies_simplified(robot):
    visual_model = robot.visual_model
    geom_model = robot.geom_model

    xyz_0 = np.array([0., 0., 0.0])
    xyz_1 = np.array([0., 0., 0.5])
    xyz_6 = np.array([0., 0., -0.1])

    base_ID = robot.model.getJointId("universe")
    hand_ID = robot.model.getJointId("arm_right_7_joint")

    geom_model.addGeometryObject(
        Box("torso_box", base_ID, 0.35, 0.40, 1, pin.SE3(np.eye(3), xyz_1))
    )
    visual_model.addGeometryObject(
        Box("torso_box", base_ID, 0.35, 0.40, 1, pin.SE3(np.eye(3), xyz_1))
    )
    geom_model.addGeometryObject(
        Box("ground_plane", base_ID, 2.5, 2.5, 0.1, pin.SE3(np.eye(3), xyz_0))
    )
    visual_model.addGeometryObject(
        Box("ground_plane", base_ID, 2.5, 2.5, 0.1, pin.SE3(np.eye(3), xyz_0))
    )
    geom_model.addGeometryObject(
        Capsule("hand_cap", hand_ID, 0.05, 0.2, pin.SE3(np.eye(3), xyz_6))
    )
    visual_model.addGeometryObject(
        Capsule("hand_cap", hand_ID, 0.05, 0.2, pin.SE3(np.eye(3), xyz_6))
    )
    arm_link_names = [
        "arm_right_3_link_0",
        "arm_right_4_link_0",
        "arm_right_5_link_0",
        "arm_right_6_link_0",
        # "wrist_ft_link_0",
        # "wrist_ft_tool_link_0",
        "hand_cap"
    ]
    arm_link_ids = [geom_model.getGeometryId(k) for k in arm_link_names]
    mask_link_names = [
        "torso_box",
        "ground_plane"
    ]
    mask_link_ids = [geom_model.getGeometryId(k) for k in mask_link_names]
    for i in mask_link_ids:
        for j in arm_link_ids:
            geom_model.addCollisionPair(pin.CollisionPair(i, j))
    print("number of collision pairs of simplified model is: ",
          len(geom_model.collisionPairs))


def build_tiago_normal(robot, srdf_dr, srdf_file):
    # # Remove collision pairs listed in the SRDF file
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "models")
    model_path = join(pinocchio_model_dir, "others/robots")
    srdf_filename = srdf_file  # "tiago.srdf"
    srdf_model_path = model_path + srdf_dr + \
        srdf_filename  # "/tiago_description/srdf/"

    geom_model = robot.geom_model
    geom_model.addAllCollisionPairs()

    collision = CollisionWrapper(robot=robot, viz=None)
    collision.remove_collisions(srdf_model_path)

    # Create data structures
    geom_data = pin.GeometryData(geom_model)


def check_tiago_autocollision(robot, q, srdf_dr='', srdf_file=''):
    build_tiago_simplified(robot)
    # build_tiago_normal(robot, srdf_dr, srdf_file)
    collision = CollisionWrapper(robot, viz=None)
    collided_idx = []
    for i in range(q.shape[0]):
        is_collision = collision.computeCollisions(q[i, :])
        if not is_collision:
            print("config %s self-collision is not violated!" % i)
        else:
            print("config %s self-collision is violated!" % i)
            collided_idx.append(i)
    return collided_idx


def main():
    print("You have to start 'meshcat-server' in a terminal ...")
    time.sleep(3)
    ros_package_path = os.getenv('ROS_PACKAGE_PATH')
    package_dirs = ros_package_path.split(':')
    robot = Robot(
    'data/tiago_schunk.urdf',
    package_dirs= package_dirs,     
    # isFext=True  # add free-flyer joint at base
    )
    # Tiago no hand
    # urdf_dr = "tiago_description/robots"
    # urdf_file = "tiago_no_hand_mod.urdf"
    # srdf_dr = "/tiago_description/srdf/"
    # srdf_file = "tiago.srdf"

    # Talos reduced
    urdf_dr = "talos_data/robots"
    urdf_file = "talos_reduced.urdf"
    srdf_dr = "/talos_data/srdf/"
    srdf_file = "talos.srdf"

    # robot = Robot(urdf_dr, urdf_file)

    q = np.empty((20, robot.q0.shape[0]))
    for i in range(20):
        q[i, :] = pin.randomConfiguration(robot.model)
    check_tiago_autocollision(robot, q, srdf_dr, srdf_file)

    # display few configurations
    viz = MeshcatVisualizer(
        model=robot.model, collision_model=robot.collision_model,
        visual_model=robot.visual_model, url='classical'
    )
    time.sleep(3)
    for i in range(20):
        viz.display(q[i, :])
        time.sleep(2)


if __name__ == '__main__':
    main()
