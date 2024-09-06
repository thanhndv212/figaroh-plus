import pinocchio as pin
from gepetto import Color

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os
from os.path import dirname, join, abspath
from tiago_utils.tiago_tools import load_robot


def add_joint(name, joint):
    # print(name)
    def rotation(q, color_):
        gui.addCylinder(name + "/axis", 0.01, 0.1, color_)
        gui.applyConfiguration(name + "/axis", q)

    def translation(q, color_):
        gui.addBox(name + "/axis", 0.015, 0.015, 0.1, color_)
        gui.applyConfiguration(name + "/axis", q)
    gui.createGroup(name)
    gui.addToGroup(name, "world")
    from math import sqrt

    if "universe" not in name:
        jointtype = joint.shortname()
        if jointtype == "JointModelRX":  # Z->X
            rotation([0, 0, 0, 0, sqrt(2) / 2, 0, sqrt(2) / 2], Color.red)
        elif jointtype == "JointModelRY":  # Y->Z
            rotation([0, 0, 0, -sqrt(2) / 2, 0, 0, sqrt(2) / 2], Color.green)
        elif jointtype == "JointModelRZ":
            rotation([0, 0, 0, 0, 0, 0, 1], Color.blue)
        elif jointtype == "JointModelPZ":
            translation([0, 0, 0, 0, 0, 0, 1], Color.red)
        # elif jointtype == "JointModelFreeFlyer":
        #    gui.addXYZaxis(name+"/axis", Color.blue, 0.01, 0.1)


def display(q):
    robot.display(q)
    for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
        gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    gui.refresh()


def add_axis_to_frame(data, model, frame):
    if "_joint" not in frame:
        gui.createGroup(frame)
        gui.addToGroup(frame, "world")
        placement = pin.SE3ToXYZQUATtuple(data.oMf[model.getFrameId(frame)])
        axis_name = frame + "/frame"
        print(axis_name, placement)
        gui.addXYZaxis(axis_name, Color.blue, 0.01, 0.1)
        gui.applyConfiguration(axis_name, placement)
    else:
        joint_frame = frame + "_frame"
        gui.createGroup(joint_frame)
        gui.addToGroup(joint_frame, "world")
        placement = pin.SE3ToXYZQUATtuple(data.oMf[model.getFrameId(frame)])

        axis_name = joint_frame + "/frame"
        print(axis_name, placement)
        gui.addXYZaxis(axis_name, Color.blue, 0.01, 0.1)
        gui.applyConfiguration(axis_name, placement)


robot = load_robot(
    abspath("urdf/tiago.urdf"),
    load_by_urdf=True,
)


data = robot.model.createData()
robot.initViewer(loadModel=True)

model = robot.model
gui = robot.viewer.gui

# draw joint axis
chain_1 = [
    "universe",
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
    "head_1_joint",
    "head_2_joint"
]
chain_2 = [
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
    "head_1_joint",
    "head_2_joint",
]
calib_joints = chain_1
for name, joint in zip(model.names, model.joints):
    if name in calib_joints:
        add_joint(name, joint)

q = robot.q0
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
# add_axis_to_frame(data, model, "gripper_right_fingertip_1_link")
# add_axis_to_frame(data, model, "right_sole_link")
# add_axis_to_frame(data, model, "gripper_left_fingertip_1_link")
# add_axis_to_frame(data, model, "left_sole_link")
# add_axis_to_frame(data, model, "desk_link_1")
for jname in model.names:
    if jname in calib_joints:
        add_axis_to_frame(data, model, jname)

# draw kinematic tree
for jid in range(0, model.njoints):
    # child frame
    name, joint = model.names[jid], model.joints[jid]
    # parent frame
    pid = model.parents[jid]
    pname, pjoint = model.names[pid], model.joints[pid]
    if pid == 1:
        print(name, pname)
    pMi = model.jointPlacements[jid]

    if name.find("hand") == -1 and name.find("wheel") == -1:
        gui.addLine(
            pname + "/to_" + name,
            [0, 0, 0],
            pMi.translation.tolist(),
            Color.black,
        )

# q = np.array([ 0.0907,0.136,-0.493,1.08,-0.545,0.0676,0.0887,0.133,-0.416,0.862,-0.425,0.0736,0,0.00676,0.258,0.573,-0.0002,-1.36,0,0,0.1,0,-0.258,-0.573,0.0002,-1.36,0,0,0.1,0,0,0 ])
# q = np.array([ 0,0,-0.4114 ,0.8594,-0.4480,-0.0017,0,0,-0.4114,0.8594,-0.4480,-0.0017 ,0,0.00676,0.258,0.573,-0.0002,-1.36,0,0,0.1,0,-0.258,-0.573,0.0002,-1.36,0,0,0.1,0,0,0 ])


# data_path = "/media/thanhndv212/4b3f9d64-01ce-4759-86c9-43e89196bdc2/thanhndv212/Cooking/raw_data/Calibration/Talos/contact-calibration/complied_csv/"
# file_name = "compiled_measurements_right_1107.csv"
# path_q = data_path + file_name
# from calibration_tools import read_data
# import time

# q_r = read_data(model, path_q)

# q = q_r[20, :]

display(q)

gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 0.3)
gui.setBackgroundColor1("python-pinocchio", [1.0, 1, 1, 1])
gui.setBackgroundColor2("python-pinocchio", [1.0, 1, 1, 1])


view = [4, 0.0, 0.0]

# r = R.from_euler("zyx", [0, 90, 0], degrees=True)
# R = np.matrix("0 1 0; 1 0 0; 0 0 1")
# quat = pin.Quaternion(r.as_matrix()).coeffs().tolist()
# view[len(view) :] = quat
# # print(view)

# gui.setCameraTransform("python-pinocchio", view)


# robot.viewer.gui.setBackgroundColor2('python-pinocchio',[1.,0.0,0.,0.0])
# robot.viewer.gui.deleteNode("world/pinocchio/visuals",True)
# robot.viewer.gui.setCameraToBestFit('python-pinocchio')

# robot.viewer.gui.addSphere("world/sphere_1",0.1, [1., 0., 0., 1.])
# robot.viewer.gui.applyConfiguration("world/sphere_1",[0, 0, 0, 1])

# Print out the placement of each joint of the kinematic tree
# for name, oMi in zip(model.frames, data.oMf):
#     print(name.name, oMi.translation)

# print(name)

# print(*oMi.homogeneous)
