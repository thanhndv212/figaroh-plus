import pinocchio as pin
from gepetto import Color
import numpy as np
from os.path import abspath
from tiago_utils.tiago_tools import load_robot


def add_joint(name, joint):
    """Add a visual representation of a joint to the scene."""
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
        # Add visual elements based on joint type
        if jointtype == "JointModelRX":  # Z->X
            rotation([0, 0, 0, 0, sqrt(2) / 2, 0, sqrt(2) / 2], Color.red)
        elif jointtype == "JointModelRY":  # Y->Z
            rotation([0, 0, 0, -sqrt(2) / 2, 0, 0, sqrt(2) / 2], Color.green)
        elif jointtype == "JointModelRZ":
            rotation([0, 0, 0, 0, 0, 0, 1], Color.blue)
        elif jointtype == "JointModelPZ":
            translation([0, 0, 0, 0, 0, 0, 1], Color.red)


def display(q):
    """Update the display with new joint configurations."""
    robot.display(q)
    for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
        gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    gui.refresh()


def add_axis_to_frame(data, model, frame):
    """Add coordinate axes to a frame."""
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


robot = load_robot(abspath("urdf/tiago.urdf"), load_by_urdf=True)

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

for jname in model.names:
    if jname in calib_joints:
        add_axis_to_frame(data, model, jname)

# draw kinematic tree
for jid in range(0, model.njoints):
    name, joint = model.names[jid], model.joints[jid]
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

display(q)

gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 0.3)
gui.setBackgroundColor1("python-pinocchio", [1.0, 1, 1, 1])
gui.setBackgroundColor2("python-pinocchio", [1.0, 1, 1, 1])

view = [4, 0.0, 0.0]
