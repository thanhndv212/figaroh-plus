import pinocchio as pin
from gepetto import Color
import numpy as np
from os.path import abspath
from utils.mate_tools import load_robot
from pinocchio.visualize import GepettoVisualizer
import time


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
    # for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
    #     gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    # gui.refresh()


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


robot = load_robot("urdf/mate.urdf", load_by_urdf=True)

data = robot.model.createData()
robot.setVisualizer(GepettoVisualizer())
robot.initViewer(loadModel=True)

gui = robot.viewer.gui

gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 1)
gui.setBackgroundColor1("python-pinocchio", [1.0, 1, 1, 1])
gui.setBackgroundColor2("python-pinocchio", [1.0, 1, 1, 1])


q = robot.q0
for i in range(10):
    q = pin.randomConfiguration(robot.model)
    robot.display(q)
    time.sleep(0.5)
