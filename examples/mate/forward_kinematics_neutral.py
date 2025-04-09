# Copyright [2021-2025] Thanh Nguyen
import csv
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from figaroh.tools.robot import Robot
import sys
import os
from utils.mate_tools import load_robot
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# import time
# from src.ik_utils import *
# from src.motion_utils import *


def place(viz, name, M):
    """_Sets an object "name" in the viewer viz using the transformation matrix M_

    Args:
        viz (_Visualiser_): _Pinocchio visualiser_
        name (_str_): _The name of the object_
        M (_SE3_): _The transformation matrix to which we want the object to be_
    """
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()


robot = load_robot("urdf/mate.urdf", load_by_urdf=True)

model = robot.model
data = robot.data

# DISPLAYS THE MARKERS
viz = GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)

try:
    viz.initViewer()
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install gepetto-viewer"
    )
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print(
        "Error while loading the viewer model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

print(viz.display_collisions)

q = pin.neutral(model)
viz.display(pin.neutral(model))

viz.viewer.gui.addXYZaxis("world/base", [0, 0.0, 1.0, 1.0], 0.008, 0.1)
viz.viewer.gui.addXYZaxis("world/angle_motor", [0, 0.0, 1.0, 1.0], 0.008, 0.1)
viz.viewer.gui.addXYZaxis("world/q_1", [0, 0.0, 1.0, 1.0], 0.008, 0.1)
viz.viewer.gui.addXYZaxis("world/lower_arm", [0, 0.0, 1.0, 1.0], 0.008, 0.1)
viz.viewer.gui.addXYZaxis("world/upper_arm", [0, 0.0, 1.0, 1.0], 0.008, 0.1)
viz.viewer.gui.addXYZaxis("world/tool", [0, 0.0, 1.0, 0.6], 0.008, 0.1)
viz.viewer.gui.addXYZaxis("world/end_effector", [0, 0.0, 1.0, 0.6], 0.008, 0.1)
viz.viewer.gui.addSphere("world/ee_sphere", 0.01, [255, 0.0, 0, 1.0])

# q[1] = np.pi/2.0
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
viz.display(q)
M_base = data.oMf[model.getFrameId("Base")]
M_angle_motor = data.oMf[model.getFrameId("angle_motor")]
M_lower_arm = data.oMf[model.getFrameId("lower_arm")]
M_upper_arm = data.oMf[model.getFrameId("upper_arm")]
M_tool = data.oMf[model.getFrameId("tool")]
M_end_effector = data.oMf[model.getFrameId("end_effector")]
M_q_1 = data.oMf[model.getFrameId("q_1")]
print(M_q_1)
place(viz, "world/base", M_base)
# place(viz,'world/angle_motor', M_angle_motor)
place(viz, "world/lower_arm", M_lower_arm)
place(viz, "world/upper_arm", M_upper_arm)
place(viz, "world/tool", M_tool)
place(viz, "world/q_1", M_q_1)
place(viz, "world/end_effector", M_end_effector)
print(M_end_effector.translation)
time.sleep(0.005)
