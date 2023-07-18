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

from sys import argv
import os
from os.path import dirname, join, abspath
import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from figaroh.tools.robot import Robot
from figaroh.calibration.calibration_tools import (
    cartesian_to_SE3
)

from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer

# create a robot object
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago_no_hand.urdf",
    package_dirs = package_dirs,
    isFext=True  # add free-flyer joint at base
)


# create a sinusoidal motion for free flyer
F = 1 # Hz 
omega = 2*np.pi*F
dt = 1/100 # sampling 
T = 5
N = int(T/dt)

XYZ = [0.0001, 0.0001, 0.0005] # amplitude x, y, z
RPY  = [1*np.pi/180, 1*np.pi/180, 0.5*np.pi/180] # amplitude roll, pitch, yaw 
A = XYZ + RPY 

q_ff = np.zeros((6,N+1))
v_ff = np.zeros((6,N+1))
a_ff = np.zeros((6,N+1))
for q_i in range(len(A)):
    for ith in range(N+1):
        q_ff[q_i, ith] = A[q_i]*np.sin(omega*ith*dt)
        v_ff[q_i, ith] = omega*A[q_i]*np.cos(omega*ith*dt)
        a_ff[q_i, ith] = -omega*omega*A[q_i]*np.sin(omega*ith*dt)

# convert rpy to quaternion 
quat = np.zeros((7, N+1))
q = np.full((N+1, robot.q0.shape[0]), robot.q0)
dq = np.zeros((N+1, robot.model.nv))
ddq = np.zeros((N+1, robot.model.nv))

for ith in range(N+1):
    SE3 = cartesian_to_SE3(q_ff[:, ith])
    quat[:, ith] = pin.SE3ToXYZQUAT(SE3)
    q[ith, 0:7] = quat[:, ith]
    dq[ith, 0:6] = v_ff[:, ith]
    ddq[ith, 0:6] = a_ff[:, ith]

tau = np.zeros((N+1, robot.model.nv))

for ii in range(N+1):
    tau[ii, :] = pin.rnea(robot.model, robot.data, q[ii, :], dq[ii, :], ddq[ii, :])

robot.initViewer(loadModel=True)
gui = robot.viewer.gui
gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 1)

# viz = MeshcatVisualizer(
#         model=robot.model, collision_model=robot.collision_
# model, visual_model=robot.visual_model, url='classical'
#     )
# time.sleep(3)
# for i in range(N+1):
#     robot.display(q[i, :])
#     time.sleep(dt)

# fig, ax = plt.subplots(6,1)
# for jj in range(6):
#     ax[jj].plot(tau[:, jj])
# plt.show()
