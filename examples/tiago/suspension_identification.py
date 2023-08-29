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
import pandas as pd
from scipy import linalg, signal

from figaroh.tools.robot import Robot
from figaroh.calibration.calibration_tools import (
    cartesian_to_SE3
)

from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer

# create a robot object
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
# robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    'data/tiago_no_hand.urdf',
    package_dirs = package_dirs,
    isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

q = pin.randomConfiguration(model)
q[:7] = np.array([0.01, 0.01, 0.01, 0.0 , 0.0, 0.0, 1.0])
dq = np.random.uniform(low=-0.1, high=0.1, size=(model.nv,))
print(model.nq, model.nv, model.njoints)
ddq = 0.5*dq

pin.centerOfMass(model, data, q, dq, ddq)
print(data.acom[0])
pin.computeCentroidalMomentumTimeVariation(model, data, q, dq, ddq)
print(data.dhg)

NbSample = 100
Y = 
for i in range(NbSample):












# B. Consider the whole base as one whole generalized suspension
# 
######## Raw data and processing ##################

# import csv file or read from rosbag

# # load csv files
# path_to_folder = dirname(str(abspath(__file__)))
# pos_csv_path = 'data/tiago_position.csv'
# vel_csv_path = 'data/tiago_velocity.csv'
# eff_csv_path = 'data/tiago_effort.csv'
# t = pd.read_csv(join(path_to_folder, pos_csv_path), usecols=[0]).to_numpy()
# q = pd.read_csv(join(path_to_folder, pos_csv_path),
#                 usecols=list(range(1, 9))).to_numpy()
# dq = pd.read_csv(join(path_to_folder, vel_csv_path),
#                  usecols=list(range(1, 9))).to_numpy()
# tau = pd.read_csv(join(path_to_folder, eff_csv_path),
#                   usecols=list(range(1, 9))).to_numpy()
# print("shape of raw data arrays (t, q, dq, effort): ", t.shape, q.shape, dq.shape, tau.shape)


# # truncate the trivial samples at starting and ending segments
# n_i = 921
# n_f = 6791
# range_tc = range(n_i, n_f)
# t = t[range_tc, :]
# q = q[range_tc, :]
# dq = dq[range_tc, :]
# tau = tau[range_tc, :]
# print("shape of data arrays (t, q, dq, effort) after truncated at both ends: ", t.shape, q.shape, dq.shape, tau.shape)
# # number of sample points
# Ntotal = t.shape[0]
# # apply zero-phase lowpass filter
# nbutter = 4
# f_butter = 2
# f_sample = 100
# b1, b2 = signal.butter(nbutter, f_butter/(f_sample/2), 'low')
# q_med = np.zeros(q.shape)
# q_butter = np.zeros(q.shape)
# dq_med = np.zeros(dq.shape)
# dq_butter = np.zeros(dq.shape)

# for j in range(dq.shape[1]):
#     q_med[:, j] = signal.medfilt(q[:, j], 5)
#     q_butter[:, j] = signal.filtfilt(
#         b1, b2, q_med[:, j], axis=0, padtype="odd", padlen=3 * (max(len(b1), len(b2)) - 1)
#     )
#     dq_med[:, j] = signal.medfilt(dq[:, j], 5)
#     dq_butter[:, j] = signal.filtfilt(
#         b1, b2, dq_med[:, j], axis=0, padtype="odd", padlen=3 * (max(len(b1), len(b2)) - 1)
#     )

# # estimate acceleration
# ddq = np.zeros(dq.shape)
# ddq_raw = np.zeros(dq.shape)

# for j in range(ddq.shape[1]):
#     ddq[:, j] = np.gradient(dq_butter[:, j])/np.gradient(t[:, 0])
#     ddq_raw[:, j] = np.gradient(dq[:, j])/np.gradient(t[:, 0])

# # get full configuration of p, v, a of all joints
# p = np.array([robot.q0]*Ntotal)
# v = np.array([robot.v0]*Ntotal)
# a = np.array([robot.v0]*Ntotal)

# # justify the useability of data set (noise level, amplitude, synchronization)


# #  joint configurations and derivatives
# q = []
# dq = []
# ddq = []

# #  observed markers data and derivatives  
# r_p = []
# dr_p = []
# ddr_p = []

# ######## Find transformation from base link to observed point ##############

# M_op = []
# r_o = []
# dr_o = []
# ddr_o = []

# ## translation 
# # t_o = M_op*t_p

# ## rotation 
# r_o[3:6,:] = r_p[3:6,:]
# dr_o[3:6,:] = dr_o[3:6,:]
# ddr_o[3:6,:] = ddr_o[3:6,:]

# ######### Inverse dynamics ########################
# q_ff = [r_o, q]
# dq_ff = [dr_o, dq]
# ddq_ff = [ddr_o, ddq]

# tau_ff = pin.rnea(model, data, q_ff, dq_ff, ddq_ff)

# ######## Parameter estimation model #############

# ## paramter vector 
# P_sus = []
# # P_sus = pinv([r_o, dr_o])*tau_ff

# ######## validation ### 
# ## given q, dq, ddq -> estimate r_o. dr_o




# # A. create a sinusoidal motion for free flyer
# F = 1 # Hz 
# omega = 2*np.pi*F
# dt = 1/100 # sampling 
# T = 5
# N = int(T/dt)

# XYZ = [0.0001, 0.0001, 0.0005] # amplitude x, y, z
# RPY  = [1*np.pi/180, 1*np.pi/180, 0.5*np.pi/180] # amplitude roll, pitch, yaw 
# A = XYZ + RPY 

# q_ff = np.zeros((6,N+1))
# v_ff = np.zeros((6,N+1))
# a_ff = np.zeros((6,N+1))
# for q_i in range(len(A)):
#     for ith in range(N+1):
#         q_ff[q_i, ith] = A[q_i]*np.sin(omega*ith*dt)
#         v_ff[q_i, ith] = omega*A[q_i]*np.cos(omega*ith*dt)
#         a_ff[q_i, ith] = -omega*omega*A[q_i]*np.sin(omega*ith*dt)

# # convert rpy to quaternion 
# quat = np.zeros((7, N+1))
# q = np.full((N+1, robot.q0.shape[0]), robot.q0)
# dq = np.zeros((N+1, robot.model.nv))
# ddq = np.zeros((N+1, robot.model.nv))

# for ith in range(N+1):
#     SE3 = cartesian_to_SE3(q_ff[:, ith])
#     quat[:, ith] = pin.SE3ToXYZQUAT(SE3)
#     q[ith, 0:7] = quat[:, ith]
#     dq[ith, 0:6] = v_ff[:, ith]
#     ddq[ith, 0:6] = a_ff[:, ith]

# tau = np.zeros((N+1, robot.model.nv))

# for ii in range(N+1):
#     tau[ii, :] = pin.rnea(robot.model, robot.data, q[ii, :], dq[ii, :], ddq[ii, :])


############ visualization ##############
# robot.initViewer(loadModel=True)
# gui = robot.viewer.gui
# gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 1)

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

# A. Consider each wheel as an individual suspension system
## general idea:
# 1. get wheel locations in base link frame, these are fixed, r_io
# 2. given a configuration q, find the wheel locations in world frame, r_ow
# which represents the moving suspension points
# 3. Retrieve coordinates data of the base from tracking system, i.e. mocap
# a) locate the markers frame in the base link frame, p_mkr
# b) get the configuration of the base in world frame, q_base
# c) filter get 1st derivs of the base configuration, dq_base
# 4. construct regressor matrix from the data, R(q, dq, r_io, r_ow)
# 5. solve for the parameters, p = R^+ * tau

# ## get wheel locations in base link frame
# from figaroh.calibration.calibration_tools import (get_rel_transform)
# wheel_locs = []
# wheel_frames = []

# robot.updateGeometryPlacements(robot.q0)
# model = robot.model
# data = robot.data
# pin.framesForwardKinematics(model, data, robot.q0)
# pin.updateFramePlacements(model, data)
# for frame in model.frames:
#     if 'wheel' in frame.name and 'joint' in frame.name:
#         wheel_frames.append(frame.name)
#     elif 'caster' in frame.name and 'joint' in frame.name:
#         wheel_frames.append(frame.name)
# for frame in wheel_frames:
#     wheel_locs.append(data.oMf[model.getFrameId(frame)])

# r_io = np.zeros((3, len(wheel_locs)))

# q = pin.randomConfiguration(model)
# q[:7] = np.array([0.01, 0.01, 0.01, 0.0 , 0.0, 0.0, 1.0])
# print(q)
# pin.framesForwardKinematics(model, data, q)
# pin.updateFramePlacements(model, data)
# print(data.oMf[model.getFrameId('base_link')].translation)
# print(data.oMf[model.getFrameId('root_joint')].translation)
# for i in range(len(wheel_locs)):
#     r_io[:, i] = wheel_locs[i].translation
#     print(wheel_frames[i], wheel_locs[i].translation)


# # get wheel locations in world frame
# # given a configuration q, find the wheel locations in world frame 
# r_ow = np.zeros((3, len(wheel_locs)))

# # for i in range(len(wheel_locs)):
#     pin.framesForwardKinematics(model, data, q)
#     pin.updateFramePlacements(model, data)
#     r_ow[:, i] = data.oMf[model.getFrameId(wheel_frames[i])].translation
