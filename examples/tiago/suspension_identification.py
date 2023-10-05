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
from os import listdir
from os.path import dirname, join, abspath, isfile

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd


from figaroh.tools.robot import Robot
from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer
from processing_utils import *
%matplotlib

# create a robot object
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
# robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    'data/tiago_schunk.urdf',
    package_dirs = package_dirs,
    isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

# A. Consider the whole base as one whole generalized suspension

######## Raw data and processing 

# # import raw data
######################################################################################
vicon_path = '/home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon'
files = [join(vicon_path, f) for f in listdir(vicon_path) if isfile(join(vicon_path, f))]
for f in files:
    if 'csv' not in f:
        files.remove(f)
files.sort()
# 0 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_around_x_vicon_1634.csv
# 1 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_around_y_vicon_1638.csv
# 2 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_around_z_vicon_1622.csv
# 3 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_calib_vicon_1508.c3d
# 4 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_calib_vicon_1607.csv
# 5 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_postures_vicon_1720.csv
# 6 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_postures_vicon_1734.csv
# 7 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_x_fold_vicon_1632.csv
# 8 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_mirror_vicon_1630.csv
# 9 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_mirror_vicon_1642.csv
# 10 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_vicon_1628.csv
# 11 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_vicon_1640.csv
# 12 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_y_fold_vicon_1636.csv
input_file = files[10]

######################################################################################
bag_path = '/home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/creps_bags/'
file_path = [f for f in listdir(bag_path)]
file_path.sort()
# 0 tiago_around_x_vicon_1634
# 1 tiago_around_y_vicon_1638
# 2 tiago_around_z_vicon_1622
# 3 tiago_calib_vicon_1508
# 4 tiago_calib_vicon_1607
# 5 tiago_x_fold_vicon_1632
# 6 tiago_xyz_mirror_vicon_1630
# 7 tiago_xyz_mirror_vicon_1642
# 8 tiago_xyz_vicon_1628
# 9 tiago_xyz_vicon_1640
# 10 tiago_y_fold_vicon_1636
input_path_joint = file_path[8]
assert input_path_joint in input_file, "ERROR: Mocap data and joitn encoder data do not match!"

######################################################################################
print('Read VICON data from ', input_file)
calib_df = read_csv_vicon(input_file)

f_res = 100
f_cutoff = 10
# selected_range = range(0, 41000)
selected_range = range(0, 5336)
plot = True # plot the coordinates
plot_raw = True
alpha = 0.25
# time_stamps_vicon = [830, 3130, 4980, 6350, 7740, 9100, 10860, 13410, 15830, 18170, 20000, 21860, 24320, 26170, 27530, 29410, 31740, 33710, 35050, 36560, 39000]
time_stamps_vicon = [0]
####################################################
## filter the data and resample
# base
base1 = filter_xyz('base1',calib_df.loc[:,['base1_x','base1_y','base1_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
base2 = filter_xyz('base2',calib_df.loc[:,['base2_x','base2_y','base2_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
base3 = filter_xyz('base3',calib_df.loc[:,['base3_x','base3_y','base3_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
# shoulder
shoulder1 = filter_xyz('shoulder1',calib_df.loc[:,['shoulder1_x','shoulder1_y','shoulder1_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
shoulder2 = filter_xyz('shoulder2',calib_df.loc[:,['shoulder2_x','shoulder2_y','shoulder2_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha) 
shoulder3 = filter_xyz('shoulder3',calib_df.loc[:,['shoulder3_x','shoulder3_y','shoulder3_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
shoulder4 = filter_xyz('shoulder4',calib_df.loc[:,['shoulder4_x','shoulder4_y','shoulder4_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
# gripper
gripper1 = filter_xyz('gripper1',calib_df.loc[:,['gripper1_x','gripper1_y','gripper1_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
gripper2 = filter_xyz('gripper2',calib_df.loc[:,['gripper2_x','gripper2_y','gripper2_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
gripper3 = filter_xyz('gripper3',calib_df.loc[:,['gripper3_x','gripper3_y','gripper3_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)

# force, moment, cop
force = filter_xyz('force',calib_df.loc[:,['F_x','F_y','F_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
moment = filter_xyz('moment',calib_df.loc[:,['M_x','M_y','M_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
cop = filter_xyz('cop',calib_df.loc[:,['COP_x','COP_y','COP_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)

####################################################
## create rigid body frame
base1_trans, base1_rot       = create_rigidbody_frame([base1, base2, base3], unit_rot=np.array([[0., 1., 0.], [1., 0., 0], [0, 0, -1]]))
base2_trans, base2_rot       = create_rigidbody_frame([base2, base1, base3], unit_rot=np.array([[0., -1., 0.], [1., 0., 0], [0, 0, 1]]))

gripper1_trans, gripper1_rot = create_rigidbody_frame([ gripper1, gripper2, gripper3,])
gripper2_trans, gripper2_rot = create_rigidbody_frame([ gripper2, gripper3, gripper1,])
gripper3_trans, gripper3_rot = create_rigidbody_frame([ gripper3, gripper2, gripper1,])

shoulder_trans, shoulder_rot = create_rigidbody_frame([shoulder1, shoulder4, shoulder2])

base = 'base2'
gripper = 'gripper3'

if base == 'base1':
    base_rot = base1_rot
    base_trans = base1_trans
elif base == 'base2':
    base_rot = base2_rot
    base_trans = base2_trans

if gripper == 'gripper1':
    gripper_rot = gripper1_rot
    gripper_trans = gripper1_trans
elif gripper == 'gripper2':
    gripper_rot = gripper2_rot
    gripper_trans = gripper2_trans
elif gripper == 'gripper3':
    gripper_rot = gripper3_rot
    gripper_trans = gripper3_trans

# plot frames
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(base1[0, 0], base1[0, 1], base1[0, 2], marker='o', label = 'base1')
ax.scatter(base2[0, 0], base2[0, 1], base2[0, 2], marker='o', label = 'base2')
ax.scatter(base3[0, 0], base3[0, 1], base3[0, 2], marker='o', label = 'base3')
ax.scatter(shoulder1[0, 0], shoulder1[0, 1], shoulder1[0, 2], marker='*', label = 'shoulder1')
ax.scatter(shoulder2[0, 0], shoulder2[0, 1], shoulder2[0, 2], marker='*', label = 'shoulder2')
ax.scatter(shoulder3[0, 0], shoulder3[0, 1], shoulder3[0, 2], marker='*', label = 'shoulder3')
ax.scatter(shoulder4[0, 0], shoulder4[0, 1], shoulder4[0, 2], marker='*', label = 'shoulder4')
ax.scatter(gripper1[0, 0], gripper1[0, 1], gripper1[0, 2], marker='>', label = 'gripper1')
ax.scatter(gripper2[0, 0], gripper2[0, 1], gripper2[0, 2], marker='>', label = 'gripper2')
ax.scatter(gripper3[0, 0], gripper3[0, 1], gripper3[0, 2], marker='>', label = 'gripper3')
ax.scatter(cop[0, 0], cop[0, 1], cop[0, 2], marker='x', label = 'cop')

ax.legend()
plot_SE3(pin.SE3.Identity())
plot_SE3(pin.SE3(base_rot[0], base_trans[0,:]), 'b_marker')
# plot_SE3(pin.SE3(shoulder_rot[0], shoulder_trans[0,:]), 's_marker')
# plot_SE3(pin.SE3(gripper_rot[0], gripper_trans[0,:]), 'g_marker')

####################################################
# convert to pinocchio frame
base_frame = list()
gripper_frame = list()
shoulder_frame = list()
for i, ti in enumerate(time_stamps_vicon):
    base_frame.append(pin.SE3(base_rot[ti], base_trans[ti, :]))
    gripper_frame.append(pin.SE3(gripper_rot[ti], gripper_trans[ti, :]))
    shoulder_frame.append(pin.SE3(shoulder_rot[ti], shoulder_trans[ti, :]))
## reproject end frame on to start frame
gripper_pos = project_frame(gripper_frame, base_frame)
gripper_shoulder = project_frame(gripper_frame, shoulder_frame)
shoulder_base = project_frame(shoulder_frame, base_frame)

######## Find transformation from base link to observed point ##############
## note: run calibration.py with the same data to get the transformation from mocap to base_link
# # translation
Mmarker_base = pin.SE3.Identity()
if base == 'base1':
    Mmarker_base.translation = np.array([0.0187, 0.2159, -0.3144])
    Mmarker_base.rotation = pin.rpy.rpyToMatrix(np.array([0.0057, -0.1964, 0.0264]))
elif base == 'base2':
    Mmarker_base.translation = np.array([0.0186, -0.1964, -0.3215])
    Mmarker_base.rotation = pin.rpy.rpyToMatrix(np.array([0.0052, 0.0118, 0.0265]))
Mbase_marker = Mmarker_base.inverse()
universe_frame = list()
xyz_u = np.zeros((len(base_rot), 3))
rpy_u = np.zeros((len(base_rot), 3))
for i in range(len(base_rot)):
    SE3_base = pin.SE3(base_rot[i], base_trans[i, :])*Mmarker_base
    universe_frame.append(SE3_base)
    xyz_u[i, :] = SE3_base.translation
    rpy_u[i, :] = pin.rpy.matrixToRpy(SE3_base.rotation)

plot_SE3(universe_frame[0], 'universe')
# plot_markertf(np.arange(len(base_rot)), xyz_u, 'xyz_u')
# plot_markertf(np.arange(len(base_rot)), rpy_u, 'rpy_u')

# # observed markers data and derivatives  
_, dxyz_u, ddxyz_u = calc_derivatives(xyz_u, f_res) 
_, drpy_u, ddrpy_u = calc_derivatives(rpy_u, f_res)

sample_range = range(3000, 4000)
NbSample = len(sample_range)
xyz_us = np.zeros_like(xyz_u[sample_range, :])
rpy_us = np.zeros_like(rpy_u[sample_range, :])
for i in range(3):
    xyz_us[:, i] = xyz_u[sample_range, i] - np.mean(xyz_u[sample_range, i])
    rpy_us[:, i] = rpy_u[sample_range, i] - np.mean(rpy_u[sample_range, i])
# plot_markertf(np.arange(NbSample), xyz_us, 'xyz_us')
# plot_markertf(np.arange(NbSample), rpy_us, 'rpy_us')

######### Regressor matrix 
reg_mat = create_R_matrix(len(sample_range), xyz_us, dxyz_u[sample_range, :], rpy_us, drpy_u[sample_range, :])  

######### Inverse dynamics 
# # get joint torques from inverse dynamics
tau_mea_vector = np.zeros(6*NbSample)
for i in range(3):
    tau_mea_vector[i*NbSample:(i+1)*NbSample] = (force[sample_range, i] - np.mean(force[sample_range, i]))
    tau_mea_vector[(3+i)*NbSample:(3+i+1)*NbSample] = moment[sample_range, i] - np.mean(moment[sample_range, i])

######## Parameter estimation model 
N_d = 3
count = 0
rmse = 1e8
var_init = 100*np.ones(2*N_d)

while rmse > 1e0 and count < 10:
    LM_solve = least_squares(cost_function, var_init, method='lm', verbose=1, \
                    args=(reg_mat[:,0:2*N_d], tau_mea_vector))
    tau_predict_vector = reg_mat[:,0:2*N_d]@LM_solve.x
    rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector)**2))
    print("*"*40)
    print("iteration: ", count, "rmse: ", rmse)
    print("solution: ", LM_solve.x)
    print("initial guess: ", var_init)
    var_init = LM_solve.x + 20000*np.random.randn(2*N_d)
    count += 1
print(LM_solve.x)
######## validation 
fig, ax = plt.subplots(N_d, 1)
tau_ylabel = ['fx (N)', 'fy(N)', 'fz(N)', 'mx(N.m)', 'my(N.m)', 'mz(N.m)']
for i in range(N_d):
    ax[i].plot(tau_mea_vector[i*len(sample_range):(i+1)*len(sample_range)], color='red')
    ax[i].plot(tau_predict_vector[i*len(sample_range):(i+1)*len(sample_range)], color='blue')
    ax[i].set_ylabel(tau_ylabel[i])
    ax[i].set_xlabel('time(ms)')
    ax[0].legend(['forceplate_measures', 'suspension_predicted_values'])
plt.show()

############ visualization
# robot.initViewer(loadModel=True)
# gui = robot.viewer.gui
# gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 1)

# robot.display(robot.q0)

# B. Consider each wheel as an individual suspension system
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
