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

from os import listdir
from os.path import join, abspath, isfile
from matplotlib import pyplot as plt
import numpy as np
import tiago_utils.suspension.processing_utils as pu
from tiago_utils.tiago_tools import load_robot
from scipy.optimize import least_squares
from figaroh.calibration.calibration_tools import cartesian_to_SE3


# vicon at creps

tiago_fb = load_robot(
    abspath("urdf/tiago_48_schunk.urdf"),
    isFext=True,
    load_by_urdf=True,
)

model = tiago_fb.model
data = tiago_fb.data

pu.initiating_robot(tiago_fb)

dir_path = "/media/thanhndv212/Cooking/processed_data/tiago/develop/data/identification/suspension/creps/creps_bags/tiago_xyz_mirror_vicon_1642/"
input_file = dir_path + "tiago_xyz_mirror_vicon_1642.csv"
marker_data = pu.process_vicon_data(input_file)

# ######## Find transformation from base link to observed marker ##############
# ## note: run calibration.py with the same data to get the transformation from mocap to base_link
# # # translation
marker_base = dict()
marker_base["base1"] = np.array(
    [0.0187, 0.2159, -0.3144, 0.0057, -0.1964, 0.0264]
)
marker_base["base2"] = np.array(
    [0.0186, -0.1964, -0.3215, 0.0052, 0.0118, 0.0265]
)


def calc_baselink_pose(marker_data, base_marker_name: str):
    Mmarker_base = cartesian_to_SE3(marker_base[base_marker_name])
    Mbase_marker = Mmarker_base.inverse()
    universe_frame = list()
    [base_trans, base_rot] = marker_data[base_marker_name]
    xyz_u = np.zeros((len(base_rot), 3))
    rpy_u = np.zeros((len(base_rot), 3))
    quat_u = np.zeros((len(base_rot), 4))
    for i in range(len(base_rot)):
        SE3_base = pin.SE3(base_rot[i], base_trans[i, :]) * Mmarker_base
        universe_frame.append(SE3_base)
        xyz_u[i, :] = SE3_base.translation
        rpy_u[i, :] = pin.rpy.matrixToRpy(SE3_base.rotation)
        quat_u[i, :] = pin.Quaternion(SE3_base.rotation).coeffs()
    return xyz_u, rpy_u, quat_u


xyz_u, rpy_u, quat_u = calc_baselink_pose(marker_data, "base2")

# # # observed markers data and derivatives
# _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res)
# _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res)

# sample_range = range(3000, 4000)
# NbSample = len(sample_range)
# xyz_us = np.zeros_like(xyz_u[sample_range, :])
# rpy_us = np.zeros_like(rpy_u[sample_range, :])
# for i in range(3):
#     xyz_us[:, i] = xyz_u[sample_range, i] - np.mean(xyz_u[sample_range, i])
#     print("mean of sample range:", np.mean(xyz_u[sample_range, i]))
#     rpy_us[:, i] = rpy_u[sample_range, i] - np.mean(rpy_u[sample_range, i])


# # ######################################################################################
# path_to_values = dir_path + "introspection_datavalues.csv"
# path_to_names = dir_path + "introspection_datanames.csv"

# # # create a robot
# # tiago = load_robot(
# #     abspath("urdf/tiago_48_schunk.urdf"),
# #     isFext=True,
# #     load_by_urdf=True,
# # )
# # add object to gripper
# pu.addBox_to_gripper(tiago_fb)

# # read values from csv files
# t_res, f_res, joint_names, q_abs_res, q_pos_res = pu.get_q_arm(
#     tiago_fb, path_to_values, path_to_names, f_cutoff
# )
# active_joints = [
#     "torso_lift_joint",
#     "arm_1_joint",
#     "arm_2_joint",
#     "arm_3_joint",
#     "arm_4_joint",
#     "arm_5_joint",
#     "arm_6_joint",
#     "arm_7_joint",
# ]
# actJoint_idx = []
# actJoint_idv = []
# for act_j in active_joints:
#     joint_idx = tiago_fb.model.getJointId(act_j)
#     actJoint_idx.append(tiago_fb.model.joints[joint_idx].idx_q)
#     actJoint_idv.append(tiago_fb.model.joints[joint_idx].idx_v)
# selected_range = range(20, t_res.shape[0]-20)
# # sample_range = selected_range
# q_arm, dq_arm, ddq_arm = pu.calc_vel_acc(
#     tiago_fb, q_pos_res, selected_range, joint_names, f_res, f_cutoff
# )
# tau_id = pin.rnea(model, data, q_arm[0, :], dq[0, :], ddq_arm[0, :])
# for ii, joint in enumerate(tiago_fb.model.joints):
#     print(
#         "{}. Joint name: {}, mass (kg): {}, joint torque (N/N.m): ".format(
#             ii, tiago_fb.model.names[ii], np.round(data.mass[ii], 2)
#         )
#     )
#     if ii == 1:  # root joint
#         print(
#             "grf est: ", np.around(tau_id[0:3], 1), np.around(tau_id[3:6], 2)
#         )
#         print(
#             "grf measured: ", np.mean(force, axis=0), np.mean(moment, axis=0)
#         )
#     print(np.round(tau_id[joint.idx_v]))
# plt.plot(force[:, 2])
# plt.show()

# calculate joint torque by inverse dynamic

# ######### Regressor matrix
# reg_mat = pu.create_R_matrix(
#     len(sample_range),
#     xyz_us,
#     dxyz_u[sample_range, :],
#     rpy_us,
#     drpy_u[sample_range, :],
# )

# ######### Inverse dynamics
# # # get joint torques from inverse dynamics
# tau_mea_vector = np.zeros(6 * NbSample)
# for i in range(3):
#     tau_mea_vector[i * NbSample : (i + 1) * NbSample] = force[
#         sample_range, i
#     ] - np.mean(force[sample_range, i])
#     tau_mea_vector[(3 + i) * NbSample : (3 + i + 1) * NbSample] = moment[
#         sample_range, i
#     ] - np.mean(moment[sample_range, i])

# ######## Parameter estimation model
# N_d = 3
# count = 0
# rmse = 1e8
# var_init = 100 * np.ones(2 * N_d)

# while rmse > 1e-1 and count < 10:
#     LM_solve = least_squares(
#         pu.cost_function,
#         var_init,
#         method="lm",
#         verbose=1,
#         args=(reg_mat[:, 0 : 2 * N_d], tau_mea_vector),
#     )
#     tau_predict_vector = reg_mat[:, 0 : 2 * N_d] @ LM_solve.x
#     rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector) ** 2))
#     print("*" * 40)
#     print("iteration: ", count, "rmse: ", rmse)
#     print("solution: ", LM_solve.x)
#     print("initial guess: ", var_init)
#     var_init = LM_solve.x + 20000 * np.random.randn(2 * N_d)
#     count += 1
# print(LM_solve.x)
# ######## validation
# fig, ax = plt.subplots(N_d, 1)
# tau_ylabel = ["fx (N)", "fy(N)", "fz(N)", "mx(N.m)", "my(N.m)", "mz(N.m)"]
# for i in range(N_d):
#     ax[i].plot(
#         tau_mea_vector[i * len(sample_range) : (i + 1) * len(sample_range)],
#         color="red",
#     )
#     ax[i].plot(
#         tau_predict_vector[
#             i * len(sample_range) : (i + 1) * len(sample_range)
#         ],
#         color="blue",
#     )
#     ax[i].set_ylabel(tau_ylabel[i])
#     ax[i].set_xlabel("time(ms)")
#     ax[0].legend(["forceplate_measures", "suspension_predicted_values"])
# plt.show()

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


# ########################################################################
# ################################ optitrack ##########################
# ########################################################################

# import numpy as np
# from scipy.optimize import least_squares
# import os
# from matplotlib import pyplot as plt
# import pinocchio as pin
# from figaroh.tools.robot import Robot
# from processing_utils import *

# ros_package_path = os.getenv("ROS_PACKAGE_PATH")
# package_dirs = ros_package_path.split(":")

# ## define path to data

# # folding motions
# # folder = 'selected_data/single_oscilation_around_xfold_weight_2023-07-28-13-33-20' # x fold weight
# # folder = 'selected_data/single_oscilation_around_yfold_weight_2023-07-28-13-12-05' # y fold weight
# # folder = 'selected_data/single_oscilation_around_z_weight_2023-07-28-13-07-11' # z weight

# # 20230912
# # folder = '/adream/selected_suspension/sinu_motion_around_z_weight_2023-09-12-16-44-18'
# # folder = '/adream/selected_suspension/sinu_motion_around_z_weight_2023-09-12-16-41-48'
# folder = "/adream/sinu_motion_around_x_fold_weight_2023-09-12-16-46-52"


# # swing motion
# # folder = 'weight_data/single_oscilation_around_x_weight_2023-07-28-13-24-46'
# # folder = 'weight_data/single_oscilation_around_y_weight_2023-07-28-13-14-03'
# # folder = 'weight_data/single_oscilation_around_z_weight_2023-07-28-13-07-11'

# dir_path = (
#     "/home/thanhndv212/Downloads/experiment_data/suspension/bags"
#     + folder
#     + "/"
# )
# path_to_values = dir_path + "introspection_datavalues.csv"
# path_to_names = dir_path + "introspection_datanames.csv"
# path_to_tf = dir_path + "natnet_rostiago_shoulderpose.csv"
# path_to_base = dir_path + "natnet_rostiago_basepose.csv"
# path_to_object = dir_path + "natnet_rostiago_objectpose.csv"

# ########################################################################

# # constant
# f_cutoff = 0.628  # lowpass filter cutoff freq/
# f_q = 100
# f_tf = 120

# # only take samples in a range
# sample_range = range(3000, 4000)
# sample_range_ext = range(sample_range[0] - 10, sample_range[-1] + 1 + 10)
# NbSample = len(sample_range)

# ########################################################################

# # create a robot
# robot = Robot(
#     "data/tiago_48.urdf",
#     package_dirs=package_dirs,
#     # isFext=True  # add free-flyer joint at base
# )

# # add object to gripper
# pu.addBox_to_gripper(robot)

# # retrieve marker data
# t_res, f_res, joint_names, q_abs_res, q_pos_res = pu.get_q_arm(
#     robot, path_to_values, path_to_names, f_cutoff
# )
# posXYZ_res, quatXYZW_res = get_XYZQUAT_marker(
#     "shoulder", path_to_tf, t_res, f_res, f_cutoff
# )
# posXYZ_ee, quatXYZW_ee = get_XYZQUAT_marker(
#     "gripper", path_to_object, t_res, f_res, f_cutoff
# )
# posXYZ_base, quatXYZW_base = get_XYZQUAT_marker(
#     "base", path_to_base, t_res, f_res, f_cutoff
# )

# rpy_shoulder = convert_quat_to_rpy(quatXYZW_res)
# rpy_ee = convert_quat_to_rpy(quatXYZW_ee)
# rpy_base = convert_quat_to_rpy(quatXYZW_base)

# ########################################################################

# # processed data
# t_sample = t_res[sample_range]
# q_arm, dq_arm, ddq_arm = pu.calc_vel_acc(
#     robot, q_abs_res, sample_range, joint_names, f_res, f_cutoff
# )
# (
#     q_marker,
#     dq_marker,
#     ddq_marker,
#     rpy_marker,
#     Mmarker0_mocap,
#     q_marker_ext,
# ) = pu.calc_fb_vel_acc(posXYZ_res, quatXYZW_res, sample_range, f_res)
# (
#     q_markerBase,
#     dq_markerBase,
#     ddq_markerBase,
#     rpy_markerBase,
#     Mmarker0_mocapBase,
#     q_marker_extBase,
# ) = pu.calc_fb_vel_acc(posXYZ_base, quatXYZW_base, sample_range, f_res)
# # u_marker, s_marker = find_isa(q_marker, dq_marker)
# # u_markerBase, s_markerBase = find_isa(q_markerBase, dq_markerBase)

# ########################################################################

# # create floating base model
# tiago_fb = Robot(
#     "data/tiago_48.urdf",
#     package_dirs=package_dirs,
#     isFext=True,  # add free-flyer joint at base
# )
# pu.addBox_to_gripper(tiago_fb)

# #######################################################################

# # identification 2
# var_init_fb = np.zeros(19)
# var_init_fb[0:3] = np.array([0.0, 0.0, 0.23])
# var_init_fb[3:7] = np.array([0.7071, 0, 0, 0.7071])
# var_init_fb[-12:] = 100 * np.ones(12)
# sol_found = False
# UNKNOWN_BASE = "xyzquat"
# base_input = None


# Mmarker0 = Mmarker0_mocap
# q_m = q_marker


# LM_solve = least_squares(
#     pu.cost_function_fb,
#     var_init_fb,
#     method="lm",
#     verbose=1,
#     args=(
#         tiago_fb,
#         Mmarker0,
#         q_m,
#         q_arm,
#         dq_arm,
#         ddq_arm,
#         f_res,
#         sol_found,
#         UNKNOWN_BASE,
#         base_input,
#     ),
# )

# print(LM_solve)

# # # estimate from solution
# (
#     tau_predict_vector,
#     tau_mea_vector,
#     q_base,
#     dq_base,
#     ddq_base,
#     rpy,
#     reg_matrix,
# ) = pu.cost_function_fb(
#     LM_solve.x,
#     tiago_fb,
#     Mmarker0,
#     q_m,
#     q_arm,
#     dq_arm,
#     ddq_arm,
#     f_res,
#     True,
#     UNKNOWN_BASE,
#     base_input,
# )
# error_tau = np.abs(tau_predict_vector - tau_mea_vector)

# print("sample_range: ", sample_range)


# # plot dynamics
# fig, ax = plt.subplots(6, 1)
# tau_ylabel = ["fx (N)", "fy(N)", "fz(N)", "mx(N.m)", "my(N.m)", "mz(N.m)"]
# for i in range(6):
#     ax[i].plot(
#         np.arange(len(sample_range)),
#         tau_mea_vector[i * len(sample_range) : (i + 1) * len(sample_range)],
#         color="red",
#     )
#     ax[i].plot(
#         np.arange(len(sample_range)),
#         tau_predict_vector[
#             i * len(sample_range) : (i + 1) * len(sample_range)
#         ],
#         color="blue",
#     )
#     ax[i].set_ylabel(tau_ylabel[i])
#     ax[i].set_xlabel("time(ms)")
#     ax[i].bar(
#         np.arange(len(sample_range)),
#         error_tau[i * len(sample_range) : (i + 1) * len(sample_range)],
#     )
#     ax[0].legend(["pinocchio_estimate", "suspension_predicted"])
# plt.show()


# # base_z = 0.3*np.ones(1)
# # base_xyz = np.concatenate((LM_solve.x[0:2], base_z), axis=0)
# base_xyz = LM_solve.x[0:3]
# # base_quat = np.array([0.7071, 0, 0, 0.7071])
# base_quat = LM_solve.x[3:7]
# Mbase_marker = pu.convert_XYZQUAT_to_SE3norm(
#     np.concatenate((base_xyz, base_quat), axis=0)
# )
# print(Mbase_marker.translation)
# print(pin.rpy.matrixToRpy(Mbase_marker.rotation) * 180 / np.pi)

# # tau_mea = np.reshape(tau_mea_vector, (NbSample, 6), order='F')

# # def find_fft_func(Y):
# #     Y = np.fft.fft(Y)
# #     ifft = np.fft.ifft(Y)
# #     return ifft

# # q_marker_est = estimate_marker_pose_from_base(Mbase_marker, Mmarker0_mocap, q_base)

# # suspension_param = LM_solve.x[2:14]
# # def compare_q_base(ij,k):
# #     k_x = suspension_param[2*ij]
# #     c_x = suspension_param[2*ij + 1]

# #     tau = tau_mea[:, ij]

# #     fx_t = find_fft_func(tau)

# #     tx_base_est = np.zeros(NbSample)
# #     dt = 1/f_res
# #     for i in range(NbSample):
# #         tx_base_est[i] = np.exp(-2*(c_x/k_x)*dt*i)*(tau[i-1])*k
# #     plt.plot(tx_base_est, label='tx_marker_est')
# #     plt.plot(q_base[:,ij], label='q_marker')
# #     plt.legend()
#
