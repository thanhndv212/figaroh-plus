import numpy as np

# import dask.dataframe as dd
from scipy.optimize import least_squares

import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pinocchio as pin
from figaroh.tools.robot import Robot

from figaroh.calibration.calibration_tools import (
    get_rel_transform,
    rank_in_configuration,
)
from figaroh.tools.robot import Robot

from processing_utils import *

########################################################################
################################ Data processing ##########################
########################################################################

ros_package_path = os.getenv("ROS_PACKAGE_PATH")
package_dirs = ros_package_path.split(":")

## define path to data

# folding motions
# folder = 'selected_data/single_oscilation_around_xfold_weight_2023-07-28-13-33-20' # x fold weight
# folder = 'selected_data/single_oscilation_around_yfold_weight_2023-07-28-13-12-05' # y fold weight
# folder = 'selected_data/single_oscilation_around_z_weight_2023-07-28-13-07-11' # z weight

# 20230912
# folder = '/adream/selected_suspension/sinu_motion_around_z_weight_2023-09-12-16-44-18'
# folder = '/adream/selected_suspension/sinu_motion_around_z_weight_2023-09-12-16-41-48'
folder = "/adream/selected_suspension/sinu_motion_around_x_fold_weight_2023-09-12-16-46-52"


# swing motion
# folder = 'weight_data/single_oscilation_around_x_weight_2023-07-28-13-24-46'
# folder = 'weight_data/single_oscilation_around_y_weight_2023-07-28-13-14-03'
# folder = 'weight_data/single_oscilation_around_z_weight_2023-07-28-13-07-11'

dir_path = (
    "/home/thanhndv212/Downloads/experiment_data/suspension/bags"
    + folder
    + "/"
)
path_to_values = dir_path + "introspection_datavalues.csv"
path_to_names = dir_path + "introspection_datanames.csv"
path_to_tf = dir_path + "natnet_rostiago_shoulderpose.csv"
path_to_base = dir_path + "natnet_rostiago_basepose.csv"
path_to_object = dir_path + "natnet_rostiago_objectpose.csv"

########################################################################

# constant
f_cutoff = 0.628  # lowpass filter cutoff freq/
f_q = 100
f_tf = 120

# only take samples in a range
sample_range = range(3000, 4000)
sample_range_ext = range(sample_range[0] - 10, sample_range[-1] + 1 + 10)
NbSample = len(sample_range)

########################################################################

# create a robot
robot = Robot(
    "data/tiago_48.urdf",
    package_dirs=package_dirs,
    # isFext=True  # add free-flyer joint at base
)

# add object to gripper
addBox_to_gripper(robot)

# retrieve marker data
t_res, f_res, joint_names, q_abs_res, q_pos_res = get_q_arm(
    robot, path_to_values, path_to_names, f_cutoff
)
posXYZ_res, quatXYZW_res = get_XYZQUAT_marker(
    "shoulder", path_to_tf, t_res, f_res, f_cutoff
)
posXYZ_ee, quatXYZW_ee = get_XYZQUAT_marker(
    "gripper", path_to_object, t_res, f_res, f_cutoff
)
posXYZ_base, quatXYZW_base = get_XYZQUAT_marker(
    "base", path_to_base, t_res, f_res, f_cutoff
)

rpy_shoulder = convert_quat_to_rpy(quatXYZW_res)
rpy_ee = convert_quat_to_rpy(quatXYZW_ee)
rpy_base = convert_quat_to_rpy(quatXYZW_base)

########################################################################

# processed data
t_sample = t_res[sample_range]
q_arm, dq_arm, ddq_arm = calc_vel_acc(
    robot, q_abs_res, sample_range, joint_names, f_res, f_cutoff
)
(
    q_marker,
    dq_marker,
    ddq_marker,
    rpy_marker,
    Mmarker0_mocap,
    q_marker_ext,
) = calc_fb_vel_acc(posXYZ_res, quatXYZW_res, sample_range, f_res)
(
    q_markerBase,
    dq_markerBase,
    ddq_markerBase,
    rpy_markerBase,
    Mmarker0_mocapBase,
    q_marker_extBase,
) = calc_fb_vel_acc(posXYZ_base, quatXYZW_base, sample_range, f_res)
# u_marker, s_marker = find_isa(q_marker, dq_marker)
# u_markerBase, s_markerBase = find_isa(q_markerBase, dq_markerBase)

########################################################################

# create floating base model
tiago_fb = Robot(
    "data/tiago_48.urdf",
    package_dirs=package_dirs,
    isFext=True,  # add free-flyer joint at base
)
addBox_to_gripper(tiago_fb)

#######################################################################

# identification 2
var_init_fb = np.zeros(19)
var_init_fb[0:3] = np.array([0.0, 0.0, 0.23])
var_init_fb[3:7] = np.array([0.7071, 0, 0, 0.7071])
var_init_fb[-12:] = 100 * np.ones(12)
sol_found = False
UNKNOWN_BASE = "xyzquat"
base_input = None


Mmarker0 = Mmarker0_mocap
q_m = q_marker


LM_solve = least_squares(
    cost_function_fb,
    var_init_fb,
    method="lm",
    verbose=1,
    args=(
        tiago_fb,
        Mmarker0,
        q_m,
        q_arm,
        dq_arm,
        ddq_arm,
        f_res,
        sol_found,
        UNKNOWN_BASE,
        base_input,
    ),
)

print(LM_solve)

# # estimate from solution
(
    tau_predict_vector,
    tau_mea_vector,
    q_base,
    dq_base,
    ddq_base,
    rpy,
    reg_matrix,
) = cost_function_fb(
    LM_solve.x,
    tiago_fb,
    Mmarker0,
    q_m,
    q_arm,
    dq_arm,
    ddq_arm,
    f_res,
    True,
    UNKNOWN_BASE,
    base_input,
)
error_tau = np.abs(tau_predict_vector - tau_mea_vector)

print("sample_range: ", sample_range)


# plot dynamics
fig, ax = plt.subplots(6, 1)
tau_ylabel = ["fx (N)", "fy(N)", "fz(N)", "mx(N.m)", "my(N.m)", "mz(N.m)"]
for i in range(6):
    ax[i].plot(
        np.arange(len(sample_range)),
        tau_mea_vector[i * len(sample_range) : (i + 1) * len(sample_range)],
        color="red",
    )
    ax[i].plot(
        np.arange(len(sample_range)),
        tau_predict_vector[
            i * len(sample_range) : (i + 1) * len(sample_range)
        ],
        color="blue",
    )
    ax[i].set_ylabel(tau_ylabel[i])
    ax[i].set_xlabel("time(ms)")
    ax[i].bar(
        np.arange(len(sample_range)),
        error_tau[i * len(sample_range) : (i + 1) * len(sample_range)],
    )
    ax[0].legend(["pinocchio_estimate", "suspension_predicted"])
plt.show()


# base_z = 0.3*np.ones(1)
# base_xyz = np.concatenate((LM_solve.x[0:2], base_z), axis=0)
base_xyz = LM_solve.x[0:3]
# base_quat = np.array([0.7071, 0, 0, 0.7071])
base_quat = LM_solve.x[3:7]
Mbase_marker = convert_XYZQUAT_to_SE3norm(
    np.concatenate((base_xyz, base_quat), axis=0)
)
print(Mbase_marker.translation)
print(pin.rpy.matrixToRpy(Mbase_marker.rotation) * 180 / np.pi)

# tau_mea = np.reshape(tau_mea_vector, (NbSample, 6), order='F')

# def find_fft_func(Y):
#     Y = np.fft.fft(Y)
#     ifft = np.fft.ifft(Y)
#     return ifft

# q_marker_est = estimate_marker_pose_from_base(Mbase_marker, Mmarker0_mocap, q_base)

# suspension_param = LM_solve.x[2:14]
# def compare_q_base(ij,k):
#     k_x = suspension_param[2*ij]
#     c_x = suspension_param[2*ij + 1]

#     tau = tau_mea[:, ij]

#     fx_t = find_fft_func(tau)

#     tx_base_est = np.zeros(NbSample)
#     dt = 1/f_res
#     for i in range(NbSample):
#         tx_base_est[i] = np.exp(-2*(c_x/k_x)*dt*i)*(tau[i-1])*k
#     plt.plot(tx_base_est, label='tx_marker_est')
#     plt.plot(q_base[:,ij], label='q_marker')
#     plt.legend()
