import csv
from re import split
import pandas as pd
import numpy as np
import rospy
# import dask.dataframe as dd

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pinocchio as pin
from figaroh.tools.robot import Robot

from figaroh.calibration.calibration_tools import (get_rel_transform, rank_in_configuration)
from figaroh.tools.robot import Robot 

from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer
import time
from processing_utils import *
########################################################################
################################ SAMPLE RANGE ##########################
########################################################################

ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')

## define path to data

# folding motions
# folder = 'selected_data/single_oscilation_around_xfold_weight_2023-07-28-13-33-20' # x fold weight
# folder = 'selected_data/single_oscilation_around_yfold_weight_2023-07-28-13-12-05' # y fold weight
folder = 'calibration/optitrack_caibration_2023-09-12-16-58-30' # z weight

# swing motion
# folder = 'weight_data/single_oscilation_around_x_weight_2023-07-28-13-24-46'
# folder = 'weight_data/single_oscilation_around_y_weight_2023-07-28-13-14-03'
# folder = 'weight_data/single_oscilation_around_z_weight_2023-07-28-13-07-11'
dir_path = '/home/thanhndv212/Downloads/experiment_data/suspension/bags/20230912/' + folder + '/'
path_to_values = dir_path + 'introspection_datavalues.csv'
path_to_names = dir_path + 'introspection_datanames.csv'
path_to_shoulder = dir_path + 'natnet_rostiago_shoulderpose.csv'
path_to_base = dir_path + 'natnet_rostiago_basepose.csv'
path_to_schunk = dir_path + 'natnet_rostiago_schunkpose.csv'
########################################################################

# constant 
f_cutoff = 0.628 # lowpass filter cutoff freq/
# f_cutoff = 5
f_q = 100
f_tf = 120

########################################################################

# create a robot
robot = Robot(
    'data/tiago_schunk.urdf',
    package_dirs= package_dirs,
    # isFext=True  # add free-flyer joint at base
)

# add object to gripper
addBox_to_gripper(robot)

########################################################################
# read values from csv files
t_res, f_res, joint_names, q_abs_res, q_pos_res = get_q_arm(robot, path_to_values, path_to_names, f_cutoff)
posXYZ_base, quatXYZW_base = get_XYZQUAT_marker('base_frame', path_to_base, t_res, f_res, f_cutoff)
posXYZ_shoulder, quatXYZW_shoulder = get_XYZQUAT_marker('shoulder_frame', path_to_shoulder, t_res, f_res, f_cutoff)
posXYZ_schunk, quatXYZW_schunk = get_XYZQUAT_marker('gripper_frame', path_to_schunk, t_res, f_res, f_cutoff)
########################################################################
time_stamps = [  750,   5000,  6450,  7920,  9300, 11000, 13100, 15000,
       17500, 19250, 21200, 25500, 26890, 28270, 30000, 32600,
       34560, 35950, 37540, 39070]

posXYZ_base_cal = posXYZ_base[time_stamps, :]
quatXYZW_base_cal = quatXYZW_base[time_stamps, :]
base_frame = np.concatenate((posXYZ_base_cal, quatXYZW_base_cal), axis=1)

posXYZ_shoulder_cal = posXYZ_shoulder[time_stamps, :]
quatXYZW_shoulder_cal = quatXYZW_shoulder[time_stamps, :]
shoulder_frame = np.concatenate((posXYZ_shoulder_cal, quatXYZW_shoulder_cal), axis=1)

posXYZ_schunk_cal = posXYZ_schunk[time_stamps, :]
quatXYZW_schunk_cal = quatXYZW_schunk[time_stamps, :]
schunk_frame = np.concatenate((posXYZ_schunk_cal, quatXYZW_schunk_cal), axis=1)

# normalize quaternion
def convert_XYZQUAT_to_SE3norm(var, se3_norm=True):
    """ Normalize quaternion and convert to SE3
    """
    SE3_out = pin.SE3.Identity()
    SE3_out.translation = var[0:3]

    quat_norm = pin.Quaternion(var[3:7]).normalize()
    if se3_norm:
        SE3_out.rotation = quat_norm.toRotationMatrix()
        return SE3_out
    else:
        var[3:7] = quat_norm.coeffs()
        return var
rotated = True
rot_mat = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
for ii in range(len(time_stamps)):
    if rotated:
        base_se3 = convert_XYZQUAT_to_SE3norm(base_frame[ii, :], se3_norm=True)
        print('base translation before: ', base_se3.translation)
        base_se3.translation = rot_mat@base_se3.translation
        print('base translation after: ', base_se3.translation)
        # base_se3.rotation = rot_mat@base_se3.rotation
        base_frame[ii, :] = pin.SE3ToXYZQUAT(base_se3)
        shoulder_se3 = convert_XYZQUAT_to_SE3norm(shoulder_frame[ii, :], se3_norm=True)
        print('shoulder translation before: ', shoulder_se3.translation)
        shoulder_se3.translation = rot_mat@shoulder_se3.translation
        print('shoulder translation after: ', shoulder_se3.translation)
        # shoulder_se3.rotation = rot_mat@shoulder_se3.rotation
        shoulder_frame[ii, :] = pin.SE3ToXYZQUAT(shoulder_se3)
        schunk_se3 = convert_XYZQUAT_to_SE3norm(schunk_frame[ii, :], se3_norm=True)
        schunk_se3.translation = rot_mat@schunk_se3.translation
        # schunk_se3.rotation = rot_mat@schunk_se3.rotation
        schunk_frame[ii, :] = pin.SE3ToXYZQUAT(schunk_se3)
    else:
        base_frame[ii, :] = convert_XYZQUAT_to_SE3norm(base_frame[ii, :], se3_norm=False)
        shoulder_frame[ii, :] = convert_XYZQUAT_to_SE3norm(shoulder_frame[ii, :], se3_norm=False)
        schunk_frame[ii, :] = convert_XYZQUAT_to_SE3norm(schunk_frame[ii, :], se3_norm=False)



xyz = project_frame(schunk_frame, base_frame)

# xyz = rot_mat.dot(xyz.T).T

active_joints = ["torso_lift_joint",
                 "arm_1_joint",
                 "arm_2_joint",
                 "arm_3_joint",
                 "arm_4_joint",
                 "arm_5_joint",
                 "arm_6_joint",
                 "arm_7_joint",
                 ]
actJoitn_idx = []
for act_j in active_joints:
    joint_idx = robot.model.getJointId(act_j)
    actJoitn_idx.append(robot.model.joints[joint_idx].idx_q)

q_cal = q_pos_res[time_stamps, :]
q_cal = q_cal[:, actJoitn_idx]

# projected gripper frame on base frame/shoudler frame
path_to_save = join(dirname(str(abspath(__file__))),
    f"data/optitrack_calibration.csv")

def save_selected_data(xyz, q, path_to_save):
    headers = ["x1", "y1", "z1"] + active_joints
    with open(path_to_save, "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(headers)
        for i in range(q.shape[0]):
            row = list(np.concatenate((xyz[i, :],
                                      q[i, :])))
            w.writerow(row)

# save_selected_data(xyz, q_cal, path_to_save)
%matplotlib
for tf in range(len(time_stamps)):
    se3_base = pin.XYZQUATToSE3(base_frame[tf,:])
    se3_shoulder = pin.XYZQUATToSE3(shoulder_frame[tf,:])
    se3_schunk = pin.XYZQUATToSE3(schunk_frame[tf,:])

    plot_SE3(se3_base, 'b')
    plot_SE3(se3_shoulder, 's' )
plot_SE3(se3_schunk, 'sch')