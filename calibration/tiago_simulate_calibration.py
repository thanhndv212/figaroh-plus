import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import least_squares
import numpy as np
import random

import pandas as pd
import csv
import time

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num


from calibration_tools import (
    extract_expData,
    extract_expData4Mkr,
    get_param,
    init_var,
    get_PEE_fullvar,
    get_PEE_var,
    get_geoOffset,
    get_jointOffset,
    get_PEE,
    Calculate_kinematics_model,
    Calculate_identifiable_kinematics_model,
    Calculate_base_kinematics_regressor)

# 1/ Load robot model and create a dictionary containing reserved constants

robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

NbSample = 50
param = get_param(robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=4)

#############################################################

# 2/ Base parameters calculation
q_rand = []
Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e = Calculate_base_kinematics_regressor(
    q_rand, model, data, param)

# naming for eeframe markers
PEE_names = []
for i in range(param['NbMarkers']):
    PEE_names.extend(['pEEx_%d' % (i+1), 'pEEy_%d' % (i+1), 'pEEz_%d' % (i+1)])
params_name = paramsrand_base + PEE_names
for i in params_name:
    print(i)

#############################################################

# 3/ Data collection/generation
dataSet = 'experimental'  # choose data source 'sample' or 'experimental'
if dataSet == 'sample':
    # create artificial offsets
    var_sample, nvars_sample = init_var(param, mode=1)
    print("%d var_sample: " % nvars_sample, var_sample)

    # create sample configurations
    q_sample = np.empty((param['NbSample'], model.nq))

    for i in range(param['NbSample']):
        config = param['q0']
        config[param['Ind_joint']] = pin.randomConfiguration(model)[
            param['Ind_joint']]
        q_sample[i, :] = config

    # create simulated end effector coordinates measures (PEEm)
    PEEm_sample = get_PEE_fullvar(
        var_sample, q_sample, model, data, param)

    q_LM = np.copy(q_sample)
    PEEm_LM = np.copy(PEEm_sample)

elif dataSet == 'experimental':
    # read csv file
    # path = '/home/dvtnguyen/calibration/figaroh/data/tiago/tiago_nov_30_64.csv'
    path = abspath('data/tiago/tiago_nov_30_64.csv')
    # path = '/home/thanhndv212/Cooking/figaroh/data/tiago/exp_data_nov_64_3011.csv'

    PEEm_exp, q_exp = extract_expData4Mkr(path, param)

    q_LM = np.copy(q_exp)
    PEEm_LM = np.copy(PEEm_exp)

# Remove potential outliers which identified from previous calibration

print(np.shape(PEEm_LM))

print('updated number of samples: ', param['NbSample'])


#############################################################

# 4/ Given a model and configurations (input), end effector positions/locations
# (output), solve an optimization problem to find offset params as variables

# # NON-LINEAR model with Levenberg-Marquardt #################
"""
    - minimize the difference between measured coordinates of end-effector
    and its estimated values from DGM by Levenberg-Marquardt
"""


coeff = 1e-3
write_to_file = True


def cost_func(var, coeff, q, model, data, param,  PEEm):
    PEEe = get_PEE_fullvar(var, q, model, data, param, noise=False)
    res_vect = np.append((PEEm - PEEe), np.sqrt(coeff)
                         * var[6:-param['NbMarkers']*3])
    # res_vect = (PEEm - PEEe)
    return res_vect


# initial guess
# mode = 1: random seed [-0.01, 0.01], mode = 0: init guess = 0
var_0, nvars = init_var(param, mode=0)
print("initial guess: ", var_0)

# solve
LM_solve = least_squares(cost_func, var_0,  method='lm', verbose=1,
                         args=(coeff, q_LM, model, data, param, PEEm_LM))

#############################################################

# 5/ Result analysis
res = LM_solve.x
# PEE estimated by solution
PEEe_sol = get_PEE_fullvar(res, q_LM, model, data, param, noise=False)

# root mean square error
rmse = np.sqrt(np.mean((PEEe_sol-PEEm_LM)**2))

print("solution: ", res)
print("minimized cost function: ", rmse)
print("optimality: ", LM_solve.optimality)

# calculate standard deviation of estimated parameter ( Khalil chapter 11)
sigma_ro_sq = (LM_solve.cost**2) / \
    (param['NbSample']*param['calibration_index'] - nvars)
J = LM_solve.jac
C_param = sigma_ro_sq*np.linalg.pinv(np.dot(J.T, J))
std_dev = []
std_pctg = []
for i in range(nvars):
    std_dev.append(np.sqrt(C_param[i, i]))
    std_pctg.append(abs(np.sqrt(C_param[i, i])/res[i]))
print("standard deviation: ", std_dev)


#############################################################

# 6/ Plot results

# calculate difference between estimated data and measured data
delta_PEE = PEEe_sol - PEEm_LM
PEE_xyz = delta_PEE.reshape((param['NbMarkers']*3, param["NbSample"]))
PEE_dist = np.zeros((param['NbMarkers'], param["NbSample"]))
for i in range(param["NbMarkers"]):
    for j in range(param["NbSample"]):
        PEE_dist[i, j] = np.sqrt(
            PEE_xyz[i*3, j]**2 + PEE_xyz[i*3 + 1, j]**2 + PEE_xyz[i*3 + 2, j]**2)

# detect "bad" data (outlierrs) => remove outliers, recalibrate 
del_list = []
scatter_size = np.zeros_like(PEE_dist)
for i in range(param['NbMarkers']):
    for k in range(param['NbSample']):
        if PEE_dist[i, k] > 0.02:
            del_list.append((i, k))
    scatter_size[i, :] = 20*PEE_dist[i, :]/np.min(PEE_dist[i, :])
print("indices of samples with >2 cm deviation: ", del_list)

# # 1// Errors between estimated position and measured position of markers

fig1, ax1 = plt.subplots(param['NbMarkers'], 1)
fig1.suptitle(
    "Relative positional errors between estimated markers and measured markers (m) by samples ")
colors = ['blue',
          'red',
          'yellow',
          'purple'
          ]
if param['NbMarkers'] == 1:
    ax1.bar(np.arange(param['NbSample']), PEE_dist[i, :])
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Error (meter)')
    ax1.grid()
else:
    for i in range(param['NbMarkers']):
        ax1[i].bar(np.arange(param['NbSample']),
                   PEE_dist[i, :], color=colors[i])
        ax1[i].set_xlabel('Sample')
        ax1[i].set_ylabel('Error of marker %s (meter)' % (i+1))
        ax1[i].grid()

# # 2// plot 3D measured poses and estimated
fig2 = plt.figure(2)
fig2.suptitle("Visualization of estimated poses and measured pose in Cartesian")
ax2 = fig2.add_subplot(111, projection='3d')
PEEm_LM2d = PEEm_LM.reshape((param['NbMarkers']*3, param["NbSample"]))
PEEe_sol2d = PEEe_sol.reshape((param['NbMarkers']*3, param["NbSample"]))
for i in range(param['NbMarkers']):
    ax2.scatter3D(PEEm_LM2d[i*3, :], PEEm_LM2d[i*3+1, :],
                  PEEm_LM2d[i*3+2, :], marker='^', color='blue')
    ax2.scatter3D(PEEe_sol2d[i*3, :], PEEe_sol2d[i*3+1, :],
                  PEEe_sol2d[i*3+2, :], marker='o', color='red')
ax2.set_xlabel('X - front (meter)')
ax2.set_ylabel('Y - side (meter)')
ax2.set_zlabel('Z - height (meter)')
ax2.grid()

# 3// visualize relative deviation between measure and estimate
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
for i in range(param['NbMarkers']):
    ax3.scatter3D(PEEm_LM2d[i*3, :], PEEm_LM2d[i*3+1, :],
                  PEEm_LM2d[i*3+2, :], s=scatter_size[i, :], color='green')
ax3.grid()

# 4// joint configurations within range bound
fig4 = plt.figure()
fig4.suptitle("Joint configurations with joint bounds")
ax4 = fig4.add_subplot(111, projection='3d')
lb = ub = []
for j in param['Ind_joint']:
    # model.names does not accept index type of numpy int64
    # and model.lowerPositionLimit index lag to model.names by 1
    lb = np.append(lb, model.lowerPositionLimit[j])
    ub = np.append(ub, model.upperPositionLimit[j])
q_actJoint = q_LM[:, param['Ind_joint']]
sample_range = np.arange(param['NbSample'])
for i in range(len(param['actJoint_idx'])):
    ax4.scatter3D(q_actJoint[:, i], sample_range, i)
for i in range(len(param['actJoint_idx'])):
    ax4.plot([lb[i], ub[i]], [sample_range[0],
             sample_range[0]], [i, i])
ax4.set_xlabel('Angle (rad)')
ax4.set_ylabel('Sample')
ax4.set_zlabel('Joint')
ax4.grid()

if dataSet == 'sample':
    plt.figure(5)
    plt.barh(params_name, (res - var_sample), align='center')
    plt.grid()
elif dataSet == 'experimental':
    plt.figure(5)
    plt.barh(params_name[0:6], res[0:6], align='center', color='blue')
    plt.grid()
    plt.figure(6)
    plt.barh(params_name[6:-3*param['NbMarkers']],
             res[6:-3*param['NbMarkers']], align='center', color='orange')
    plt.grid()
    plt.figure(7)
    plt.barh(params_name[-3*param['NbMarkers']:],
             res[-3*param['NbMarkers']:], align='center', color='green')
    plt.grid()
plt.show()

#############################################################

# 7/ Write estimated parameters 


def write_results_tofile(res, file_type='xacro'):
    """ Save offset parameters to file
        file_type = 'csv': for storing only
        file_type = 'xacro' or 'yaml': for updating kinematic model
    """
    torso_list = [0, 1, 2, 3, 4, 5]
    arm1_list = [6, 7, 8, 11]
    arm2_list = [13, 16]
    arm3_list = [19, 22]
    arm4_list = [24, 27]
    arm5_list = [30, 33]
    arm6_list = [36, 39]
    arm7_list = [43, 46]  # include phiz7
    total_list = [torso_list, arm1_list, arm2_list, arm3_list, arm4_list,
                  arm5_list, arm6_list, arm7_list]

    zero_list = []
    for i in range(len(total_list)):
        zero_list = [*zero_list, *total_list[i]]

    param_list = np.zeros((param['NbJoint'], 6))

    # torso all zeros

    # arm 1
    param_list[1, 3] = res[6]
    param_list[1, 4] = res[7]

    # arm 2
    param_list[2, 0] = res[8]
    param_list[2, 2] = res[9]
    param_list[2, 3] = res[10]
    param_list[2, 5] = res[11]

    # arm 3
    param_list[3, 0] = res[12]
    param_list[3, 2] = res[13]
    param_list[3, 3] = res[14]
    param_list[3, 5] = res[15]

    # arm 4
    param_list[4, 1] = res[16]
    param_list[4, 2] = res[17]
    param_list[4, 4] = res[18]
    param_list[4, 5] = res[19]

    # arm 5
    param_list[5, 1] = res[20]
    param_list[5, 2] = res[21]
    param_list[5, 4] = res[22]
    param_list[5, 5] = res[23]

    # arm 6
    param_list[6, 1] = res[24]
    param_list[6, 2] = res[25]
    param_list[6, 4] = res[26]
    param_list[6, 5] = res[27]

    # arm 7
    param_list[7, 0] = res[28]
    param_list[7, 2] = res[29]
    param_list[7, 3] = res[30]
    param_list[7, 5] = res[31]

    joint_names = [name for i, name in enumerate(model.names)]
    offset_name = ['_x_offset', '_y_offset', '_z_offset', '_roll_offset',
                   '_pitch_offset', '_yaw_offset']
    
    if file_type == 'xacro':
        path_save_xacro = join(
            dirname(dirname(str(abspath(__file__)))),
            f"data/tiago/post_estimation/offset.xacro")
        with open(path_save_xacro, "w") as output_file:
            for i in range(param['NbJoint']):
                for j in range(6):
                    update_name = joint_names[i+1] + offset_name[j]
                    update_value = param_list[i, j]
                    update_line = "<xacro:property name=\"{}\" value=\"{}\" / >".format(
                        update_name, update_value)
                    output_file.write(update_line)
                    output_file.write('\n')

    elif file_type == 'yaml':
        path_save_yaml = join(
            dirname(dirname(str(abspath(__file__)))),
            f"data/tiago/post_estimation/offset.yaml")
        with open(path_save_yaml, "w") as output_file:
            for i in range(param['NbJoint']):
                for j in range(6):
                    update_name = joint_names[i+1] + offset_name[j]
                    update_value = param_list[i, j]
                    update_line = "{}: {}".format(
                        update_name, update_value)
                    output_file.write(update_line)
                    output_file.write('\n')
    
    elif file_type == 'csv':
        path_save_ep = join(
            dirname(dirname(str(abspath(__file__)))),
            f"data/tiago/post_estimation/offset.csv")
        with open(path_save_ep, "w") as output_file:
            w = csv.writer(output_file)
            for i in range(nvars):
                w.writerow(
                    [
                        params_name[i],
                        res[i],
                        std_dev[i],
                        std_pctg[i]
                    ]
                )
############################################################