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

import os
from os.path import dirname, join, abspath

import pinocchio as pin
from pinocchio.utils import *

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import least_squares
import numpy as np

import yaml
from yaml.loader import SafeLoader
import pprint

from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer
from figaroh.tools.robot import Robot
from figaroh.calibration.calibration_tools import (
    get_param,
    get_param_from_yaml,
    add_pee_name,
    load_data,
    init_var,
    get_PEE_fullvar,
    calculate_base_kinematics_regressor,
    update_forward_kinematics,
    get_LMvariables,
    get_rel_transform)

# 1. Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago.urdf",
    package_dirs=package_dirs,
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

# Load robot configuration from YAML file
with open('config/tiago_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config['calibration'])
calib_data = config['calibration']
param = get_param_from_yaml(robot, calib_data)

for jointID in param['actJoint_idx']:
    print(model.joints[jointID].shortname())

#############################################################

# 2. Base parameters calculation
q_rand = []
Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e = calculate_base_kinematics_regressor(
    q_rand, model, data, param)

# Add markers name to param['param_name']
add_pee_name(param)

# Display total calibrating parameter names
for i, pn in enumerate(param['param_name']):
    print(i, pn)

#############################################################

# 3. Data collection/generation
dataSet = 'experimental'  # Choose data source 'sample' or 'experimental'
if dataSet == 'sample':
    # Create artificial offsets
    var_sample, nvars_sample = get_LMvariables(param, mode=1, seed=0.05)
    print("%d var_sample: " % nvars_sample, var_sample)

    # Create sample configurations
    q_sample = np.empty((param['NbSample'], model.nq))

    for i in range(param['NbSample']):
        config = param['q0']
        config[param['config_idx']] = pin.randomConfiguration(model)[
            param['config_idx']]
        q_sample[i, :] = config

    # Create simulated end effector coordinates measures (PEEm)
    PEEm_sample = update_forward_kinematics(model, data, var_sample, q_sample, param)

    q_LM = np.copy(q_sample)
    PEEm_LM = np.copy(PEEm_sample)

elif dataSet == 'experimental':
    # Read CSV file
    path = abspath('data/eye_hand_calibration_recorded_data_500.csv')

    # Load data from file
    PEEm_exp, q_exp = load_data(path, model, param)

    q_LM = np.copy(q_exp)
    PEEm_LM = np.copy(PEEm_exp)

# Remove potential outliers which identified from previous calibration
print(np.shape(PEEm_LM))

# Update the number of samples
print('updated number of samples: ', param['NbSample'])

# # Play configurations on visualizer
# from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer
# import time
# viz = MeshcatVisualizer(
#     model=robot.model, collision_model=robot.collision_model,
#     visual_model=robot.visual_model, url='classical'
# )
# time.sleep(1)
# for i in range(param['NbSample']):
#     viz.display(q_LM[i, :])
#     time.sleep(1)
#############################################################

# 4. Given a model and configurations (input), end effector positions/locations
# (output), solve an optimization problem to find offset params as variables

# NON-LINEAR model with Levenberg-Marquardt #################

"""
Minimize the difference between measured coordinates of end-effector
and its estimated values from DGM by Levenberg-Marquardt
"""

coeff = 1e-2

def cost_func(var, coeff, q, model, data, param, PEEm):
    """
    Cost function for the optimization problem.

    Args:
        var (array): Variable parameters.
        coeff (float): Scaling factor for the variable parameters.
        q (array): Robot joint configurations.
        model (pin.Model): Robot model.
        data (pin.Data): Data structure for the robot model.
        param (dict): Dictionary of parameters.
        PEEm (array): Measured end-effector coordinates.

    Returns:
        res_vect (array): Residual vector for the optimization problem.
    """
    
    PEEe = update_forward_kinematics(model, data, var, q, param)
    res_vect = np.append((PEEm - PEEe), np.sqrt(coeff)
                         * var[6:-param['NbMarkers']*param['calibration_index']])

    return res_vect

# Initial guess
# mode = 1: random seed [-0.01, 0.01], mode = 0: init guess = 0
# var_0, nvars = init_var(param, mode=0)
var_0, nvars = get_LMvariables(param, mode=0)

# Write base position in initial guess
# hey5 
# pose:
#     position: [ 0.2163, 0.03484, 0.004, ]
#     rpy: [0., -1.57, -1.57]
# frame: hand_tool_link
#   <xacro:property name="camera_position_x" value="0.0908" />
#   <xacro:property name="camera_position_y" value="0.08" />
#   <xacro:property name="camera_position_z" value="0.0" />
#   <xacro:property name="camera_orientation_r" value="${-90 * deg_to_rad}" />
#   <xacro:property name="camera_orientation_p" value="0.0" />
#   <xacro:property name="camera_orientation_y" value="0.0" />
# camera varible 
var_0[0:6] = [0.0908, 0.08, 0.0, -1.57, 0.0, 0.0]
# tip variable
var_0[-param['calibration_index']:] = np.array([ 0.2163, 0.03484, 0.004]) 
iterate = True
count = 0
del_list = []

while count <2:
    count += 1
    print(f"{count} iter guess", var_0)
    # Solve the optimization problem
    LM_solve = least_squares(cost_func, var_0, method='lm', verbose=1,
                            args=(coeff, q_LM, model, data, param, PEEm_LM))

    #############################################################

    # 5. Result analysis
    res = LM_solve.x

    # PEE estimated by solution
    PEEe_sol = update_forward_kinematics(model, data, res, q_LM, param)

    # Root mean square error
    rmse = np.sqrt(np.mean((PEEe_sol-PEEm_LM)**2))

    print("solution: ")
    for x_i, xname in enumerate(param['param_name']):
        print(x_i+1, xname, list(res)[x_i])
    print("minimized cost function: ", rmse)
    print("optimality: ", LM_solve.optimality)

    # calculate difference between estimated data and measured data
    delta_PEE = PEEe_sol - PEEm_LM
    PEE_xyz = delta_PEE.reshape((param['NbMarkers']*param['calibration_index'], param["NbSample"]))
    PEE_dist = np.zeros((param['NbMarkers'], param["NbSample"]))
    for i in range(param["NbMarkers"]):
        for j in range(param["NbSample"]):
            PEE_dist[i, j] = np.sqrt(
                PEE_xyz[i*3, j]**2 + PEE_xyz[i*3 + 1, j]**2 + PEE_xyz[i*3 + 2, j]**2)

    # detect "bad" data (outlierrs) => remove outliers, recalibrate 
    scatter_size = np.zeros_like(PEE_dist)
    eps = 0.02
    for i in range(param['NbMarkers']):
        for k in range(param['NbSample']):
            if PEE_dist[i, k] > eps:
                del_list.append((i, k))
        scatter_size[i, :] = 20*PEE_dist[i, :]/np.min(PEE_dist[i, :])
    print(f"indices of samples with >{eps} cm deviation: ", del_list)
    if del_list is not None and count <2:
        path = abspath('data/eye_hand_calibration_recorded_data_500_wtorso.csv')

        # Load data from file
        PEEm_LM, q_LM = load_data(path, model, param, del_list)
        param['NbSample'] = q_LM.shape[0]
    else:
        iterate = False
# # uncalibrated
# var_0[0:6] = [0.0908, 0.08, 0.0, -1.57, 0.0, 0.0]
# tip variable
if param['calibration_index'] == 6:
    var_0[-param['calibration_index']:] = np.array([ 0.2163, 0.03484, 0.004, 0., -1.57, -1.57]) 
elif param['calibration_index'] == 3:
    var_0[-param['calibration_index']:] = np.array([ 0.2163, 0.03484, 0.004]) 

uncalib_res = var_0
# # uncalib_res[:3] = res[:3]
# # uncalib_res[-3:] = res[-3:]
PEEe_uncalib = update_forward_kinematics(model, data, uncalib_res, q_LM, param)
rmse_uncalib = np.sqrt(np.mean((PEEe_uncalib-PEEm_LM)**2))
print("minimized cost function uncalib: ", rmse_uncalib)
calib_result = dict(zip(param['param_name'], list(res)))
# # calculate standard deviation of estimated parameter ( Khalil chapter 11)
# sigma_ro_sq = (LM_solve.cost**2) / \
#     (param['NbSample']*param['calibration_index'] - nvars)
# J = LM_solve.jac
# C_param = sigma_ro_sq*np.linalg.pinv(np.dot(J.T, J))
# std_dev = []
# std_pctg = []
# for i in range(nvars):
#     std_dev.append(np.sqrt(C_param[i, i]))
#     std_pctg.append(abs(np.sqrt(C_param[i, i])/res[i]))
# print("standard deviation: ", std_dev)


##############################################################

# 6. Plot results

# # 1// Errors between estimated position and measured position of markers

fig1, ax1 = plt.subplots(param['NbMarkers'], 1)
# fig1.suptitle(
    # "Relative positional errors between estimated markers and measured markers (m) by samples ")
colors = ['blue',
          'red',
          'yellow',
          'purple'
          ]

if param['NbMarkers'] == 1:
    print(param['NbSample'], PEE_dist.shape)
    ax1.bar(np.arange(param['NbSample']), PEE_dist[i, :])
    ax1.set_xlabel('Sample',fontsize=25)
    ax1.set_ylabel('Error (meter)',fontsize=30)
    ax1.tick_params(axis='both', labelsize=30)
    ax1.grid()
else:
    for i in range(param['NbMarkers']):
        ax1[i].bar(np.arange(param['NbSample']),
                   PEE_dist[i, :], color=colors[i])
        ax1[i].set_xlabel('Sample',fontsize=25)
        ax1[i].set_ylabel('Error of marker %s (meter)' % (i+1),fontsize=25)
        ax1[i].tick_params(axis='both', labelsize=30)
        ax1[i].grid()

# # 2// plot 3D measured poses and estimated
fig2 = plt.figure(2)
fig2.suptitle("Visualization of estimated poses and measured pose in Cartesian")
ax2 = fig2.add_subplot(111, projection='3d')
PEEm_LM2d = PEEm_LM.reshape((param['NbMarkers']*param['calibration_index'], param["NbSample"]))
PEEe_sol2d = PEEe_sol.reshape((param['NbMarkers']*param['calibration_index'], param["NbSample"]))
for i in range(param['NbMarkers']):
    ax2.scatter3D(PEEm_LM2d[i*3, :], PEEm_LM2d[i*3+1, :],
                  PEEm_LM2d[i*3+2, :], marker='^', color='red', label='measured')
    ax2.scatter3D(PEEe_sol2d[i*3, :], PEEe_sol2d[i*3+1, :],
                  PEEe_sol2d[i*3+2, :], marker='o', color='green', label='estimated')
ax2.set_xlabel('X - front (meter)')
ax2.set_ylabel('Y - side (meter)')
ax2.set_zlabel('Z - height (meter)')
ax2.grid()
ax2.legend()

# 3// visualize relative deviation between measure and estimate
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
for i in range(param['NbMarkers']):
    ax3.scatter3D(PEEm_LM2d[i*3, :], PEEm_LM2d[i*3+1, :],
                  PEEm_LM2d[i*3+2, :], s=scatter_size[i, :], color='blue')
ax3.set_xlabel('X - front (meter)')
ax3.set_ylabel('Y - side (meter)')
ax3.set_zlabel('Z - height (meter)')
ax3.grid()

# 4// joint configurations within range bound
fig4 = plt.figure()
fig4.suptitle("Joint configurations with joint bounds")
ax4 = fig4.add_subplot(111, projection='3d')
lb = ub = []
for j in param['config_idx']:
    # model.names does not accept index type of numpy int64
    # and model.lowerPositionLimit index lag to model.names by 1
    lb = np.append(lb, model.lowerPositionLimit[j])
    ub = np.append(ub, model.upperPositionLimit[j])
q_actJoint = q_LM[:, param['config_idx']]
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


# # v. Optional plots depending on the dataset
# if dataSet == 'sample':
#     plt.figure(5)
#     plt.barh(params_name, (res - var_sample), align='center')
#     plt.grid()
# elif dataSet == 'experimental':
#     plt.figure(5)
#     plt.barh(params_name[0:6], res[0:6], align='center', color='blue')
#     plt.grid()
#     plt.figure(6)
#     plt.barh(params_name[6:-3*param['NbMarkers']],
#              res[6:-3*param['NbMarkers']], align='center', color='orange')
#     plt.grid()
#     plt.figure(7)
#     plt.barh(params_name[-3*param['NbMarkers']:],
#              res[-3*param['NbMarkers']:], align='center', color='green')
#     plt.grid()
plt.show()
##############################################################

# # 6. Save to file
# mapping 
calibration_parameters = {}
calibration_parameters['camera_position_x'] = calib_result['base_px']
calibration_parameters['camera_position_y'] = calib_result['base_py']
calibration_parameters['camera_position_z'] = calib_result['base_pz']
calibration_parameters['camera_orientation_r'] = calib_result['base_phix']
calibration_parameters['camera_orientation_p'] = calib_result['base_phiy']
calibration_parameters['camera_orientation_y'] = calib_result['base_phiz']

for idx in param['actJoint_idx']:
    joint = model.names[idx]
    for key in calib_result.keys():
        if joint in key:    
            calibration_parameters[joint+'_joint_offset'] = calib_result[key]

path_save_xacro = abspath('data/offset.xacro')
# with open(path_save_xacro, "w") as output_file:
#     for parameter in calibration_parameters.keys():
#             update_name = parameter
#             update_value = calibration_parameters[parameter]
#             update_line = "<xacro:property name=\"{}\" value=\"{}\" / >".format(
#                 update_name, update_value)
#             output_file.write(update_line)
#             output_file.write('\n')
