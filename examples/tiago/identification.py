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

from numpy.core.arrayprint import DatetimeFormat
from datetime import datetime
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer

from sys import argv
import os
from os.path import dirname, join, abspath
import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.linalg import norm, solve
from scipy import linalg, signal

import pandas as pd
import json
import csv
import yaml
from yaml.loader import SafeLoader
import pprint

from figaroh.tools.robot import Robot
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
    get_index_eliminate,
    eliminate_non_dynaffect,
    add_actuator_inertia,
    add_friction,
    add_joint_offset,
)
from figaroh.tools.qrdecomposition import get_baseParams, double_QR


# create a robot object
ros_package_path = os.getenv("ROS_PACKAGE_PATH")
package_dirs = ros_package_path.split(":")
robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago_no_hand.urdf",
    package_dirs=package_dirs,
    # isFext=True  # add free-flyer joint at base
)
active_joints = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
]
idx_act_joints = [robot.model.getJointId(i) - 1 for i in active_joints]

# load standard parameters
with open("examples/tiago/config/tiago_config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)
identif_data = config["identification"]
pprint.pprint(dir(robot.model))
for i, joint in enumerate(robot.model.joints):
    print(i, robot.model.names[i], joint)
params_settings = get_param_from_yaml(robot, identif_data)
params_std = robot.get_standard_parameters(params_settings)

params_settings["idx_act_joints"] = idx_act_joints

eff_lims = robot.model.effortLimit[params_settings["idx_act_joints"]]

# load csv files
path_to_folder = dirname(str(abspath(__file__)))
pos_csv_path = "data/tiago_position.csv"
vel_csv_path = "data/tiago_velocity.csv"
eff_csv_path = "data/tiago_effort.csv"
t = pd.read_csv(join(path_to_folder, pos_csv_path), usecols=[0]).to_numpy()
q = pd.read_csv(
    join(path_to_folder, pos_csv_path), usecols=list(range(1, 9))
).to_numpy()
dq = pd.read_csv(
    join(path_to_folder, vel_csv_path), usecols=list(range(1, 9))
).to_numpy()
tau = pd.read_csv(
    join(path_to_folder, eff_csv_path), usecols=list(range(1, 9))
).to_numpy()
print(
    "shape of raw data arrays (t, q, dq, effort): ",
    t.shape,
    q.shape,
    dq.shape,
    tau.shape,
)


# truncate the trivial samples at starting and ending segments
n_i = 921
n_f = 6791
range_tc = range(n_i, n_f)
t = t[range_tc, :]
q = q[range_tc, :]
dq = dq[range_tc, :]
tau = tau[range_tc, :]
print(
    "shape of data arrays (t, q, dq, effort) after truncated at both ends: ",
    t.shape,
    q.shape,
    dq.shape,
    tau.shape,
)

# number of sample points9
Ntotal = t.shape[0]

# reduction ratio
red_tor = 1
red_arm1 = 100
red_arm2 = 100
red_arm3 = 100
red_arm4 = 100
red_arm5 = 336
red_arm6 = 336
red_arm7 = 336

# current constant (Nm/A)
ka_tor = 1
ka_arm1 = 0.136
ka_arm2 = 0.136
ka_arm3 = -0.087
ka_arm4 = -0.087
ka_arm5 = -0.087
ka_arm6 = -0.0613
ka_arm7 = -0.0613
ka = [ka_tor, ka_arm1, ka_arm2, ka_arm3, ka_arm4, ka_arm5, ka_arm6, ka_arm7]
red = [red_tor, red_arm1, red_arm2, red_arm3, red_arm4, red_arm5, red_arm6, red_arm7]
for i in range(len(red)):
    if i == 0:
        tau[:, i] = red[i] * ka[i] * tau[:, i] + 193
    else:
        tau[:, i] = red[i] * ka[i] * tau[:, i]

# median and lowpass filter
nbutter = 4
f_butter = 2
f_sample = 100
b1, b2 = signal.butter(nbutter, f_butter / (f_sample / 2), "low")
q_med = np.zeros(q.shape)
q_butter = np.zeros(q.shape)
dq_med = np.zeros(dq.shape)
dq_butter = np.zeros(dq.shape)

for j in range(dq.shape[1]):
    q_med[:, j] = signal.medfilt(q[:, j], 5)
    q_butter[:, j] = signal.filtfilt(
        b1,
        b2,
        q_med[:, j],
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(b1), len(b2)) - 1),
    )
    dq_med[:, j] = signal.medfilt(dq[:, j], 5)
    dq_butter[:, j] = signal.filtfilt(
        b1,
        b2,
        dq_med[:, j],
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(b1), len(b2)) - 1),
    )

# estimate acceleration
ddq = np.zeros(dq.shape)
ddq_raw = np.zeros(dq.shape)

for j in range(ddq.shape[1]):
    ddq[:, j] = np.gradient(dq_butter[:, j]) / np.gradient(t[:, 0])
    ddq_raw[:, j] = np.gradient(dq[:, j]) / np.gradient(t[:, 0])

# get full configuration of p, v, a of all joints
p = np.array([robot.q0] * Ntotal)
v = np.array([robot.v0] * Ntotal)
a = np.array([robot.v0] * Ntotal)

p[:, params_settings["idx_act_joints"]] = q_butter
v[:, params_settings["idx_act_joints"]] = dq_butter
a[:, params_settings["idx_act_joints"]] = ddq

#  eliminate border effect

# # plot filtered and raw p,v,a
# plot1 = plt.figure(1)
# axs1 = plot1.subplots(8, 1)
# for i in range(len(params_settings["idx_act_joints"])):
#     if i == 0:
#         axs1[i].plot(t, q[:, i], label='pos raw')
#         axs1[i].plot(t, q_butter[:, i], label='pos filtered')
#         axs1[i].set_ylabel("torso_lift_joint (m)")
#     else:
#         axs1[i].plot(t, q[:, i])
#         axs1[i].plot(t, q_butter[:, i])
#         axs1[i].set_ylabel("arm_%d_joint (rad)" % i)
# plot1.legend()
# plot3 = plt.figure(3)
# axs3 = plot3.subplots(8, 1)
# for i in range(len(params_settings["idx_act_joints"])):
#     if i == 0:
#         axs3[i].plot(t, dq[:, i], label='vel raw')
#         axs3[i].plot(t, dq_butter[:, i], label='vel filtered')
#         axs3[i].set_ylabel("arm_%d_joint (m/s)" % i)

#     else:
#         axs3[i].plot(t, dq[:, i])
#         axs3[i].plot(t, dq_butter[:, i])

# plot3.legend()

# plot4 = plt.figure(4)
# axs4 = plot4.subplots(8, 1)
# for i in range(len(params_settings["idx_act_joints"])):
#     if i == 0:
#         axs4[i].plot(t, ddq_raw[:, i], label='acc raw')
#         axs4[i].set_ylabel("torso_lift_joint (m/s^2)")
#         axs4[i].plot(t, ddq[:, i], label='acc filtered')
#     else:
#         axs4[i].plot(t, ddq_raw[:, i])
#         axs4[i].plot(t, ddq[:, i])
#         axs4[i].set_ylabel("arm_%d_joint (rad/s^2)" % i)


# plot4.legend()
# plt.show()

# build basic regressor
W = build_regressor_basic(robot, p, v, a, params_settings)
print(W.shape, len(params_std))

# remove zero columns
idx_e, params_r = get_index_eliminate(W, params_std, tol_e=0.001)
W_e = build_regressor_reduced(W, idx_e)
print(W_e.shape)
# eliminate zero crossing if considering friction model

#######separate link-by-link and parallel decimate########
# joint torque
tau_dec = []
for i in range(len(params_settings["idx_act_joints"])):
    tau_dec.append(signal.decimate(tau[:, i], q=10, zero_phase=True))

tau_rf = tau_dec[0]
for i in range(1, len(tau_dec)):
    tau_rf = np.append(tau_rf, tau_dec[i])

# regressor
W_list = []  # list of sub regressor for each joitnt
for i in range(len(params_settings["idx_act_joints"])):
    W_dec = []
    for j in range(W_e.shape[1]):
        W_dec.append(
            signal.decimate(
                W_e[
                    range(
                        params_settings["idx_act_joints"][i] * Ntotal,
                        (params_settings["idx_act_joints"][i] + 1) * Ntotal,
                    ),
                    j,
                ],
                q=10,
                zero_phase=True,
            )
        )

    W_temp = np.zeros((W_dec[0].shape[0], len(W_dec)))
    for i in range(len(W_dec)):
        W_temp[:, i] = W_dec[i]
    W_list.append(W_temp)

# rejoining sub  regresosrs into one complete regressor
W_rf = np.zeros((tau_rf.shape[0], W_list[0].shape[1]))
for i in range(len(W_list)):
    W_rf[range(i * W_list[i].shape[0], (i + 1) * W_list[i].shape[0]), :] = W_list[i]

# time
t_dec = signal.decimate(t[:, 0], q=10, zero_phase=True)


# calculate base parameters
W_b, bp_dict, params_b, phi_b, phi_std = double_QR(tau_rf, W_rf, params_r, params_std)
# import pprint
# pprint.pprint(bp_dict)
print("condition number: ", np.linalg.cond(W_b))
print(
    "rmse norm (N/m): ",
    np.linalg.norm(tau_rf - np.dot(W_b, phi_b)) / np.sqrt(tau_rf.shape[0]),
)
print(
    "relative residue norm: ",
    np.linalg.norm(tau_rf - np.dot(W_b, phi_b)) / np.linalg.norm(tau_rf),
)

# joint torque estimated from p,v,a with base params
tau_base = np.dot(W_b, phi_b)
print("tau_base shape", tau_base.shape)

# # joint torque estimated from p,v,a with std params
phi_ref = np.array(list(params_std.values()))
tau_ref = np.dot(W, phi_ref)
tau_ref = tau_ref[range(len(params_settings["idx_act_joints"]) * Ntotal)]
plt.rcParams.update({"font.size": 30})
# plot joint torque
plot2 = plt.figure(2)
axs2 = plot2.subplots(8, 1)
for i in range(len(params_settings["idx_act_joints"])):
    if i == 0:
        axs2[i].plot(t_dec, tau_dec[i], color="red", label="effort measured")
        axs2[i].plot(
            t_dec,
            tau_base[range(i * tau_dec[i].shape[0], (i + 1) * tau_dec[i].shape[0])],
            color="green",
            label="effort estimated",
        )
        axs2[i].plot(
            t,
            tau_ref[range(i * Ntotal, (i + 1) * Ntotal)],
            color="blue",
            label="notional effort estimated",
        )
        axs2[i].set_ylabel("torso", fontsize=25)
        axs2[i].tick_params(labelbottom=False, bottom=False)
        # axs2[i].axhline(eff_lims[i], t[0], t[-1])
        axs2[i].grid()
    elif i < 8:
        axs2[i].plot(t_dec, tau_dec[i], color="red")
        axs2[i].plot(
            t_dec,
            tau_base[range(i * tau_dec[i].shape[0], (i + 1) * tau_dec[i].shape[0])],
            color="green",
        )
        axs2[i].plot(t, tau_ref[range(i * Ntotal, (i + 1) * Ntotal)], color="blue")
        axs2[i].set_ylabel(
            "arm %d" % i,
            fontsize=25,
        )
        axs2[i].tick_params(labelbottom=False, bottom=False)
        axs2[i].grid()

        if i == 7:
            axs2[i].set_xlabel("time (sec)", fontsize=25)
            axs2[i].tick_params(axis="y", color="black")
            axs2[i].tick_params(labelbottom=True, bottom=True)

        # axs2[i].axhline(eff_lims[i], t[0], t[-1])
# axs2[8].plot(t, ddq[:, 0])

plot2.legend()
plt.show()


# x = np.arange(len(params_b))
# width = 0.5
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, phi_b, width, label='identified')
# rects2 = ax.bar(x + width/2, phi_std, width, label='reference')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.legend()


# plt.grid()
# plt.show(block=True)


# std_xr_ols = relative_stdev(W_b, phi_b, tau_rf)
# path_to_folder = dirname(dirname(str(abspath(__file__))))
# dt = datetime.now()
# current_time = dt.strftime("%d_%b_%Y_%H%M")
# bp_csv = join(path_to_folder,
#               'identification_toolbox/src/tiago/tiago_bp_{current_time}.csv')
# # pd.DataFrame(bp_dict).to_csv(bp_csv, index=True)
# with open(bp_csv, "w") as output_file:
#     w = csv.writer(output_file)
#     for i in range(len(params_b)):
#         w.writerow(
#             [params_b[i], phi_b[i], phi_std[i], 100*std_xr_ols[i]]
#         )
