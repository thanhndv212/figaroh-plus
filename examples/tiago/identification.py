from numpy.core.arrayprint import DatetimeFormat
from datetime import datetime
from numpy.core.fromnumeric import shape
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
from pinocchio.utils import *

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

from figaroh.tools.robot import Robot
from figaroh.tools.regressor import *
from figaroh.tools.qrdecomposition import *
from figaroh.tools.randomdata import *
from figaroh.tools.robotcollisions import *


# create a robot object
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago_no_hand.urdf",
    package_dirs = package_dirs,
    # isFext=True  # add free-flyer joint at base
)
active_joints = ["torso_lift_joint",
                 "arm_1_joint",
                 "arm_2_joint",
                 "arm_3_joint",
                 "arm_4_joint",
                 "arm_5_joint",
                 "arm_6_joint",
                 "arm_7_joint"]
idx_act_joints = [robot.model.getJointId(i)-1 for i in active_joints]

# load standard parameters
params_std = robot.get_standard_parameters()  # basic std params
eff_lims = robot.model.effortLimit[idx_act_joints]

# load csv files
path_to_folder = dirname(str(abspath(__file__)))
pos_csv_path = 'data/tiago_position.csv'
vel_csv_path = 'data/tiago_velocity.csv'
eff_csv_path = 'data/tiago_effort.csv'
t = pd.read_csv(join(path_to_folder, pos_csv_path), usecols=[0]).to_numpy()
q = pd.read_csv(join(path_to_folder, pos_csv_path),
                usecols=list(range(1, 9))).to_numpy()
dq = pd.read_csv(join(path_to_folder, vel_csv_path),
                 usecols=list(range(1, 9))).to_numpy()
tau = pd.read_csv(join(path_to_folder, eff_csv_path),
                  usecols=list(range(1, 9))).to_numpy()
print("shape of raw data arrays (t, q, dq, effort): ", t.shape, q.shape, dq.shape, tau.shape)


# truncate the trivial samples at starting and ending segments
n_i = 921
n_f = 6791
range_tc = range(n_i, n_f)
t = t[range_tc, :]
q = q[range_tc, :]
dq = dq[range_tc, :]
tau = tau[range_tc, :]
print("shape of data arrays (t, q, dq, effort) after truncated at both ends: ", t.shape, q.shape, dq.shape, tau.shape)

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
ka = [ka_tor, ka_arm1, ka_arm2, ka_arm3,
      ka_arm4, ka_arm5, ka_arm6, ka_arm7]
red = [red_tor, red_arm1, red_arm2, red_arm3,
       red_arm4, red_arm5, red_arm6, red_arm7]
for i in range(len(red)):
    if i == 0:
        tau[:, i] = red[i]*ka[i]*tau[:, i] + 193
    else:
        tau[:, i] = red[i]*ka[i]*tau[:, i]

# median and lowpass filter
nbutter = 4
f_butter = 2
f_sample = 100
b1, b2 = signal.butter(nbutter, f_butter/(f_sample/2), 'low')
q_med = np.zeros(q.shape)
q_butter = np.zeros(q.shape)
dq_med = np.zeros(dq.shape)
dq_butter = np.zeros(dq.shape)

for j in range(dq.shape[1]):
    q_med[:, j] = signal.medfilt(q[:, j], 5)
    q_butter[:, j] = signal.filtfilt(
        b1, b2, q_med[:, j], axis=0, padtype="odd", padlen=3 * (max(len(b1), len(b2)) - 1)
    )
    dq_med[:, j] = signal.medfilt(dq[:, j], 5)
    dq_butter[:, j] = signal.filtfilt(
        b1, b2, dq_med[:, j], axis=0, padtype="odd", padlen=3 * (max(len(b1), len(b2)) - 1)
    )

# estimate acceleration
ddq = np.zeros(dq.shape)
ddq_raw = np.zeros(dq.shape)

for j in range(ddq.shape[1]):
    ddq[:, j] = np.gradient(dq_butter[:, j])/np.gradient(t[:, 0])
    ddq_raw[:, j] = np.gradient(dq[:, j])/np.gradient(t[:, 0])

# get full configuration of p, v, a of all joints
p = np.array([robot.q0]*Ntotal)
v = np.array([robot.v0]*Ntotal)
a = np.array([robot.v0]*Ntotal)

p[:, idx_act_joints] = q_butter
v[:, idx_act_joints] = dq_butter
a[:, idx_act_joints] = ddq

#  eliminate border effect

# # plot filtered and raw p,v,a
# plot1 = plt.figure(1)
# axs1 = plot1.subplots(8, 1)
# for i in range(len(idx_act_joints)):
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
# for i in range(len(idx_act_joints)):
#     if i == 0:
#         axs3[i].plot(t, dq[:, i], label='vel raw')
#         axs3[i].plot(t, dq_butter[:, i], label='vel filtered')
#         axs3[i].set_ylabel("arm_%d_joint (m/s)" % i)

#     else:
#         axs3[i].plot(t, dq[:, i])
#         axs3[i].plot(t, dq_butter[:, i])

# plot3.legend()

plot4 = plt.figure(4)
axs4 = plot4.subplots(8, 1)
for i in range(len(idx_act_joints)):
    if i == 0:
        axs4[i].plot(t, ddq_raw[:, i], label='acc raw')
        axs4[i].set_ylabel("torso_lift_joint (m/s^2)")
        axs4[i].plot(t, ddq[:, i], label='acc filtered')
    else:
        axs4[i].plot(t, ddq_raw[:, i])
        axs4[i].plot(t, ddq[:, i])
        axs4[i].set_ylabel("arm_%d_joint (rad/s^2)" % i)


# plot4.legend()
# plt.show()

# build basic regressor
W = build_regressor_basic(Ntotal, robot, p, v, a)

# add friction manually ot dictionary of standard params


def add_frict_params(params_std, fv_std, fs_std, idx_act_joints):
    # check lenght of friction constants
    assert len(fv_std) == len(
        idx_act_joints), "fv should have the size of active joints"
    assert len(fs_std) == len(
        idx_act_joints), "fv should have the size of active joints"

    list_keys = list(params_std.keys())
    list_values = list(params_std.values())
    for i in range(len(idx_act_joints)):
        joint_num = idx_act_joints[i] + 1
        # add friction params after mass params
        mass_index = list_keys.index('m%d' % joint_num)
        list_keys.insert(mass_index + 1, 'fv%d' % joint_num)
        list_keys.insert(mass_index + 2, 'fs%d' % joint_num)
        list_values.insert(mass_index + 1, fv_std[i])
        list_values.insert(mass_index + 2, fs_std[i])

    new_params_std = dict(zip(list_keys, list_values))
    return new_params_std


fv_std = np.zeros(len(idx_act_joints))
fs_std = np.zeros(len(idx_act_joints))
# basic std params + friction params
params_std_f = add_frict_params(params_std, fv_std, fs_std, idx_act_joints)


def add_frict_cols(W_act, params_std_f, idx_act_joints, v):
    # check correct size of W
    assert W_act.shape[1] + 2 * \
        len(idx_act_joints) == len(
            params_std_f), "check size of W or params_std"
    list_keys = list(params_std_f.keys())
    new_W = W_act
    for i in range(len(idx_act_joints)):
        joint_num = idx_act_joints[i] + 1
        # add friction columns after mass columns
        mass_index = list_keys.index('m%d' % joint_num)
        new_W = np.insert(new_W, mass_index+1,
                          np.zeros(new_W.shape[0]), axis=1)
        new_W = np.insert(new_W, mass_index+2,
                          np.zeros(new_W.shape[0]), axis=1)
        vi = v[:, i]
        sign_vi = np.array([np.sign(k) for k in v[:, i]])
        range_i = range(idx_act_joints[i]*v.shape[0],
                        (idx_act_joints[i]+1)*v.shape[0])
        new_W[range_i, mass_index + 1] = vi
        new_W[range_i, mass_index + 2] = sign_vi
    return new_W


# extract rows of active joints/remove rows of inactive joints
list_idx = []
for k in range(len(idx_act_joints)):
    list_idx.extend(
        list(range(idx_act_joints[k]*Ntotal, (idx_act_joints[k]+1)*Ntotal)))
W_act = W[list_idx, :]

# add friction columns to regressor = basic observation matrix + friction columns
W_f = add_frict_cols(W_act, params_std_f, idx_act_joints, v)


# add offset torque
def add_offset_params(params_std_f, idx_act_joints):
    # check lenght of friction constants
    list_keys = list(params_std_f.keys())
    list_values = list(params_std_f.values())
    for i in range(len(idx_act_joints)):
        joint_num = idx_act_joints[i] + 1
        # add offset params after friction params
        fs_index = list_keys.index('fs%d' % joint_num)
        list_keys.insert(fs_index + 1, 'off%d' % joint_num)
        list_values.insert(fs_index + 1, 0)
    new_params_std = dict(zip(list_keys, list_values))
    return new_params_std

# basic std params + friction params + offset


params_std_os = add_offset_params(params_std_f, idx_act_joints)


def add_offset_cols(Ntotal, W_f, params_std_os, idx_act_joint):
    # check correct size of W
    assert W_f.shape[1] + \
        len(idx_act_joints) == len(
            params_std_os), "check size of W or params_std"
    list_keys = list(params_std_os.keys())
    new_W = W_f
    for i in range(len(idx_act_joints)):
        joint_num = idx_act_joints[i] + 1
        # add offset columns after friction columns
        fs_index = list_keys.index('fs%d' % joint_num)
        new_W = np.insert(new_W, fs_index+1,
                          np.zeros(new_W.shape[0]), axis=1)
        offset_i = np.ones(Ntotal)
        range_i = range(idx_act_joints[i]*Ntotal,
                        (idx_act_joints[i]+1)*Ntotal)
        new_W[range_i, fs_index + 1] = offset_i
    return new_W


# basic observation matrix + friction columns + offset columns
W_os = add_offset_cols(Ntotal, W_f, params_std_os, idx_act_joints)


# remove zero columns
idx_e, params_r = get_index_eliminate(W_os, params_std_os, tol_e=0.001)
W_e = build_regressor_reduced(W_os, idx_e)

# eliminate zero crossing if considering friction model

#######separate link-by-link and parallel decimate########
# joint torque
tau_dec = []
for i in range(len(idx_act_joints)):
    tau_dec.append(signal.decimate(tau[:, i], q=10, zero_phase=True))

tau_rf = tau_dec[0]
for i in range(1, len(tau_dec)):
    tau_rf = np.append(tau_rf, tau_dec[i])

# regressor
W_list = []  # list of sub regressor for each joitnt
for i in range(len(idx_act_joints)):
    W_dec = []
    for j in range(W_e.shape[1]):
        W_dec.append(signal.decimate(W_e[range(
            idx_act_joints[i]*Ntotal, (idx_act_joints[i]+1)*Ntotal), j], q=10, zero_phase=True))

    W_temp = np.zeros((W_dec[0].shape[0], len(W_dec)))
    for i in range(len(W_dec)):
        W_temp[:, i] = W_dec[i]
    W_list.append(W_temp)

    # rejoining sub  regresosrs into one complete regressor
W_rf = np.zeros((tau_rf.shape[0], W_list[0].shape[1]))
for i in range(len(W_list)):
    W_rf[range(i*W_list[i].shape[0], (i+1)*W_list[i].shape[0]), :] = W_list[i]

# time
t_dec = signal.decimate(t[:, 0], q=10, zero_phase=True)


# calculate base parameters
W_b, bp_dict, params_b, phi_b, phi_std = double_QR(
    tau_rf, W_rf, params_r, params_std_os)
# import pprint
# pprint.pprint(bp_dict)
print("condition number: ", np.linalg.cond(W_b))
print("residue norm: ", np.linalg.norm(tau_rf - np.dot(W_b, phi_b)))
print("relative residue norm: ", np.linalg.norm(
    tau_rf - np.dot(W_b, phi_b))/np.linalg.norm(tau_rf))

# joint torque estimated from p,v,a with base params
tau_base = np.dot(W_b, phi_b)
print("tau_base shape", tau_base.shape)

# joint torque estimated from p,v,a with std params
phi_f = np.array(list(params_std_f.values()))
tau_f = np.dot(W_f, phi_f)
tau_ref_f = tau_f[range(len(idx_act_joints)*Ntotal)]

phi_os = np.array(list(params_std_os.values()))
tau_os = np.dot(W_os, phi_os)
tau_ref_os = tau_os[range(len(idx_act_joints)*Ntotal)]

# plot joint torque
plot2 = plt.figure(2)
axs2 = plot2.subplots(8, 1)
for i in range(len(idx_act_joints)):
    if i == 0:
        axs2[i].plot(t_dec, tau_dec[i], label='effort measures-decimated')
        axs2[i].plot(t_dec, tau_base[range(i*tau_dec[i].shape[0], (i+1)*tau_dec[i].shape[0])],
                     label='base params effort estimated')
        axs2[i].plot(t, tau_ref_os[range(i*Ntotal, (i+1)*Ntotal)],
                     label='standard params effort estimated')
        axs2[i].set_ylabel("torso_lift_joint (N)")
        # axs2[i].axhline(eff_lims[i], t[0], t[-1])
    else:
        axs2[i].plot(t_dec, tau_dec[i])
        axs2[i].plot(t_dec, tau_base[range(
            i*tau_dec[i].shape[0], (i+1)*tau_dec[i].shape[0])])
        axs2[i].plot(t, tau_ref_os[range(i*Ntotal, (i+1)*Ntotal)])
        axs2[i].set_ylabel("arm_%d_joint (N.m)" % i)
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
