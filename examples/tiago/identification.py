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
from os.path import abspath
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from scipy import signal
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
    get_index_eliminate,
)
from figaroh.tools.qrdecomposition import double_QR
from tiago_utils.tiago_tools import load_robot


robot = load_robot(
    abspath("urdf/tiago.urdf"),
    load_by_urdf=True,
)
active_joints = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint"
]
idx_act_joints = [robot.model.getJointId(i) - 1 for i in active_joints]

# load standard parameters
with open("config/tiago_config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)
identif_data = config["identification"]
params_settings = get_param_from_yaml(robot, identif_data)
standard_parameter = robot.get_standard_parameters(params_settings)
params_settings["idx_act_joints"] = idx_act_joints
eff_lims = robot.model.effortLimit[params_settings["idx_act_joints"]]

# load csv files
pos_csv_path = "data/identification/dynamic/tiago_position.csv"
vel_csv_path = "data/identification/dynamic/tiago_velocity.csv"
eff_csv_path = "data/identification/dynamic/tiago_effort.csv"

t = pd.read_csv(abspath(pos_csv_path), usecols=[0]).to_numpy()
pos = pd.read_csv(abspath(pos_csv_path))
vel = pd.read_csv(abspath(vel_csv_path))
eff = pd.read_csv(abspath(eff_csv_path))

pos_cols = []
vel_cols = []
eff_cols = []
for jn in active_joints:
    for ii in pos.columns:
        if jn in ii:
            pos_cols.append(ii)
    for jj in vel.columns:
        if jn in jj:
            vel_cols.append(jj)
    for kk in eff.columns:
        if jn in kk:
            eff_cols.append(kk)

q = pos[pos_cols].to_numpy()
dq = vel[vel_cols].to_numpy()
tau = eff[eff_cols].to_numpy()
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
reduction_ratio = dict()
reduction_ratio["torso_lift_joint"] = 1
reduction_ratio["arm_1_joint"] = 100
reduction_ratio["arm_2_joint"] = 100
reduction_ratio["arm_3_joint"] = 100
reduction_ratio["arm_4_joint"] = 100
reduction_ratio["arm_5_joint"] = 336
reduction_ratio["arm_6_joint"] = 336
reduction_ratio["arm_7_joint"] = 336

# current constant (Nm/A)
k_motor = dict()
k_motor["torso_lift_joint"] = 1
k_motor["arm_1_joint"] = 0.136
k_motor["arm_2_joint"] = 0.136
k_motor["arm_3_joint"] = -0.087
k_motor["arm_4_joint"] = -0.087
k_motor["arm_5_joint"] = -0.0613
k_motor["arm_6_joint"] = -0.0613
k_motor["arm_7_joint"] = -0.0613

pin.computeSubtreeMasses(robot.model, robot.data)
for i, jointn_ in enumerate(active_joints):
    if jointn_ == "torso_lift_joint":
        tau[:, i] = (
            reduction_ratio[jointn_] * k_motor[jointn_] * tau[:, i]
            + 9.81 * robot.data.mass[robot.model.getJointId(jointn_)]
        )
    else:
        tau[:, i] = reduction_ratio[jointn_] * k_motor[jointn_] * tau[:, i]

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
print(W.shape, len(standard_parameter))

# remove zero columns
idx_e, active_parameter = get_index_eliminate(
    W, standard_parameter, tol_e=0.001
)
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
    W_rf[range(i * W_list[i].shape[0], (i + 1) * W_list[i].shape[0]), :] = (
        W_list[i]
    )

# time
t_dec = signal.decimate(t[:, 0], q=10, zero_phase=True)


# calculate base parameters
W_b, bp_dict, base_parameter, phi_b, phi_std = double_QR(
    tau_rf, W_rf, active_parameter, standard_parameter
)
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
phi_ref = np.array(list(standard_parameter.values()))
tau_ref = np.dot(W, phi_ref)
tau_ref = tau_ref[range(len(params_settings["idx_act_joints"]) * Ntotal)]

# plot joint torque
with plt.style.context(["seaborn-darkgrid"]):
    plot2 = plt.figure(2)
    axs2 = plot2.subplots(len(active_joints), 1)
    plt.style.library["seaborn-darkgrid"].update(
        {"font.size": 18, "grid.color": "grey"}
    )
    for i in range(len(params_settings["idx_act_joints"])):
        if i == 0:
            axs2[i].plot(
                t_dec,
                tau_dec[i],
                color="red",
                label="torque estimated from motor current",
            )
            axs2[i].plot(
                t_dec,
                tau_base[
                    range(
                        i * tau_dec[i].shape[0], (i + 1) * tau_dec[i].shape[0]
                    )
                ],
                color="green",
                label="torque estimated by identified model",
            )
            axs2[i].plot(
                t,
                tau_ref[range(i * Ntotal, (i + 1) * Ntotal)],
                color="blue",
                label="torque estimated by initial model",
                linestyle="--",
            )
            axs2[i].set_ylabel(
                "torso [N]", rotation="horizontal", fontsize=15, ha="right"
            )
            axs2[i].tick_params(labelbottom=False, bottom=False)
            # axs2[i].axhline(eff_lims[i], t[0], t[-1])
            # axs2[i].grid()
        elif i < len(active_joints):
            axs2[i].plot(t_dec, tau_dec[i], color="red")
            axs2[i].plot(
                t_dec,
                tau_base[
                    range(
                        i * tau_dec[i].shape[0], (i + 1) * tau_dec[i].shape[0]
                    )
                ],
                color="green",
            )
            axs2[i].plot(
                t,
                tau_ref[range(i * Ntotal, (i + 1) * Ntotal)],
                color="blue",
                linestyle="--",
            )
            axs2[i].set_ylabel(
                "arm %d [N.m]" % i,
                rotation="horizontal",
                fontsize=15,
                ha="right",
            )
            axs2[i].tick_params(labelbottom=False, bottom=False)
            # axs2[i].grid()

            if i == len(active_joints) - 1:
                axs2[i].set_xlabel("time [sec]", fontsize=15)
                axs2[i].tick_params(axis="y", color="black")
                axs2[i].tick_params(labelbottom=True, bottom=True)

    plot2.legend()
plt.show()

# x = np.arange(len(base_parameter))
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
#     for i in range(len(base_parameter)):
#         w.writerow(
#             [base_parameter[i], phi_b[i], phi_std[i], 100*std_xr_ols[i]]
#         )
