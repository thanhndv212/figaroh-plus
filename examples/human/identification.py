# Copyright [2021-2025] Thanh Nguyen
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
import numpy as np
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import (
    build_regressor_basic,
    get_index_eliminate,
    build_regressor_reduced,
)
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import (
    get_param_from_yaml,
    base_param_from_standard,
    calculate_first_second_order_differentiation,
    calculate_standard_parameters,
    low_pass_filter_data,
)
import matplotlib.pyplot as plt
import csv
import yaml
import pandas as pd
from yaml.loader import SafeLoader
from human_calibration_tools import (
    make_markers_dict_notime,
    compute_joint_centers,
    compute_mean_joints_centers,
    scale_human_model_mocap,
    calibrate_human_model_mocap,
    get_local_markers,
    markers_local_for_df,
    mean_local_markers,
    add_plug_in_gait_markers,
)

robot_mocap = Robot(
    "models/human_description/urdf/human.urdf",
    "models",
    True,
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
)
model_mocap = robot_mocap.model
data_mocap = robot_mocap.data

joints_width = [
    0.124 / 2,
    0.124 / 2,
    0.089 / 2,
    0.089 / 2,
    0.104 / 2,
    0.104 / 2,
    0.074 / 2,
    0.074 / 2,
]  # KNEE ANKLE ELBOW WRIST

path_qmocap = "examples/human/data/qmocap.txt"
path_mocap_data = "examples/human/data/Calib.csv"
path_tau_meas_raw = "examples/human/data/tau_meas_raw.txt"

with open("examples/human/config/human_config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)

identif_data = config["identification"]

params_settings = get_param_from_yaml(robot_mocap, identif_data)
params_settings["ts"] = 0.02

# LOAD KINEMATICS DATA

q_mocap = np.loadtxt(path_qmocap, delimiter=",")

with open(path_mocap_data, newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
    no_lines = len(list(spamreader)) - 1

markers_names = [
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "C7",
    "T10",
    "CLAV",
    "STRN",
    "RBAK",
    "LSHO",
    "LELB",
    "LWRA",
    "LWRB",
    "LFIN",
    "RSHO",
    "RELB",
    "RWRA",
    "RWRB",
    "RFIN",
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LTHI",
    "LKNE",
    "LTIB",
    "LANK",
    "LHEE",
    "LTOE",
    "RTHI",
    "RKNE",
    "RTIB",
    "RANK",
    "RHEE",
    "RTOE",
]

markers_trajectories = np.zeros((no_lines, 3 * len(markers_names)))
forces_trajectories_mocap = np.loadtxt(path_tau_meas_raw, delimiter=",")

forces_trajectories_mocap[:, 0] -= np.mean(forces_trajectories_mocap[:, 0])
forces_trajectories_mocap[:, 1] -= np.mean(forces_trajectories_mocap[:, 1])

c = 0

with open(path_mocap_data, newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
    for row in spamreader:
        if "time" in row[0]:
            print("First")
        else:
            new_row = row[0].split(",")
            for ii in range(1, 3 * len(markers_names) + 1):
                markers_trajectories[c, ii - 1] = float(new_row[ii])
            c += 1

for ii in range(forces_trajectories_mocap.shape[1]):
    if ii == 0:
        forces_trajectories_mocap_filt = low_pass_filter_data(
            forces_trajectories_mocap[:, ii], params_settings, 5
        )
    else:
        forces_trajectories_mocap_filt = np.column_stack(
            (
                forces_trajectories_mocap_filt,
                low_pass_filter_data(
                    forces_trajectories_mocap[:, ii], params_settings, 5
                ),
            )
        )

forces_trajectories_mocap_filt = forces_trajectories_mocap_filt[:-2, :]

# Model Calibration

q_calib = []
dataframe = []
joints_centers_list = []

q_tPose = 0.0001 * np.ones((model_mocap.nq,))

q_tPose[23] = np.pi / 2
q_tPose[25] = 0.1
q_tPose[26] = 4 * np.pi / 9
q_tPose[34] = np.pi / 2
q_tPose[36] = 0.1
q_tPose[37] = 4 * np.pi / 9

q_init = q_tPose

for ii in range(10):
    Position_markers_calib = np.zeros((len(markers_names), 3))
    for jj in range(len(markers_names)):
        Position_markers_calib[jj, :] = markers_trajectories[ii, 3 * jj : 3 * (jj + 1)]

    dictio = make_markers_dict_notime(
        Position_markers_calib, markers_names, joints_width
    )

    joints_centers = compute_joint_centers(dictio)
    joints_centers_list.append(joints_centers)

    # Change body segments sizes
    model_mocap, data_mocap = scale_human_model_mocap(model_mocap, joints_centers)

    q0 = calibrate_human_model_mocap(model_mocap, data_mocap, joints_centers, q_init)
    q_init = q0

    q_calib.append(q0)

    markers_local = get_local_markers(model_mocap, data_mocap, q0, dictio)

    markers_for_df = markers_local_for_df(markers_local)

    dataframe.append(markers_for_df)

df = pd.DataFrame(dataframe)

for ii in [
    "LFHDx",
    "LFHDy",
    "LFHDz",
    "RFHDx",
    "RFHDy",
    "RFHDz",
    "LBHDx",
    "LBHDy",
    "LBHDz",
    "RBHDx",
    "RBHDy",
    "RBHDz",
    "C7x",
    "C7y",
    "C7z",
    "T10x",
    "T10y",
    "T10z",
    "CLAVx",
    "CLAVy",
    "CLAVz",
    "STRNx",
    "STRNy",
    "STRNz",
    "RBAKx",
    "RBAKy",
    "RBAKz",
    "LSHOx",
    "LSHOy",
    "LSHOz",
    "LELBx",
    "LELBy",
    "LELBz",
    "LWRAx",
    "LWRAy",
    "LWRAz",
    "LWRBx",
    "LWRBy",
    "LWRBz",
    "LFINx",
    "LFINy",
    "LFINz",
    "RSHOx",
    "RSHOy",
    "RSHOz",
    "RELBx",
    "RELBy",
    "RELBz",
    "RWRAx",
    "RWRAy",
    "RWRAz",
    "RWRBx",
    "RWRBy",
    "RWRBz",
    "RFINx",
    "RFINy",
    "RFINz",
    "LASIx",
    "LASIy",
    "LASIz",
    "RASIx",
    "RASIy",
    "RASIz",
    "LPSIx",
    "LPSIy",
    "LPSIz",
    "RPSIx",
    "RPSIy",
    "RPSIz",
    "LTHIx",
    "LTHIy",
    "LTHIz",
    "LKNEx",
    "LKNEy",
    "LKNEz",
    "LTIBx",
    "LTIBy",
    "LTIBz",
    "LANKx",
    "LANKy",
    "LANKz",
    "LHEEx",
    "LHEEy",
    "LHEEz",
    "LTOEx",
    "LTOEy",
    "LTOEz",
    "RTHIx",
    "RTHIy",
    "RTHIz",
    "RKNEx",
    "RKNEy",
    "RKNEz",
    "RTIBx",
    "RTIBy",
    "RTIBz",
    "RANKx",
    "RANKy",
    "RANKz",
    "RHEEx",
    "RHEEy",
    "RHEEz",
    "RTOEx",
    "RTOEy",
    "RTOEz",
]:
    q1 = df[ii].quantile(q=0.25)
    q3 = df[ii].quantile(q=0.75)
    IQR = q3 - q1
    borne_inf = q1 - 1.5 * IQR
    borne_sup = q3 + 1.5 * IQR
    df = df[df[ii] < borne_sup]
    df = df[df[ii] > borne_inf]

mean = df.mean()
joints_centers_glo = compute_mean_joints_centers(joints_centers_list)

markers_local = mean_local_markers(mean)
model_mocap, data_mocap = scale_human_model_mocap(model_mocap, joints_centers_glo)
data_mocap, i_markers = add_plug_in_gait_markers(model_mocap, data_mocap, markers_local)

rotvec_list = []

for ii in range(len(q_mocap)):
    quat = pin.Quaternion(
        np.array([q_mocap[ii, 3], q_mocap[ii, 4], q_mocap[ii, 5], q_mocap[ii, 6]])
    )
    rotvec_ii = pin.log3(quat)
    rotvec_list.append(rotvec_ii)

rotvec_nofilt = np.array(rotvec_list)

for ii in range(rotvec_nofilt.shape[1]):
    if ii == 0:
        rotvec = low_pass_filter_data(rotvec_nofilt[:, ii], params_settings, 5)
    else:
        rotvec = np.column_stack(
            (rotvec, low_pass_filter_data(rotvec_nofilt[:, ii], params_settings, 5))
        )

quat_filt = np.zeros((rotvec.shape[0], 4))

for ii in range(rotvec.shape[0]):
    rotvec_ii = rotvec[ii, :]
    quat_ref = np.array(
        [q_mocap[ii, 3], q_mocap[ii, 4], q_mocap[ii, 5], q_mocap[ii, 6]]
    )
    quat_ii = pin.Quaternion((pin.exp3(rotvec_ii)))
    quat_ii = np.array([quat_ii[0], quat_ii[1], quat_ii[2], quat_ii[3]])
    dot_product = np.dot(quat_ii, quat_ref)
    angle = np.arccos(dot_product) * 2
    if angle < np.pi / 2:
        quat_ii = quat_ii
    else:
        quat_ii = -quat_ii
    quat_filt[ii, :] = quat_ii

for ii in range(model_mocap.nq):
    if ii == 0:
        q_m = low_pass_filter_data(q_mocap[:, ii], params_settings, 5)
    else:
        q_m = np.column_stack(
            (q_m, low_pass_filter_data(q_mocap[:, ii], params_settings, 5))
        )

q_m[:, 3:7] = quat_filt

# Identif

# generate a list containing the full set of standard parameters
params_standard_mocap = robot_mocap.get_standard_parameters(params_settings)

# 1. First we build the structural base identification model, i.e. the one that can
# be observed, using random samples

q_rand = []
dq_rand = []
ddq_rand = []

for ii in range(params_settings["nb_samples"] + 2):
    q_rand.append(pin.randomConfiguration(model_mocap))

for ii in range(params_settings["nb_samples"] + 1):
    dq_rand.append(pin.difference(model_mocap, q_rand[ii + 1], q_rand[ii]))

for ii in range(params_settings["nb_samples"]):
    ddq_rand.append(dq_rand[ii + 1] - dq_rand[ii] / params_settings["ts"])

q_rand = np.array(q_rand[:-2])
dq_rand = np.array(dq_rand[:-1])
ddq_rand = np.array(ddq_rand)

nb_samples = len(q_rand)

W = build_regressor_basic(robot_mocap, q_rand, dq_rand, ddq_rand, params_settings)

# remove zero cols and build a zero columns free regressor matrix
idx_e, params_r = get_index_eliminate(W, params_standard_mocap, 1e-6)
W_e = build_regressor_reduced(W, idx_e)

# Calulate the base regressor matrix, the base regroupings equations params_base and
# get the idx_base, ie. the index of base parameters in the initial regressor matrix

W_base, params_base, idx_base = get_baseParams(W_e, params_r, params_standard_mocap)

num_phi_base = base_param_from_standard(params_standard_mocap, params_base)

print("When using random trajectories the cond num is", int(np.linalg.cond(W_base)))

print("The structural base parameters are: ")
for ii in range(len(params_base)):
    print(params_base[ii], ", urdf value =", num_phi_base[ii])

# Identif with real data

q_m, dq_m, ddq_m = calculate_first_second_order_differentiation(
    model_mocap, q_m, params_settings
)

nb_samples = len(q_m)

tau_meas_mocap = forces_trajectories_mocap_filt.flatten("F")

# Forces transported

tau_meast_mocap = np.empty(nb_samples * 6)

for ii in range(nb_samples):
    pin.forwardKinematics(model_mocap, data_mocap, q_m[ii, :])
    pin.updateFramePlacements(model_mocap, data_mocap)
    M_pelvis = data_mocap.oMi[model_mocap.getJointId("root_joint")]
    tau_temp = pin.Force(forces_trajectories_mocap_filt[ii, :])
    tau_temp = tau_temp.se3ActionInverse(M_pelvis)
    for j in range(6):
        tau_temp_sub = np.array(
            [
                tau_temp.linear[0],
                tau_temp.linear[1],
                tau_temp.linear[2],
                tau_temp.angular[0],
                tau_temp.angular[1],
                tau_temp.angular[2],
            ]
        )
        tau_meast_mocap[j * nb_samples + ii] = tau_temp_sub[j]

W_mocap = build_regressor_basic(robot_mocap, q_m, dq_m, ddq_m, params_settings)

# remove zero cols and build a zero columns free regressor matrix
W_e_mocap = build_regressor_reduced(W_mocap, idx_e)

# select only the columns of the regressor corresponding to the structural base
# parameters
W_base_mocap = W_e_mocap[:, idx_base]

print("When using all trajectories the cond num is", int(np.linalg.cond(W_base_mocap)))

# Least-square identification process
phi_base_mocap = np.matmul(np.linalg.pinv(W_base_mocap), tau_meast_mocap)

tau_identif = W_base_mocap @ phi_base_mocap

tau_simu_mocap = np.empty(nb_samples * 6)

for ii in range(nb_samples):
    pin.forwardKinematics(model_mocap, data_mocap, q_m[ii, :])
    pin.updateFramePlacements(model_mocap, data_mocap)
    M_pelvis = data_mocap.oMi[model_mocap.getJointId("root_joint")]
    tau_temp = pin.Force(
        np.array(
            pin.rnea(model_mocap, data_mocap, q_m[ii, :], dq_m[ii, :], ddq_m[ii, :])[:6]
        )
    )  # force de réaction
    tau_temp = pin.Force(
        np.array(
            [
                tau_temp.linear[0],
                tau_temp.linear[1],
                tau_temp.linear[2],
                tau_temp.angular[0],
                tau_temp.angular[1],
                tau_temp.angular[2],
            ]
        )
    )
    tau_temp = tau_temp.se3Action(M_pelvis)
    for j in range(6):
        tau_temp_sub = np.array(
            [
                tau_temp.linear[0],
                tau_temp.linear[1],
                tau_temp.linear[2],
                tau_temp.angular[0],
                tau_temp.angular[1],
                tau_temp.angular[2],
            ]
        )
        tau_simu_mocap[j * nb_samples + ii] = tau_temp_sub[j]

# Forces visualisation

Fx_identif = []
Fy_identif = []
Fz_identif = []
Mx_identif = []
My_identif = []
Mz_identif = []

for ii in range(int(len(tau_identif) / 6)):
    Fx_identif.append(tau_identif[ii])
    Fy_identif.append(tau_identif[int(1 * (len(tau_identif) / 6)) + ii])
    Fz_identif.append(tau_identif[int(2 * (len(tau_identif) / 6)) + ii])
    Mx_identif.append(tau_identif[int(3 * (len(tau_identif) / 6)) + ii])
    My_identif.append(tau_identif[int(4 * (len(tau_identif) / 6)) + ii])
    Mz_identif.append(tau_identif[int(5 * (len(tau_identif) / 6)) + ii])

tau_identif_proj_mocap = np.empty(nb_samples * 6)

for ii in range(nb_samples):
    pin.forwardKinematics(model_mocap, data_mocap, q_m[ii, :])
    pin.updateFramePlacements(model_mocap, data_mocap)
    M_pelvis = data_mocap.oMi[model_mocap.getJointId("root_joint")]
    tau_temp = pin.Force(
        np.array(
            [
                Fx_identif[ii],
                Fy_identif[ii],
                Fz_identif[ii],
                Mx_identif[ii],
                My_identif[ii],
                Mz_identif[ii],
            ]
        )
    )  # force de réaction
    tau_temp = tau_temp.se3Action(M_pelvis)
    for j in range(6):
        tau_temp_sub = np.array(
            [
                tau_temp.linear[0],
                tau_temp.linear[1],
                tau_temp.linear[2],
                tau_temp.angular[0],
                tau_temp.angular[1],
                tau_temp.angular[2],
            ]
        )
        tau_identif_proj_mocap[j * nb_samples + ii] = tau_temp_sub[j]

# PLOTTING FORCE TRACKING

plt.plot(tau_meas_mocap, "k", label="measured")
plt.plot(tau_simu_mocap, "b", label="Mocap AT")
plt.plot(tau_identif_proj_mocap, "g", label="Mocap identified")
plt.legend()
plt.title("Overall wrench (measured, simulated with AT and identified")
plt.show()

# QP SIP

phi_ref = []
id_inertias = []

for jj in range(len(model_mocap.inertias.tolist())):
    if model_mocap.inertias.tolist()[jj].mass != 0:
        id_inertias.append(jj)

nreal = len(id_inertias)

params_name = (
    "Ixx",
    "Ixy",
    "Ixz",
    "Iyy",
    "Iyz",
    "Izz",
    "mx",
    "my",
    "mz",
    "m",
)

for k in range(nreal):
    for j in params_name:
        phi_ref_temp = params_standard_mocap[j + str(id_inertias[k])]
        phi_ref.append(phi_ref_temp)

phi_ref = np.array(phi_ref)

COM_max = []
COM_min = []

for ii in range(nreal):
    for kk in range(3):
        if phi_ref[10 * ii + 6 + kk] > 0:
            COM_max.append(1.3 * phi_ref[10 * ii + 6 + kk])
            COM_min.append(0.7 * phi_ref[10 * ii + 6 + kk])
        elif phi_ref[10 * ii + 6 + kk] < 0:
            COM_max.append(0.7 * phi_ref[10 * ii + 6 + kk])
            COM_min.append(1.3 * phi_ref[10 * ii + 6 + kk])
        elif phi_ref[10 * ii + 6 + kk] == 0:
            COM_max.append(0.001)
            COM_min.append(-0.001)

COM_max = np.array(COM_max)
COM_min = np.array(COM_min)

print(W_e_mocap.shape)

phi_standard_mocap, phi_AT = calculate_standard_parameters(
    model_mocap,
    W_e_mocap,
    tau_meast_mocap,
    COM_max,
    COM_min,
    params_standard_mocap,
    0.33,
)

### COMPARATIVE BAR PLOT OF THE STANDARD PARAMETERS ###

id_inertias_mocap = []

for jj in range(len(model_mocap.inertias.tolist())):
    if model_mocap.inertias.tolist()[jj].mass != 0:
        id_inertias_mocap.append(jj)

phi_m_AT = []
phi_m_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        phi_m_AT.append(phi_AT[10 * ii + 9])
        phi_m_est_mocap.append(phi_standard_mocap[10 * ii + 9])

phi_m_AT = np.array(phi_m_AT)
phi_m_est_mocap = np.array(phi_m_est_mocap)

phi_comx_AT = []
phi_comx_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        phi_comx_AT.append(phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9])
        phi_comx_est_mocap.append(
            phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9]
        )

phi_comx_AT = np.array(phi_comx_AT)
phi_comx_est_mocap = np.array(phi_comx_est_mocap)

phi_comy_AT = []
phi_comy_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        phi_comy_AT.append(phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9])
        phi_comy_est_mocap.append(
            phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9]
        )

phi_comy_AT = np.array(phi_comy_AT)
phi_comy_est_mocap = np.array(phi_comy_est_mocap)

phi_comz_AT = []
phi_comz_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        phi_comz_AT.append(phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9])
        phi_comz_est_mocap.append(
            phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9]
        )

phi_comz_AT = np.array(phi_comz_AT)
phi_comz_est_mocap = np.array(phi_comz_est_mocap)

phi_ixx_AT = []
phi_ixx_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        S_AT = np.array(
            [
                [
                    0,
                    -phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9],
                    phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9],
                ],
                [
                    phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9],
                    0,
                    -phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9],
                ],
                [
                    -phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9],
                    phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9],
                    0,
                ],
            ]
        )
        mSTS_AT = phi_AT[10 * ii + 9] * S_AT.T @ S_AT
        phi_ixx_AT.append(phi_AT[10 * ii] - mSTS_AT[0, 0])

        S_mocap = np.array(
            [
                [
                    0,
                    -phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9],
                    phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9],
                ],
                [
                    phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9],
                    0,
                    -phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9],
                ],
                [
                    -phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9],
                    phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9],
                    0,
                ],
            ]
        )
        mSTS_mocap = phi_standard_mocap[10 * ii + 9] * S_mocap.T @ S_mocap
        phi_ixx_est_mocap.append(phi_standard_mocap[10 * ii] - mSTS_mocap[0, 0])

phi_ixx_AT = np.array(phi_ixx_AT)
phi_ixx_est_mocap = np.array(phi_ixx_est_mocap)

phi_iyy_AT = []
phi_iyy_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        S_AT = np.array(
            [
                [
                    0,
                    -phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9],
                    phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9],
                ],
                [
                    phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9],
                    0,
                    -phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9],
                ],
                [
                    -phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9],
                    phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9],
                    0,
                ],
            ]
        )
        mSTS_AT = phi_AT[10 * ii + 9] * S_AT.T @ S_AT
        phi_iyy_AT.append(phi_AT[10 * ii + 3] - mSTS_AT[1, 1])

        S_mocap = np.array(
            [
                [
                    0,
                    -phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9],
                    phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9],
                ],
                [
                    phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9],
                    0,
                    -phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9],
                ],
                [
                    -phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9],
                    phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9],
                    0,
                ],
            ]
        )
        mSTS_mocap = phi_standard_mocap[10 * ii + 9] * S_mocap.T @ S_mocap
        phi_iyy_est_mocap.append(phi_standard_mocap[10 * ii + 3] - mSTS_mocap[1, 1])

phi_iyy_AT = np.array(phi_iyy_AT)
phi_iyy_est_mocap = np.array(phi_iyy_est_mocap)

phi_izz_AT = []
phi_izz_est_mocap = []

for ii in range(len(id_inertias_mocap)):
    if ii != 0 and ii != 7 and ii != 12:
        S_AT = np.array(
            [
                [
                    0,
                    -phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9],
                    phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9],
                ],
                [
                    phi_AT[10 * ii + 8] / phi_AT[10 * ii + 9],
                    0,
                    -phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9],
                ],
                [
                    -phi_AT[10 * ii + 7] / phi_AT[10 * ii + 9],
                    phi_AT[10 * ii + 6] / phi_AT[10 * ii + 9],
                    0,
                ],
            ]
        )
        mSTS_AT = phi_AT[10 * ii + 9] * S_AT.T @ S_AT
        phi_izz_AT.append(phi_AT[10 * ii + 5] - mSTS_AT[2, 2])

        S_mocap = np.array(
            [
                [
                    0,
                    -phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9],
                    phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9],
                ],
                [
                    phi_standard_mocap[10 * ii + 8] / phi_standard_mocap[10 * ii + 9],
                    0,
                    -phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9],
                ],
                [
                    -phi_standard_mocap[10 * ii + 7] / phi_standard_mocap[10 * ii + 9],
                    phi_standard_mocap[10 * ii + 6] / phi_standard_mocap[10 * ii + 9],
                    0,
                ],
            ]
        )
        mSTS_mocap = phi_standard_mocap[10 * ii + 9] * S_mocap.T @ S_mocap
        phi_izz_est_mocap.append(phi_standard_mocap[10 * ii + 5] - mSTS_mocap[2, 2])

phi_izz_AT = np.array(phi_izz_AT)
phi_izz_est_mocap = np.array(phi_izz_est_mocap)

names = [
    "l_ul",
    "l_ll",
    "l_f",
    "pel",
    "abd",
    "tho",
    "l_ua",
    "l_la",
    "l_h",
    "hd",
    "r_ua",
    "r_la",
    "r_h",
    "r_ul",
    "r_ll",
    "r_f",
]

list_of_dict_AT = []
list_of_dict_mocap = []

m_AT = dict(zip(names, phi_m_AT))
m_est_mocap = dict(zip(names, phi_m_est_mocap))

list_of_dict_AT.append(m_AT)
list_of_dict_mocap.append(m_est_mocap)

print("m_AT = ", m_AT)
print("m_est_mocap = ", m_est_mocap)

X = np.arange(len(m_AT))
ax = plt.subplot(111)
ax.bar(X, m_AT.values(), width=0.2, color="b", align="center")
ax.bar(X - 0.2, m_est_mocap.values(), width=0.2, color="g", align="center")
ax.legend(("AT", "mocap"))
plt.xticks(X, m_AT.keys())
plt.title("MASS [kg]", fontsize=17)
plt.show()


comx_AT = dict(zip(names, phi_comx_AT))
comx_est_mocap = dict(zip(names, phi_comx_est_mocap))

list_of_dict_AT.append(comx_AT)
list_of_dict_mocap.append(comx_est_mocap)

print("comx_AT = ", comx_AT)
print("comx_est_mocap = ", comx_est_mocap)

X = np.arange(len(comx_AT))
ax = plt.subplot(111)
ax.bar(X, comx_AT.values(), width=0.2, color="b", align="center", label="AT")
ax.bar(
    X - 0.2,
    comx_est_mocap.values(),
    width=0.2,
    color="g",
    align="center",
    label="mocap",
)
ax.legend()
plt.xticks(X, comx_AT.keys())
plt.title("MSX [kg.m]", fontsize=17)
plt.show()


comy_AT = dict(zip(names, phi_comy_AT))
comy_est_mocap = dict(zip(names, phi_comy_est_mocap))

list_of_dict_AT.append(comy_AT)
list_of_dict_mocap.append(comy_est_mocap)

print("comy_AT = ", comy_AT)
print("comy_est_mocap = ", comy_est_mocap)

X = np.arange(len(comy_AT))
ax = plt.subplot(111)
ax.bar(X, comy_AT.values(), width=0.2, color="b", align="center", label="AT")
ax.bar(
    X - 0.2,
    comy_est_mocap.values(),
    width=0.2,
    color="g",
    align="center",
    label="mocap",
)
ax.legend()
plt.xticks(X, comy_AT.keys())
plt.title("MSY [kg.m]", fontsize=17)
plt.show()


comz_AT = dict(zip(names, phi_comz_AT))
comz_est_mocap = dict(zip(names, phi_comz_est_mocap))

list_of_dict_AT.append(comz_AT)
list_of_dict_mocap.append(comz_est_mocap)

print("comz_AT = ", comz_AT)
print("comz_est_mocap = ", comz_est_mocap)

X = np.arange(len(comz_AT))
ax = plt.subplot(111)
ax.bar(X, comz_AT.values(), width=0.2, color="b", align="center", label="AT")
ax.bar(
    X - 0.2,
    comz_est_mocap.values(),
    width=0.2,
    color="g",
    align="center",
    label="mocap",
)
ax.legend()
plt.xticks(X, comz_AT.keys())
plt.title("MSZ [kg.m]", fontsize=17)
plt.show()


ixx_AT = dict(zip(names, phi_ixx_AT))
ixx_est_mocap = dict(zip(names, phi_ixx_est_mocap))

list_of_dict_AT.append(ixx_AT)
list_of_dict_mocap.append(ixx_est_mocap)

X = np.arange(len(ixx_AT))
ax = plt.subplot(111)
ax.bar(X, ixx_AT.values(), width=0.2, color="b", align="center")
ax.bar(X - 0.2, ixx_est_mocap.values(), width=0.2, color="g", align="center")
ax.legend(("AT", "mocap", "Xsens"))
plt.xticks(X, ixx_AT.keys())
plt.title("Ixx [kg.m²]", fontsize=17)
plt.show()


iyy_AT = dict(zip(names, phi_iyy_AT))
iyy_est_mocap = dict(zip(names, phi_iyy_est_mocap))

list_of_dict_AT.append(iyy_AT)
list_of_dict_mocap.append(iyy_est_mocap)

X = np.arange(len(iyy_AT))
ax = plt.subplot(111)
ax.bar(X, iyy_AT.values(), width=0.2, color="b", align="center")
ax.bar(X - 0.2, iyy_est_mocap.values(), width=0.2, color="g", align="center")
ax.legend(("AT", "mocap", "Xsens"))
plt.xticks(X, iyy_AT.keys())
plt.title("Iyy [kg.m²]", fontsize=17)
plt.show()


izz_AT = dict(zip(names, phi_izz_AT))
izz_est_mocap = dict(zip(names, phi_izz_est_mocap))

list_of_dict_AT.append(izz_AT)
list_of_dict_mocap.append(izz_est_mocap)

X = np.arange(len(izz_AT))
ax = plt.subplot(111)
ax.bar(X, izz_AT.values(), width=0.2, color="b", align="center")
ax.bar(X - 0.2, izz_est_mocap.values(), width=0.2, color="g", align="center")
ax.legend(("AT", "mocap", "Xsens"))
plt.xticks(X, izz_AT.keys())
plt.title("Izz [kg.m²]", fontsize=17)
plt.show()
