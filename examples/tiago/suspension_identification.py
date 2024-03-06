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
from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
    get_index_eliminate,
)
import yaml
from yaml.loader import SafeLoader
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.tools.qrdecomposition import get_baseParams


def calc_floatingbase_pose(
    base_marker_name: str,
    marker_data: dict,
    marker_base: dict,
    Mref=pin.SE3.Identity(),
):
    """Calculate floatingbase pose w.r.t a fixed frame from measures expressed
    in mocap frame.

    Args:
        marker_data (dict): vicon marker data
        marker_base (dict): SE3 from a marker to a fixed base
        base_marker_name (str): base marker name
        Mref (_type_, optional): Defaults to pin.SE3.Identity().

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray : position, rpy, quaternion
    """

    Mmarker_floatingbase = cartesian_to_SE3(marker_base[base_marker_name])
    [base_trans, base_rot] = marker_data[base_marker_name]

    n_ = len(base_rot)
    xyz_u = np.zeros((n_, 3))
    rpy_u = np.zeros((n_, 3))
    quat_u = np.zeros((n_, 4))

    for i in range(n_):
        SE3_floatingbase = (
            Mref
            * pin.SE3(base_rot[i], base_trans[i, :])
            * Mmarker_floatingbase
        )

        xyz_u[i, :] = SE3_floatingbase.translation
        rpy_u[i, :] = pin.rpy.matrixToRpy(SE3_floatingbase.rotation)
        quat_u[i, :] = pin.Quaternion(SE3_floatingbase.rotation).coeffs()

    return xyz_u, rpy_u, quat_u


def estimate_fixed_base(
    base_marker_name: str,
    marker_data: dict,
    marker_fixedbase: dict,
    stationary_range: range,
    Mref=pin.SE3.Identity(),
):
    """Estimate fixed base frame of robot expressed in a fixed frame

    Args:
        marker_data (dict): vicon marker data
        marker_fixedbase (dict): SE3 from a marker to a fixed base
        base_marker_name (str): base marker name
        stationary_range (range): range to extract data
        Mref (SE3, optional): Defaults to pin.SE3.Identity() as mocap frame.

    Returns:
        SE3: transformatiom SE3
    """

    Mmarker_fixedbase = cartesian_to_SE3(marker_fixedbase[base_marker_name])
    [base_trans, base_rot] = marker_data[base_marker_name]

    ne_ = len(stationary_range)
    xyz_u_ = np.zeros((ne_, 3))
    rpy_u_ = np.zeros((ne_, 3))

    for j, i in enumerate(stationary_range):
        SE3_fixdebase = (
            pin.SE3(base_rot[i], base_trans[i, :]) * Mmarker_fixedbase
        )
        xyz_u_[j, :] = SE3_fixdebase.translation
        rpy_u_[j, :] = pin.rpy.matrixToRpy(SE3_fixdebase.rotation)

    xyz_mean = np.mean(xyz_u_, axis=0)
    rpy_mean = np.mean(rpy_u_, axis=0)

    return cartesian_to_SE3(np.append(xyz_mean, rpy_mean))


def compute_estimated_pee(
    endframe_name: str,
    base_marker_name: str,
    marker_fixedbase: dict,
    Mmocap_fixedbase,
    Mendframe_marker,
    mocap_range_: range,
    q_arm: np.ndarray,
):
    """_summary_

    Args:
        endframe_name (str): name of the end of kinematic chain
        base_marker_name (str): name of base marker measured
        marker_fixedbase (dict): fixed transformation from marker to fixed base
        Mmocap_fixedbase (SE3): transf. from mocap to fixed base
        Mendframe_marker (SE3): transf. from end of kin.chain to last marker
        mocap_range_ (range): range of selected mocap data
        q_arm (np.ndarray): encoder data within 'encoder_range'

    Returns:
        numpy.ndarray, numpy.ndarray: estimate of last marker in mocap frame
    """
    assert len(q_arm) == len(
        mocap_range_
    ), "joint configuration range is not matched with mocap range."

    endframe_id = model.getFrameId(endframe_name)
    Mmarker_fixedbase = cartesian_to_SE3(marker_fixedbase[base_marker_name])
    [base_trans, base_rot] = marker_data[base_marker_name]

    nc_ = len(mocap_range_)
    pee_est = np.zeros((nc_, 3))
    peews_est = np.zeros((nc_, 3))

    for jj, ii in enumerate(mocap_range_):
        pin.framesForwardKinematics(model, data, q_arm[jj])
        pin.updateFramePlacements(model, data)

        Mmocap_floatingbase = (
            pin.SE3(base_rot[ii], base_trans[ii, :]) * Mmarker_fixedbase
        )

        pee_SE3 = Mmocap_fixedbase * data.oMf[endframe_id] * Mendframe_marker

        peews_SE3 = (
            Mmocap_floatingbase * data.oMf[endframe_id] * Mendframe_marker
        )

        pee_est[jj, :] = pee_SE3.translation
        peews_est[jj, :] = peews_SE3.translation

    return pee_est, peews_est


def plot_compare_suspension(
    pee_est=None,
    peews_est=None,
    gripper3_pos=None,
    lookup_marker="gripper",
    plot_err=True,
):
    fig, ax = plt.subplots(3, 1)
    t_tf2 = []
    t_tf = []
    if pee_est is not None:
        t_tf = np.arange(len(pee_est))
        ax[0].plot(
            t_tf,
            peews_est[:, 0],
            color="red",
            label="estimated with suspension added",
        )
        ax[1].plot(
            t_tf,
            peews_est[:, 1],
            color="blue",
            label="estimated with suspension added",
        )
        ax[2].plot(
            t_tf,
            peews_est[:, 2],
            color="green",
            label="estimated with suspension added",
        )
    if peews_est is not None:
        t_tf = np.arange(len(peews_est))
        ax[0].plot(
            t_tf,
            pee_est[:, 0],
            color="red",
            linestyle="dotted",
            label="estimated without suspension added",
        )
        ax[1].plot(
            t_tf,
            pee_est[:, 1],
            color="blue",
            linestyle="dotted",
            label="estimated without suspension added",
        )
        ax[2].plot(
            t_tf,
            pee_est[:, 2],
            color="green",
            linestyle="dotted",
            label="estimated without suspension added",
        )
    if gripper3_pos is not None:
        t_tf2 = np.arange(len(gripper3_pos))
        ax[0].plot(
            t_tf2,
            gripper3_pos[:, 0],
            color="red",
            label="measured",
            linestyle="--",
        )
        ax[1].plot(
            t_tf2,
            gripper3_pos[:, 1],
            color="blue",
            label="measured",
            linestyle="--",
        )
        ax[2].plot(
            t_tf2,
            gripper3_pos[:, 2],
            color="green",
            label="measured",
            linestyle="--",
        )

    ax3 = ax[0].twinx()
    ax4 = ax[1].twinx()
    ax5 = ax[2].twinx()
    if plot_err:
        if pee_est is not None and gripper3_pos is not None:
            ax3.bar(
                t_tf,
                pee_est[:, 0] - gripper3_pos[:, 0],
                color="black",
                label="errors - x axis",
                alpha=0.3,
            )
            ax4.bar(
                t_tf,
                pee_est[:, 1] - gripper3_pos[:, 1],
                color="black",
                label="errors - without susspension added",
                alpha=0.3,
            )
            ax5.bar(
                t_tf,
                pee_est[:, 2] - gripper3_pos[:, 2],
                color="black",
                label="errors - z axis",
                alpha=0.3,
            )
        if peews_est is not None and gripper3_pos is not None:

            ax3.bar(
                t_tf,
                peews_est[:, 0] - gripper3_pos[:, 0],
                color="red",
                label="errors - x axis",
                alpha=0.3,
            )
            ax4.bar(
                t_tf,
                peews_est[:, 1] - gripper3_pos[:, 1],
                color="blue",
                label="errors - with susspension added",
                alpha=0.3,
            )
            ax5.bar(
                t_tf,
                peews_est[:, 2] - gripper3_pos[:, 2],
                color="green",
                label="errors - z axis",
                alpha=0.3,
            )
    ax4.legend()
    ax[0].legend()
    ax[0].set_ylabel("x component (meter)")
    ax[1].set_ylabel("y component (meter)")
    ax[2].set_ylabel("z component (meter)")
    ax[2].set_xlabel("sample")
    fig.suptitle("Marker Position of {}".format(lookup_marker))


# fixed frame:
# mocap frame = marker position measures
# forceplate frame = force and moment measures
# fixed base frame = base_footprint (free-flyer joint) at stationary state

# dynamic frame:
# floating base frame = base_footprint in dynamic state
# marker_shoulder frame = marker on shoulder
# marker_base frame = marker on base
# marker_gripper frame = marker on gripper

# [mocap frame] ---- <marker base frame> ---(4) Mmarker_floatingbase---><floating base frame>
#     |        \
#     |                  \
# (1) Mmocap_forceplate                \
#     |                                 (2) Mmocap_fixedbase
#     |                                             \
#     |                                                         \
# [forceplate frame]---(3) Mforceplate_fixedbase----->[fixed base frame or robot frame]


# (1) forceplate - mocap reference frame transformation
forceplate_frame_rot = np.array([[-1, 0.0, 0.0], [0, 1.0, 0], [0, 0, -1]])
forceplate_frame_trans = np.array([0.9, 0.45, 0.0])

Mmocap_forceplate = pin.SE3(forceplate_frame_rot, forceplate_frame_trans)
Mforceplate_mocap = Mmocap_forceplate.inverse()


# (2.1) mocap - fixedbase footprint
Mmocap_fixedfootprint = cartesian_to_SE3(
    np.array(
        [
            9.21473783e-01,
            5.16372267e-01,
            -5.44391042e-03,
            1.42087717e-02,
            7.31715478e-04,
            -1.61396925e00,
        ]
    )
)
Mfixedfootprint_mocap = Mmocap_fixedfootprint.inverse()

# (2.2) mocap - fixedbase baselink
Mmocap_fixedbaselink = cartesian_to_SE3(
    np.array(
        [
            9.20782862e-01,
            5.17310886e-01,
            9.31457902e-02,
            1.42957502e-02,
            6.76446542e-04,
            -1.61388066e00,
        ]
    )
)
Mfixedbaselink_mocap = Mmocap_fixedbaselink.inverse()


# (3.1) forceplate - robot fixed baselink frame transformation
Mforceplate_fixedfootprint = Mforceplate_mocap * Mmocap_fixedfootprint
Mfixedfootprint_forceplate = Mforceplate_fixedfootprint.inverse()

# (3.2) forceplate - robot fixed baselink frame transformation
Mforceplate_fixedbaselink = Mforceplate_mocap * Mmocap_fixedbaselink
Mfixedbaselink_forceplate = Mforceplate_fixedbaselink.inverse()

# (4.1) marker on the base - robot floating base_footprint transformation
marker_footprint = dict()
marker_footprint["base2"] = np.array(
    [
        0.01905164792882577,
        -0.20057504109760418,
        -0.3148863380453684,
        0.006911212803801684,
        0.009815807728356198,
        0.053830497405014326,
    ]
)
Mmarker_footprint = cartesian_to_SE3(marker_footprint["base2"])
Mfootprint_marker = Mmarker_footprint.inverse()

# (4.2) marker on the base - robot floating baselink transformation
marker_baselink = dict()
marker_baselink["base2"] = np.array(
    [
        0.01904,
        -0.200587,
        -0.21628975,
        0.006999,
        0.00976118669378908,
        0.0539194899,
    ]
)
Mmarker_baselink = cartesian_to_SE3(marker_baselink["base2"])
Mbaselink_marker = Mmarker_baselink.inverse()

# (5) gripper_tool_link to gripper 3 marker
Mwrist_gripper3 = cartesian_to_SE3(
    np.array(
        [
            -0.02686200922255708,
            -0.00031620763696974614,
            -0.1514577985136796,
            0,
            0,
            0,
        ]
    )
)

# (6) torso_lift_link to shoulder_1 marker
Mshoulder_torso = cartesian_to_SE3(
    np.array(
        [
            0.14574771716972612,
            0.1581171862116727,
            -0.0176625098292798,
            -0.005016136500979825,
            0.006322745940971755,
            0.027530310085705736,
        ]
    )
)
Mtorso_shoulder = Mshoulder_torso.inverse()


# pu.plot_SE3(pin.SE3.Identity())

# pu.plot_SE3(estimate_fixed_base(marker_datas[0], marker_baselink, "base2", range(50, 1000)), "baselink")

# pu.plot_SE3(pin.SE3(marker_datas[0]["base2"][1][0], marker_datas[0]["base2"][0][0]), "base2")
# pu.plot_SE3(estimate_fixed_base(marker_datas[0], marker_footprint, "base2", range(50, 1000)), "footprint")
# pu.plot_markertf(marker_datas[0]["base2"][0])

##########################################
f_res = 100
f_cutoff = 2
tiago_fb = load_robot(
    abspath("urdf/tiago_48_schunk.urdf"),
    isFext=True,
    load_by_urdf=True,
)
model = tiago_fb.model
data = tiago_fb.data
pu.addBox_to_gripper(tiago_fb)
pu.initiating_robot(tiago_fb)

# load settings
with open("config/tiago_config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)
identif_data = config["identification"]
params_settings = get_param_from_yaml(tiago_fb, identif_data)
standard_parameter = tiago_fb.get_standard_parameters(params_settings)

# load data
file_names = [
    "tiago_around_x_vicon_1634",  # 0
    "tiago_around_y_vicon_1638",  # 1
    "tiago_around_z_vicon_1622",  # 2
    "tiago_x_fold_vicon_1632",  # 3
    "tiago_xyz_mirror_vicon_1630",  # 4
    "tiago_xyz_mirror_vicon_1642",  # 5
    "tiago_xyz_vicon_1628",  # 6
    "tiago_xyz_vicon_1640",  # 7
    "tiago_y_fold_vicon_1636",  # 8
]

marker_datas = []
marker_meas = []
dir_paths = []
for file_name in file_names:
    dir_path_ = "/media/thanhndv212/Cooking/processed_data/tiago/develop/data/identification/suspension/creps/creps_bags/{}/".format(
        file_name
    )
    input_file = dir_path_ + "{}.csv".format(file_name)
    marker_datas.append(pu.process_vicon_data(input_file, f_cutoff))
    marker_meas.append(pu.process_vicon_data(input_file, 70))
    dir_paths.append(dir_path_)
# process data
mocap_ranges = [
    range(2450, 3450),  # 0
    range(2700, 4200),  # 1
    range(2700, 4200),  # 2
    range(5053, 6553),  # 3
    range(2700, 4200),  # 4
    range(2700, 4200),  # 5
    range(2700, 4200),  # 6
    range(3000, 4500),  # 7
    range(3000, 4500),  # 8
]

selected_data = [
    0,  # good fit on mx, my
    # 1, # bad
    # 2, # very good fit on fx
    # 3,  # good fit on mx, my
    # 4, # good fit on fz
    # 5, # good fit on fx
    # 6, # bad
    # 7, # good fit on fz
    # 8, # my, and somehow mz good
]
xyz = []
rpy = []
dxyz = []
drpy = []
tau_mea = []
for ij_ in selected_data:
    marker_data = marker_datas[ij_]
    mocap_range_ = mocap_ranges[ij_]
    NbSample = len(mocap_range_)

    # calculate floatingbase poses and velocities
    xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
        "base2", marker_data, marker_footprint, Mforceplate_mocap
    )
    _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res)
    _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res)

    # measured force plate data
    force = marker_data["force"]
    moment = marker_data["moment"]
    tau_ = np.concatenate(
        (force[mocap_range_, :], moment[mocap_range_, :]), axis=1
    )

    xyz.append(xyz_u)
    rpy.append(rpy_u)
    dxyz.append(dxyz_u)
    drpy.append(drpy_u)
    tau_mea.append(tau_)

# concatenate over several dataset
xyz_u = xyz[0][mocap_ranges[selected_data[0]], :]
rpy_u = rpy[0][mocap_ranges[selected_data[0]], :]
dxyz_u = dxyz[0][mocap_ranges[selected_data[0]], :]
drpy_u = drpy[0][mocap_ranges[selected_data[0]], :]
tau_mea_concat = tau_mea[0]
if len(selected_data) > 1:
    for ij in range(1, len(selected_data)):
        xyz_u = np.concatenate(
            (xyz_u, xyz[ij][mocap_ranges[selected_data[ij]], :]), axis=0
        )
        rpy_u = np.concatenate(
            (rpy_u, rpy[ij][mocap_ranges[selected_data[ij]], :]), axis=0
        )
        dxyz_u = np.concatenate(
            (dxyz_u, dxyz[ij][mocap_ranges[selected_data[ij]], :]), axis=0
        )
        drpy_u = np.concatenate(
            (drpy_u, drpy[ij][mocap_ranges[selected_data[ij]], :]), axis=0
        )
        tau_mea_concat = np.concatenate((tau_mea_concat, tau_mea[ij]), axis=0)
tau_mea_vector = np.reshape(tau_mea_concat.T, (6 * len(tau_mea_concat)))

# Regressor matrix
Ntotal = len(xyz_u)
reg_mat = pu.create_R_matrix(
    Ntotal,
    xyz_u,
    dxyz_u,
    rpy_u,
    drpy_u,
)
param_base = ["k", "c", "n"]
axis = ["x", "y", "z"]
dof = ["t", "r"]
suspension_parameter = []
for d_ in dof:
    for a_ in axis:
        for b_ in param_base:
            suspension_parameter.append(b_ + a_ + "_" + d_)

# Parameter estimation
N_d = 6
count = 0
rmse = 1e8
var_init = 1000 * np.ones(3 * N_d)

while rmse > 5 * 1e-1 and count < 100:
    LM_solve = least_squares(
        pu.cost_function,
        var_init,
        method="lm",
        verbose=1,
        args=(reg_mat[:, 0 : 3 * N_d], tau_mea_vector),
    )
    tau_predict_vector = reg_mat[:, 0 : 3 * N_d] @ LM_solve.x
    rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector) ** 2))
    print("*" * 40)
    print("iteration: ", count, "rmse: ", rmse)
    print("solution: ", dict(zip(suspension_parameter, LM_solve.x)))
    print("initial guess: ", var_init)
    var_init = LM_solve.x + 20000 * np.random.randn(3 * N_d)
    count += 1
print(LM_solve.x)

######## validation
fig, ax = plt.subplots(N_d, 1)
tau_ylabel = ["fx (N)", "fy(N)", "fz(N)", "mx(N.m)", "my(N.m)", "mz(N.m)"]
for i in range(N_d):
    ax[i].plot(
        tau_mea_vector[i * Ntotal : (i + 1) * Ntotal],
        color="red",
    )
    ax[i].plot(
        tau_predict_vector[i * Ntotal : (i + 1) * Ntotal],
        color="blue",
    )
    ax[i].set_ylabel(tau_ylabel[i])
    ax[i].set_xlabel("time(ms)")
    ax[0].legend(["forceplate_measures", "suspension_predicted_values"])
plt.show()


######################################################################################
dir_path = dir_paths[0]
path_to_values = dir_path + "introspection_datavalues.csv"
path_to_names = dir_path + "introspection_datanames.csv"

# read values from csv files
t_res, f_res, joint_names, q_abs_res, q_rel_res = pu.get_q_arm(
    tiago_fb, path_to_values, path_to_names, f_cutoff=0.625
)
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
# encoder_range = range(10, t_res.shape[0] - 10)
encoder_range = range(2430, 3430)

# mocap_range = encoder_range
q_arm, dq_arm, ddq_arm = pu.calc_vel_acc(
    tiago_fb, q_rel_res, encoder_range, joint_names, f_res, f_cutoff=0.625
)
# [base_trans, base_rot] = marker_datas[0]["base2"]


shoulder1_pos = marker_meas[selected_data[0]]["shoulder2"][0][
    mocap_ranges[selected_data[0]], :
]
gripper3_pos = marker_meas[selected_data[0]]["gripper3"][0][
    mocap_ranges[selected_data[0]], :
]
base2_pos = marker_meas[selected_data[0]]["base2"][0][
    mocap_ranges[selected_data[0]], :
]


def lookup_compare(lookup_marker="shoulder"):
    if lookup_marker == "gripper":
        pee_est, peews_est = compute_estimated_pee(
            "gripper_tool_link",
            "base2",
            marker_footprint,
            Mmocap_fixedfootprint,
            Mwrist_gripper3,
            mocap_ranges[selected_data[0]],
            q_arm,
        )
        plot_compare_suspension(
            pee_est, peews_est, gripper3_pos, lookup_marker=lookup_marker
        )

    elif lookup_marker == "shoulder":
        pee_est, peews_est = compute_estimated_pee(
            "torso_lift_link",
            "base2",
            marker_footprint,
            Mmocap_fixedfootprint,
            Mtorso_shoulder,
            mocap_ranges[selected_data[0]],
            q_arm,
        )
        pee_est[:, 0] += 0.02
        peews_est[:, 0] += 0.02
        pee_est[:, 1] += 0.007
        peews_est[:, 1] += 0.007
        pee_est[:, 2] += 0.0035
        peews_est[:, 2] += 0.0035

        plot_compare_suspension(
            pee_est, peews_est, shoulder1_pos, lookup_marker=lookup_marker
        )
        for ii, xx in enumerate(["x", "y", "z"]):
            print(
                "rmse without --> with susp. along {} axis at shoulder in millimeter: ".format(
                    xx
                ),
                np.around(
                    1e3 * pu.rmse(shoulder1_pos[:, ii], pee_est[:, ii]), 2
                ),
                " --> ",
                np.around(
                    1e3 * pu.rmse(shoulder1_pos[:, ii], peews_est[:, ii]), 2
                ),
            )

    elif lookup_marker == "base":
        pee_est, peews_est = compute_estimated_pee(
            "base_link",
            "base2",
            marker_footprint,
            Mmocap_fixedfootprint,
            Mmarker_baselink.inverse(),
            mocap_ranges[selected_data[0]],
            q_arm,
        )
        pee_est[:, 0] += 0.00075
        peews_est[:, 0] += 0.00075
        pee_est[:, 1] += 0.001
        peews_est[:, 1] += 0.001
        pee_est[:, 2] += 0.0001
        peews_est[:, 2] += 0.0001
        plot_compare_suspension(
            pee_est, peews_est, base2_pos, lookup_marker=lookup_marker
        )
        for ii, xx in enumerate(["x", "y", "z"]):
            print(
                "rmse without --> with susp. along {} axis at base in millimeter: ".format(
                    xx
                ),
                np.around(1e3 * pu.rmse(base2_pos[:, ii], pee_est[:, ii]), 2),
                " --> ",
                np.around(
                    1e3 * pu.rmse(base2_pos[:, ii], peews_est[:, ii]), 2
                ),
            )

for lm in ['base', 'shoulder', 'gripper']:
    lookup_compare(lm)
# # concatenate with floating base kinematic entities
# q_arm[:, 0:3] = xyz_u[2463:4863, :]
# q_arm[:, 3:7] = quat_u[2463:4863, :]
# dq_arm[:, 0:3] = dxyz_u[2463:4863, :]
# dq_arm[:, 3:6] = drpy_u[2463:4863, :]
# ddq_arm[:, 0:3] = ddxyz_u[2463:4863, :]
# ddq_arm[:, 3:6] = ddrpy_u[2463:4863, :]

# # modify baselink mass

# init_root_inertia = model.inertias[1].copy()
# delta_base_mass = np.mean(-force[:, 2]) / 9.81 - total_mass
# model.inertias[1].mass += -8


# # build basic regressor
# W = build_regressor_basic(tiago_fb, q_arm, dq_arm, ddq_arm, params_settings)
# print(W.shape, len(standard_parameter))

# # remove zero columns
# idx_e, active_parameter = get_index_eliminate(W, standard_parameter, tol_e=0.001)
# W_e = build_regressor_reduced(W, idx_e)
# print(W_e.shape)

# calculate joint torque by inverse dynamic

# tau_id = []
# for i_ in range(q_arm.shape[0]):
#     tau_id.append(
#         pin.rnea(model, data, q_arm[i_, :], dq_arm[i_, :], ddq_arm[i_, :])
#     )
# tau_id = np.array(tau_id)
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


# fig_, ax_ = plt.subplots(6, 1)
# invdyn_tau = tau_id[:, 0:6]
# invdyn_tau_in_fpframe = np.zeros_like(invdyn_tau)
# for jj in range(len(invdyn_tau)):
#     invdyn_tau_in_fpframe[jj, :] = np.dot(Mforceplate_baselink.action, invdyn_tau[jj, :])
# f = [force[2463:4863, 0], force[2463:4863, 1], -force[2463:4863, 2]]
# m = [moment[2463:4863, 0], moment[2463:4863, 1], moment[2463:4863, 2]]

# f_tick = ["fx", "fy", "fz"]
# m_tick = ["mx", "my", "mz"]

# for j_ in range(3):
#     ax_[j_].plot(f[j_], label="measured force")
#     ax_[j_].plot(
#         np.array(invdyn_tau_in_fpframe)[:, j_], label="inverse dyn force"
#     )
#     ax_[j_].set_ylabel(f_tick[j_])
#     ax_[j_].legend(loc="upper left")
#     ax_[3 + j_].plot(m[j_], label="measured force")
#     ax_[3 + j_].plot(
#         np.array(invdyn_tau_in_fpframe)[:, 3 + j_], label="inverse dyn moment"
#     )
#     ax_[3 + j_].set_ylabel(m_tick[j_])
#     ax_[3 + j_].legend(loc="upper left")


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
# mocap_range = range(3000, 4000)
# mocap_range_ext = range(mocap_range[0] - 10, mocap_range[-1] + 1 + 10)
# NbSample = len(mocap_range)

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
# t_res, f_res, joint_names, q_abs_res, q_rel_res = pu.get_q_arm(
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
# t_sample = t_res[mocap_range]
# q_arm, dq_arm, ddq_arm = pu.calc_vel_acc(
#     robot, q_abs_res, mocap_range, joint_names, f_res, f_cutoff
# )
# (
#     q_marker,
#     dq_marker,
#     ddq_marker,
#     rpy_marker,
#     Mmarker0_mocap,
#     q_marker_ext,
# ) = pu.calc_fb_vel_acc(posXYZ_res, quatXYZW_res, mocap_range, f_res)
# (
#     q_markerBase,
#     dq_markerBase,
#     ddq_markerBase,
#     rpy_markerBase,
#     Mmarker0_mocapBase,
#     q_marker_extBase,
# ) = pu.calc_fb_vel_acc(posXYZ_base, quatXYZW_base, mocap_range, f_res)
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

# print("mocap_range: ", mocap_range)


# # plot dynamics
# fig, ax = plt.subplots(6, 1)
# tau_ylabel = ["fx (N)", "fy(N)", "fz(N)", "mx(N.m)", "my(N.m)", "mz(N.m)"]
# for i in range(6):
#     ax[i].plot(
#         np.arange(len(mocap_range)),
#         tau_mea_vector[i * len(mocap_range) : (i + 1) * len(mocap_range)],
#         color="red",
#     )
#     ax[i].plot(
#         np.arange(len(mocap_range)),
#         tau_predict_vector[
#             i * len(mocap_range) : (i + 1) * len(mocap_range)
#         ],
#         color="blue",
#     )
#     ax[i].set_ylabel(tau_ylabel[i])
#     ax[i].set_xlabel("time(ms)")
#     ax[i].bar(
#         np.arange(len(mocap_range)),
#         error_tau[i * len(mocap_range) : (i + 1) * len(mocap_range)],
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
