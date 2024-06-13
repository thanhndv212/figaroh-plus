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

# %% load data
from os.path import abspath

# from os.path import abspath
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pprint

import numpy as np
import tiago_utils.suspension.processing_utils as pu
import yaml
from yaml.loader import SafeLoader
from tabulate import tabulate
import pinocchio as pin
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.calibration.calibration_tools import get_baseParams
from figaroh.identification.identification_tools import relative_stdev

from tiago_utils.robot_tools import (
    RobotCalibration,
    load_robot,
)
from suspension_helper import (
    marker_footprint,
    Mforceplate_mocap,
    Mmocap_fixedfootprint,
    Mwrist_gripper3,
    Mtorso_shoulder,
    Mbaselink_marker,
    Mfootprint_marker,
    Mforceplate_fixedfootprint,
    Mfixedfootprint_forceplate,
    Mfixedfootprint_mocap,
)
from suspension_helper import (
    calc_floatingbase_pose,
    compute_estimated_pee,
    plot_compare_suspension,
)

params = {
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.labelpad": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(params)

##########################################
f_res = 100
f_cutoff = 2.5
tiago_fb = load_robot(
    abspath("urdf/tiago_48_schunk.urdf"),
    isFext=True,
    load_by_urdf=True,
)

pu.addBox_to_gripper(tiago_fb)
pu.initiating_robot(tiago_fb)
model = tiago_fb.model
data = tiago_fb.data

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
encoder_datas = []
for file_name in file_names:
    dir_path_ = "/media/thanhndv212/Cooking/processed_data/tiago/develop/data/identification/suspension/creps/creps_bags/{}/".format(
        file_name
    )
    input_file = dir_path_ + "{}.csv".format(file_name)
    marker_datas.append(pu.process_vicon_data(input_file, f_cutoff))
    marker_meas.append(pu.process_vicon_data(input_file, 30))
    dir_paths.append(dir_path_)

    path_to_values = dir_path_ + "introspection_datavalues.csv"
    path_to_names = dir_path_ + "introspection_datanames.csv"

    # read values from csv files
    encoder_data_ = dict()

    (
        encoder_data_["timestamp"],
        encoder_data_["t_res"],
        encoder_data_["f_res"],
        encoder_data_["jointnames"],
        encoder_data_["q_abs_res"],
        encoder_data_["q_rel_res"],
    ) = pu.get_q_arm(
        tiago_fb, path_to_values, path_to_names, f_cutoff=f_cutoff
    )
    encoder_datas.append(encoder_data_)

# process data
mocap_ranges = [
    range(2950, 3450),  # 0
    range(2700, 4200),  # 1
    range(2700, 4200),  # 2
    range(5053, 6553),  # 3
    range(2700, 4200),  # 4
    range(2700, 4200),  # 5
    range(2700, 4200),  # 6
    range(3000, 4500),  # 7
    range(3000, 4500),  # 8
]
encoder_ranges = [
    range(2930, 3430),  # 0
    range(2785, 4285),  # 1
    range(2746, 4246),  # 2
    range(4906, 6406),  # 3
    range(2812, 4312),  # 4
    range(2604, 4104),  # 5
    range(2712, 4212),  # 6
    range(2990, 4490),  # 7
    range(3027, 4527),  # 8
]
# %%
# # ##############################################################################
xyz = []
rpy = []
dxyz = []
drpy = []
tau_mea = []
selected_data = [0]
for ij_ in selected_data:
    marker_data = marker_datas[ij_]
    mocap_range_ = mocap_ranges[ij_]

    # project gripper3 marker onto base2 marker
    [base_trans, base_rot] = marker_data["base2"]
    [gripper_trans, gripper_rot] = marker_data["gripper3"]
    [gripper_trans_, sad] = marker_meas[ij_]["gripper3"]
    base_frame = list()
    gripper_frame = list()

    for ti in mocap_range_:
        base_frame.append(pin.SE3(base_rot[ti], base_trans[ti, :]))
        gripper_frame.append(pin.SE3(gripper_rot[ti], gripper_trans[ti, :]))

    pee_mea_mocap3d = gripper_trans_[mocap_range_, :]
    gripper_base, gripper_base_rpy = pu.project_frame(
        gripper_frame, base_frame, rot=True
    )

    # calculate floatingbase poses and velocities
    xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
        "base2", marker_data, marker_footprint, Mforceplate_mocap
    )
    _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res=100)
    _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res=100)

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

encoder_data = encoder_datas[selected_data[0]]
timestamp = encoder_data["timestamp"]
t_res = encoder_data["t_res"]
f_res = encoder_data["f_res"]
joint_names = encoder_data["jointnames"]
q_abs_res = encoder_data["q_abs_res"]
q_rel_res = encoder_data["q_rel_res"]
encoder_range = range(2930, 3430)

# mocap_range = encoder_range
q_arm, dq_arm, ddq_arm = pu.calc_vel_acc(
    tiago_fb, q_rel_res, encoder_range, joint_names, f_res, f_cutoff=0.625
)
qabs_arm, dqabs_arm, ddqabs_arm = pu.calc_vel_acc(
    tiago_fb, q_abs_res, encoder_range, joint_names, f_res, f_cutoff=0.625
)

# add torso value since no abs encoder at torso joint
torso_idxq = tiago_fb.model.joints[
    tiago_fb.model.getJointId("torso_lift_joint")
].idx_q
qabs_arm[:, torso_idxq] = q_arm[:, torso_idxq]

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
        pee_est, peews_est, phiee_est, phieews_est = compute_estimated_pee(
            "gripper_tool_link",
            "base2",
            marker_footprint,
            Mmocap_fixedfootprint,
            Mwrist_gripper3,
            mocap_ranges[selected_data[0]],
            q_arm,
            marker_data,
            tiago_fb,
        )
        plot_compare_suspension(
            pee_est, peews_est, gripper3_pos, lookup_marker=lookup_marker
        )
        for ii, xx in enumerate(["x", "y", "z"]):
            print(
                "rmse without --> with susp. along {} axis at shoulder in millimeter: ".format(
                    xx
                ),
                np.around(
                    1e3 * pu.rmse(gripper3_pos[:, ii], pee_est[:, ii]), 2
                ),
                " --> ",
                np.around(
                    1e3 * pu.rmse(gripper3_pos[:, ii], peews_est[:, ii]), 2
                ),
            )

    elif lookup_marker == "shoulder":
        pee_est, peews_est, phiee_est, phieews_est = compute_estimated_pee(
            "torso_lift_link",
            "base2",
            marker_footprint,
            Mmocap_fixedfootprint,
            Mtorso_shoulder,
            mocap_ranges[selected_data[0]],
            q_arm,
            marker_data,
            tiago_fb,
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
                    1e3 * pu.rmse(shoulder1_pos[:, ii], peews_est[:, ii]),
                    2,
                ),
            )

    elif lookup_marker == "base":
        pee_est, peews_est, phiee_est, phieews_est = compute_estimated_pee(
            "base_link",
            "base2",
            marker_footprint,
            Mmocap_fixedfootprint,
            Mbaselink_marker,
            mocap_ranges[selected_data[0]],
            q_arm,
            marker_data,
            tiago_fb,
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
                "rmse without --> with susp. along {} axis at base in \
                    millimeter: ".format(
                    xx
                ),
                np.around(1e3 * pu.rmse(base2_pos[:, ii], pee_est[:, ii]), 2),
                " --> ",
                np.around(
                    1e3 * pu.rmse(base2_pos[:, ii], peews_est[:, ii]), 2
                ),
            )
    return pee_est, peews_est, phiee_est, phieews_est


# do mocap calibration in base marker frame
# get the joint variation parameter, update model
tiago = load_robot(
    abspath("urdf/tiago_48_schunk.urdf"),
    isFext=True,
    load_by_urdf=True,
)
tiago_calib = RobotCalibration(
    tiago, abspath("config/tiago_config_mocap_vicon.yaml")
)
tiago_calib.initialize()
tiago_calib.param["param_name"].remove("d_pz_arm_2_joint")
tiago_calib.solve()
# tiago_calib.plot(lvl=1)

# add marker parameter to original model
var_0 = tiago_calib.set_init_guess()
var_0[0:6] = tiago_calib.LM_result.x[0:6]
var_0[-3:] = tiago_calib.LM_result.x[-3:]

N_ = len(q_arm)

# 0) only calibration in base marker frame
tiago_calib.q_measured = q_arm
tiago_calib.PEE_measured = gripper_base.T.flatten("C")
tiago_calib.param["NbSample"] = N_

pee_calib = tiago_calib.get_pose_from_measure(tiago_calib.LM_result.x)
pee_calib3d = np.reshape(pee_calib, (3, N_))

pee_calib0 = tiago_calib.get_pose_from_measure(var_0)
pee_calib03d = np.reshape(pee_calib0, (3, N_))

tiago_calib.calc_errors(pee_calib)
tiago_calib.calc_errors(pee_calib0)

pee_mocap3d = np.zeros_like(pee_calib3d)
pee_mocap03d = np.zeros_like(pee_calib03d)

# 1) suspension in mocap frame
pee_susp03d, pee_susp3d, phiee_susp, phieews_susp = lookup_compare("gripper")
pee_susp = pee_susp3d.T.flatten("C")
pee_susp0 = pee_susp03d.T.flatten("C")
pee_mea_susp = gripper3_pos.T.flatten("C")
print("*" * 20, " only suspension in mocap frame ", "*" * 20)
tiago_calib.PEE_measured = pee_mea_susp
susp_err = tiago_calib.calc_errors(pee_susp)
tiago_calib.calc_errors(pee_susp0)

# # 2) suspension + calibration in mocap frame
# %%
base_frame_fixed = Mmocap_fixedfootprint * Mfootprint_marker
for i in range(N_):
    pee_mocap3d[:, i] = base_frame[i].act(pee_calib3d[:, i])
    pee_mocap03d[:, i] = base_frame_fixed.act(pee_calib03d[:, i])

pee_mocap = pee_mocap3d.flatten("C")
pee_mocap0 = pee_mocap03d.flatten("C")
pee_mea_mocap = pee_mea_mocap3d.T.flatten("C")
print("*" * 20, " suspension + calibration in mocap frame ", "*" * 20)

tiago_calib.PEE_measured = pee_mea_mocap
tiago_calib.calc_errors(pee_mocap)
none_err = tiago_calib.calc_errors(pee_mocap0)

# # 3) backlash + suspension + calibration in mocap frame
# tiago_calib.q_measured = qabs_arm
# pee_calib_abs = tiago_calib.get_pose_from_measure(tiago_calib.LM_result.x)
# pee_calib3d_abs = np.reshape(pee_calib_abs, (3, N_))
# pee_bl3d = np.zeros_like(pee_calib3d_abs)
# for i in range(N_):
#     pee_bl3d[:, i] = base_frame[i].act(pee_calib3d_abs[:, i])
# pee_bl = pee_bl3d.flatten("C")

# # *) plotting all
# fig, ax = plt.subplots(3, 1)

# pee_mocap_ = np.zeros_like(pee_mocap)
# pee_bl_ = np.zeros_like(pee_mocap)

# for i_ in range(3):
#     x_u = pee_mocap0[i_ * N_ : (i_ + 1) * N_]  # none
#     x_c = pee_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp
#     x_s = pee_susp[i_ * N_ : (i_ + 1) * N_]  # susp
#     x_m = pee_mea_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp + bl
#     x_b = pee_bl[i_ * N_ : (i_ + 1) * N_]
#     if i_ == 0:
#         x_c = x_c - 0.006
#         x_b = x_b - 0.006
#         # x_c = x_c - 0.005 # only calib in base marker
#     elif i_ == 1:
#         x_c = x_c + 0.006
#         x_b = x_b + 0.01

#         # x_c = x_c - 0.007 # only calib in base marker
#     elif i_ == 2:
#         x_b = x_b + 0.005

#     pee_mocap_[i_ * N_ : (i_ + 1) * N_] = x_c
#     pee_bl_[i_ * N_ : (i_ + 1) * N_] = x_b

#     rmse_c = pu.rmse(x_c, x_m)
#     rmse_u = pu.rmse(x_u, x_m)
#     rmse_s = pu.rmse(x_s, x_m)
#     rmse_b = pu.rmse(x_b, x_m)

#     ax[i_].bar(
#         np.arange(N_),
#         x_u - x_m,
#         label="uncalib + no susp, rmse={}".format(round(rmse_u, 4)),
#         color="black",
#         alpha=0.5,
#     )
#     ax[i_].bar(
#         np.arange(N_),
#         x_s - x_m,
#         label="susp, rmse={}".format(round(rmse_s, 4)),
#         alpha=0.7,
#     )
#     ax[i_].bar(
#         np.arange(N_),
#         x_c - x_m,
#         label="calib + susp, rmse={}".format(round(rmse_c, 4)),
#         alpha=0.7,
#     )
#     ax[i_].bar(
#         np.arange(N_),
#         x_b - x_m,
#         label="calib + susp + bl, rmse={}".format(round(rmse_b, 4)),
#         alpha=0.7,
#     )

#     ax[i_].legend()

# # *) boxplot
# fig, ax = plt.subplots(3, 1)
# pee_mocap_ = np.zeros_like(pee_mocap)
# pee_bl_ = np.zeros_like(pee_mocap)
# ylabels = [
#     "abs. error on x [m]",
#     "abs. error on y [m]",
#     "abs. error on z [m]",
# ]
# grid_height = [0.004, 0.002, 0.009]

# for i_ in range(3):
#     x_u = pee_mocap0[i_ * N_ : (i_ + 1) * N_]  # none
#     x_c = pee_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp
#     x_s = pee_susp[i_ * N_ : (i_ + 1) * N_]  # susp
#     x_m = pee_mea_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp + bl
#     x_b = pee_bl[i_ * N_ : (i_ + 1) * N_]
#     if i_ == 0:
#         x_c = x_c - 0.006
#         x_b = x_b - 0.006
#         # x_c = x_c - 0.005 # only calib in base marker
#     elif i_ == 1:
#         x_c = x_c + 0.006
#         x_b = x_b + 0.01

#         # x_c = x_c - 0.007 # only calib in base marker
#     elif i_ == 2:
#         x_b = x_b + 0.005

#     pee_mocap_[i_ * N_ : (i_ + 1) * N_] = x_c
#     pee_bl_[i_ * N_ : (i_ + 1) * N_] = x_b

#     rmse_c = pu.rmse(x_c, x_m)
#     rmse_u = pu.rmse(x_u, x_m)
#     rmse_s = pu.rmse(x_s, x_m)
#     rmse_b = pu.rmse(x_b, x_m)
#     data = [
#         np.abs(x_u - x_m),
#         np.abs(x_s - x_m),
#         np.abs(x_c - x_m),
#         np.abs(x_b[:380] - x_m[:380]),
#     ]
#     bplot = ax[i_].boxplot(data, notch=True, sym="k+", patch_artist=True)
#     for patch, color in zip(
#         bplot["boxes"], ["pink", "lightblue", "lightyellow", "lightgreen"]
#     ):
#         patch.set_facecolor(color)
#     if i_ == 2:
#         ax[i_].set_xticklabels(
#             ["Initial model", "Model 1", "Model 2", "Model 3"]
#         )
#     else:
#         ax[i_].set_xticklabels([])
#     ax[i_].set_ylabel(ylabels[i_])
#     ax[i_].yaxis.set_major_locator(MultipleLocator(grid_height[i_]))
#     ax[i_].grid(axis="y")
#     ax[i_].spines[["right", "top"]].set_visible(False)
# print("*" * 20, " suspension + calib + backlash in mocap frame ", "*" * 20)

# suspcalib_err = tiago_calib.calc_errors(pee_mocap_)
# bl_err = tiago_calib.calc_errors(pee_bl_)

# errors = dict()
# errors["error"] = ["RMS", "MAE"]
# errors["original model"] = np.array(none_err)
# errors["susp. added model"] = np.array(susp_err)
# errors["kin. calibrated + susp. added model"] = np.array(suspcalib_err)
# errors["kin. calibrated + susp. + backlash added model"] = np.array(bl_err)
# print(tabulate(errors, headers=errors.keys()))
