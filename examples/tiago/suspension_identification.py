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
from scipy.optimize import least_squares
import yaml
from yaml.loader import SafeLoader
from tabulate import tabulate
import pinocchio as pin
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.calibration.calibration_tools import get_baseParams

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


def plot_validation(file_name_, tau_mea_vector, tau_predict_vector, Ntotal):
    #  validation
    N_d = 6
    fig, ax = plt.subplots(N_d, 1)
    tau_ylabel = [
        "Fx [N]",
        "Fy [N]",
        "Fz [N]",
        "Mx [N.m]",
        "My [N.m]",
        "Mz [N.m]",
    ]
    for i in range(N_d):
        ax[i].plot(
            tau_mea_vector[i * Ntotal : (i + 1) * Ntotal],
            color="red",
        )
        ax[i].plot(
            tau_predict_vector[i * Ntotal : (i + 1) * Ntotal],
            color="blue",
            linestyle="--",
        )
        ax[i].set_ylabel(tau_ylabel[i])
        ax[i].spines[["right", "top"]].set_visible(False)
        ax[0].legend(["measured", "predicted"], loc="lower right")
        if i != int(N_d - 1):
            ax[i].set_xticklabels([])
        else:
            ax[i].set_xlabel("sample")
    fig.align_ylabels(ax)
    fig.suptitle(file_name_)


# %% process data

from figaroh.identification.identification_tools import relative_stdev


def new_suspension_solve(
    file_name_: str,
    xyz_u: np.ndarray,
    dxyz_u: np.ndarray,
    rpy_u: np.ndarray,
    drpy_u: np.ndarray,
    tau_mea_concat: np.ndarray,
):
    """Assume each wheel could be modeled as a 3D spring-damper.
    The displacement of wheel i with ri is the constant position vector to
    fixed reference frame of robot, is calculated by: delta_ri = R*ri+t-ri,
    where [t theta] is floating base 6D pose. Since theta is small,
    R = I + exp([theta]_X).
    => delta_ri = t + [theta]+X * ri.
    Suspension model:
    F = sum(k*delta_ri) + sum(c*dot(delta_ri))
    M = sum(ri X Fi)

    Args:
        file_name_ (str): imported data file
        xyz_u (np.ndarray): floating base linear position
        dxyz_u (np.ndarray): floating base linear velocity
        rpy_u (np.ndarray): floating base angular position
        drpy_u (np.ndarray): floating base angular velocity
        tau_mea_concat (np.ndarray): measured ground reaction wrench

    Returns:
        x: solution of parameters
        tau_est: predicted values of wrench
    """
    # Regressor matrix
    Ntotal = len(xyz_u)

    # measure output
    tau_mea_vector = np.reshape(tau_mea_concat.T, (6 * len(tau_mea_concat)))
    # Parameter estimation
    # n_wheel = len(wheel_names)
    count = 0
    rmse = 1e8
    var_init = 1000 * np.ones(len(suspension_parameter))
    # var_init[-12:-6] = np.array(
    #     [4000, 12500, 37700, 200, 21500, 14400]
    # )  # torsional
    # var_init[-6:] = np.array([0, 0, 725, 0, 0, 0])  # offset

    lower_bound = (
        6 * [0, 0, 0, 0, 0, 0]
        # + 4 * [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        + [0] * 6
        + [-np.inf] * 6
    )
    while rmse > 5 * 1e-1 and count < 1:
        LM_solve = least_squares(
            pu.cost_function_multiwheels,
            var_init,
            # method="trf",
            bounds=(lower_bound, np.inf),
            verbose=1,
            args=(
                Ntotal,
                wheels,
                xyz_u,
                dxyz_u,
                rpy_u,
                drpy_u,
                tau_mea_vector,
            ),
        )
        tau_predict_vector = pu.multi_linear_models(
            LM_solve.x, Ntotal, wheels, xyz_u, dxyz_u, rpy_u, drpy_u
        )
        rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector) ** 2))
        # print("*" * 40)
        # print("iteration: ", count, "rmse: ", rmse)
        # susp_param = dict(zip(suspension_parameter, np.array(list(LM_solve.x), dtype=float64)))
        var_init = LM_solve.x + 1000 * np.random.randn(len(var_init))
        print(
            "Iteration {}".format(count),
            dict(zip(suspension_parameter, np.array(list(LM_solve.x)))),
        )
        count += 1
    plot_validation(file_name_, tau_mea_vector, tau_predict_vector, Ntotal)
    tau_predict_ = np.reshape(tau_predict_vector, (6, Ntotal))
    return np.array(list(LM_solve.x), dtype=np.float64), tau_predict_.T


# wheels
wheel_names = [
    "wheel_left_joint",
    "wheel_right_joint",
    "caster_back_left_2_joint",
    "caster_back_right_2_joint",
    "caster_front_left_2_joint",
    "caster_front_right_2_joint",
]
wheels = dict()
for wn in wheel_names:
    wheel_ri = data.oMf[model.getFrameId(wn)]
    wheels[wn] = wheel_ri.translation
    # wheels[wn][2] = 0

# comb = []
# for i1 in [-1, 1]:
#     for i2 in [-1, 1]:
#         for i3 in [-1, 1]:
#             for i4 in [-1, 1]:
#                 for i5 in [-1, 1]:
#                     for i6 in [-1, 1]:
#                         comb.append([i1, i2, i3, i4, i5, i6])
# comb = [[-1, 1, 1, -1, 1, 1]]
# comb = [[-1, 1, 1, 1, 1, 1]]

comb = [
    [-1, 1, 1, -1, 1, 1],
    # [-1, -1, -1, -1, -1, -1],

    # [-1, -1, -1, 1, 1, 1],

    # [-1, -1, 1, -1, -1, 1],

    # [-1, -1, 1, 1, 1, 1],

    # [-1, 1, -1, -1, -1, -1],

    # [-1, 1, -1, 1, 1, 1],

    # [-1, 1, 1, -1, -1, -1],

    # [-1, 1, 1, 1, 1, 1],

    # [1, -1, -1, -1, -1, -1],

    # [1, -1, -1, 1, 1, 1],

    # [1, -1, 1, -1, -1, -1],

    # [1, -1, 1, 1, 1, 1],

    # [1, 1, -1, -1, -1, -1],

    # [1, 1, -1, 1, 1, 1],

    # [1, 1, 1, -1, -1, -1],

    # [1, 1, 1, 1, 1, 1],
]
susp_list = []
suspbase_list = []
for const in comb:
    print("*******{}********".format(const))
    pee_mea_mocap3d = None
    gripper_base = None
    ref_frame = "footprint"

    param_base = ["k", "c"]
    axis = ["x", "y", "z"]
    # dof = ["t", "r"]
    suspension_parameter = []

    for wn in wheels.keys():
        for b_ in param_base:
            for a_ in axis:
                suspension_parameter.append(b_ + a_ + "_" + wn)
    # suspension_parameter += ['kt_x', 'kt_y', 'kt_z', 'ct_x', 'ct_y', 'ct_z']

    suspension_parameter += ["kr_x", "kr_y", "kr_z", "cr_x", "cr_y", "cr_z"]

    suspension_parameter += ["fo_x", "fo_y", "fo_z", "mo_x", "mo_y", "mo_z"]
    xyz = []
    rpy = []
    dxyz = []
    drpy = []
    tau_mea = []
    selected_data = [
        # 0,  # "tiago_around_x_vicon_1634",  # 0
        # 1,  # "tiago_around_y_vicon_1638",  # 1
        # 2,  # "tiago_around_z_vicon_1622",  # 2
        # 3,  # "tiago_x_fold_vicon_1632",  # 3
        # 4,  # "tiago_xyz_mirror_vicon_1630",  # 4
        # 5,  # "tiago_xyz_mirror_vicon_1642",  # 5
        # 6,  # "tiago_xyz_vicon_1628",  # 6
        7,  # "tiago_xyz_vicon_1640",  # 7
        # 8,  # "tiago_y_fold_vicon_1636",  # 8
    ]
    # load data

    for ij_ in selected_data:
        marker_data = marker_datas[ij_]
        mocap_range_ = mocap_ranges[ij_]

        if ref_frame == "footprint":
            # calculate floatingbase poses and velocities
            xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
                "base2", marker_data, marker_footprint, Mfixedfootprint_mocap
            )
            _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res)
            _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res)

            # measured force plate data
            force = np.zeros_like(marker_data["force"])
            force[:, 0] = const[0] * marker_data["force"][:, 0]
            force[:, 1] = const[1] * marker_data["force"][:, 1]
            force[:, 2] = const[2] * marker_data["force"][:, 2]

            moment = np.zeros_like(marker_data["moment"])
            moment[:, 0] = const[3] * marker_data["moment"][:, 0]
            moment[:, 1] = const[4] * marker_data["moment"][:, 1]
            moment[:, 2] = const[5] * marker_data["moment"][:, 2]

            tau_ = np.concatenate(
                (force[mocap_range_, :], moment[mocap_range_, :]), axis=1
            )

            # transform measured force and moment to robot fixed ref frame
            for jj in range(len(tau_)):
                tau_[jj, :] = np.dot(
                    Mfixedfootprint_forceplate.action, tau_[jj, :]
                )
        elif ref_frame == "forceplate":
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

    ## LS_susp
    pos_res = xyz[0][mocap_ranges[selected_data[0]], :]
    pos_vel = dxyz[0][mocap_ranges[selected_data[0]], :]
    rpy_res = rpy[0][mocap_ranges[selected_data[0]], :]
    ang_vel = drpy[0][mocap_ranges[selected_data[0]], :]
    tau_mea_s = tau_mea[0]

    var = len(suspension_parameter) * [0]
    NbSample = len(pos_res)

    wheels = dict()
    for wn in wheel_names:
        wheel_ri = data.oMf[model.getFrameId(wn)]
        wheels[wn] = wheel_ri.translation

    M = pu.create_linmodel_regmatrix(
        var, NbSample, wheels, pos_res, pos_vel, rpy_res, ang_vel
    )

    MB, base_supsParam, _ = get_baseParams(M, suspension_parameter)
    tau_mea_vec = tau_mea_s.T.flatten("C")
    sol = np.dot(np.linalg.pinv(MB), tau_mea_vec)

    rel_stddev = relative_stdev(MB, sol, tau_mea_vec)
    plot_validation("base param {}".format(const), tau_mea_vec, np.dot(MB, sol), NbSample)
    susp_baseparam = dict(zip(base_supsParam, np.around(sol, 2)))
    pprint.pprint(susp_baseparam)
    pprint.pprint(dict(zip(base_supsParam, np.around(rel_stddev, 2))))

    # # new_susp
    # xyz_s = []
    # dxyz_s = []
    # rpy_s = []
    # drpy_s = []
    # tau_s = []
    # tau_pred_s = []
    # for ij in range(len(selected_data)):
    #     xyz_s.append(xyz[ij][mocap_ranges[selected_data[ij]], :])
    #     dxyz_s.append(dxyz[ij][mocap_ranges[selected_data[ij]], :])
    #     rpy_s.append(rpy[ij][mocap_ranges[selected_data[ij]], :])
    #     drpy_s.append(drpy[ij][mocap_ranges[selected_data[ij]], :])
    #     tau_s.append(tau_mea[ij])
    # sols = []
    # for i in range(len(selected_data)):
    #     sol_, tau_pred_ = new_suspension_solve(
    #         # str(selected_data[i]) + file_names[selected_data[i]],
    #         "full param {}".format(const),
    #         xyz_s[i],
    #         dxyz_s[i],
    #         rpy_s[i],
    #         drpy_s[i],
    #         tau_s[i],
    #     )

    #     sols.append(sol_)
    #     tau_pred_s.append(tau_pred_)

    # susp_param = dict(zip(suspension_parameter, np.around(sols[0], 2)))
    # susp_list.append(susp_param)
    suspbase_list.append(susp_baseparam)

# %% new_susp
def new_suspension_solve(
    file_name_: str,
    xyz_u: np.ndarray,
    dxyz_u: np.ndarray,
    rpy_u: np.ndarray,
    drpy_u: np.ndarray,
    tau_mea_concat: np.ndarray,
):
    """Assume each wheel could be modeled as a 3D spring-damper.
    The displacement of wheel i with ri is the constant position vector to
    fixed reference frame of robot, is calculated by: delta_ri = R*ri+t-ri,
    where [t theta] is floating base 6D pose. Since theta is small,
    R = I + exp([theta]_X).
    => delta_ri = t + [theta]+X * ri.
    Suspension model:
    F = sum(k*delta_ri) + sum(c*dot(delta_ri))
    M = sum(ri X Fi)

    Args:
        file_name_ (str): imported data file
        xyz_u (np.ndarray): floating base linear position
        dxyz_u (np.ndarray): floating base linear velocity
        rpy_u (np.ndarray): floating base angular position
        drpy_u (np.ndarray): floating base angular velocity
        tau_mea_concat (np.ndarray): measured ground reaction wrench

    Returns:
        x: solution of parameters
        tau_est: predicted values of wrench
    """
    # Regressor matrix
    Ntotal = len(xyz_u)

    # measure output
    tau_mea_vector = np.reshape(tau_mea_concat.T, (6 * len(tau_mea_concat)))
    # Parameter estimation
    # n_wheel = len(wheel_names)
    count = 0
    rmse = 1e8
    var_init = 1000 * np.ones(len(suspension_parameter))
    var_init[-12:-6] = np.array(
        [4000, 12500, 37700, 200, 21500, 14400]
    )  # torsional
    var_init[-6:] = np.array([0, 0, 725, 0, 0, 0])  # offset

    lower_bound = 6 * [0, 0, 0, -np.inf, -np.inf, -np.inf] + [-np.inf] * 12
    while rmse > 5 * 1e-1 and count < 1:
        LM_solve = least_squares(
            pu.cost_function_multiwheels,
            var_init,
            # method="trf",
            # bounds=(lower_bound, np.inf),
            verbose=1,
            args=(
                Ntotal,
                wheels,
                xyz_u,
                dxyz_u,
                rpy_u,
                drpy_u,
                tau_mea_vector,
            ),
        )
        tau_predict_vector = pu.multi_linear_models(
            LM_solve.x, Ntotal, wheels, xyz_u, dxyz_u, rpy_u, drpy_u
        )
        rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector) ** 2))
        # print("*" * 40)
        # print("iteration: ", count, "rmse: ", rmse)
        # susp_param = dict(zip(suspension_parameter, np.array(list(LM_solve.x), dtype=float64)))
        var_init = LM_solve.x + 1000 * np.random.randn(len(var_init))
        print(
            "Iteration {}".format(count),
            dict(zip(suspension_parameter, np.array(list(LM_solve.x)))),
        )
        count += 1
    plot_validation(file_name_, tau_mea_vector, tau_predict_vector, Ntotal)
    tau_predict_ = np.reshape(tau_predict_vector, (6, Ntotal))
    return np.array(list(LM_solve.x), dtype=np.float64), tau_predict_.T


# xyz_s = []
# dxyz_s = []
# rpy_s = []
# drpy_s = []
# tau_s = []
# tau_pred_s = []
# for ij in range(len(selected_data)):
#     xyz_s.append(xyz[ij][mocap_ranges[selected_data[ij]], :])
#     dxyz_s.append(dxyz[ij][mocap_ranges[selected_data[ij]], :])
#     rpy_s.append(rpy[ij][mocap_ranges[selected_data[ij]], :])
#     drpy_s.append(drpy[ij][mocap_ranges[selected_data[ij]], :])
#     tau_s.append(tau_mea[ij])
# sols = []
# for i in range(len(selected_data)):
#     sol_, tau_pred_ = new_suspension_solve(
#         str(selected_data[i]) + file_names[selected_data[i]],
#         xyz_s[i],
#         dxyz_s[i],
#         rpy_s[i],
#         drpy_s[i],
#         tau_s[i],
#     )

#     sols.append(sol_)
#     tau_pred_s.append(tau_pred_)

# susp_param = dict(zip(suspension_parameter, np.around(sols[0], 2)))


# %%
def LS_susp_identification():
    xyz = []
    rpy = []
    dxyz = []
    drpy = []
    tau_mea = []
    selected_data = [
        # 0,  # good fit on mx, my
        # 1,  # bad
        # 2,  # very good fit on fx
        # 3,  # good fit on mx, my
        # 4,  # good fit on fz
        # 5,  # good fit on fx
        # 6,  # bad
        7,  # good fit on fz
        # 8,  # my, and somehow mz good
    ]
    for ij_ in selected_data:
        marker_data = marker_datas[ij_]
        mocap_range_ = mocap_ranges[ij_]

        if ref_frame == "footprint":
            # calculate floatingbase poses and velocities
            xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
                "base2", marker_data, marker_footprint, Mfixedfootprint_mocap
            )
            _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res)
            _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res)

            # measured force plate data
            force = marker_data["force"]
            force[:, 0] = -force[:, 0]
            force[:, 1] = -force[:, 1]
            # force[:, 2] = -force[:, 2]

            moment = marker_data["moment"]
            moment[:, 0] = -moment[:, 0]
            moment[:, 1] = -moment[:, 1]
            # moment[:, 2] = -moment[:, 2]

            tau_ = np.concatenate(
                (force[mocap_range_, :], moment[mocap_range_, :]), axis=1
            )

            # transform measured force and moment to robot fixed ref frame
            for jj in range(len(tau_)):
                tau_[jj, :] = np.dot(
                    Mfixedfootprint_forceplate.action, tau_[jj, :]
                )
        elif ref_frame == "forceplate":
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
    pos_res = xyz[0][mocap_ranges[selected_data[0]], :]
    pos_vel = dxyz[0][mocap_ranges[selected_data[0]], :]
    rpy_res = rpy[0][mocap_ranges[selected_data[0]], :]
    ang_vel = drpy[0][mocap_ranges[selected_data[0]], :]
    tau_mea_s = tau_mea[0]

    var = 48 * [0]
    NbSample = len(pos_res)

    def skew(a):
        """Returns a skew-symmetric matrix from a vector."""
        return np.cross(np.identity(a.shape[0]), a)

    def create_linmodel_regmatrix(
        var, NbSample, wheels, pos_res, pos_vel, rpy_res, ang_vel
    ):
        M = np.zeros((6 * NbSample, len(var)))

        for i in range(NbSample):
            A_ = np.zeros((3, len(var)))
            B_ = np.zeros((3, len(var)))
            t = pos_res[i, :]
            t_dot = pos_vel[i, :]
            theta = rpy_res[i, :]
            theta_dot = ang_vel[i, :]
            # linear wheel spring-dampers
            for wi, key in enumerate(wheels.keys()):
                AF_i = np.zeros((3, 6))
                ri = wheels[key]
                AF_i[:, 0:3] = np.diag(t - np.matmul(skew(ri), theta))
                AF_i[:, 3:6] = np.diag(t_dot - np.matmul(skew(ri), theta_dot))
                A_[:, 6 * wi : 6 * (wi + 1)] = AF_i
                BF_i = np.matmul(skew(ri), AF_i)
                B_[:, 6 * wi : 6 * (wi + 1)] = BF_i

            # generalized torsion spring-damper
            B_[:, -12:-9] = np.diag(theta)
            B_[:, -9:-6] = np.diag(theta_dot)

            # offset nominal
            A_[:, -6:-3] = np.diag([1, 1, 1])
            B_[:, -3:] = np.diag([1, 1, 1])

            # update M
            M[0 * NbSample + i, :] = A_[0, :]  # fx
            M[1 * NbSample + i, :] = A_[1, :]  # fy
            M[2 * NbSample + i, :] = A_[2, :]  # fz
            M[3 * NbSample + i, :] = B_[0, :]  # mx
            M[4 * NbSample + i, :] = B_[1, :]  # my
            M[5 * NbSample + i, :] = B_[2, :]  # mz
        return M

    wheels = dict()
    for wn in wheel_names:
        # wheel_ri = Mforceplate_fixedfootprint * data.oMf[model.getFrameId(wn)]
        wheel_ri = data.oMf[model.getFrameId(wn)]
        wheels[wn] = wheel_ri.translation

    M = create_linmodel_regmatrix(
        var, NbSample, wheels, pos_res, pos_vel, rpy_res, ang_vel
    )

    from figaroh.calibration.calibration_tools import get_baseParams
    
    MB, base_supsParam, _ = get_baseParams(M, suspension_parameter)
    tau_mea_vec = tau_mea_s.T.flatten("C")
    sol = np.dot(np.linalg.pinv(MB), tau_mea_vec)

    def plot_validation(
        file_name_, tau_mea_vector, tau_predict_vector, Ntotal
    ):
        #  validation
        N_d = 6
        fig, ax = plt.subplots(N_d, 1)
        tau_ylabel = [
            "Fx [N]",
            "Fy [N]",
            "Fz [N]",
            "Mx [N.m]",
            "My [N.m]",
            "Mz [N.m]",
        ]
        for i in range(N_d):
            ax[i].plot(
                tau_mea_vector[i * Ntotal : (i + 1) * Ntotal],
                color="red",
            )
            ax[i].plot(
                tau_predict_vector[i * Ntotal : (i + 1) * Ntotal],
                color="blue",
                linestyle="--",
            )
            ax[i].set_ylabel(tau_ylabel[i])
            ax[i].spines[["right", "top"]].set_visible(False)
            ax[0].legend(["measured", "predicted"], loc="lower right")
            if i != int(N_d - 1):
                ax[i].set_xticklabels([])
            else:
                ax[i].set_xlabel("sample")
        fig.align_ylabels(ax)
        fig.suptitle(file_name_)

    plot_validation("base param", tau_mea_vec, np.dot(MB, sol), NbSample)
    return dict(zip(base_supsParam, sol))


def new_susp_identification():
    xyz = []
    rpy = []
    dxyz = []
    drpy = []
    tau_mea = []
    selected_data = [
        # 0,  # good fit on mx, my
        # 1,  # bad
        # 2,  # very good fit on fx
        # 3,  # good fit on mx, my
        # 4,  # good fit on fz
        # 5,  # good fit on fx
        # 6,  # bad
        7,  # good fit on fz
        # 8,  # my, and somehow mz good
    ]
    for ij_ in selected_data:
        marker_data = marker_datas[ij_]
        mocap_range_ = mocap_ranges[ij_]

        # # project gripper3 marker onto base2 marker
        # [base_trans, base_rot] = marker_data["base2"]
        # [gripper_trans, gripper_rot] = marker_data["gripper3"]
        # [gripper_trans_, sad] = marker_meas[ij_]["gripper3"]
        # base_frame = list()
        # gripper_frame = list()

        # for ti in mocap_range_:
        #     base_frame.append(pin.SE3(base_rot[ti], base_trans[ti, :]))
        #     gripper_frame.append(
        #         pin.SE3(gripper_rot[ti], gripper_trans[ti, :])
        #     )

        if ref_frame == "footprint":
            # calculate floatingbase poses and velocities
            xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
                "base2", marker_data, marker_footprint, Mfixedfootprint_mocap
            )
            _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res)
            _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res)

            # measured force plate data
            force = marker_data["force"]
            force[:, 0] = -force[:, 0]
            force[:, 1] = -force[:, 1]
            # force[:, 2] = - force[:, 2]

            moment = marker_data["moment"]
            moment[:, 0] = -moment[:, 0]
            moment[:, 1] = -moment[:, 1]
            # moment[:, 2] = - moment[:, 2]

            tau_ = np.concatenate(
                (force[mocap_range_, :], moment[mocap_range_, :]), axis=1
            )

            # transform measured force and moment to robot fixed ref frame
            for jj in range(len(tau_)):
                tau_[jj, :] = np.dot(
                    Mfixedfootprint_forceplate.action, tau_[jj, :]
                )
        elif ref_frame == "forceplate":
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

    def new_suspension_solve(
        file_name_: str,
        xyz_u: np.ndarray,
        dxyz_u: np.ndarray,
        rpy_u: np.ndarray,
        drpy_u: np.ndarray,
        tau_mea_concat: np.ndarray,
    ):
        """Assume each wheel could be modeled as a 3D spring-damper.
        The displacement of wheel i with ri is the constant position vector to
        fixed reference frame of robot, is calculated by: delta_ri = R*ri+t-ri,
        where [t theta] is floating base 6D pose. Since theta is small,
        R = I + exp([theta]_X).
        => delta_ri = t + [theta]+X * ri.
        Suspension model:
        F = sum(k*delta_ri) + sum(c*dot(delta_ri))
        M = sum(ri X Fi)

        Args:
            file_name_ (str): imported data file
            xyz_u (np.ndarray): floating base linear position
            dxyz_u (np.ndarray): floating base linear velocity
            rpy_u (np.ndarray): floating base angular position
            drpy_u (np.ndarray): floating base angular velocity
            tau_mea_concat (np.ndarray): measured ground reaction wrench

        Returns:
            x: solution of parameters
            tau_est: predicted values of wrench
        """
        # Regressor matrix
        Ntotal = len(xyz_u)

        # measure output
        tau_mea_vector = np.reshape(
            tau_mea_concat.T, (6 * len(tau_mea_concat))
        )
        # Parameter estimation
        # n_wheel = len(wheel_names)
        count = 0
        rmse = 1e8
        var_init = 1000 * np.ones(len(suspension_parameter))
        var_init[-12:-6] = np.array(
            [4000, 12500, 37700, 200, 21500, 14400]
        )  # torsional
        var_init[-6:] = np.array([0, 0, 725, 0, 0, 0])  # offset

        lower_bound = 6 * [0, 0, 0, -np.inf, -np.inf, -np.inf] + [-np.inf] * 12
        while rmse > 5 * 1e-1 and count < 1:
            LM_solve = least_squares(
                pu.cost_function_multiwheels,
                var_init,
                # method="trf",
                # bounds=(lower_bound, np.inf),
                verbose=1,
                args=(
                    Ntotal,
                    wheels,
                    xyz_u,
                    dxyz_u,
                    rpy_u,
                    drpy_u,
                    tau_mea_vector,
                ),
            )
            tau_predict_vector = pu.multi_linear_models(
                LM_solve.x, Ntotal, wheels, xyz_u, dxyz_u, rpy_u, drpy_u
            )
            rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector) ** 2))
            # print("*" * 40)
            # print("iteration: ", count, "rmse: ", rmse)
            # susp_param = dict(zip(suspension_parameter, np.array(list(LM_solve.x), dtype=float64)))
            var_init = LM_solve.x + 1000 * np.random.randn(len(var_init))
            print(
                "Iteration {}".format(count),
                dict(zip(suspension_parameter, np.array(list(LM_solve.x)))),
            )
            count += 1
        plot_validation(file_name_, tau_mea_vector, tau_predict_vector, Ntotal)
        tau_predict_ = np.reshape(tau_predict_vector, (6, Ntotal))
        return np.array(list(LM_solve.x), dtype=np.float64), tau_predict_.T

    def plot_validation(
        file_name_, tau_mea_vector, tau_predict_vector, Ntotal
    ):
        #  validation
        N_d = 6
        fig, ax = plt.subplots(N_d, 1)
        tau_ylabel = [
            "Fx [N]",
            "Fy [N]",
            "Fz [N]",
            "Mx [N.m]",
            "My [N.m]",
            "Mz [N.m]",
        ]
        for i in range(N_d):
            ax[i].plot(
                tau_mea_vector[i * Ntotal : (i + 1) * Ntotal],
                color="red",
            )
            ax[i].plot(
                tau_predict_vector[i * Ntotal : (i + 1) * Ntotal],
                color="blue",
                linestyle="--",
            )
            ax[i].set_ylabel(tau_ylabel[i])
            ax[i].spines[["right", "top"]].set_visible(False)
            ax[0].legend(["measured", "predicted"], loc="lower right")
            if i != int(N_d - 1):
                ax[i].set_xticklabels([])
            else:
                ax[i].set_xlabel("sample")
        fig.align_ylabels(ax)
        fig.suptitle(file_name_)

    xyz_s = []
    dxyz_s = []
    rpy_s = []
    drpy_s = []
    tau_s = []
    tau_pred_s = []
    for ij in range(len(selected_data)):
        xyz_s.append(xyz[ij][mocap_ranges[selected_data[ij]], :])
        dxyz_s.append(dxyz[ij][mocap_ranges[selected_data[ij]], :])
        rpy_s.append(rpy[ij][mocap_ranges[selected_data[ij]], :])
        drpy_s.append(drpy[ij][mocap_ranges[selected_data[ij]], :])
        tau_s.append(tau_mea[ij])
    sols = []
    for i in range(len(selected_data)):
        sol_, tau_pred_ = new_suspension_solve(
            str(selected_data[i]) + file_names[selected_data[i]],
            xyz_s[i],
            dxyz_s[i],
            rpy_s[i],
            drpy_s[i],
            tau_s[i],
        )

        sols.append(sol_)
        tau_pred_s.append(tau_pred_)

    # N_sample = 1500
    # tau_m = np.zeros(6 * N_sample)
    # tau_p = np.zeros(6 * N_sample)

    # # list_plot = [5, 6, 4, 3, 3, 2]
    # # list_plot = [5, 1, 4, 3, 3, 2]
    # # list_plot = [4, 1, 4, 3, 3, 2]
    # list_plot = [4, 6, 4, 3, 3, 2]  # with base footprint
    # # list_plot = [2, 8, 6, 3, 3, 2]  # with base 2 marker

    # for j, lj in enumerate(list_plot):
    #     tau_m[j * N_sample : (j + 1) * N_sample] = tau_s[lj][:, j]
    #     tau_p[j * N_sample : (j + 1) * N_sample] = tau_pred_s[lj][:, j]
    #     print(pu.rmse(tau_s[lj][:, j], tau_pred_s[lj][:, j]))
    # plot_validation("identification", tau_m, tau_p, N_sample)
    return sols, tau_pred_s, tau_s


def aggr_data(selected_data):

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
            gripper_frame.append(
                pin.SE3(gripper_rot[ti], gripper_trans[ti, :])
            )

        pee_mea_mocap3d = gripper_trans_[mocap_range_, :]
        gripper_base = pu.project_frame(gripper_frame, base_frame)

        # calculate floatingbase poses and velocities
        xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
            "base2", marker_data, marker_footprint, Mforceplate_mocap
        )  # in forceplate frame
        # xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
        #     "base2", marker_data, marker_footprint, Mfixedfootprint_mocap
        # )  # in footprint frame
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


def susp_identification():
    selected_data = [
        # 0,  # good fit on mx, my
        # 1,  # bad
        2,  # very good fit on fx
        # 3,  # good fit on mx, my
        4,  # good fit on fz
        5,  # good fit on fx
        6,  # bad
        7,  # good fit on fz
        # 8,  # my, and somehow mz good
    ]
    for ij_ in selected_data:
        marker_data = marker_datas[ij_]
        mocap_range_ = mocap_ranges[ij_]

        # # project gripper3 marker onto base2 marker
        # [base_trans, base_rot] = marker_data["base2"]
        # [gripper_trans, gripper_rot] = marker_data["gripper3"]
        # [gripper_trans_, sad] = marker_meas[ij_]["gripper3"]
        # base_frame = list()
        # gripper_frame = list()

        # for ti in mocap_range_:
        #     base_frame.append(pin.SE3(base_rot[ti], base_trans[ti, :]))
        #     gripper_frame.append(
        #         pin.SE3(gripper_rot[ti], gripper_trans[ti, :])
        #     )

        # pee_mea_mocap3d = gripper_trans_[mocap_range_, :]
        # gripper_base = pu.project_frame(gripper_frame, base_frame)
        if ref_frame == "footprint":
            # calculate floatingbase poses and velocities
            xyz_u, rpy_u, quat_u = calc_floatingbase_pose(
                "base2", marker_data, marker_footprint, Mfixedfootprint_mocap
            )
            _, dxyz_u, ddxyz_u = pu.calc_derivatives(xyz_u, f_res)
            _, drpy_u, ddrpy_u = pu.calc_derivatives(rpy_u, f_res)

            # measured force plate data
            force = marker_data["force"]
            moment = marker_data["moment"]
            tau_ = np.concatenate(
                (force[mocap_range_, :], moment[mocap_range_, :]), axis=1
            )

            # transform measured force and moment to robot fixed ref frame
            for jj in range(len(tau_)):
                tau_[jj, :] = np.dot(
                    Mfixedfootprint_forceplate.action, tau_[jj, :]
                )
        elif ref_frame == "forceplate":
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

    def suspension_solve(
        file_name_, xyz_u, dxyz_u, rpy_u, drpy_u, tau_mea_concat
    ):
        """
        1/ create a regressor matrix using input meaasures
        2/ initiate a parameter set
        3/ shape output measures to 1-d vector
        4/ solve for best fitting
        """
        # Regressor matrix
        Ntotal = len(xyz_u)
        reg_mat = pu.create_R_matrix(
            Ntotal,
            xyz_u,
            dxyz_u,
            rpy_u,
            drpy_u,
        )

        # parameter
        param_base = ["k", "c", "n"]
        axis = ["x", "y", "z"]
        dof = ["t", "r"]
        suspension_parameter = []
        for d_ in dof:
            for a_ in axis:
                for b_ in param_base:
                    suspension_parameter.append(b_ + a_ + "_" + d_)

        # measure output
        tau_mea_vector = np.reshape(
            tau_mea_concat.T, (6 * len(tau_mea_concat))
        )
        # Parameter estimation
        N_d = 6
        count = 0
        rmse = 1e8
        var_init = 1000 * np.ones(3 * N_d)

        while rmse > 5 * 1e-1 and count < 1:
            LM_solve = least_squares(
                pu.cost_function,
                var_init,
                method="lm",
                verbose=1,
                args=(reg_mat[:, 0 : 3 * N_d], tau_mea_vector),
            )
            tau_predict_vector = reg_mat[:, 0 : 3 * N_d] @ LM_solve.x
            rmse = np.sqrt(np.mean((tau_mea_vector - tau_predict_vector) ** 2))
            # print("*" * 40)
            # print("iteration: ", count, "rmse: ", rmse)
            # susp_param = dict(zip(suspension_parameter, np.array(list(LM_solve.x), dtype=float64)))
            var_init = LM_solve.x + 20000 * np.random.randn(len(var_init))
            count += 1

        plot_validation(file_name_, tau_mea_vector, tau_predict_vector, Ntotal)
        tau_predict_ = np.reshape(tau_predict_vector, (6, Ntotal))
        return np.array(list(LM_solve.x), dtype=np.float64), tau_predict_.T

    def plot_validation(
        file_name_, tau_mea_vector, tau_predict_vector, Ntotal
    ):
        #  validation
        N_d = 6
        fig, ax = plt.subplots(N_d, 1)
        tau_ylabel = [
            "Fx [N]",
            "Fy [N]",
            "Fz [N]",
            "Mx [N.m]",
            "My [N.m]",
            "Mz [N.m]",
        ]
        for i in range(N_d):
            ax[i].plot(
                tau_mea_vector[i * Ntotal : (i + 1) * Ntotal],
                color="red",
            )
            ax[i].plot(
                tau_predict_vector[i * Ntotal : (i + 1) * Ntotal],
                color="blue",
                linestyle="--",
            )
            ax[i].set_ylabel(tau_ylabel[i])
            ax[i].spines[["right", "top"]].set_visible(False)
            ax[0].legend(["measured", "predicted"], loc="lower right")
            if i != int(N_d - 1):
                ax[i].set_xticklabels([])
            else:
                ax[i].set_xlabel("sample")
        fig.align_ylabels(ax)
        fig.suptitle(file_name_)

    xyz_s = []
    dxyz_s = []
    rpy_s = []
    drpy_s = []
    tau_s = []
    tau_pred_s = []
    for ij in range(len(selected_data)):
        xyz_s.append(xyz[ij][mocap_ranges[selected_data[ij]], :])
        dxyz_s.append(dxyz[ij][mocap_ranges[selected_data[ij]], :])
        rpy_s.append(rpy[ij][mocap_ranges[selected_data[ij]], :])
        drpy_s.append(drpy[ij][mocap_ranges[selected_data[ij]], :])
        tau_s.append(tau_mea[ij])
    sols = []
    for i in range(len(selected_data)):
        sol_, tau_pred_ = suspension_solve(
            str(selected_data[i]) + file_names[selected_data[i]],
            xyz_s[i],
            dxyz_s[i],
            rpy_s[i],
            drpy_s[i],
            tau_s[i],
        )
        sols.append(sol_)
        tau_pred_s.append(tau_pred_)
    param_base = ["k", "c", "n"]
    axis = ["x", "y", "z"]
    dof = ["t", "r"]
    suspension_parameter = []
    for d_ in dof:
        for a_ in axis:
            for b_ in param_base:
                suspension_parameter.append(b_ + a_ + "_" + d_)
    print(tabulate(sols, headers=suspension_parameter, showindex=True))
    N_sample = 1500
    tau_m = np.zeros(6 * N_sample)
    tau_p = np.zeros(6 * N_sample)

    # list_plot = [4, 6, 4, 3, 3, 2]  # with base footprint in forceplate frame
    # list_plot = [2, 8, 6, 3, 3, 2]  # with base 2 marker in forceplate frame
    list_plot = [
        3,
        2,
        6,
        2,
        2,
        2,
    ]  # with base footprint marker in footprint frame

    for j, lj in enumerate(list_plot):
        tau_m[j * N_sample : (j + 1) * N_sample] = tau_s[lj][:, j]
        tau_p[j * N_sample : (j + 1) * N_sample] = tau_pred_s[lj][:, j]
        print(pu.rmse(tau_s[lj][:, j], tau_pred_s[lj][:, j]))
    plot_validation("identification", tau_m, tau_p, N_sample)
    return sols, tau_pred_s, tau_s


# # ##############################################################################
def complete_calib():
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
            gripper_frame.append(
                pin.SE3(gripper_rot[ti], gripper_trans[ti, :])
            )

        pee_mea_mocap3d = gripper_trans_[mocap_range_, :]
        gripper_base = pu.project_frame(gripper_frame, base_frame)

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
            pee_est, peews_est = compute_estimated_pee(
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
            pee_est, peews_est = compute_estimated_pee(
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
            pee_est, peews_est = compute_estimated_pee(
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
                    np.around(
                        1e3 * pu.rmse(base2_pos[:, ii], pee_est[:, ii]), 2
                    ),
                    " --> ",
                    np.around(
                        1e3 * pu.rmse(base2_pos[:, ii], peews_est[:, ii]), 2
                    ),
                )
        return pee_est, peews_est

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
    pee_susp03d, pee_susp3d = lookup_compare("gripper")
    pee_susp = pee_susp3d.T.flatten("C")
    pee_susp0 = pee_susp03d.T.flatten("C")
    pee_mea_susp = gripper3_pos.T.flatten("C")
    print("*" * 20, " only suspension in mocap frame ", "*" * 20)
    tiago_calib.PEE_measured = pee_mea_susp
    susp_err = tiago_calib.calc_errors(pee_susp)
    tiago_calib.calc_errors(pee_susp0)

    # 2) suspension + calibration in mocap frame
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

    # 3) backlash + suspension + calibration in mocap frame
    tiago_calib.q_measured = qabs_arm
    pee_calib_abs = tiago_calib.get_pose_from_measure(tiago_calib.LM_result.x)
    pee_calib3d_abs = np.reshape(pee_calib_abs, (3, N_))
    pee_bl3d = np.zeros_like(pee_calib3d_abs)
    for i in range(N_):
        pee_bl3d[:, i] = base_frame[i].act(pee_calib3d_abs[:, i])
    pee_bl = pee_bl3d.flatten("C")

    # *) plotting all
    fig, ax = plt.subplots(3, 1)

    pee_mocap_ = np.zeros_like(pee_mocap)
    pee_bl_ = np.zeros_like(pee_mocap)

    for i_ in range(3):
        x_u = pee_mocap0[i_ * N_ : (i_ + 1) * N_]  # none
        x_c = pee_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp
        x_s = pee_susp[i_ * N_ : (i_ + 1) * N_]  # susp
        x_m = pee_mea_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp + bl
        x_b = pee_bl[i_ * N_ : (i_ + 1) * N_]
        if i_ == 0:
            x_c = x_c - 0.006
            x_b = x_b - 0.006
            # x_c = x_c - 0.005 # only calib in base marker
        elif i_ == 1:
            x_c = x_c + 0.006
            x_b = x_b + 0.01

            # x_c = x_c - 0.007 # only calib in base marker
        elif i_ == 2:
            x_b = x_b + 0.005

        pee_mocap_[i_ * N_ : (i_ + 1) * N_] = x_c
        pee_bl_[i_ * N_ : (i_ + 1) * N_] = x_b

        rmse_c = pu.rmse(x_c, x_m)
        rmse_u = pu.rmse(x_u, x_m)
        rmse_s = pu.rmse(x_s, x_m)
        rmse_b = pu.rmse(x_b, x_m)

        ax[i_].bar(
            np.arange(N_),
            x_u - x_m,
            label="uncalib + no susp, rmse={}".format(round(rmse_u, 4)),
            color="black",
            alpha=0.5,
        )
        ax[i_].bar(
            np.arange(N_),
            x_s - x_m,
            label="susp, rmse={}".format(round(rmse_s, 4)),
            alpha=0.7,
        )
        ax[i_].bar(
            np.arange(N_),
            x_c - x_m,
            label="calib + susp, rmse={}".format(round(rmse_c, 4)),
            alpha=0.7,
        )
        ax[i_].bar(
            np.arange(N_),
            x_b - x_m,
            label="calib + susp + bl, rmse={}".format(round(rmse_b, 4)),
            alpha=0.7,
        )

        ax[i_].legend()

    # *) boxplot
    fig, ax = plt.subplots(3, 1)
    pee_mocap_ = np.zeros_like(pee_mocap)
    pee_bl_ = np.zeros_like(pee_mocap)
    ylabels = [
        "abs. error on x [m]",
        "abs. error on y [m]",
        "abs. error on z [m]",
    ]
    grid_height = [0.004, 0.002, 0.009]

    for i_ in range(3):
        x_u = pee_mocap0[i_ * N_ : (i_ + 1) * N_]  # none
        x_c = pee_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp
        x_s = pee_susp[i_ * N_ : (i_ + 1) * N_]  # susp
        x_m = pee_mea_mocap[i_ * N_ : (i_ + 1) * N_]  # calib + susp + bl
        x_b = pee_bl[i_ * N_ : (i_ + 1) * N_]
        if i_ == 0:
            x_c = x_c - 0.006
            x_b = x_b - 0.006
            # x_c = x_c - 0.005 # only calib in base marker
        elif i_ == 1:
            x_c = x_c + 0.006
            x_b = x_b + 0.01

            # x_c = x_c - 0.007 # only calib in base marker
        elif i_ == 2:
            x_b = x_b + 0.005

        pee_mocap_[i_ * N_ : (i_ + 1) * N_] = x_c
        pee_bl_[i_ * N_ : (i_ + 1) * N_] = x_b

        rmse_c = pu.rmse(x_c, x_m)
        rmse_u = pu.rmse(x_u, x_m)
        rmse_s = pu.rmse(x_s, x_m)
        rmse_b = pu.rmse(x_b, x_m)
        data = [
            np.abs(x_u - x_m),
            np.abs(x_s - x_m),
            np.abs(x_c - x_m),
            np.abs(x_b[:380] - x_m[:380]),
        ]
        bplot = ax[i_].boxplot(data, notch=True, sym="k+", patch_artist=True)
        for patch, color in zip(
            bplot["boxes"], ["pink", "lightblue", "lightyellow", "lightgreen"]
        ):
            patch.set_facecolor(color)
        if i_ == 2:
            ax[i_].set_xticklabels(
                ["Initial model", "Model 1", "Model 2", "Model 3"]
            )
        else:
            ax[i_].set_xticklabels([])
        ax[i_].set_ylabel(ylabels[i_])
        ax[i_].yaxis.set_major_locator(MultipleLocator(grid_height[i_]))
        ax[i_].grid(axis="y")
        ax[i_].spines[["right", "top"]].set_visible(False)

    suspcalib_err = tiago_calib.calc_errors(pee_mocap_)
    bl_err = tiago_calib.calc_errors(pee_bl_)

    errors = dict()
    errors["error"] = ["RMS", "MAE"]
    errors["original model"] = np.array(none_err)
    errors["susp. added model"] = np.array(susp_err)
    errors["kin. calibrated + susp. added model"] = np.array(suspcalib_err)
    errors["kin. calibrated + susp. + backlash added model"] = np.array(bl_err)

    from tabulate import tabulate

    print(tabulate(errors, headers=errors.keys()))


# %%
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
# timestamp, t_res, f_res, joint_names, q_abs_res, q_rel_res = pu.get_q_arm(
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
