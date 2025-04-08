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
from scipy import signal
from ..tools.regressor import build_regressor_reduced, get_index_eliminate
import quadprog
import operator


def get_param_from_yaml(robot, identif_data):
    """_This function allows to create a dictionnary of the settings set in a yaml file._

    Args:
        robot (_robot_): _Pinocchio robot_
        identif_data (_dict_): _a dictionnary containing the parameters settings for identification (set in a config yaml file) _

    Returns:
        _dict_: _a dictionnary of parameters settings_
    """

    # robot_name: anchor as a reference point for executing
    robot_name = robot.model.name

    robots_params = identif_data["robot_params"][0]
    problem_params = identif_data["problem_params"][0]
    process_params = identif_data["processing_params"][0]
    tls_params = identif_data["tls_params"][0]

    param = {
        "robot_name": robot_name,
        "nb_samples": int(1 / (process_params["ts"])),
        "q_lim_def": robots_params["q_lim_def"],
        "dq_lim_def": robots_params["dq_lim_def"],
        "is_external_wrench": problem_params["is_external_wrench"],
        "is_joint_torques": problem_params["is_joint_torques"],
        "force_torque": problem_params["force_torque"],
        "external_wrench_offsets": problem_params["external_wrench_offsets"],
        "has_friction": problem_params["has_friction"],
        "fv": robots_params["fv"],
        "fs": robots_params["fs"],
        "has_actuator_inertia": problem_params["has_actuator_inertia"],
        "Ia": robots_params["Ia"],
        "has_joint_offset": problem_params["has_joint_offset"],
        "off": robots_params["offset"],
        "has_coupled_wrist": problem_params["has_coupled_wrist"],
        "Iam6": robots_params["Iam6"],
        "fvm6": robots_params["fvm6"],
        "fsm6": robots_params["fsm6"],
        "N": robots_params["N"],
        "ratio_essential": robots_params["ratio_essential"],
        "cut_off_frequency_butterworth": process_params[
            "cut_off_frequency_butterworth"
        ],
        "ts": process_params["ts"],
        "mass_load": tls_params["mass_load"],
        "which_body_loaded": tls_params["which_body_loaded"],
    }
    return param


def set_missing_params_setting(robot, params_settings):
    """_Set the missing parameters of a robot, for instance if the urdf does not mention them_

    Args:
        robot (_robot_): _Pinocchio robot_
        params_settings (_dict_): _Dictionnary of parameters settings_

    Returns:
        _dict_: _Dictionnary of updated parameter settings. Note : the robot is updated also_
    """

    diff_limit = np.setdiff1d(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    # upper and lower joint limits are the same so use defaut values for all joints
    if not diff_limit.any:
        print("No joint limits. Set default values")
        for ii in range(robot.model.nq):
            robot.model.lowerPositionLimit[ii] = -params_settings["q_lim_def"]
            robot.model.upperPositionLimit[ii] = params_settings["q_lim_def"]

    if np.sum(robot.model.velocityLimit) == 0:
        print("No velocity limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.velocityLimit[ii] = params_settings["dq_lim_def"]

    # maybe we need to check an other field somethnibg weird here
    if np.sum(robot.model.velocityLimit) == 0:
        print("No joint torque limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.effortLimit[ii] = -params_settings["tau_lim_def"]

    accelerationLimit = np.zeros(robot.model.nq)
    for ii in range(robot.model.nq):
        # accelerationLimit to be consistent with PIN naming
        accelerationLimit[ii] = params_settings["ddq_lim_def"]
    params_settings["accelerationLimit"] = accelerationLimit
    # print(model.accelerationLimit)

    if params_settings["has_friction"]:

        for ii in range(robot.model.nv):
            if ii == 0:
                # default values of the joint viscous friction in case they are used
                fv = [(ii + 1) / 10]
                # default value of the joint static friction in case they are used
                fs = [(ii + 1) / 10]
            else:
                fv.append((ii + 1) / 10)
                fs.append((ii + 1) / 10)

        params_settings["fv"] = fv
        params_settings["fs"] = fs

    if params_settings[
        "external_wrench_offsets"
    ]:  # set for a fp of dim (1.8mx0.9m) at its center
        params_settings["OFFX"] = 900
        params_settings["OFFY"] = 450
        params_settings["OFFZ"] = 0

    return params_settings


def base_param_from_standard(phi_standard, params_base):
    """_This function allows to calculate numerically the base parameters with the values of the standard ones._

    Args:
        phi_standard (_dict_): _a dictionnary containing the values of the standard parameters of the model (usually from get_standard_parameters)_
        params_base (_list_): _a list containing the analytical relations between standard parameters to give the base parameters_

    Returns:
        _list_: _a list containing the numeric values of the base parameters_
    """

    phi_base = []
    ops = {"+": operator.add, "-": operator.sub}
    for ii in range(len(params_base)):
        param_base_i = params_base[ii].split(" ")
        values = []
        list_ops = []
        for jj in range(len(param_base_i)):
            param_base_j = param_base_i[jj].split("*")
            if len(param_base_j) == 2:
                value = float(param_base_j[0]) * phi_standard[param_base_j[1]]
                values.append(value)
            elif param_base_j[0] != "+" and param_base_j[0] != "-":
                value = phi_standard[param_base_j[0]]
                values.append(value)
            else:
                list_ops.append(ops[param_base_j[0]])
        value_phi_base = values[0]
        for kk in range(len(list_ops)):
            value_phi_base = list_ops[kk](value_phi_base, values[kk + 1])
        phi_base.append(value_phi_base)
    return phi_base


def relative_stdev(W_b, phi_b, tau):
    """_Calculates relative standard deviation of estimated parameters using the residual errro[Pressé & Gautier 1991]_

    Args:
        W_b (_array_): _Base regressor matrix_
        phi_b (_list_): _List containing the relationship for the base parameters_
        tau (_array_): _Array of force measurements_

    Returns:
        _array_: _An array containing the relative standard deviation for each base parameter_
    """
    # stdev of residual error ro
    sig_ro_sqr = np.linalg.norm((tau - np.dot(W_b, phi_b))) ** 2 / (
        W_b.shape[0] - phi_b.shape[0]
    )

    # covariance matrix of estimated parameters
    C_x = sig_ro_sqr * np.linalg.inv(np.dot(W_b.T, W_b))

    # relative stdev of estimated parameters
    std_x_sqr = np.diag(C_x)
    std_xr = np.zeros(std_x_sqr.shape[0])
    for i in range(std_x_sqr.shape[0]):
        std_xr[i] = np.round(100 * np.sqrt(std_x_sqr[i]) / np.abs(phi_b[i]), 2)

    return std_xr


def index_in_base_params(params, id_segments):
    """_This function finds the base parameters in which a parameter of a given segment (referenced with its id) appear_

    Args:
        params (_list_): _A list containing the relation for the base parameters_
        id_segments (_list_): _A list containing the ids of the segments_

    Returns:
        _dict_: _A dictionnary containing the link between base parameters and id of segments_
    """
    base_index = []
    params_name = [
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
    ]

    id_segments_new = [i for i in range(len(id_segments))]

    for id in id_segments:
        for ii in range(len(params)):
            param_base_i = params[ii].split(" ")
            for jj in range(len(param_base_i)):
                param_base_j = param_base_i[jj].split("*")
                for ll in range(len(param_base_j)):
                    for kk in params_name:
                        if kk + str(id) == param_base_j[ll]:
                            base_index.append((id, ii))

    base_index[:] = list(set(base_index))
    base_index = sorted(base_index)

    dictio = {}

    for i in base_index:
        dictio.setdefault(i[0], []).append(i[1])

    values = []
    for ii in dictio:
        values.append(dictio[ii])

    return dict(zip(id_segments_new, values))


def weigthed_least_squares(robot, phi_b, W_b, tau_meas, tau_est, param):
    """_This function computes the weigthed least square solution of the identification problem see [Gautier, 1997] for details_

    Args:
        robot (_robot_): _Pinocchio robot_
        phi_b (_liste_): _A list containing the relation for the base parameters_
        W_b (_array_): _The base regressor matrix_
        tau_meas (_array_): _An array containing the measured forces_
        tau_est (_array_): _An array containing the estimated forces_
        param (_dict_): _A dictionnary settings_

    Returns:
        _array_: _An array for the weighted base parameters_
    """
    sigma = np.zeros(
        robot.model.nq
    )  # Needs to be modified for taking into account the GRFM
    zero_identity_matrix = np.identity(len(tau_meas))
    P = np.zeros((len(tau_meas), len(tau_meas)))
    nb_samples = int(param["idx_tau_stop"][0])
    start_idx = int(0)
    for ii in range(robot.model.nq):

        sigma[ii] = np.linalg.norm(
            tau_meas[int(start_idx) : int(param["idx_tau_stop"][ii])]
            - tau_est[int(start_idx) : int(param["idx_tau_stop"][ii])]
        ) / (
            len(tau_meas[int(start_idx) : int(param["idx_tau_stop"][ii])]) - len(phi_b)
        )

        start_idx = param["idx_tau_stop"][ii]

        for jj in range(nb_samples):

            P[jj + ii * (nb_samples), jj + ii * (nb_samples)] = 1 / sigma[ii]

        phi_b = np.matmul(np.linalg.pinv(np.matmul(P, W_b)), np.matmul(P, tau_meas))

    phi_b = np.around(phi_b, 6)

    return phi_b


def calculate_first_second_order_differentiation(model, q, param, dt=None):
    """_This function calculates the derivatives (velocities and accelerations here) by central difference for given angular configurations accounting that the robot has a freeflyer or not (which is indicated in the params_settings)._

    Args:
        model (_model_): _Pinocchio model_
        q (_array_): _the angular configurations whose derivatives need to be calculated_
        param (_dict_): _a dictionnary containing the settings_
        dt (_list_, optional): _ a list containing the different timesteps between the samples (set to None by default, which means that the timestep is constant and to be found in param['ts'])_. Defaults to None.

    Returns:
        _array_: _angular configurations (whose size match the samples removed by central differences)_
        _array_: _angular velocities_
        _array_: _angular accelerations_
    """

    if param["is_joint_torques"]:
        dq = np.zeros([q.shape[0] - 1, q.shape[1]])
        ddq = np.zeros([q.shape[0] - 1, q.shape[1]])

    if param["is_external_wrench"]:
        dq = np.zeros([q.shape[0] - 1, q.shape[1] - 1])
        ddq = np.zeros([q.shape[0] - 1, q.shape[1] - 1])

    if dt is None:
        dt = param["ts"]
        for ii in range(q.shape[0] - 1):
            dq[ii, :] = pin.difference(model, q[ii, :], q[ii + 1, :]) / dt

        for jj in range(model.nq - 1):
            ddq[:, jj] = np.gradient(dq[:, jj], edge_order=1) / dt
    else:
        for ii in range(q.shape[0] - 1):
            dq[ii, :] = pin.difference(model, q[ii, :], q[ii + 1, :]) / dt[ii]

        for jj in range(model.nq - 1):
            ddq[:, jj] = np.gradient(dq[:, jj], edge_order=1) / dt

    q = np.delete(q, len(q) - 1, 0)
    q = np.delete(q, len(q) - 1, 0)

    dq = np.delete(dq, len(dq) - 1, 0)
    ddq = np.delete(ddq, len(ddq) - 1, 0)

    return q, dq, ddq


def low_pass_filter_data(data, param, nbutter=5):
    """_This function filters and elaborates data used in the identification process. The filter used is a zero phase lag butterworth filter_

    Args:
        data (_array_): _The data to filter_
        param (_dict_): _Dictionnary of settings_
        nbutter (_int_): _Order of the butterworth filter_ Defaults to 5

    Returns:
        _array_: _The filtered data_
    """

    b, a = signal.butter(
        nbutter, param["ts"] * param["cut_off_frequency_butterworth"] / 2, "low"
    )

    # data = signal.medfilt(data, 3)
    data = signal.filtfilt(
        b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)
    )

    # suppress end segments of samples due to the border effect
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    data = np.delete(data, np.s_[(data.shape[0] - nbord) : data.shape[0]], axis=0)

    return data


# SIP QP OPTIMISATION


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = 0.5 * (P + P.T) + np.eye(P.shape[0]) * (
        1e-5
    )  # make sure P is symmetric, pos,def
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


# def sample_spherical(npoints, ndim=3):
#     vec = np.random.randn(ndim, npoints)
#     vec /= np.linalg.norm(vec, axis=0)
#     return vec


def calculate_standard_parameters(
    model, W, tau, COM_max, COM_min, params_standard_u, alpha
):
    """_This function solves a qp problem to find standard parameters that are not too far from the reference value while fitting the measurements_

    Args:
        model (_model_): _Pinocchio model_
        W (_array_): _Basic regressor matrix_
        tau (_array_): _External wrench measurements_
        COM_max (_array_): _Upper boundaries for the center of mass position_
        COM_min (_array_): _Upper boundaries for the center of mass position_
        params_standard_u (_dict_): _Dictionnary containg the standard parameters from the urdf_
        alpha (_float_): _Pareto front coefficient_

    Returns:
        _array_: _An array containing the standard parameters values_
        _array_: -An array containing the standard parameters values as they are set in the urdf_
    """
    np.set_printoptions(threshold=np.inf)
    phi_ref = []
    id_inertias = []

    for jj in range(len(model.inertias.tolist())):
        if model.inertias.tolist()[jj].mass != 0:
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
            phi_ref_temp = params_standard_u[j + str(id_inertias[k])]
            phi_ref.append(phi_ref_temp)

    phi_ref = np.array(phi_ref)

    sf1 = 1 / (np.max(phi_ref) * len(phi_ref))
    sf2 = 1 / (np.max(tau) * len(tau))

    P = (1 - alpha) * sf1 * np.eye(W.shape[1]) + alpha * sf2 * np.matmul(W.T, W)
    r = -((1 - alpha) * sf1 * phi_ref.T + sf2 * alpha * np.matmul(tau.T, W))

    # Setting constraints
    G = np.zeros(((14) * (nreal), 10 * nreal))
    h = np.zeros((((14) * (nreal), 1)))

    for ii in range(nreal):
        G[14 * ii][ii * 10 + 6] = 1  # mx<mx+
        h[14 * ii] = COM_max[3 * ii]
        G[14 * ii + 1][ii * 10 + 6] = -1  # mx>mx-
        h[14 * ii + 1] = -COM_min[3 * ii]
        G[14 * ii + 2][ii * 10 + 7] = 1  # my<my+
        h[14 * ii + 2] = COM_max[3 * ii + 1]
        G[14 * ii + 3][ii * 10 + 7] = -1  # my>my-
        h[14 * ii + 3] = -COM_min[3 * ii + 1]
        G[14 * ii + 4][ii * 10 + 8] = 1  # mz<mz+
        h[14 * ii + 4] = COM_max[3 * ii + 2]
        G[14 * ii + 5][ii * 10 + 8] = -1  # mz>mz-
        h[14 * ii + 5] = -COM_min[3 * ii + 2]
        G[14 * ii + 6][ii * 10 + 9] = 1  # m<m+
        h[14 * ii + 6] = 1.3 * phi_ref[ii * 10 + 9]
        G[14 * ii + 7][ii * 10 + 9] = -1  # m>m-
        h[14 * ii + 7] = -0.7 * phi_ref[ii * 10 + 9]
        G[14 * ii + 8][ii * 10 + 0] = 1  # Ixx<Ixx+
        h[14 * ii + 8] = 1.3 * phi_ref[ii * 10 + 0]
        G[14 * ii + 9][ii * 10 + 0] = -1  # Ixx>Ixx-
        h[14 * ii + 9] = -0.7 * phi_ref[ii * 10 + 0]
        G[14 * ii + 10][ii * 10 + 3] = 1  # Iyy<Iyy+
        h[14 * ii + 10] = 1.3 * phi_ref[ii * 10 + 3]
        G[14 * ii + 11][ii * 10 + 3] = -1  # Iyy>Iyy-
        h[14 * ii + 11] = -0.7 * phi_ref[ii * 10 + 3]
        G[14 * ii + 12][ii * 10 + 5] = 1  # Izz<Izz+
        h[14 * ii + 12] = 1.3 * phi_ref[ii * 10 + 5]
        G[14 * ii + 13][ii * 10 + 5] = -1  # Izz>Izz-
        h[14 * ii + 13] = -0.7 * phi_ref[ii * 10 + 5]

    # print(G.shape,h.shape,A.shape,b.shape)

    # SOLVING
    phi_standard = quadprog_solve_qp(P, r, G, h.reshape((G.shape[0],)))

    return phi_standard, phi_ref
