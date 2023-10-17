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
    """This function allows to create a dictionnary of the settings set in a
    yaml file.
    Input:  robot: (Robot Tpl) a robot (extracted from an URDF for instance)
            identif_data: (dict) a dictionnary containing the parameters
            settings for identification (set in a config yaml file)
    Output: param: (dict) a dictionnary of parameters settings
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
    diff_limit = np.setdiff1d(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    # upper and lower joint limits are the same , defaut values for all joints
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
                # default values of the joint viscous friction
                fv = [(ii + 1) / 10]
                # default value of the joint static friction
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
    """This function allows to calculate numerically the base parameters with
    the values of the standard ones.
    Input:  phi_standard: (tuple) a dictionnary containing the values of the
            standard parameters of the model (from get_standard_parameters)
            params_base: (list) a list containing the analytical relations
            between standard parameters to give the base parameters
    Output: phi_base: (list) a list containing the numeric values of the base
            parameters
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
    """Calculates relative standard deviation of estimated parameters using
    the residual errro[Pressé & Gautier 1991]"""
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


def weigthed_least_squares(robot, phi_b, W_b, tau_meas, tau_est, param):
    """This function computes the weigthed least square solution of the
    identification problem see [Gautier, 1997] for details
    Input:  robot: pinocchio robot structure
            W_b: base regressor matrix
            tau: measured joint torques
    """
    sigma = np.zeros(
        robot.model.nq
    )  # Needs to be modified for taking into account the GRFM
    P = np.zeros((len(tau_meas), len(tau_meas)))
    nb_samples = int(param["idx_tau_stop"][0])
    start_idx = int(0)
    for ii in range(robot.model.nq):
        sigma[ii] = np.linalg.norm(
            tau_meas[int(start_idx): int(param["idx_tau_stop"][ii])]
            - tau_est[int(start_idx): int(param["idx_tau_stop"][ii])]
        ) / (
            len(tau_meas[int(start_idx): int(param["idx_tau_stop"][ii])])
            - len(phi_b)
        )

        start_idx = param["idx_tau_stop"][ii]

        for jj in range(nb_samples):
            P[jj + ii * (nb_samples), jj + ii * (nb_samples)] = 1 / sigma[ii]

        phi_b = np.matmul(
            np.linalg.pinv(np.matmul(P, W_b)), np.matmul(P, tau_meas)
        )

    phi_b = np.around(phi_b, 6)

    return phi_b


def calculate_first_second_order_differentiation(model, q, param, dt=None):
    """This function calculates the derivatives (velocities and accelerations
    here) by central difference for given angular configurations accounting
    that the robot has a freeflyer or not (which is indicated in the
    params_settings).
    Input:  model: (Model Tpl) the robot model
            q: (array) the angular configurations whose derivatives need to be
            calculated
            param: (dict) a dictionnary containing the settings
            dt:  (list) a list containing the different timesteps between the
            samples (set to None by default, which means that the timestep is
            constant and to be found in param['ts'])
    Output: q: (array) angular configurations (whose size match the samples
            removed by central differences)
            dq: (array) angular velocities
            ddq: (array) angular accelerations
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


def low_pass_filter_data(data, param, nbutter):
    """This function filters and elaborates data used in the identification
    process. It is based on a return of experience  of Prof Maxime Gautier
    (LS2N, Nantes, France)
    """

    b, a = signal.butter(
        nbutter,
        param["ts"] * param["cut_off_frequency_butterworth"] / 2,
        "low",
    )

    # data = signal.medfilt(data, 3)
    data = signal.filtfilt(
        b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)
    )

    # suppress end segments of samples due to the border effect
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    data = np.delete(
        data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0
    )

    return data


# Function for the total least square


def build_total_regressor(
    W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, param
):
    """Inputs:  W_b_u  base regressor for unloaded case
            W_b_l:  base Regressor for loaded case
            W_l: Full  regressor for loaded case
            I_u: measured current in uloaded case in A
            I_l: measured current in loaded case in A
    Ouputs: W_tot: total regressor matrix
            V_norm= Normalised solution vector
            residue"""

    # build the total regressor matrix for TLS
    # we have to add a minus in front of the regressors for TLS
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    nb_j = int(len(I_u) / param["nb_samples"])

    # nv (or 6) columns for the current
    V_a = np.concatenate(
        (
            I_u[0: param["nb_samples"]].reshape(param["nb_samples"], 1),
            np.zeros(((nb_j - 1) * param["nb_samples"], 1)),
        ),
        axis=0,
    )
    V_b = np.concatenate(
        (
            I_l[0: param["nb_samples"]].reshape(param["nb_samples"], 1),
            np.zeros(((nb_j - 1) * param["nb_samples"], 1)),
        ),
        axis=0,
    )

    for ii in range(1, nb_j):
        V_a_ii = np.concatenate(
            (
                np.concatenate(
                    (
                        np.zeros((param["nb_samples"] * (ii), 1)),
                        I_u[
                            param["nb_samples"]
                            * (ii): (ii + 1)
                            * param["nb_samples"]
                        ].reshape(param["nb_samples"], 1),
                    ),
                    axis=0,
                ),
                np.zeros((param["nb_samples"] * (5 - (ii)), 1)),
            ),
            axis=0,
        )
        V_b_ii = np.concatenate(
            (
                np.concatenate(
                    (
                        np.zeros((param["nb_samples"] * (ii), 1)),
                        I_l[
                            param["nb_samples"]
                            * (ii): (ii + 1)
                            * param["nb_samples"]
                        ].reshape(param["nb_samples"], 1),
                    ),
                    axis=0,
                ),
                np.zeros((param["nb_samples"] * (5 - (ii)), 1)),
            ),
            axis=0,
        )
        V_a = np.concatenate((V_a, V_a_ii), axis=1)
        V_b = np.concatenate((V_b, V_b_ii), axis=1)

    W_current = np.concatenate((V_a, V_b), axis=0)

    W_tot = np.concatenate((W_tot, W_current), axis=1)

    # selection and reduction of the regressor for the unknown parameters

    if param["has_friction"]:  # adds fv and fs
        W_l_temp = np.zeros((len(W_l), 12))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
            W_l_temp[:, k] = W_l[
                :, (param["which_body_loaded"]) * 12 + k
            ]  # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz fs fv
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (param["which_body_loaded"]) * 12 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    elif param["has_actuator_inertia"]:  # adds ia fv fs off
        W_l_temp = np.zeros((len(W_l), 14))
        # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz ia fv fs off
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]:
            W_l_temp[:, k] = W_l[:, (param["which_body_loaded"]) * 14 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (param["which_body_loaded"]) * 14 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    else:
        W_l_temp = np.zeros((len(W_l), 9))
        for k in range(9):
            W_l_temp[:, k] = W_l[
                :, (param["which_body_loaded"]) * 10 + k
            ]  # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (param["which_body_loaded"]) * 10 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)

    V = np.transpose(Vh).conj()

    # for validation purpose
    # W_tot_est=W_tot#-S[-1]*np.matmul(U[:,-1].reshape(len(W_tot),1),np.transpose(V[:,-1].reshape(len(Vh),1)))

    V_norm = param["mass_load"] * np.divide(V[:, -1], V[-1, -1])

    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue


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


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def calculate_standard_parameters_vold(
    robot, W, tau, COM_max, COM_min, params_settings
):
    """This function retrieves the 10 standard parameters (m, 3D COM, 6D
    Inertias) for each body in the body tree thanks to a QP optimisation
    (cf Jovic 2016).
    Input:  robot : (Robot Tpl) a robot extracted from an urdf for instance
            W : ((Nsamples*njoints,14*nbodies) array) the full dynamic
            regressor (calculated thanks to build regressor basic )
            tau : ((Nbsamples*njoints,) array) the joint torque array
            COM_max : (list) sup boundaries for COM in the form (x,y,z) for
            each body
            COM_min : (list) sup boundaries for COM in the form (x,y,z) for
            each body
            params_settings : (dict) a dictionnary indicating the settings
            (extracted from get_parameters_from_yaml)
    Output: phi_standard: (list) a list containing the numeric values of the
            standard parameters
            phi_ref : (list) a list containing the numeric values of the
            standard parameters as they are set in the urdf
    """
    alpha = 0.9
    phi_ref = []
    id_inertias = []
    id_virtual = []

    if params_settings["is_external_wrench"]:
        for jj in range(1, len(robot.model.inertias.tolist())):
            if robot.model.inertias.tolist()[jj].mass != 0:
                id_inertias.append(jj - 1)
            else:
                id_virtual.append(jj - 1)
    else:
        for jj in range(len(robot.model.inertias.tolist())):
            if robot.model.inertias.tolist()[jj].mass != 0:
                id_inertias.append(jj)
            else:
                id_virtual.append(jj)

    nreal = len(id_inertias)
    nvirtual = len(id_virtual)
    nbodies = nreal + nvirtual

    params_standard_u = robot.get_standard_parameters(params_settings)

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

    for k in range(1, nbodies + 1):
        for j in params_name:
            phi_ref_temp = params_standard_u[j + str(k)]
            phi_ref.append(phi_ref_temp)

    phi_ref = np.array(phi_ref)

    P = np.matmul(W.T, W) + alpha * np.eye(10 * (nbodies))
    r = -(np.matmul(tau.T, W) + alpha * phi_ref.T)

    # Setting constraints
    epsilon = 0.001
    v = sample_spherical(2000)  # vectors over the unit sphere

    G = np.zeros(((7 + len(v[0])) * (nreal), 10 * (nbodies)))
    h = np.zeros((((7 + len(v[0])) * (nreal), 1)))
    # A=np.zeros((10*(nvirtual),10*(nbodies)))
    # b=np.zeros((10*nvirtual,1))

    for ii in range(len(id_inertias)):
        for k in range(
            len(v[0])
        ):  # inertia matrix def pos for enough vectors onunit sphere
            G[ii * (len(v[0]) + 7) + k][id_inertias[ii] * 10 + 0] = (
                -v[0][k] ** 2
            )
            G[ii * (len(v[0]) + 7) + k][id_inertias[ii] * 10 + 1] = (
                -2 * v[0][k] * v[1][k]
            )
            G[ii * (len(v[0]) + 7) + k][id_inertias[ii] * 10 + 2] = (
                -2 * v[0][k] * v[2][k]
            )
            G[ii * (len(v[0]) + 7) + k][id_inertias[ii] * 10 + 3] = (
                -v[1][k] ** 2
            )
            G[ii * (len(v[0]) + 7) + k][id_inertias[ii] * 10 + 4] = (
                -2 * v[1][k] * v[2][k]
            )
            G[ii * (len(v[0]) + 7) + k][id_inertias[ii] * 10 + 5] = (
                -v[2][k] ** 2
            )
            h[ii * (len(v[0]) + 7) + k] = epsilon
        G[len(v[0]) + ii * (len(v[0]) + 7)][
            id_inertias[ii] * 10 + 6
        ] = 1  # mx<mx+
        G[len(v[0]) + ii * (len(v[0]) + 7)][
            id_inertias[ii] * 10 + 9
        ] = -COM_max[
            3 * ii
        ]  # mx<mx+
        G[len(v[0]) + ii * (len(v[0]) + 7) + 1][
            id_inertias[ii] * 10 + 6
        ] = -1  # mx>mx-
        G[len(v[0]) + ii * (len(v[0]) + 7) + 1][
            id_inertias[ii] * 10 + 9
        ] = COM_min[
            3 * ii
        ]  # mx>mx-
        G[len(v[0]) + ii * (len(v[0]) + 7) + 2][
            id_inertias[ii] * 10 + 7
        ] = 1  # my<my+
        G[len(v[0]) + ii * (len(v[0]) + 7) + 2][
            id_inertias[ii] * 10 + 9
        ] = -COM_max[
            3 * ii + 1
        ]  # my<my+
        G[len(v[0]) + ii * (len(v[0]) + 7) + 3][
            id_inertias[ii] * 10 + 7
        ] = -1  # my>my-
        G[len(v[0]) + ii * (len(v[0]) + 7) + 3][
            id_inertias[ii] * 10 + 9
        ] = COM_min[
            3 * ii + 1
        ]  # my>my-
        G[len(v[0]) + ii * (len(v[0]) + 7) + 4][
            id_inertias[ii] * 10 + 8
        ] = 1  # mz<mz+
        G[len(v[0]) + ii * (len(v[0]) + 7) + 4][
            id_inertias[ii] * 10 + 9
        ] = -COM_max[
            3 * ii + 2
        ]  # mz<mz+
        G[len(v[0]) + ii * (len(v[0]) + 7) + 5][
            id_inertias[ii] * 10 + 8
        ] = -1  # mz>mz-
        G[len(v[0]) + ii * (len(v[0]) + 7) + 5][
            id_inertias[ii] * 10 + 9
        ] = COM_min[
            3 * ii + 2
        ]  # mz>mz-
        G[len(v[0]) + ii * (len(v[0]) + 7) + 6][
            id_inertias[ii] * 10 + 9
        ] = -1  # m>0

    # SOLVING
    phi_standard = quadprog_solve_qp(
        P, r, G, h.reshape(((7 + len(v[0])) * (nreal),))
    )  # ,A,b.reshape((10*(nvirtual),)))

    return phi_standard, phi_ref


# def calculate_standard_parameters(
#     robot, W, tau, COM_max, COM_min, params_settings
# ):
#     """This function retrieves the 10 standard parameters (m, 3D COM, 6D
#     Inertias) for each body in the body tree thanks to a QP optimisation
#     (cf Jovic 2016).
#     Input:  robot : (Robot Tpl) a robot extracted from an urdf for instance
#             W : ((Nsamples*njoints,14*nbodies) array) the full dynamic
#             regressor (calculated thanks to build regressor basic )
#             tau : ((Nbsamples*njoints,) array) the joint torque array
#             COM_max : (list) sup boundaries for COM in the form (x,y,z) for
#             each body
#             COM_min : (list) sup boundaries for COM in the form (x,y,z) for
#             each body
#             params_settings : (dict) a dictionnary indicating the settings
#             (extracted from get_parameters_from_yaml)
#     Output: phi_standard: (list) a list containing the numeric values of the
#             standard parameters
#             phi_ref : (list) a list containing the numeric values of the
#             standard parameters as they are set in the urdf
#     """

#     alpha = 0.8
#     phi_ref = []

#     nbodies = len(robot.model.inertias)

#     params_standard_u = robot.get_standard_parameters(params_settings)

#     params_name = (
#         "Ixx",
#         "Ixy",
#         "Ixz",
#         "Iyy",
#         "Iyz",
#         "Izz",
#         "mx",
#         "my",
#         "mz",
#         "m",
#         "Ia",
#         "fv",
#         "fs",
#         "off",
#     )

#     for k in range(1, nbodies):
#         for j in params_name:
#             if params_standard_u[j + str(k)] is not None:
#                 phi_ref_temp = params_standard_u[j + str(k)]
#             else:
#                 phi_ref_temp = 0
#             phi_ref.append(phi_ref_temp)

#     phi_ref = np.array(phi_ref)
#     P = np.matmul(W.T, W) + alpha * np.eye(14 * (nbodies - 1))
#     r = -(np.matmul(tau.T, W) + alpha * phi_ref.T)

#     # Setting constraints

#     epsilon = 0.001
#     v = sample_spherical(2000)  # vectors over the unit sphere

#     G = np.zeros(((7 + len(v[0])) * (nbodies - 1), 14 * (nbodies - 1)))
#     h = np.zeros((((7 + len(v[0])) * (nbodies - 1), 1)))

#     for ii in range(nbodies - 1):
#         for k in range(
#             len(v[0])
#         ):  # inertia matrix def pos for enough vectors on unit sphere
#             G[ii * (len(v[0]) + 7) + k][ii * 14 + 0] = -v[0][k] ** 2
#             G[ii * (len(v[0]) + 7) + k][ii * 14 + 1] = -2 * v[0][k] * v[1][k]
#             G[ii * (len(v[0]) + 7) + k][ii * 14 + 2] = -2 * v[0][k] * v[2][k]
#             G[ii * (len(v[0]) + 7) + k][ii * 14 + 3] = -v[1][k] ** 2
#             G[ii * (len(v[0]) + 7) + k][ii * 14 + 4] = -2 * v[1][k] * v[2][k]
#             G[ii * (len(v[0]) + 7) + k][ii * 14 + 5] = -v[2][k] ** 2
#             h[ii * (len(v[0]) + 7) + k] = epsilon
#         G[len(v[0]) + ii * (len(v[0]) + 7)][ii * 14 + 6] = 1  # mx<mx+
#         G[len(v[0]) + ii * (len(v[0]) + 7)][ii * 14 + 9] = -COM_max[
#             3 * ii
#         ]  # mx<mx+
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 1][ii * 14 + 6] = -1  # mx>mx-
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 1][ii * 14 + 9] = COM_min[
#             3 * ii
#         ]  # mx>mx-
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 2][ii * 14 + 7] = 1  # my<my+
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 2][ii * 14 + 9] = -COM_max[
#             3 * ii + 1
#         ]  # my<my+
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 3][ii * 14 + 7] = -1  # my>my-
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 3][ii * 14 + 9] = COM_min[
#             3 * ii + 1
#         ]  # my>my-
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 4][ii * 14 + 8] = 1  # mz<mz+
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 4][ii * 14 + 9] = -COM_max[
#             3 * ii + 2
#         ]  # mz<mz+
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 5][ii * 14 + 8] = -1  # mz>mz-
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 5][ii * 14 + 9] = COM_min[
#             3 * ii + 2
#         ]  # mz>mz-
#         G[len(v[0]) + ii * (len(v[0]) + 7) + 6][ii * 14 + 9] = -1  # m>0

#     # SOLVING
#     phi_standard = quadprog_solve_qp(
#         P, r, G, h.reshape(((7 + len(v[0])) * (nbodies - 1),))
#     )

#     return phi_standard, phi_ref
