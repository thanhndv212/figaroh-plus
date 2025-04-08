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


# def build_regressor_basic(N, robot, q, v, a):
#     # TODO: reorgnize columns from ['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz']
#     # to ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
#     W = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
#     for i in range(N):
#         W_temp = pin.computeJointTorqueRegressor(
#             robot.model, robot.data, q[i, :], v[i, :], a[i, :]
#         )
#         for j in range(W_temp.shape[0]):
#             W[j * N + i, :] = W_temp[j, :]
#     W_mod = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
#     for k in range(robot.model.nv):
#         W_mod[:, 10 * k + 9] = W[:, 10 * k + 0]  # m
#         W_mod[:, 10 * k + 8] = W[:, 10 * k + 3]  # mz
#         W_mod[:, 10 * k + 7] = W[:, 10 * k + 2]  # my
#         W_mod[:, 10 * k + 6] = W[:, 10 * k + 1]  # mx
#         W_mod[:, 10 * k + 5] = W[:, 10 * k + 9]  # Izz
#         W_mod[:, 10 * k + 4] = W[:, 10 * k + 8]  # Iyz
#         W_mod[:, 10 * k + 3] = W[:, 10 * k + 6]  # Iyy
#         W_mod[:, 10 * k + 2] = W[:, 10 * k + 7]  # Ixz
#         W_mod[:, 10 * k + 1] = W[:, 10 * k + 5]  # Ixy
#         W_mod[:, 10 * k + 0] = W[:, 10 * k + 4]  # Ixx
#     return W_mod


def build_regressor_basic(robot, q, v, a, param, tau=None):
    """This function builds the basic regressor of the 10(+4) parameters
    'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m'+ ('ia','fs','fv','off') using pinocchio
    library depending on param.
    Input:  robot: (robot) a robot extracted from an urdf (for instance)
            q: (ndarray) a configuration position vector (size robot.model.nq)
            v: (ndarray) a configuration velocity vector (size robot.model.nv)
            a: (ndarray) a configutation acceleration vectore (size robot.model.na)
            param: (dict) a dictionnary setting the options, i.e., here add two
            parameters, 'ia' if the flag 'has_actuator_inertia' is true,'fs' and 'fv' if the flag 'has friction' is true, 'off' is the flag "has_joint_offset' is true
            tau : (ndarray) of stacked torque measurements (Fx,Fy,Fz), None if the torque offsets are not identified
    Output: W_mod: (ndarray) basic regressor for 10(+4) parameters
    """
    # TODO : test phase with all the different cases between ia, fv+fs, off to see if all have been correctly handled + add similiar code for external wrench case (+ friction, ia,off,etc..)

    N = len(q)  # nb of samples

    id_inertias = []
    for jj in range(len(robot.model.inertias.tolist())):
        if robot.model.inertias.tolist()[jj].mass != 0:
            id_inertias.append(jj)
    nb_in = len(id_inertias)

    nb_in_total = (
        len(robot.model.inertias) - 1
    )  # -1 if base link has inertia without external wrench, else -1 for freeflyer

    nv = robot.model.nv

    add_col = 4
    # TODO: build regressor for the case of both joint torques and external wrenches.
    if param["is_joint_torques"]:
        W = np.zeros([N * nv, (10 + add_col) * nv])
        W_mod = np.zeros([N * nv, (10 + add_col) * nv])
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for j in range(W_temp.shape[0]):
                W[j * N + i, 0 : 10 * nv] = W_temp[j, :]

                if param["has_friction"]:
                    W[j * N + i, 10 * nv + 2 * j] = v[i, j]  # fv
                    W[j * N + i, 10 * nv + 2 * j + 1] = np.sign(v[i, j])  # fs
                else:
                    W[j * N + i, 10 * nv + 2 * j] = 0  # fv
                    W[j * N + i, 10 * nv + 2 * j + 1] = 0  # fs

                if param["has_actuator_inertia"]:
                    W[j * N + i, 10 * nv + 2 * nv + j] = a[i, j]  # ia
                else:
                    W[j * N + i, 10 * nv + 2 * nv + j] = 0  # ia

                if param["has_joint_offset"]:
                    W[j * N + i, 10 * nv + 2 * nv + nv + j] = 1  # off
                else:
                    W[j * N + i, 10 * nv + 2 * nv + nv + j] = 0  # off

        for k in range(nv):
            W_mod[:, (10 + add_col) * k + 9] = W[:, 10 * k + 0]  # m
            W_mod[:, (10 + add_col) * k + 8] = W[:, 10 * k + 3]  # mz
            W_mod[:, (10 + add_col) * k + 7] = W[:, 10 * k + 2]  # my
            W_mod[:, (10 + add_col) * k + 6] = W[:, 10 * k + 1]  # mx
            W_mod[:, (10 + add_col) * k + 5] = W[:, 10 * k + 9]  # Izz
            W_mod[:, (10 + add_col) * k + 4] = W[:, 10 * k + 8]  # Iyz
            W_mod[:, (10 + add_col) * k + 3] = W[:, 10 * k + 6]  # Iyy
            W_mod[:, (10 + add_col) * k + 2] = W[:, 10 * k + 7]  # Ixz
            W_mod[:, (10 + add_col) * k + 1] = W[:, 10 * k + 5]  # Ixy
            W_mod[:, (10 + add_col) * k + 0] = W[:, 10 * k + 4]  # Ixx

            W_mod[:, (10 + add_col) * k + 10] = W[:, 10 * nv + 2 * nv + k]  # ia
            W_mod[:, (10 + add_col) * k + 11] = W[:, 10 * nv + 2 * k]  # fv
            W_mod[:, (10 + add_col) * k + 12] = W[:, 10 * nv + 2 * k + 1]  # fs
            W_mod[:, (10 + add_col) * k + 13] = W[:, 10 * nv + 2 * nv + nv + k]  # off

    elif param["is_external_wrench"]:
        ft = param["force_torque"]
        W = np.zeros([N * 6, (10 + add_col) * (nb_in_total)])
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for k in range(len(ft)):
                if ft[k] == "Fx":
                    j = 0
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10 : 10 * idx_in
                        ]
                elif ft[k] == "Fy":
                    j = 1
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10 : 10 * idx_in
                        ]
                elif ft[k] == "Fz":
                    j = 2
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10 : 10 * idx_in
                        ]
                elif ft[k] == "Mx":
                    j = 3
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10 : 10 * idx_in
                        ]
                elif ft[k] == "My":
                    j = 4
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10 : 10 * idx_in
                        ]
                elif ft[k] == "Mz":
                    j = 5
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10 : 10 * idx_in
                        ]
                elif ft[k] == "All":
                    for j in range(6):
                        for idx_in in id_inertias:
                            W[j * N + i, (idx_in - 1) * 10 : 10 * idx_in] = W_temp[
                                j, (idx_in - 1) * 10 : 10 * idx_in
                            ]
                else:
                    raise ValueError("Please enter valid parameters")

            for j in range(nb_in_total):
                for k in range(6):
                    if param["has_friction"]:
                        W[k * N + i, 10 * nb_in_total + 2 * j] = v[i, j]  # fv
                        W[k * N + i, 10 * nb_in_total + 2 * j + 1] = np.sign(
                            v[i, j]
                        )  # fs
                    else:
                        W[k * N + i, 10 * nb_in_total + 2 * j] = 0  # fv
                        W[k * N + i, 10 * nb_in_total + 2 * j + 1] = 0  # fs

                    if param["has_actuator_inertia"]:
                        W[k * N + i, 10 * nb_in_total + 2 * nb_in_total + j] = a[
                            i, j
                        ]  # ia
                    else:
                        W[k * N + i, 10 * nb_in_total + 2 * nb_in_total + j] = 0  # ia

                    if param["has_joint_offset"]:
                        W[
                            k * N + i,
                            10 * nb_in_total + 2 * nb_in_total + nb_in_total + j,
                        ] = 1  # off
                    else:
                        W[
                            k * N + i,
                            10 * nb_in_total + 2 * nb_in_total + nb_in_total + j,
                        ] = 0  # off

        W_mod = np.zeros([N * 6, (10 + add_col) * (nb_in_total)])

        for k in range(nb_in_total):
            W_mod[:, (10 + add_col) * k + 9] = W[:, 10 * k + 0]  # m
            W_mod[:, (10 + add_col) * k + 8] = W[:, 10 * k + 3]  # mz
            W_mod[:, (10 + add_col) * k + 7] = W[:, 10 * k + 2]  # my
            W_mod[:, (10 + add_col) * k + 6] = W[:, 10 * k + 1]  # mx
            W_mod[:, (10 + add_col) * k + 5] = W[:, 10 * k + 9]  # Izz
            W_mod[:, (10 + add_col) * k + 4] = W[:, 10 * k + 8]  # Iyz
            W_mod[:, (10 + add_col) * k + 3] = W[:, 10 * k + 6]  # Iyy
            W_mod[:, (10 + add_col) * k + 2] = W[:, 10 * k + 7]  # Ixz
            W_mod[:, (10 + add_col) * k + 1] = W[:, 10 * k + 5]  # Ixy
            W_mod[:, (10 + add_col) * k + 0] = W[:, 10 * k + 4]  # Ixx

            W_mod[:, (10 + add_col) * k + 10] = W[
                :, 10 * nb_in_total + 2 * nb_in_total + k
            ]  # ia
            W_mod[:, (10 + add_col) * k + 11] = W[:, 10 * nb_in_total + 2 * k]  # fv
            W_mod[:, (10 + add_col) * k + 12] = W[:, 10 * nb_in_total + 2 * k + 1]  # fs
            W_mod[:, (10 + add_col) * k + 13] = W[
                :, 10 * nb_in_total + 2 * nb_in_total + nb_in_total + k
            ]  # off

    return W_mod


def add_actuator_inertia(W, robot, q, v, a, param):
    N = len(q)  # nb of samples
    nv = robot.model.nv
    add_col = 4
    for k in range(nv):
        W[:, (10 + add_col) * k + 10] = a[i, j]
    return W


def add_friction(W, robot, q, v, a, param):
    N = len(q)  # nb of samples
    nv = robot.model.nv
    add_col = 4
    for k in range(nv):
        W[:, (10 + add_col) * k + 11] = v[i, j]
        W[:, (10 + add_col) * k + 12] = np.sign(v[i, j])
    return W


def add_joint_offset(W, robot, q, v, a, param):
    N = len(q)  # nb of samples
    nv = robot.model.nv
    add_col = 4
    for k in range(nv):
        W[:, (10 + add_col) * k + 13] = 1
    return W


def add_coupling_TX40(W, model, data, N, nq, nv, njoints, q, v, a):
    """Dedicated function for Staubli TX40"""
    W = np.c_[W, np.zeros([W.shape[0], 3])]
    for i in range(N):
        # joint 5
        W[4 * N + i, W.shape[1] - 3] = a[i, 5]
        W[4 * N + i, W.shape[1] - 2] = v[i, 5]
        W[4 * N + i, W.shape[1] - 1] = np.sign(v[i, 4] + v[i, 5])
        # joint 6
        W[5 * N + i, W.shape[1] - 3] = a[i, 4]
        W[5 * N + i, W.shape[1] - 2] = v[i, 4]
        W[5 * N + i, W.shape[1] - 1] = np.sign(v[i, 4] + v[i, 5])

    return W


def eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
    """This function eliminates columns which has L2 norm smaller than tolerance.
    Input:  W: (ndarray) joint torque regressor
            params_std: (dict) standard parameters
            tol_e: (float) tolerance
    Output: W_e: (ndarray) reduced regressor
            params_r: [list] corresponding parameters to columns of reduced regressor"""
    col_norm = np.diag(np.dot(W.T, W))
    idx_e = []
    params_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
            params_e.append(list(params_std.keys())[i])
        else:
            params_r.append(list(params_std.keys())[i])
    idx_e = tuple(idx_e)
    W_e = np.delete(W, idx_e, 1)
    return W_e, params_r


def get_index_eliminate(W, params_std, tol_e=1e-6):
    col_norm = np.diag(np.dot(W.T, W))
    idx_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
        else:
            params_r.append(list(params_std.keys())[i])
    return idx_e, params_r


def build_regressor_reduced(W, idx_e):
    W_e = np.delete(W, idx_e, 1)
    return W_e


# Function for the total least square


def build_total_regressor_current(W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, param):
    """_This function computes the regressor associated to the Total Least Square algorithm when the measurements are joints currents. For more details see [Gautier 2013]_

    Args:
        W_b_u (_array_): _base regressor matrix for unloaded case _
        W_b_l (_array_): _base regressor matrix for loaded case _
        W_l (_array_): _Full  regressor matrix for loaded case_
        I_u (_array_): _Joint current in the unloaded case_
        I_l (_array_): _Joint current in the loaded case_
        param_standard_l (_dict_): _A list of the standard parameters value in the loaded case_
        param (_dict_): _Dictionnary of settings_

    Returns:
        _array_: _The total regressor matrix_
        _array_: _The normalized vector of standard parameters_
        _array_: _The residue associated_
    """

    # build the total regressor matrix for TLS
    # we have to add a minus in front of the regressors for TLS
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    nb_j = int(len(I_u) / param["nb_samples"])

    # nv (or 6) columns for the current
    V_a = np.concatenate(
        (
            I_u[0 : param["nb_samples"]].reshape(param["nb_samples"], 1),
            np.zeros(((nb_j - 1) * param["nb_samples"], 1)),
        ),
        axis=0,
    )
    V_b = np.concatenate(
        (
            I_l[0 : param["nb_samples"]].reshape(param["nb_samples"], 1),
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
                            param["nb_samples"] * (ii) : (ii + 1) * param["nb_samples"]
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
                            param["nb_samples"] * (ii) : (ii + 1) * param["nb_samples"]
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

    # selection and reduction of the regressor for the unknown parameters for the mass

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
                -W_l[:, (param["which_body_loaded"]) * 12 + 9].reshape(len(W_l), 1),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    elif param["has_actuator_inertia"]:  # adds ia fv fs off
        W_l_temp = np.zeros((len(W_l), 14))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]:
            W_l_temp[:, k] = W_l[
                :, (param["which_body_loaded"]) * 14 + k
            ]  # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz ia fv fs off
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
                -W_l[:, (param["which_body_loaded"]) * 14 + 9].reshape(len(W_l), 1),
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
                -W_l[:, (param["which_body_loaded"]) * 10 + 9].reshape(len(W_l), 1),
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


def build_total_regressor_wrench(
    W_b_u, W_b_l, W_l, tau_u, tau_l, param_standard_l, param
):
    """_This function computes the regressor associated to the Total Least Square algorithm when the measurements are external wrenches. For more details see [Gautier 2013]_

    Args:
        W_b_u (_array_): _base regressor matrix for unloaded case _
        W_b_l (_array_): _base regressor matrix for loaded case _
        W_l (_array_): _Full  regressor matrix for loaded case_
        tau_u (_array_): _External wrench in the unloaded case_
        tau_l (_array_): _External wrench in the loaded case_
        param_standard_l (_dict_): _A list of the standard parameters value in the loaded case_
        param (_dict_): _Dictionnary of settings_

    Returns:
        _array_: _The total regressor matrix_
        _array_: _The normalized vector of standard parameters_
        _array_: _The residue associated_
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    tau_meast_ul = np.reshape(tau_u, (len(tau_u), 1))
    tau_meast_l = np.reshape(tau_l, (len(tau_l), 1))

    tau_meast_all = np.concatenate((tau_meast_ul, tau_meast_l), axis=0)

    nb_samples_ul = int(len(tau_meast_ul) / 6)
    nb_samples_l = int(len(tau_meast_l) / 6)

    tau_ul = np.concatenate(
        (
            tau_meast_ul[:nb_samples_ul],
            np.zeros(((len(tau_meast_ul) - nb_samples_ul), 1)),
        ),
        axis=0,
    )
    tau_l = np.concatenate(
        (tau_meast_l[:nb_samples_l], np.zeros(((len(tau_meast_l) - nb_samples_l), 1))),
        axis=0,
    )

    for ii in range(1, 6):
        tau_ul_ii = np.concatenate(
            (
                np.concatenate(
                    (
                        np.zeros((nb_samples_ul * (ii), 1)),
                        tau_meast_ul[nb_samples_ul * (ii) : (ii + 1) * nb_samples_ul],
                    ),
                    axis=0,
                ),
                np.zeros((nb_samples_ul * (5 - (ii)), 1)),
            ),
            axis=0,
        )
        tau_l_ii = np.concatenate(
            (
                np.concatenate(
                    (
                        np.zeros((nb_samples_l * (ii), 1)),
                        tau_meast_l[nb_samples_l * (ii) : (ii + 1) * nb_samples_l],
                    ),
                    axis=0,
                ),
                np.zeros((nb_samples_l * (5 - (ii)), 1)),
            ),
            axis=0,
        )
        tau_ul = np.concatenate((tau_ul, tau_ul_ii), axis=1)
        tau_l = np.concatenate((tau_l, tau_l_ii), axis=1)

    W_tau = np.concatenate((tau_ul, tau_l), axis=0)

    W_tot = np.concatenate((W_tot, W_tau), axis=1)

    W_l_temp = np.zeros((len(W_e_l), 9))
    for k in range(9):
        W_l_temp[:, k] = W_e_l[
            :, (params_settings["which_body_loaded"]) * 10 + k
        ]  # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz
    W_upayload = np.concatenate(
        (np.zeros((len(W_base_ul), W_l_temp.shape[1])), -W_l_temp), axis=0
    )
    W_tot = np.concatenate((W_tot, W_upayload), axis=1)
    W_kpayload = np.concatenate(
        (
            np.zeros((len(W_base_ul), 1)),
            -W_e_l[:, (params_settings["which_body_loaded"]) * 10 + 9].reshape(
                len(W_e_l), 1
            ),
        ),
        axis=0,
    )  # the mass
    W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)

    V = np.transpose(Vh).conj()

    V_norm = params_settings["mass_load"] * np.divide(V[:, -1], V[-1, -1])

    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue
