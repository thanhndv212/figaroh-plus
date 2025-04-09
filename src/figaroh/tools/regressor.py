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


def build_regressor_basic(robot, q, v, a, param, tau=None):
    """Build basic regressor for dynamic parameters.

    Args:
        robot: Robot model object
        q: Configuration position array
        v: Configuration velocity array
        a: Configuration acceleration array
        param: Dictionary of options controlling which parameters to include
        tau: Optional torque measurements array for torque offsets

    Returns:
        ndarray: Basic regressor matrix for standard parameters
    """
    N = len(q)  # Number of samples

    id_inertias = [
        jj for jj in range(len(robot.model.inertias.tolist()))
        if robot.model.inertias.tolist()[jj].mass != 0
    ]

    nb_in_total = len(robot.model.inertias) - 1  # Number of inertias
    nv = robot.model.nv
    add_col = 4

    if param["is_joint_torques"]:
        W = np.zeros([N * nv, (10 + add_col) * nv])
        W_mod = np.zeros([N * nv, (10 + add_col) * nv])
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for j in range(W_temp.shape[0]):
                W[j * N + i, 0:10 * nv] = W_temp[j, :]

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
                        W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10:10 * idx_in
                        ]
                elif ft[k] == "Fy":
                    j = 1
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10:10 * idx_in
                        ]
                elif ft[k] == "Fz":
                    j = 2
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10:10 * idx_in
                        ]
                elif ft[k] == "Mx":
                    j = 3
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10:10 * idx_in
                        ]
                elif ft[k] == "My":
                    j = 4
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10:10 * idx_in
                        ]
                elif ft[k] == "Mz":
                    j = 5
                    for idx_in in id_inertias:
                        W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                            j, (idx_in - 1) * 10:10 * idx_in
                        ]
                elif ft[k] == "All":
                    for j in range(6):
                        for idx_in in id_inertias:
                            W[j * N + i, (idx_in - 1) * 10:10 * idx_in] = W_temp[
                                j, (idx_in - 1) * 10:10 * idx_in
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



def add_coupling_TX40(W, model, data, N, nq, nv, njoints, q, v, a):
    """Dedicated function for Staubli TX40.

    Args:
        W: Input regressor matrix
        model: Robot model
        data: Robot data
        N: Number of samples
        nq: Number of positions
        nv: Number of velocities
        njoints: Number of joints
        q: Joint positions
        v: Joint velocities
        a: Joint accelerations

    Returns:
        ndarray: Updated regressor matrix
    """
    W = np.c_[W, np.zeros([W.shape[0], 3])]
    for i in range(N):
        # Joint 5
        W[4 * N + i, W.shape[1] - 3] = a[i, 5]
        W[4 * N + i, W.shape[1] - 2] = v[i, 5]
        W[4 * N + i, W.shape[1] - 1] = np.sign(v[i, 4] + v[i, 5])
        # Joint 6
        W[5 * N + i, W.shape[1] - 3] = a[i, 4]
        W[5 * N + i, W.shape[1] - 2] = v[i, 4]
        W[5 * N + i, W.shape[1] - 1] = np.sign(v[i, 4] + v[i, 5])

    return W


def eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
    """Eliminate columns with L2 norm smaller than tolerance.

    Args:
        W: Joint torque regressor matrix
        params_std: Standard parameters dictionary
        tol_e: Tolerance value

    Returns:
        tuple:
            - Reduced regressor matrix
            - List of parameters corresponding to reduced regressor columns
    """
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
    """Get indices of columns to eliminate based on tolerance.

    Args:
        W: Joint torque regressor matrix
        params_std: Standard parameters dictionary
        tol_e: Tolerance value

    Returns:
        tuple:
            - List of indices to eliminate
            - List of remaining parameters
    """
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
    """Build reduced regressor matrix.

    Args:
        W: Input regressor matrix
        idx_e: Indices of columns to eliminate

    Returns:
        ndarray: Reduced regressor matrix
    """
    W_e = np.delete(W, idx_e, 1)
    return W_e


def build_total_regressor_current(
    W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, param
):
    """Build regressor for total least squares with current measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        I_u: Joint currents in unloaded case
        I_l: Joint currents in loaded case
        param_standard_l: Standard parameters in loaded case
        param: Dictionary of settings

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized parameter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    nb_joints = int(len(I_u) / param["nb_samples"])
    n_samples = param["nb_samples"]

    V_a = np.concatenate([
        I_u[:n_samples].reshape(n_samples, 1),
        np.zeros(((nb_joints - 1) * n_samples, 1))
    ], axis=0)

    V_b = np.concatenate([
        I_l[:n_samples].reshape(n_samples, 1),
        np.zeros(((nb_joints - 1) * n_samples, 1))
    ], axis=0)

    for ii in range(1, nb_joints):
        V_a_ii = np.concatenate([
            np.zeros((n_samples * ii, 1)),
            I_u[n_samples * ii:(ii + 1) * n_samples].reshape(n_samples, 1),
            np.zeros((n_samples * (nb_joints - ii - 1), 1))
        ], axis=0)
        V_b_ii = np.concatenate([
            np.zeros((n_samples * ii, 1)),
            I_l[n_samples * ii:(ii + 1) * n_samples].reshape(n_samples, 1),
            np.zeros((n_samples * (nb_joints - ii - 1), 1))
        ], axis=0)
        V_a = np.concatenate((V_a, V_a_ii), axis=1)
        V_b = np.concatenate((V_b, V_b_ii), axis=1)

    W_current = np.concatenate((V_a, V_b), axis=0)
    W_tot = np.concatenate((W_tot, W_current), axis=1)

    if param["has_friction"]:
        W_l_temp = np.zeros((len(W_l), 12))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
            W_l_temp[:, k] = W_l[:, (param["which_body_loaded"]) * 12 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(W_l_temp, param_standard_l, 1e-6)
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
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    elif param["has_actuator_inertia"]:
        W_l_temp = np.zeros((len(W_l), 14))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]:
            W_l_temp[:, k] = W_l[:, (param["which_body_loaded"]) * 14 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(W_l_temp, param_standard_l, 1e-6)
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
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    else:
        W_l_temp = np.zeros((len(W_l), 9))
        for k in range(9):
            W_l_temp[:, k] = W_l[:, (param["which_body_loaded"]) * 10 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(W_l_temp, param_standard_l, 1e-6)
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
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = param["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue


def build_total_regressor_wrench(
    W_b_u, W_b_l, W_l, tau_u, tau_l, param_standard_l, param
):
    """Build regressor for total least squares with external wrench measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        tau_u: External wrench in unloaded case
        tau_l: External wrench in loaded case
        param_standard_l: Standard parameters in loaded case
        param: Dictionary of settings

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized parameter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    tau_meast_ul = np.reshape(tau_u, (len(tau_u), 1))
    tau_meast_l = np.reshape(tau_l, (len(tau_l), 1))

    nb_samples_ul = int(len(tau_meast_ul) / 6)
    nb_samples_l = int(len(tau_meast_l) / 6)

    tau_ul = np.concatenate([
        tau_meast_ul[:nb_samples_ul],
        np.zeros((len(tau_meast_ul) - nb_samples_ul, 1))
    ], axis=0)
    
    tau_l = np.concatenate([
        tau_meast_l[:nb_samples_l],
        np.zeros((len(tau_meast_l) - nb_samples_l, 1))
    ], axis=0)

    for ii in range(1, 6):
        tau_ul_ii = np.concatenate([
            np.concatenate([
                np.zeros((nb_samples_ul * ii, 1)),
                tau_meast_ul[
                    nb_samples_ul * ii:(ii + 1) * nb_samples_ul
                ]
            ], axis=0),
            np.zeros((nb_samples_ul * (5 - ii), 1))
        ], axis=0)

        tau_l_ii = np.concatenate([
            np.concatenate([
                np.zeros((nb_samples_l * ii, 1)),
                tau_meast_l[
                    nb_samples_l * ii:(ii + 1) * nb_samples_l
                ]
            ], axis=0),
            np.zeros((nb_samples_l * (5 - ii), 1))
        ], axis=0)

        tau_ul = np.concatenate((tau_ul, tau_ul_ii), axis=1)
        tau_l = np.concatenate((tau_l, tau_l_ii), axis=1)

    W_tau = np.concatenate((tau_ul, tau_l), axis=0)
    W_tot = np.concatenate((W_tot, W_tau), axis=1)

    W_l_temp = np.zeros((len(W_l), 9))
    for k in range(9):
        W_l_temp[:, k] = W_l[
            :, (param["which_body_loaded"]) * 10 + k
        ]
    W_upayload = np.concatenate(
        (np.zeros((len(W_l), W_l_temp.shape[1])), -W_l_temp),
        axis=0
    )
    W_tot = np.concatenate((W_tot, W_upayload), axis=1)
    
    W_kpayload = np.concatenate([
        np.zeros((len(W_l), 1)),
        -W_l[:, param["which_body_loaded"] * 10 + 9].reshape(len(W_l), 1)
    ], axis=0)
    W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = param["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue
