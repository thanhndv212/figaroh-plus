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
    """This function builds the basic regressor of the 10(+4) parameters
    'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m'+ ('ia','fs','fv',
    'off') using pinocchio library depending on param.
    Input:  robot: (robot) a robot extracted from an urdf (for instance)
            q: (ndarray) (size robot.model.nq)
            v: (ndarray) (size robot.model.nv)
            a: (ndarray) (size robot.model.nv)
            param: (dict) a dictionnary setting the options
            tau : (ndarray) of stacked torque measurements
    Output: W_mod: (ndarray) basic regressor for 10(+4) parameters
    """
    # TODO : test phase with all the different cases between ia, fv+fs, off to
    # see if all have been correctly handled + add similiar code for external
    # wrench case (+ friction, ia,off,etc..)

    nsample = len(q)  # number of samples
    nb_in = len(robot.model.inertias) - 1
    # -1 if base link has inertia without external wrench, -1 for freeflyer
    njoints = robot.model.njoints - 1  # number of joints 
    nv = robot.model.nv  # degree of freedom
    add_col = 4

    # TODO: build regressor for  both joint torques/external wrenches.
    if param["is_joint_torques"]:
        W = np.zeros([nsample * nv, (10 + add_col) * njoints])
        W_mod = np.zeros([nsample * nv, (10 + add_col) * njoints])
        for i in range(nsample):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            assert (
                W_temp.shape[0] == nv
            ), "Torque Regressor's size does not match model.nv"
            for j in range(nv):
                W[j * nsample + i, 0 : 10 * njoints] = W_temp[j, :]
                for k in range(njoints):
                    if param["has_friction"]:
                        W[j * nsample + i, 10 * njoints + 2 * k] = v[i, j]  # fv
                        W[j * nsample + i, 10 * njoints + 2 * k + 1] = np.sign(
                            v[i, j]
                        )  # fs
                    else:
                        W[j * nsample + i, 10 * njoints + 2 * k] = 0  # fv
                        W[j * nsample + i, 10 * njoints + 2 * k + 1] = 0  # fs

                    if param["has_actuator_inertia"]:
                        W[j * nsample + i, 10 * njoints + 2 * njoints + k] = a[
                            i, j
                        ]  # ia
                    else:
                        W[j * nsample + i, 10 * njoints + 2 * njoints + k] = (
                            0  # ia
                        )

                    if param["has_joint_offset"]:
                        W[
                            j * nsample + i,
                            10 * njoints + 2 * njoints + njoints + k,
                        ] = 1  # off
                    else:
                        W[
                            j * nsample + i,
                            10 * njoints + 2 * njoints + njoints + k,
                        ] = 0  # off

        for k_ in range(njoints):
            W_mod[:, (10 + add_col) * k_ + 9] = W[:, 10 * k_ + 0]  # m
            W_mod[:, (10 + add_col) * k_ + 8] = W[:, 10 * k_ + 3]  # mz
            W_mod[:, (10 + add_col) * k_ + 7] = W[:, 10 * k_ + 2]  # my
            W_mod[:, (10 + add_col) * k_ + 6] = W[:, 10 * k_ + 1]  # mx
            W_mod[:, (10 + add_col) * k_ + 5] = W[:, 10 * k_ + 9]  # Izz
            W_mod[:, (10 + add_col) * k_ + 4] = W[:, 10 * k_ + 8]  # Iyz
            W_mod[:, (10 + add_col) * k_ + 3] = W[:, 10 * k_ + 6]  # Iyy
            W_mod[:, (10 + add_col) * k_ + 2] = W[:, 10 * k_ + 7]  # Ixz
            W_mod[:, (10 + add_col) * k_ + 1] = W[:, 10 * k_ + 5]  # Ixy
            W_mod[:, (10 + add_col) * k_ + 0] = W[:, 10 * k_ + 4]  # Ixx

            W_mod[:, (10 + add_col) * k_ + 10] = W[
                :, 10 * njoints + 2 * njoints + k_
            ]  # ia
            W_mod[:, (10 + add_col) * k_ + 11] = W[
                :, 10 * njoints + 2 * k_
            ]  # fv
            W_mod[:, (10 + add_col) * k_ + 12] = W[
                :, 10 * njoints + 2 * k_ + 1
            ]  # fs
            W_mod[:, (10 + add_col) * k_ + 13] = W[
                :, 10 * njoints + 2 * njoints + njoints + k_
            ]  # off

    elif param["is_external_wrench"]:
        ft = param["force_torque"]
        W = np.zeros([nsample * 6, (10 + add_col) * (nb_in)])
        for i in range(nsample):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for jj in range(len(ft)):
                if ft[jj] == "Fx":
                    j = 0
                    W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                elif ft[jj] == "Fy":
                    j = 1
                    W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                elif ft[jj] == "Fz":
                    j = 2
                    W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                elif ft[jj] == "Mx":
                    j = 3
                    W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                elif ft[jj] == "My":
                    j = 4
                    W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                elif ft[jj] == "Mz":
                    j = 5
                    W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                elif ft[jj] == "All":
                    for j in range(6):
                        W[j * nsample + i, 0 : 10 * nb_in] = W_temp[j, :]
                else:
                    raise ValueError("Please enter valid parameters")

            if param["has_friction"]:
                W[j * nsample + i, 10 * nb_in + 2 * j] = v[i, j]  # fv
                W[j * nsample + i, 10 * nb_in + 2 * j + 1] = np.sign(
                    v[i, j]
                )  # fs
            else:
                W[j * nsample + i, 10 * nb_in + 2 * j] = 0  # fv
                W[j * nsample + i, 10 * nb_in + 2 * j + 1] = 0  # fs

            if param["has_actuator_inertia"]:
                W[j * nsample + i, 10 * nb_in + 2 * nb_in + j] = a[i, j]  # ia
            else:
                W[j * nsample + i, 10 * nb_in + 2 * nb_in + j] = 0  # ia

            if param["has_joint_offset"]:
                W[j * nsample + i, 10 * nb_in + 2 * nb_in + nb_in + j] = (
                    1  # off
                )
            else:
                W[j * nsample + i, 10 * nb_in + 2 * nb_in + nb_in + j] = (
                    0  # off
                )

        W_mod = np.zeros([nsample * 6, (10 + add_col) * (nb_in)])

        if param["external_wrench_offsets"]:
            W_mod = np.zeros([nsample * 6, (10 + add_col) * (nb_in) + 3])

            if tau is not None:
                for jj_ in range(3, 6):
                    if jj_ == 3:
                        for ii in range(nsample):
                            W_mod[ii + jj_ * nsample, -2] = tau[ii + nsample]
                            W_mod[ii + jj_ * nsample, -1] = tau[ii + 2 * nsample]
                    if jj_ == 4:
                        for ii in range(nsample):
                            W_mod[ii + jj_ * nsample, -3] = tau[ii]
                            W_mod[ii + jj_ * nsample, -1] = tau[ii + 2 * nsample]
                    if jj_ == 5:
                        for ii in range(nsample):
                            W_mod[ii + jj_ * nsample, -3] = tau[ii]
                            W_mod[ii + jj_ * nsample, -2] = tau[ii + nsample]

        for k_ in range(nb_in):
            W_mod[:, (10 + add_col) * k_ + 9] = W[:, 10 * k_ + 0]  # m
            W_mod[:, (10 + add_col) * k_ + 8] = W[:, 10 * k_ + 3]  # mz
            W_mod[:, (10 + add_col) * k_ + 7] = W[:, 10 * k_ + 2]  # my
            W_mod[:, (10 + add_col) * k_ + 6] = W[:, 10 * k_ + 1]  # mx
            W_mod[:, (10 + add_col) * k_ + 5] = W[:, 10 * k_ + 9]  # Izz
            W_mod[:, (10 + add_col) * k_ + 4] = W[:, 10 * k_ + 8]  # Iyz
            W_mod[:, (10 + add_col) * k_ + 3] = W[:, 10 * k_ + 6]  # Iyy
            W_mod[:, (10 + add_col) * k_ + 2] = W[:, 10 * k_ + 7]  # Ixz
            W_mod[:, (10 + add_col) * k_ + 1] = W[:, 10 * k_ + 5]  # Ixy
            W_mod[:, (10 + add_col) * k_ + 0] = W[:, 10 * k_ + 4]  # Ixx

            W_mod[:, (10 + add_col) * k_ + 10] = W[
                :, 10 * nb_in + 2 * nb_in + k_
            ]  # ia
            W_mod[:, (10 + add_col) * k_ + 11] = W[:, 10 * nb_in + 2 * k_]  # fv
            W_mod[:, (10 + add_col) * k_ + 12] = W[
                :, 10 * nb_in + 2 * k_ + 1
            ]  # fs
            W_mod[:, (10 + add_col) * k_ + 13] = W[
                :, 10 * nb_in + 2 * nb_in + nb_in + k_
            ]  # off

    return W_mod


def add_coupling_TX40(W, model, data, nsample, nq, nv, njoints, q, v, a):
    """Dedicated function for Staubli TX40"""
    W = np.c_[W, np.zeros([W.shape[0], 3])]
    for i in range(nsample):
        # joint 5
        W[4 * nsample + i, W.shape[1] - 3] = a[i, 5]
        W[4 * nsample + i, W.shape[1] - 2] = v[i, 5]
        W[4 * nsample + i, W.shape[1] - 1] = np.sign(v[i, 4] + v[i, 5])
        # joint 6
        W[5 * nsample + i, W.shape[1] - 3] = a[i, 4]
        W[5 * nsample + i, W.shape[1] - 2] = v[i, 4]
        W[5 * nsample + i, W.shape[1] - 1] = np.sign(v[i, 4] + v[i, 5])

    return W


def eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
    """This function eliminates columns which has L2 norm smaller than
    tolerance.
    Input:  W: (ndarray) joint torque regressor
            params_std: (dict) standard parameters
            tol_e: (float) tolerance
    Output: W_e: (ndarray) reduced regressor
            params_r:   [list] corresponding parameters to columns of reduced
                        regressor"""
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
