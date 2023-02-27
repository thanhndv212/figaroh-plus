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
    
    N = len(q) # nb of samples 
    nb_in=len(robot.model.inertias)-1 # -1 if base link has inertia without external wrench, else -1 for freeflyer
    nv=robot.model.nv

    add_col = 4
    #TODO: build regressor for the case of both joint torques and external wrenches. 
    if param["is_joint_torques"]:
        W = np.zeros([N*nv, (10+add_col)*nv])
        W_mod = np.zeros([N*nv, (10+add_col)*nv])
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
        W = np.zeros([N * 6, (10 + add_col) * (nb_in)])
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for k in range(len(ft)):
                if ft[k] == "Fx":
                    j = 0
                    W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                elif ft[k] == "Fy":
                    j = 1
                    W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                elif ft[k] == "Fz":
                    j = 2
                    W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                elif ft[k] == "Mx":
                    j = 3
                    W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                elif ft[k] == "My":
                    j = 4
                    W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                elif ft[k] == "Mz":
                    j = 5
                    W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                elif ft[k] == "All":
                    for j in range(6):
                        W[j * N + i,  0 : 10 * nb_in] = W_temp[j, :]
                else:
                    raise ValueError("Please enter valid parameters")

            if param["has_friction"]:
                W[j * N + i, 10 * nb_in + 2 * j] = v[i, j]  # fv
                W[j * N + i, 10 * nb_in + 2 * j + 1] = np.sign(v[i, j])  # fs
            else:
                W[j * N + i, 10 * nb_in + 2 * j] = 0  # fv
                W[j * N + i, 10 * nb_in + 2 * j + 1] = 0  # fs

            if param["has_actuator_inertia"]:
                W[j * N + i, 10 * nb_in + 2 * nb_in + j] = a[i, j]  # ia
            else:
                W[j * N + i, 10 * nb_in + 2 * nb_in + j] = 0  # ia

            if param["has_joint_offset"]:
                W[j * N + i, 10 * nb_in + 2 * nb_in + nb_in + j] = 1  # off
            else:
                W[j * N + i, 10 * nb_in + 2 * nb_in + nb_in + j] = 0  # off

        W_mod = np.zeros([N * 6, (10 + add_col) * (nb_in)])
        
        if param["external_wrench_offsets"]:
            W_mod = np.zeros([N * 6, (10 + add_col) * (nb_in) + 3])

            if tau is not None:
                for k in range(3,6):
                    if k == 3:  
                        for ii in range(N):
                            W_mod[ii + k * N, -2] = tau[ii+N]
                            W_mod[ii + k * N, -1] = tau[ii+2*N]
                    if k == 4 :
                        for ii in range(N):
                            W_mod[ii + k * N, -3] = tau[ii]
                            W_mod[ii + k * N, -1] = tau[ii+2*N]
                    if k == 5:
                        for ii in range(N):
                            W_mod[ii + k * N, -3] = tau[ii]
                            W_mod[ii + k * N, -2] = tau[ii+N]

        for k in range(nb_in):
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

            W_mod[:, (10 + add_col) * k + 10] = W[:, 10 * nb_in + 2 * nb_in + k]  # ia
            W_mod[:, (10 + add_col) * k + 11] = W[:, 10 * nb_in + 2 * k]  # fv
            W_mod[:, (10 + add_col) * k + 12] = W[:, 10 * nb_in + 2 * k + 1]  # fs
            W_mod[:, (10 + add_col) * k + 13] = W[:, 10 * nb_in + 2 * nb_in + nb_in + k]  # off

    return W_mod

def add_actuator_inertia(W, robot, q, v, a, param):
    N = len(q) # nb of samples 
    nv = robot.model.nv
    add_col = 4
    for k in range(nv):
        W[:, (10 + add_col) * k + 10] = a[i, j]
    return W

def add_friction(W,robot, q, v, a, param):
    N = len(q) # nb of samples 
    nv = robot.model.nv
    add_col = 4
    for k in range(nv):
        W[:, (10 + add_col) * k + 11] = v[i, j]
        W[:, (10 + add_col) * k + 12] = np.sign(v[i, j])
    return W

def add_joint_offset(W, robot, q, v, a, param):
    N = len(q) # nb of samples 
    nv = robot.model.nv
    add_col = 4
    for k in range(nv):
        W[:, (10 + add_col) * k + 13] = 1
    return W

def add_coupling_TX40(W, model, data, N, nq, nv, njoints, q, v, a):
    """ Dedicated function for Staubli TX40
    """
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
