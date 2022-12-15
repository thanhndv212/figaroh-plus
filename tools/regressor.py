import pinocchio as pin
import numpy as np


def build_regressor_basic_pinocchio(N, robot, q, v, a):
    W = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
    for i in range(N):
        W_temp = pin.computeJointTorqueRegressor(
            robot.model, robot.data, q[i, :], v[i, :], a[i, :]
        )
        for j in range(W_temp.shape[0]):
            W[j * N + i, :] = W_temp[j, :]
    return W


def build_regressor_basic(N, robot, q, v, a):
    # TODO: reorgnize columns from ['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz']
    # to ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
    W = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
    for i in range(N):
        W_temp = pin.computeJointTorqueRegressor(
            robot.model, robot.data, q[i, :], v[i, :], a[i, :]
        )
        for j in range(W_temp.shape[0]):
            W[j * N + i, :] = W_temp[j, :]
    W_mod = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
    for k in range(robot.model.nv):
        W_mod[:, 10 * k + 9] = W[:, 10 * k + 0]  # m
        W_mod[:, 10 * k + 8] = W[:, 10 * k + 3]  # mz
        W_mod[:, 10 * k + 7] = W[:, 10 * k + 2]  # my
        W_mod[:, 10 * k + 6] = W[:, 10 * k + 1]  # mx
        W_mod[:, 10 * k + 5] = W[:, 10 * k + 9]  # Izz
        W_mod[:, 10 * k + 4] = W[:, 10 * k + 8]  # Iyz
        W_mod[:, 10 * k + 3] = W[:, 10 * k + 6]  # Iyy
        W_mod[:, 10 * k + 2] = W[:, 10 * k + 7]  # Ixz
        W_mod[:, 10 * k + 1] = W[:, 10 * k + 5]  # Ixy
        W_mod[:, 10 * k + 0] = W[:, 10 * k + 4]  # Ixx
    return W_mod

def build_regressor_basic_v2(robot, q, v, a, param, tau=None):
    """This function builds the basic regressor of the 10(+2) parameters
    'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m'+ ('fs','fv') using pinocchio
    library depending on param.
    Input:  robot: (robot) a robot extracted from an urdf (for instance)
            q: (ndarray) a configuration position vector (size robot.model.nq)
            v: (ndarray) a configuration velocity vector (size robot.model.nv)
            a: (ndarray) a configutation acceleration vectore (size robot.model.na)
            param: (dict) a dictionnary setting the options, i.e., here add two
            parameters, 'fs' and 'fv' if the flag 'has friction' is true
            tau : (ndarray) of stacked torque measurements (Fx,Fy,Fz), None if the torque offsets are not identified 
    Output: W_mod: (ndarray) basic regressor for 10(+2) parameters
    """
    err = False
    N = len(q) # nb of samples 
    nb_in=len(robot.model.inertias)
    if param["is_joint_torques"]:
        W = np.zeros([N * robot.model.nv, 10 * (nb_in-1)])
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for j in range(W_temp.shape[0]):
                W[j * N + i, :] = W_temp[j, :]

    if param["is_external_wrench"]:
        ft = param["force_torque"]
        W = np.zeros([N * 6, 10 * (nb_in-1)]) 
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for k in range(len(ft)):
                if ft[k] == "Fx":
                    j = 0
                    W[j * N + i, :] = W_temp[j, :]
                elif ft[k] == "Fy":
                    j = 1
                    W[j * N + i, :] = W_temp[j, :]
                elif ft[k] == "Fz":
                    j = 2
                    W[j * N + i, :] = W_temp[j, :]
                elif ft[k] == "Mx":
                    j = 3
                    W[j * N + i, :] = W_temp[j, :]
                elif ft[k] == "My":
                    j = 4
                    W[j * N + i, :] = W_temp[j, :]
                elif ft[k] == "Mz":
                    j = 5
                    W[j * N + i, :] = W_temp[j, :]
                elif ft[k] == "All":
                    for j in range(6):
                        W[j * N + i, :] = W_temp[j, :]
                else:
                    err = True
        if err:
            raise SyntaxError("Please enter valid parameters")

    add_col = 0

    if param["has_friction"]:  # adds two other parameters fs and fv
        add_col += 2

    if param["is_joint_torques"]:
        W_mod = np.zeros([N * robot.model.nv, (10 + add_col) * nb_in])

    if param["is_external_wrench"]:
        W_mod = np.zeros([N * 6, (10 + add_col) * (nb_in-1)])
    
    if param["is_external_wrench"] and param["external_wrench_offsets"]:  # adds OFFX, OFFY, OFFZ
        W_mod = np.zeros([N * 6, (10 + add_col) * (nb_in-1) + 3])

    if param["is_joint_torques"]:
        for k in range(nb_in-1):
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

            if param[
                "has_friction"
            ]:  # builds the regressor for the augmented parameters
                W_mod = np.c_[W_mod, np.zeros([N * robot.model.nv, 2 * nb_in])]
                for ii in range(N):
                    W_mod[ii + k * N, (10 + add_col) * k + 10] = v[ii, k]  # fv
                    W_mod[ii + k * N, (10 + add_col) * k + 11] = np.sign(v[ii, k])  # fs

    if param["is_external_wrench"]:
        for k in range(nb_in-1):
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

            if param[
                "has_friction"
            ]:  # builds the regressor for the augmented parameters
                W_mod = np.c_[
                    W_mod,
                    np.zeros([N * 6, 2 * (nb_in-1)]),
                ]
                for ii in range(N):
                    W_mod[ii + k * N, (10 + add_col) * k + 10] = v[ii, k]  # fv
                    W_mod[ii + k * N, (10 + add_col) * k + 11] = np.sign(v[ii, k])  # fs

    if param["external_wrench_offsets"] and tau is not None:
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

    return W_mod



# TODO: clean up this mess
# def build_regressor_w_friction(model, data, N, nq, nv, njoints, q, v, a):
#     W = np.zeros([N * nv, 10 * nv + 2 * nv])
#     for i in range(N):
#         W_temp = pin.computeJointTorqueRegressor(
#             model, data, q[i, :], v[i, :], a[i, :])
#         for j in range(W_temp.shape[0]):
#             W[j * N + i, 0:10 * nv] = W_temp[j, :]
#             W[j * N + i, 10 * nv + 2 * j] = v[i, j]
#             W[j * N + i, 10 * nv + 2 * j + 1] = np.sign(v[i, j])
#     return W


# def build_regressor_full(model, data, N, nq, nv, njoints, q, v, a):
#     W = np.zeros([N * nv, 10 * nv + 2 * nv + nv + nv])

#     for i in range(N):
#         W_temp = pin.computeJointTorqueRegressor(
#             model, data, q[i, :], v[i, :], a[i, :])
#         for j in range(W_temp.shape[0]):
#             W[j * N + i, 0:10 * nv] = W_temp[j, :]
#             W[j * N + i, 10 * nv + 2 * j] = v[i, j]
#             W[j * N + i, 10 * nv + 2 * j + 1] = np.sign(v[i, j])
#             W[j * N + i, 10 * nv + 2 * nv + j] = a[i, j]
#             W[j * N + i, 10 * nv + 2 * nv + nv + j] = 1
#     return W


def build_regressor_full_modified(model, data, N, nq, nv, njoints, q, v, a):
    W = np.zeros([N * nv, 10 * nv + 2 * nv + nv + nv])

    for i in range(N):
        W_temp = pin.computeJointTorqueRegressor(
            model, data, q[i, :], v[i, :], a[i, :])
        for j in range(W_temp.shape[0]):
            W[j * N + i, 0: 10 * nv] = W_temp[j, :]
            W[j * N + i, 10 * nv + 2 * j] = v[i, j]  # fv
            W[j * N + i, 10 * nv + 2 * j + 1] = np.sign(v[i, j])  # fs
            W[j * N + i, 10 * nv + 2 * nv + j] = a[i, j]  # ia
            W[j * N + i, 10 * nv + 2 * nv + nv + j] = 1  # off
    W_mod = np.zeros([N * nv, 10 * nv + 2 * nv + nv + nv])
    for k in range(nv):
        W_mod[:, 14 * k + 10] = W[:, 10 * nv + 2 * nv + k]  # ia
        W_mod[:, 14 * k + 11] = W[:, 10 * nv + 2 * k]  # fv
        W_mod[:, 14 * k + 12] = W[:, 10 * nv + 2 * k + 1]  # fs
        W_mod[:, 14 * k + 13] = W[:, 10 * nv + 2 * nv + nv + k]  # off
        W_mod[:, 14 * k + 9] = W[:, 10 * k + 0]  # m
        W_mod[:, 14 * k + 8] = W[:, 10 * k + 3]  # mz
        W_mod[:, 14 * k + 7] = W[:, 10 * k + 2]  # my
        W_mod[:, 14 * k + 6] = W[:, 10 * k + 1]  # mx
        W_mod[:, 14 * k + 5] = W[:, 10 * k + 9]  # Izz
        W_mod[:, 14 * k + 4] = W[:, 10 * k + 8]  # Iyz
        W_mod[:, 14 * k + 3] = W[:, 10 * k + 6]  # Iyy
        W_mod[:, 14 * k + 2] = W[:, 10 * k + 7]  # Ixz
        W_mod[:, 14 * k + 1] = W[:, 10 * k + 5]  # Ixy
        W_mod[:, 14 * k + 0] = W[:, 10 * k + 4]  # Ixx
    return W_mod


# def add_friction(W, model, data, N, nq, nv, njoints, q, v, a):
#     # TODO: break the model modular
#     W = np.c_[W, np.zeros([W.shape[0], 2 * nv])]
#     for i in range(N):
#         for j in range(nv):
#             W[j * N + i, W.shape[1] + 2 * j] = v[i, j]
#             W[j * N + i, W.shape[1] + 2 * j + 1] = np.sign(v[i, j])
#     pass


# def add_motor_inertia(W, model, data, N, nq, nv, njoints, q, v, a):
#     # TODO: break the model modular
#     W = np.c_[W, np.zeros([W.shape[0], 2 * nv])]
#     for i in range(N):
#         for j in range(nv):
#             W[j * N + i, W.shape[1] + j] = a[i, j]
#             W[j * N + i, W.shape[1] + nv + j] = 1
# till here


def add_coupling(W, model, data, N, nq, nv, njoints, q, v, a):
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


def get_index_eliminate(W, params_std, tol_e):
    col_norm = np.diag(np.dot(W.T, W))
    idx_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
        else:
            params_r.append(list(params_std.keys())[i])
    # idx_e = tuple(idx_e)
    return idx_e, params_r


def build_regressor_reduced(W, idx_e):
    W_e = np.delete(W, idx_e, 1)
    return W_e
