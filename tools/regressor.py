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
