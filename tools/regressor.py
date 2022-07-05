import pinocchio as pin
import numpy as np


def set_missing_urdf_parameters(robot, param):
    """This function append the param dictionnary with additionnal parameters that are
    required for the generation of optimal exciting model but that are not present in
    the URDF.
    Output: params : appended dictionary of parameter names and their values"""

    # Set model specific parameters - use default values in case some fields are blank
    # in the URDF
    param["q0"] = np.array(robot.q0)
    diff_limit = np.setdiff1d(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    print(diff_limit)
    if not np.sum(
        diff_limit
    ):  # upper and lower joint limits are the same so use defaut values for all joints
        print("No joint limits. Set default values")
        for ii in range(robot.model.nq):
            robot.model.lowerPositionLimit[ii] = -param["q_lim_def"]
            robot.model.upperPositionLimit[ii] = param["q_lim_def"]

    if np.sum(robot.model.velocityLimit) == 0:
        print("No velocity limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.velocityLimit[ii] = param["dq_lim_def"]

    if np.sum(robot.model.velocityLimit) == 0:
        print("No joint torque limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.effortLimit[ii] = -param["tau_lim_def"]

    accelerationLimit = np.zeros(robot.model.nq)
    for ii in range(robot.model.nq):
        accelerationLimit[ii] = param[
            "ddq_lim_def"
        ]  # accelerationLimit to be consistent with PIN naming
    param["accelerationLimit"] = accelerationLimit
    # print(model.accelerationLimit)

    if param["has_friction"]:

        for ii in range(robot.model.nv):
            if ii == 0:
                fv = [
                    (ii + 1) / 10
                ]  # default values of the joint viscous friction in case they are used
                fs = [
                    (ii + 1) / 10
                ]  # default value of the joint static friction in case they are used
            else:
                fv.append((ii + 1) / 10)
                fs.append((ii + 1) / 10)

        param["fv"] = fv
        param["fs"] = fs

    return param


def get_standard_parameters(robot, param):
    """This function prints out the standard inertial parameters extracted from the urdf
    model.
    Input: robot : (robot) a robot extracted from an urdf (for instance)
           param : ([list]) a dictionnary of str to set the options
    Output: params_std: ([list]) a dictionary of parameter names (str) and their values
    (float)"""
    model = robot.model
    phi = []
    params = []

    # manual input addtiional parameters
    # example TX40
    # self.fv = (8.05e0, 5.53e0, 1.97e0, 1.11e0, 1.86e0, 6.5e-1)
    # self.fs = (7.14e0, 8.26e0, 6.34e0, 2.48e0, 3.03e0, 2.82e-1)
    # self.Ia = (3.62e-1, 3.62e-1, 9.88e-2, 3.13e-2, 4.68e-2, 1.05e-2)
    # self.off = (3.92e-1, 1.37e0, 3.26e-1, -1.02e-1, -2.88e-2, 1.27e-1)
    # self.Iam6 = 9.64e-3
    # self.fvm6 = 6.16e-1
    # self.fsm6 = 1.95e0
    # self.N1 = 32
    # self.N2 = 32
    # self.N3 = 45
    # self.N4 = -48
    # self.N5 = 45
    # self.N6 = 32
    # self.qd_lim = 0.01 * \
    #     np.array([287, 287, 430, 410, 320, 700]) * np.pi / 180
    # self.ratio_essential = 30

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

    # change order of values in phi['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz',
    # 'Iyz','Izz'] - from pinoccchio
    # corresponding to params_name
    # ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
    for i in range(1, model.njoints):
        P = model.inertias[i].toDynamicParameters()
        P_mod = np.zeros(P.shape[0])
        P_mod[9] = P[0]  # m
        P_mod[8] = P[3]  # mz
        P_mod[7] = P[2]  # my
        P_mod[6] = P[1]  # mx
        P_mod[5] = P[9]  # Izz
        P_mod[4] = P[8]  # Iyz
        P_mod[3] = P[6]  # Iyy
        P_mod[2] = P[7]  # Ixz
        P_mod[1] = P[5]  # Ixy
        P_mod[0] = P[4]  # Ixx
        for j in params_name:
            if not param["is_external_wrench"]:  # self.isFext:
                params.append(j + str(i))
            else:
                params.append(j + str(i - 1))
        for k in P_mod:
            phi.append(k)
        # if param['hasActuatorInertia']:
        # phi.extend([self.Ia[i - 1]])
        # params.extend(["Ia" + str(i)])
        if param["has_friction"]:

            phi.extend([param["fv"][i - 1], param["fv"][i - 1]])
            params.extend(["fv" + str(i), "fs" + str(i)])
            # phi.extend([self.fv[i - 1], self.fs[i - 1]])
            # params.extend(["fv" + str(i), "fs" + str(i)])
        # if param['hasJointOffset']:
        # phi.extend([self.off[i - 1]])
        # params.extend(["off" + str(i)])
    # if param['hasCoupledWrist']:#self.isCoupling:
    # phi.extend([self.Iam6, self.fvm6, self.fsm6])
    # params.extend(["Iam6", "fvm6", "fsm6"])
    params_std = dict(zip(params, phi))
    return params_std


def build_regressor_basic_pinocchio(N, robot, q, v, a):
    """This function builds the basic regressor of the 10 parameters
    'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m' using pinocchio library
    Input:  N: (int) number of samples
            robot: (robot) a robot extracted from an urdf (for instance)
            q: (ndarray) a configuration position vector (size robot.model.nq)
            v: (ndarray) a configuration velocity vector (size robot.model.nv)
            a: (ndarray) a configutation acceleration vectore (size robot.model.na)
    Output: W: (ndarray) basic regressor
    """
    W = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
    for i in range(N):
        W_temp = pin.computeJointTorqueRegressor(
            robot.model, robot.data, q[i, :], v[i, :], a[i, :]
        )
        for j in range(W_temp.shape[0]):
            W[j * N + i, :] = W_temp[j, :]
    return W


def build_regressor_basic(robot, q, v, a, param):
    """This function builds the basic regressor of the 10(+2) parameters
    'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m'+ ('fs','fv') using pinocchio
    library depending on param.
    Input:  robot: (robot) a robot extracted from an urdf (for instance)
            q: (ndarray) a configuration position vector (size robot.model.nq)
            v: (ndarray) a configuration velocity vector (size robot.model.nv)
            a: (ndarray) a configutation acceleration vectore (size robot.model.na)
            param: (dict) a dictionnary setting the options, i.e., here add two
            parameters, 'fs' and 'fv' if the flag 'has friction' is true
    Output: W_mod: (ndarray) basic regressor for 10(+2) parameters
    """
    err = False
    N = len(q)
    # TODO: reorgnize columns from ['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz',
    # 'Iyz','Izz']
    # to ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
    if param["is_joint_torques"]:
        W = np.zeros([N * robot.model.nv, 10 * robot.model.nv])
        for i in range(N):
            W_temp = pin.computeJointTorqueRegressor(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )
            for j in range(W_temp.shape[0]):
                W[j * N + i, :] = W_temp[j, :]

    if param["is_external_wrench"]:
        ft = param["force_torque"]
        W = np.zeros([N * 6, 10 * (robot.model.nv - 5)])
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
        add_col = 2

    if param["is_joint_torques"]:
        W_mod = np.zeros([N * robot.model.nv, (10 + add_col) * robot.model.nv])

    if param["is_external_wrench"]:
        W_mod = np.zeros([N * 6, (10 + add_col) * (robot.model.nv - 5)])

    if param["is_joint_torques"]:
        for k in range(robot.model.nv):
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
                W_mod = np.c_[W_mod, np.zeros([N * robot.model.nq, 2 * robot.model.nq])]
                for ii in range(N):
                    W_mod[ii + k * N, (10 + add_col) * k + 10] = v[ii, k]  # fv
                    W_mod[ii + k * N, (10 + add_col) * k + 11] = np.sign(v[ii, k])  # fs

    if param["is_external_wrench"]:
        for k in range(robot.model.nv - 5):
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
                    np.zeros([N * (robot.model.nq - 7), 2 * (robot.model.nq - 7)]),
                ]
                for ii in range(N):
                    W_mod[ii + k * N, (10 + add_col) * k + 10] = v[ii, k]  # fv
                    W_mod[ii + k * N, (10 + add_col) * k + 11] = np.sign(v[ii, k])  # fs
    return W_mod


def build_regressor_full_modified(model, data, N, nq, nv, njoints, q, v, a):
    """This function builds the full regressor of the 14 parameters
    'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m','off','fs','fv','ia' using
    pinocchio library.
    Input:  model: (model) model of a given robot
            data: (data) data of a given robot
            N: (int) number of robot body segments
            nv:(int) shape of the velocity vector
            q: (ndarray) a configuration position vector (size robot.model.nq)
            v: (ndarray) a configuration velocity vector (size robot.model.nv)
            a: (ndarray) a configutation acceleration vectore (size robot.model.na)
    Output:  W_mod: (ndarray) full regressor for the 14 parameters
    """
    W = np.zeros(
        [N * nv, 10 * nv + 2 * nv + nv + nv]
    )  # add two other new parameters ia and off

    for i in range(N):
        W_temp = pin.computeJointTorqueRegressor(model, data, q[i, :], v[i, :], a[i, :])
        for j in range(W_temp.shape[0]):
            W[j * N + i, 0 : 10 * nv] = W_temp[j, :]
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
    """This function ?.
    Input:  W: (ndarray) joint torque regressor
            N: (int) number of robot body segments
            v: (ndarray) a configuration velocity vector (size robot.model.nv)
            a: (ndarray) a configutation acceleration vectore (size robot.model.na)
    Output: W: (ndarray)
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
    """This function get the indexes of all the parameters whose L2 norm is smaller than
    tolerance.
    Input:  W: (ndarray) joint torque regressor
            params_std: (dict) standard parameters
            tol_e: (float) tolerance
    Output: idx_e: [list(int)] list of indexes of the eliminated parameters
            params_r: [list] corresponding parameters to columns of reduced regressor
    """
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
    """This function eliminates columns whose indexes are given by the indexes of idx_e
    Input:  W: (ndarray) joint torque regressor
            idx_e: ([list(int)]) list of indexes to eliminate
    Output: W_e: (ndarray) reduced regressor
    """
    W_e = np.delete(W, idx_e, 1)
    return W_e
