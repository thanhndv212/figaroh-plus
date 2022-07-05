import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from tabulate import tabulate
import numpy as np

# from pinocchio.utils import *
from os.path import dirname, join, abspath
from scipy import signal


from tools.regressor import build_regressor_reduced, get_index_eliminate


def remove_zero_crossing_velocity(robot, W, tau):
    # eliminate qd crossing zero
    for i in range(len(W)):
        idx_qd_cross_zero = []
        for j in range(W[i].shape[0]):
            if abs(W[i][j, i * 14 + 11]) < robot.qd_lim[i]:  # check columns of fv_i
                idx_qd_cross_zero.append(j)
        # if i == 4 or i == 5:  # joint 5 and 6
        # for k in range(W_list[i].shape[0]):
        # # check sum cols of fv_5 + fv_6
        # if (
        # abs(W_list[i][k, 4 * 14 + 11] + W_list[i][k, 5 * 14 + 11])
        # < qd_lim[4] + qd_lim[5]
        # ):
        # idx_qd_cross_zero.append(k)
        # indices with vels around zero
        idx_eliminate = list(set(idx_qd_cross_zero))
        W[i] = np.delete(W[i], idx_eliminate, axis=0)
        W[i] = np.delete(tau[i], idx_eliminate, axis=0)
        print(W[i].shape, tau[i].shape)


def relative_stdev(W_b, phi_b, tau):
    """Calculates relative standard deviation of estimated parameters using the residual
    errro[Pressé & Gautier 1991]"""
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
    """This function computes the weigthed least square solution of the identification
    problem see [Gautier, 1997] for details inputs:
    - robot: pinocchio robot structure
    - W_b: base regressor matrix
    - tau: measured joint torques
    """
    # Needs to be modified for taking into account the GRFM
    sigma = np.zeros(robot.model.nq)
    # zero_identity_matrix = np.identity(len(tau_meas))
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

        # sig_ro_joint[ii] = np.linalg.norm(
        # tau[ii] - np.dot(W_b[a : (a + tau[ii]), :], phi_b)
        # ) ** 2 / (tau[ii])
        # diag_SIGMA[a: (a + tau[ii])] = np.full(    tau[ii], sig_ro_joint[ii])
        # a += tau[ii]
    # SIGMA = np.diag(diag_SIGMA)

    # Covariance matrix
    # (W^T*SIGMA^-1*W)^-1
    # C_X = np.linalg.inv(np.matmul(np.matmul(W_b.T, np.linalg.inv(SIGMA)), W_b))
    # WLS solution
    # (W^T*SIGMA^-1*W)^-1*W^T*SIGMA^-1*TAU
    # phi_b = np.matmul(np.matmul(np.matmul(C_X, W_b.T), np.linalg.inv(SIGMA)), tau)
    phi_b = np.around(phi_b, 6)

    return phi_b


def calculate_first_second_order_differentiation(data, param):
    print(data.shape)
    # calculate vel and acc by central difference
    dt = param["ts"]
    ddata = np.zeros([data.shape[0], data.shape[1]])
    dddata = np.zeros([data.shape[0], data.shape[1]])
    # t = np.linspace(0, data.shape[0], num=data.shape[0]) * param["ts"]
    for ii in range(data.shape[1]):
        ddata[:, ii] = np.gradient(data[:, ii], edge_order=1) / dt
        dddata[:, ii] = np.gradient(ddata[:, ii], edge_order=1) / dt

    return ddata, dddata


def low_pass_filter_data(data, param):
    """This function filters and elaborates data used in the identification process.
    It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes,
    France)"""

    # Apply median filter order 3 and then butterworth zerophase filtering order 4
    nbutter = 4

    b, a = signal.butter(
        nbutter, param["ts"] * param["cut_off_frequency_butterworth"] / 2, "low"
    )

    data = signal.medfilt(data, 3)
    data = signal.filtfilt(
        b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)
    )

    # suppress end segments of samples due to the border effect
    nbord = 5 * nbutter
    print(nbord)
    data = np.delete(data, np.s_[0:nbord], axis=0)
    data = np.delete(data, np.s_[(data.shape[0] - nbord) : data.shape[0]], axis=0)

    return data


def load_model(robotname, robot_urdf_file):
    """This function create a robot model and its data by inputting a URDF file that
    describes the robot.
    Input: 	robotname: directory containing model of robot
            robot_urdf_file: URDF file of robot
    Output: robot: robot model and data created by Pinocchio"""

    pinocchio_model_dir = join(
        dirname(dirname(str(abspath(__file__)))), "identification_toolbox/models"
    )
    model_path = join(pinocchio_model_dir, "others/robots")
    mesh_dir = model_path
    urdf_filename = robot_urdf_file
    urdf_dir = robotname + "/urdf"
    urdf_model_path = join(join(model_path, urdf_dir), urdf_filename)
    robot = RobotWrapper.BuildFromURDF(
        urdf_model_path, mesh_dir
    )  # ,pin.JointModelFreeFlyer())

    return robot


# inertial parameters of link2 from urdf model


def buildAugmentedRegressor(W_b_u, W_l, W_b_l, tau_u, tau_l, param):
    """Inputs:  W_b_u  base regressor for unloaded case
            W_b_l:  base Regressor for loaded case
            W_l: Full  regressor for loaded case
            I_u: measured current in uloaded case in A
            I_l: measured current in loaded case in A
    Ouputs: W_tot: total regressor matrix
            V_norm= Normalised solution vector"""

    # augmented regressor matrix

    tau = np.concatenate((tau_u, tau_l), axis=0)

    W = np.concatenate((W_b_u, W_b_l), axis=0)

    W_upayload = np.concatenate((np.zeros((len(W_l), 2)), W_l[:, [-9, -7]]), axis=0)

    W = np.concatenate((W, W_upayload), axis=1)

    W_kpayload = np.concatenate(
        (np.zeros((len(W_l), 1)), W_l[:, -10].reshape(len(W_l), 1)), axis=0
    )
    W = np.concatenate((W, W_kpayload), axis=1)

    Phi_b = np.matmul(np.linalg.pinv(W), tau)

    # Phi_b_ref=np.copy(Phi_b)

    return W, Phi_b


def build_total_regressor(W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, param):
    """Inputs:  W_b_u  base regressor for unloaded case
            W_b_l:  base Regressor for loaded case
            W_l: Full  regressor for loaded case
            I_u: measured current in uloaded case in A
            I_l: measured current in loaded case in A
    Ouputs: W_tot: total regressor matrix
            V_norm= Normalised solution vector
            residue"""

    # build the total regressor matrix for TLS
    # we have to add a minus in front of the regressors for tTLS
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    # Two columns for the current
    V_a = np.concatenate(
        (
            I_u[0 : param["nb_samples"]].reshape(param["nb_samples"], 1),
            np.zeros((param["nb_samples"], 1)),
        ),
        axis=0,
    )
    V_a = np.concatenate(
        (
            V_a,
            np.concatenate(
                (
                    np.zeros((param["nb_samples"], 1)),
                    I_u[param["nb_samples"] :].reshape(param["nb_samples"], 1),
                ),
                axis=0,
            ),
        ),
        axis=1,
    )

    V_b = np.concatenate(
        (
            I_l[0 : param["nb_samples"]].reshape(param["nb_samples"], 1),
            np.zeros((param["nb_samples"], 1)),
        ),
        axis=0,
    )
    V_b = np.concatenate(
        (
            V_b,
            np.concatenate(
                (
                    np.zeros((param["nb_samples"], 1)),
                    I_l[param["nb_samples"] :].reshape(param["nb_samples"], 1),
                ),
                axis=0,
            ),
        ),
        axis=1,
    )

    W_current = np.concatenate((V_a, V_b), axis=0)

    W_tot = np.concatenate((W_tot, W_current), axis=1)

    # selection and reduction of the regressor for the unknown parameters for the mass

    if param["has_friction"]:  # adds fv and fs
        W_l_temp = np.zeros((len(W_l), 12))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
            W_l_temp[:, k] = W_l[
                :, (param["which_body_loaded"] - 1) * 12 + k
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
                -W_l[:, (param["which_body_loaded"] - 1) * 12 + 9].reshape(len(W_l), 1),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    elif param["has_actuator_inertia"]:  # adds ia fv fs off
        W_l_temp = np.zeros((len(W_l), 14))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]:
            W_l_temp[:, k] = W_l[
                :, (param["which_body_loaded"] - 1) * 14 + k
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
                -W_l[:, (param["which_body_loaded"] - 1) * 14 + 9].reshape(len(W_l), 1),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    else:
        W_l_temp = np.zeros((len(W_l), 9))
        for k in range(9):
            W_l_temp[:, k] = W_l[
                :, (param["which_body_loaded"] - 1) * 10 + k
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
                -W_l[:, (param["which_body_loaded"] - 1) * 10 + 9].reshape(len(W_l), 1),
            ),
            axis=0,
        )  # the mass
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    print(W_tot.shape)
    print(np.linalg.matrix_rank(W_tot))
    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    # ind_min = np.argmin(S)

    V = np.transpose(Vh).conj()

    # for validation purpose
    # W_tot_est = W_tot  # -S[-1]*np.matmul(U[:,-1].reshape(
    # len(W_tot),1),np.transpose(V[:,-1].reshape(len(Vh),1)))

    V_norm = param["mass_load"] * np.divide(V[:, -1], V[-1, -1])

    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue


def standardParameters(model, param):
    """This function prints out the standard inertial parameters obtained from 3D
    design.
    Note: a flag IsFrictioncld to include in standard parameters
    Input: 	njoints: number of joints
    Output: params_std: a dictionary of parameter names and their values"""
    params_name = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]
    phi = []
    params = []

    for i in range(1, model.njoints):
        P = model.inertias[i].toDynamicParameters()
        for k in P:
            phi.append(k)
        for j in params_name:
            params.append(j + str(i))

    if param["isFrictionincld"]:
        for k in range(1, model.njoints):
            # Here we add arbitrary values

            phi.extend([param["fv"], param["fc"]])
            params.extend(["fv" + str(k), "fc" + str(k)])
    params_std = dict(zip(params, phi))
    return params_std


# Building regressor


def iden_model(model, data, q, dq, ddq, param):
    """This function calculates joint torques and generates the joint torque regressor.
    Note: a parameter Friction as to be set to include in dynamic model
    Input: 	model, data: model and data structure of robot from Pinocchio
            q, v, a: joint's position, velocity, acceleration
            N : number of samples
            nq: length of q
    Output: tau: vector of joint torque
            W : joint torque regressor"""
    nb_samples = len(q)
    tau = np.empty(model.nq * nb_samples)
    W = np.empty([nb_samples * model.nq, 10 * model.nq])

    for i in range(nb_samples):
        tau_temp = pin.rnea(model, data, q[i, :], dq[i, :], ddq[i, :])
        W_temp = pin.computeJointTorqueRegressor(
            model, data, q[i, :], dq[i, :], ddq[i, :]
        )
        for j in range(model.nq):
            tau[j * nb_samples + i] = tau_temp[j]
            W[j * nb_samples + i, :] = W_temp[j, :]

    if param["Friction"]:
        W = np.c_[W, np.zeros([nb_samples * model.nq, 2 * model.nq])]
        for i in range(nb_samples):
            for j in range(model.nq):
                tau[j * nb_samples + i] = (
                    tau[j * nb_samples + i]
                    + dq[i, j] * param["fv"]
                    + np.sign(dq[i, j]) * param["fc"]
                )
                W[j * nb_samples + i, 10 * model.nq + 2 * j] = dq[i, j]
                W[j * nb_samples + i, 10 * model.nq + 2 * j + 1] = np.sign(dq[i, j])

    return tau, W


# eliminate zero columns


def eliminateNonAffecting(W_, params_std, tol_e):
    """This function eliminates columns which has L2 norm smaller than tolerance.
    Input: 	W: joint torque regressor
            tol_e: tolerance
    Output: W_e: reduced regressor
                    params_r: corresponding parameters to columns of reduced regressor
    """
    col_norm = np.diag(np.dot(W_.T, W_))
    idx_e = []
    params_e = []
    params_r = []

    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
            params_e.append(list(params_std.keys())[i])
        else:
            params_r.append(list(params_std.keys())[i])

    W_e = np.delete(W_, idx_e, 1)
    return W_e, params_r


# QR decompostion, rank revealing


def double_QR(tau, W_e, params_r, params_std=None):
    """This function calculates QR decompostion 2 times, first to find symbolic
    expressions of base parameters, second to find their values after re-organizing
    regressor matrix.
            Input:  W_e: regressor matrix (normally after eliminating zero columns)
                    params_r: a list of parameters corresponding to W_e
            Output: W_b: base regressor
                    base_parametes: a dictionary of base parameters"""
    # scipy has QR pivoting using Householder reflection
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    # find rank of regressor
    epsilon = np.finfo(float).eps  # machine epsilon
    tolpal = W_e.shape[0] * abs(np.diag(R).max()) * epsilon  # rank revealing tolerance
    # tolpal = 0.02
    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tolpal:
            idx_base.append(i)
        else:
            idx_regroup.append(i)

    numrank_W = len(idx_base)

    # rebuild W and params after sorted
    W1 = np.zeros([W_e.shape[0], len(idx_base)])
    W2 = np.zeros([W_e.shape[0], len(idx_regroup)])

    params_base = []
    params_regroup = []
    print(idx_base)
    for i in range(len(idx_base)):
        W1[:, i] = W_e[:, idx_base[i]]
        params_base.append(params_r[idx_base[i]])
    for j in range(len(idx_regroup)):
        W2[:, j] = W_e[:, idx_regroup[j]]
        params_regroup.append(params_r[idx_regroup[j]])

    W_regrouped = np.c_[W1, W2]

    # perform QR decomposition second time on regrouped regressor
    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W : R.shape[1]]

    print(Q1.shape)
    print(np.linalg.matrix_rank(Q1))

    # regrouping coefficient
    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)

    # values of base params
    phi_b = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)
    print(phi_b)

    # base regressor
    W_b = np.dot(Q1, R1)

    """phi_pinv=np.round(np.matmul(np.linalg.pinv(W_b),tau), 6)
    print(phi_pinv)
    phi_b=phi_pinv[0:len(phi_b)]"""

    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!  "

    # reference values from std params
    if params_std is not None:
        phi_std = []
        for x in params_base:
            phi_std.append(params_std[x])
        for i in range(numrank_W):
            for j in range(beta.shape[1]):
                phi_std[i] = phi_std[i] + beta[i, j] * params_std[params_regroup[j]]
        phi_std = np.around(phi_std, 5)

    tol_beta = 1e-6  # for scipy.signal.decimate
    for i in range(numrank_W):
        for j in range(beta.shape[1]):
            if abs(beta[i, j]) < tol_beta:

                params_base[i] = params_base[i]

            elif beta[i, j] < -tol_beta:

                params_base[i] = (
                    params_base[i]
                    + " - "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )

            else:

                params_base[i] = (
                    params_base[i]
                    + " + "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )

    base_parameters = dict(zip(params_base, phi_b))

    print("base parameters and their identified values: ")
    table = [params_base, phi_b]
    print(tabulate(table))

    if params_std is not None:
        return W_b, base_parameters, params_base, phi_b, phi_std
    else:
        return W_b, base_parameters, params_base, phi_b
