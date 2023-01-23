import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
from pinocchio.utils import *

from sys import argv
import os
from os.path import dirname, join, abspath
import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.linalg import norm, solve
from scipy import linalg, signal


import pandas as pd
import json
import csv
# from tabulate import tabulate

from tools.robot import Robot
from tools.regressor import *
from tools.qrdecomposition import *
from tools.randomdata import *
from tools.robotcollisions import *


robot = Robot(
    "staubli_tx40_description/urdf",
    "tx40_mdh_modified.urdf",
    False,
    True,
    True,
    True,
    True,
)


if __name__ == "__main__":
    model = robot.model
    print(model)
    data = robot.data
    nq, nv, njoints = model.nq, model.nv, model.njoints
    # params_std = standardParameters(njoints,fv,fs,Ia,off,Iam6,fvm6,fsm6)
    params_std = robot.get_standard_parameters()
    # table_stdparams = pd.DataFrame(params_std.items(), columns=[
    #                                "Standard Parameters", "Value"])
    # print(table_stdparams)

    # print("identification on exp data")

    f_sample = 5000
    curr_data = pd.read_csv(
        "/home/ecn/Cooking/figaro/identification_toolbox/src/thanh/curr_data.csv").to_numpy()
    pos_data = pd.read_csv(
        "/home/ecn/Cooking/figaro/identification_toolbox/src/thanh/pos_read_data.csv").to_numpy()
    # Nyquist freq/0.5*sampling rate fs = 0.5 *5 kHz

    N = pos_data.shape[0]

    # cut off tail and head
    head = int(0.1 * f_sample)
    tail = int(7.5 * f_sample + 1)
    # y = curr_data[head:tail,:]
    # q_motor = pos_data[head:tail,:]
    y = curr_data
    q_motor = pos_data

    # calculate joint position = inv(reduction ration matrix)*motor_encoder_angle
    red_q = np.diag([robot.N1, robot.N2, robot.N3,
                    robot.N4, robot.N5, robot.N6])
    red_q[5, 4] = robot.N6
    q_T = np.dot(np.linalg.inv(red_q), q_motor.T)
    q = q_T.T

    # median order 3 => butterworth zerophase filtering
    nbutter = 4
    f_butter = 100
    b, a = signal.butter(nbutter, f_butter / (f_sample / 2), "low")
    for j in range(q.shape[1]):
        q[:, j] = signal.medfilt(q[:, j], 3)
        q[:, j] = signal.filtfilt(
            b, a, q[:, j], axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)
        )

    # calibration between joint mdh position and robot measure
    q[:, 1] += -np.pi / 2
    q[:, 2] += np.pi / 2
    # q[:, 5] += np.pi #already calibrated in urdf for joint 6

    # calculate vel and acc by central difference
    dt = 1 / f_sample
    dq = np.zeros([q.shape[0], q.shape[1]])
    ddq = np.zeros([q.shape[0], q.shape[1]])
    t = np.linspace(0, dq.shape[0], num=dq.shape[0]) / f_sample
    for i in range(pos_data.shape[1]):
        dq[:, i] = np.gradient(q[:, i], edge_order=1) / dt
        ddq[:, i] = np.gradient(dq[:, i], edge_order=1) / dt
        plt.plot(t, q[:, i])

    plt.axvline(0.1)
    plt.axvline(7.5)
    plt.show(block=True)

    # suppress end segments of samples due to the border effect
    nbord = 5 * nbutter
    q = np.delete(q, np.s_[0:nbord], axis=0)
    q = np.delete(q, np.s_[(q.shape[0] - nbord): q.shape[0]], axis=0)
    dq = np.delete(dq, np.s_[0:nbord], axis=0)
    dq = np.delete(dq, np.s_[(dq.shape[0] - nbord): dq.shape[0]], axis=0)
    ddq = np.delete(ddq, np.s_[0:nbord], axis=0)
    ddq = np.delete(ddq, np.s_[(ddq.shape[0] - nbord): ddq.shape[0]], axis=0)

    # build regressor matrix
    qd = dq
    qdd = ddq
    N = q.shape[0]
    W = build_regressor_full_modified(
        model, data, N, nq, nv, njoints, q, qd, qdd)
    W = add_coupling(W, model, data, N, nq, nv, njoints, q, qd, qdd)

    # calculate joint torques = reduction gear ratio matrix*motor_torques
    red_tau = np.diag([robot.N1, robot.N2, robot.N3,
                      robot.N4, robot.N5, robot.N6])
    red_tau[4, 5] = robot.N6
    tau_T = np.dot(red_tau, y.T)

    # suppress end segments of  samples due to the border effect
    tau_T = np.delete(tau_T, np.s_[0:nbord], axis=1)
    tau_T = np.delete(
        tau_T, np.s_[(tau_T.shape[1] - nbord): tau_T.shape[1]], axis=1)

    # straight a matrix n-by-6 to a vector 6n-by-1
    tau = np.asarray(tau_T).ravel()

    # decimate by scipy.signal.decimate best_factor q = 25
    # parallel decimate joint-by-joint on joint torques and columns of regressor
    nj_ = tau.shape[0] // 6
    print("ni: ", nj_)
    tau_list = []
    W_list = []
    for i in range(nv):
        tau_temp = tau[(i * nj_): ((i + 1) * nj_)]
        for m in range(2):
            print(tau_temp.shape)
            tau_temp = signal.decimate(tau_temp, q=10, zero_phase=True)
            print(tau_temp.shape)
        tau_list.append(tau_temp)
        W_joint_temp = np.zeros((tau_temp.shape[0], W.shape[1]))
        for j in range(W_joint_temp.shape[1]):
            W_joint_temp_col = W[(i * nj_): (i * nj_ + nj_), j]
            for n in range(2):
                W_joint_temp_col = signal.decimate(
                    W_joint_temp_col, q=10, zero_phase=True
                )
            W_joint_temp[:, j] = W_joint_temp_col
        W_list.append(W_joint_temp)

    # eliminate qd crossing zero
    for i in range(len(W_list)):
        idx_qd_cross_zero = []
        for j in range(W_list[i].shape[0]):
            if abs(W_list[i][j, i * 14 + 11]) < robot.qd_lim[i]:  # check columns of fv_i
                idx_qd_cross_zero.append(j)
        # if i == 4 or i == 5:  # joint 5 and 6
        #     for k in range(W_list[i].shape[0]):
        #         if abs(W_list[i][k, 4 * 14 + 11] + W_list[i][k, 5 * 14 + 11]) < qd_lim[4] + qd_lim[5]:  # check sum cols of fv_5 + fv_6
        #             idx_qd_cross_zero.append(k)
        # indices with vels around zero
        idx_eliminate = list(set(idx_qd_cross_zero))
        W_list[i] = np.delete(W_list[i], idx_eliminate, axis=0)
        tau_list[i] = np.delete(tau_list[i], idx_eliminate, axis=0)
        print(W_list[i].shape, tau_list[i].shape)

    # rejoining
    # note:length of data on each joint different
    row_size = 0
    for i in range(len(tau_list)):
        row_size += tau_list[i].shape[0]

    tau_ = np.zeros(row_size)
    W_ = np.zeros((row_size, W_list[0].shape[1]))
    a = 0
    for i in range(len(tau_list)):
        tau_[a: (a + tau_list[i].shape[0])] = tau_list[i]
        W_[a: (a + tau_list[i].shape[0]), :] = W_list[i]
        a += tau_list[i].shape[0]
    print(tau_.shape, W_.shape)

    # base parameters
    # elimate and QR decomposition for ordinary LS
    W_e, idx_e, params_r = eliminate_non_dynaffect(W_, params_std, 0.001)
    W_b, base_parameters, params_base, phi_b = double_QR(tau_, W_e, params_r)
    std_xr_ols = relative_stdev(W_b, phi_b, tau_)
    phi_b_ols = np.around(np.linalg.lstsq(W_b, tau_, rcond=None)[0], 6)

    # weighted LS
    a = 0
    sig_ro_joint = np.zeros(nv)
    diag_SIGMA = np.zeros(row_size)
    # variance to each joint estimates
    for i in range(len(tau_list)):
        sig_ro_joint[i] = np.linalg.norm(
            tau_list[i] - np.dot(W_b[a: (a + tau_list[i].shape[0]), :], phi_b)
        ) ** 2 / (tau_list[i].shape[0])
        diag_SIGMA[a: (a + tau_list[i].shape[0])] = np.full(
            tau_list[i].shape[0], sig_ro_joint[i]
        )
        a += tau_list[i].shape[0]
    SIGMA = np.diag(diag_SIGMA)
    # Covariance matrix
    C_X = np.linalg.inv(
        np.matmul(np.matmul(W_b.T, np.linalg.inv(SIGMA)), W_b)
    )  # (W^T*SIGMA^-1*W)^-1
    # WLS solution
    phi_b = np.matmul(
        np.matmul(np.matmul(C_X, W_b.T), np.linalg.inv(SIGMA)), tau_)  # (W^T*SIGMA^-1*W)^-1*W^T*SIGMA^-1*TAU
    phi_b = np.around(phi_b, 6)

    # residual
    print("number of equations(after preproccesing): ", row_size)
    print("residual norm: ", np.linalg.norm(tau_ - np.dot(W_b, phi_b)))
    print(
        "relative residual norm: ",
        np.linalg.norm(tau_ - np.dot(W_b, phi_b)) / np.linalg.norm(tau_),
    )

    # WLS standard deviation
    STD_X = np.diag(C_X)
    std_xr = np.zeros(STD_X.shape[0])
    for i in range(STD_X.shape[0]):
        std_xr[i] = np.round(100 * np.sqrt(STD_X[i]) / np.abs(phi_b[i]), 2)

    # print("eleminanted parameters: ", params_e)
    print("condition number of base regressor: ", np.linalg.cond(W_b))
    # print('condition number of observation matrix: ', np.linalg.cond(W_e))

    path_save_bp = join(
        dirname(dirname(str(abspath(__file__)))),
        "identification_toolbox/src/thanh/TX40_bp_5.csv",
    )
    with open(path_save_bp, "w") as output_file:
        w = csv.writer(output_file)
        for i in range(len(params_base)):
            w.writerow(
                [params_base[i], phi_b_ols[i], std_xr_ols[i], phi_b[i], std_xr[i]]
            )

    ############################################################################################################################
    # essential parameter
    min_std_e = min(std_xr)
    max_std_e = max(std_xr)
    std_xr_e = std_xr
    params_essential = params_base
    W_essential = W_b
    while not (max_std_e < robot.ratio_essential * min_std_e):
        (i,) = np.where(np.isclose(std_xr_e, max_std_e))
        del params_essential[int(i)]
        W_essential = np.delete(W_essential, i, 1)

        # OLS
        phi_e_ols = np.around(np.linalg.lstsq(
            W_essential, tau_, rcond=None)[0], 6)
        std_e_ols = relative_stdev(W_essential, phi_e_ols, tau_)
        print("condition number of essential regressor: ",
              np.linalg.cond(W_essential))

        # weighted LS
        a = 0
        sig_ro_joint_e = np.zeros(nv)
        diag_SIGMA_e = np.zeros(row_size)
        # variance to each joint estimates
        for i in range(len(tau_list)):
            sig_ro_joint_e[i] = np.linalg.norm(
                tau_list[i]
                - np.dot(W_essential[a: (a + tau_list[i].shape[0]), :], phi_e_ols)
            ) ** 2 / (tau_list[i].shape[0])
            diag_SIGMA_e[a: (a + tau_list[i].shape[0])] = np.full(
                tau_list[i].shape[0], sig_ro_joint[i]
            )
            a += tau_list[i].shape[0]
        SIGMA_e = np.diag(diag_SIGMA_e)
        # Covariance matrix
        C_X_e = np.linalg.inv(
            np.matmul(np.matmul(W_essential.T,
                      np.linalg.inv(SIGMA_e)), W_essential)
        )  # (W^T*SIGMA^-1*W)^-1
        # WLS solution
        phi_e_wls = np.matmul(
            np.matmul(np.matmul(C_X_e, W_essential.T),
                      np.linalg.inv(SIGMA_e)), tau_
        )  # (W^T*SIGMA^-1*W)^-1*W^T*SIGMA^-1*TAU
        phi_e_wls = np.around(phi_e_wls, 6)
        # WLS standard deviation
        STD_X_e = np.diag(C_X_e)
        std_xr_e = np.zeros(STD_X_e.shape[0])
        for i in range(STD_X_e.shape[0]):
            std_xr_e[i] = np.round(
                100 * np.sqrt(STD_X_e[i]) / np.abs(phi_e_wls[i]), 2)
        min_std_e = min(std_xr_e)
        max_std_e = max(std_xr_e)
    print("number of equations(after preproccesing): ", row_size)
    print("residual norm: ", np.linalg.norm(
        tau_ - np.dot(W_essential, phi_e_wls)))
    print(
        "relative residual norm: ",
        np.linalg.norm(tau_ - np.dot(W_essential, phi_e_wls)) /
        np.linalg.norm(tau_),
    )
    print("condition number of essential regressor: ",
          np.linalg.cond(W_essential))
    # save results to csv
    path_save_ep = join(
        dirname(dirname(str(abspath(__file__)))),
        "identification_toolbox/src/thanh/TX40_ep_5.csv",
    )
    with open(path_save_ep, "w") as output_file:
        w = csv.writer(output_file)
        for i in range(len(params_essential)):
            w.writerow(
                [
                    params_essential[i],
                    phi_e_ols[i],
                    std_e_ols[i],
                    phi_e_wls[i],
                    std_xr_e[i],
                ]
            )
