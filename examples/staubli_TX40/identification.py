import hppfcl
import eigenpy
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
from numpy.testing import assert_allclose
from scipy import linalg, signal


import pandas as pd
import json
import csv
import yaml
from yaml.loader import SafeLoader
import pprint
# from tabulate import tabulate

from src.figaroh.tools.robot import Robot
from src.figaroh.tools.regressor import *
from src.figaroh.tools.qrdecomposition import *
from src.figaroh.tools.randomdata import *
from src.figaroh.tools.robotcollisions import *
from src.figaroh.identification.identification_tools import get_param_from_yaml,calculate_first_second_order_differentiation, base_param_from_standard, calculate_standard_parameters, low_pass_filter_data


robot = Robot(
   'models/others/robots/staubli_tx40_description/urdf/tx40_mdh_modified.urdf','models/others/robots',
    False,
    True,
    True,
    True,
    True,
)


if __name__ == "__main__":

    model = robot.model
    data = robot.data

    nq, nv, njoints = model.nq, model.nv, model.njoints

    with open('examples/staubli_TX40/config/T40X_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
        pprint.pprint(config)
    
    identif_data = config['identification']

    params_settings = get_param_from_yaml(robot, identif_data)
    params_std = robot.get_standard_parameters_v2(params_settings)

    q_rand = np.random.uniform(low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nq))

    dq_rand = np.random.uniform(
        low=-10, high=10, size=(10 * params_settings["nb_samples"], model.nv)
    )

    ddq_rand = np.random.uniform(
        low=-30, high=30, size=(10 * params_settings["nb_samples"], model.nv)
    )
    W = build_regressor_basic_v2(robot, q_rand, dq_rand, ddq_rand, params_settings)
    W = add_coupling(W, model, data, len(q_rand), nq, nv, njoints, q_rand, dq_rand, ddq_rand)
    
    # remove zero cols and build a zero columns free regressor matrix
    idx_e, params_r = get_index_eliminate(W, params_std, 1e-6)
    W_e = build_regressor_reduced(W, idx_e)

    # Calulate the base regressor matrix, the base regroupings equations params_base and
    # get the idx_base, ie. the index of base parameters in the initial regressor matrix
    _, params_base, idx_base = get_baseParams_v2(W_e, params_r, params_std)

    print("The structural base parameters are: ")
    for ii in range(len(params_base)):
        print(params_base[ii])

    f_sample = 1/params_settings['ts']

    curr_data = pd.read_csv(
        "examples/staubli_TX40/data/curr_data.csv").to_numpy()
    pos_data = pd.read_csv(
        "examples/staubli_TX40/data/pos_read_data.csv").to_numpy()
    # Nyquist freq/0.5*sampling rate fs = 0.5 *5 kHz

    N_robot = params_settings['N']

    # cut off tail and head
    head = int(0.1 * f_sample)
    tail = int(7.5 * f_sample + 1)
    y = curr_data
    q_motor = pos_data

    # calculate joint position = inv(reduction ration matrix)*motor_encoder_angle
    red_q = np.diag([N_robot[0], N_robot[1], N_robot[2],
                    N_robot[3], N_robot[4], N_robot[5]])
    red_q[5, 4] = N_robot[5]
    q_T = np.dot(np.linalg.inv(red_q), q_motor.T)
    q = q_T.T

    q_nofilt = np.array(q)

    nbutter = 4
    nbord = 5 * nbutter

    for ii in range(model.nq):
        if ii == 0:
            q= low_pass_filter_data(q_nofilt[:,ii], params_settings,nbutter)
        else:
            q = np.column_stack((q,low_pass_filter_data(q_nofilt[:,ii], params_settings,nbutter)))


    # calibration between joint mdh position and robot measure
    q[:, 1] += -np.pi / 2
    q[:, 2] += np.pi / 2

    q, dq, ddq = calculate_first_second_order_differentiation(model,q,params_settings)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Verif q')

    ax1.plot(q[:,1])
    ax1.set_ylabel('q1')

    ax2.plot(dq[:,1])
    ax2.set_ylabel('dq1')

    ax3.plot(ddq[:,1])
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('ddq1')

    plt.show()

    # build regressor matrix
    qd = dq
    qdd = ddq
    N = q.shape[0]

    W = build_regressor_basic_v2(robot,q,qd,qdd,params_settings)
    W = add_coupling(W, model, data, N, nq, nv, njoints, q, qd, qdd)

    # calculate joint torques = reduction gear ratio matrix*motor_torques
    red_tau = np.diag([N_robot[0], N_robot[1], N_robot[2],
                    N_robot[3], N_robot[4], N_robot[5]])
    red_tau[4, 5] = N_robot[5]
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
            if abs(W_list[i][j, i * 14 + 11]) < params_settings['dq_lim_def'][i]:  # check columns of fv_i
                idx_qd_cross_zero.append(j)
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
    W_e, params_r = eliminate_non_dynaffect(W_, params_std, 0.001)
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
        "staubli_TX40/results/TX40_bp_5.csv",
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
    while not (max_std_e < params_settings['ratio_essential'] * min_std_e):
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
        "staubli_TX40/results/TX40_ep_5.csv",
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
