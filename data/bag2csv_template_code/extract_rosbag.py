import csv
from re import split
import pandas as pd
import numpy as np
import rospy
# import dask.dataframe as dd

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pinocchio as pin


def extract_t_list(path_to_tf):
    """ Extracts list of timestampts where samples were recorded
        """
    pass


def test_readCSV(path_to_values, path_to_names):
    # read names and values from csv to dataframe
    dt_names = pd.read_csv(path_to_names)
    dt_values = pd.read_csv(path_to_values)

    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')


def extract_tf(path_to_tf, frame_names):
    """ Extract Qualysis data from tf bag of PAL robots,
        Input:  path_to_tf: path to csv file
                frame_name: list of str, frame defined in Qualisys streamed and recorded in rosbag
        Output: a dictionary
                keys: frame_names /values: 7xn array of [time, xyzquaternion]
    """
    tf_dict = {}

    # create data frame
    df = pd.read_csv(path_to_tf)

    # get collumns names
    frame_col = "child_frame_id"

    # translation
    x_col = "x"
    y_col = "y"
    z_col = "z"

    # orientation
    ux_col = "ux"
    uy_col = "uy"
    uz_col = "uz"
    w_col = "w"

    # time
    sec_col = "secs"
    nsec_col = "nsecs"

    # TODO: check if all names are correctly presented in headers of csv file

    # read values
    frame_val = df.loc[:, frame_col].values

    x_val = df.loc[:, x_col].values
    y_val = df.loc[:, y_col].values
    z_val = df.loc[:, z_col].values

    ux_val = df.loc[:, ux_col].values
    uy_val = df.loc[:, uy_col].values
    uz_val = df.loc[:, uz_col].values
    w_val = df.loc[:, w_col].values

    sec_val = df.loc[:, sec_col].values
    nsec_val = df.loc[:, nsec_col].values

    # t_val (list): extract and covert rostime to second
    t_val = []
    # starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec() # mark up t0
    starting_t = 0
    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

    # tf_dict (dict): return a dict contain key/item = frame_name(str)/numpy array
    for frame_name in frame_names:
        t = []
        x = []
        y = []
        z = []
        ux = []
        uy = []
        uz = []
        w = []
        for i in range(frame_val.shape[0]):
            if frame_val[i] == frame_name:
                t.append(t_val[i])
                x.append(x_val[i])
                y.append(y_val[i])
                z.append(z_val[i])
                ux.append(ux_val[i])
                uy.append(uy_val[i])
                uz.append(uz_val[i])
                w.append(w_val[i])
        tf_dict[frame_name] = np.transpose(
            np.array([t, x, y, z, ux, uy, uz, w]))
    return tf_dict


def extract_instrospection(path_to_values, path_to_names, value_names=[], t_list=[]):
    """ Extracts joint angles from Introspection Msg data from rosbag -> csv
        value_names: names of values to be extracted
        t_list: selected extracting timestamps
    """
    joint_dict = {}
    # read names and values from csv to dataframe
    dt_names = pd.read_csv(path_to_names)
    dt_values = pd.read_csv(path_to_values)

    # t_val (list): extract and convert rostime to second
    sec_col = "secs"
    nsec_col = "nsecs"
    sec_val = dt_values.loc[:, sec_col].values
    nsec_val = dt_values.loc[:, nsec_col].values
    t_val = []

    # starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec() # mark up t0
    starting_t = 0
    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

    # t_idx (list): get list of instants where data samples are picked up based on t_list
    # if t_list = [], extract the whole collumn
    if not t_list:
        t_list = t_val

    t_idx = []
    eps = 0.01
    for t in t_list:
        t_min = min(t_val, key=lambda x: abs(x-t))
        if abs(t-t_min) < eps:
            t_idx.append(t_val.index(t_min))

    # names (list): slice names in datanames corressponding to "values" column in datavalues
    names = []
    if dt_names.columns[-1] == "names_version":
        last_col = "names_version"
        if dt_names.columns[7] == "names":
            first_col = "names"
            first_idx = dt_names.columns.get_loc(first_col)
            last_idx = dt_names.columns.get_loc(last_col)
            names = list(dt_names.columns[range(first_idx+1, last_idx)])

    print("total number of columns in data_names: ", len(names))

    # joint_idx (list): get indices of corresponding to active joints
    # if value_names = [], extract all available values
    if not value_names:
        value_names = names

    joint_idx = []
    for element in value_names:
        if element in names:
            joint_idx.append(names.index(element))
        else:
            print(element, "Mentioned joint is not present in the names list.")
            break
    print("Joint indices corresponding to active joints: ", joint_idx)

    # joint_val (np.darray): split data in "values" column (str) to numpy array
    # extracted_val (np.darray):extract only values of interest from joint_val
    extracted_val = np.empty((len(t_idx), len(joint_idx)))
    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')

    if not len(split_data) == len(names):
        print("Names and value collumns did not match!")
    else:
        joint_val = []
        # slicing along axis 0 given t_idx
        for i in t_idx:
            # each msg is A STRING, it needs to be splitted and group into a list of float
            msg = dt_values_val[i]
            first_row = msg.replace('[', '')
            first_row = first_row.replace(']', '')
            row_data = first_row.split(',')
            joint_val.append(row_data)
        joint_val = np.asarray(joint_val, dtype=np.float64)

        # slicing along axis 1 given value_idx
        for i in range(len(joint_idx)):
            extracted_val[:, i] = joint_val[:, joint_idx[i]]
    return extracted_val


def get_data_sample(pos, t_list, eps=0.1):
    """ Extracts data samples give a list of specific instants
    """
    pos_idx = []
    count = 0
    for t in t_list:
        count += 1
        t_min = min(list(pos[:, 0]), key=lambda x: abs(x-t))
        print("deviation of time step: ", abs(t-t_min))

        if abs(t-t_min) < eps:
            curr_idx = list(pos[:, 0]).index(t_min)
            pos_idx.append(curr_idx)
        else:
            print("Missing data at %f" % t)
            print(count)
            break

    pos_sample = np.empty((len(pos_idx), pos.shape[1]))
    for i in range(len(pos_idx)):
        pos_sample[i, :] = pos[pos_idx[i], :]
    return pos_sample

#     # project prj_frame onto ref_frame


def project_frame(prj_frame, ref_frame):
    projected_pos = np.empty((prj_frame.shape[0], 3))
    if prj_frame.shape != ref_frame.shape:
        print("projecting two frames have different sizes! Projected positions are empty!")
    else:
        for i in range(prj_frame.shape[0]):
            ref_se3 = pin.XYZQUATToSE3(ref_frame[i, 1:])
            prj_se3 = pin.XYZQUATToSE3(prj_frame[i, 1:])
            projected_se3 = pin.SE3.inverse(ref_se3)*prj_se3
            projected_pos[i, :] = projected_se3.translation
    return projected_pos


def plot_position(frame, fig=[]):
    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(frame[:, 1], frame[:, 2], frame[:, 3], color='blue')


def save_csv(t, xyz, q, path_to_save, side=''):
    if not side:
        print("Error! Take side!")

    # # talos left arm
    elif side == 'left':
        path_save_ep = join(
            dirname(dirname(str(abspath(__file__)))),
            path_to_save)
        headers = [
            "x1", "y1", "z1",
            "torso1", "torso2", "armL1", "armL2", "armL3", "armL4", "armL5", "armL6", "armL7"]

    # # talos right arm
    elif side == 'right':
        path_save_ep = join(
            dirname(dirname(str(abspath(__file__)))),
            path_to_save)
        headers = [
            "x1", "y1", "z1",
            "torso1", "torso2", "armR1", "armR2", "armR3", "armR4", "armR5", "armR6", "armR7"]

    with open(path_save_ep, "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(headers)
        for i in range(len(t)):
            row = list(np.concatenate((xyz[i, :],
                                      q[i, :])))
            w.writerow(row)


def main():
    # list of instants where data samples are picked up
    # NOTE: cycle is not periodic!!!
    # sample test

    t_pick = []

    # left arm
    t_pick = [29.6]
    # , 46.6, 64.47, 80.34,
    #           98.2, 115.05, 131.91, 149.77,
    #           166.62, 183.49, 200.35, 218.2,
    #           235.06, 251.93, 268.80, 286.64,
    #           302.51, 320.37, 338.22, 356.08,
    #           371.95, 388.80, 405.66, 423.52,
    #           440.37, 457.24, 471.13, 491.95,
    #           507.83, 524.68, 541.55, 560.39,
    #           577.25, 294.12, 609.99, 627.83,
    #           645.70, 661.56, 679.41, 696.27,
    #           713.14, 730.99, 748.85, 764.72,
    #           782.56, 799.43, 816.29, 834.14,
    #           851.0, 869.84, 884.73, 903.57,
    #           920.44, 935.31, 954.15, 972.02,
    #           985.89, 1003.75, 1020.62, 1037.47,
    #           1056.31, 1089.04,
    #           ]
    t_pick.sort()
    print(t_pick)

    frame_names = ['"waist_frame"', '"endeffector_frame"']

    # extract mocap data
    # Talos

    path_to_tf = '/home/thanhndv212/Cooking/bag2csv/Calibration/Talos/talos_mars/calib_left_15_03_2022-03-15-13-54-45/tf_throttle.csv'
    ###################################### Talos 1 marker ###############
    # get full data
    talos_dict = extract_tf(path_to_tf, frame_names)
    W_pos = talos_dict[frame_names[0]]
    EE_pos = talos_dict[frame_names[1]]
    print("Endeffector and waist frame read from csv: ",
          EE_pos.shape, W_pos.shape)
    plot_position(EE_pos)
    plt.show()

    # select only data given timestamps
    t_list = [x + W_pos[0, 0] for x in t_pick]
    print("Endeffector and waist frame read from csv: ",
          EE_pos.shape, W_pos.shape)
    print("length of record:", EE_pos[-1, 0] -
          EE_pos[0, 0], W_pos[-1, 0] - W_pos[0, 0])

    W_sample = get_data_sample(W_pos, t_list)
    EE_sample = get_data_sample(EE_pos, t_list)
    print("Endeffector and waist frame at static postures: ",
          EE_sample.shape, W_sample.shape)

    # project endeffector onto waist
    EE_prj_sample = project_frame(EE_sample, W_sample)
    print("projected endefffector: ", EE_prj_sample.shape)
    ########################################################################

    # extract joint configurations data
    # Talos
    path_to_values = '/home/thanhndv212/Cooking/bag2csv/Calibration/Talos/talos_mars/calib_left_15_03_2022-03-15-13-54-45/introspection_datavalues_throttle.csv'
    path_to_names = '/home/thanhndv212/Cooking/bag2csv/Calibration/Talos/talos_mars/calib_left_15_03_2022-03-15-13-54-45/introspection_datanames_throttle.csv'

    # talos left arm
    # torso_1 = '- torso_1_joint_position'
    # torso_2 = '- torso_2_joint_position'
    # arm_left_1 = '- arm_left_1_joint_position'
    # arm_left_2 = '- arm_left_2_joint_position'
    # arm_left_3 = '- arm_left_3_joint_position'
    # arm_left_4 = '- arm_left_4_joint_position'
    # arm_left_5 = '- arm_left_5_joint_position'
    # arm_left_6 = '- arm_left_6_joint_position'
    # arm_left_7 = '- arm_left_7_joint_position'

    # joint_names = [torso_1, torso_2, arm_left_1, arm_left_2,
    #                arm_left_3, arm_left_4, arm_left_5, arm_left_6, arm_left_7]

    # talos right arm
    torso_1 = '- torso_1_joint_position'
    torso_2 = '- torso_2_joint_position'
    arm_right_1 = '- arm_right_1_joint_position'
    arm_right_2 = '- arm_right_2_joint_position'
    arm_right_3 = '- arm_right_3_joint_position'
    arm_right_4 = '- arm_right_4_joint_position'
    arm_right_5 = '- arm_right_5_joint_position'
    arm_right_6 = '- arm_right_6_joint_position'
    arm_right_7 = '- arm_right_7_joint_position'

    joint_names = [torso_1, torso_2, arm_right_1, arm_right_2,
                   arm_right_3, arm_right_4, arm_right_5, arm_right_6, arm_right_7]

    actJoint_val = extract_instrospection(
        path_to_values, path_to_names, joint_names, t_list)
    print("expectedd NbSamplexNbjoints: ", actJoint_val.shape)

    # write to csv
    save_csv(t_list, EE_prj_sample, actJoint_val,
             f"talos/sample.csv", side='right')
    ###################################### Tiago 4 markers ###############
    # t_pick = []
    # t_pick.sort()
    # print(t_pick)
    # frame_names = ['"base_frame"',
    #                '"eeframe_BL"',
    #                '"eeframe_BR"',
    #                '"eeframe_TL"',
    #                '"eeframe_TR"']

    # extract mocap data
    # Tiago
    # path_to_tf = '/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/calib_Nov/calib_mocap_2021-11-30-15-44-33/tf.csv'

    # get full data
    # talos_dict = extract_tf(path_to_tf, frame_names)
    # W_pos = talos_dict[frame_names[0]]
    # BL_pos = talos_dict[frame_names[1]]
    # BR_pos = talos_dict[frame_names[2]]
    # TL_pos = talos_dict[frame_names[3]]
    # TR_pos = talos_dict[frame_names[4]]

    # select only data given timestamps
    # t_list = [x + W_pos[0, 0] for x in t_pick]
    ########################################################################
    # extract joint configurations data

    # Tiago
    # path_to_values = '/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/calib_Nov/calib_mocap_2021-11-30-15-44-33/introspection_datavalues.csv'
    # path_to_names = '/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/calib_Nov/calib_mocap_2021-11-30-15-44-33/introspection_datanames.csv'

    # joint names
    # torso = '- torso_lift_joint_position'
    # arm_1 = '- arm_1_joint_position'
    # arm_2 = '- arm_2_joint_position'
    # arm_3 = '- arm_3_joint_position'
    # arm_4 = '- arm_4_joint_position'
    # arm_5 = '- arm_5_joint_position'
    # arm_6 = '- arm_6_joint_position'
    # arm_7 = '- arm_7_joint_position'
    # joint_names = [torso, arm_1, arm_2, arm_3, arm_4, arm_5, arm_6, arm_7]

    # W_sample = get_data_sample(W_pos, t_list)
    # BL_sample = get_data_sample(BL_pos, t_list)
    # BR_sample = get_data_sample(BR_pos, t_list)
    # TL_sample = get_data_sample(TL_pos, t_list)
    # TR_sample = get_data_sample(TR_pos, t_list)

    # project endeffector onto waist
    # BL_prj_sample = project_frame(BL_sample, W_sample)
    # BR_prj_sample = project_frame(BR_sample, W_sample)
    # TL_prj_sample = project_frame(TL_sample, W_sample)
    # TR_prj_sample = project_frame(TR_sample, W_sample)

    # plot markers in cartesian
    # fig2 = plt.figure(2)
    # ax2 = fig2.add_subplot(111, projection='3d')
    # ax2.scatter3D(BL_sample[:, 1], BL_sample[:, 2],
    #               BL_sample[:, 3], color='blue')
    # ax2.scatter3D(BR_sample[:, 1], BR_sample[:, 2],
    #               BR_sample[:, 3], color='red')
    # ax2.scatter3D(TL_sample[:, 1], TL_sample[:, 2],
    #               TL_sample[:, 3], color='green')
    # ax2.scatter3D(TR_sample[:, 1], TR_sample[:, 2],
    #               TR_sample[:, 3], color='yellow')
    # plt.show()

    # write to csv
    # path_save_ep = join(
    #     dirname(dirname(str(abspath(__file__)))),
    #     f"tiago/tiago_nov_30_64.csv")
    # headers = [
    #     "x1", "y1", "z1",
    #     "x2", "y2", "z2",
    #     "x3", "y3", "z3",
    #     "x4", "y4", "z4",
    #     "torso", "arm1", "arm2", "arm3", "arm4", "arm5", "arm6", "arm7"]
    # with open(path_save_ep, "w") as output_file:
    #     w = csv.writer(output_file)
    #     w.writerow(headers)
    #     for i in range(len(t_list)):
    #         row = list(np.concatenate((BL_prj_sample[i, :], BR_prj_sample[i, :],
    #                                   TR_prj_sample[i, :], TR_prj_sample[i, :],
    #                                   actJoint_val[i, :])))
    #         w.writerow(row)


if __name__ == '__main__':
    main()
