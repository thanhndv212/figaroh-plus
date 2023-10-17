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

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal
import pandas as pd

# data set of 4 corners and middle points
""" algo to extract 10 points near corner point and middle point
    store list of 2D arrays"""


def extract_setpts(start_idx, reps, path_to_csv, frame_name):
    tf_eeframe = pd.read_csv(path_to_csv)

    # names
    frame_column = "child_frame_id"
    x_column = "x"
    y_column = "y"
    z_column = "z"

    # read_values
    frame_values = tf_eeframe.loc[:, frame_column].values
    x_values = tf_eeframe.loc[:, x_column].values
    y_values = tf_eeframe.loc[:, y_column].values
    z_values = tf_eeframe.loc[:, z_column].values

    # extract particular frame
    eeframe = frame_name
    eeframe_x = []
    eeframe_y = []
    eeframe_z = []

    # save to all measured coordinates to lists
    for i in range(frame_values.shape[0]):
        if frame_values[i] == eeframe:
            eeframe_x.append(x_values[i])
            eeframe_y.append(y_values[i])
            eeframe_z.append(z_values[i])

    # extract center, corners, middle points
    """ check the abrupt change of 1st numerical derivatives of z coordinates
    # #################
    # # C3     M2    C2
    # #
    # # M3     P     M1
    # #
    # # C4     M4    C1
    # #################
    # P -> C1 -> M1 -> C2 -> M2 -> C3 -> M3-> C4 -> M4 -> C1' ->P"""
    Nb_dp = 10

    period_p = 16500
    # long_step = 2000  # P-> C1, M4->C1', C1->P
    # short_step = 1500  # otherwise

    # incre_step = [0,
    #               2000,
    #               1500,
    #               1500,
    #               1500,
    #               1500,
    #               1500,
    #               1500,
    #               1500,
    #               2000]
    # mocap incremental steps
    incre_step = [0, 1500, 1500, 1500, 1500, 1500, 1300, 1300, 1500, 2300]
    init_idx = []

    for i in incre_step:
        start_idx += i
        init_idx.append(start_idx)

    # center
    dp = []
    count = 0
    for t in range(len(init_idx)):
        dp_i = []
        for i in range(reps):
            idx_p = init_idx[t] + i * period_p
            print(idx_p)
            k_set = np.zeros((Nb_dp, 3))
            if i == reps - 1 and t == len(init_idx) - 1:
                print(len(eeframe_x), len(eeframe_y), len(eeframe_z))
                idx_p = -10
            for j in range(Nb_dp):
                k_set[j, :] = np.array(
                    [
                        eeframe_x[idx_p + j],
                        eeframe_y[idx_p + j],
                        eeframe_z[idx_p + j],
                    ]
                )

            dp_i.append(k_set)
        dp.append(dp_i)
        count += 1
    dp_center = dp[0]
    dp_corner_list = [dp[1], dp[3], dp[5], dp[7]]
    dp_mid_list = [dp[2], dp[4], dp[6], dp[8]]
    dp_C1_return = dp[9]
    # 1 st derivs
    freq = 100  # hz
    dt = 1 / freq
    dzdt = np.gradient(np.array(eeframe_z), dt)
    dydt = np.gradient(np.array(eeframe_y), dt)
    dzdt = signal.medfilt(dzdt, 5)
    dydt = signal.medfilt(dydt, 5)

    # plot coordinates and their 1st derivs
    fig = plt.figure()
    ax = fig.subplots(2, 1)
    ax[0].plot(dzdt, label="1st derivative of z coordinates")
    ax[1].plot(dydt, label="1st derivative of y coordinates", color="g")
    ax[0].plot(np.array(eeframe_z), label="z coordinatesd")
    ax[1].plot(np.array(eeframe_y), label="y coordinatesd", color="black")
    fig.legend()
    # plt.show()
    return dp_center, dp_corner_list, dp_mid_list, dp_C1_return


def create_square(var, width, height, edge):
    """var: an array containing position of center point normalized nomal
            vector
    w, h: size of the rectangle
    edge: a vector of upside edge

    # #################
    # # C4     M4    C3
    # #
    # # M1     P     M3
    # #
    # # C1     M2    C2
    # #################"""
    # center point position
    center_p = np.copy(var)[0:3]

    # normal vector
    vec = np.copy(var)[3:6]
    normal_vec = vec / np.linalg.norm(vec)

    # size
    w = width
    h = height

    # one edge unit vector
    up_vec = np.copy(edge)

    # edge unit vector
    U = np.cross(normal_vec, up_vec)
    W = np.cross(normal_vec, U)

    # corner points
    C4 = center_p + w / 2 * U + h / 2 * W
    C1 = center_p - w / 2 * U + h / 2 * W
    C2 = center_p - w / 2 * U - h / 2 * W
    C3 = center_p + w / 2 * U - h / 2 * W

    # middle points
    M3 = center_p + w / 2 * U
    M4 = center_p + h / 2 * W
    M1 = center_p - w / 2 * U
    M2 = center_p - h / 2 * W

    corner_points = np.array([C1, C2, C3, C4])
    mid_points = np.array([M1, M2, M3, M4])
    return corner_points, mid_points


def plot_square(ax, corner_points, mid_points):
    for i in range(mid_points.shape[0]):
        ax.scatter(
            mid_points[i, 0], mid_points[i, 1], mid_points[i, 2], color="r"
        )

    for i in range(corner_points.shape[0]):
        ax.scatter(
            corner_points[i, 0],
            corner_points[i, 1],
            corner_points[i, 2],
            color="r",
        )
        ax.plot(
            [corner_points[i, 0], corner_points[i - 1, 0]],
            [corner_points[i, 1], corner_points[i - 1, 1]],
            [corner_points[i, 2], corner_points[i - 1, 2]],
            color="black",
        )

    # origin and axes show
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")
    ax.grid()
    # plt.show()


def plot_setpts(ax, dp_corner, dp_mid):
    for i in range(4):
        for j in range(dp_corner[i].shape[0]):
            ax.scatter(
                dp_corner[i][j, 0],
                dp_corner[i][j, 1],
                dp_corner[i][j, 2],
                color="blue",
            )
        for k in range(dp_mid[i].shape[0]):
            ax.scatter(
                dp_mid[i][k, 0], dp_mid[i][k, 1], dp_mid[i][k, 2], color="blue"
            )


# square fitting optimization problem


def cost_function(var, w, h, edge, dp_corner, dp_mid):
    c_pts, m_pts = create_square(var, w, h, edge)
    res_vec = []
    for i in range(4):
        for j in range(dp_corner[i].shape[0]):
            res_vec.append(np.linalg.norm([c_pts[i, :] - dp_corner[i][j, :]]))
        for k in range(dp_mid[i].shape[0]):
            res_vec.append(np.linalg.norm([m_pts[i, :] - dp_mid[i][k, :]]))

    res_sum = np.linalg.norm(np.array(res_vec))
    if res_sum == 0:
        print(" Cost function cannot be calculated!!!")
    return res_sum


def main():
    # predefined constants
    edge = np.array([0, 0, 1])
    w = 0.5
    h = 0.5

    # optimization problem
    init_guess = np.array([0.7, 0.0, 0.7, 1, 0, 0])

    # fitting data
    c_pts_sample, m_pts_sample = create_square(init_guess, w, h, edge)

    # artificial test data
    dp_corner = []
    dp_mid = []
    Nb_pts = 10
    for i in range(4):
        k_corner = np.zeros([Nb_pts, 3])
        k_mid = np.zeros([Nb_pts, 3])
        for k in range(Nb_pts):
            k_corner[k, :] = c_pts_sample[i, :] + 0.005 * np.random.rand(
                c_pts_sample[i, :].shape[0]
            )
            k_mid[k, :] = m_pts_sample[i, :] + 0.005 * np.random.rand(
                m_pts_sample[i, :].shape[0]
            )
        dp_corner.append(k_corner)
        dp_mid.append(k_mid)
    print(len(dp_corner))

    rslt = optimize.least_squares(
        cost_function,
        init_guess,
        jac="3-point",
        args=(w, h, edge, dp_corner, dp_mid),
        verbose=1,
    )

    print(rslt.x)
    print(rslt.cost)
    minimum = optimize.fmin(
        cost_function, init_guess, args=(w, h, edge, dp_corner, dp_mid)
    )
    print(minimum)

    # plot solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    LS_cpts, LS_mpts = create_square(rslt.x, w, h, edge)
    fmin_cpts, fmin_mpts = create_square(minimum, w, h, edge)

    plot_square(ax, LS_cpts, LS_mpts)
    plot_square(ax, fmin_cpts, fmin_mpts)

    # plot give data points
    for i in range(1):
        for j in range(dp_corner[i].shape[0]):
            ax.scatter(
                dp_corner[i][j, 0], dp_corner[i][j, 1], dp_corner[i][j, 2]
            )
        for k in range(dp_mid[i].shape[0]):
            ax.scatter(dp_mid[i][k, 0], dp_mid[i][k, 1], dp_mid[i][k, 2])
    plot_square(ax, c_pts_sample, m_pts_sample)

    plt.show()


if __name__ == "__main__":
    main()
    # path_to_csv = "/home/thanhndv212/Cooking/bag2csv/calib_Jan/rosbag/\
    # square_motion_4_zero_offsets_2022-01-12-10-55-17/tf.csv"
    # reps = 1
    # start_idx = 1050

    # path_to_csv = "/home/thanhndv212/Cooking/bag2csv/calib_Jan/rosbag/\
    # square_motion_4_pal_offsets_2022-01-12-11-38-43/tf.csv"
    # reps = 1
    # start_idx = 700

    # path_to_csv = "/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/\
    # calib_Jan/rosbag/square_motion_4_mocap_offsets_2022-01-12-11-25-38/tf.csv"
    # reps = 1
    # start_idx = 700

    # path_to_csv = "/home/thanhndv212/Cooking/bag2csv/calib_Jan/rosbag/\
    # square_motion_4_mocap_inv_offsets_2022-01-12-11-10-58/tf.csv"
    # # reps = 1
    # # start_idx = 500

    # frame_name = '"endeffector_frame"'
    # dp_center, dp_corner_list, dp_mid_list, dp_C1_return = extract_setpts(
    #     start_idx, reps, path_to_csv, frame_name
    # )

    # # predefined constants
    # edge = np.array([0, 0, 1])
    # w = 0.5
    # h = 0.5

    # # optimization problem
    # init_guess = np.array([0.7, 0.0, 0.7, 1, 0, 0])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # # repeatability
    # pt_name = ["center", "C1", "C2", "C3", "C4", "M1", "M2", "M3", "M4"]
    # # res_list = []
    # # res_list.append(np.linalg.norm(dp_center[0] - dp_center[1]))
    # # for i in range(len(dp_corner_list)):
    # #     res_list.append(np.linalg.norm(
    # #         dp_corner_list[i][0] - dp_corner_list[i][1]))
    # # for i in range(len(dp_mid_list)):
    # #     res_list.append(np.linalg.norm(
    # #         dp_mid_list[i][0] - dp_mid_list[i][1]))
    # # plt.bar(pt_name, res_list)
    # pt_list = [dp_center] + dp_corner_list + dp_mid_list

    # pt_array = np.empty((3, 9))
    # for i in range(9):
    #     pt_array[:, i] = np.mean(pt_list[i][0], axis=0)
    # import csv

    # path_save_ep = join(
    #     dirname(dirname(str(abspath(__file__)))),
    #     f"data/tiago/square_static_postures_mocap_offset.csv",
    # )
    # with open(path_save_ep, "w") as output_file:
    #     w = csv.writer(output_file)
    #     w.writerow(pt_name)
    #     for i in range(3):
    #         w.writerow(pt_array[i, :])

    # # square fitting

    # for j in range(reps):
    #     dp_corner = []
    #     dp_mid = []
    #     for i in range(len(dp_corner_list)):
    #         dp_corner.append(dp_corner_list[i][j])
    #         dp_mid.append(dp_mid_list[i][j])
    #     plot_setpts(ax, dp_corner, dp_mid)
    #     rslt = optimize.least_squares(
    #         cost_function,
    #         init_guess,
    #         jac="3-point",
    #         args=(w, h, edge, dp_corner, dp_mid),
    #         verbose=1,
    #     )
    #     print(rslt.x)
    #     print(rslt.cost)
    #     LS_cpts, LS_mpts = create_square(rslt.x, w, h, edge)
    #     plot_square(ax, LS_cpts, LS_mpts)

    # # plt.show()
