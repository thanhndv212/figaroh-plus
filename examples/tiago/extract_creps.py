### This scripts extracts time-stamped static posture data for tiago
### calibration using vicon nexus mocap system.
from os import listdir
from os.path import isfile, join, dirname, abspath

import numpy as np
import matplotlib.pyplot as plt

import pinocchio as pin

import tiago_utils.suspension.processing_utils as pu
from tiago_utils.tiago_tools import load_robot

# ######################################################################################
# vicon_path = '/home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon'
# files = [join(vicon_path, f) for f in listdir(vicon_path) if isfile(join(vicon_path, f))]
# for f in files:
#     if 'csv' not in f:
#         files.remove(f)
# files.sort()
# # 0 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_around_x_vicon_1634.csv
# # 1 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_around_y_vicon_1638.csv
# # 2 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_around_z_vicon_1622.csv
# # 3 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_calib_vicon_1508.c3d
# # 4 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_calib_vicon_1607.csv
# # 5 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_postures_vicon_1720.csv
# # 6 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_postures_vicon_1734.csv
# # 7 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_x_fold_vicon_1632.csv
# # 8 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_mirror_vicon_1630.csv
# # 9 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_mirror_vicon_1642.csv
# # 10 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_vicon_1628.csv
# # 11 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_xyz_vicon_1640.csv
# # 12 /home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/vicon/tiago_y_fold_vicon_1636.csv
# input_file = files[4]

# ######################################################################################
# bag_path = '/home/thanhndv212/Downloads/experiment_data/suspension/bags/creps/creps_bags/'
# file_path = [f for f in listdir(bag_path)]
# file_path.sort()
# # 0 tiago_around_x_vicon_1634
# # 1 tiago_around_y_vicon_1638
# # 2 tiago_around_z_vicon_1622
# # 3 tiago_calib_vicon_1508
# # 4 tiago_calib_vicon_1607
# # 5 tiago_x_fold_vicon_1632
# # 6 tiago_xyz_mirror_vicon_1630
# # 7 tiago_xyz_mirror_vicon_1642
# # 8 tiago_xyz_vicon_1628
# # 9 tiago_xyz_vicon_1640
# # 10 tiago_y_fold_vicon_1636
# input_path_joint = file_path[4]
# assert input_path_joint in input_file, "ERROR: Mocap data and joitn encoder data do not match!"
dir_path = "/media/thanhndv212/Cooking/processed_data/tiago/develop/data/identification/suspension/creps/calibration/tiago_calib_vicon_1607/"

######################################################################################
input_file = dir_path + "tiago_calib_vicon_1607.csv"
print("Read VICON data from ", input_file)
calib_df = rpu.ead_csv_vicon(input_file)

f_res = 100
f_cutoff = 5
selected_range = range(0, 41000)
# selected_range = range(0, 5336)
plot = True  # plot the coordinates
plot_raw = True
alpha = 0.25
time_stamps_vicon = [
    830,
    3130,
    4980,
    6350,
    7740,
    9100,
    10860,
    13410,
    15830,
    18170,
    20000,
    21860,
    24320,
    26170,
    27530,
    29410,
    31740,
    33710,
    35050,
    36560,
    39000,
]
# time_stamps_vicon = None
####################################################


def filter_data(name: str, col1: str, col2: str, col3: str):
    return pu.filter_xyz(
        name,
        calib_df.loc[:, [col1, col2, col3]].to_numpy()[selected_range],
        f_res,
        f_cutoff,
        plot,
        time_stamps_vicon,
        plot_raw,
        alpha,
    )


[base1, base2, base3] = [
    filter_data(
        "base{}".format(ij),
        "base{}_x".format(ij),
        "base{}_y".format(ij),
        "base{}_z".format(ij),
    )
    for ij in [1, 2, 3]
]

[shoulder1, shoulder2, shoulder3, shoulder4] = [
    filter_data(
        "shoulder{}".format(ij),
        "shoulder{}_x".format(ij),
        "shoulder{}_y".format(ij),
        "shoulder{}_z".format(ij),
    )
    for ij in [1, 2, 3, 4]
]

[gripper1, gripper2, gripper3] = [
    filter_data(
        "gripper{}".format(ij),
        "gripper{}_x".format(ij),
        "gripper{}_y".format(ij),
        "gripper{}_z".format(ij),
    )
    for ij in [1, 2, 3]
]

[force, moment, cop] = [
    filter_data(
        "{}".format(ij),
        "{}_x".format(ij),
        "{}_y".format(ij),
        "{}_z".format(ij),
    )
    for ij in ["F", "M", "COP"]
]

marker_data = dict()

## create a frame based on three markers
marker_data["base1"] = pu.create_rigidbody_frame(
    [base1, base2, base3],
    unit_rot=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0], [0, 0, -1]]),
)
marker_data["base2"] = pu.create_rigidbody_frame(
    [base2, base1, base3],
    unit_rot=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0], [0, 0, 1]]),
)
marker_data["base3"] = pu.create_rigidbody_frame(
    [base3, base1, base2],
    unit_rot=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0], [0, 0, 1]]),
)

# gripper 1, 2 in vicon dataset were inaccurate, large deviation.
# consequently orientation data were reliable as well.

marker_data["gripper3"] = pu.create_rigidbody_frame(
    [
        gripper3,
        gripper2,
        gripper1,
    ]
)

marker_data["shoulder1"] = pu.create_rigidbody_frame(
    [shoulder1, shoulder4, shoulder2]
)

marker_data["shoulder2"] = pu.create_rigidbody_frame(
    [shoulder2, shoulder4, shoulder1]
)

marker_data["shoulder3"] = pu.create_rigidbody_frame(
    [shoulder3, shoulder4, shoulder2]
)

marker_data["shoulder4"] = pu.create_rigidbody_frame(
    [shoulder4, shoulder2, shoulder3]
)

#####################################################################################

path_to_values = dir_path + "introspection_datavalues.csv"
path_to_names = dir_path + "introspection_datanames.csv"

# create a robot
robot = load_robot(
    abspath("urdf/tiago_48_schunk.urdf"),
    load_by_urdf=True,
)

# add object to gripper
# addBox_to_gripper(robot)

# read values from csv files
t_res, f_res, joint_names, q_abs_res, q_pos_res = pu.get_q_arm(
    robot, path_to_values, path_to_names, f_cutoff
)
active_joints = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
]
actJoint_idx = []
actJoint_idv = []
for act_j in active_joints:
    joint_idx = robot.model.getJointId(act_j)
    actJoint_idx.append(robot.model.joints[joint_idx].idx_q)
    actJoint_idv.append(robot.model.joints[joint_idx].idx_v)

q_arm, dq_arm, ddq_arm = pu.calc_vel_acc(
    robot, q_abs_res, selected_range, joint_names, f_res, f_cutoff
)
q_arm = q_abs_res[:, actJoint_idx]
time_stamps = [
    960,
    3367,
    5100,
    6500,
    7860,
    9260,
    11580,
    13500,
    16000,
    18400,
    20320,
    22300,
    24590,
    26530,
    27860,
    29740,
    32320,
    34170,
    35520,
    37000,
    39000,
]

q_sel = np.zeros([len(time_stamps), q_arm.shape[1]])
for i, ti in enumerate(time_stamps):
    q_sel[i, :] = q_arm[ti, :]

fig_joint = plt.figure()
for ai in range(q_arm.shape[1]):
    plt.plot(np.arange(q_arm.shape[0]), q_arm[:, ai])
    plt.scatter(x=np.array(time_stamps), y=q_sel[:, ai], s=20)

# convert to pinocchio frame
for base in ["base1", "base2", "base3"]:
    for shoulder in ["shoulder1", "shoulder2", "shoulder3", "shoulder4"]:
        for gripper in ["gripper3"]:

            [base_trans, base_rot] = marker_data[base]
            [shoulder_trans, shoulder_rot] = marker_data[shoulder]
            [gripper_trans, gripper_rot] = marker_data[gripper]

            base_frame = list()
            gripper_frame = list()
            shoulder_frame = list()

            for i, ti in enumerate(time_stamps_vicon):
                base_frame.append(pin.SE3(base_rot[ti], base_trans[ti, :]))
                gripper_frame.append(
                    pin.SE3(gripper_rot[ti], gripper_trans[ti, :])
                )
                shoulder_frame.append(
                    pin.SE3(shoulder_rot[ti], shoulder_trans[ti, :])
                )

            # reproject end frame on to start frame
            gripper_base = pu.project_frame(gripper_frame, base_frame)
            gripper_shoulder = pu.project_frame(gripper_frame, shoulder_frame)
            shoulder_base = pu.project_frame(shoulder_frame, base_frame)

            pu.save_selected_data(
                active_joints,
                gripper_base,
                q_sel,
                join(
                    dirname(str(abspath(__file__))),
                    "data/calibration/mocap/vicon/absolute_encoder/vicon_calibration_fc10_{}_{}.csv".format(
                        gripper, base
                    ),
                ),
            )
            pu.save_selected_data(
                active_joints,
                gripper_shoulder,
                q_sel,
                join(
                    dirname(str(abspath(__file__))),
                    "data/calibration/mocap/vicon/absolute_encoder/vicon_calibration_fc10_{}_{}.csv".format(
                        gripper, shoulder
                    ),
                ),
            )
            pu.save_selected_data(
                active_joints,
                shoulder_base,
                q_sel,
                join(
                    dirname(str(abspath(__file__))),
                    "data/calibration/mocap/vicon/absolute_encoder/vicon_calibration_fc10_{}_{}.csv".format(
                        shoulder, base
                    ),
                ),
            )
