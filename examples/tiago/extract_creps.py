import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import pinocchio as pin
from os.path import isfile, join, dirname, abspath
from tiago_utils.suspension.processing_utils import *
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
# ######################################################################################
dir_path = (
    "/media/thanhndv212/Cooking/processed_data/tiago/develop/data/identification/suspension/creps/calibration/tiago_calib_vicon_1607/"
)

######################################################################################
input_file = dir_path + "tiago_calib_vicon_1607.csv"
print('Read VICON data from ', input_file)
calib_df = read_csv_vicon(input_file)

f_res = 100
f_cutoff = 5
selected_range = range(0, 41000)
# selected_range = range(0, 5336)
plot = True # plot the coordinates
plot_raw = True
alpha = 0.25
time_stamps_vicon = [830, 3130, 4980, 6350, 7740, 9100, 10860, 13410, 15830, 18170, 20000, 21860, 24320, 26170, 27530, 29410, 31740, 33710, 35050, 36560, 39000]
# time_stamps_vicon = None
####################################################
## filter the data and resample
# base
base1 = filter_xyz('base1',calib_df.loc[:,['base1_x','base1_y','base1_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
base2 = filter_xyz('base2',calib_df.loc[:,['base2_x','base2_y','base2_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
base3 = filter_xyz('base3',calib_df.loc[:,['base3_x','base3_y','base3_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
# shoulder
shoulder1 = filter_xyz('shoulder1',calib_df.loc[:,['shoulder1_x','shoulder1_y','shoulder1_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
shoulder2 = filter_xyz('shoulder2',calib_df.loc[:,['shoulder2_x','shoulder2_y','shoulder2_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha) 
shoulder3 = filter_xyz('shoulder3',calib_df.loc[:,['shoulder3_x','shoulder3_y','shoulder3_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
shoulder4 = filter_xyz('shoulder4',calib_df.loc[:,['shoulder4_x','shoulder4_y','shoulder4_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
# gripper
gripper1 = filter_xyz('gripper1',calib_df.loc[:,['gripper1_x','gripper1_y','gripper1_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
gripper2 = filter_xyz('gripper2',calib_df.loc[:,['gripper2_x','gripper2_y','gripper2_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
gripper3 = filter_xyz('gripper3',calib_df.loc[:,['gripper3_x','gripper3_y','gripper3_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)

# force, moment, cop
force = filter_xyz('force',calib_df.loc[:,['F_x','F_y','F_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
moment = filter_xyz('moment',calib_df.loc[:,['M_x','M_y','M_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)
cop = filter_xyz('cop',calib_df.loc[:,['COP_x','COP_y','COP_z']].to_numpy()[selected_range], f_res, f_cutoff, plot, time_stamps_vicon, plot_raw, alpha)

####################################################
## create rigid body frame
base1_trans, base1_rot       = create_rigidbody_frame([base1, base2, base3], unit_rot=np.array([[0., 1., 0.], [1., 0., 0], [0, 0, -1]]))
base2_trans, base2_rot       = create_rigidbody_frame([base2, base1, base3], unit_rot=np.array([[0., -1., 0.], [1., 0., 0], [0, 0, 1]]))

gripper1_trans, gripper1_rot = create_rigidbody_frame([ gripper1, gripper2, gripper3,])
gripper2_trans, gripper2_rot = create_rigidbody_frame([ gripper2, gripper3, gripper1,])
gripper3_trans, gripper3_rot = create_rigidbody_frame([ gripper3, gripper2, gripper1,])

shoulder_trans, shoulder_rot = create_rigidbody_frame([shoulder1, shoulder4, shoulder2])

base = 'base2'
gripper = 'gripper1'
def shuffle_base_gripper(base, gripper):
    if base == 'base1':
        base_rot = base1_rot
        base_trans = base1_trans
    elif base == 'base2':
        base_rot = base2_rot
        base_trans = base2_trans

    if gripper == 'gripper1':
        gripper_rot = gripper1_rot
        gripper_trans = gripper1_trans
    elif gripper == 'gripper2':
        gripper_rot = gripper2_rot
        gripper_trans = gripper2_trans
    elif gripper == 'gripper3':
        gripper_rot = gripper3_rot
        gripper_trans = gripper3_trans
    return base_rot, base_trans, gripper_rot, gripper_trans

base_rot, base_trans, gripper_rot, gripper_trans = shuffle_base_gripper(base, gripper)


# plot frames
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(base1[0, 0], base1[0, 1], base1[0, 2], marker='o', label = 'base1')
ax.scatter(base2[0, 0], base2[0, 1], base2[0, 2], marker='o', label = 'base2')
ax.scatter(base3[0, 0], base3[0, 1], base3[0, 2], marker='o', label = 'base3')
ax.scatter(shoulder1[0, 0], shoulder1[0, 1], shoulder1[0, 2], marker='*', label = 'shoulder1')
ax.scatter(shoulder2[0, 0], shoulder2[0, 1], shoulder2[0, 2], marker='*', label = 'shoulder2')
ax.scatter(shoulder3[0, 0], shoulder3[0, 1], shoulder3[0, 2], marker='*', label = 'shoulder3')
ax.scatter(shoulder4[0, 0], shoulder4[0, 1], shoulder4[0, 2], marker='*', label = 'shoulder4')
ax.scatter(gripper1[0, 0], gripper1[0, 1], gripper1[0, 2], marker='>', label = 'gripper1')
ax.scatter(gripper2[0, 0], gripper2[0, 1], gripper2[0, 2], marker='>', label = 'gripper2')
ax.scatter(gripper3[0, 0], gripper3[0, 1], gripper3[0, 2], marker='>', label = 'gripper3')
ax.legend()
plot_SE3(pin.SE3.Identity())
plot_SE3(pin.SE3(base_rot[0], base_trans[0,:]), 'b')
plot_SE3(pin.SE3(shoulder_rot[0], shoulder_trans[0,:]), 's')
plot_SE3(pin.SE3(gripper_rot[0], gripper_trans[0,:]), 'g')

######################################################################################

######################################################################################
# path to joint encoder data
# print('Read encoder data from ', input_path_joint)
path_to_values = dir_path + 'introspection_datavalues.csv'
path_to_names = dir_path + 'introspection_datanames.csv'

# create a robot
robot = load_robot(
    abspath("urdf/tiago_48_schunk.urdf"),
    load_by_urdf=True,
)

# add object to gripper
# addBox_to_gripper(robot)

# read values from csv files
t_res, f_res, joint_names, q_abs_res, q_pos_res = get_q_arm(robot, path_to_values, path_to_names, f_cutoff)
active_joints = ["torso_lift_joint",
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

q_arm, dq_arm, ddq_arm = calc_vel_acc(robot, q_abs_res, selected_range, joint_names, f_res, f_cutoff)
q_arm = q_abs_res[:, actJoint_idx]
time_stamps = [960, 3367, 5100, 6500, 7860, 9260, 11580, 13500, 16000, 18400, 20320, 22300, 24590, 26530, 27860, 29740, 32320, 34170, 35520, 37000, 39000]

q_sel = np.zeros([len(time_stamps), q_arm.shape[1]])
for i, ti in enumerate(time_stamps):
    q_sel[i, :] = q_arm[ti, :]

fig_joint = plt.figure()
for ai in range(q_arm.shape[1]):
    plt.plot(np.arange(q_arm.shape[0]), q_arm[:, ai])
    plt.scatter(x=np.array(time_stamps), y=q_sel[:, ai], s=20)

######################################################################################

######################################################################################
####################################################
# convert to pinocchio frame
for base in ['base1', 'base2']:
    for gripper in ['gripper1', 'gripper2', 'gripper3']:      
        base_rot, base_trans, gripper_rot, gripper_trans = shuffle_base_gripper(base, gripper)
        base_frame = list()
        gripper_frame = list()
        shoulder_frame = list()
        for i, ti in enumerate(time_stamps_vicon):
            base_frame.append(pin.SE3(base_rot[ti], base_trans[ti, :]))
            gripper_frame.append(pin.SE3(gripper_rot[ti], gripper_trans[ti, :]))
            shoulder_frame.append(pin.SE3(shoulder_rot[ti], shoulder_trans[ti, :]))
        ## reproject end frame on to start frame
        gripper_pos = project_frame(gripper_frame, base_frame)
        gripper_shoulder = project_frame(gripper_frame, shoulder_frame)
        shoulder_base = project_frame(shoulder_frame, base_frame)

        save_selected_data(active_joints, gripper_pos, q_sel, join(dirname(str(abspath(__file__))), "data/calibration/mocap/vicon/absolute_encoder/vicon_calibration_fc10_{}_{}.csv".format(gripper, base)))
        save_selected_data(active_joints, gripper_shoulder, q_sel, join(dirname(str(abspath(__file__))), "data/calibration/mocap/vicon/absolute_encoder/vicon_calibration_fc10_{}_shoulder.csv".format(gripper)))
        save_selected_data(active_joints, shoulder_base, q_sel, join(dirname(str(abspath(__file__))), "data/calibration/mocap/vicon/absolute_encoder/vicon_calibration_fc10_shoulder_{}.csv".format(base)))


# TODO:
# reselect time_stamps and time_stamps_vicon
# calibrate -> identify shoulder frame in robot frame
# %matplotlib
# actJoint_idv = []
# fig, ax = plt.subplots(2,1)
# for act_j in active_joints:
#     joint_idx = robot.model.getJointId(act_j)
#     actJoint_idx.append(robot.model.joints[joint_idx].idx_q)
#     actJoint_idv.append(robot.model.joints[joint_idx].idx_v)
# for jqi in actJoint_idx:
#     ax[0].plot(q_arm[:, jqi])
# for ji in actJoint_idv:
#     ax[1].plot(dq_arm[:, ji])

# plt.grid()
# fig1, ax1 = plt.subplots(2,1)
# for i in range(3):
#     ax1[0].plot(gripper_trans[:, i])
#     ax1[1].plot(gripper_vel[:, i])
# ax1[0].grid()
# ax1[0].set_title('gripper position')
# ax1[1].grid()
# ax[1].set_title('gripper velocity')
