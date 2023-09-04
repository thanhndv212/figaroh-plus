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
from figaroh.tools.robot import Robot

from figaroh.calibration.calibration_tools import (get_rel_transform, rank_in_configuration)
from figaroh.tools.robot import Robot 

from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer
import time
def extract_t_list(path_to_tf):
    """ Extracts list of timestampts where samples were recorded
        """
    pass


def test_readCSV(path_to_values, path_to_names):
    # read names and values from csv to dataframe~
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
    df_cols = df.columns
    if "child_frame_id" in df_cols:
        frame_col = "child_frame_id"
    else:
        frame_col = "frame_id"

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
    print(frame_val)
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

    starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec() # mark up t0
    # starting_t = 0
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
    print(names[0], names[-1])
    print("total number of columns in data_names: ", len(names))

    # joint_idx (list): get indices of corresponding to active joints
    # if value_names = [], extract all available values
    if not value_names:
        value_names = names

    joint_idx = []
    nonexist = []
    for element in value_names:
        if element in names:
            joint_idx.append(names.index(element))
        else:
            print(element, "Mentioned joint is not present in the names list.")
            nonexist.append(element)
    for element in nonexist:
        value_names.remove(element)
    print("Joint indices corresponding to active joints: ", joint_idx)

    # joint_val (np.darray): split data in "values" column (str) to numpy array
    # extracted_val (np.darray):extract only values of interest from joint_val
    extracted_val = np.empty((len(t_idx), len(joint_idx)))
    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')
    print("name of elements in values:",len(split_data))
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

        # remove "-" in names
        # for name in value_names:
        #     if "- " in name:
        #         value_names[value_names.index(name)] = name.replace("- ", "")

        val_dict = dict.fromkeys(['t'] + value_names)
        val_dict['t'] = t_val
        # slicing along axis 1 given value_idx
        for i in range(len(joint_idx)):
            extracted_val[:, i] = joint_val[:, joint_idx[i]]
            val_dict[value_names[i]] = joint_val[:, joint_idx[i]]
    return val_dict


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

def fourier_series(sig, sampling_rate, sig_period, plot_first=10):
    from scipy import fftpack
    sample_freq = fftpack.fftfreq(sig.size, d=1/sampling_rate)
    sig_fft = fftpack.fft(sig)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    power = np.abs(sig_fft)[pidxs]
    plt.figure()
    plt.plot(freqs, power, color="#e41a1c")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('plower')
    axes = plt.axes([0.3, 0.3, 0.5, 0.5])
    plt.title('Peak frequency')
    plt.plot(freqs[:plot_first], power[:plot_first], color="#e41a1c")
    plt.setp(axes, yticks=[])
    freq = freqs[power.argmax()]
    np.allclose(freq, 1./sig_period)  # check that correct freq is found
    sig_fft[np.abs(sample_freq) > freq] = 0
    main_sig = fftpack.ifft(sig_fft)
    plt.figure()
    plt.plot(sig, color="#e41a1c")
    plt.plot(np.real(main_sig), linewidth=3, color="#377eb8")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

def sig_filter(sig, med_fil, nbutter, f_cutoff, f_sample):
    from scipy import signal
    # medium filtering to remove noise
    sig_medf = signal.medfilt(sig, med_fil)

    # low pass filter by butterworth
    b1, b2 = signal.butter(nbutter, f_cutoff/(f_sample/2), 'low')
    sig_lpf = signal.filtfilt(
        b1, b2, sig_medf, axis=0, padtype="odd", padlen=3 * (max(len(b1), len(b2)) - 1)
    )
    return sig_lpf

def sig_resample(sig, timestamps, desired_length, f_sample):
    """
    sig: original signal
    timestamps: timestamps of original signal
    desired_length: length of resampled signal
    f_sample: sampling frequency of desired signal
    """
    from scipy import signal
    # resample to desired length
    t_new = np.linspace(timestamps[0], timestamps[-1], desired_length)
    sig_res = signal.resample(sig, desired_length)
    return sig_res


def plot_markertf(t_tf, posX, posY, posZ, fig=[]):
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t_tf, posX, color='red')
    ax[1].plot(t_tf, posY, color='blue')
    ax[2].plot(t_tf, posZ, color='green')
    ax[0].set_ylabel('x')
    ax[1].set_ylabel('y')
    ax[2].set_ylabel('z')
    ax[2].set_xlabel('time')
        
def calc_deriv_fft(signal, sample_rate):
    from scipy.fft import fft, ifft, fftfreq
    y = signal
    N = y.shape[0]
    dx = 1/sample_rate
    L = dx*(N-1)
    k = (2*np.pi)*fftfreq(len(y), d=dx)
    dydx = ifft(1j*k*fft(y)).real
    return dydx

def calc_angvel_fromQuat(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

def create_R_matrix(NbSample, pos_res, pos_vel, rpy_res, ang_vel):
    R = np.zeros((6*NbSample, 12))
    # translation
    for i in range(3):
        R[(i)*NbSample:(i+1)*NbSample, 2*i] = pos_res[:,i] - np.mean(pos_res[:,i])
        R[(i)*NbSample:(i+1)*NbSample, 2*i+1] = pos_vel[:,i]
    # rotation
    for i in range(3):
        R[(3 + i)*NbSample:(3 + i+1)*NbSample, 2*3 + 2*i] = rpy_res[:,i] - np.mean(rpy_res[:,i])
        R[ (3 + i)*NbSample:(3 + i+1)*NbSample, 2*3 + 2*i+1] = ang_vel[:,i]
    return R

def create_tau_vector(tau):
    # estimate dynamics
    tau_vector = tau.flatten('F')
    return tau_vector

def cost_function(var, R, tau_vector):
    return R.dot(var) - tau_vector

def estimate_base_pose_from_marker(Mbase_marker, NbSample, Mmarker0_mocap, q_marker, f_res):
    # estimate base pose from marker
    # q_marker: NbSamplex7 numpy array marker data
    # Mbase_marker: constant SE3 
    # Mmarker0_mocap: constant SE3
    # q_base: NbSamplex7 numpy array floating base pose

    q_base = np.zeros((NbSample, 7))
    Mmarker_base_hom = np.linalg.inv(Mbase_marker.homogeneous)
    Mmarker_base = pin.SE3.Identity()
    Mmarker_base.rotation = Mmarker_base_hom[0:3,0:3]
    Mmarker_base.translation = Mmarker_base_hom[0:3,3]

    # estimate base pose from marker
    for i in range(NbSample):
        Mbase0_base = Mbase_marker*Mmarker0_mocap*pin.XYZQUATToSE3(q_marker[i, :])*Mmarker_base
        q_base[i, :] = pin.SE3ToXYZQUAT(Mbase0_base)
    
    # convert quaternion to euler angles
    rpy = convert_quat_to_rpy(q_base[:, 3:7])

    # estimate velocity and acceleration of the base
    dt = 1/f_res
    dq_base = np.zeros((NbSample, 6))
    ddq_base = np.zeros((NbSample, 6))

    # calculate differentiation by calc_deriv_fft
    for i in range(3):
        dq_base[:, i] = np.gradient(q_base[:, i])*f_res
        dq_base[:, 3 + i] = np.gradient(rpy[:, i])*f_res
    for i in range(6):
        ddq_base[:, i] = np.gradient(dq_base[:, i])*f_res
    
    return q_base, dq_base, ddq_base, rpy

# 1/ input measures: q_marker, variables are Mmarker_base, var (12x1)
# 2/ then estimate q_base, dq_base, ddq_base
# 3/ then create_tau_vector
# 4/ then create_R_matrix
# 5/ then calculate cost_fucntion and perform least square
def cost_function_fb(var, robot_fb, Mmarker0_mocap, q_marker, NbSample, q_arm, dq_arm, ddq_arm, f_res):
    # var: variables (7+12)x1 vector
    # robot_fb: robot model with free-flyer joint
    # q_marker: NbSamplex7 numpy array
    # sample_range: list of indices of samples to be used for estimation

    Mbase_marker = convert_XYZQUAT_to_SE3norm(var[0:7])

    var_susp = var[7:19]

    # estimate kinematic quantities of the floating base
    q_base, dq_base, ddq_base, rpy = estimate_base_pose_from_marker(Mbase_marker, NbSample, Mmarker0_mocap, q_marker, f_res)

    # create arrays of kinematic quantities for whole body with fb
    # concatenate base and arm joints
    q_fb = np.concatenate((q_base, q_arm), axis=1)
    dq_fb = np.concatenate((dq_base, dq_arm), axis=1)
    ddq_fb = np.concatenate((ddq_base, ddq_arm), axis=1)

    # estimate dynamics
    tau_fb = np.zeros((NbSample, 6))
    for i in range(NbSample):
        pin.computeCentroidalMomentumTimeVariation(robot_fb.model, robot_fb.data, q_fb[i, :], dq_fb[i, :], ddq_fb[i, :])
        tau_fb[i, :] = robot_fb.data.dhg.vector
    tau_vector = tau_fb.flatten('F')

    # create R matrix
    R = create_R_matrix(NbSample, q_base[:, 0:3], dq_base[:, 0:3], rpy, dq_base[:, 3:6])

    # calculate cost function
    return R.dot(var_susp) - tau_vector

def calc_results(var, robot_fb, Mmarker0_mocap, q_marker, NbSample, q_arm, dq_arm, ddq_arm, f_res):
    # var: variables (7+12)x1 vector
    # robot_fb: robot model with free-flyer joint
    # q_marker: NbSamplex7 numpy array
    # sample_range: list of indices of samples to be used for estimation

    Mbase_marker = convert_XYZQUAT_to_SE3norm(var[0:7])

    var_susp = var[7:19]

    # estimate kinematic quantities of the floating base
    q_base, dq_base, ddq_base, rpy = estimate_base_pose_from_marker(Mbase_marker, NbSample, Mmarker0_mocap, q_marker, f_res)

    # create arrays of kinematic quantities for whole body with fb
    # concatenate base and arm joints
    q_fb = np.concatenate((q_base, q_arm), axis=1)
    dq_fb = np.concatenate((dq_base, dq_arm), axis=1)
    ddq_fb = np.concatenate((ddq_base, ddq_arm), axis=1)

    # estimate dynamics
    tau_fb = np.zeros((NbSample, 6))
    for i in range(NbSample):
        pin.computeCentroidalMomentumTimeVariation(robot_fb.model, robot_fb.data, q_fb[i, :], dq_fb[i, :], ddq_fb[i, :])
        tau_fb[i, :] = robot_fb.data.dhg.vector
    tau_vector = tau_fb.flatten('F')

    # create R matrix
    R = create_R_matrix(NbSample, q_base[:, 0:3], dq_base[:, 0:3], rpy, dq_base[:, 3:6])

    # calculate cost function
    return R.dot(var_susp), tau_vector

def convert_quat_to_rpy(quat):
    rpy = np.zeros((len(quat), 3))
    for i in range(len(quat)):
        rpy[i, :] = pin.rpy.matrixToRpy(pin.Quaternion(quat[i, :]).normalize().toRotationMatrix())
    return rpy

########################################################################

ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')

## define path to data

# folder = 'single_oscilation_around_xfold_weight_2023-07-28-13-33-20' # x fold weight
folder = 'single_oscilation_around_yfold_weight_2023-07-28-13-12-05' # y fold weight
# folder = 'single_oscilation_around_z_weight_2023-07-28-13-07-11' # z weight
dir_path = '/home/thanhndv212/Downloads/experiment_data/suspension/bags/selected_data/' + folder + '/'
path_to_values = dir_path + 'introspection_datavalues.csv'
path_to_names = dir_path + 'introspection_datanames.csv'
path_to_tf = dir_path + 'natnet_rostiago_shoulderpose.csv'

# create a robot
robot = Robot(
    'data/tiago_schunk.urdf',
    package_dirs= package_dirs,
    # isFext=True  # add free-flyer joint at base
)

# constant 
f_cutoff = 0.628 # lowpass filter cutoff freq
f_q = 100
f_tf = 120


def get_q_arm(robot, path_to_values, path_to_names):
    ########################################################################
    abs_encoders = ['- {}_absolute_encoder_position'.format(name) for name in robot.model.names[1:]]
    positions = ['- {}_position'.format(name) for name in robot.model.names[1:]]

    encoder_dict = extract_instrospection(
        path_to_values, path_to_names, abs_encoders, [])
    position_dict = extract_instrospection(
        path_to_values, path_to_names, positions, []
    )

    ########################################################################

    # joint data
    # original sampling rate
    t_q = encoder_dict['t']

    q_abs = np.zeros((len(t_q), robot.model.nq))
    q_pos = np.zeros((len(t_q), robot.model.nq))

    # get encoder values and joint positions only for active joints
    joint_names = []
    for name in robot.model.names[1:]:
        joint_idx = robot.model.getJointId(name)
        idx_q = robot.model.joints[joint_idx].idx_q
        for key in encoder_dict.keys():
            if name in key:
                joint_names.append(name)
                q_abs[:, idx_q] = encoder_dict['- {}_absolute_encoder_position'.format(name)]
                q_pos[:, idx_q] = position_dict['- {}_position'.format(name)]

    # fill nan with 0
    q_abs = np.nan_to_num(q_abs, nan=0)
    q_pos = np.nan_to_num(q_pos, nan=0)

    # resample timestamps to uniform sampling rate f_res
    t_res = np.linspace(t_q[0], t_q[-1], len(t_q))
    f_res = 1/(t_res[1] - t_res[0])

    # filter 
    q_abs_butf = np.zeros_like(q_abs)
    q_pos_butf = np.zeros_like(q_pos)
    for i in range(q_abs.shape[1]):
        q_abs_butf[:, i] = sig_filter(q_abs[:, i], med_fil=5, nbutter=4, f_cutoff=f_cutoff, f_sample=f_res)
        q_pos_butf[:, i] = sig_filter(q_pos[:, i], med_fil=5, nbutter=4, f_cutoff=f_cutoff, f_sample=f_res)

    # resample to uniform sampling rate f_res
    q_abs_res = np.zeros_like(q_abs)
    q_pos_res = np.zeros_like(q_pos)
    for i in range(q_abs.shape[1]):
        q_abs_res[:, i] = sig_resample(q_abs_butf[:, i], t_q, len(t_res), f_res)
        q_pos_res[:, i] = sig_resample(q_pos_butf[:, i], t_q, len(t_res), f_res)
    plot_markertf(t_res, q_abs[:, 24], q_abs_butf[:, 24], q_abs_res[:, 24])
    return t_res, f_res, joint_names, q_abs_res
########################################################################
def get_XYZQUAT_marker(robot, t_res, f_res):
    # marker data
    shoulder_pose = extract_tf(path_to_tf, frame_names=['"world"'])

    # original sampling rate
    t_tf = shoulder_pose['"world"'][:, 0]

    posXYZ = shoulder_pose['"world"'][:, 1:4]
    quatXYZW = shoulder_pose['"world"'][:, 4:8]

    # filter
    posXYZ_butf = np.zeros_like(posXYZ)
    quatXYZW_butf = np.zeros_like(quatXYZW)
    for i in range(3):
        posXYZ_butf[:, i] = sig_filter(posXYZ[:, i], med_fil=5, nbutter=4, f_cutoff=f_cutoff, f_sample=120)
    for i in range(4):
        quatXYZW_butf[:, i] = sig_filter(quatXYZW[:, i], med_fil=5, nbutter=4, f_cutoff=f_cutoff, f_sample=120)

    # resample to f_res
    posXYZ_res = np.zeros((len(t_res), 3))
    quatXYZW_res = np.zeros((len(t_res), 4))
    for i in range(3):
        posXYZ_res[:, i] = sig_resample(posXYZ_butf[:, i], t_tf, len(t_res), f_res)
    for i in range(4):
        quatXYZW_res[:, i] = sig_resample(quatXYZW_butf[:, i], t_tf, len(t_res), f_res)
    plot_markertf(t_res, posXYZ_res[:, 0], posXYZ_res[:, 1], posXYZ_res[:, 2])
    return posXYZ_res, quatXYZW_res
########################################################################
################################ SAMPLE RANGE ##########################
########################################################################
t_res, f_res, joint_names, q_abs_res = get_q_arm(robot, path_to_values, path_to_names)
posXYZ_res, quatXYZW_res = get_XYZQUAT_marker(robot, t_res, f_res)
# only take samples in a range
sample_range = range(2500, 4000)
NbSample = len(sample_range)

# processed data
t_sample = t_res[sample_range]
q_arm = q_abs_res[sample_range, :]
q_marker = np.concatenate((posXYZ_res[sample_range, :], quatXYZW_res[sample_range, :]), axis=1)

def get_SE3_inv(SE3_obj):
    SE3_inv = pin.SE3.Identity()
    SE3_inv.rotation = SE3_obj.rotation.transpose()
    SE3_inv.translation = -SE3_obj.rotation.transpose()*SE3_obj.translation
    return SE3_inv
# convert quaternion to rpy
rpy = convert_quat_to_rpy(q_marker[:, 3:7])
def get_Mmarker_mocap( q_marker, rpy):
    # average position of marker
    rpy = convert_quat_to_rpy(q_marker[:, 3:7])
    q0_marker = np.zeros(6)
    for i in range(3):
        q0_marker[i] = np.mean(q_marker[:, i])
        q0_marker[3 + i] = np.mean(rpy[:, i])
    Mmocap_marker0 = pin.SE3.Identity()
    Mmocap_marker0.rotation = pin.rpy.rpyToMatrix(q0_marker[3:6])
    Mmocap_marker0.translation = q0_marker[0:3]
    Mmarker0_mocap = Mmocap_marker0.inverse()
    return Mmarker0_mocap
Mmarker0_mocap = get_Mmarker_mocap( q_marker, rpy)
########################################################################
def calc_vel_acc(robot, q, NbSample, joint_names, f_res, f_cutoff):
    # joint velocities for active joints
    dq_arm = np.zeros((NbSample, robot.model.nv))
    for jname in joint_names:
        # config index and velocity index
        joint_idv = robot.model.joints[robot.model.getJointId(jname)].idx_v
        joint_idq = robot.model.joints[robot.model.getJointId(jname)].idx_q

        # calculate joint velocities
        dq_arm[:, joint_idv] = np.gradient(q_arm[:, joint_idq])*f_res

    # filter velocities
    for i in range(dq_arm.shape[1]):
        dq_arm[:, i] = sig_filter(dq_arm[:, i], med_fil=5, nbutter=4, f_cutoff=f_cutoff, f_sample=f_res)

    # joint accelerations for active joints
    ddq_arm = np.zeros((NbSample, robot.model.nv))
    for i in range(robot.model.nv):
        ddq_arm[:, i] = np.gradient(dq_arm[:, i])*f_res
    return dq_arm, ddq_arm 
dq_arm, ddq_arm = calc_vel_acc(robot, q_arm, NbSample, joint_names, f_res, f_cutoff)
# body dynamics without floating base
def get_centroid_tau(robot, NbSample, q, dq, ddq):
    """
    robot: robot model
    NbSample: number of samples
    q: joint positions
    dq: joint velocities
    ddq: joint accelerations
    return: tau_centroid: centroidal momentum time variation
    """
    tau_centroid = np.zeros((NbSample, 6))
    for i in range(NbSample):
        pin.computeCentroidalMomentumTimeVariation(robot.model, robot.data, q[i, :], dq[i, :], ddq[i, :])
        tau_centroid[i, :] = robot.data.dhg.vector
    return tau_centroid
tau_centroid = get_centroid_tau(robot, NbSample, q_arm, dq_arm, ddq_arm)
# tau_centroid = np.zeros((NbSample, 6))
# for i in range(NbSample):
#     pin.computeCentroidalMomentumTimeVariation(robot.model, robot.data, q_arm[i, :], dq_arm[i, :], ddq_arm[i, :])
#     tau_centroid[i, :] = robot.data.dhg.vector

########################################################################
def calc_fb_vel_acc(q_marker, NbSample, f_res):
    # calculate marker velocities
    pos_vel = np.zeros((NbSample, 3))
    for i in range(3):
        pos_vel[:, i] = np.gradient(q_marker[:, i])*f_res

    # marker angular velocities by rpy
    angrpy_vel = np.zeros((NbSample, 3))
    for i in range(3):
        angrpy_vel[:, i] = np.gradient(rpy[:, i])*f_res

    dq_marker = np.concatenate((pos_vel, angrpy_vel), axis=1)

    # calculate marker accelerations
    ang_acc = np.zeros((NbSample, 3))
    pos_acc = np.zeros((NbSample, 3))
    for i in range(3):
        pos_acc[:, i] = np.gradient(pos_vel[:, i])*f_res
        ang_acc[:, i] = np.gradient(angrpy_vel[:, i])*f_res

    ddq_marker = np.concatenate((pos_acc, ang_acc), axis=1)
    return dq_marker, ddq_marker
dq_marker, ddq_marker = calc_fb_vel_acc(q_marker, NbSample, f_res)
########################################################################

# body dynamics with floating base

# concatenate base and arm joints
q_fb = np.concatenate((q_marker, q_arm), axis=1)
dq_fb = np.concatenate((dq_marker, dq_arm), axis=1)
ddq_fb = np.concatenate((ddq_marker, ddq_arm), axis=1)

# create floating base model
tiago_fb = Robot(
    'data/tiago_schunk.urdf',
    package_dirs= package_dirs,
    isFext=True  # add free-flyer joint at base
)

# body dynamics without base
tau_fb = get_centroid_tau(tiago_fb, NbSample, q_fb, dq_fb, ddq_fb)
# tau_fb = np.zeros((NbSample, 6))
# for i in range(NbSample):
#     pin.computeCentroidalMomentumTimeVariation(tiago_fb.model, tiago_fb.data, q_fb[i, :], dq_fb[i, :], ddq_fb[i, :])
#     tau_fb[i, :] = tiago_fb.data.dhg.vector


########################################################################
from scipy.optimize import least_squares

# get dynamics measurems within sample range
# tau_centroid_vector = create_tau_vector(tau_centroid)
# tau_fb_vector = create_tau_vector(tau_fb)

# identification
# R = create_R_matrix(NbSample, q_marker[:, 0:3], pos_vel, rpy, angrpy_vel)
# tau_mea_vector = tau_fb_vector

# least squares
# var_init = np.ones(12)*100

# cost_function_fb(var, robot_fb, q_marker, sample_range, q_abs_res, dq_abs, ddq_abs, t_res, f_res)
# LM_solve = least_squares(cost_function, var_init, verbose = 1, args=(R, tau_mea_vector))
# print(LM_solve.x)

# # estimate from solution
# tau_predict_vector = R.dot(LM_solve.x)

# identification 2
var_init_fb = np.zeros(19)
var_init_fb[0:3] = np.ones(3)
var_init_fb[3:7] = np.zeros(4)
var_init_fb[7:19] = np.zeros(12)

LM_solve = least_squares(cost_function_fb, var_init_fb, method='lm', verbose=1, args=(tiago_fb, Mmarker0_mocap, q_marker, NbSample, q_arm, dq_arm, ddq_arm, f_res))

# # estimate from solution
tau_predict_vector, tau_mea_vector = calc_results(LM_solve.x, tiago_fb, Mmarker0_mocap, q_marker, NbSample, q_arm, dq_arm, ddq_arm, f_res)

print(LM_solve.x)
# Mbase_marker = pin.XYZQUATToSE3(LM_solve.x[0:7])
def convert_XYZQUAT_to_SE3norm(var):
    SE3_out = pin.SE3.Identity()
    Mbase_marker.translation = var[0:3]
    quat_norm = pin.Quaternion(var[3:7]).normalize()
    SE3_out.rotation = quat_norm.toRotationMatrix()
    return SE3_out
Mbase_marker = convert_XYZQUAT_to_SE3norm(LM_solve.x[0:7])
# plot dynamics
fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].plot(tau_mea_vector[i*len(sample_range):(i+1)*len(sample_range)], color='red')
    ax[i].plot(tau_predict_vector[i*len(sample_range):(i+1)*len(sample_range)], color='blue')
    ax[i].set_ylabel('tau')
    ax[i].set_xlabel('time')
    ax[i].legend(['estimate', 'predicted'])
plt.show()












# # find difference between two encoder readings at two instants

# t_bf = 20
# t_af = 50

# excld_joint = 'leg_right_2' # exceptionally drifting on absolute encoder
# # pierre commented: absolute encoder has no influence on controller

# t_pg = [t_bf, t_af]
# pg_bf = []
# pg_af = []
# q_bf = []
# q_af = []

# for i in range(1, len(encoder_dict)):
#     if excld_joint not in list(encoder_dict.keys())[i]:
#         pg_bf.append(list(encoder_dict.values())[i][t_bf] \
#             - list(position_dict.values())[i][t_bf])
#         pg_af.append(list(encoder_dict.values())[i][t_af] \
#             - list(position_dict.values())[i][t_af])
#     q_bf.append(list(position_dict.values())[i][t_bf])
#     q_af.append(list(position_dict.values())[i][t_af])
# names = []
# for name in robot.model.names[1:]:
#     if 'leg_right_2' not in name:
#         names.append(name)
# idx = np.arange(len(names))
# width = 0.4

# fig, ax = plt.subplots()
# ax.barh(idx, pg_bf, width, color='red', label='drifting at pregrasp before contact')
# ax.barh(idx + width, pg_af, width, color='blue', label='drifting at pregrasp after contact')

# ax.set(yticks=idx+width, yticklabels=names, ylim=[2*width-1, len(names)])
# ax.legend()
# plt.grid()
# plt.show()

# model = robot.model
# data = robot.data
# pin.framesForwardKinematics(model, data, np.array(q_bf))
# pin.updateFramePlacements(model, data)
# leftEE_bf = get_rel_transform(model, data, 'left_sole_link', 'gripper_left_joint')
# pin.framesForwardKinematics(model, data, np.array(q_af))
# pin.updateFramePlacements(model, data)
# leftEE_af = get_rel_transform(model, data, 'left_sole_link', 'gripper_left_joint')


# write to csv
# save_csv(t_list, EE_prj_sample, actJoint_val,
#          f"talos/sample.csv", side='right')
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

