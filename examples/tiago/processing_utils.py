import csv
from re import split
import pandas as pd
import numpy as np
import rospy
# import dask.dataframe as dd
from scipy.optimize import least_squares

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


def create_rigidbody_frame(markers=[], unit_rot=None):
    """ create a rigid body frame from markers, in which:
        - the origin of the frame is the first element of markers list.
        - the x-axis is the vector from the first element to the second element of markers list.
        - the xy-plane is the plane formed by the first three elements of markers list.
        - the z-axis is the cross product of the x-axis and y-axis.
    Input: 
        - markers: a list of markers' position ndarray (n, 3)
    Output:
        - a list of origin coordinates.
        - a list of rotation matrices.
    """
    # create the origin of the frame
    origin = markers[0]
    # create the x-axis
    x_axis = markers[1] - markers[0]
    unit_x = np.zeros_like(x_axis)
    for i in range(x_axis.shape[0]):
        unit_x[i, :] = x_axis[i, :] / np.linalg.norm(x_axis[i, :])
    # create the z-axis
    m20 = markers[2] - markers[0]
    unit_z = np.zeros_like(unit_x)
    for i in range(unit_x.shape[0]):
        z_vec = np.cross(unit_x[i, :], m20[i, :])
        unit_z[i, :] = z_vec/ np.linalg.norm(z_vec)
    # create the y-axis
    unit_y = np.zeros_like(unit_x)
    for i in range(unit_x.shape[0]):
        y_vec = np.cross(unit_z[i, :], unit_x[i, :])
        unit_y[i, :] = y_vec/ np.linalg.norm(y_vec)
    # create the rotation matrix
    rot_mat = list()
    for i in range(origin.shape[0]):
        rot_ = np.zeros([3,3])
        rot_[:,0] = unit_x[i, :]
        rot_[:,1] = unit_y[i, :]
        rot_[:,2] = unit_z[i, :]
        if unit_rot is not None:
            rot_ = rot_.dot(unit_rot)
        rot_mat.append(rot_)

    assert len(origin) == len(rot_mat), "ERROR: origin and rotation matrix do not match!"
    return origin, rot_mat

def save_selected_data(active_joints, xyz, q, path_to_save):
    headers = ["x1", "y1", "z1"] + active_joints
    with open(path_to_save, "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(headers)
        for i in range(q.shape[0]):
            row = list(np.concatenate((xyz[i, :],
                                      q[i, :])))
            w.writerow(row)

# raw extracted data
########################################################################
# vicon
def read_csv_vicon(filename):
    vicon_xyz = pd.read_csv(filename, header=None, skiprows=1).to_numpy()
    vicon_headers = ['time',
                    'base1_x', 'base1_y', 'base1_z',
                    'base2_x', 'base2_y', 'base2_z',
                    'base3_x', 'base3_y', 'base3_z',
                    'shoulder1_x', 'shoulder1_y', 'shoulder1_z',
                    'shoulder2_x', 'shoulder2_y', 'shoulder2_z',
                    'shoulder3_x', 'shoulder3_y', 'shoulder3_z',
                    'shoulder4_x', 'shoulder4_y', 'shoulder4_z',
                    'gripper1_x', 'gripper1_y', 'gripper1_z',
                    'gripper2_x', 'gripper2_y', 'gripper2_z',
                    'gripper3_x', 'gripper3_y', 'gripper3_z',
                    'F_x', 'F_y', 'F_z',
                    'M_x', 'M_y', 'M_z',
                    'COP_x', 'COP_y', 'COP_z',]
    vicon_xyz = pd.DataFrame(vicon_xyz, columns=vicon_headers)
    return vicon_xyz
########################################################################

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
########################################################################
def get_XYZQUAT_marker(marker_frame_name, path_to_tf, t_res, f_res, f_cutoff, plot=True):
    """ Extract marker data from optitrack and resample to uniform sampling rate f_res,
    after filtering with median filter and butterworth filter.
    """
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
    if plot:
        plot_markertf(t_res, posXYZ_res, marker_frame_name)
    return posXYZ_res, quatXYZW_res


def filter_xyz(marker_frame_name, posXYZ, f_res, f_cutoff, plot=True, time_stamps=None, plot_raw=False, alpha=1):
    t_tf = np.arange(posXYZ.shape[0])/f_res
    # filter
    posXYZ_butf = np.zeros_like(posXYZ)
    for i in range(3):
        posXYZ_butf[:, i] = sig_filter(posXYZ[:, i], med_fil=5, nbutter=4, f_cutoff=f_cutoff, f_sample=120)

    # resample to f_res
    posXYZ_res = np.zeros((len(t_tf), 3))
    for i in range(3):
        posXYZ_res[:, i] = sig_resample(posXYZ_butf[:, i], t_tf, len(t_tf), f_res)
    if plot:
        if plot_raw == True:
            fig_, ax_ = plot_markertf(t_tf, posXYZ, marker_frame_name, alpha=alpha)
            fig, ax = plot_markertf(t_tf, posXYZ_res, marker_frame_name, fig_, ax_)
        else: 
            fig, ax = plot_markertf(t_tf, posXYZ_res, marker_frame_name)

        if time_stamps is not None:
            for ti in time_stamps:
                ax[0].scatter(ti/f_res, posXYZ_res[ti, 0], color='red')
                ax[1].scatter(ti/f_res, posXYZ_res[ti, 1], color='red')
                ax[2].scatter(ti/f_res, posXYZ_res[ti, 2], color='red')
    return posXYZ_res

# plot 
def plot_markertf(t_tf, posXYZ, marker_frame_name, fig=None, ax=None, alpha=1):
    if ax is None:
        fig, ax = plt.subplots(3, 1)
    ax[0].plot(t_tf, posXYZ[:,0], color='red', alpha=alpha)
    ax[1].plot(t_tf, posXYZ[:,1], color='blue', alpha=alpha)
    ax[2].plot(t_tf, posXYZ[:,2], color='green', alpha=alpha)
    ax[0].set_ylabel('x component')
    ax[1].set_ylabel('y component')
    ax[2].set_ylabel('z component')
    ax[2].set_xlabel('time')
    fig.suptitle('Marker Position of {}'.format(marker_frame_name))
    return fig, ax 

def plot_markerQuat(t_tf, quat, marker_frame_name, fig=[]):
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t_tf, quat[:,0], color='red')
    ax[1].plot(t_tf, quat[:,1], color='blue')
    ax[2].plot(t_tf, quat[:,2], color='green')
    ax[3].plot(t_tf, quat[:,3], color='black')
    ax[0].set_ylabel('x')
    ax[1].set_ylabel('y')
    ax[2].set_ylabel('z')
    ax[3].set_ylabel('w')
    ax[3].set_xlabel('time (s)')
    fig.suptitle('Marker Quaternion of {}'.format(marker_frame_name))

# cherry pick data
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

# project prj_frame onto ref_frame

def project_frame(prj_frame, ref_frame, rot=False):
    """ Project prj_frame onto ref_frame
    Input:
        - prj_frame: 7xn array of [xyzquaternion] or list of SE3
        - ref_frame: 7xn array of [xyzquaternion] or list of SE3
    Output:
        - projected_pos: nx3 array of xyz coordinates
        - projected_rot: nx3 array of rotation matrices
    """
    projected_rot = list()

    if isinstance(prj_frame[0], pin.SE3) and isinstance(ref_frame[0], pin.SE3):
        projected_pos = np.empty((len(prj_frame), 3))
        assert len(prj_frame) == len(ref_frame), "projecting two frames have different sizes! Projected positions are empty!"
        for i in range(len(prj_frame)):
            prj_se3 = prj_frame[i]
            ref_se3 = ref_frame[i]
            projected_se3 = pin.SE3.inverse(ref_se3)*prj_se3
            projected_pos[i, :] = projected_se3.translation
            projected_rot.append(projected_se3.rotation)

    elif isinstance(prj_frame[0], np.ndarray) and isinstance(ref_frame[0], np.ndarray):
        projected_pos = np.empty((prj_frame.shape[0], 3))
        assert prj_frame.shape == ref_frame.shape, "projecting two frames have different sizes! Projected positions are empty!"
        for i in range(prj_frame.shape[0]):
            ref_se3 = pin.XYZQUATToSE3(ref_frame[i, 1:])
            prj_se3 = pin.XYZQUATToSE3(prj_frame[i, 1:])
            projected_se3 = pin.SE3.inverse(ref_se3)*prj_se3
            projected_pos[i, :] = projected_se3.translation
            projected_rot.append(projected_se3.rotation)
    if rot:
        return projected_pos, projected_rot
    else:
        return projected_pos
    
def plot_3d_points(frame, fig=[]):
    """ plot 3d points with 3d coordinates
    """
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

def calc_angvel_fromQuatandDeriv(quat, dquat):
    """ Calculate rotation velocity from quaternion and its derivative
        quat = [x,y,z,w]
        dquat = [dx, dy, dz, dw]
    """

    angvel = np.zeros((dquat.shape[0], 3))
    for i in range(dquat.shape[0]):
        q0 = quat[i, 3]
        q1 = quat[i, 0]
        q2 = quat[i, 1]
        q3 = quat[i, 2] 
        mat_E = np.array([
            [-q1, q0, -q3, q2],
            [-q2, q3, q0, -q1],
            [-q3, -q2, q1, q0]])
        angvel[i, :] = 2*mat_E.dot(dquat[i, :])
    return angvel

def normalize_array(array, axis='row'):
    """ normalize each row of an array
    """
    norm_array = np.zeros_like(array)
    if axis == 'row':
        for i in range(array.shape[0]):
            norm_array[i, :] = array[i, :]/np.linalg.norm(array[i, :])
    elif axis == 'column':
        for i in range(array.shape[1]):
            norm_array[:, i] = array[:, i]/np.linalg.norm(array[:, i])
    return norm_array


def calc_dQuat(quat, f_res):
    """ Calculate derivative of quaternion
    """
    dquat = np.zeros((quat.shape[0], 4))
    for i in range(quat.shape[0]):
        if i == 0:
            dquat[i, :] = (quat[i+1, :] - quat[i, :])*f_res
        elif i == quat.shape[0] - 1:
            dquat[i, :] = (quat[i, :] - quat[i-1, :])*f_res
        else:
            dquat[i, :] = (quat[i+1, :] - quat[i-1, :])*f_res/2
    return dquat

def convert_XYZQUAT_to_SE3norm(var, se3_norm=True):
    """ Normalize quaternion and convert to SE3
    """
    SE3_out = pin.SE3.Identity()
    SE3_out.translation = var[0:3]

    quat_norm = pin.Quaternion(var[3:7]).normalize()
    if se3_norm:
        SE3_out.rotation = quat_norm.toRotationMatrix()
        return SE3_out
    else:
        var[3:7] = quat_norm.coeffs()
        return var
    
def create_R_matrix(NbSample, pos_res, pos_vel, rpy_res, ang_vel):
    """ Create a regressor matrix R for least square of size 6*NbSample x 12
        (12: 6 for translation and 6 for rotation)
    """
    R = np.zeros((6*NbSample, 12))
    # translation: fx, fy, fz
    for i_f in range(3):
        R[(i_f)*NbSample:(i_f+1)*NbSample, 2*i_f] = pos_res[:,i_f] # linear stiffness
        R[(i_f)*NbSample:(i_f+1)*NbSample, 2*i_f+1] = pos_vel[:,i_f] # linear damping
    # rotation: mx, my, mz
    for i_m in range(3): 
        R[(3 + i_m)*NbSample:(3 + i_m+1)*NbSample, 2*3 + 2*i_m] = rpy_res[:,i_m] # angular stiffness
        R[(3 + i_m)*NbSample:(3 + i_m+1)*NbSample, 2*3 + 2*i_m+1] = ang_vel[:,i_m] # angulardamping
    
    # force-contributation moment: m_fx, m_fy, m_fz, skew_matrix of r (3x3) * f (3x1)
    # # m_fx -> mx
    # R[(3)*NbSample:(4)*NbSample, 2] = np.multiply(-pos_res[:, 2], pos_res[:, 1]) # - z*y
    # R[(3)*NbSample:(4)*NbSample, 3] = np.multiply(-pos_res[:, 2], pos_vel[:, 1]) # - z*y'
    # R[(3)*NbSample:(4)*NbSample, 4] = np.multiply(pos_res[:, 1], pos_res[:, 2]) # y*z
    # R[(3)*NbSample:(4)*NbSample, 5] = np.multiply(pos_res[:, 1], pos_vel[:, 2]) # y*z'
    # # m_fy -> my
    # R[(4)*NbSample:(5)*NbSample, 0] = np.multiply(pos_res[:, 2], pos_res[:, 0]) # z*x
    # R[(4)*NbSample:(5)*NbSample, 1] = np.multiply(pos_res[:, 2], pos_vel[:, 0]) # z*x'
    # R[(4)*NbSample:(5)*NbSample, 4] = np.multiply(-pos_res[:, 0], pos_res[:, 2]) # - x*z
    # R[(4)*NbSample:(5)*NbSample, 5] = np.multiply(-pos_res[:, 0], pos_vel[:, 2]) # - x*z'
    # # m_fz -> mz
    # R[(5)*NbSample:(6)*NbSample, 0] = np.multiply(-pos_res[:, 1], pos_res[:, 0]) # - y*x
    # R[(5)*NbSample:(6)*NbSample, 1] = np.multiply(-pos_res[:, 1], pos_vel[:, 0]) # - y*x'
    # R[(5)*NbSample:(6)*NbSample, 2] = np.multiply(pos_res[:, 0], pos_res[:, 1]) # x*y
    # R[(5)*NbSample:(6)*NbSample, 3] = np.multiply(pos_res[:, 0], pos_vel[:, 1]) # x*y'
     # m_fx -> mx
    R[(3)*NbSample:(4)*NbSample, 2] = np.multiply(-pos_res[:, 2], pos_res[:, 1]) # - z*y
    R[(3)*NbSample:(4)*NbSample, 3] = np.multiply(-pos_res[:, 2], pos_vel[:, 1]) # - z*y'
    R[(3)*NbSample:(4)*NbSample, 4] = np.multiply(pos_res[:, 1], pos_res[:, 2]) # y*z
    R[(3)*NbSample:(4)*NbSample, 5] = np.multiply(pos_res[:, 1], pos_vel[:, 2]) # y*z'
    # m_fy -> my
    R[(4)*NbSample:(5)*NbSample, 0] = np.multiply(pos_res[:, 2], pos_res[:, 0]) # z*x
    R[(4)*NbSample:(5)*NbSample, 1] = np.multiply(pos_res[:, 2], pos_vel[:, 0]) # z*x'
    R[(4)*NbSample:(5)*NbSample, 4] = np.multiply(-pos_res[:, 0], pos_res[:, 2]) # - x*z
    R[(4)*NbSample:(5)*NbSample, 5] = np.multiply(-pos_res[:, 0], pos_vel[:, 2]) # - x*z'
    # m_fz -> mz
    R[(5)*NbSample:(6)*NbSample, 0] = np.multiply(-pos_res[:, 1], pos_res[:, 0]) # - y*x
    R[(5)*NbSample:(6)*NbSample, 1] = np.multiply(-pos_res[:, 1], pos_vel[:, 0]) # - y*x'
    R[(5)*NbSample:(6)*NbSample, 2] = np.multiply(pos_res[:, 0], pos_res[:, 1]) # x*y
    R[(5)*NbSample:(6)*NbSample, 3] = np.multiply(pos_res[:, 0], pos_vel[:, 1]) # x*y'
    

    return R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def create_tau_vector(tau):
    # estimate dynamics
    tau_vector = tau.flatten('F')
    return tau_vector

def cost_function(var, R, tau_vector):
    return R.dot(var) - tau_vector

def estimate_marker_pose_from_base(Mbase_marker, Mmarker0_mocap, q_base):
    # Mcm = Mc_m0*Mm_b*SE3(q_base)*Mb_m
    N_ = q_base.shape[0]
    q_marker = np.zeros((N_, 7))

    for i in range(N_):
        Mmarker_base = Mbase_marker.inverse()
        Mmocap_marker0 = Mmarker0_mocap.inverse()
        Mmarker = Mmocap_marker0*Mmarker_base*pin.XYZQUATToSE3(q_base[i, :])*Mbase_marker
        q_marker[i, :] = pin.SE3ToXYZQUAT(Mmarker)
        assert abs(np.linalg.norm(q_marker[i, 3:7]) - 1) < 1e-3, "Quaternion is not normalized!"
    return q_marker

def estimate_base_pose_from_marker(Mbase_marker, NbSample, Mmarker0_mocap, q_marker_ext, f_res):
    # estimate base pose from marker
    # q_marker: NbSamplex7 numpy array marker data
    # Mbase_marker: constant SE3 
    # Mmarker0_mocap: constant SE3
    # q_base: NbSamplex7 numpy array floating base pose

    # SE3 transformation from marker to floating base
    Mmarker_base = Mbase_marker.inverse()

    # padding two ends to avoid edge effects
    NbSample_ext = q_marker_ext.shape[0]

    # trim off the first and last 10 samples to avoid edge effects
    trimoff_idx = int((NbSample_ext - NbSample)/2)
    sample_range_int = range(trimoff_idx, NbSample_ext-trimoff_idx)

    # initialize pose of floating base
    q_base = np.zeros((NbSample_ext, 7))

    # estimate floating base pose in robot frame from marker pose in mocap frame
    for i in range(NbSample_ext):
        # SE3 transformation from mocap to marker
        Mmocap_marker = pin.XYZQUATToSE3(q_marker_ext[i, :])
        # SE3 transformation from resting base to floating base
        Mbase0_base = Mbase_marker*Mmarker0_mocap*Mmocap_marker*Mmarker_base
        # convert SE3 to quaternion and position
        q_base[i, :] = pin.SE3ToXYZQUAT(Mbase0_base)
        # flag to check if quaternion is normalized
        assert abs(np.linalg.norm(q_base[i, 3:7]) - 1) < 1e-3, "Quaternion is not normalized!"

    # convert quaternion to euler angles
    rpy_base = convert_quat_to_rpy(q_base[:, 3:7])

    # estimate velocity and acceleration of the base
    dq_base = np.zeros((NbSample_ext, 6))
    ddq_base = np.zeros((NbSample_ext, 6))

    # calculate differentiation by calc_deriv_fft
    for i in range(3):
        dq_base[:, i] = np.gradient(q_base[:, i])*f_res
        dq_base[:, 3 + i] = np.gradient(rpy_base[:, i])*f_res
    for i in range(6):
        ddq_base[:, i] = np.gradient(dq_base[:, i])*f_res

    # truncating the first and last 10 samples to avoid edge effects
    q_base = q_base[sample_range_int, :]
    dq_base = dq_base[sample_range_int, :]
    ddq_base = ddq_base[sample_range_int, :]
    rpy_base = rpy_base[sample_range_int, :]
    
    return q_base, dq_base, ddq_base, rpy_base

# 1/ input measures: q_marker, variables are Mmarker_base, var (12x1)
# 2/ then estimate q_base, dq_base, ddq_base
# 3/ then create_tau_vector
# 4/ then create_R_matrix
# 5/ then calculate cost_fucntion and perform least square
def cost_function_fb(var, robot_fb, Mmarker0, q_m, q_arm, dq_arm, ddq_arm, f_res, sol_found=False, UNKNOWN_BASE=None, base_input=None):
    
    """ Cost function is the error vector between predicted (force,torque) by the suspension model and estiamted
        (force,torque) from the robot inverse dynamics. The error vector is calculated by the following steps:
        1/ estimate kinematic quantities of the floating base
        2/ create arrays of kinematic quantities for whole body with fb
        3/ estimate dynamics which is the torque vector at free-flyer joint (floating base)
        4/ create R matrix
        5/ calculate cost function
        Args:
            var: variables (7+12)x1 vector
            robot_fb: robot model with free-flyer joint
            Mmarker0: SE3 transformation marker frame at rest to mocap frame
            q_m: NbSamplex7 numpy array marker data
            q_arm, dq_arm, ddq_arm: kinematic quantities of the arm
            f_res: sampling rate
            sol_found: boolean, if True, identification solution is found. If False, only cost function is calculated
    """
    # create SE3 transformation from floating base to marker from first 6 variables
    if UNKNOWN_BASE == 'xyzquat':
        assert base_input is None, "Mbase_marker is not correctly defined!"
        Mbase_marker_ = convert_XYZQUAT_to_SE3norm(var[0:7])
    elif UNKNOWN_BASE == 'xyz':
        assert np.array(base_input).shape[0] == 4, 'Quaternion is not correctly defined!'
        Mbase_marker_ = pin.SE3.Identity()
        Mbase_marker_.translation = var[0:3]
        Mbase_marker_.rotation = pin.rpy.rpyToMatrix(base_input)
    elif UNKNOWN_BASE == 'quat':
        assert np.array(base_input).shape[0] == 3, 'XYZ is not correctly defined!'
        Mbase_marker_ = pin.SE3.Identity()
        Mbase_marker_.translation = base_input
        Mbase_marker_.rotation = pin.Quaternion(var[0:4]).toRotationMatrix()
    else:
        assert np.array(base_input).shape[0] == 7, "Mbase_marker is not correctly defined!"
        Mbase_marker_ = convert_XYZQUAT_to_SE3norm(np.array(base_input))

    # create variables for suspension model from last 12 variables
    var_susp = var[-12:]

    NbSample = q_arm.shape[0]


    # estimate kinematic quantities of the floating base
    q_base, dq_base, ddq_base, rpy_base = estimate_base_pose_from_marker(Mbase_marker_, NbSample, Mmarker0, q_m, f_res)

    # create arrays of kinematic quantities for whole body with fb
    q_fb = np.concatenate((q_base, q_arm), axis=1)
    dq_fb = np.concatenate((dq_base, dq_arm), axis=1)
    ddq_fb = np.concatenate((ddq_base, ddq_arm), axis=1)

    # estimate dynamics
    tau_fb = np.zeros((NbSample, 6))
    for i in range(NbSample):
        pin.computeCentroidalMomentumTimeVariation(robot_fb.model, robot_fb.data, q_fb[i, :], dq_fb[i, :], ddq_fb[i, :])
        tau_fb[i, :] = robot_fb.data.f[1].vector
    tau_vector = tau_fb.flatten('F')

    # create R matrix
    R = create_R_matrix(NbSample, q_base[:, 0:3], dq_base[:, 0:3], rpy_base, dq_base[:, 3:6])
 
    # calculate cost function
    if sol_found:
        return R.dot(var_susp), tau_vector, q_base, dq_base, ddq_base, rpy_base, R
    else:
        return R.dot(var_susp) - tau_vector

def convert_quat_to_rpy(quat):
    rpy = np.zeros((len(quat), 3))
    for i in range(len(quat)):
        assert abs(np.linalg.norm(quat[i, :]) - 1) < 1e-2, "Quaternion is not normalized! Norm is {}".format(np.linalg.norm(quat[i, :]))
        rpy[i, :] = pin.rpy.matrixToRpy(pin.Quaternion(quat[i, :]).toRotationMatrix())
    return rpy


########################################################################

def get_q_arm(robot, path_to_values, path_to_names, f_cutoff):
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
    return t_res, f_res, joint_names, q_abs_res, q_pos_res

########################################################################
def calc_vel_acc(robot, q_abs_res, sample_range, joint_names, f_res, f_cutoff):
    # take extra 10 samples to both ends of sample to range
    external_range = range(sample_range[0]-10, sample_range[-1] + 1 +10)
    q_arm = q_abs_res[external_range, :]
    internal_range = range(10, q_arm.shape[0]-10)
    NbSample = q_arm.shape[0]

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
    
    q_arm = q_arm[internal_range, :]
    dq_arm = dq_arm[internal_range, :]
    ddq_arm = ddq_arm[internal_range, :]
    return q_arm, dq_arm, ddq_arm 


########################################################################
def calc_derivatives(posXYZ_res, f_res):
    NbSample = posXYZ_res.shape[0]
        
    # calculate marker velocities
    pos_vel = np.zeros((NbSample, 3))
    for i in range(posXYZ_res.shape[1]):
        pos_vel[:, i] = np.gradient(posXYZ_res[:, i])*f_res

    # calculate marker accelerations
    pos_acc = np.zeros((NbSample, 3))
    for i in range(posXYZ_res.shape[1]):
        pos_acc[:, i] = np.gradient(pos_vel[:, i])*f_res
    return posXYZ_res, pos_vel, pos_acc

def calc_fb_vel_acc(posXYZ_res, quatXYZW_res, sample_range, f_res):
    # concatenate marker positions and orientations
    # take extra 10 samples to both ends of sample to range
    external_range = range(sample_range[0]-10, sample_range[-1] + 1 +10)
    q_marker = np.concatenate((posXYZ_res[external_range, :], quatXYZW_res[external_range, :]), axis=1)
    internal_range = range(10, q_marker.shape[0]-10)
    NbSample = q_marker.shape[0]
    
    rpy_marker = convert_quat_to_rpy(q_marker[:, 3:7])
    
    # calculate marker velocities
    pos_vel = np.zeros((NbSample, 3))
    for i in range(3):
        pos_vel[:, i] = np.gradient(q_marker[:, i])*f_res

    # marker angular velocities by rpy
    angrpy_vel = np.zeros((NbSample, 3))
    for i in range(3):
        angrpy_vel[:, i] = np.gradient(rpy_marker[:, i])*f_res

    dq_marker = np.concatenate((pos_vel, angrpy_vel), axis=1)

    # calculate marker accelerations
    ang_acc = np.zeros((NbSample, 3))
    pos_acc = np.zeros((NbSample, 3))
    for i in range(3):
        pos_acc[:, i] = np.gradient(pos_vel[:, i])*f_res
        ang_acc[:, i] = np.gradient(angrpy_vel[:, i])*f_res

    ddq_marker = np.concatenate((pos_acc, ang_acc), axis=1)
    
    # truncating the first and last 10 samples to avoid edge effects
    q_marker_ext = q_marker.copy()
    q_marker = q_marker[internal_range, :]
    dq_marker = dq_marker[internal_range, :]
    ddq_marker = ddq_marker[internal_range, :]

    # convert quaternion to euler angles
    rpy_marker = convert_quat_to_rpy(q_marker[:, 3:7])
    Mmarker0_mocap = get_Mmarker_mocap( q_marker, rpy_marker)
    return q_marker, dq_marker, ddq_marker, rpy_marker, Mmarker0_mocap, q_marker_ext

########################################################################
# get SE3 inverse
def get_SE3_inv(SE3_obj):
    SE3_inv_hom = np.linalg.inv(SE3_obj.homogeneous)
    SE3_inv = pin.SE3.Identity()
    SE3_inv.rotation = SE3_inv_hom[0:3,0:3]
    SE3_inv.translation = SE3_inv_hom[0:3,3]
    return SE3_inv

# convert quaternion to rpy
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

def addBox_to_gripper(robot):
    boxjoint_id = robot.model.getJointId('gripper_finger_joint')
    inertiabox = pin.Inertia.FromBox(0.437, 12*1e-2, 22*1e-2, 4.5*1e-2)
    robot.model.appendBodyToJoint(boxjoint_id, inertiabox, pin.SE3.Identity())
    pin.computeCentroidalMomentum(robot.model, robot.data)

########################################################################

# visualize with spatialmath-python
from spatialmath.base import *
import spatialmath as sm

def plot_marker_and_ee(sample_id):
    plt.figure()
    T_shoulder = sm.SE3(posXYZ_res[sample_id,:])
    T_shoulder.RPY(rpy_shoulder[sample_id, :])
    T_shoulder.plot(frame='marker', color='rgb')
    T_ee = sm.SE3(posXYZ_ee[sample_id, :])
    T_ee.RPY(rpy_ee[sample_id, :])
    T_ee.plot(frame='ee', color='rgb')
    T_base = sm.SE3(posXYZ_base[sample_id, :])
    T_base.RPY(rpy_bse[sample_id, :])
    T_base.plot(frame='base', color='rgb')
    plt.show()
def plot_marker_serial(sample_range, posXYZ, rpy):
    T_series = []
    for sample_id in sample_range:
        if sample_id%100==0:
            T = sm.SE3(posXYZ[sample_id,:])
            T.RPY(rpy[sample_id, :])
            T_series.append(T)
            tranimate(T_series, repeat=True, interval=1, movie=True)
    plt.show()
def plot_marker_serial(sample_range, posXYZ, rpy):
    T_series = []
    for sample_id in sample_range:
        if sample_id%10==0:
            T = sm.SE3(posXYZ[sample_id,:])
            T.RPY(rpy[sample_id, :])
            T_series.append(T)
    for i, Tk in enumerate(T_series):
        if i>0:
            Tk.animate(start=T_series[i-1])

def plot_SE3(SE3_pin, frame_name=None, ax=None):
    """ Plot SE3 object or a list of SE3 objects from pinocchio 
    """
    from spatialmath import SE3
    if frame_name is None:
        frame_name = 'world'
    T = SE3(np.array(SE3_pin.homogeneous))
    print('rotation of {}: '.format(frame_name) , T.R)
    print('translation of {}: '.format(frame_name), T.t)
    T.plot( frame=frame_name, color='rgb', ax=ax)

def find_isa(q_marker, dq_marker, f_res):
    dquat_marker = calc_dQuat(q_marker[:, 3:7], f_res)
    angvel_marker = calc_angvel_fromQuatandDeriv(q_marker[:, 3:7], dquat_marker)
    linvel_marker = dq_marker[:, 0:3]
    posrel_marker = q_marker[:, 0:3]
    u_marker, s_marker = find_int_screw_axis(angvel_marker, linvel_marker, posrel_marker)
    return u_marker, s_marker

def find_int_screw_axis(angvel, linvel, pos_rel):
    unit_vec = normalize_array(angvel)
    s = np.zeros((unit_vec.shape[0], 3))
    for i in range(unit_vec.shape[0]):
        s[i, :] = pos_rel[i, :] + np.cross(angvel[i, :], linvel[i, :])/(np.linalg.norm(angvel[i, :])**2)
    return unit_vec, s