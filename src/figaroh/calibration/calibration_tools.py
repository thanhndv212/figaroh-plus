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

from os.path import abspath, dirname, join
import numpy as np
from scipy.optimize import approx_fprime, least_squares
from scipy.linalg import svd, svdvals
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import time
import pprint
import matplotlib.pyplot as plt
import numdifftools as nd
# import quadprog as qp
import pandas as pd
from ..tools.regressor import eliminate_non_dynaffect
from ..tools.qrdecomposition import (
    get_baseParams,
    get_baseIndex,
    build_baseRegressor,
    cond_num)
from ..tools.robot import Robot

######################## INITIALIZATION TOOLS ########################################
# pinocchio does not parse axis of motion, below is munually imported from urdf of talos
# arranged in the same order with model.joints no free_flyer
AXIS_MOTION =[
    [None, None, None],  # universe joint
    # leg left
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    # leg right
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    # torso
    [0, 0, 1],
    [0, 1, 0],
    # arm left
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    # arm right
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    # head
    [0, 1, 0],
    [0, 0, 1]
]


def get_param_from_yaml(robot, calib_data):

    # NOTE: since joint 0 is universe and it is trivial,
    # indices of joints are different from indices of joint configuration,
    # different from indices of joint velocities

    # robot_name: anchor as a reference point for executing
    robot_name = robot.model.name

    # End-effector sensing measurability: a BOOLEAN vector of 6 stands for px,
    # py, pz, phix, phiy, phiz (mocap/camera/laser tracker/close-loop)
    # number of "True" = calb_idx
    NbMarkers = len(calib_data['markers'])
    measurability = calib_data['markers'][0]['measure']
    calib_idx = measurability.count(True)

    # Calibration model: level 1: joint_offset (only joint offsets),
    # level 2: full_params(geometric parameters),
    # level 3: non_geom(non-geometric parameters)

    # Kinematic chain: base frame: start_frame, end-effector frame: end_frame
    start_frame = calib_data['base_frame']  # default
    end_frame = calib_data['tool_frame']

    frames = [f.name for f in robot.model.frames]
    assert (start_frame in frames), "Start_frame {} does not exist.".format(start_frame)

    assert (end_frame in frames), "End_frame {} does not exist.".format(end_frame)

    # q0: default zero configuration
    q0 = robot.q0
    NbSample = calib_data['nb_sample']

    # IDX_TOOL: frame ID of the tool
    IDX_TOOL = robot.model.getFrameId(end_frame)

    # tool_joint: ID of the joint right before the tool's frame (parent)
    tool_joint = robot.model.frames[IDX_TOOL].parent

    # indices of active joints: from base to tool_joint (exclude the first universe joint)
    actJoint_idx = get_sup_joints(robot.model, start_frame, end_frame)

    # indices of joint configuration corresponding to active joints
    config_idx = [robot.model.joints[i].idx_q for i in actJoint_idx]

    # number of active joints
    NbJoint = len(actJoint_idx)

    # optimizing variable in optimization code
    x_opt_prev = np.zeros([NbJoint])
    
    

    # list of calibrating parameters name
    param_name = []
    if calib_data['non_geom']:
        # list of elastic gain parameter names
        elastic_gain = []
        axis = ['kx', 'ky', 'kz']
        for j_id, joint_name in enumerate(robot.model.names.tolist()):
            if joint_name == 'universe':
                axis_motion = 'null'
            else:
                for ii, ax in enumerate(AXIS_MOTION[j_id]):
                    if ax == 1:
                        axis_motion = axis[ii]
            elastic_gain.append(axis_motion+'_'+joint_name)
        for i in actJoint_idx:
            param_name.append(elastic_gain[i])

    param = {
        'robot_name': robot_name,
        'q0': q0,
        'x_opt_prev': x_opt_prev,
        'NbSample': NbSample,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'IDX_TOOL': IDX_TOOL,
        'tool_joint': tool_joint,
        'eps': 1e-3,
        'config_idx': config_idx,
        'actJoint_idx': actJoint_idx,
        'PLOT': 0,
        'NbMarkers': NbMarkers,
        'calib_model': calib_data['calib_level'],  # 'joint_offset' / 'full_params'
        'measurability': measurability,
        'calibration_index': calib_idx,  # 3 / 6
        'NbJoint': NbJoint,
        'free_flyer': calib_data['free_flyer'],
        'param_name': param_name,
        'non_geom': calib_data['non_geom'],
    }
    pprint.pprint(param)
    return param


def get_param(robot, NbSample, start_frame='universe',
              end_frame='ee_marker_joint', NbMarkers=1,
              calib_model='full_params',
              free_flyer=False, non_geom=False, EE_measurability=None):
    """ Read a .yaml config, then contain parsed parameters in a dictionary.
    """
    # NOTE: since joint 0 is universe and it is trivial,
    # indices of joints are different from indices of joint configuration,
    # different from indices of joint velocities

    # robot_name: anchor as a reference point for executing
    robot_name = robot.model.name

    # End-effector sensing measurability: a BOOLEAN vector of 6 stands for px,
    # py, pz, phix, phiy, phiz (mocap/camera/laser tracker/close-loop)
    # number of "True" = calb_idx
    
    calib_idx = EE_measurability.count(True)

    # Calibration model: level 1: joint_offset (only joint offsets),
    # level 2: full_params(geometric parameters),
    # level 3: non_geom(non-geometric parameters)
    calib_model = calib_model

    # Kinematic chain: base frame: start_frame, end-effector frame: end_frame
    start_frame = start_frame  # default
    end_frame = end_frame

    frames = [f.name for f in robot.model.frames]
    assert (start_frame in frames), "Start_frame {} does not exist.".format(start_frame)

    assert (end_frame in frames), "End_frame {} does not exist.".format(end_frame)

    # q0: default zero configuration
    q0 = robot.q0

    # IDX_TOOL: frame ID of the tool
    IDX_TOOL = robot.model.getFrameId(end_frame)

    # tool_joint: ID of the joint right before the tool's frame (parent)
    tool_joint = robot.model.frames[IDX_TOOL].parent

    # indices of active joints: from base to tool_joint (exclude the first universe joint)
    actJoint_idx = get_sup_joints(robot.model, start_frame, end_frame)

    # indices of joint configuration corresponding to active joints
    config_idx = [robot.model.joints[i].idx_q for i in actJoint_idx]

    # number of active joints
    NbJoint = len(actJoint_idx)

    # optimizing variable in optimization code
    x_opt_prev = np.zeros([NbJoint])
    
    # list of elastic gain parameter names
    elastic_gain = []
    axis = ['kx', 'ky', 'kz']
    for j_id, joint_name in enumerate(robot.model.names.tolist()):
        if joint_name == 'universe':
            axis_motion = 'null'
        else:
            for ii, ax in enumerate(AXIS_MOTION[j_id]):
                if ax == 1:
                    axis_motion = axis[ii]
        elastic_gain.append(axis_motion+'_'+joint_name)

    # list of calibrating parameters name
    param_name = []
    if non_geom:
        for i in actJoint_idx:
            param_name.append(elastic_gain[i])

    param = {
        'robot_name': robot_name,
        'q0': q0,
        'x_opt_prev': x_opt_prev,
        'NbSample': NbSample,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'IDX_TOOL': IDX_TOOL,
        'tool_joint': tool_joint,
        'eps': 1e-3,
        'config_idx': config_idx,
        'actJoint_idx': actJoint_idx,
        'PLOT': 0,
        'NbMarkers': NbMarkers,
        'calib_model': calib_model,  # 'joint_offset' / 'full_params'
        'measurability': EE_measurability,
        'calibration_index': calib_idx,  # 3 / 6
        'NbJoint': NbJoint,
        'free_flyer': free_flyer,
        'param_name': param_name,
        'non_geom': non_geom,
    }
    print(param)
    return param


def get_joint_offset(joint_names):
    """ This function give a dictionary of joint offset parameters.
            Input:  joint_names: a list of joint names (from model.names)
            Output: joint_off: a dictionary of joint offsets.
    """
    joint_off = []

    for i, name in enumerate(joint_names):
        joint_off.append("off_" + name)

    phi_jo = [0] * len(joint_off)  # default zero values
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off


def get_geo_offset(joint_names):
    """ This function give a dictionary of variations (offset) of kinematics parameters.
            Input:  joint_names: a list of joint names (from model.names)
            Output: geo_params: a dictionary of variations of kinematics parameters.
    """
    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
    geo_params = []

    for i in range(len(joint_names)):
        for j in tpl_names:
            # geo_params.append(j + ("_%d" % i))
            geo_params.append(j + "_" + joint_names[i])

    phi_gp = [0] * len(geo_params)  # default zero values
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params

# TODO: to add to tools

def add_pee_name(param):
    PEE_names = []
    pee_tpl = ['pEEx', 'pEEy', 'pEEz', 'phiEEx', 'phiEEy','phiEEz']
    for i in range(param['NbMarkers']):
        for j, state in enumerate(param['measurability']):
            if state:
                PEE_names.extend(['{}_{}'.format(pee_tpl[j], i+1)])
    param['param_name'] = param['param_name'] + PEE_names

    
def add_eemarker_frame(frame_name, p, rpy, model, data):
    """ Adds a frame at the end_effector.
    """
    p = np.array([0.1, 0.1, 0.1])
    R = pin.rpy.rpyToMatrix(rpy)
    frame_placement = pin.SE3(R, p)

    parent_jointId = model.getJointId("arm_7_joint")
    prev_frameId = model.getFrameId("arm_7_joint")
    ee_frame_id = model.addFrame(
        pin.Frame(frame_name, parent_jointId, prev_frameId, frame_placement, pin.FrameType(0), pin.Inertia.Zero()), False)
    return ee_frame_id


######################## DATA PROCESSING TOOLS ########################################
# data collected and saved in various formats and structure, it has
# always been a pain in the ass to handles. Need to think a way to simplify
# and optimize this procedure.
def read_config_data(model, path_to_file):
    df = pd.read_csv(path_to_file)
    q = np.zeros([len(df), model.njoints-1])
    for i in range(len(df)):
        for j, name in enumerate(model.names[1:].tolist()):
            jointidx = rank_in_configuration(model, name)
            q[i, jointidx] = df[name][i]
    return q


def extract_expData(path_to_file, param):
    # first 7 cols: coordinates xyzquat of end effector by mocap
    xyz_rotQuat = pd.read_csv(
        path_to_file, usecols=list(range(0, 7))).to_numpy()

    # next 8 cols: joints position
    q_act = pd.read_csv(path_to_file, usecols=list(range(7, 15))).to_numpy()

    # delete the 4th data sample/outlier
    xyz_rotQuat = np.delete(xyz_rotQuat, 4, 0)
    q_act = np.delete(q_act, 4, 0)

    param['NbSample'] = q_act.shape[0]

    # extract measured end effector coordinates
    PEEm_exp = np.empty((param['calibration_index'], param['NbSample']))
    for i in range(param['NbSample']):
        PEE_se3 = pin.XYZQUATToSE3(xyz_rotQuat[i, :])
        PEEm_exp[0:3, i] = PEE_se3.translation
        if param['calibration_index'] == 6:
            PEEm_exp[3:6, i] = pin.rpy.matrixToRpy(PEE_se3.rotation)
    PEEm_exp = PEEm_exp.flatten('C')

    # extract measured joint configs
    q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
    for i in range(param['NbSample']):
        q_exp[i, 0:8] = q_act[i, :]
        # ATTENTION: need to check robot.q0 vs TIAGo.q0
        q_exp[i, 8:] = param['q0'][8:]
    return PEEm_exp, q_exp


def load_data(path_to_file, model, param, del_list=[]):
    """ Read a csv file into dataframe by pandas, then transform to the form
    of full joint configuration and markers' position/location.
    NOTE: indices matter! Pay attention.
        Input:  path_to_file: str, path to csv
                param: Param, a class contain neccesary constant info.
        Output: np.ndarray, joint configs
                1D np.ndarray, markers' position/location 
        Csv headers: 
                i-th marker position: xi, yi, zi
                i-th marker orientation: phixi, phiyi, phizi (not used atm)
                active joint angles: 
                    tiago: torso, arm1, arm2, arm3, arm4, arm5, arm6, arm7
                    talos: torso1, torso2, armL1, armL2, armL3, armL4, armL5, armL6, armL7
    """
    # read_csv
    df = pd.read_csv(path_to_file)

    # create headers for marker position
    PEE_headers = []
    pee_tpl = ['x', 'y', 'z', 'phix','phiy','phiz']
    for i in range(param['NbMarkers']):
        for j, state in enumerate(param['measurability']):
            if state:
                PEE_headers.extend(['{}{}'.format(pee_tpl[j], i+1)])

    # create headers for joint configurations
    joint_headers = [model.names[i] for i in param['actJoint_idx']]
    print(joint_headers)
    # check if all created headers present in csv file
    csv_headers = list(df.columns)
    for header in (PEE_headers + joint_headers):
        if header not in csv_headers:
            print("%s does not exist in the file." % header)
            break

    # Extract marker position/location
    xyz_4Mkr = df[PEE_headers].to_numpy()

    # Extract joint configurations
    q_act = df[joint_headers].to_numpy()

    # remove bad data
    if del_list:
        xyz_4Mkr = np.delete(xyz_4Mkr, del_list, axis=0)
        q_act = np.delete(q_act, del_list, axis=0)

    # update number of data points
    param['NbSample'] = q_act.shape[0]

    PEEm_exp = xyz_4Mkr.T
    PEEm_exp = PEEm_exp.flatten('C')

    q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
    for i in range(param['NbSample']):
            config = param['q0']
            config[param['config_idx']] = q_act[i, :]
            q_exp[i, :] = config

    return PEEm_exp, q_exp


######################## COMMON TOOLS ########################################


def rank_in_configuration(model, joint_name):
    """ Get index of a given joint in joint configuration vector
    """
    assert joint_name in model.names, 'Given joint name does not exist.'
    jointId = model.getJointId(joint_name)
    joint_idx = model.joints[jointId].idx_q
    return joint_idx


def cartesian_to_SE3(X):
    """ Convert (6,) cartesian coordinates to SE3
            Input: 1D (6,) numpy array
            Output: SE3 placement
    """
    X = np.array(X)
    X = X.flatten('C')
    translation = X[0:3]
    rot_matrix = pin.rpy.rpyToMatrix(X[3:6])
    placement = pin.SE3(rot_matrix, translation)
    return placement


def get_rel_transform(model, data, start_frame, end_frame):
    """ Calculate relative transformation between any two frames
        in the same kinematic structure in pinocchio
            Output: SE3 placement
        Note: assume framesForwardKinematics and updateFramePlacements already
        updated
    """
    # update frames given a configuration
    # if q is None:
    #     pass
    # else:
    #     pin.framesForwardKinematics(model, data, q)
    #     pin.updateFramePlacements(model, data)
    frames = [f.name for f in model.frames]
    assert (start_frame in frames), \
        "{} does not exist.".format(start_frame)
    assert (end_frame in frames), \
        "{} does not exist.".format(end_frame)
    # assert (start_frame != end_frame), "Two frames are identical."
    # transformation from base link to start_frame
    start_frameId = model.getFrameId(start_frame)
    oMsf = data.oMf[start_frameId]
    # transformation from base link to end_frame
    end_frameId = model.getFrameId(end_frame)
    oMef = data.oMf[end_frameId]
    # relative transformation from start_frame to end_frame
    sMef = oMsf.actInv(oMef)
    return sMef


def get_sup_joints(model, start_frame, end_frame):
    """ Find supporting joints between two frames
            Output: a list of supporting joints' Id 
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)
    start_par = model.frames[start_frameId].parent
    end_par = model.frames[end_frameId].parent
    branch_s = model.supports[start_par].tolist()
    branch_e = model.supports[end_par].tolist()
    # remove 'universe' joint from branches
    if model.names[branch_s[0]] == 'universe':
        branch_s.remove(branch_s[0])
    if model.names[branch_e[0]] == 'universe':
        branch_e.remove(branch_e[0])
    
    # find over-lapping joints in two branches
    shared_joints = list(set(branch_s) & set(branch_e))
    # create a list of supporting joints between two frames
    list_1 = [x for x in branch_s if x not in branch_e]
    list_1.reverse()
    list_2 = [y for y in branch_e if y not in branch_s]
    # case 2: root_joint is fixed joint; branch_s and branch_e are completely separate
    if shared_joints == []:
        sup_joints = list_1 + list_2
    else:
        # case 1: branch_s is part of branch_e
        if shared_joints == branch_s:
            sup_joints = [branch_s[-1]] + list_2
        else:
            assert shared_joints != branch_e, "End frame should be before start frame."
            # case 3: there are overlapping joints between two branches
            sup_joints = list_1 + [shared_joints[-1]] + list_2
    return sup_joints


def get_rel_kinreg(model, data, start_frame, end_frame, q):
    """ Calculate kinematic regressor between start_frame and end_frame
            Output: 6x6n matrix 
    """ 
    sup_joints = get_sup_joints(model, start_frame, end_frame)
    # update frameForwardKinematics and updateFramePlacements
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # a set-zero kinreg
    kinreg = np.zeros((6, 6*(model.njoints-1)))
    frame = model.frames[model.getFrameId(end_frame)]
    oMf = data.oMi[frame.parent]*frame.placement
    for p in sup_joints:
        oMp = data.oMi[model.parents[p]]*model.jointPlacements[p]
        # fMp = get_rel_transform(model, data, end_frame, model.names[p])
        fMp = oMf.actInv(oMp)
        fXp = fMp.toActionMatrix()
        kinreg[:, 6*(p-1):6*p] = fXp
    return kinreg

def get_rel_jac(model, data, start_frame, end_frame, q):
    """ Calculate frameJacobian between start_frame and end_frame
            Output: 6xn matrix 
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)
    
    # update frameForwardKinematics and updateFramePlacements
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # relative Jacobian
    J_start = pin.computeFrameJacobian(
        model, data, q, start_frameId, pin.LOCAL)
    J_end = pin.computeFrameJacobian(
        model, data, q, end_frameId, pin.LOCAL)
    J_rel = J_end - J_start
    return J_rel
######################## LEVENBERG-MARQUARDT TOOLS ########################################


def get_LMvariables(param, mode=0, seed=0):
    """ Create a initial zero/range-bounded random search varibale for Leverberg-Marquardt algo.
    """
    # initialize all variables at zeros
    nvar = len(param['param_name'])
    if mode==0:
        var = np.zeros(nvar)
    elif mode==1:
        var = np.random.uniform(-seed, seed, nvar)
    return var, nvar


def update_forward_kinematics(model, data, var, q, param):
    """ Update jointplacements with offset parameters, recalculate forward kinematics
        to find end-effector's position and orientation.
    """
    # read param['param_name'] to allocate offset parameters to correct SE3
    # convert translation: add a vector of 3 to SE3.translation
    # convert orientation: convert SE3.rotation 3x3 matrix to vector rpy, add
    #  to vector rpy, convert back to to 3x3 matrix
    axis_tpl = ['d_px', 'd_py', 'd_pz', 'd_phix', 'd_phiy', 'd_phiz']
    elas_tpl = ['kx', 'ky', 'kz']
    pee_tpl = ['pEEx', 'pEEy', 'pEEz', 'phiEEx', 'phiEEy','phiEEz']
    # order of joint in variables are arranged as in param['actJoint_idx']
    assert len(var) == len(param['param_name']), "Length of variables != length of params"
    param_dict = dict(zip(param['param_name'], var))
    origin_model = model.copy()

    # update model.jointPlacements
    updated_params = []
    for j_id in param['actJoint_idx']:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy with kinematic errors
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)
        model = update_joint_placement(model, j_id, xyz_rpy)
    PEE = np.zeros((param['calibration_index'], param['NbSample']))

    # get transform
    q_ = np.copy(q)
    for i in range(param['NbSample']):
        pin.framesForwardKinematics(model, data, q_[i, :])
        pin.updateFramePlacements(model, data)
        # update model.jointPlacements with joint elastic error
        if param['non_geom']:
            tau = pin.computeGeneralizedGravity(model, data, q_[i,:]) # vector size of 32 = nq < njoints
            # update xyz_rpy with joint elastic error
            for j_id in param['actJoint_idx']:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1] # nq = njoints -1 
                if j_name in key:
                    for elas_id, elas in enumerate(elas_tpl):
                        if elas in key:
                            param_dict[key] = param_dict[key]*tau_j
                            xyz_rpy[elas_id + 3] += param_dict[key] # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, xyz_rpy)
            # get relative transform with updated model
            oMee = get_rel_transform(model, data, param['start_frame'], param['end_frame'])
            # revert model back to origin from added joint elastic error
            for j_id in param['actJoint_idx']:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1] # nq = njoints -1 
                if j_name in key:
                    for elas_id, elas in enumerate(elas_tpl):
                        if elas in key:
                            param_dict[key] = param_dict[key]*tau_j
                            xyz_rpy[elas_id + 3] += param_dict[key] # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, -xyz_rpy)

        else:
            oMee = get_rel_transform(model, data, param['start_frame'], param['end_frame'])


        # update last frame if there is 
        if len(updated_params) < len(param_dict):
            pee = np.zeros(6)
            for n_id in range(len(updated_params), len(param_dict)):
                for axis_id, axis in enumerate(pee_tpl):
                    if axis in param['param_name'][n_id]:
                        pee[axis_id] = var[n_id]
            eeMf = cartesian_to_SE3(pee)
            oMf = oMee*eeMf
        else:
            oMf = oMee
        
        # final transform
        trans = oMf.translation.tolist()
        orient = pin.rpy.matrixToRpy(oMf.rotation).tolist()
        loc = trans + orient
        measure = []
        for mea_id, mea in enumerate(param['measurability']):
            if mea:
                measure.append(loc[mea_id])
        PEE[:, i] = np.array(measure)
    PEE = PEE.flatten('C')
    # revert model back to original 
    assert origin_model.jointPlacements != model.jointPlacements, 'before revert'
    for j_id in param['actJoint_idx']:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        xyz_rpy[axis_id] = param_dict[key]
        model = update_joint_placement(model, j_id, -xyz_rpy)

    assert origin_model.jointPlacements != model.jointPlacements, 'after revert'

    return PEE


def update_joint_placement(model, joint_idx, xyz_rpy):
    """ Update joint placement given a vector of 6 offsets.
    """
    tpl_translation = model.jointPlacements[joint_idx].translation
    tpl_rotation = model.jointPlacements[joint_idx].rotation
    tpl_orientation = pin.rpy.matrixToRpy(tpl_rotation)
    #update axes 
    updt_translation = tpl_translation + xyz_rpy[0:3]
    updt_orientation = tpl_orientation + xyz_rpy[3:6]
    updt_rotation = pin.rpy.rpyToMatrix(updt_orientation)
    #update placements
    model.jointPlacements[joint_idx].translation = updt_translation
    model.jointPlacements[joint_idx].rotation = updt_rotation
    return model


def init_var(param, mode=0, base_model=True):
    """ Creates variable vector, mode = 0: initial guess, mode = 1: predefined values(randomized)
    """
    # x,y,z,r,p,y from mocap to base ( 6 parameters for pos and orient)
    # create artificial offsets
    if mode == 0:
        # 6D base frame
        # qBase_0 = np.array([0.0, 0., 0., 0., 0., 0.])
        # tiago + mocap ref frame
        # qBase_0 = np.array([0.5245, 0.3291, -0.02294, 0., 0., 0.])
        # tiago + base ref frame
        qBase_0 = np.array([0.01647, 0.22073, -0.33581, 0., 0., 0.])

        # parameter variation at joints
        if param['calib_model'] == 'joint_ offset':
            offset_0 = np.zeros(param['NbJoint'])
        elif param['calib_model'] == 'full_params':
            offset_0 = np.zeros(param['NbJoint']*6)
        # markers variables
        qEE_0 = np.full((param['NbMarkers']*param['calibration_index'],), 0.0)

    elif mode == 1:
        # 6D base frame
        # talos
        if param['robot_name'] == 'talos':
            qBase_0 = np.array([-0.16, 0.047, 0.16, 0., 0., 0.])
        # tiago
        elif param['robot_name'] == 'tiago':
            qBase_0 = np.array([0.5245, 0.3291, -0.02294, 0., 0., 0.])
        else:
            qBase_0 = np.array([0., 0., 0., 0., 0., 0.])

        # parameter variation at joints
        if param['calib_model'] == 'joint_ offset':
            offset_0 = np.random.uniform(-0.005, 0.005, (param['NbJoint'],))
        elif param['calib_model'] == 'full_params':
            offset_0 = np.random.uniform(-0.005, 0.005, (param['NbJoint']*6,))
        # markers variables
        qEE_0 = np.full((param['NbMarkers']*param['calibration_index'],), 0)
    # robot_name = "tiago"
    # robot_name = "talos"
    zero_list = []
    if param['robot_name'] == 'tiago':
        # create list of parameters to be set as zero for tiago, respect the order
        # TODO: to be imported from a config file
        torso_list = [0, 1, 2, 3, 4, 5]
        arm1_list = [6, 7, 8, 11]
        arm2_list = [13, 16]
        arm3_list = [19, 22]
        arm4_list = [24, 27]
        arm5_list = [30, 33]
        arm6_list = [36, 39]
        arm7_list = [43, 46]  # include phiz7
        total_list = [torso_list, arm1_list, arm2_list, arm3_list, arm4_list,
                      arm5_list, arm6_list, arm7_list]
        zero_list = np.concatenate(total_list)
    elif param['robot_name'] == 'talos':
        # create list of parameters to be set as zero for tiago, respect the order
        # TODO: to be imported from a config file
        torso1_list = [0, 1, 2, 3, 4, 5]
        torso2_list = [2, 5]
        arm1_list = [1, 4]
        arm2_list = [2, 5]
        arm3_list = [0, 3]
        arm4_list = [2, 5]
        arm5_list = [1, 4]
        arm6_list = [2, 5]
        arm7_list = [0, 3]  # include phiz7
        total_list = [torso1_list, torso2_list, arm1_list, arm2_list,
                      arm3_list, arm4_list, arm5_list, arm6_list, arm7_list]
        for i in range(len(total_list)):
            total_list[i] = np.array(total_list[i]) + i*6
        zero_list = np.concatenate(total_list)
    print("list of elements to be set zero: ", zero_list)

    # remove parameters are set to zero (dependent params)
    if base_model == True:
        offset_0 = np.delete(offset_0, zero_list, None)
        # x,y,z,r,p,y from wrist to end_effector ( 6 parameters for pos and orient)

    var = np.append(np.append(qBase_0, offset_0), qEE_0)
    nvars = var.shape[0]
    return var, nvars


def get_PEE_fullvar(var, q, model, data, param, noise=False, base_model=True):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
        var: an 1D array containing estimated offset parameters since scipy.optimize
        only takes 1D array variables. Reshape to ((1 + NbJoints + NbMarkers, 6)), replacing zeros 
        those missing.
        base_model: bool, choice to choose using base parameters to estimate ee's coordinates.
        Use jointplacement to add offset to 6 axes of joint
    """
    PEE = []

    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']    
    q_temp = np.copy(q)

    # reshape variable vector to vectors of 6
    NbFrames = 1 + param['NbJoint'] + param['NbMarkers']

    if param['robot_name'] == 'tiago':
        if not base_model:
            var_rs = np.reshape(var, (NbFrames, 6))
        elif base_model:
            var_rs = np.zeros((NbFrames, 6))

            # first 6 params of base frame var_rs[0:6,:]
            var_rs[0, 0: 6] = var[0: 6]

            # offset parameters var_rs[1:(1+Nbjoint),:]
            if param['NbJoint'] > 0:
                # torso
                var_rs[1, :] = np.zeros(6)
                if param['NbJoint'] > 1:
                    # arm 1
                    var_rs[2, 3] = var[6]
                    var_rs[2, 4] = var[7]
                    if param['NbJoint'] > 2:
                        # arm 2
                        var_rs[3, 0] = var[8]
                        var_rs[3, 2] = var[9]
                        var_rs[3, 3] = var[10]
                        var_rs[3, 5] = var[11]
                        if param['NbJoint'] > 3:
                            # arm3
                            var_rs[4, 0] = var[12]
                            var_rs[4, 2] = var[13]
                            var_rs[4, 3] = var[14]
                            var_rs[4, 5] = var[15]
                            if param['NbJoint'] > 4:
                                # arm4
                                var_rs[5, 1] = var[16]
                                var_rs[5, 2] = var[17]
                                var_rs[5, 4] = var[18]
                                var_rs[5, 5] = var[19]
                                if param['NbJoint'] > 5:
                                    # arm5
                                    var_rs[6, 1] = var[20]
                                    var_rs[6, 2] = var[21]
                                    var_rs[6, 4] = var[22]
                                    var_rs[6, 5] = var[23]
                                    if param['NbJoint'] > 6:
                                        # arm6
                                        var_rs[7, 1] = var[24]
                                        var_rs[7, 2] = var[25]
                                        var_rs[7, 4] = var[26]
                                        var_rs[7, 5] = var[27]
                                        if param['NbJoint'] > 7:
                                            # arm7
                                            var_rs[8, 0] = var[28]
                                            var_rs[8, 2] = var[29]
                                            var_rs[8, 3] = var[30]
                                            var_rs[8, 5] = var[31]
            # 32 base parameters for tiago, first 6 assigned to base frame

    elif param['robot_name'] == 'talos':
        if not base_model:
            var_rs = np.reshape(var, (NbFrames, 6))
        elif base_model:
            var_rs = np.zeros((NbFrames, 6))

            # 6D base frame
            var_rs[0, 0: 6] = var[0: 6]

            # offset parameters var_rs[1:(1+Nbjoint),:]
            if param['NbJoint'] > 0:
                # torso_1
                var_rs[1, :] = np.zeros(6)
                if param['NbJoint'] > 1:
                    #  torso_2
                    var_rs[2, 0] = var[6]
                    var_rs[2, 1] = var[7]
                    var_rs[2, 3] = var[8]
                    var_rs[2, 4] = var[9]
                    if param['NbJoint'] > 2:
                        # arm1
                        var_rs[3, 0] = var[10]
                        var_rs[3, 2] = var[11]
                        var_rs[3, 3] = var[12]
                        var_rs[3, 5] = var[13]
                        if param['NbJoint'] > 3:
                            # arm2
                            var_rs[4, 0] = var[14]
                            var_rs[4, 1] = var[15]
                            var_rs[4, 3] = var[16]
                            var_rs[4, 4] = var[17]
                            if param['NbJoint'] > 4:
                                # arm3
                                var_rs[5, 1] = var[18]
                                var_rs[5, 2] = var[19]
                                var_rs[5, 4] = var[20]
                                var_rs[5, 5] = var[21]
                                if param['NbJoint'] > 5:
                                    # arm4
                                    var_rs[6, 0] = var[22]
                                    var_rs[6, 1] = var[23]
                                    var_rs[6, 3] = var[24]
                                    var_rs[6, 4] = var[25]
                                    if param['NbJoint'] > 6:
                                        # arm5
                                        var_rs[7, 0] = var[26]
                                        var_rs[7, 2] = var[27]
                                        var_rs[7, 3] = var[28]
                                        var_rs[7, 5] = var[29]
                                        if param['NbJoint'] > 7:
                                            # arm6
                                            var_rs[8, 0] = var[30]
                                            var_rs[8, 1] = var[31]
                                            var_rs[8, 3] = var[32]
                                            var_rs[8, 4] = var[33]
                                            if param['NbJoint'] > 8:
                                                # arm6
                                                var_rs[9, 1] = var[34]
                                                var_rs[9, 2] = var[35]
                                                var_rs[9, 4] = var[36]
                                                var_rs[9, 5] = var[37]

            # 38 base parameters for talos torso-arm, first 6 assigned to base frame

    # frame trasformation matrix from mocap to base
    base_placement = cartesian_to_SE3(var_rs[0, 0: 6])

    for k in range(param['NbMarkers']):
        # The last calibration_index(3/6)*NbMarkers of var array to be assigned to 
        # the last NbMarkers rows of var_rs
        # kth marker's index in var_rs
        markerId = 1 + param['NbJoint'] + k

        # beginning index belongs to markers in var
        curr_varId = var.shape[0] - \
            param['NbMarkers']*param['calibration_index']

        # kth marker frame var_rs[1+NbJoint+k]
        var_rs[markerId, 0: param['calibration_index']] = var[(
            curr_varId+k*param['calibration_index']): (curr_varId+(k+1)*param['calibration_index'])]

        PEE_marker = np.empty((nrow, ncol))
        for i in range(ncol):
            config = q_temp[i, :]
            # some fckps here in exceptional case wherer no parameters of the joint are to be variables
            # even the joint is active
            # update joint geometric parameters with values stored in var_rs
            # NOTE: jointPlacements modify the model of robot, revert the
            # updates after done calculatioin
            for j, joint_idx in enumerate(param['actJoint_idx']):
                joint_placement = cartesian_to_SE3(var_rs[j+1, :])
                # model.jointPlacements[j].translation += joint_placement.translation
                # model.jointPlacements[j].rotation += joint_placement.rotation (matrix addition => possibly wrong)
                temp_translation = model.jointPlacements[joint_idx].translation
                model.jointPlacements[joint_idx].translation += var_rs[j+1, 0: 3]
                new_rpy = pin.rpy.matrixToRpy(
                    model.jointPlacements[joint_idx].rotation) + var_rs[j+1, 3:6]
                model.jointPlacements[joint_idx].rotation = pin.rpy.rpyToMatrix(
                    new_rpy)
                after_translation = model.jointPlacements[joint_idx].translation
            pin.framesForwardKinematics(model, data, config)
            pin.updateFramePlacements(model, data)

            # calculate oMf from the 1st join tto last joint (wrist)
            lastJoint_name = model.names[param['tool_joint']]
            lastJoint_frameId = model.getFrameId(lastJoint_name)
            inter_placements = data.oMf[lastJoint_frameId]

            # # calculate oMf from wrist to the last frame
            # print("marker row: ", markerId)
            last_placement = cartesian_to_SE3(var_rs[markerId, :])

            new_oMf = base_placement * \
                inter_placements * \
                last_placement  # from baseframe -> joints -> last frame

            # create a 2D array containing coordinates of end_effector
            # calibration_index = 3
            PEE_marker[0:3, i] = new_oMf.translation
            # calibrtion_index = 6
            if nrow == 6:
                PEE_rot = new_oMf.rotation
                PEE_marker[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)

            # revert the updates above/restore original state of robot.model
            for j, joint_idx in enumerate(param['actJoint_idx']):
                joint_placement = cartesian_to_SE3(var_rs[j+1, :])
                # model.jointPlacements[j].translation -= joint_placement.translation
                # model.jointPlacements[j].rotation -= joint_placement.rotation (matrix addition => possibly wrong)
                model.jointPlacements[joint_idx].translation -= var_rs[j+1, 0:3]
                update_rpy = pin.rpy.matrixToRpy(
                    model.jointPlacements[joint_idx].rotation) - var_rs[j+1, 3:6]
                model.jointPlacements[joint_idx].rotation = pin.rpy.rpyToMatrix(
                    update_rpy)

            # TODO: to add changement of joint placements after reverting back to original version
            # stop process, send warning message!!

        # flatten ee's coordinates to 1D array
        PEE_marker = PEE_marker.flatten('C')
        PEE = np.append(PEE, PEE_marker)
    return PEE


def get_PEE_var(var, q, model, data, param, noise=False):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
        var: consist of geometric parameters from base->1st joint, joint offsets, wirst-> end effector
    """
    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']
    PEE = np.empty((nrow, ncol))
    q_temp = np.copy(q)
    for i in range(ncol):
        config = q_temp[i, :]
        # frame trasformation matrix from mocap to base
        p_base = var[0:3]
        rpy_base = var[3:6]
        R_base = pin.rpy.rpyToMatrix(rpy_base)
        base_placement = pin.SE3(R_base, p_base)

        # adding offset (8 for 8 joints)
        config[0:8] = config[0:8] + var[6:14]

        # adding zero mean additive noise to simulated measured coordinates
        if noise:
            noise = np.random.normal(0, 0.001, var[0:8].shape)
            config[0:8] = config[0:8] + noise

        pin.framesForwardKinematics(model, data, config)
        pin.updateFramePlacements(model, data)

        # calculate oMf from wrist to the last frame
        p_ee = var[14:17]
        rpy_ee = var[17:20]
        R_ee = pin.rpy.rpyToMatrix(rpy_ee)
        last_placement = pin.SE3(R_ee, p_ee)

        base_oMf = base_placement * \
            data.oMf[param['IDX_TOOL']]  # from mocap to wirst
        new_oMf = base_oMf*last_placement  # from wrist to end effector

        # create a matrix containing coordinates of end_effector
        PEE[0:3, i] = new_oMf.translation
        if nrow == 6:
            PEE_rot = new_oMf.rotation
            PEE[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)

    PEE = PEE.flatten('C')
    return PEE


def get_PEE(offset_var, q, model, data, param, noise=False):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
        offset_var: consist of only joint offsets
    """
    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']
    PEE = np.empty((nrow, ncol))
    q_temp = np.copy(q)
    for i in range(ncol):
        config = q_temp[i, :]
        config[0:8] = config[0:8] + offset_var
        # adding zero mean additive noise to simulated measured coordinates
        if noise:
            noise = np.random.normal(0, 0.001, offset_var.shape)
            config[0:8] = config[0:8] + noise

        pin.framesForwardKinematics(model, data, config)
        pin.updateFramePlacements(model, data)

        PEE[0:3, i] = data.oMf[param['IDX_TOOL']].translation
        if nrow == 6:
            PEE_rot = data.oMf[param['IDX_TOOL']].rotation
            PEE[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)
    PEE = PEE.flatten('C')
    return PEE


######################## BASE REGRESSOR TOOLS ########################################


def calculate_kinematics_model(q_i, model, data, param):
    """ Calculate jacobian matrix and kinematic regressor given ONE configuration.
        Details of calculation at regressor.hxx and se3-tpl.hpp
    """
    # print(mp.current_process())
    # pin.updateGlobalPlacements(model , data)
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)

    J = pin.computeFrameJacobian(
        model, data, q_i, param['IDX_TOOL'], pin.LOCAL)
    R = pin.computeFrameKinematicRegressor(
        model, data, param['IDX_TOOL'], pin.LOCAL)
    return model, data, R, J


def calculate_identifiable_kinematics_model(q, model, data, param):
    """ Calculate jacobian matrix and kinematic regressor and aggreating into one matrix,
        given a set of configurations or random configurations if not given.
    """
    q_temp = np.copy(q)
    # Note if no q id given then use random generation of q to determine the minimal kinematics model
    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain aggreated Jacobian matrix J and kinematic regressor R
    calib_idx = param['calibration_index']
    R = np.zeros([6*param['NbSample'], 6*(model.njoints-1)])
    J = np.zeros([6*param['NbSample'], model.njoints-1])
    for i in range(param['NbSample']):
        if MIN_MODEL == 1:
            q_rand = pin.randomConfiguration(model)
            q_i = param['q0']
            q_i[param['config_idx']] = q_rand[param['config_idx']]
        else:
            q_i = q_temp[i, :]
        if param['start_frame'] == 'universe':
            model, data, Ri, Ji = calculate_kinematics_model(
                q_i, model, data, param)
        else:
            Ri = get_rel_kinreg(model, data, param['start_frame'],
                                param['end_frame'], q_i)
            # Ji = np.zeros([6, model.njoints-1]) ## TODO: get_rel_jac
            Ji = get_rel_jac(model, data, param['start_frame'],
                                param['end_frame'], q_i)
        for j, state in enumerate(param['measurability']):
            if state:
                R[param['NbSample']*j + i, :] = Ri[j, :]
                J[param['NbSample']*j + i, :] = Ji[j, :]
    # remove zero rows
    zero_rows = []
    for r_idx in range(R.shape[0]):
        if np.linalg.norm(R[r_idx, :]) < 1e-6:
            zero_rows.append(r_idx)
    R = np.delete(R, zero_rows, axis=0)
    zero_rows = []
    for r_idx in range(J.shape[0]):
        if np.linalg.norm(J[r_idx, :]) < 1e-6:
            zero_rows.append(r_idx)
    J = np.delete(J, zero_rows, axis=0)
    print(R.shape, J.shape)
    if param['calib_model'] == "joint_offset":
        return J
    elif param['calib_model'] == "full_params":
        return R


def calculate_base_kinematics_regressor(q, model, data, param):
    """ Calculate base regressor and base parameters for a calibration model from given configuration data.
    """
    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names[1:])]
    geo_params = get_geo_offset(joint_names)
    joint_offsets = get_joint_offset(joint_names)

    # calculate kinematic regressor with random configs
    if not param["free_flyer"]:
        Rrand = calculate_identifiable_kinematics_model([], model, data, param)
    else:
        Rrand = calculate_identifiable_kinematics_model(q, model, data, param)
    # calculate kinematic regressor with input configs
    if np.any(np.array(q)):
        R = calculate_identifiable_kinematics_model(q, model, data, param)
    else:
        R = Rrand

    ############## only joint offset parameters ########
    if param['calib_model'] == 'joint_offset':
        # particularly select columns/parameters corresponding to joint and 6 last parameters
        # actJoint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
        #                 50, 51, 52, 53]  # all on z axis - checked!!

        # a dictionary of selected parameters
        # gp_listItems = list(geo_params.items())
        # geo_params_sel = []
        # for i in actJoint_idx:
        #     geo_params_sel.append(gp_listItems[i])
        # geo_params_sel = dict(geo_params_sel)
        geo_params_sel = joint_offsets

        # select columns corresponding to joint_idx
        Rrand_sel = Rrand

        # select columns corresponding to joint_idx
        R_sel = R

    ############## full 6 parameters ###################
    elif param['calib_model'] == 'full_params':
        geo_params_sel = geo_params
        Rrand_sel = Rrand
        R_sel = R

    # remove non affect columns from random data => reduced regressor
    Rrand_e, paramsrand_e = eliminate_non_dynaffect(
        Rrand_sel, geo_params_sel, tol_e=1e-6)

    # get indices of independent columns (base param) w.r.t to reduced regressor
    idx_base = get_baseIndex(Rrand_e, paramsrand_e)

    # get base regressor and base params from random data
    Rrand_b, paramsrand_base, _ = get_baseParams(Rrand_e, paramsrand_e)

    # remove non affect columns from GIVEN data
    R_e, params_e = eliminate_non_dynaffect(
        R_sel, geo_params_sel, tol_e=1e-6)

    # get base param from given data 
    idx_gbase = get_baseIndex(R_e, params_e)
    R_gb, params_gbase, _ = get_baseParams(R_e, params_e)

    # get base regressor from GIVEN data
    R_b = build_baseRegressor(R_e, idx_base)

    # update calibrating param['param_name']/calibrating parameters
    for j in idx_base:
        param['param_name'].append(paramsrand_e[j])

    print('shape of full regressor, reduced regressor, base regressor: ',
          Rrand.shape, Rrand_e.shape, Rrand_b.shape)
    return Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e
