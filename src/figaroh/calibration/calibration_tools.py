# Copyright [2021-2025] Thanh Nguyen
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
    cond_num,
)
from ..tools.robot import Robot

TOL_QR = 1e-8
FULL_PARAMTPL = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
JOINT_OFFSETTPL = [
    "offsetPX",
    "offsetPY",
    "offsetPZ",
    "offsetRX",
    "offsetRY",
    "offsetRZ",
]
ELAS_TPL = [
    "k_PX",
    "k_PY",
    "k_PZ",
    "k_RX",
    "k_RY",
    "k_RZ",
]  # ONLY REVOLUTE JOINT FOR NOW
EE_TPL = ["pEEx", "pEEy", "pEEz", "phiEEx", "phiEEy", "phiEEz"]
BASE_TPL = [
    "base_px",
    "base_py",
    "base_pz",
    "base_phix",
    "base_phiy",
    "base_phiz",
]


# INITIALIZATION TOOLS ########################################
def get_param_from_yaml(robot, calib_data) -> dict:
    """Get parameter from the calibration config yaml file and return a
    dictionary.

    Args:
        robot (_type_): robot instance
        calib_data (_type_): calibration parameters parsed from config yaml file

    Returns:
        param (dict): a dictionary containing calibration parameters
    """
    # NOTE: since joint 0 is universe and it is trivial,
    # indices of joints are different from indices of joint configuration,
    # different from indices of joint velocities
    param = dict()
    robot_name = robot.model.name
    frames = [f.name for f in robot.model.frames]
    param["robot_name"] = robot_name

    # End-effector sensing measurability:
    # number of "True" = calib_idx
    NbMarkers = len(calib_data["markers"])
    measurability = calib_data["markers"][0]["measure"]
    calib_idx = measurability.count(True)
    param["NbMarkers"] = NbMarkers
    param["measurability"] = measurability
    param["calibration_index"] = calib_idx

    # Calibration model:
    # level 1: joint_offset (only joint offsets),
    # level 2: full_params(geometric parameters),
    # level 3: non_geom(non-geometric parameters)
    param["calib_model"] = calib_data["calib_level"]  # 'joint_offset' / 'full_params'

    # Kinematic chain: base frame: start_frame, end-effector frame: end_frame
    start_frame = calib_data["base_frame"]  # default
    end_frame = calib_data["tool_frame"]
    assert start_frame in frames, "Start_frame {} does not exist.".format(start_frame)

    assert end_frame in frames, "End_frame {} does not exist.".format(end_frame)
    param["start_frame"] = start_frame
    param["end_frame"] = end_frame

    # eye-hand calibration
    try:
        base_to_ref_frame = calib_data["base_to_ref_frame"]
        ref_frame = calib_data["ref_frame"]
    except KeyError:
        base_to_ref_frame = None
        ref_frame = None
        print("base_to_ref_frame and ref_frame are not defined.")

    if base_to_ref_frame is not None:
        assert (
            base_to_ref_frame in frames
        ), "base_to_ref_frame {} does not exist.".format(base_to_ref_frame)
    else:
        base_to_ref_frame = None

    if ref_frame is not None:
        assert ref_frame in frames, "ref_frame {} does not exist.".format(ref_frame)
    else:
        ref_frame = None
    param["base_to_ref_frame"] = base_to_ref_frame
    param["ref_frame"] = ref_frame

    try:
        # initial pose of base frame and marker frame
        camera_pose = calib_data["camera_pose"]
        tip_pose = calib_data["tip_pose"]
    except KeyError:
        camera_pose = None
        tip_pose = None
        print("camera_pose and tip_pose are not defined.")

    param["camera_pose"] = camera_pose
    param["tip_pose"] = tip_pose

    # q0: default zero configuration
    param["q0"] = robot.q0
    param["NbSample"] = calib_data["nb_sample"]

    # IDX_TOOL: frame ID of the tool
    IDX_TOOL = robot.model.getFrameId(end_frame)
    param["IDX_TOOL"] = IDX_TOOL

    # tool_joint: ID of the joint right before the tool's frame (parent)
    tool_joint = robot.model.frames[IDX_TOOL].parent
    param["tool_joint"] = tool_joint

    # indices of active joints: from base to tool_joint
    actJoint_idx = get_sup_joints(robot.model, start_frame, end_frame)
    param["actJoint_idx"] = actJoint_idx

    # indices of joint configuration corresponding to active joints
    config_idx = [robot.model.joints[i].idx_q for i in actJoint_idx]
    param["config_idx"] = config_idx

    # number of active joints
    NbJoint = len(actJoint_idx)
    param["NbJoint"] = NbJoint

    # initialize a list of calibrating parameters name
    param_name = []
    if calib_data["non_geom"]:
        # list of elastic gain parameter names
        elastic_gain = []
        joint_axes = ["PX", "PY", "PZ", "RX", "RY", "RZ"]
        for j_id, joint_name in enumerate(robot.model.names.tolist()):
            if joint_name == "universe":
                axis_motion = "null"
            else:
                # for ii, ax in enumerate(AXIS_MOTION[j_id]):
                #     if ax == 1:
                #         axis_motion = axis[ii]
                shortname = robot.model.joints[
                    j_id
                ].shortname()  # ONLY TAKE PRISMATIC AND REVOLUTE JOINT
                for ja in joint_axes:
                    if ja in shortname:
                        axis_motion = ja
                    elif "RevoluteUnaligned" in shortname:
                        axis_motion = "RZ"  # hard coded fix for canopies

            elastic_gain.append("k_" + axis_motion + "_" + joint_name)
        for i in actJoint_idx:
            param_name.append(elastic_gain[i])
    param["param_name"] = param_name

    param.update(
        {
            "free_flyer": calib_data["free_flyer"],
            "non_geom": calib_data["non_geom"],
            "eps": 1e-3,
            "PLOT": 0,
        }
    )
    try:
        param.update(
            {
                "coeff_regularize": calib_data["coeff_regularize"],
                "data_file": calib_data["data_file"],
                "sample_configs_file": calib_data["sample_configs_file"],
                "outlier_eps": calib_data["outlier_eps"],
            }
        )
    except KeyError:
        param.update(
            {
                "coeff_regularize": None,
                "data_file": None,
                "sample_configs_file": None,
                "outlier_eps": None,
            }
        )
    pprint.pprint(param)
    return param


def get_joint_offset(model, joint_names):
    """This function give a dictionary of joint offset parameters.
    Input:  joint_names: a list of joint names (from model.names)
    Output: joint_off: a dictionary of joint offsets.
    """
    joint_off = []
    joint_names = list(model.names[1:])
    joints = list(model.joints[1:])
    assert len(joint_names) == len(
        joints
    ), "Number of jointnames does not match number of joints! Please check\
        imported model."
    for id, joint in enumerate(joints):
        name = joint_names[id]
        shortname = joint.shortname()
        if model.name == "canopies":
            if "RevoluteUnaligned" in shortname:
                shortname = shortname.replace("RevoluteUnaligned", "RZ")
        for i in range(joint.nv):
            if i > 0:
                offset_param = (
                    shortname.replace("JointModel", "offset")
                    + "{}".format(i + 1)
                    + "_"
                    + name
                )
            else:
                offset_param = shortname.replace("JointModel", "offset") + "_" + name
            joint_off.append(offset_param)

    phi_jo = [0] * len(joint_off)  # default zero values
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off


def get_geo_offset(joint_names):
    """This function give a dictionary of variations (offset) of kinematics parameters.
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


def add_base_name(param):
    # base_names = []
    # for i in range(param['NbMarkers']):
    #     for j, state in enumerate(param['measurability']):
    #         if state:
    #             base_names.extend(['{}_{}'.format(BASE_TPL[j], i+1)])
    if param["calib_model"] == "full_params":
        param["param_name"][0:6] = BASE_TPL
    elif param["calib_model"] == "joint_offset":
        param["param_name"] = BASE_TPL + param["param_name"]


def add_pee_name(param):
    PEE_names = []
    for i in range(param["NbMarkers"]):
        for j, state in enumerate(param["measurability"]):
            if state:
                PEE_names.extend(["{}_{}".format(EE_TPL[j], i + 1)])
    param["param_name"] = param["param_name"] + PEE_names


def add_eemarker_frame(frame_name, p, rpy, model, data):
    """Adds a frame at the end_effector."""
    p = np.array([0.1, 0.1, 0.1])
    R = pin.rpy.rpyToMatrix(rpy)
    frame_placement = pin.SE3(R, p)

    parent_jointId = model.getJointId("arm_7_joint")
    prev_frameId = model.getFrameId("arm_7_joint")
    ee_frame_id = model.addFrame(
        pin.Frame(
            frame_name,
            parent_jointId,
            prev_frameId,
            frame_placement,
            pin.FrameType(0),
            pin.Inertia.Zero(),
        ),
        False,
    )
    return ee_frame_id


######################## DATA PROCESSING TOOLS ########################################
# data collected and saved in various formats and structure, it has
# always been a pain in the ass to handles. Need to think a way to simplify
# and optimize this procedure.
def read_config_data(model, path_to_file):
    df = pd.read_csv(path_to_file)
    q = np.zeros([len(df), model.njoints - 1])
    for i in range(len(df)):
        for j, name in enumerate(model.names[1:].tolist()):
            jointidx = rank_in_configuration(model, name)
            q[i, jointidx] = df[name][i]
    return q


def load_data(path_to_file, model, param, del_list=[]):
    """Read a csv file into dataframe by pandas, then transform to the form
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
            Marker is a broad term for measurment. Important to know that any type of
            measurement should be able to be converted to a 6D vector (position and orientation)
            and be treated as a marker. The measurability hightlight the DOF that measurement
            deliver and reference joint indicates the parent where the measurement is taken.
            However, respecting the order of x, y, z, phix, phiy, phiz is critical as well as
            the numbered order of the markers.
    """
    # read_csv
    df = pd.read_csv(path_to_file)

    # create headers for marker position
    PEE_headers = []
    pee_tpl = ["x", "y", "z", "phix", "phiy", "phiz"]
    for i in range(param["NbMarkers"]):
        for j, state in enumerate(param["measurability"]):
            if state:
                PEE_headers.extend(["{}{}".format(pee_tpl[j], i + 1)])

    # create headers for joint configurations
    joint_headers = [model.names[i] for i in param["actJoint_idx"]]
    print(joint_headers)
    # check if all created headers present in csv file
    csv_headers = list(df.columns)
    for header in PEE_headers + joint_headers:
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
    param["NbSample"] = q_act.shape[0]

    PEEm_exp = xyz_4Mkr.T
    PEEm_exp = PEEm_exp.flatten("C")

    q_exp = np.empty((param["NbSample"], param["q0"].shape[0]))
    for i in range(param["NbSample"]):
        config = param["q0"]
        config[param["config_idx"]] = q_act[i, :]
        q_exp[i, :] = config

    return PEEm_exp, q_exp


######################## COMMON TOOLS ########################################


def rank_in_configuration(model, joint_name):
    """Get index of a given joint in joint configuration vector"""
    assert joint_name in model.names, "Given joint name does not exist."
    jointId = model.getJointId(joint_name)
    joint_idx = model.joints[jointId].idx_q
    return joint_idx


def cartesian_to_SE3(X):
    """Convert (6,) cartesian coordinates to SE3
    Input: 1D (6,) numpy array
    Output: SE3 placement
    """
    X = np.array(X)
    X = X.flatten("C")
    translation = X[0:3]
    rot_matrix = pin.rpy.rpyToMatrix(X[3:6])
    placement = pin.SE3(rot_matrix, translation)
    return placement


def xyzquat_to_SE3(xyzquat):
    """Convert (7,) xyzquat coordinates to SE3
    Input: 1D (7,) numpy array
    Output: SE3 placement
    """
    xyzquat = np.array(xyzquat)
    xyzquat = xyzquat.flatten("C")
    translation = xyzquat[0:3]
    rot_matrix = pin.Quaternion(xyzquat[3:7]).normalize().toRotationMatrix()
    placement = pin.SE3(rot_matrix, translation)
    return placement


def get_rel_transform(model, data, start_frame, end_frame):
    """Calculate relative transformation between any two frames
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
    assert start_frame in frames, "{} does not exist.".format(start_frame)
    assert end_frame in frames, "{} does not exist.".format(end_frame)
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
    """Find supporting joints between two frames
    Output: a list of supporting joints' Id
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)
    start_par = model.frames[start_frameId].parent
    end_par = model.frames[end_frameId].parent
    branch_s = model.supports[start_par].tolist()
    branch_e = model.supports[end_par].tolist()
    # remove 'universe' joint from branches
    if model.names[branch_s[0]] == "universe":
        branch_s.remove(branch_s[0])
    if model.names[branch_e[0]] == "universe":
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
    """Calculate kinematic regressor between start_frame and end_frame
    Output: 6x6n matrix
    """
    sup_joints = get_sup_joints(model, start_frame, end_frame)
    # update frameForwardKinematics and updateFramePlacements
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # a set-zero kinreg
    kinreg = np.zeros((6, 6 * (model.njoints - 1)))
    frame = model.frames[model.getFrameId(end_frame)]
    oMf = data.oMi[frame.parent] * frame.placement
    for p in sup_joints:
        oMp = data.oMi[model.parents[p]] * model.jointPlacements[p]
        # fMp = get_rel_transform(model, data, end_frame, model.names[p])
        fMp = oMf.actInv(oMp)
        fXp = fMp.toActionMatrix()
        kinreg[:, 6 * (p - 1) : 6 * p] = fXp
    return kinreg


def get_rel_jac(model, data, start_frame, end_frame, q):
    """Calculate frameJacobian between start_frame and end_frame
    Output: 6xn matrix
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)

    # update frameForwardKinematics and updateFramePlacements
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # relative Jacobian
    J_start = pin.computeFrameJacobian(model, data, q, start_frameId, pin.LOCAL)
    J_end = pin.computeFrameJacobian(model, data, q, end_frameId, pin.LOCAL)
    J_rel = J_end - J_start
    return J_rel


######################## LEVENBERG-MARQUARDT TOOLS ########################################


def get_LMvariables(param, mode=0, seed=0):
    """Create a initial zero/range-bounded random search varibale for Leverberg-Marquardt algo."""
    # initialize all variables at zeros
    nvar = len(param["param_name"])
    if mode == 0:
        var = np.zeros(nvar)
    elif mode == 1:
        var = np.random.uniform(-seed, seed, nvar)
    return var, nvar


def update_forward_kinematics(model, data, var, q, param, verbose=0):
    """Update jointplacements with offset parameters, recalculate forward kinematics
    to find end-effector's position and orientation.
    """
    # read param['param_name'] to allocate offset parameters to correct SE3
    # convert translation: add a vector of 3 to SE3.translation
    # convert orientation: convert SE3.rotation 3x3 matrix to vector rpy, add
    #  to vector rpy, convert back to to 3x3 matrix

    # name reference of calibration parameters
    if param["calib_model"] == "full_params":
        axis_tpl = FULL_PARAMTPL
    elif param["calib_model"] == "joint_offset":
        axis_tpl = JOINT_OFFSETTPL

    # order of joint in variables are arranged as in param['actJoint_idx']
    assert len(var) == len(
        param["param_name"]
    ), "Length of variables != length of params"
    param_dict = dict(zip(param["param_name"], var))
    origin_model = model.copy()

    # update model.jointPlacements
    updated_params = []
    start_f = param["start_frame"]
    end_f = param["end_frame"]

    # define transformation for camera frame
    if param["base_to_ref_frame"] is not None:
        start_f = param["ref_frame"]
        # base frame to ref frame (i.e. Tiago: camera transformation)
        base_tf = np.zeros(6)
        for key in param_dict.keys():
            for base_id, base_ax in enumerate(BASE_TPL):
                if base_ax in key:
                    base_tf[base_id] = param_dict[key]
                    updated_params.append(key)
        b_to_cam = get_rel_transform(
            model, data, param["start_frame"], param["base_to_ref_frame"]
        )
        ref_to_cam = cartesian_to_SE3(base_tf)
        cam_to_ref = ref_to_cam.actInv(pin.SE3.Identity())
        bMo = b_to_cam * cam_to_ref
    else:
        if param["calib_model"] == "joint_offset":
            base_tf = np.zeros(6)
            for key in param_dict.keys():
                for base_id, base_ax in enumerate(BASE_TPL):
                    if base_ax in key:
                        base_tf[base_id] = param_dict[key]
                        updated_params.append(key)
            bMo = cartesian_to_SE3(base_tf)

    # update model.jointPlacements with joint 'full_params'/'joint_offset'
    for j_id in param["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy with kinematic errors
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose == 1:
                            print(
                                "Updating [{}] joint placement at axis {} with [{}]".format(
                                    j_name, axis, key
                                )
                            )
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)
        model = update_joint_placement(model, j_id, xyz_rpy)
    PEE = np.zeros((param["calibration_index"], param["NbSample"]))

    # update end_effector frame
    for marker_idx in range(1, param["NbMarkers"] + 1):
        pee = np.zeros(6)
        ee_name = "EE"
        for key in param_dict.keys():
            if ee_name in key and str(marker_idx) in key:
                # update xyz_rpy with kinematic errors
                for axis_pee_id, axis_pee in enumerate(EE_TPL):
                    if axis_pee in key:
                        if verbose == 1:
                            print(
                                "Updating [{}_{}] joint placement at axis {} with [{}]".format(
                                    ee_name, str(marker_idx), axis_pee, key
                                )
                            )
                        pee[axis_pee_id] += param_dict[key]
                        # updated_params.append(key)

        eeMf = cartesian_to_SE3(pee)

    # get transform
    q_ = np.copy(q)
    for i in range(param["NbSample"]):
        pin.framesForwardKinematics(model, data, q_[i, :])
        pin.updateFramePlacements(model, data)
        # update model.jointPlacements with joint elastic error
        if param["non_geom"]:
            tau = pin.computeGeneralizedGravity(
                model, data, q_[i, :]
            )  # vector size of 32 = nq < njoints
            # update xyz_rpy with joint elastic error
            for j_id in param["actJoint_idx"]:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1]  # nq = njoints -1
                if j_name in key:
                    for elas_id, elas in enumerate(ELAS_TPL):
                        if elas in key:
                            param_dict[key] = param_dict[key] * tau_j
                            xyz_rpy[elas_id + 3] += param_dict[
                                key
                            ]  # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, xyz_rpy)
            # get relative transform with updated model
            oMee = get_rel_transform(
                model, data, param["start_frame"], param["end_frame"]
            )
            # revert model back to origin from added joint elastic error
            for j_id in param["actJoint_idx"]:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1]  # nq = njoints -1
                if j_name in key:
                    for elas_id, elas in enumerate(ELAS_TPL):
                        if elas in key:
                            param_dict[key] = param_dict[key] * tau_j
                            xyz_rpy[elas_id + 3] += param_dict[
                                key
                            ]  # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, -xyz_rpy)

        else:
            oMee = get_rel_transform(
                model, data, param["start_frame"], param["end_frame"]
            )

        if len(updated_params) < len(param_dict):

            oMf = oMee * eeMf
            # final transform
            trans = oMf.translation.tolist()
            orient = pin.rpy.matrixToRpy(oMf.rotation).tolist()
            loc = trans + orient
            measure = []
            for mea_id, mea in enumerate(param["measurability"]):
                if mea:
                    measure.append(loc[mea_id])
            # PEE[(marker_idx-1)*param['calibration_index']:marker_idx*param['calibration_index'], i] = np.array(measure)
            PEE[:, i] = np.array(measure)

            # assert len(updated_params) == len(param_dict), "Not all parameters are updated"

    PEE = PEE.flatten("C")
    # revert model back to original
    assert origin_model.jointPlacements != model.jointPlacements, "before revert"
    for j_id in param["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        xyz_rpy[axis_id] = param_dict[key]
        model = update_joint_placement(model, j_id, -xyz_rpy)

    assert origin_model.jointPlacements != model.jointPlacements, "after revert"

    return PEE


def update_forward_kinematics_2(model, data, var, q, param, verbose=0):
    """Update jointplacements with offset parameters, recalculate forward kinematics
    to find end-effector's position and orientation.
    """
    # read param['param_name'] to allocate offset parameters to correct SE3
    # convert translation: add a vector of 3 to SE3.translation
    # convert orientation: convert SE3.rotation 3x3 matrix to vector rpy, add
    #  to vector rpy, convert back to to 3x3 matrix
    # name reference of calibration parameters
    if param["calib_model"] == "full_params":
        axis_tpl = FULL_PARAMTPL
    elif param["calib_model"] == "joint_offset":
        axis_tpl = JOINT_OFFSETTPL

    # order of joint in variables are arranged as in param['actJoint_idx']
    assert len(var) == len(
        param["param_name"]
    ), "Length of variables != length of params"
    param_dict = dict(zip(param["param_name"], var))
    origin_model = model.copy()

    # update model.jointPlacements
    updated_params = []

    # base placement or 'universe' joint
    # TODO: add axis of base
    if "base_placement" in list(param_dict.keys())[0]:
        base_placement = cartesian_to_SE3(var[0:6])
        updated_params = param["param_name"][0:6]

    for j_id in param["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy with kinematic errors
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose == 1:
                            print(
                                "Updating [{}] joint placement at axis {} with [{}]".format(
                                    j_name, axis, key
                                )
                            )
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)
        model = update_joint_placement(model, j_id, xyz_rpy)
    PEE = np.zeros((param["NbMarkers"] * param["calibration_index"], param["NbSample"]))

    # update end_effector frame
    ee_frames = []
    for marker_idx in range(1, param["NbMarkers"] + 1):
        pee = np.zeros(6)
        ee_name = "EE"
        for key in param_dict.keys():
            if ee_name in key and str(marker_idx) in key:
                # update xyz_rpy with kinematic errors
                for axis_pee_id, axis_pee in enumerate(EE_TPL):
                    if axis_pee in key:
                        if verbose == 1:
                            print(
                                "Updating [{}_{}] joint placement at axis {} with [{}]".format(
                                    ee_name, str(marker_idx), axis_pee, key
                                )
                            )
                        pee[axis_pee_id] += param_dict[key]
                        # updated_params.append(key)

        eeMf = cartesian_to_SE3(pee)
        ee_frames.append(eeMf)
    assert (
        len(ee_frames) == param["NbMarkers"]
    ), "Number of end-effector frames != number of markers"

    # get transform
    q_ = np.copy(q)
    for i in range(param["NbSample"]):
        pin.framesForwardKinematics(model, data, q_[i, :])
        pin.updateFramePlacements(model, data)
        # update model.jointPlacements with joint elastic error
        if param["non_geom"]:
            tau = pin.computeGeneralizedGravity(
                model, data, q_[i, :]
            )  # vector size of 32 = nq < njoints
            # update xyz_rpy with joint elastic error
            for j_id in param["actJoint_idx"]:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1]  # nq = njoints -1
                if j_name in key:
                    for elas_id, elas in enumerate(ELAS_TPL):
                        if elas in key:
                            param_dict[key] = param_dict[key] * tau_j
                            xyz_rpy[elas_id + 3] += param_dict[
                                key
                            ]  # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, xyz_rpy)
            # get relative transform with updated model
            oMee = get_rel_transform(
                model, data, param["start_frame"], param["end_frame"]
            )
            # revert model back to origin from added joint elastic error
            for j_id in param["actJoint_idx"]:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1]  # nq = njoints -1
                if j_name in key:
                    for elas_id, elas in enumerate(ELAS_TPL):
                        if elas in key:
                            param_dict[key] = param_dict[key] * tau_j
                            xyz_rpy[elas_id + 3] += param_dict[
                                key
                            ]  # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, -xyz_rpy)

        else:
            oMf = oMee

        if param["base_to_ref_frame"] is not None:
            oMf = bMo * oMf
        else:
            if param["calib_model"] == "joint_offset":
                oMf = bMo * oMf
        # final transform
        trans = oMf.translation.tolist()
        orient = pin.rpy.matrixToRpy(oMf.rotation).tolist()
        loc = trans + orient
        measure = []
        for mea_id, mea in enumerate(param["measurability"]):
            if mea:
                measure.append(loc[mea_id])
        PEE_marker[:, i] = np.array(measure)
    PEE_marker = PEE_marker.flatten("C")
    PEE = np.append(PEE, PEE_marker)

    # revert model back to original
    assert origin_model.jointPlacements != model.jointPlacements, "before revert"
    for j_id in param["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        xyz_rpy[axis_id] = param_dict[key]
        model = update_joint_placement(model, j_id, -xyz_rpy)

    assert origin_model.jointPlacements != model.jointPlacements, "after revert"

    return PEE


def calc_updated_fkm(model, data, var, q, param, verbose=0):
    """Update jointplacements with offset parameters, recalculate fkm
    to find end-effector's position and orientation.
    """
    # read param['param_name'] to allocate offset parameters to correct SE3
    # convert translation: add a vector of 3 to SE3.translation
    # convert orientation: convert SE3.rotation 3x3 matrix to vector rpy, add
    #  to vector rpy, convert back to to 3x3 matrix

    # name reference of calibration parameters
    if param["calib_model"] == "full_params":
        axis_tpl = FULL_PARAMTPL

    elif param["calib_model"] == "joint_offset":
        axis_tpl = JOINT_OFFSETTPL

    # order of joint in variables are arranged as in param['actJoint_idx']
    assert len(var) == len(
        param["param_name"]
    ), "Length of variables != length of params"
    param_dict = dict(zip(param["param_name"], var))
    origin_model = model.copy()

    # store parameter updated to the model
    updated_params = []

    # check if baseframe and end--effector frame are known
    for key in param_dict.keys():
        if "base" in key:
            base_param_incl = True
            break
        else:
            base_param_incl = False
    for key in param_dict.keys():
        if "EE" in key:
            ee_param_incl = True
            break
        else:
            ee_param_incl = False

    # kinematic chain
    start_f = param["start_frame"]
    end_f = param["end_frame"]

    # if world frame (measurement ref frame) to the start frame is not known,
    # base_tpl needs to be used to define the first 6 parameters

    # 1/ calc transformation from the world frame to start frame: wMo
    if base_param_incl:
        base_tf = np.zeros(6)
        for key in param_dict.keys():
            for base_id, base_ax in enumerate(BASE_TPL):
                if base_ax in key:
                    base_tf[base_id] = param_dict[key]
                    updated_params.append(key)

        wMo = cartesian_to_SE3(base_tf)
    else:
        wMo = pin.SE3.Identity()

    # 2/ calculate transformation from the end frame to the end-effector frame,
    # if not known: eeMf
    if ee_param_incl and param["NbMarkers"] == 1:
        for marker_idx in range(1, param["NbMarkers"] + 1):
            pee = np.zeros(6)
            ee_name = "EE"
            for key in param_dict.keys():
                if ee_name in key and str(marker_idx) in key:
                    # update xyz_rpy with kinematic errors
                    for axis_pee_id, axis_pee in enumerate(EE_TPL):
                        if axis_pee in key:
                            if verbose == 1:
                                print(
                                    "Updating [{}_{}] joint placement at axis {} with [{}]".format(
                                        ee_name, str(marker_idx), axis_pee, key
                                    )
                                )
                            pee[axis_pee_id] += param_dict[key]
                            updated_params.append(key)

            eeMf = cartesian_to_SE3(pee)
    else:
        if param["NbMarkers"] > 1:
            print("Multiple markers are not supported.")
        else:
            eeMf = pin.SE3.Identity()

    # 3/ calculate transformation from start frame to end frame of kinematic chain using updated model: oMee

    # update model.jointPlacements with kinematic error parameter
    for j_id in param["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]

        # check joint name in param dict
        for key in param_dict.keys():
            if j_name in key:

                # update xyz_rpy with kinematic errors based on identifiable axis
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose == 1:
                            print(
                                "Updating [{}] joint placement at axis {} with [{}]".format(
                                    j_name, axis, key
                                )
                            )
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)

        # updaet joint placement
        model = update_joint_placement(model, j_id, xyz_rpy)

    # check if all parameters are updated to the model
    assert len(updated_params) == len(
        list(param_dict.keys())
    ), "Not all parameters are updated {} and {}".format(
        updated_params, list(param_dict.keys())
    )

    # pose vector of the end-effector
    PEE = np.zeros((param["calibration_index"], param["NbSample"]))

    q_ = np.copy(q)
    for i in range(param["NbSample"]):

        pin.framesForwardKinematics(model, data, q_[i, :])
        pin.updateFramePlacements(model, data)

        # NOTE: joint elastic error is not considered in this version

        oMee = get_rel_transform(model, data, start_f, end_f)

        # calculate transformation from world frame to end-effector frame
        wMee = wMo * oMee
        wMf = wMee * eeMf

        # final transform
        trans = wMf.translation.tolist()
        orient = pin.rpy.matrixToRpy(wMf.rotation).tolist()
        loc = trans + orient
        measure = []
        for mea_id, mea in enumerate(param["measurability"]):
            if mea:
                measure.append(loc[mea_id])
        PEE[:, i] = np.array(measure)

    # final result of updated fkm
    PEE = PEE.flatten("C")

    # revert model back to original
    assert origin_model.jointPlacements != model.jointPlacements, "before revert"
    for j_id in param["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        xyz_rpy[axis_id] = param_dict[key]
        model = update_joint_placement(model, j_id, -xyz_rpy)

    assert origin_model.jointPlacements != model.jointPlacements, "after revert"

    return PEE


def update_joint_placement(model, joint_idx, xyz_rpy):
    """Update joint placement given a vector of 6 offsets."""
    tpl_translation = model.jointPlacements[joint_idx].translation
    tpl_rotation = model.jointPlacements[joint_idx].rotation
    tpl_orientation = pin.rpy.matrixToRpy(tpl_rotation)
    # update axes
    updt_translation = tpl_translation + xyz_rpy[0:3]
    updt_orientation = tpl_orientation + xyz_rpy[3:6]
    updt_rotation = pin.rpy.rpyToMatrix(updt_orientation)
    # update placements
    model.jointPlacements[joint_idx].translation = updt_translation
    model.jointPlacements[joint_idx].rotation = updt_rotation
    return model


######################## BASE REGRESSOR TOOLS ########################################


def calculate_kinematics_model(q_i, model, data, param):
    """Calculate jacobian matrix and kinematic regressor given ONE configuration.
    Details of calculation at regressor.hxx and se3-tpl.hpp
    """
    # print(mp.current_process())
    # pin.updateGlobalPlacements(model , data)
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)

    J = pin.computeFrameJacobian(model, data, q_i, param["IDX_TOOL"], pin.LOCAL)
    R = pin.computeFrameKinematicRegressor(model, data, param["IDX_TOOL"], pin.LOCAL)
    return model, data, R, J


def calculate_identifiable_kinematics_model(q, model, data, param):
    """Calculate jacobian matrix and kinematic regressor and aggreating into one matrix,
    given a set of configurations or random configurations if not given.
    """
    q_temp = np.copy(q)
    # Note if no q id given then use random generation of q to determine the minimal kinematics model
    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain aggreated Jacobian matrix J and kinematic regressor R
    calib_idx = param["calibration_index"]
    R = np.zeros([6 * param["NbSample"], 6 * (model.njoints - 1)])
    J = np.zeros([6 * param["NbSample"], model.njoints - 1])
    for i in range(param["NbSample"]):
        if MIN_MODEL == 1:
            q_rand = pin.randomConfiguration(model)
            q_i = param["q0"]
            q_i[param["config_idx"]] = q_rand[param["config_idx"]]
        else:
            q_i = q_temp[i, :]
        if param["start_frame"] == "universe":
            model, data, Ri, Ji = calculate_kinematics_model(q_i, model, data, param)
        else:
            Ri = get_rel_kinreg(
                model, data, param["start_frame"], param["end_frame"], q_i
            )
            # Ji = np.zeros([6, model.njoints-1]) ## TODO: get_rel_jac
            Ji = get_rel_jac(model, data, param["start_frame"], param["end_frame"], q_i)
        for j, state in enumerate(param["measurability"]):
            if state:
                R[param["NbSample"] * j + i, :] = Ri[j, :]
                J[param["NbSample"] * j + i, :] = Ji[j, :]
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
    if param["calib_model"] == "joint_offset":
        return J
    elif param["calib_model"] == "full_params":
        return R


def calculate_base_kinematics_regressor(q, model, data, param, tol_qr=TOL_QR):
    """Calculate base regressor and base parameters for a calibration model from given configuration data."""
    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names[1:])]
    geo_params = get_geo_offset(joint_names)
    joint_offsets = get_joint_offset(model, joint_names)

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
    if param["calib_model"] == "joint_offset":
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
    elif param["calib_model"] == "full_params":
        geo_params_sel = geo_params
        Rrand_sel = Rrand
        R_sel = R

    # remove non affect columns from random data => reduced regressor
    Rrand_e, paramsrand_e = eliminate_non_dynaffect(
        Rrand_sel, geo_params_sel, tol_e=1e-6
    )

    # get indices of independent columns (base param) w.r.t to reduced regressor
    idx_base = get_baseIndex(Rrand_e, paramsrand_e, tol_qr=tol_qr)

    # get base regressor and base params from random data
    Rrand_b, paramsrand_base, _ = get_baseParams(Rrand_e, paramsrand_e, tol_qr=tol_qr)

    # remove non affect columns from GIVEN data
    R_e, params_e = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

    # get base param from given data
    idx_gbase = get_baseIndex(R_e, params_e, tol_qr=tol_qr)
    R_gb, params_gbase, _ = get_baseParams(R_e, params_e, tol_qr=tol_qr)

    # get base regressor from GIVEN data
    R_b = build_baseRegressor(R_e, idx_base)

    # update calibrating param['param_name']/calibrating parameters
    for j in idx_base:
        param["param_name"].append(paramsrand_e[j])

    print(
        "shape of full regressor, reduced regressor, base regressor: ",
        Rrand.shape,
        Rrand_e.shape,
        Rrand_b.shape,
    )
    return Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e
