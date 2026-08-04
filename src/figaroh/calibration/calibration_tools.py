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

"""
Calibration tools and algorithms for robot kinematic calibration.

This module contains the implementation of calibration algorithms including:
- Forward kinematics update functions
- Levenberg-Marquardt optimization
- Base regressor calculation
- Data loading and processing utilities
"""

import logging
import numpy as np
import pinocchio as pin

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from ..tools.regressor import eliminate_non_dynaffect
from ..tools.qrdecomposition import (
    get_baseParams,
    get_baseIndex,
    build_baseRegressor,
)

# Import configuration functions and constants from config module
from .config import (
    get_param_from_yaml,
    unified_to_legacy_config,
    get_sup_joints,
)

# Import parameter management functions and constants from parameter module
from .parameter import (
    get_joint_offset,
    get_fullparam_offset,
    add_base_name,
    add_pee_name,
    add_eemarker_frame,
    FULL_PARAMTPL,
    JOINT_OFFSETTPL,
    ELAS_TPL,
    EE_TPL,
    BASE_TPL,
)

# Import data loading functions from data_loader module
from .data_loader import (
    read_config_data,
    load_data,
    get_idxq_from_jname,
)

# Constants for calibration
TOL_QR = 1e-8
# Re-export for backward compatibility
__all__ = [
    "get_param_from_yaml",
    "unified_to_legacy_config",
    "get_sup_joints",
    "get_joint_offset",
    "get_fullparam_offset",
    "add_base_name",
    "add_pee_name",
    "add_eemarker_frame",
    "read_config_data",
    "load_data",
    "get_idxq_from_jname",
    "cartesian_to_SE3",
    "xyzquat_to_SE3",
    "get_rel_transform",
    "get_rel_kinreg",
    "get_rel_jac",
    "initialize_variables",
    "calc_updated_fkm",
    "update_joint_placement",
    "calculate_kinematics_model",
    "calculate_identifiable_kinematics_model",
    "calculate_base_kinematics_regressor",
]


# COMMON TOOLS


def cartesian_to_SE3(X):
    """Convert cartesian coordinates to SE3 transformation.

    Args:
        X (ndarray): (6,) array with [x,y,z,rx,ry,rz] coordinates

    Returns:
        pin.SE3: SE3 transformation with:
            - translation from X[0:3]
            - rotation matrix from RPY angles X[3:6]
    """
    X = np.array(X)
    X = X.flatten("C")
    translation = X[0:3]
    rot_matrix = pin.rpy.rpyToMatrix(X[3:6])
    placement = pin.SE3(rot_matrix, translation)
    return placement


def xyzquat_to_SE3(xyzquat):
    """Convert XYZ position and quaternion orientation to SE3 transformation.

    Takes a 7D vector containing XYZ position and WXYZ quaternion and creates
    an SE3 transformation matrix.

    Args:
        xyzquat (ndarray): (7,) array containing:
            - xyzquat[0:3]: XYZ position coordinates
            - xyzquat[3:7]: WXYZ quaternion orientation

    Returns:
        pin.SE3: Rigid body transformation with:
            - Translation from XYZ position
            - Rotation matrix from normalized quaternion

    Example:
        >>> pos_quat = np.array([0.1, 0.2, 0.3, 1.0, 0, 0, 0])
        >>> transform = xyzquat_to_SE3(pos_quat)
    """
    xyzquat = np.array(xyzquat)
    xyzquat = xyzquat.flatten("C")
    translation = xyzquat[0:3]
    rot_matrix = pin.Quaternion(xyzquat[3:7]).normalize().toRotationMatrix()
    placement = pin.SE3(rot_matrix, translation)
    return placement


def get_rel_transform(model, data, start_frame, end_frame):
    """Get relative transformation between two frames.

    Calculates the transform from start_frame to end_frame in the kinematic chain.
    Assumes forward kinematics has been updated.

    Args:
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        start_frame (str): Starting frame name
        end_frame (str): Target frame name

    Returns:
        pin.SE3: Relative transformation sMt from start to target frame

    Raises:
        AssertionError: If frame names don't exist in model
    """
    frames = [f.name for f in model.frames]
    assert start_frame in frames, "{} does not exist.".format(start_frame)
    assert end_frame in frames, "{} does not exist.".format(end_frame)
    start_frameId = model.getFrameId(start_frame)
    oMsf = data.oMf[start_frameId]
    end_frameId = model.getFrameId(end_frame)
    oMef = data.oMf[end_frameId]
    sMef = oMsf.actInv(oMef)
    return sMef


def get_rel_kinreg(model, data, start_frame, end_frame, q, backend=None):
    """Calculate relative kinematic regressor between frames.

    Computes frame Jacobian-based regressor matrix mapping small joint displacements
    to spatial velocities.

    Args:
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        start_frame (str): Starting frame name
        end_frame (str): Target frame name
        q (ndarray): Joint configuration vector
        backend (DynamicsBackend, optional): If provided, routes forward kinematics
            calls through the backend abstraction.

    Returns:
        ndarray: (6, 6n) regressor matrix for n joints
    """
    sup_joints = get_sup_joints(model, start_frame, end_frame)
    if backend is not None:
        backend.compute_forward_kinematics(q)
    else:
        pin.framesForwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
    kinreg = np.zeros((6, 6 * (model.njoints - 1)))
    frame = model.frames[model.getFrameId(end_frame)]
    oMf = data.oMi[frame.parent] * frame.placement
    for p in sup_joints:
        oMp = data.oMi[model.parents[p]] * model.jointPlacements[p]
        fMp = oMf.actInv(oMp)
        fXp = fMp.toActionMatrix()
        kinreg[:, 6 * (p - 1) : 6 * p] = fXp
    return kinreg


def get_rel_jac(model, data, start_frame, end_frame, q, backend=None):
    """Calculate relative Jacobian matrix between two frames.

    Computes the difference between Jacobians of end_frame and start_frame,
    giving the differential mapping from joint velocities to relative spatial velocity.

    Args:
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        start_frame (str): Starting frame name
        end_frame (str): Target frame name
        q (ndarray): Joint configuration vector
        backend (DynamicsBackend, optional): If provided, routes forward kinematics
            and Jacobian calls through the backend abstraction.

    Returns:
        ndarray: (6, n) relative Jacobian matrix where:
            - Rows represent [dx,dy,dz,wx,wy,wz] spatial velocities
            - Columns represent joint velocities
            - n is number of joints

    Note:
        Updates forward kinematics before computing Jacobians
    """
    if backend is not None:
        # compute_jacobian updates FK internally
        J_start = backend.compute_jacobian(q, start_frame)
        J_end = backend.compute_jacobian(q, end_frame)
    else:
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


# LEVENBERG-MARQUARDT TOOLS


def initialize_variables(calib_config, mode=0, seed=0):
    """Initialize variables for Levenberg-Marquardt optimization.

    Creates initial parameter vector either as zeros or random values within bounds.

    Args:
        calib_config (dict): Parameter dictionary containing:
            - param_name: List of parameter names to initialize
        mode (int, optional): Initialization mode:
            - 0: Zero initialization
            - 1: Random uniform initialization. Defaults to 0.
        seed (float, optional): Range [-seed,seed] for random init. Defaults to 0.

    Returns:
        tuple:
            - var (ndarray): Initial parameter vector
            - nvar (int): Number of parameters

    Example:
        >>> var, n = initialize_variables(params, mode=1, seed=0.1)
        >>> print(var.shape)
        (42,)
    """
    # initialize all variables at zeros
    nvar = len(calib_config["param_name"])
    if mode == 0:
        var = np.zeros(nvar)
    elif mode == 1:
        var = np.random.uniform(-seed, seed, nvar)
    return var, nvar


def calc_updated_fkm(model, data, var, q, calib_config, verbose=0, backend=None):
    """Update forward kinematics with world frame transformations.

    Single, unified FK-update function for calibration: composes the full
    chain of transformations::

        wMf = wMo * oMee * eeMf

    where:
        - ``wMo``: world (measurement) frame to the kinematic chain's start
          frame. Estimated directly when ``BASE_TPL`` params are present in
          ``param_name`` (unknown base frame, e.g. ``known_baseframe=False``),
          estimated via a known camera/ref-frame anchor when
          ``calib_config["base_to_ref_frame"]``/``"ref_frame"`` are set (e.g.
          eye-hand calibration), or identity otherwise.
        - ``oMee``: start frame to end frame, through the updated kinematic
          chain (``full_params``/``joint_offset`` geometric error
          parameters), optionally including joint elasticity when
          ``calib_config["non_geom"]`` is set — a per-joint compliance
          parameter (``ELAS_TPL``) that adds a gravity-torque-proportional
          deflection about that joint's own motion axis, then reverts it,
          on every sample.
        - ``eeMf``: end frame to the measured marker frame (``EE_TPL``
          params), or identity if not estimated.

    Args:
        model (pin.Model): Robot model to update
        data (pin.Data): Robot data
        var (ndarray): Parameter vector matching calib_config["param_name"]
        q (ndarray): Joint configurations matrix (n_samples, n_joints)
        calib_config (dict): Calibration parameters containing:
            - calib_model: "full_params" or "joint_offset"
            - start_frame, end_frame: Frame names
            - base_to_ref_frame, ref_frame: Optional camera-style known
              chain anchor (eye-hand calibration); None to disable
            - non_geom: Whether to apply joint elasticity
            - actJoint_idx: Active joint indices
            - measurability: Active DOFs
            - NbMarkers: Must be 1 (multi-marker is not supported)
        verbose (int, optional): Print update info. Defaults to 0.
        backend (DynamicsBackend, optional): If provided, routes forward
            kinematics and gravity calls through the backend abstraction.

    Returns:
        ndarray: Flattened marker measurements in world frame

    Raises:
        NotImplementedError: If calib_config["NbMarkers"] > 1.

    Notes:
        - Requires base or end-effector parameters in param_name to
          estimate wMo / eeMf; otherwise they default to identity.
        - Validates all parameters in param_name are consumed exactly once.
    """

    # name reference of calibration parameters
    if calib_config["calib_model"] == "full_params":
        axis_tpl = FULL_PARAMTPL

    elif calib_config["calib_model"] == "joint_offset":
        axis_tpl = JOINT_OFFSETTPL

    # order of joint in variables are arranged as in calib_config['actJoint_idx']
    assert len(var) == len(
        calib_config["param_name"]
    ), "Length of variables != length of params"
    param_dict = dict(zip(calib_config["param_name"], var))
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
    start_f = calib_config["start_frame"]
    end_f = calib_config["end_frame"]

    # 1/ calc transformation from the world frame to start frame: wMo
    base_to_ref_frame = calib_config.get("base_to_ref_frame")
    if base_to_ref_frame is not None:
        # known chain anchor + unknown camera/ref pose (e.g. eye-hand calib)
        start_f = calib_config["ref_frame"]
        base_tf = np.zeros(6)
        for key in param_dict.keys():
            for base_id, base_ax in enumerate(BASE_TPL):
                if base_ax in key:
                    base_tf[base_id] = param_dict[key]
                    updated_params.append(key)
        b_to_cam = get_rel_transform(
            model, data, calib_config["start_frame"], base_to_ref_frame
        )
        ref_to_cam = cartesian_to_SE3(base_tf)
        cam_to_ref = ref_to_cam.actInv(pin.SE3.Identity())
        wMo = b_to_cam * cam_to_ref
    elif base_param_incl:
        # fully unknown world (measurement) frame to start frame transform
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
    if ee_param_incl and calib_config["NbMarkers"] == 1:
        for marker_idx in range(1, calib_config["NbMarkers"] + 1):
            pee = np.zeros(6)
            ee_name = "EE"
            for key in param_dict.keys():
                if ee_name in key and str(marker_idx) in key:
                    # update xyz_rpy with kinematic errors
                    for axis_pee_id, axis_pee in enumerate(EE_TPL):
                        if axis_pee in key:
                            if verbose == 1:
                                logger.debug(
                                    "Updating [{}_{}] joint placement at axis {} with [{}]".format(
                                        ee_name, str(marker_idx), axis_pee, key
                                    )
                                )
                            pee[axis_pee_id] += param_dict[key]
                            updated_params.append(key)

            eeMf = cartesian_to_SE3(pee)
    else:
        if calib_config["NbMarkers"] > 1:
            raise NotImplementedError(
                "calc_updated_fkm only supports NbMarkers == 1, got "
                "NbMarkers={}.".format(calib_config["NbMarkers"])
            )
        eeMf = pin.SE3.Identity()

    # 3/ calculate transformation from start frame to end frame of kinematic chain using updated model: oMee

    # update model.jointPlacements with kinematic error parameter
    for j_id in calib_config["actJoint_idx"]:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]

        # check joint name in param dict
        for key in param_dict.keys():
            if j_name in key:

                # update xyz_rpy with kinematic errors based on identifiable axis
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose == 1:
                            logger.debug(
                                "Updating [{}] joint placement at axis {} with [{}]".format(
                                    j_name, axis, key
                                )
                            )
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)

        # updaet joint placement
        model = update_joint_placement(model, j_id, xyz_rpy)

    # joint elasticity: one compliance parameter per active joint (ELAS_TPL,
    # see _build_elastic_param_names), matched once here since the mapping
    # joint -> param is static; the deflection itself is gravity-torque
    # dependent and recomputed per sample below.
    elastic_map = {}
    if calib_config.get("non_geom"):
        for j_id in calib_config["actJoint_idx"]:
            j_name = model.names[j_id]
            for key in param_dict.keys():
                if j_name in key:
                    for elas_id, elas in enumerate(ELAS_TPL):
                        if elas in key:
                            if verbose == 1:
                                logger.debug(
                                    "Joint [{}] elastic gain [{}] on axis {}".format(
                                        j_name, key, elas
                                    )
                                )
                            elastic_map[j_id] = (key, elas_id)
                            updated_params.append(key)

    # check if all parameters are updated to the model
    assert len(updated_params) == len(
        list(param_dict.keys())
    ), "Not all parameters are updated {} and {}".format(
        updated_params, list(param_dict.keys())
    )

    # pose vector of the end-effector
    PEE = np.zeros((calib_config["calibration_index"], calib_config["NbSample"]))

    q_ = np.copy(q)
    for i in range(calib_config["NbSample"]):

        if backend is not None:
            backend.compute_forward_kinematics(q_[i, :])
        else:
            pin.framesForwardKinematics(model, data, q_[i, :])
            pin.updateFramePlacements(model, data)

        if elastic_map:
            if backend is not None:
                tau = backend.compute_gravity_vector(q_[i, :])
            else:
                tau = pin.computeGeneralizedGravity(model, data, q_[i, :])

            for j_id, (key, elas_id) in elastic_map.items():
                xyz_rpy = np.zeros(6)
                xyz_rpy[elas_id] = param_dict[key] * tau[j_id - 1]
                model = update_joint_placement(model, j_id, xyz_rpy)

            # jointPlacements changed: data.oMf is stale until FK is redone
            if backend is not None:
                backend.compute_forward_kinematics(q_[i, :])
            else:
                pin.framesForwardKinematics(model, data, q_[i, :])
                pin.updateFramePlacements(model, data)

            oMee = get_rel_transform(model, data, start_f, end_f)

            # revert model back to origin from the added joint elastic error
            for j_id, (key, elas_id) in elastic_map.items():
                xyz_rpy = np.zeros(6)
                xyz_rpy[elas_id] = param_dict[key] * tau[j_id - 1]
                model = update_joint_placement(model, j_id, -xyz_rpy)
        else:
            oMee = get_rel_transform(model, data, start_f, end_f)

        # calculate transformation from world frame to end-effector frame
        wMee = wMo * oMee
        wMf = wMee * eeMf

        # final transform
        trans = wMf.translation.tolist()
        orient = pin.rpy.matrixToRpy(wMf.rotation).tolist()
        loc = trans + orient
        measure = []
        for mea_id, mea in enumerate(calib_config["measurability"]):
            if mea:
                measure.append(loc[mea_id])
        PEE[:, i] = np.array(measure)

    # final result of updated fkm
    PEE = PEE.flatten("C")

    # revert model back to original
    assert origin_model.jointPlacements != model.jointPlacements, "before revert"
    for j_id in calib_config["actJoint_idx"]:
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
    """Update joint placement with offset parameters.

    Modifies a joint's placement transform by adding position and orientation offsets.

    Args:
        model (pin.Model): Robot model to modify
        joint_idx (int): Index of joint to update
        xyz_rpy (ndarray): (6,) array of offsets:
            - xyz_rpy[0:3]: Translation offsets (x,y,z)
            - xyz_rpy[3:6]: Rotation offsets (roll,pitch,yaw)

    Returns:
        pin.Model: Updated robot model

    Side Effects:
        Modifies model.jointPlacements[joint_idx] in place
    """
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


# BASE REGRESSOR TOOLS


def calculate_kinematics_model(q_i, model, data, calib_config, backend=None):
    """Calculate Jacobian and kinematic regressor for single configuration.

    Computes frame Jacobian and kinematic regressor matrices for tool frame
    at given joint configuration.

    Args:
        q_i (ndarray): Joint configuration vector
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        calib_config (dict): Parameters containing "IDX_TOOL" frame index
        backend (DynamicsBackend, optional): If provided, routes forward kinematics
            and Jacobian calls through the backend abstraction.

    Returns:
        tuple:
            - model (pin.Model): Updated model
            - data (pin.Data): Updated data
            - R (ndarray): (6,6n) Kinematic regressor matrix
            - J (ndarray): (6,n) Frame Jacobian matrix
    """
    if backend is not None:
        # compute_forward_kinematics updates FK internally
        backend.compute_forward_kinematics(q_i)
        # Convert frame ID to name for backend Jacobian
        frame_id = calib_config["IDX_TOOL"]
        frame_name = model.frames[frame_id].name
        J = backend.compute_jacobian(q_i, frame_name)
    else:
        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)
        J = pin.computeFrameJacobian(
            model, data, q_i, calib_config["IDX_TOOL"], pin.LOCAL
        )

    # computeFrameKinematicRegressor has no backend equivalent — use escape hatch
    R = pin.computeFrameKinematicRegressor(
        model, data, calib_config["IDX_TOOL"], pin.LOCAL
    )
    return model, data, R, J


def calculate_identifiable_kinematics_model(q, model, data, calib_config, backend=None):
    """Calculate identifiable Jacobian and regressor matrices.

    Builds aggregated Jacobian and regressor matrices from either:
    1. Given set of configurations, or
    2. Random configurations if none provided

    Args:
        q (ndarray, optional): Joint configurations matrix. If empty, uses random configs.
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        calib_config (dict): Parameters containing:
            - NbSample: Number of configurations
            - calibration_index: Number of active DOFs
            - start_frame, end_frame: Frame names
            - calib_model: Model type
        backend (DynamicsBackend, optional): If provided, routes random configuration
            and forwards backend to called functions.

    Returns:
        ndarray: Either:
            - Joint offset case: Frame Jacobian matrix
            - Full params case: Kinematic regressor matrix

    Note:
        Removes rows corresponding to inactive DOFs and zero elements
    """
    q_temp = np.copy(q)
    # Note if no q id given then use random generation of q to determine the
    # minimal kinematics model
    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain aggreated Jacobian matrix J and kinematic regressor R
    calib_idx = calib_config["calibration_index"]
    R = np.zeros([6 * calib_config["NbSample"], 6 * (model.njoints - 1)])
    J = np.zeros([6 * calib_config["NbSample"], model.njoints - 1])
    for i in range(calib_config["NbSample"]):
        if MIN_MODEL == 1:
            if backend is not None:
                q_rand = backend.random_configuration()
            else:
                q_rand = pin.randomConfiguration(model)
            q_i = calib_config["q0"]
            q_i[calib_config["config_idx"]] = q_rand[calib_config["config_idx"]]
        else:
            q_i = q_temp[i, :]
        if calib_config["start_frame"] == "universe":
            model, data, Ri, Ji = calculate_kinematics_model(
                q_i, model, data, calib_config, backend=backend
            )
        else:
            Ri = get_rel_kinreg(
                model,
                data,
                calib_config["start_frame"],
                calib_config["end_frame"],
                q_i,
                backend=backend,
            )
            # Ji = np.zeros([6, model.njoints-1]) ## TODO: get_rel_jac
            Ji = get_rel_jac(
                model,
                data,
                calib_config["start_frame"],
                calib_config["end_frame"],
                q_i,
                backend=backend,
            )
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                R[calib_config["NbSample"] * j + i, :] = Ri[j, :]
                J[calib_config["NbSample"] * j + i, :] = Ji[j, :]
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

    # select regressor matrix based on calibration model
    if calib_config["calib_model"] == "joint_offset":
        return J
    elif calib_config["calib_model"] == "full_params":
        return R


def calculate_base_kinematics_regressor(
    q, model, data, calib_config, tol_qr=TOL_QR, backend=None
):
    """Calculate base regressor matrix for calibration parameters.

    Identifies base (identifiable) parameters by:
    1. Computing regressors with random/given configurations
    2. Eliminating unidentifiable parameters
    3. Finding independent regressor columns

    Args:
        q (ndarray): Joint configurations matrix
        model (pin.Model): Robot model
        data (pin.Data): Robot data
        calib_config (dict): Contains calibration settings:
            - free_flyer: Whether base is floating
            - calib_model: Either "joint_offset" or "full_params"
        tol_qr (float, optional): QR decomposition tolerance. Defaults to TOL_QR.
        backend (DynamicsBackend, optional): If provided, forwards backend to
            called functions for backend-aware computation.

    Returns:
        tuple:
            - Rrand_b (ndarray): Base regressor from random configs
            - R_b (ndarray): Base regressor from given configs
            - R_e (ndarray): Full regressor after eliminating unidentifiable params
            - paramsrand_base (list): Names of base parameters from random configs
            - paramsrand_e (list): Names of identifiable parameters

    Side Effects:
        - Updates calib_config["param_name"] with identified base parameters
        - Prints regressor matrix shapes
    """
    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names[1:])]
    geo_params = get_fullparam_offset(joint_names)
    joint_offsets = get_joint_offset(model, joint_names)

    # calculate kinematic regressor with random configs
    if not calib_config["free_flyer"]:
        Rrand = calculate_identifiable_kinematics_model(
            [], model, data, calib_config, backend=backend
        )
    else:
        Rrand = calculate_identifiable_kinematics_model(
            q, model, data, calib_config, backend=backend
        )
    # calculate kinematic regressor with input configs
    if np.any(np.array(q)):
        R = calculate_identifiable_kinematics_model(
            q, model, data, calib_config, backend=backend
        )
    else:
        R = Rrand

    # only joint offset parameters
    if calib_config["calib_model"] == "joint_offset":
        geo_params_sel = joint_offsets

        # select columns corresponding to joint_idx
        Rrand_sel = Rrand

        # select columns corresponding to joint_idx
        R_sel = R

    # full 6 parameters
    elif calib_config["calib_model"] == "full_params":
        geo_params_sel = geo_params
        Rrand_sel = Rrand
        R_sel = R

    # remove non affect columns from random data => reduced regressor
    Rrand_e, paramsrand_e = eliminate_non_dynaffect(
        Rrand_sel, geo_params_sel, tol_e=1e-6
    )

    # indices of independent columns (base param) w.r.t to reduced regressor
    idx_base = get_baseIndex(Rrand_e, paramsrand_e, tol_qr=tol_qr)

    # get base regressor and base params from random data
    Rrand_b, paramsrand_base, _ = get_baseParams(Rrand_e, paramsrand_e, tol_qr=tol_qr)

    # remove non affect columns from GIVEN data
    R_e, params_e = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

    # get base param from given data
    # idx_gbase = get_baseIndex(R_e, params_e, tol_qr=tol_qr)
    R_gb, params_gbase, _ = get_baseParams(R_e, params_e, tol_qr=tol_qr)

    # get base regressor from GIVEN data
    R_b = build_baseRegressor(R_e, idx_base)

    # update calibrating calib_config['param_name']/calibrating parameters
    for j in idx_base:
        calib_config["param_name"].append(paramsrand_e[j])

    return Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e
