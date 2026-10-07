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

from ..tools.regressor import eliminate_non_dynaffect
from ..tools.qrdecomposition import (
    QRDecomposer,
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

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
    "apply_joint_offset",
    "random_joint_configuration",
    "estimate_frames_closed_form",
    "measurement_jacobian",
    "select_identifiable_parameters",
    "drop_calibration_parameters",
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
    oMf = data.oMi[frame.parentJoint] * frame.placement
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
          params), or identity if not estimated. With several markers
          (``NbMarkers`` > 1, e.g. the points of one rigid body), each
          marker ``k`` has its own ``eeMf_k`` (``pEEx_k`` ... ``phiEEz_k``)
          on the same tool frame, and its rows follow marker ``k - 1``'s
          (figaroh-plus#119).

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
            - NbMarkers: Number of markers on the tool frame
        verbose (int, optional): Print update info. Defaults to 0.
        backend (DynamicsBackend, optional): If provided, routes forward
            kinematics and gravity calls through the backend abstraction.

    Returns:
        ndarray: Flattened marker measurements in world frame, ordered
        marker, then measured component, then sample (as ``load_data``)

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

    # store parameter updated to the model
    updated_params = []

    # check if baseframe and end--effector frame are known; with no
    # parameters at all (nominal FK) neither is (#129)
    base_param_incl = any("base" in key for key in param_dict)
    ee_param_incl = any("EE" in key for key in param_dict)

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

    # 2/ calculate transformation from the end frame to each marker frame,
    # if not known: eeMf (one per marker, identity when not estimated)
    n_markers = calib_config["NbMarkers"]
    eeMf = [pin.SE3.Identity()] * n_markers
    if ee_param_incl:
        for marker_idx in range(1, n_markers + 1):
            pee = np.zeros(6)
            for axis_pee_id, axis_pee in enumerate(EE_TPL):
                key = "{}_{}".format(axis_pee, marker_idx)
                if key in param_dict:
                    if verbose == 1:
                        logger.debug("Updating marker frame with [{}]".format(key))
                    pee[axis_pee_id] += param_dict[key]
                    updated_params.append(key)
            eeMf[marker_idx - 1] = cartesian_to_SE3(pee)

    # 3/ calculate transformation from start frame to end frame of kinematic chain using updated model: oMee

    # placements are restored from this copy once the samples are evaluated
    saved_placements = {
        j_id: model.jointPlacements[j_id].copy()
        for j_id in calib_config["actJoint_idx"]
    }

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

        # update joint placement: full_params perturb the placement in the
        # parent frame; joint offsets act about the joint's own axes (q + offset)
        if calib_config["calib_model"] == "joint_offset":
            model = apply_joint_offset(model, j_id, xyz_rpy)
        else:
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

    # pose vector of the markers: rows are marker-major, then component
    n_dofs = calib_config["calibration_index"]
    PEE = np.zeros((n_markers * n_dofs, calib_config["NbSample"]))

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

            geometric_placements = {
                j_id: model.jointPlacements[j_id].copy() for j_id in elastic_map
            }
            for j_id, (key, elas_id) in elastic_map.items():
                xyz_rpy = np.zeros(6)
                xyz_rpy[elas_id] = param_dict[key] * tau[j_id - 1]
                model = apply_joint_offset(model, j_id, xyz_rpy)

            # jointPlacements changed: data.oMf is stale until FK is redone
            if backend is not None:
                backend.compute_forward_kinematics(q_[i, :])
            else:
                pin.framesForwardKinematics(model, data, q_[i, :])
                pin.updateFramePlacements(model, data)

            oMee = get_rel_transform(model, data, start_f, end_f)

            # revert model back to origin from the added joint elastic error
            for j_id, placement in geometric_placements.items():
                model.jointPlacements[j_id] = placement
        else:
            oMee = get_rel_transform(model, data, start_f, end_f)

        # calculate transformation from world frame to each marker frame
        wMee = wMo * oMee
        for k, eeMf_k in enumerate(eeMf):
            wMf = wMee * eeMf_k
            trans = wMf.translation.tolist()
            orient = pin.rpy.matrixToRpy(wMf.rotation).tolist()
            loc = trans + orient
            measure = [
                loc[mea_id]
                for mea_id, mea in enumerate(calib_config["measurability"])
                if mea
            ]
            PEE[k * n_dofs : (k + 1) * n_dofs, i] = np.array(measure)

    # final result of updated fkm
    PEE = PEE.flatten("C")

    # revert model back to original
    for j_id, placement in saved_placements.items():
        model.jointPlacements[j_id] = placement

    return PEE


def update_joint_placement(model, joint_idx, xyz_rpy):
    """Apply a ``full_params`` placement error in the joint frame.

    ``M <- M * SE3(exp3(xyz_rpy[3:6]), xyz_rpy[0:3])``: the translation is
    expressed in the joint frame and the rotation is a rotation vector. This
    is the convention of Pinocchio's kinematic regressor
    (``computeFrameKinematicRegressor(..., LOCAL)``), from which the base
    parameters and the base-mapping matrix are derived, so the fitted model
    has exactly the dependencies that selection assumed (figaroh-plus#110).
    Earlier versions added the translation in the parent frame and the
    rotation to the placement's RPY angles, which disagrees with the
    regressor wherever the nominal placement is rotated.

    Args:
        model (pin.Model): Robot model to modify
        joint_idx (int): Index of joint to update
        xyz_rpy (ndarray): (6,) ``d_px, d_py, d_pz`` (m) and ``d_phix,
            d_phiy, d_phiz`` (rad, rotation vector), in the joint frame

    Returns:
        pin.Model: Updated robot model

    Side Effects:
        Modifies model.jointPlacements[joint_idx] in place
    """
    xyz_rpy = np.asarray(xyz_rpy, dtype=float)
    delta = pin.SE3(pin.exp3(xyz_rpy[3:6]), xyz_rpy[0:3])
    model.jointPlacements[joint_idx] = model.jointPlacements[joint_idx] * delta
    return model


def apply_joint_offset(model, joint_idx, offset):
    """Offset a joint about its own axes (joint-angle / joint-position offset).

    Composes the offset on the child side of the joint placement,
    ``M <- M * SE3(R(offset[3:6]), offset[0:3])``, so that it is expressed in
    the joint frame. For a revolute joint about z, ``offset[5] = d`` is then
    exactly the configuration ``q + d``; for a prismatic joint along x,
    ``offset[0] = d`` is ``q + d``. This is the meaning of the
    ``offset{PX,PY,PZ,RX,RY,RZ}_<joint>`` parameters (``JOINT_OFFSETTPL``,
    axis taken from the joint's ``shortname()``) and of the elastic
    deflections (``ELAS_TPL``).

    :func:`update_joint_placement` composes on the same side for the
    ``full_params`` (``d_p*``, ``d_phi*``) placement errors, with all three
    rotation components as one rotation vector.

    Args:
        model (pin.Model): Robot model to modify
        joint_idx (int): Index of joint to update
        offset (ndarray): (6,) translation (x, y, z) and rotation (roll,
            pitch, yaw) offsets, expressed in the joint frame

    Returns:
        pin.Model: Updated robot model

    Side Effects:
        Modifies model.jointPlacements[joint_idx] in place
    """
    offset = np.asarray(offset, dtype=float)
    delta = pin.SE3(pin.rpy.rpyToMatrix(offset[3:6]), offset[0:3])
    model.jointPlacements[joint_idx] = model.jointPlacements[joint_idx] * delta
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
    R = np.zeros([6 * calib_config["NbSample"], 6 * (model.njoints - 1)])
    J = np.zeros([6 * calib_config["NbSample"], model.njoints - 1])
    # seeded, so the selected parameter set does not depend on global RNG
    # state (figaroh-plus#99)
    rng = np.random.default_rng(calib_config.get("random_seed", 0))
    for i in range(calib_config["NbSample"]):
        if MIN_MODEL == 1:
            q_rand = random_joint_configuration(model, rng)
            # a copy: q0 is robot.q0 and must not be overwritten (#125)
            q_i = calib_config["q0"].copy()
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

    # Several markers of one body (figaroh-plus#119) observe the tool
    # frame's orientation, even when each marker's position alone is
    # measured: the structural regressor then uses every component.
    # Dependencies the actual points leave are removed at the data level.
    reg_config = calib_config
    if calib_config.get("NbMarkers", 1) > 1:
        reg_config = dict(calib_config, measurability=[True] * 6, calibration_index=6)

    # calculate kinematic regressor with random configs
    if not calib_config["free_flyer"]:
        Rrand = calculate_identifiable_kinematics_model(
            [], model, data, reg_config, backend=backend
        )
    else:
        Rrand = calculate_identifiable_kinematics_model(
            q, model, data, reg_config, backend=backend
        )
    # calculate kinematic regressor with input configs
    if np.any(np.array(q)):
        R = calculate_identifiable_kinematics_model(
            q, model, data, reg_config, backend=backend
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

    # indices of independent columns (base param) w.r.t the reduced
    # regressor, and the base-mapping matrix M s.t. phi_base = M @ theta_r
    # (theta_r ordered as paramsrand_e) -- one QR pass on the structural
    # (random-config) regressor covers both, since the base/dependent
    # column split is a property of the kinematic chain, not of any
    # particular dataset.
    decomposer = QRDecomposer(tolerance=tol_qr)
    M, paramsrand_base, idx_base, _ = decomposer.get_base_mapping_matrix_double(
        Rrand_e, paramsrand_e
    )

    # get base regressor from random data
    Rrand_b = build_baseRegressor(Rrand_e, idx_base)

    # remove non affect columns from GIVEN data
    R_e, _ = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

    # get base regressor from GIVEN data
    R_b = build_baseRegressor(R_e, idx_base)

    # update calibrating calib_config['param_name']/calibrating parameters
    # -- record the slice these base entries land at (param_name may
    # already carry earlier entries, e.g. elastic-gain params) so callers
    # can locate the corresponding values in a solved variable vector
    # (which is built one-to-one, in order, from param_name) regardless of
    # any later in-place renaming (add_base_name) or appending (add_pee_name).
    _base_slice_start = len(calib_config["param_name"])
    for j in idx_base:
        calib_config["param_name"].append(paramsrand_e[j])
    _base_slice_end = len(calib_config["param_name"])

    # Structural base-mapping, stashed for optional downstream
    # redistribution (see figaroh.tools.qrdecomposition.redistribute_min_norm
    # and BaseCalibration.redistribute_parameters). base_mapping_row_names
    # is a snapshot of the base-parameter name order *before*
    # add_base_name/add_pee_name may rename/prepend entries in
    # calib_config["param_name"], so callers should locate fitted values by
    # base_mapping_slice (positional), not by re-looking-up these names.
    calib_config["base_mapping_matrix"] = M
    calib_config["base_mapping_param_names"] = list(paramsrand_e)
    calib_config["base_mapping_row_names"] = [paramsrand_e[j] for j in idx_base]
    calib_config["base_mapping_slice"] = (_base_slice_start, _base_slice_end)

    return Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e


# FRAME INITIALISATION AND DATA-LEVEL IDENTIFIABILITY


def random_joint_configuration(model, rng):
    """Draw a configuration uniformly within joint limits from ``rng``.

    One-DoF joints are drawn within their position limits, or within
    [-pi, pi] when the limits are missing or wider than a turn; other joints
    stay at the neutral configuration. Unlike ``pin.randomConfiguration``,
    the draw depends only on ``rng``, not on Pinocchio's global generator.

    Args:
        model (pin.Model): Robot model
        rng (np.random.Generator): Random generator

    Returns:
        ndarray: (nq,) configuration
    """
    q = pin.neutral(model)
    for joint in model.joints[1:]:
        if joint.nq != 1:
            continue
        lo = model.lowerPositionLimit[joint.idx_q]
        hi = model.upperPositionLimit[joint.idx_q]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo > 2 * np.pi:
            lo, hi = -np.pi, np.pi
        q[joint.idx_q] = rng.uniform(lo, hi)
    return q


def _kabsch(source, target):
    """Rotation R minimising sum |R (s - s_mean) - (t - t_mean)|^2."""
    H = (source - source.mean(0)).T @ (target - target.mean(0))
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _chordal_mean(rotations):
    """Rotation closest (Frobenius) to the mean of ``rotations``."""
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def estimate_frames_closed_form(model, data, q, PEE, calib_config, n_iter=50):
    """Closed-form initial guess for the unknown base and tip frames.

    With nominal joint parameters, the measured marker satisfies
    ``P_i = R_b (p_i + R_i t_tip) + t_b`` (and ``R_meas_i = R_b R_i R_tip``
    when orientation is measured), where ``(R_i, p_i)`` is the nominal
    start-to-end frame transform. Rotations are estimated by Kabsch
    (positions) or chordal averaging (orientations), alternated with a linear
    least-squares solve for ``t_b`` and ``t_tip``.

    With several markers (``NbMarkers`` > 1, points of one rigid body,
    figaroh-plus#119), each has its own ``t_tip_k`` and the base frame is
    fitted to all of them; orientations, if measured, are not used then and
    tip rotations are guessed as zero.

    Only applies to markers whose position is fully measured, with the
    base frame estimated directly (``base_*`` parameters, no camera
    ``base_to_ref_frame`` anchor). Otherwise returns an empty dict.

    Args:
        model (pin.Model): Robot model (nominal joint placements)
        data (pin.Data): Robot data
        q (ndarray): (NbSample, nq) joint configurations
        PEE (ndarray): Flattened DOF-major measurements, as from ``load_data``
        calib_config (dict): Calibration configuration
        n_iter (int): Alternation iterations

    Returns:
        dict: Initial values keyed by the ``base_*``, ``pEE*``/``phiEE*``
        names present in ``calib_config["param_name"]``.
    """
    names = list(calib_config["param_name"])
    meas = list(calib_config["measurability"])
    n_markers = calib_config.get("NbMarkers", 1)
    if (
        not any(n in names for n in BASE_TPL)
        or calib_config.get("base_to_ref_frame") is not None
        or not all(meas[:3])
    ):
        return {}
    n = len(q)
    M = np.asarray(PEE, dtype=float).reshape(n_markers, sum(meas), n)
    P = np.transpose(M[:, :3], (0, 2, 1))  # (n_markers, n, 3)
    orient = n_markers == 1 and len(meas) == 6 and all(meas[3:6])
    if orient:
        R_meas = np.array([pin.rpy.rpyToMatrix(M[0, 3:6, i]) for i in range(n)])

    R, p = np.empty((n, 3, 3)), np.empty((n, 3))
    for i in range(n):
        pin.framesForwardKinematics(model, data, q[i])
        T = get_rel_transform(
            model, data, calib_config["start_frame"], calib_config["end_frame"]
        )
        R[i], p[i] = T.rotation, T.translation

    tip_pos = [
        any(f"{e}_{k + 1}" in names for e in EE_TPL[:3]) for k in range(n_markers)
    ]
    tip_rot = orient and any(f"{e}_1" in names for e in EE_TPL[3:])
    R_b, R_tip = np.eye(3), np.eye(3)
    t_b, t_tip = np.zeros(3), np.zeros((n_markers, 3))
    free = [k for k in range(n_markers) if tip_pos[k]]
    for _ in range(n_iter):
        if orient:
            R_b = _chordal_mean(R_meas @ np.transpose(R @ R_tip, (0, 2, 1)))
            if tip_rot:
                R_tip = _chordal_mean(np.transpose(R_b @ R, (0, 2, 1)) @ R_meas)
        else:
            source = np.concatenate([p + R @ t_tip[k] for k in range(n_markers)])
            R_b = _kabsch(source, P.reshape(-1, 3))
        # P_k,i - R_b p_i = R_b R_i t_tip_k + t_b, linear in (t_tip_k, t_b);
        # tips not estimated stay at 0
        rhs = (P - (p @ R_b.T)[None]).reshape(-1)
        A = np.zeros((n_markers, n, 3, 3 * len(free) + 3))
        for col, k in enumerate(free):
            A[k, :, :, 3 * col : 3 * col + 3] = R_b @ R
        A[:, :, :, -3:] = np.eye(3)
        sol = np.linalg.lstsq(A.reshape(-1, A.shape[-1]), rhs, rcond=None)[0]
        for col, k in enumerate(free):
            t_tip[k] = sol[3 * col : 3 * col + 3]
        t_b = sol[-3:]

    values = np.concatenate([t_b, pin.rpy.matrixToRpy(R_b)])
    guess = {n_: v for n_, v in zip(BASE_TPL, values) if n_ in names}
    for k in range(n_markers):
        tip = np.concatenate([t_tip[k], pin.rpy.matrixToRpy(R_tip)])
        for e, v in zip(EE_TPL, tip):
            if f"{e}_{k + 1}" in names:
                guess[f"{e}_{k + 1}"] = v
    return guess


def measurement_jacobian(model, data, var, q, calib_config, step=1e-6):
    """Central-difference Jacobian of ``calc_updated_fkm`` w.r.t. ``var``.

    Rows follow the flattened, DOF-major measurement vector; columns follow
    ``calib_config["param_name"]``.
    """
    var = np.asarray(var, dtype=float)
    cfg = dict(calib_config, NbSample=len(q))
    cols = []
    for j in range(len(var)):
        dv = np.zeros_like(var)
        dv[j] = step
        hi = calc_updated_fkm(model, data, var + dv, q, cfg)
        lo = calc_updated_fkm(model, data, var - dv, q, cfg)
        cols.append((hi - lo) / (2 * step))
    return np.column_stack(cols)


def select_identifiable_parameters(jacobian, names, always_keep=(), tol=1e-4):
    """Split parameters into identifiable and absorbed ones, deterministically.

    Columns are normalised to unit length (so units do not matter) and taken
    in order: first ``always_keep`` (e.g. base and tip frames), then the rest
    of ``names`` in order. A column is kept when its component orthogonal to
    the columns already kept exceeds ``tol``; otherwise it is a combination of
    them and is reported as absorbed. ``always_keep`` columns are never
    dropped.

    Args:
        jacobian (ndarray): (n_meas, n_params) measurement Jacobian
        names (list): Parameter names, one per column
        always_keep (iterable): Names kept unconditionally and tested first
        tol (float): Threshold on the orthogonal residual of a unit column.
            Exact dependencies give ~1e-10 (finite-difference noise); the
            default 1e-4 also drops near-dependencies whose column alone
            would have a condition number above 1e4 (a direction the data
            barely excites).

    Returns:
        tuple: (kept names in original order, absorbed names in original order)
    """
    names = list(names)
    always_keep = [n for n in names if n in set(always_keep)]
    order = always_keep + [n for n in names if n not in set(always_keep)]
    basis = np.zeros((jacobian.shape[0], 0))
    kept = set()
    for name in order:
        col = jacobian[:, names.index(name)]
        norm = np.linalg.norm(col)
        if norm > 0:
            r = col / norm
            for _ in range(2):  # re-orthogonalise for numerical stability
                r = r - basis @ (basis.T @ r)
            res = np.linalg.norm(r)
        else:
            res = 0.0
        if name in always_keep or res > tol:
            kept.add(name)
            if res > tol:
                basis = np.column_stack([basis, r / res])
    return [n for n in names if n in kept], [n for n in names if n not in kept]


def drop_calibration_parameters(calib_config, dropped):
    """Remove parameters from ``param_name``, keeping the base mapping aligned.

    A dropped name inside ``base_mapping_slice`` also loses its row of
    ``base_mapping_matrix`` / ``base_mapping_row_names``; names before the
    slice shift it left.
    """
    names = list(calib_config["param_name"])
    for idx in sorted((names.index(n) for n in dropped), reverse=True):
        if "base_mapping_slice" in calib_config:
            start, end = calib_config["base_mapping_slice"]
            if start <= idx < end:
                row = idx - start
                calib_config["base_mapping_matrix"] = np.delete(
                    calib_config["base_mapping_matrix"], row, axis=0
                )
                rows = list(calib_config["base_mapping_row_names"])
                del rows[row]
                calib_config["base_mapping_row_names"] = rows
                calib_config["base_mapping_slice"] = (start, end - 1)
            elif idx < start:
                calib_config["base_mapping_slice"] = (start - 1, end - 1)
        del names[idx]
    calib_config["param_name"] = names
