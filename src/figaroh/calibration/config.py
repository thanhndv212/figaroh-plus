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
Configuration parsing and parameter management for robot calibration.

This module handles all configuration-related functionality including:
- YAML configuration file parsing
- Unified to legacy config format conversion
- Parameter extraction and validation
- Frame and joint configuration management
"""

import logging

from ..utils.error_handling import ConfigurationError

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def regularization_coefficient(value):
    """``coeff_regularize`` from a config value; a non-zero value is an error.

    The coefficient weighted ``sqrt(c) * theta`` rows that robot cost
    functions appended for every joint parameter, so a metre and a radian
    were penalised alike (#120). It was deprecated in 0.6.0 and removed in
    1.0: ``estimation.method: map`` replaces it with priors of physical size
    per parameter group. Zero or absent is accepted and passed through.

    Raises:
        ConfigurationError: ``value`` is non-zero.
    """
    if value:
        raise ConfigurationError(
            f"regularization_coefficient ({value}) was removed in 1.0: it "
            "penalised metres and radians alike. Remove it (or set it to 0) "
            "and use estimation.method: map (priors per parameter group, "
            "docs/source/concepts/calibration_estimation.md; figaroh-plus#120)."
        )
    return value


def get_sup_joints(model, start_frame, end_frame):
    """Get list of supporting joints between two frames in kinematic chain.

    Finds all joints that contribute to relative motion between start_frame and
    end_frame by analyzing their support branches in the kinematic tree.

    Args:
        model (pin.Model): Robot model
        start_frame (str): Name of starting frame
        end_frame (str): Name of ending frame

    Returns:
        list[int]: Joint IDs ordered from start to end frame, handling cases:
            1. Branch entirely contained in another branch
            2. Disjoint branches with fixed root
            3. Partially overlapping branches

    Raises:
        AssertionError: If end frame appears before start frame in chain

    Note:
        Excludes "universe" joints from returned list since they don't
        contribute to relative motion.
    """
    start_frameId = model.getFrameId(start_frame)
    end_frameId = model.getFrameId(end_frame)
    start_par = model.frames[start_frameId].parentJoint
    end_par = model.frames[end_frameId].parentJoint
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
    # case 2: root_joint is fixed; branch_s and branch_e are separate
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


def unified_to_legacy_config(robot, unified_calib_config) -> dict:
    """Convert unified configuration format to legacy calib_config format.

    Maps the unified configuration structure to the ``calib_config`` dict the
    calibration classes consume.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_calib_config (dict): Configuration from create_task_config

    Returns:
        dict: Calibration configuration dict (``calib_config``)

    Raises:
        KeyError: If required fields are missing from unified config
        AssertionError: If frame validation fails

    Example:
        >>> unified_config = create_task_config(robot, parsed_config,
        ...                                    "calibration")
        >>> legacy_config = unified_to_legacy_config(robot, unified_config)
    """
    # Initialize output configuration
    calib_config = {}

    # Extract unified config sections
    joints = unified_calib_config.get("joints", {})
    kinematics = unified_calib_config.get("kinematics", {})
    parameters = unified_calib_config.get("parameters", {})
    measurements = unified_calib_config.get("measurements", {})
    data = unified_calib_config.get("data", {})

    # 1. Extract basic robot information
    calib_config["robot_name"] = robot.model.name
    calib_config["q0"] = robot.q0
    calib_config["instance"] = unified_calib_config.get("instance", {})

    # 2. Extract and validate markers/measurements
    _extract_marker_info(calib_config, measurements)

    # 3. Extract and validate kinematic frames
    _extract_frame_info(calib_config, robot, kinematics)

    # 4. Extract tool frame information
    _extract_tool_info(calib_config, robot, calib_config["end_frame"])

    # 5. Determine active joints
    _determine_active_joints(
        calib_config,
        robot,
        joints,
        calib_config["start_frame"],
        calib_config["end_frame"],
    )

    # 6. Extract poses
    _extract_poses(calib_config, measurements)

    # 7. Extract calibration parameters
    _extract_calibration_params(calib_config, robot, parameters)

    # 7b. Extract eye-hand (camera) calibration frames, if enabled
    _extract_eye_hand_params(calib_config, unified_calib_config)

    # 8. Extract data configuration
    calib_config["NbSample"] = data.get("number_of_samples", 500)
    calib_config["data_file"] = data.get("source_file")
    # held-out measurements in the same format; "" means none
    calib_config["validation_data_file"] = data.get("validation_data_file") or None
    calib_config["sample_configs_file"] = data.get("sample_configurations_file")

    return calib_config


def _extract_marker_info(calib_config, measurements):
    """Extract marker and measurement information.

    Args:
        calib_config (dict): Configuration dictionary to update
        measurements (dict): Measurements section from unified config

    Raises:
        KeyError: If markers are not defined
    """
    markers = measurements.get("markers", [{}])
    if not markers:
        raise KeyError("No markers defined in unified configuration")

    first_marker = markers[0]
    measurability = first_marker.get(
        "measurable_dof", [True, True, True, False, False, False]
    )

    calib_config["NbMarkers"] = len(markers)
    calib_config["measurability"] = measurability
    calib_config["calibration_index"] = measurability.count(True)


def _extract_frame_info(calib_config, robot, kinematics):
    """Extract and validate kinematic frame information.

    Args:
        calib_config (dict): Configuration dictionary to update
        robot: Robot instance
        kinematics (dict): Kinematics section from unified config

    Raises:
        KeyError: If tool_frame not specified
        AssertionError: If frames don't exist in robot model
    """
    frames = [f.name for f in robot.model.frames]
    start_frame = kinematics.get("base_frame", "universe")
    end_frame = kinematics.get("tool_frame")

    if not end_frame:
        raise KeyError("tool_frame not specified in unified configuration")

    # Validate frames exist
    if start_frame not in frames:
        raise AssertionError(f"Start_frame {start_frame} does not exist")
    if end_frame not in frames:
        raise AssertionError(f"End_frame {end_frame} does not exist")

    calib_config["start_frame"] = start_frame
    calib_config["end_frame"] = end_frame
    # True if the robot base itself is unconstrained/floating. Lives under
    # kinematics (not parameters) in the unified schema -- see
    # templates/base_robot_config.yaml's tasks.calibration.kinematics.
    calib_config["free_flyer"] = kinematics.get("free_flying_base", False)


def _extract_eye_hand_params(calib_config, unified_calib_config):
    """Extract eye-hand (camera) calibration frames, if enabled.

    Eye-hand calibration estimates the pose of a camera/sensor mounted
    somewhere in the middle of the kinematic chain: ``camera_frame`` is
    reachable from ``base_frame`` via a *known* transform, and
    ``reference_frame`` is where the chain resumes after the *unknown*
    camera pose being estimated. See "Why the camera pose is a free
    parameter, not a small offset" in
    docs/source/concepts/config_parameters.md.

    Args:
        calib_config (dict): Configuration dictionary to update
        unified_calib_config (dict): Task-level calibration config (the
            dict returned by ``create_task_config(..., "calibration")``)
    """
    eye_hand = unified_calib_config.get("eye_hand", {})
    if isinstance(eye_hand, dict) and eye_hand.get("enabled"):
        calib_config["base_to_ref_frame"] = eye_hand.get("camera_frame")
        calib_config["ref_frame"] = eye_hand.get("reference_frame")
    else:
        calib_config["base_to_ref_frame"] = None
        calib_config["ref_frame"] = None


def _extract_tool_info(calib_config, robot, end_frame):
    """Extract tool frame information.

    Args:
        calib_config (dict): Configuration dictionary to update
        robot: Robot instance
        end_frame (str): End effector frame name
    """
    IDX_TOOL = robot.model.getFrameId(end_frame)
    tool_joint = robot.model.frames[IDX_TOOL].parentJoint

    calib_config["IDX_TOOL"] = IDX_TOOL
    calib_config["tool_joint"] = tool_joint


def _determine_active_joints(calib_config, robot, joints, start_frame, end_frame):
    """Determine active joints from configuration or kinematic chain.

    Args:
        calib_config (dict): Configuration dictionary to update
        robot: Robot instance
        joints (dict): Joints section from unified config
        start_frame (str): Starting frame name
        end_frame (str): Ending frame name
    """
    active_joint_names = joints.get("active_joints", [])

    if active_joint_names:
        # Map joint names to indices
        actJoint_idx = []
        for joint_name in active_joint_names:
            if joint_name in robot.model.names:
                joint_id = robot.model.getJointId(joint_name)
                actJoint_idx.append(joint_id)
    else:
        # Compute from kinematic chain
        actJoint_idx = get_sup_joints(robot.model, start_frame, end_frame)

    # Store joint information
    calib_config["actJoint_idx"] = actJoint_idx
    calib_config["config_idx"] = [robot.model.joints[i].idx_q for i in actJoint_idx]
    calib_config["NbJoint"] = len(actJoint_idx)


def _extract_poses(calib_config, measurements):
    """Extract base and tip pose information.

    Args:
        calib_config (dict): Configuration dictionary to update
        measurements (dict): Measurements section from unified config
    """
    poses = measurements.get("poses", {})
    base_pose = poses.get("base_pose")
    tip_pose = poses.get("tool_pose")

    calib_config["base_pose"] = base_pose
    calib_config["tip_pose"] = tip_pose

    if not base_pose and not tip_pose:
        logger.warning("base_pose and tip_pose are not defined.")


def _extract_calibration_params(calib_config, robot, parameters):
    """Extract calibration parameters including non-geometric terms.

    Args:
        calib_config (dict): Configuration dictionary to update
        robot: Robot instance
        parameters (dict): Parameters section from unified config
    """
    # Extract calibration model and settings
    calib_config["calib_model"] = parameters.get("calibration_level", "full_params")
    non_geom = parameters.get("include_non_geometric", False)

    # Build parameter names for non-geometric parameters
    param_name = []
    if non_geom:
        param_name = _build_elastic_param_names(robot, calib_config["actJoint_idx"])

    calib_config["param_name"] = param_name

    # Store calibration settings (free_flyer is extracted from kinematics
    # in _extract_frame_info, not from parameters -- see that function)
    calib_config.update(
        {
            "non_geom": non_geom,
            "eps": 1e-3,
            "PLOT": 0,
            "coeff_regularize": regularization_coefficient(
                parameters.get("regularization_coefficient", 0.0)
            ),
            "outlier_eps": parameters.get("outlier_threshold", 0.05),
            # seeds the structural selection's random configurations (#99)
            "random_seed": parameters.get("random_seed", 0),
            # parameter selection / estimation method (estimation.py, #113)
            "estimation": dict(parameters.get("estimation") or {}),
        }
    )


def _build_elastic_param_names(robot, actJoint_idx):
    """Build elastic gain parameter names for active joints.

    Args:
        robot: Robot instance
        actJoint_idx (list): List of active joint indices

    Returns:
        list: List of elastic parameter names
    """
    elastic_gain = []
    joint_axes = ["PX", "PY", "PZ", "RX", "RY", "RZ"]

    # Build elastic gain names for all joints
    for j_id, joint_name in enumerate(robot.model.names.tolist()):
        if joint_name == "universe":
            axis_motion = "null"
        else:
            shortname = robot.model.joints[j_id].shortname()
            axis_motion = _determine_axis_motion(shortname, joint_axes)

        elastic_gain.append(f"k_{axis_motion}_{joint_name}")

    # Select only active joints
    return [elastic_gain[i] for i in actJoint_idx]


def _determine_axis_motion(shortname, joint_axes):
    """Determine axis of motion from joint short name.

    Args:
        shortname (str): Joint short name from Pinocchio
        joint_axes (list): List of possible joint axes

    Returns:
        str: Axis of motion identifier
    """
    # Check for standard joint axes
    for ja in joint_axes:
        if ja in shortname:
            return ja

    # Handle special cases
    if "RevoluteUnaligned" in shortname:
        return "RZ"  # Hard-coded fix for canopies and similar robots

    return "RZ"  # Default fallback
