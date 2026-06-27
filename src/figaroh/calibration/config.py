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

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def get_param_from_yaml(robot, calib_data) -> dict:
    """Parse calibration parameters from YAML configuration file.

    Processes robot and calibration data to build a parameter dictionary containing
    all necessary settings for robot calibration. Handles configuration of:
    - Frame identifiers and relationships
    - Marker/measurement settings
    - Joint indices and configurations
    - Non-geometric parameters
    - Eye-hand calibration setup

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        calib_data (dict): Calibration parameters parsed from YAML file containing:
            - markers: List of marker configurations
            - calib_level: Calibration model type
            - base_frame: Starting frame name
            - tool_frame: End frame name
            - free_flyer: Whether base is floating
            - non_geom: Whether to include non-geometric params

    Returns:
        dict: Parameter dictionary containing:
            - robot_name: Name of robot model
            - NbMarkers: Number of markers
            - measurability: Measurement DOFs per marker
            - start_frame, end_frame: Frame names
            - base_to_ref_frame: Optional camera frame
            - IDX_TOOL: Tool frame index
            - actJoint_idx: Active joint indices
            - param_name: List of parameter names
            - Additional settings from YAML

    Side Effects:
        Prints warning messages if optional frames undefined
        Prints final parameter dictionary

    Example:
        >>> calib_data = yaml.safe_load(config_file)
        >>> params = get_param_from_yaml(robot, calib_data)
        >>> print(params['NbMarkers'])
        2
    """
    # NOTE: since joint 0 is universe and it is trivial,
    # indices of joints are different from indices of joint configuration,
    # different from indices of joint velocities
    calib_config = dict()
    robot_name = robot.model.name
    frames = [f.name for f in robot.model.frames]
    calib_config["robot_name"] = robot_name

    # End-effector sensing measurability:
    NbMarkers = len(calib_data["markers"])
    measurability = calib_data["markers"][0]["measure"]
    calib_idx = measurability.count(True)
    calib_config["NbMarkers"] = NbMarkers
    calib_config["measurability"] = measurability
    calib_config["calibration_index"] = calib_idx

    # Calibration model
    calib_config["calib_model"] = calib_data["calib_level"]

    # Get start and end frames
    start_frame = calib_data["base_frame"]
    end_frame = calib_data["tool_frame"]

    # Validate frames exist
    err_msg = "{}_frame {} does not exist"
    if start_frame not in frames:
        raise AssertionError(err_msg.format("Start", start_frame))
    if end_frame not in frames:
        raise AssertionError(err_msg.format("End", end_frame))

    calib_config["start_frame"] = start_frame
    calib_config["end_frame"] = end_frame

    # Handle eye-hand calibration frames
    try:
        base_to_ref_frame = calib_data["base_to_ref_frame"]
        ref_frame = calib_data["ref_frame"]
    except KeyError:
        base_to_ref_frame = None
        ref_frame = None
        logger.warning("base_to_ref_frame and ref_frame are not defined.")

    # Validate base-to-ref frame if provided
    if base_to_ref_frame is not None:
        if base_to_ref_frame not in frames:
            err_msg = "base_to_ref_frame {} does not exist"
            raise AssertionError(err_msg.format(base_to_ref_frame))

    # Validate ref frame if provided
    if ref_frame is not None:
        if ref_frame not in frames:
            err_msg = "ref_frame {} does not exist"
            raise AssertionError(err_msg.format(ref_frame))

    calib_config["base_to_ref_frame"] = base_to_ref_frame
    calib_config["ref_frame"] = ref_frame

    # Get initial poses
    try:
        base_pose = calib_data["base_pose"]
        tip_pose = calib_data["tip_pose"]
    except KeyError:
        base_pose = None
        tip_pose = None
        logger.warning("base_pose and tip_pose are not defined.")

    calib_config["base_pose"] = base_pose
    calib_config["tip_pose"] = tip_pose

    # q0: default zero configuration
    calib_config["q0"] = robot.q0
    calib_config["NbSample"] = calib_data["nb_sample"]

    # IDX_TOOL: frame ID of the tool
    IDX_TOOL = robot.model.getFrameId(end_frame)
    calib_config["IDX_TOOL"] = IDX_TOOL

    # tool_joint: ID of the joint right before the tool's frame (parent)
    tool_joint = robot.model.frames[IDX_TOOL].parentJoint
    calib_config["tool_joint"] = tool_joint

    # indices of active joints: from base to tool_joint
    actJoint_idx = get_sup_joints(robot.model, start_frame, end_frame)
    calib_config["actJoint_idx"] = actJoint_idx

    # indices of joint configuration corresponding to active joints
    config_idx = [robot.model.joints[i].idx_q for i in actJoint_idx]
    calib_config["config_idx"] = config_idx

    # number of active joints
    NbJoint = len(actJoint_idx)
    calib_config["NbJoint"] = NbJoint

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
    calib_config["param_name"] = param_name

    calib_config.update(
        {
            "free_flyer": calib_data["free_flyer"],
            "non_geom": calib_data["non_geom"],
            "eps": 1e-3,
            "PLOT": 0,
        }
    )
    try:
        calib_config.update(
            {
                "coeff_regularize": calib_data["coeff_regularize"],
                "data_file": calib_data["data_file"],
                "sample_configs_file": calib_data["sample_configs_file"],
                "outlier_eps": calib_data["outlier_eps"],
            }
        )
    except KeyError:
        calib_config.update(
            {
                "coeff_regularize": None,
                "data_file": None,
                "sample_configs_file": None,
                "outlier_eps": None,
            }
        )
    return calib_config


def unified_to_legacy_config(robot, unified_calib_config) -> dict:
    """Convert unified configuration format to legacy calib_config format.

    Maps the new unified configuration structure to the exact format expected
    by get_param_from_yaml. This ensures backward compatibility while using
    the new unified parser.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_calib_config (dict): Configuration from create_task_config

    Returns:
        dict: Legacy format calibration configuration matching
              get_param_from_yaml output

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

    # 8. Extract data configuration
    calib_config["NbSample"] = data.get("number_of_samples", 500)
    calib_config["data_file"] = data.get("source_file")
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

    # Store calibration settings
    calib_config.update(
        {
            "free_flyer": parameters.get("free_flyer", False),
            "non_geom": non_geom,
            "eps": 1e-3,
            "PLOT": 0,
            "coeff_regularize": parameters.get("regularization_coefficient", 0.01),
            "outlier_eps": parameters.get("outlier_threshold", 0.05),
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


# Backward compatibility wrapper for get_param_from_yaml
def get_param_from_yaml_legacy(robot, calib_data) -> dict:
    """Legacy calibration parameter parser - kept for backward compatibility.

    This is the original implementation. New code should use the unified
    config parser from figaroh.utils.config_parser.

    Args:
        robot: Robot instance
        calib_data: Calibration data dictionary

    Returns:
        Calibration configuration dictionary
    """
    # Keep the original implementation here for compatibility
    return get_param_from_yaml(robot, calib_data)


# Import the new unified parser as the default
try:
    from ..utils.config_parser import (
        get_param_from_yaml as unified_get_param_from_yaml,
    )

    # Replace the function with unified version while maintaining signature
    def get_param_from_yaml_unified(robot, calib_data) -> dict:
        """Enhanced parameter parser using unified configuration system.

        This function provides backward compatibility while using the new
        unified configuration parser when possible.

        Args:
            robot: Robot instance
            calib_data: Configuration data (dict or file path)

        Returns:
            Calibration configuration dictionary
        """
        try:
            return unified_get_param_from_yaml(robot, calib_data, "calibration")
        except Exception as e:
            # Fall back to legacy parser if unified parser fails
            import warnings

            warnings.warn(
                f"Unified parser failed ({e}), falling back to legacy "
                "parser. Consider updating your configuration format.",
                UserWarning,
            )
            return get_param_from_yaml_legacy(robot, calib_data)

    # Keep the old function available but with warning
    def get_param_from_yaml_with_warning(robot, calib_data) -> dict:
        """Original function with deprecation notice."""
        import warnings

        warnings.warn(
            "Direct use of get_param_from_yaml is deprecated. "
            "Consider using the unified config parser from "
            "figaroh.utils.config_parser",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_param_from_yaml_unified(robot, calib_data)

except ImportError:
    # If unified parser is not available, keep using original function
    pass
