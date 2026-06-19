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
Parameter management utilities for robot calibration.

This module contains functions for creating and managing calibration parameter
dictionaries, including:
- Joint offset parameters
- Geometric parameter offsets
- Base frame parameters
- End-effector marker parameters
- Frame management utilities
"""

import numpy as np
import pinocchio as pin

# Constants for parameter templates
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
]
EE_TPL = ["pEEx", "pEEy", "pEEz", "phiEEx", "phiEEy", "phiEEz"]
BASE_TPL = [
    "base_px",
    "base_py",
    "base_pz",
    "base_phix",
    "base_phiy",
    "base_phiz",
]


# Export public API
__all__ = [
    "get_joint_offset",
    "get_fullparam_offset",
    "add_base_name",
    "add_pee_name",
    "add_eemarker_frame",
    "FULL_PARAMTPL",
    "JOINT_OFFSETTPL",
    "ELAS_TPL",
    "EE_TPL",
    "BASE_TPL",
]


def get_joint_offset(model, joint_names):
    """Get dictionary of joint offset parameters.

    Maps joint names to their offset parameters, handling special cases for
    different joint types and multiple DOF joints.

    Args:
        model: Pinocchio robot model
        joint_names: List of joint names from model.names

    Returns:
        dict: Mapping of joint offset parameter names to initial zero values.
            Keys have format: "{offset_type}_{joint_name}"

    Example:
        >>> offsets = get_joint_offset(robot.model, robot.model.names[1:])
        >>> print(offsets["offsetRZ_joint1"])
        0.0
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


def get_fullparam_offset(joint_names):
    """Get dictionary of geometric parameter variations.

    Creates mapping of geometric offset parameters for each joint's
    position and orientation.

    Args:
        joint_names: List of joint names from robot model

    Returns:
        dict: Mapping of geometric parameter names to initial zero values.
            Keys have format: "d_{param}_{joint_name}" where param is:
            - px, py, pz: Position offsets
            - phix, phiy, phiz: Orientation offsets

    Example:
        >>> geo_params = get_fullparam_offset(robot.model.names[1:])
        >>> print(geo_params["d_px_joint1"])
        0.0
    """
    geo_params = []

    for i in range(len(joint_names)):
        for j in FULL_PARAMTPL:
            # geo_params.append(j + ("_%d" % i))
            geo_params.append(j + "_" + joint_names[i])

    phi_gp = [0] * len(geo_params)  # default zero values
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params


def add_base_name(calib_config):
    """Add base frame parameters to parameter list.

    Updates calib_config["param_name"] with base frame parameters depending on
    calibration model type.

    Args:
        calib_config: Parameter dictionary containing:
            - calib_model: "full_params" or "joint_offset"
            - param_name: List of parameter names to update

    Side Effects:
        Modifies calib_config["param_name"] in place by:
        - For full_params: Replaces first 6 entries with base parameters
        - For joint_offset: Prepends base parameters to list
    """
    if calib_config["calib_model"] == "full_params":
        calib_config["param_name"][0:6] = BASE_TPL
    elif calib_config["calib_model"] == "joint_offset":
        calib_config["param_name"] = BASE_TPL + calib_config["param_name"]


def add_pee_name(calib_config):
    """Add end-effector marker parameters to parameter list.

    Adds parameters for each active measurement DOF of each marker.

    Args:
        calib_config: Parameter dictionary containing:
            - NbMarkers: Number of markers
            - measurability: List of booleans for active DOFs
            - param_name: List of parameter names to update

    Side Effects:
        Modifies calib_config["param_name"] in place by appending marker
        parameters in format: "{param_type}_{marker_num}"
    """
    PEE_names = []
    for i in range(calib_config["NbMarkers"]):
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                PEE_names.extend(["{}_{}".format(EE_TPL[j], i + 1)])
    calib_config["param_name"] = calib_config["param_name"] + PEE_names


def add_eemarker_frame(frame_name, p, rpy, model, data):
    """Add a new frame attached to the end-effector.

    Creates and adds a fixed frame to the robot model at the end-effector
    location, typically used for marker or tool frames.

    Args:
        frame_name (str): Name for the new frame
        p (ndarray): 3D position offset from parent frame
        rpy (ndarray): Roll-pitch-yaw angles for frame orientation
        model (pin.Model): Robot model to add frame to
        data (pin.Data): Robot data structure

    Returns:
        int: ID of newly created frame

    Note:
        Currently hardcoded to attach to "arm_7_joint". This should be made
        configurable in future versions.
    """
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
