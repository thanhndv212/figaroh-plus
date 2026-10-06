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
Data loading and processing utilities for robot calibration.

This module provides functions for loading and processing calibration data
from various file formats, including:
- CSV file reading for joint configurations
- Marker position/orientation data loading
- Data validation and cleanup
- Configuration vector management
"""

import logging
import numpy as np
import pandas as pd

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Export public API
__all__ = [
    "read_config_data",
    "load_data",
    "get_idxq_from_jname",
]


def get_idxq_from_jname(model, joint_name):
    """Get index of joint in configuration vector.

    Args:
        model (pin.Model): Robot model
        joint_name (str): Name of joint to find index for

    Returns:
        int: Index of joint in configuration vector q

    Raises:
        AssertionError: If joint name does not exist in model
    """
    assert joint_name in model.names, "Given joint name does not exist."
    jointId = model.getJointId(joint_name)
    joint_idx = model.joints[jointId].idx_q
    return joint_idx


def read_config_data(model, path_to_file):
    """Read joint configurations from CSV file.

    Args:
        model (pin.Model): Robot model containing joint information
        path_to_file (str): Path to CSV file containing joint configurations

    Returns:
        ndarray: Matrix of shape (n_samples, n_joints-1) containing joint
            positions
    """
    df = pd.read_csv(path_to_file)
    q = np.zeros([len(df), model.njoints - 1])
    for i in range(len(df)):
        for j, name in enumerate(model.names[1:].tolist()):
            jointidx = get_idxq_from_jname(model, name)
            q[i, jointidx] = df[name][i]
    return q


def load_data(path_to_file, model, calib_config, del_list=[]):
    """Load joint configuration and marker data from CSV file.

    Reads marker positions/orientations and joint configurations from a CSV
    file. Handles data validation, bad sample removal, and conversion to
    numpy arrays.

    Args:
        path_to_file (str): Path to CSV file containing recorded data
        model (pin.Model): Robot model containing joint information
        calib_config (dict): Parameter dictionary containing:
            - NbMarkers: Number of markers to load
            - measurability: List indicating which DOFs are measured
            - actJoint_idx: List of active joint indices
            - config_idx: Configuration vector indices
            - q0: Default configuration vector
        del_list (list, optional): Indices of bad samples to remove.
            Defaults to [].

    Returns:
        tuple:
            - PEEm_exp (ndarray): Flattened marker measurements of shape
                (n_active_dofs,)
            - q_exp (ndarray): Joint configurations of shape
                (n_samples, n_joints)

    Note:
        CSV file must contain columns:
        - For each marker i: [xi, yi, zi, phixi, phiyi, phizi]
        - Joint names matching model.names for active joints

    Raises:
        KeyError: If required columns are missing from CSV

    Side Effects:
        - Prints joint headers
        - Updates calib_config["NbSample"] with number of valid samples
    """
    # read_csv
    df = pd.read_csv(path_to_file)

    # create headers for marker position
    PEE_headers = []
    pee_tpl = ["x", "y", "z", "phix", "phiy", "phiz"]
    for i in range(calib_config["NbMarkers"]):
        for j, state in enumerate(calib_config["measurability"]):
            if state:
                PEE_headers.extend(["{}{}".format(pee_tpl[j], i + 1)])

    # create headers for joint configurations
    joint_headers = [model.names[i] for i in calib_config["actJoint_idx"]]

    # check if all created headers present in csv file
    csv_headers = list(df.columns)
    for header in PEE_headers + joint_headers:
        if header not in csv_headers:
            logger.warning("%s does not exist in the file.", header)
            break

    # Extract marker position/location
    pose_ee = df[PEE_headers].to_numpy()

    # Extract joint configurations
    q_act = df[joint_headers].to_numpy()

    # remove bad data
    if del_list:
        pose_ee = np.delete(pose_ee, del_list, axis=0)
        q_act = np.delete(q_act, del_list, axis=0)

    # a missing (occluded) point would make every residual NaN: name the rows
    bad = ~np.isfinite(pose_ee).all(axis=1)
    if bad.any():
        rows = np.delete(np.arange(len(df)), del_list or [])[bad]
        raise ValueError(
            f"{path_to_file}: non-finite marker measurements in CSV rows "
            f"{rows.tolist()} (e.g. an occluded point); exclude them with "
            "del_list or clean the file"
        )

    # update number of data points
    calib_config["NbSample"] = q_act.shape[0]

    PEEm_exp = pose_ee.T
    PEEm_exp = PEEm_exp.flatten("C")

    q_exp = np.empty((calib_config["NbSample"], calib_config["q0"].shape[0]))
    for i in range(calib_config["NbSample"]):
        # a copy: q0 is robot.q0, which must not end up as the last sample (#125)
        config = calib_config["q0"].copy()
        config[calib_config["config_idx"]] = q_act[i, :]
        q_exp[i, :] = config

    return PEEm_exp, q_exp
