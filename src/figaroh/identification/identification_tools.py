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

import pinocchio as pin
import numpy as np
from scipy import signal
import operator

# Import configuration parsing functions
from .config import (  # noqa: F401
    get_param_from_yaml,
    get_param_from_yaml_legacy,
    unified_to_legacy_identif_config,
)

# Import parameter management functions
from .parameter import (  # noqa: F401
    reorder_inertial_parameters,
    add_standard_additional_parameters,
    add_custom_parameters,
    get_standard_parameters,
    get_parameter_info,
)


def base_param_from_standard(phi_standard, params_base):
    """Convert standard parameters to base parameters.

    Takes standard dynamic parameters and calculates the corresponding base
    parameters using analytical relationships between them.

    Args:
        phi_standard (dict): Standard parameters from model/URDF
        params_base (list): Analytical parameter relationships

    Returns:
        list: Base parameter values calculated from standard parameters
    """
    phi_base = []
    ops = {"+": operator.add, "-": operator.sub}
    for ii in range(len(params_base)):
        param_base_i = params_base[ii].split(" ")
        values = []
        list_ops = []
        for jj in range(len(param_base_i)):
            param_base_j = param_base_i[jj].split("*")
            if len(param_base_j) == 2:
                value = float(param_base_j[0]) * phi_standard[param_base_j[1]]
                values.append(value)
            elif param_base_j[0] != "+" and param_base_j[0] != "-":
                value = phi_standard[param_base_j[0]]
                values.append(value)
            else:
                list_ops.append(ops[param_base_j[0]])
        value_phi_base = values[0]
        for kk in range(len(list_ops)):
            value_phi_base = list_ops[kk](value_phi_base, values[kk + 1])
        phi_base.append(value_phi_base)
    return phi_base


def relative_stdev(W_b, phi_b, tau):
    """Calculate relative standard deviation of identified parameters.

    Implements the residual error method from [Pressé & Gautier 1991] to
    estimate parameter uncertainty.

    Args:
        W_b (ndarray): Base regressor matrix
        phi_b (list): Base parameter values
        tau (ndarray): Measured joint torques/forces

    Returns:
        ndarray: Relative standard deviation (%) for each base parameter
    """
    # stdev of residual error ro
    sig_ro_sqr = np.linalg.norm((tau - np.dot(W_b, phi_b))) ** 2 / (
        W_b.shape[0] - phi_b.shape[0]
    )

    # covariance matrix of estimated parameters
    C_x = sig_ro_sqr * np.linalg.inv(np.dot(W_b.T, W_b))

    # relative stdev of estimated parameters
    std_x_sqr = np.diag(C_x)
    std_xr = np.zeros(std_x_sqr.shape[0])
    for i in range(std_x_sqr.shape[0]):
        std_xr[i] = np.round(100 * np.sqrt(std_x_sqr[i]) / np.abs(phi_b[i]), 2)

    return std_xr


def index_in_base_params(params, id_segments):
    """Map segment IDs to their base parameters.

    For each segment ID, finds which base parameters contain inertial
    parameters from that segment.

    Args:
        params (list): Base parameter expressions
        id_segments (list): Segment IDs to map

    Returns:
        dict: Maps segment IDs to lists of base parameter indices
    """
    base_index = []
    params_name = [
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
        "mx",
        "my",
        "mz",
        "m",
    ]

    id_segments_new = [i for i in range(len(id_segments))]

    for id in id_segments:
        for ii in range(len(params)):
            param_base_i = params[ii].split(" ")
            for jj in range(len(param_base_i)):
                param_base_j = param_base_i[jj].split("*")
                for ll in range(len(param_base_j)):
                    for kk in params_name:
                        if kk + str(id) == param_base_j[ll]:
                            base_index.append((id, ii))

    base_index[:] = list(set(base_index))
    base_index = sorted(base_index)

    dictio = {}

    for i in base_index:
        dictio.setdefault(i[0], []).append(i[1])

    values = []
    for ii in dictio:
        values.append(dictio[ii])

    return dict(zip(id_segments_new, values))


def weigthed_least_squares(robot, phi_b, W_b, tau_meas, tau_est, identif_config):
    """Compute weighted least squares solution for parameter identification.

    Implements iteratively reweighted least squares method from
    [Gautier, 1997]. Accounts for heteroscedastic noise.

    Args:
        robot (pin.Robot): Robot model
        phi_b (ndarray): Initial base parameters
        W_b (ndarray): Base regressor matrix
        tau_meas (ndarray): Measured joint torques
        tau_est (ndarray): Estimated joint torques
        param (dict): Settings including idx_tau_stop

    Returns:
        ndarray: Identified base parameters
    """
    sigma = np.zeros(robot.model.nq)  # For ground reaction force model
    P = np.zeros((len(tau_meas), len(tau_meas)))
    nb_samples = int(identif_config["idx_tau_stop"][0])
    start_idx = int(0)
    for ii in range(robot.model.nq):
        tau_slice = slice(int(start_idx), int(identif_config["idx_tau_stop"][ii]))
        diff = tau_meas[tau_slice] - tau_est[tau_slice]
        denom = len(tau_meas[tau_slice]) - len(phi_b)
        sigma[ii] = np.linalg.norm(diff) / denom

        start_idx = identif_config["idx_tau_stop"][ii]

        for jj in range(nb_samples):
            idx = jj + ii * nb_samples
            P[idx, idx] = 1 / sigma[ii]

        phi_b = np.matmul(np.linalg.pinv(np.matmul(P, W_b)), np.matmul(P, tau_meas))

    phi_b = np.around(phi_b, 6)

    return phi_b


def calculate_first_second_order_differentiation(
    model, q, identif_config, dt=None, backend=None
):
    """Calculate joint velocities and accelerations from positions.

    Computes first and second order derivatives of joint positions using central
    differences. Handles both constant and variable timesteps.

    Args:
        model (pin.Model): Robot model (used when backend is None)
        q (ndarray): Joint position matrix (n_samples, n_joints)
        param (dict): Parameters containing:
            - is_joint_torques: Whether using joint torques
            - is_external_wrench: Whether using external wrench
            - ts: Timestep if constant
        dt (ndarray, optional): Variable timesteps between samples.
        backend (DynamicsBackend, optional): If provided, uses backend.compute_difference
            instead of pin.difference for Lie group operations.

    Returns:
        tuple:
            - q (ndarray): Trimmed position matrix
            - dq (ndarray): Joint velocity matrix
            - ddq (ndarray): Joint acceleration matrix

    Note:
        Two samples are removed from start/end due to central differences
    """
    nq = backend.nq if backend is not None else model.nq

    if identif_config["is_joint_torques"]:
        dq = np.zeros([q.shape[0] - 1, q.shape[1]])
        ddq = np.zeros([q.shape[0] - 1, q.shape[1]])

    if identif_config["is_external_wrench"]:
        dq = np.zeros([q.shape[0] - 1, q.shape[1] - 1])
        ddq = np.zeros([q.shape[0] - 1, q.shape[1] - 1])

    if dt is None:
        dt = identif_config["ts"]
        for ii in range(q.shape[0] - 1):
            if backend is not None:
                dq[ii, :] = backend.compute_difference(q[ii, :], q[ii + 1, :]) / dt
            else:
                dq[ii, :] = pin.difference(model, q[ii, :], q[ii + 1, :]) / dt

        for jj in range(nq - 1):
            ddq[:, jj] = np.gradient(dq[:, jj], edge_order=1) / dt
    else:
        for ii in range(q.shape[0] - 1):
            if backend is not None:
                dq[ii, :] = backend.compute_difference(q[ii, :], q[ii + 1, :]) / dt[ii]
            else:
                dq[ii, :] = pin.difference(model, q[ii, :], q[ii + 1, :]) / dt[ii]

        for jj in range(nq - 1):
            ddq[:, jj] = np.gradient(dq[:, jj], edge_order=1) / dt

    q = np.delete(q, len(q) - 1, 0)
    q = np.delete(q, len(q) - 1, 0)

    dq = np.delete(dq, len(dq) - 1, 0)
    ddq = np.delete(ddq, len(ddq) - 1, 0)

    return q, dq, ddq


def low_pass_filter_data(data, identif_config, nbutter=5):
    """Apply zero-phase Butterworth low-pass filter to measurement data.

    Uses scipy's filtfilt for zero-phase digital filtering. Removes high
    frequency noise while preserving signal phase. Handles border effects by
    trimming filtered data.

    Args:
        data (ndarray): Raw measurement data to filter
        param (dict): Filter parameters containing:
            - ts: Sample time
            - cut_off_frequency_butterworth: Cutoff frequency in Hz
        nbutter (int, optional): Filter order. Higher order gives sharper
            frequency cutoff. Defaults to 5.

    Returns:
        ndarray: Filtered data with border regions removed

    Note:
        Border effects are handled by removing nborder = 5*nbutter samples
        from start and end of filtered signal.
    """
    cutoff = identif_config["ts"] * identif_config["cut_off_frequency_butterworth"] / 2
    b, a = signal.butter(nbutter, cutoff, "low")

    padlen = 3 * (max(len(b), len(a)) - 1)
    data = signal.filtfilt(b, a, data, axis=0, padtype="odd", padlen=padlen)

    # Remove border effects
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    end_slice = slice(data.shape[0] - nbord, data.shape[0])
    data = np.delete(data, end_slice, axis=0)

    return data
