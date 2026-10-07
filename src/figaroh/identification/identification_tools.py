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

    return phi_b  # full precision (#142)


def calculate_first_second_order_differentiation(
    model, q, identif_config, dt=None, backend=None
):
    """Estimate interval tangent velocities and their coordinate derivatives.

    Configuration differences use the model/backend Lie-group operation, so
    positions have width ``nq`` and velocities/accelerations have width ``nv``.
    Effort-selection flags do not determine these dimensions.

    Args:
        model (pin.Model): Robot model; unused when backend is supplied.
        q (ndarray): Finite configurations of shape (n_samples, nq), with at
            least three samples. Manifold configurations must be valid for
            the selected model (e.g. normalized quaternions).
        identif_config (dict): Contains positive finite ``ts`` when dt is None.
        dt (float or ndarray, optional): Positive finite sample interval, or
            one interval per consecutive configuration pair (n_samples - 1).
            Pass timestamp differences, not absolute timestamps.
        backend (DynamicsBackend, optional): Supplies nq, nv and
            compute_difference instead of the Pinocchio model.

    Returns:
        tuple: Positions (n_samples - 2, nq), velocities and accelerations
        (n_samples - 2, nv). All tangent coordinates are differentiated.

    Raises:
        ValueError: Invalid configurations, sample count or timesteps.

    Note:
        Preserves the historical alignment: q[:-2] and the first n_samples-2
        forward interval velocities. Velocities represent interval midpoints,
        not the returned position timestamps. Acceleration is the gradient of
        those tangent components at the midpoint times (one-sided at endpoints).
        No interpolation to position timestamps or moving-frame transport is
        performed. Both discarded position samples are at the end.
    """
    nq = backend.nq if backend is not None else model.nq
    nv = backend.nv if backend is not None else model.nv
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != nq:
        raise ValueError(f"q must have shape (n_samples, {nq})")
    if q.shape[0] < 3:
        raise ValueError("q must contain at least three samples")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must contain only finite configurations")

    intervals = np.asarray(identif_config["ts"] if dt is None else dt, dtype=float)
    constant_dt = intervals.ndim == 0
    if constant_dt:
        intervals = np.full(q.shape[0] - 1, float(intervals))
    if intervals.shape != (q.shape[0] - 1,):
        raise ValueError("dt must be scalar or have one timestep per sample pair")
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise ValueError("Every timestep must be finite and strictly positive")

    dq = np.empty((q.shape[0] - 1, nv))
    for ii, timestep in enumerate(intervals):
        if backend is not None:
            difference = backend.compute_difference(q[ii], q[ii + 1])
        else:
            difference = pin.difference(model, q[ii], q[ii + 1])
        dq[ii] = difference / timestep

    if constant_dt:
        ddq = np.gradient(dq, axis=0, edge_order=1) / intervals[0]
    else:
        midpoint_times = np.cumsum(intervals) - 0.5 * intervals
        if not np.all(np.isfinite(midpoint_times)) or np.any(
            np.diff(midpoint_times) <= 0
        ):
            raise ValueError("dt produces invalid interval midpoint times")
        ddq = np.gradient(dq, midpoint_times, axis=0, edge_order=1)

    return q[:-2].copy(), dq[:-1], ddq[:-1]


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
