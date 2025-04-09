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
import quadprog
import operator


def get_param_from_yaml(robot, identif_data):
    """Parse identification parameters from YAML configuration file.

    Extracts robot parameters, problem settings, signal processing options and total 
    least squares parameters from a YAML config file.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model
        identif_data (dict): YAML configuration containing:
            - robot_params: Joint limits, friction, inertia settings
            - problem_params: External wrench, friction, actuator settings
            - processing_params: Sample rate, filter settings
            - tls_params: Load mass and location

    Returns:
        dict: Parameter dictionary with unified settings

    Example:
        >>> config = yaml.safe_load(config_file)
        >>> params = get_param_from_yaml(robot, config)
        >>> print(params["nb_samples"])
    """

    # robot_name: anchor as a reference point for executing
    robot_name = robot.model.name

    robots_params = identif_data["robot_params"][0]
    problem_params = identif_data["problem_params"][0]
    process_params = identif_data["processing_params"][0]
    tls_params = identif_data["tls_params"][0]

    param = {
        "robot_name": robot_name,
        "nb_samples": int(1 / (process_params["ts"])),
        "q_lim_def": robots_params["q_lim_def"],
        "dq_lim_def": robots_params["dq_lim_def"],
        "is_external_wrench": problem_params["is_external_wrench"],
        "is_joint_torques": problem_params["is_joint_torques"],
        "force_torque": problem_params["force_torque"],
        "external_wrench_offsets": problem_params["external_wrench_offsets"],
        "has_friction": problem_params["has_friction"],
        "fv": robots_params["fv"],
        "fs": robots_params["fs"],
        "has_actuator_inertia": problem_params["has_actuator_inertia"],
        "Ia": robots_params["Ia"],
        "has_joint_offset": problem_params["has_joint_offset"],
        "off": robots_params["offset"],
        "has_coupled_wrist": problem_params["has_coupled_wrist"],
        "Iam6": robots_params["Iam6"],
        "fvm6": robots_params["fvm6"],
        "fsm6": robots_params["fsm6"],
        "N": robots_params["N"],
        "ratio_essential": robots_params["ratio_essential"],
        "cut_off_frequency_butterworth": process_params[
            "cut_off_frequency_butterworth"
        ],
        "ts": process_params["ts"],
        "mass_load": tls_params["mass_load"],
        "which_body_loaded": tls_params["which_body_loaded"],
    }
    return param


def set_missing_params_setting(robot, params_settings):
    """Set default values for missing robot parameters.
    
    Fills in missing parameters in the robot model with default values from 
    params_settings when URDF doesn't specify them.

    Args:
        robot (pin.RobotWrapper): Robot instance to update 
        params_settings (dict): Default parameter values containing:
            - q_lim_def: Default joint position limits
            - dq_lim_def: Default joint velocity limits
            - tau_lim_def: Default joint torque limits
            - ddq_lim_def: Default acceleration limits
            - fv, fs: Default friction parameters

    Returns:
        dict: Updated params_settings with all required parameters

    Side Effects:
        Modifies robot model in place:
        - Sets joint limits if undefined
        - Sets velocity limits if zero
        - Sets effort limits if zero
        - Adds acceleration limits
    """
    # Compare lower and upper position limits
    diff_limit = np.setdiff1d(
        robot.model.lowerPositionLimit, 
        robot.model.upperPositionLimit
    )
    
    # Set default joint limits if none defined
    if not diff_limit.any:
        print("No joint limits. Set default values")
        for ii in range(robot.model.nq):
            robot.model.lowerPositionLimit[ii] = -params_settings["q_lim_def"]
            robot.model.upperPositionLimit[ii] = params_settings["q_lim_def"]

    # Set default velocity limits if zero
    if np.sum(robot.model.velocityLimit) == 0:
        print("No velocity limit. Set default value") 
        for ii in range(robot.model.nq):
            robot.model.velocityLimit[ii] = params_settings["dq_lim_def"]

    # Set default torque limits if zero
    if np.sum(robot.model.velocityLimit) == 0:
        print("No joint torque limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.effortLimit[ii] = -params_settings["tau_lim_def"]

    # Set acceleration limits
    accelerationLimit = np.zeros(robot.model.nq)
    for ii in range(robot.model.nq):
        # accelerationLimit to be consistent with PIN naming
        accelerationLimit[ii] = params_settings["ddq_lim_def"]
    params_settings["accelerationLimit"] = accelerationLimit

    # Set friction parameters if needed
    if params_settings["has_friction"]:
        for ii in range(robot.model.nv):
            if ii == 0:
                # Default viscous friction values
                fv = [(ii + 1) / 10]
                # Default static friction values  
                fs = [(ii + 1) / 10]
            else:
                fv.append((ii + 1) / 10)
                fs.append((ii + 1) / 10)

        params_settings["fv"] = fv
        params_settings["fs"] = fs

    # Set external wrench offsets if needed
    if params_settings["external_wrench_offsets"]:
        # Set for a footprint of dim (1.8mx0.9m) at its center
        params_settings["OFFX"] = 900
        params_settings["OFFY"] = 450
        params_settings["OFFZ"] = 0

    return params_settings


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
    sig_ro_sqr = (np.linalg.norm((tau - np.dot(W_b, phi_b))) ** 2 / 
                 (W_b.shape[0] - phi_b.shape[0]))

    # covariance matrix of estimated parameters
    C_x = sig_ro_sqr * np.linalg.inv(np.dot(W_b.T, W_b))

    # relative stdev of estimated parameters
    std_x_sqr = np.diag(C_x)
    std_xr = np.zeros(std_x_sqr.shape[0])
    for i in range(std_x_sqr.shape[0]):
        std_xr[i] = np.round(
            100 * np.sqrt(std_x_sqr[i]) / np.abs(phi_b[i]), 
            2
        )

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


def weigthed_least_squares(robot, phi_b, W_b, tau_meas, tau_est, param):
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
    nb_samples = int(param["idx_tau_stop"][0])
    start_idx = int(0)
    for ii in range(robot.model.nq):
        tau_slice = slice(int(start_idx), int(param["idx_tau_stop"][ii]))
        diff = (tau_meas[tau_slice] - tau_est[tau_slice])
        denom = len(tau_meas[tau_slice]) - len(phi_b)
        sigma[ii] = np.linalg.norm(diff) / denom

        start_idx = param["idx_tau_stop"][ii]
        
        for jj in range(nb_samples):
            idx = jj + ii * nb_samples
            P[idx, idx] = 1 / sigma[ii]

        phi_b = np.matmul(
            np.linalg.pinv(np.matmul(P, W_b)), 
            np.matmul(P, tau_meas)
        )

    phi_b = np.around(phi_b, 6)

    return phi_b


def calculate_first_second_order_differentiation(model, q, param, dt=None):
    """Calculate joint velocities and accelerations from positions.

    Computes first and second order derivatives of joint positions using central 
    differences. Handles both constant and variable timesteps.

    Args:
        model (pin.Model): Robot model
        q (ndarray): Joint position matrix (n_samples, n_joints)
        param (dict): Parameters containing:
            - is_joint_torques: Whether using joint torques
            - is_external_wrench: Whether using external wrench
            - ts: Timestep if constant
        dt (ndarray, optional): Variable timesteps between samples.

    Returns:
        tuple:
            - q (ndarray): Trimmed position matrix
            - dq (ndarray): Joint velocity matrix  
            - ddq (ndarray): Joint acceleration matrix

    Note:
        Two samples are removed from start/end due to central differences
    """

    if param["is_joint_torques"]:
        dq = np.zeros([q.shape[0] - 1, q.shape[1]])
        ddq = np.zeros([q.shape[0] - 1, q.shape[1]])

    if param["is_external_wrench"]:
        dq = np.zeros([q.shape[0] - 1, q.shape[1] - 1])
        ddq = np.zeros([q.shape[0] - 1, q.shape[1] - 1])

    if dt is None:
        dt = param["ts"]
        for ii in range(q.shape[0] - 1):
            dq[ii, :] = pin.difference(model, q[ii, :], q[ii + 1, :]) / dt

        for jj in range(model.nq - 1):
            ddq[:, jj] = np.gradient(dq[:, jj], edge_order=1) / dt
    else:
        for ii in range(q.shape[0] - 1):
            dq[ii, :] = pin.difference(model, q[ii, :], q[ii + 1, :]) / dt[ii]

        for jj in range(model.nq - 1):
            ddq[:, jj] = np.gradient(dq[:, jj], edge_order=1) / dt

    q = np.delete(q, len(q) - 1, 0)
    q = np.delete(q, len(q) - 1, 0)

    dq = np.delete(dq, len(dq) - 1, 0)
    ddq = np.delete(ddq, len(ddq) - 1, 0)

    return q, dq, ddq


def low_pass_filter_data(data, param, nbutter=5):
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
    cutoff = param["ts"] * param["cut_off_frequency_butterworth"] / 2
    b, a = signal.butter(nbutter, cutoff, "low")

    padlen = 3 * (max(len(b), len(a)) - 1)
    data = signal.filtfilt(b, a, data, axis=0, padtype="odd", padlen=padlen)

    # Remove border effects
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    end_slice = slice(data.shape[0] - nbord, data.shape[0]) 
    data = np.delete(data, end_slice, axis=0)

    return data


# SIP QP OPTIMISATION


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """Solve a Quadratic Program defined as:
        
    minimize    1/2 x^T P x + q^T x
    subject to  G x <= h
                A x = b
                
    Args:
        P (ndarray): Symmetric quadratic cost matrix (n x n)
        q (ndarray): Linear cost vector (n)
        G (ndarray, optional): Linear inequality constraint matrix (m x n).
        h (ndarray, optional): Linear inequality constraint vector (m).
        A (ndarray, optional): Linear equality constraint matrix (p x n).
        b (ndarray, optional): Linear equality constraint vector (p).
    
    Returns:
        ndarray: Optimal solution vector x* (n)
        
    Note:
        Ensures P is symmetric positive definite by adding small identity matrix
    """
    qp_G = 0.5 * (P + P.T) + np.eye(P.shape[0]) * (
        1e-5
    )  # make sure P is symmetric, pos,def
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def calculate_standard_parameters(
    model, W, tau, COM_max, COM_min, params_standard_u, alpha
):
    """Calculate optimal standard inertial parameters via quadratic
    programming.
    
    Finds standard parameters that:
    1. Best fit measurement data (tau)
    2. Stay close to reference values from URDF
    3. Keep COM positions within physical bounds

    Args:
        model (pin.Model): Robot model containing inertias
        W (ndarray): Regressor matrix mapping parameters to measurements
        tau (ndarray): Measured force/torque data
        COM_max (ndarray): Upper bounds on center of mass positions
        COM_min (ndarray): Lower bounds on center of mass positions 
        params_standard_u (dict): Reference parameters from URDF
        alpha (float): Weight between fitting data (1) vs staying near refs (0)

    Returns:
        tuple:
            - phi_standard (ndarray): Optimized standard parameters
            - phi_ref (ndarray): Reference parameters from URDF
            
    Note:
        Constrains parameters to stay within [0.7, 1.3] times reference values
        except for COM positions which use explicit bounds.
    """
    np.set_printoptions(threshold=np.inf)
    phi_ref = []
    id_inertias = []

    for jj in range(len(model.inertias.tolist())):
        if model.inertias.tolist()[jj].mass != 0:
            id_inertias.append(jj)

    nreal = len(id_inertias)

    params_name = (
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
    )

    for k in range(nreal):
        for j in params_name:
            phi_ref_temp = params_standard_u[j + str(id_inertias[k])]
            phi_ref.append(phi_ref_temp)

    phi_ref = np.array(phi_ref)

    sf1 = 1 / (np.max(phi_ref) * len(phi_ref))
    sf2 = 1 / (np.max(tau) * len(tau))

    P = ((1 - alpha) * sf1 * np.eye(W.shape[1]) + 
         alpha * sf2 * np.matmul(W.T, W))
    r = (-((1 - alpha) * sf1 * phi_ref.T + 
           sf2 * alpha * np.matmul(tau.T, W)))

    # Setting constraints
    G = np.zeros(((14) * (nreal), 10 * nreal))
    h = np.zeros((((14) * (nreal), 1)))

    for ii in range(nreal):
        G[14 * ii][ii * 10 + 6] = 1  # mx<mx+
        h[14 * ii] = COM_max[3 * ii]
        G[14 * ii + 1][ii * 10 + 6] = -1  # mx>mx-
        h[14 * ii + 1] = -COM_min[3 * ii]
        G[14 * ii + 2][ii * 10 + 7] = 1  # my<my+
        h[14 * ii + 2] = COM_max[3 * ii + 1]
        G[14 * ii + 3][ii * 10 + 7] = -1  # my>my-
        h[14 * ii + 3] = -COM_min[3 * ii + 1]
        G[14 * ii + 4][ii * 10 + 8] = 1  # mz<mz+
        h[14 * ii + 4] = COM_max[3 * ii + 2]
        G[14 * ii + 5][ii * 10 + 8] = -1  # mz>mz-
        h[14 * ii + 5] = -COM_min[3 * ii + 2]
        G[14 * ii + 6][ii * 10 + 9] = 1  # m<m+
        h[14 * ii + 6] = 1.3 * phi_ref[ii * 10 + 9]
        G[14 * ii + 7][ii * 10 + 9] = -1  # m>m-
        h[14 * ii + 7] = -0.7 * phi_ref[ii * 10 + 9]
        G[14 * ii + 8][ii * 10 + 0] = 1  # Ixx<Ixx+
        h[14 * ii + 8] = 1.3 * phi_ref[ii * 10 + 0]
        G[14 * ii + 9][ii * 10 + 0] = -1  # Ixx>Ixx-
        h[14 * ii + 9] = -0.7 * phi_ref[ii * 10 + 0]
        G[14 * ii + 10][ii * 10 + 3] = 1  # Iyy<Iyy+
        h[14 * ii + 10] = 1.3 * phi_ref[ii * 10 + 3]
        G[14 * ii + 11][ii * 10 + 3] = -1  # Iyy>Iyy-
        h[14 * ii + 11] = -0.7 * phi_ref[ii * 10 + 3]
        G[14 * ii + 12][ii * 10 + 5] = 1  # Izz<Izz+
        h[14 * ii + 12] = 1.3 * phi_ref[ii * 10 + 5]
        G[14 * ii + 13][ii * 10 + 5] = -1  # Izz>Izz-
        h[14 * ii + 13] = -0.7 * phi_ref[ii * 10 + 5]

    # print(G.shape,h.shape,A.shape,b.shape)

    # SOLVING
    phi_standard = quadprog_solve_qp(P, r, G, h.reshape((G.shape[0],)))

    return phi_standard, phi_ref
