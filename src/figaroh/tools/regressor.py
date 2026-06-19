"""Regressor matrix computation utilities for robot dynamic identification."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pinocchio as pin
from dataclasses import dataclass


@dataclass
class RegressorConfig:
    """Configuration for regressor computation."""

    has_friction: bool = False
    has_actuator_inertia: bool = False
    has_joint_offset: bool = False
    is_joint_torques: bool = True
    is_external_wrench: bool = False
    force_torque: Optional[List[str]] = None
    additional_columns: int = 0


class RegressorBuilder:
    """Enhanced regressor builder with better organization."""

    def __init__(self, robot, config: Optional[RegressorConfig] = None):
        self.robot = robot
        self.config = config or RegressorConfig()
        # Use backend if available, fall back to direct model access
        self._has_backend = hasattr(robot, "backend")
        self.nv = robot.backend.nv if self._has_backend else robot.model.nv
        self.nonzero_inertias = self._get_nonzero_inertias()

    def build_basic_regressor(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray, identif_config=None
    ) -> np.ndarray:
        """Build basic regressor matrix."""
        # Normalize inputs
        Q, V, A, N = self._normalize_inputs(q, v, a)

        if self.config.is_joint_torques:
            return self._build_joint_torque_regressor(Q, V, A, N, identif_config)
        elif self.config.is_external_wrench:
            return self._build_external_wrench_regressor(Q, V, A, N, identif_config)
        else:
            raise ValueError(
                "Must specify either joint_torques or external_wrench mode"
            )

    def _normalize_inputs(
        self, q, v, a
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Normalize and validate inputs."""
        nq = self.robot.backend.nq if self._has_backend else self.robot.model.nq
        Q = self._ensure_2d(q, nq, "q")
        V = self._ensure_2d(v, self.nv, "v")
        A = self._ensure_2d(a, self.nv, "a")

        N = Q.shape[0]
        if V.shape[0] != N or A.shape[0] != N:
            raise ValueError(
                f"Inconsistent sample counts: q={N}, v={V.shape[0]}, a={A.shape[0]}"
            )

        return Q, V, A, N

    def _ensure_2d(self, x, expected_width: int, name: str) -> np.ndarray:
        """Ensure input is 2D array with correct width."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != expected_width:
            raise ValueError(
                f"{name} must have {expected_width} columns, got {x.shape[1]}"
            )
        return x

    def _get_nonzero_inertias(self) -> List[int]:
        """Get indices of bodies with non-zero mass."""
        if self._has_backend:
            inertias = self.robot.backend.get_inertias()
        else:
            inertias = self.robot.model.inertias.tolist()
        return [i for i, inertia in enumerate(inertias) if inertia.mass != 0]

    def _build_joint_torque_regressor(
        self, Q, V, A, N, identif_config=None
    ) -> np.ndarray:
        """Build regressor for joint torque identification."""
        W_ = np.zeros([N * self.nv, (10 + self.config.additional_columns) * self.nv])

        for i in range(N):
            if self._has_backend:
                W_temp = self.robot.backend.compute_regressor(Q[i], V[i], A[i])
                if W_temp.ndim == 1:
                    W_temp = W_temp.reshape(self.nv, -1)
            else:
                W_temp = pin.computeJointTorqueRegressor(
                    self.robot.model, self.robot.data, Q[i], V[i], A[i]
                )
            self._fill_joint_regressor_sample(W_, W_temp, V, A, i, N, identif_config)
        return W_
        # return self._reorder_parameters(W_, self.nv)

    def _build_external_wrench_regressor(
        self, Q, V, A, N, identif_config=None
    ) -> np.ndarray:
        """Build regressor for external wrench identification."""
        if self._has_backend:
            nb_bodies = len(self.robot.backend.get_inertias()) - 1
        else:
            nb_bodies = len(self.robot.model.inertias) - 1
        ft_components = self.config.force_torque or []

        W_ = np.zeros([N * 6, (10 + self.config.additional_columns) * nb_bodies])

        for i in range(N):
            if self._has_backend:
                W_temp = self.robot.backend.compute_regressor(Q[i], V[i], A[i])
                if W_temp.ndim == 1:
                    W_temp = W_temp.reshape(self.nv, -1)
            else:
                W_temp = pin.computeJointTorqueRegressor(
                    self.robot.model, self.robot.data, Q[i], V[i], A[i]
                )
            self._fill_wrench_regressor_sample(
                W_, W_temp, V, A, ft_components, i, N, nb_bodies, identif_config
            )
        return W_
        # return self._reorder_parameters(W, nb_bodies)

    def _fill_joint_regressor_sample(
        self, W, W_temp, V, A, sample_idx, N, identif_config=None
    ):
        """Fill regressor for one sample in joint torque mode."""
        for j in range(W_temp.shape[0]):
            base_idx = j * N + sample_idx
            W[base_idx, : 10 * self.nv] = W_temp[j, :]

            # Additional parameters
            param_start = 10 * self.nv

            if j in identif_config["act_idxv"]:
                if self.config.has_friction:
                    W[base_idx, param_start + j] = V[sample_idx, j]  # fv
                    W[base_idx, param_start + self.nv + j] = np.sign(
                        V[sample_idx, j]
                    )  # fs

                if self.config.has_actuator_inertia:
                    W[base_idx, param_start + 2 * self.nv + j] = A[sample_idx, j]  # ia

                if self.config.has_joint_offset:
                    W[base_idx, param_start + 2 * self.nv + self.nv + j] = 1.0  # offset

    def _fill_wrench_regressor_sample(
        self,
        W,
        W_temp,
        V,
        A,
        ft_components,
        sample_idx,
        N,
        nb_bodies,
        identif_config=None,
    ):
        """Fill regressor for one sample in external wrench mode."""
        for k, ft in enumerate(ft_components):
            j = "Fx Fy Fz Mx My Mz".split().index(ft)
            base_idx = j * N + sample_idx
            W[base_idx, : 10 * nb_bodies] = W_temp[j, : 10 * nb_bodies]

        for j in range(nb_bodies):
            base_idx = j * 6 + sample_idx

            if identif_config and j in identif_config["act_idxv"]:
                if self.config.has_friction:
                    W[base_idx, 10 * nb_bodies + j] = V[sample_idx, j]  # fv
                    W[base_idx, 10 * nb_bodies + nb_bodies + j] = np.sign(
                        V[sample_idx, j]
                    )  # fs

                if self.config.has_actuator_inertia:
                    W[base_idx, 10 * nb_bodies + 2 * nb_bodies + j] = A[
                        sample_idx, j
                    ]  # ia

                if self.config.has_joint_offset:
                    W[base_idx, 10 * nb_bodies + 2 * nb_bodies + nb_bodies + j] = (
                        1  # offset
                    )

    def _reorder_parameters(self, W: np.ndarray, num_params: int) -> np.ndarray:
        """Reorder parameters to standard format."""
        cols = 10 + self.config.additional_columns
        W_reordered = np.zeros([W.shape[0], cols * num_params])

        # Parameter order: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m, ia, fv, fs, offset]
        param_order = [4, 5, 7, 6, 8, 9, 1, 2, 3, 0]  # Pinocchio to standard order

        for k in range(num_params):
            base_out = k * cols
            base_in = k * 10

            # Reorder inertial parameters
            for i, old_idx in enumerate(param_order):
                W_reordered[:, base_out + i] = W[:, base_in + old_idx]

            # Add additional parameters
            if self.config.additional_columns > 0:
                param_start = 10 * num_params
                W_reordered[:, base_out + 10] = W[
                    :, param_start + 2 * num_params + k
                ]  # ia
                W_reordered[:, base_out + 11] = W[:, param_start + 2 * k]  # fv
                W_reordered[:, base_out + 12] = W[:, param_start + 2 * k + 1]  # fs
                W_reordered[:, base_out + 13] = W[
                    :, param_start + 2 * num_params + num_params + k
                ]  # offset

        return W_reordered


# Backward compatibility functions
def build_regressor_basic(robot, q, v, a, identif_config, tau=None):
    """Legacy function for backward compatibility."""
    # Calculate additional columns based on enabled options
    additional_columns = sum(
        [
            2 if identif_config.get("has_friction", False) else 0,  # fv and fs
            1 if identif_config.get("has_actuator_inertia", False) else 0,  # ia
            1 if identif_config.get("has_joint_offset", False) else 0,  # offset
        ]
    )

    config = RegressorConfig(
        has_friction=identif_config.get("has_friction", False),
        has_actuator_inertia=identif_config.get("has_actuator_inertia", False),
        has_joint_offset=identif_config.get("has_joint_offset", False),
        is_joint_torques=identif_config.get("is_joint_torques", True),
        is_external_wrench=identif_config.get("is_external_wrench", False),
        force_torque=identif_config.get("force_torque", None),
        additional_columns=additional_columns,
    )

    builder = RegressorBuilder(robot, config)
    return builder.build_basic_regressor(q, v, a, identif_config)


# Keep other functions with minor improvements...
def eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
    """Eliminate columns with small L2 norm."""
    col_norms = np.diag(W.T @ W)
    param_keys = list(params_std.keys())

    keep_indices = []
    keep_params = []

    for i, norm in enumerate(col_norms):
        if norm >= tol_e and i < len(param_keys):
            keep_indices.append(i)
            keep_params.append(param_keys[i])

    W_reduced = W[:, keep_indices]
    return W_reduced, keep_params


def get_index_eliminate(W, params_std, tol_e=1e-6):
    """Get indices of columns to eliminate based on tolerance.

    Args:
        W: Joint torque regressor matrix
        params_std: Standard parameters dictionary
        tol_e: Tolerance value

    Returns:
        tuple:
            - List of indices to eliminate
            - List of remaining parameters
    """
    col_norm = np.diag(np.dot(W.T, W))
    idx_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
        else:
            params_r.append(list(params_std.keys())[i])
    return idx_e, params_r


def build_regressor_reduced(W, idx_e):
    """Build reduced regressor matrix.

    Args:
        W: Input regressor matrix
        idx_e: Indices of columns to eliminate

    Returns:
        ndarray: Reduced regressor matrix
    """
    W_e = np.delete(W, idx_e, 1)
    return W_e


def build_total_regressor_current(
    W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, identif_config
):
    """Build regressor for total least squares with current measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        I_u: Joint currents in unloaded case
        I_l: Joint currents in loaded case
        param_standard_l: Standard parameters in loaded case
        identif_config: Dictionary of settings

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized identif_configeter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    nb_joints = int(len(I_u) / identif_config["nb_samples"])
    n_samples = identif_config["nb_samples"]

    V_a = np.concatenate(
        [
            I_u[:n_samples].reshape(n_samples, 1),
            np.zeros(((nb_joints - 1) * n_samples, 1)),
        ],
        axis=0,
    )

    V_b = np.concatenate(
        [
            I_l[:n_samples].reshape(n_samples, 1),
            np.zeros(((nb_joints - 1) * n_samples, 1)),
        ],
        axis=0,
    )

    for ii in range(1, nb_joints):
        V_a_ii = np.concatenate(
            [
                np.zeros((n_samples * ii, 1)),
                I_u[n_samples * ii : (ii + 1) * n_samples].reshape(n_samples, 1),
                np.zeros((n_samples * (nb_joints - ii - 1), 1)),
            ],
            axis=0,
        )
        V_b_ii = np.concatenate(
            [
                np.zeros((n_samples * ii, 1)),
                I_l[n_samples * ii : (ii + 1) * n_samples].reshape(n_samples, 1),
                np.zeros((n_samples * (nb_joints - ii - 1), 1)),
            ],
            axis=0,
        )
        V_a = np.concatenate((V_a, V_a_ii), axis=1)
        V_b = np.concatenate((V_b, V_b_ii), axis=1)

    W_current = np.concatenate((V_a, V_b), axis=0)
    W_tot = np.concatenate((W_tot, W_current), axis=1)

    if identif_config["has_friction"]:
        W_l_temp = np.zeros((len(W_l), 12))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
            W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 12 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (identif_config["which_body_loaded"]) * 12 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    elif identif_config["has_actuator_inertia"]:
        W_l_temp = np.zeros((len(W_l), 14))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]:
            W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 14 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (identif_config["which_body_loaded"]) * 14 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    else:
        W_l_temp = np.zeros((len(W_l), 9))
        for k in range(9):
            W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 10 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (identif_config["which_body_loaded"]) * 10 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = identif_config["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue


def build_total_regressor_wrench(
    W_b_u, W_b_l, W_l, tau_u, tau_l, param_standard_l, param
):
    """Build regressor for total least squares with external wrench measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        tau_u: External wrench in unloaded case
        tau_l: External wrench in loaded case
        param_standard_l: Standard parameters in loaded case
        param: Dictionary of settings

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized parameter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    tau_meast_ul = np.reshape(tau_u, (len(tau_u), 1))
    tau_meast_l = np.reshape(tau_l, (len(tau_l), 1))

    nb_samples_ul = int(len(tau_meast_ul) / 6)
    nb_samples_l = int(len(tau_meast_l) / 6)

    tau_ul = np.concatenate(
        [
            tau_meast_ul[:nb_samples_ul],
            np.zeros((len(tau_meast_ul) - nb_samples_ul, 1)),
        ],
        axis=0,
    )

    tau_l = np.concatenate(
        [tau_meast_l[:nb_samples_l], np.zeros((len(tau_meast_l) - nb_samples_l, 1))],
        axis=0,
    )

    for ii in range(1, 6):
        tau_ul_ii = np.concatenate(
            [
                np.concatenate(
                    [
                        np.zeros((nb_samples_ul * ii, 1)),
                        tau_meast_ul[nb_samples_ul * ii : (ii + 1) * nb_samples_ul],
                    ],
                    axis=0,
                ),
                np.zeros((nb_samples_ul * (5 - ii), 1)),
            ],
            axis=0,
        )

        tau_l_ii = np.concatenate(
            [
                np.concatenate(
                    [
                        np.zeros((nb_samples_l * ii, 1)),
                        tau_meast_l[nb_samples_l * ii : (ii + 1) * nb_samples_l],
                    ],
                    axis=0,
                ),
                np.zeros((nb_samples_l * (5 - ii), 1)),
            ],
            axis=0,
        )

        tau_ul = np.concatenate((tau_ul, tau_ul_ii), axis=1)
        tau_l = np.concatenate((tau_l, tau_l_ii), axis=1)

    W_tau = np.concatenate((tau_ul, tau_l), axis=0)
    W_tot = np.concatenate((W_tot, W_tau), axis=1)

    W_l_temp = np.zeros((len(W_l), 9))
    for k in range(9):
        W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 10 + k]
    W_upayload = np.concatenate(
        (np.zeros((len(W_l), W_l_temp.shape[1])), -W_l_temp), axis=0
    )
    W_tot = np.concatenate((W_tot, W_upayload), axis=1)

    W_kpayload = np.concatenate(
        [
            np.zeros((len(W_l), 1)),
            -W_l[:, identif_config["which_body_loaded"] * 10 + 9].reshape(len(W_l), 1),
        ],
        axis=0,
    )
    W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = identif_config["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue
