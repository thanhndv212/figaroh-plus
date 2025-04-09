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
from numpy.linalg import norm, solve
from scipy import linalg, signal

TOL_QR = 1e-8


def QR_pivoting(tau: np.ndarray, W_e: np.ndarray, params_r: list, tol_qr: float = TOL_QR) -> tuple:
    """Calculate QR decomposition with pivoting and find base parameters.
    
    Args:
        tau: Measurement vector
        W_e: Regressor matrix after eliminating zero columns
        params_r: List of parameters corresponding to W_e
        tol_qr: Tolerance for rank determination
        
    Returns:
        tuple: (W_b, base_parameters) containing:
            - W_b: Base regressor matrix
            - base_parameters: Dictionary mapping parameter names to values
    """
    Q, R, P = linalg.qr(W_e, pivoting=True)

    params_rsorted = []
    for i in range(P.shape[0]):
        params_rsorted.append(params_r[P[i]])

    # find rank of regressor
    numrank_W = 0
    for i in range(np.diag(R).shape[0]):
        if abs(np.diag(R)[i]) > tol_qr:
            continue
        else:
            numrank_W = i
            break

    R1 = R[0:numrank_W, 0:numrank_W]
    Q1 = Q[:, 0:numrank_W]
    R2 = R[0:numrank_W, numrank_W:R.shape[1]]

    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)
    phi_b = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)
    W_b = np.dot(Q1, R1)

    params_base = params_rsorted[:numrank_W]
    params_rgp = params_rsorted[numrank_W:]

    tol_beta = 1e-6
    for i in range(numrank_W):
        for j in range(beta.shape[1]):
            if abs(beta[i, j]) < tol_beta:
                params_base[i] = params_base[i]
            elif beta[i, j] < -tol_beta:
                params_base[i] = (
                    params_base[i]
                    + " - "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_rgp[j])
                )
            else:
                params_base[i] = (
                    params_base[i]
                    + " + "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_rgp[j])
                )
    base_parameters = dict(zip(params_base, phi_b))
    return W_b, base_parameters


def double_QR(tau: np.ndarray, W_e: np.ndarray, params_r: list, params_std: dict = None, tol_qr: float = TOL_QR) -> tuple:
    """Perform double QR decomposition to find base parameters.
    
    Args:
        tau: Measurement vector
        W_e: Regressor matrix after eliminating zero columns  
        params_r: List of parameters corresponding to W_e
        params_std: Standard parameters dictionary (optional)
        tol_qr: Tolerance for rank determination
        
    Returns:
        tuple: Contains combinations of:
            - W_b: Base regressor matrix
            - base_parameters: Dictionary of base parameters
            - params_base: List of base parameter names
            - phi_b: Base parameter values
            - phi_std: Standard parameter values (if params_std provided)
    """
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tol_qr:
            idx_base.append(i)
        else:
            idx_regroup.append(i)

    numrank_W = len(idx_base)

    W1 = np.zeros([W_e.shape[0], len(idx_base)])
    W2 = np.zeros([W_e.shape[0], len(idx_regroup)])

    params_base = []
    params_regroup = []

    for i in range(len(idx_base)):
        W1[:, i] = W_e[:, idx_base[i]]
        params_base.append(params_r[idx_base[i]])
    for j in range(len(idx_regroup)):
        W2[:, j] = W_e[:, idx_regroup[j]]
        params_regroup.append(params_r[idx_regroup[j]])

    W_regrouped = np.c_[W1, W2]

    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W:R.shape[1]]

    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)
    phi_b = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)
    W_b = np.dot(Q1, R1)
    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!"

    if params_std is not None:
        phi_std = []
        for x in params_base:
            phi_std.append(params_std[x])
        for i in range(numrank_W):
            for j in range(beta.shape[1]):
                phi_std[i] = phi_std[i] + beta[i, j] * params_std[params_regroup[j]]
        phi_std = np.around(phi_std, 5)

    tol_beta = 1e-6
    for i in range(numrank_W):
        for j in range(beta.shape[1]):
            if abs(beta[i, j]) < tol_beta:
                params_base[i] = params_base[i]
            elif beta[i, j] < -tol_beta:
                params_base[i] = (
                    params_base[i]
                    + " - "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )
            else:
                params_base[i] = (
                    params_base[i]
                    + " + "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )

    base_parameters = dict(zip(params_base, phi_b))

    if params_std is not None:
        return W_b, base_parameters, params_base, phi_b, phi_std
    else:
        return W_b, base_parameters, params_base, phi_b


def get_baseParams(W_e: np.ndarray, params_r: list, params_std: dict = None, tol_qr: float = TOL_QR) -> tuple:
    """Get base parameters and regressor matrix.
    
    Args:
        W_e: Regressor matrix
        params_r: List of parameters
        params_std: Standard parameters (optional)
        tol_qr: Tolerance for rank determination
        
    Returns:
        tuple: (W_b, params_base, idx_base) containing:
            - W_b: Base regressor matrix
            - params_base: List of base parameter expressions
            - idx_base: Indices of independent parameters
    """
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    for i in range(len(params_r)):
        if abs(np.diag(R))[i] > tol_qr:
            idx_base.append(i)
        else:
            idx_regroup.append(i)

    numrank_W = len(idx_base)

    W1 = np.zeros([W_e.shape[0], len(idx_base)])
    W2 = np.zeros([W_e.shape[0], len(idx_regroup)])

    params_base = []
    params_regroup = []

    for i in range(len(idx_base)):
        W1[:, i] = W_e[:, idx_base[i]]
        params_base.append(params_r[idx_base[i]])
    for j in range(len(idx_regroup)):
        W2[:, j] = W_e[:, idx_regroup[j]]
        params_regroup.append(params_r[idx_regroup[j]])

    W_regrouped = np.c_[W1, W2]

    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W:R.shape[1]]

    beta = np.around(np.matmul(np.linalg.inv(R1), R2), 6)

    tol_beta = 1e-6
    for i in range(numrank_W):
        for j in range(beta.shape[1]):
            if abs(beta[i, j]) < tol_beta:
                params_base[i] = params_base[i]
            elif beta[i, j] < -tol_beta:
                params_base[i] = (
                    params_base[i]
                    + " - "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )
            else:
                params_base[i] = (
                    params_base[i]
                    + " + "
                    + str(abs(beta[i, j]))
                    + "*"
                    + str(params_regroup[j])
                )

    W_b = np.dot(Q1, R1)
    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!"

    return W_b, params_base, idx_base


def get_baseIndex(W_e: np.ndarray, params_r: list, tol_qr: float = TOL_QR) -> tuple:
    """Find linearly independent parameters.
    
    Args:
        W_e: Regressor matrix
        params_r: Parameter dictionary 
        tol_qr: Tolerance for rank determination
        
    Returns:
        tuple: Indices of independent parameters
    """
    Q, R = np.linalg.qr(W_e)
    # print(np.diag(R))
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tol_qr:
            idx_base.append(i)
    idx_base = tuple(idx_base)
    return idx_base


def build_baseRegressor(W_e: np.ndarray, idx_base: tuple) -> np.ndarray:
    """Create base regressor matrix.
    
    Args:
        W_e: Original regressor matrix
        idx_base: Indices of base parameters
        
    Returns:
        ndarray: Base regressor matrix
    """
    W_b = np.zeros([W_e.shape[0], len(idx_base)])

    for i in range(len(idx_base)):
        W_b[:, i] = W_e[:, idx_base[i]]
    return W_b


def cond_num(W_b: np.ndarray, norm_type: str = None) -> float:
    """Calculate condition number of a matrix.
    
    Args:
        W_b: Input matrix
        norm_type: Type of norm to use ('fro' or 'max_over_min_sigma')
        
    Returns:
        float: Condition number
    """
    if norm_type == "fro":
        cond_num = np.linalg.cond(W_b, "fro")
    elif norm_type == "max_over_min_sigma":
        cond_num = np.linalg.cond(W_b, 2) / np.linalg.cond(W_b, -2)
    else:
        cond_num = np.linalg.cond(W_b)
    return cond_num


# def relative_stdev(W_b, phi_b, tau):
#     """ Calculates relative deviation of estimated parameters."""
#     # stdev of residual error ro
#     sig_ro_sqr = np.linalg.norm((tau - np.dot(W_b, phi_b))) ** 2 / (
#         W_b.shape[0] - phi_b.shape[0]
#     )

#     # covariance matrix of estimated parameters
#     C_x = sig_ro_sqr * np.linalg.inv(np.dot(W_b.T, W_b))

#     # relative stdev of estimated parameters
#     std_x_sqr = np.diag(C_x)
#     std_xr = np.zeros(std_x_sqr.shape[0])
#     for i in range(std_x_sqr.shape[0]):
#         std_xr[i] = np.round(100 * np.sqrt(std_x_sqr[i]) / np.abs(phi_b[i]), 2)

#     return std_xr
