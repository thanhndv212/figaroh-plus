import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
from scipy import linalg, signal

# epsilon = np.finfo(float).eps  # machine epsilon
# tolpal = W_e.shape[0]*abs(np.diag(R).max()) * \
#     epsilon  # rank revealing tolerance
tolpal = 1e-3


def QR_pivoting(tau, W_e, params_r):
    """This function calculates QR decompostion with pivoting, finds rank of regressor,
    and calculates base parameters
            Input:  W_e: regressor matrix (normally after eliminating zero columns)
                    params_r: a list of parameters corresponding to W_e
            Output: W_b: base regressor
                    base_parametes: a dictionary of base parameters"""

    # scipy has QR pivoting using Householder reflection
    Q, R, P = linalg.qr(W_e, pivoting=True)

    # sort params as decreasing order of diagonal of R
    params_rsorted = []
    for i in range(P.shape[0]):
        # print(i, ": ", params_r[P[i]], "---", abs(np.diag(R)[i]))
        params_rsorted.append(params_r[P[i]])

    # find rank of regressor
    numrank_W = 0

    for i in range(np.diag(R).shape[0]):
        if abs(np.diag(R)[i]) > tolpal:
            continue
        else:
            numrank_W = i
            break

    # regrouping, calculating base params, base regressor
    R1 = R[0:numrank_W, 0:numrank_W]
    Q1 = Q[:, 0:numrank_W]
    R2 = R[0:numrank_W, numrank_W: R.shape[1]]

    # regrouping coefficient
    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)

    # values of base params
    phi_b = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)

    # base regressor
    W_b = np.dot(Q1, R1)

    params_base = params_rsorted[:numrank_W]
    params_rgp = params_rsorted[numrank_W:]

    tol_beta = 1e-6  # for scipy.signal.decimate
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


def double_QR(tau, W_e, params_r, params_std=None):
    """This function calculates QR decompostion 2 times, first to find symbolic 
    expressions of base parameters, second to find their values after re-organizing 
    regressor matrix.
            Input:  W_e: regressor matrix (normally after eliminating zero columns)
                    params_r: a list of parameters corresponding to W_e
            Output: W_b: base regressor
                    base_parametes: a dictionary of base parameters"""
    # scipy has QR pivoting using Householder reflection
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    # find rank of regressor

    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tolpal:
            idx_base.append(i)
        else:
            idx_regroup.append(i)

    numrank_W = len(idx_base)

    # rebuild W and params after sorted
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

    # perform QR decomposition second time on regrouped regressor
    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W: R.shape[1]]

    # regrouping coefficient
    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)

    # values of base params
    phi_b = np.round(np.dot(np.linalg.inv(R1), np.dot(Q1.T, tau)), 6)

    # base regressor
    W_b = np.dot(Q1, R1)
    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!  "

    # reference values from std params
    if params_std is not None:
        phi_std = []
        for x in params_base:
            phi_std.append(params_std[x])
        for i in range(numrank_W):
            for j in range(beta.shape[1]):
                phi_std[i] = phi_std[i] + beta[i, j] * \
                    params_std[params_regroup[j]]
        phi_std = np.around(phi_std, 5)

    tol_beta = 1e-6  # for scipy.signal.decimate
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


def get_baseParams(W_e, params_r):
    """ Returns symbolic expressions of base parameters and base regressor matrix. """
    # scipy has QR pivoting using Householder reflection
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    # find rank of regressor
    # print("Diagonal values of matrix R: ")
    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tolpal:
            idx_base.append(i)
        else:
            idx_regroup.append(i)
        # print(params_r[i], np.diag(R)[i])
    numrank_W = len(idx_base)

    # rebuild W and params after sorted
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

    # perform QR decomposition second time on regrouped regressor
    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W: R.shape[1]]

    # regrouping coefficient
    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)

    tol_beta = 1e-6  # for scipy.signal.decimate
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

    # base regressor
    W_b = np.dot(Q1, R1)
    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!  "

    return W_b, params_base

def get_baseParams_v2(W_e, params_r, params_std):
    """ Returns symbolic expressions of base parameters and base regressor matrix and idenx of the base regressor matrix. """
    # scipy has QR pivoting using Householder reflection
    Q, R = np.linalg.qr(W_e)

    # sort params as decreasing order of diagonal of R
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    idx_regroup = []

    # find rank of regressor
    epsilon = np.finfo(float).eps  # machine epsilon
    tolpal = W_e.shape[0]*abs(np.diag(R).max()) * \
        epsilon  # rank revealing tolerance
    # tolpal = 0.02
    for i in range(len(params_r)):
        if abs(np.diag(R)[i]) > tolpal:
            idx_base.append(i)
        else:
            idx_regroup.append(i)

    numrank_W = len(idx_base)

    # rebuild W and params after sorted
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
    # return base param indices 
    idx_base = []
    params_names = list(params_std.keys())
    for i in params_base:
        if i in params_names:
            idx_base.append(params_names.index(i))
            
    W_regrouped = np.c_[W1, W2]

    # perform QR decomposition second time on regrouped regressor
    Q_r, R_r = np.linalg.qr(W_regrouped)

    R1 = R_r[0:numrank_W, 0:numrank_W]
    Q1 = Q_r[:, 0:numrank_W]
    R2 = R_r[0:numrank_W, numrank_W: R.shape[1]]

    # regrouping coefficient
    beta = np.around(np.dot(np.linalg.inv(R1), R2), 6)

    tol_beta = 1e-6  # for scipy.signal.decimate
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

    # base regressor
    W_b = np.dot(Q1, R1)
    assert np.allclose(W1, W_b), "base regressors is wrongly calculated!  "


    return W_b, params_base, idx_base


def get_baseIndex(W_e, params_r):
    """ This function finds the linearly independent parameters.
            Input:  W_e: regressor matrix
                    params_r: a dictionary of parameters
            Output: idx_base: a tuple of indices of only independent parameters.
    """
    Q, R = np.linalg.qr(W_e)
    # print(np.diag(R))
    assert np.diag(R).shape[0] == len(
        params_r
    ), "params_r does not have same length with R"

    idx_base = []
    epsilon = np.finfo(float).eps  # machine epsilon
    for i in range(len(params_r)):
        # print("R-value: ", i+1, params_r[i], abs(np.diag(R)[i]))
        if abs(np.diag(R)[i]) > tolpal:
            idx_base.append(i)
    idx_base = tuple(idx_base)
    return idx_base


def build_baseRegressor(W_e, idx_base):
    """ Create base regressor matrix corresponding to base parameters."""
    W_b = np.zeros([W_e.shape[0], len(idx_base)])

    for i in range(len(idx_base)):
        W_b[:, i] = W_e[:, idx_base[i]]
    return W_b


def cond_num(W_b, norm_type=None):
    """ Calculates different types of condition number of a matrix."""
    if norm_type == 'fro':
        cond_num = np.linalg.cond(W_b, 'fro')
    elif norm_type == 'max_over_min_sigma':
        cond_num = np.linalg.cond(W_b, 2)/np.linalg.cond(W_b, -2)
    else:
        cond_num = np.linalg.cond(W_b)
    return cond_num


def relative_stdev(W_b, phi_b, tau):
    """ Calculates relative deviation of estimated parameters."""
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
