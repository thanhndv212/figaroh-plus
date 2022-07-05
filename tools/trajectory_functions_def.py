# import time

import numpy as np
import ipopt
from scipy.optimize import approx_fprime

from tools.regressor import build_regressor_basic


def generate_trapezoidal_axis_motion(robot, param):
    """This function generatesjoints' position,velocity, acceleration using trapezoidal
    interpolation to excite motor inertia, and joint friction (Khalil & Dombre, ch.13
    2008).
    It generates a continuous trajectory axis per axis starting from the neutral robot's
    position.
        Input:
                model: pin model
                param: parameters
        Output: q, dq, ddq: joint's position, velocity, acceleration as time series"""
    model = robot.model
    for ii in range(model.nq):

        q_ii = model.lowerPositionLimit[ii] + param["q_safety"]
        dq_ii = 0
        ddq_ii = param["accelerationLimit"][ii]  # accelerationLimit[ii]

        for jj in range(param["nb_repet_trap"]):
            q_jj, dq_jj, ddq_jj = generate_trapezoidal_traj(
                model.lowerPositionLimit[ii] + param["q_safety"],
                model.upperPositionLimit[ii] - param["q_safety"],
                param["trapez_vel"][ii],
                param["accelerationLimit"][ii],
                model,
                param,
            )
            q_ii = np.append(q_ii, q_jj)
            dq_ii = np.append(dq_ii, dq_jj)
            ddq_ii = np.append(ddq_ii, ddq_jj)

            q_jj, dq_jj, ddq_jj = generate_trapezoidal_traj(
                model.upperPositionLimit[ii] - param["q_safety"],
                model.lowerPositionLimit[ii] + param["q_safety"],
                param["trapez_vel"][ii],
                param["accelerationLimit"][ii],
                model,
                param,
            )
            q_ii = np.append(q_ii, q_jj)
            dq_ii = np.append(dq_ii, dq_jj)
            ddq_ii = np.append(ddq_ii, ddq_jj)

        if ii == 0:

            q_temp_pol, dq_temp_pol, ddq_temp_pol = generate_quinticpoly_traj(
                np.array([param["q0"][0], 0, 0]),
                np.array([q_ii[0], dq_ii[0], ddq_ii[0]]),
                model.velocityLimit[ii],
                param["accelerationLimit"][ii],
                model,
                param,
            )

            q = np.zeros((q_temp_pol.shape[0], model.nq))
            q[:, ii] = q_temp_pol

            dq = np.zeros((q_temp_pol.shape[0], model.nq))
            dq[:, ii] = dq_temp_pol

            ddq = np.zeros((q_temp_pol.shape[0], model.nq))
            ddq[:, ii] = ddq_temp_pol

            q_temp = np.zeros((q_ii.shape[0], model.nq))
            q_temp[:, ii] = q_ii
            q = np.vstack((q, q_temp))

            dq_temp = np.zeros((q_ii.shape[0], model.nq))
            dq_temp[:, ii] = dq_ii
            dq = np.vstack((dq, dq_temp))

            ddq_temp = np.zeros((q_ii.shape[0], model.nq))
            ddq_temp[:, ii] = ddq_ii
            ddq = np.vstack((ddq, ddq_temp))

            q_temp_pol, dq_temp_pol, ddq_temp_pol = generate_quinticpoly_traj(
                np.array([q_ii[-1], dq_ii[-1], ddq_ii[-1]]),
                np.array([param["q0"][0], 0, 0]),
                model.velocityLimit[ii],
                param["accelerationLimit"][ii],
                model,
                param,
            )

            q_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            q_temp[:, ii] = q_temp_pol
            q = np.vstack((q, q_temp))

            dq_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            dq_temp[:, ii] = dq_temp_pol
            dq = np.vstack((dq, dq_temp))

            ddq_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            ddq_temp[:, ii] = ddq_temp_pol
            ddq = np.vstack((ddq, ddq_temp))

        else:

            q_temp_pol, dq_temp_pol, ddq_temp_pol = generate_quinticpoly_traj(
                np.array([param["q0"][0], 0, 0]),
                np.array([q_ii[0], dq_ii[0], ddq_ii[0]]),
                model.velocityLimit[ii],
                param["accelerationLimit"][ii],
                model,
                param,
            )

            q_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            q_temp[:, ii] = q_temp_pol
            q = np.vstack((q, q_temp))

            dq_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            dq_temp[:, ii] = dq_temp_pol
            dq = np.vstack((dq, dq_temp))

            ddq_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            ddq_temp[:, ii] = ddq_temp_pol
            ddq = np.vstack((ddq, ddq_temp))

            q_temp = np.zeros((q_ii.shape[0], model.nq))
            q_temp[:, ii] = q_ii
            q = np.vstack((q, q_temp))

            dq_temp = np.zeros((q_ii.shape[0], model.nq))
            dq_temp[:, ii] = dq_ii
            dq = np.vstack((dq, dq_temp))

            ddq_temp = np.zeros((q_ii.shape[0], model.nq))
            ddq_temp[:, ii] = ddq_ii
            ddq = np.vstack((ddq, ddq_temp))

            q_temp_pol, dq_temp_pol, ddq_temp_pol = generate_quinticpoly_traj(
                np.array([q_ii[-1], dq_ii[-1], ddq_ii[-1]]),
                np.array([param["q0"][0], 0, 0]),
                model.velocityLimit[ii],
                param["accelerationLimit"][ii],
                model,
                param,
            )

            q_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            q_temp[:, ii] = q_temp_pol
            q = np.vstack((q, q_temp))

            dq_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            dq_temp[:, ii] = dq_temp_pol
            dq = np.vstack((dq, dq_temp))

            ddq_temp = np.zeros((q_temp_pol.shape[0], model.nq))
            ddq_temp[:, ii] = ddq_temp_pol
            ddq = np.vstack((ddq, ddq_temp))

    return q, dq, ddq


def generate_trapezoidal_traj(Jc0, Jcf, vmax, amax, model, param):
    """This function generatesjoints' position,velocity, acceleration for ne joint using
    trapezoidal interpolation  (Khalil & Dombre, ch.13, pp. 322, 2008).
    Input: 	Jc0: joint configuration at time 0
            Jcf: joint configuration at final time
            vmax: maximal (desired) joint velocity
            amax: maximal (desired) joint acceleation
            model: pin model
            param: parameters (NbSample, sampling time)
    Output: q, dq, ddq: joint's position, velocity, acceleration as time series"""

    tau = vmax / amax
    D = Jcf - Jc0
    tf = tau + np.abs(D) / vmax

    q = np.zeros((int(tf / param["ts"]), 1))
    dq = np.zeros((int(tf / param["ts"]), 1))
    ddq = np.zeros((int(tf / param["ts"]), 1))

    if np.abs(D) > ((vmax * vmax) / amax):

        t = 0
        for i in range(int(tf / param["ts"])):

            if t <= tau:
                q[i] = Jc0 + 0.5 * t * t * amax * np.sign(D)
                dq[i] = t * amax * np.sign(D)
                ddq[i] = amax * np.sign(D)

                # print("phase 1")
            elif t >= tau and t <= (tf - tau):
                q[i] = Jc0 + (t - tau / 2) * vmax * np.sign(D)
                dq[i] = vmax * np.sign(D)
                ddq[i] = 0
            elif t >= (tf - tau):
                q[i] = Jcf - 1 / 2 * (tf - t) * (tf - t) * amax * np.sign(D)
                dq[i] = -(t - tf) * amax * np.sign(D)
                ddq[i] = -amax * np.sign(D)

            t = t + param["ts"]
            D = Jcf - q[i]
    else:
        print("Motion not feasible. please modify vmax or amax")
    return q, dq, ddq


def generate_quinticpoly_traj(Jc0, Jcf, vmax, amax, model, param):
    """This function generatesjoints' position,velocity, acceleration using qunitic
    polynomial curve (Khalil & Dombre, ch.13, pp. 315, 2008).
    Input: 	Jc0: joint configuration at time 0
            Jcf: joint configuration at final time
            model: pin model
            param: parameters (NbSample, sampling time)
    Output: q, dq, ddq: joint's position, velocity, acceleration as time series"""

    if param["sync_joint_motion"]:  # synchronise on the slowest joint
        nq = model.nq

        for ii in range(model.nq):

            tf = 15 * np.abs(Jcf[0] - Jc0[0]) / (8 * vmax[ii])

        tf = np.max(tf)
        # print("time for interpolation",tf)
        nb_sample_interpolate = int(tf / param["ts"]) + 1
        q = np.zeros((nb_sample_interpolate, nq))
        dq = np.zeros((nb_sample_interpolate, nq))
        ddq = np.zeros((nb_sample_interpolate, nq))

        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )

        for ii in range(nq):

            a = np.matmul(np.linalg.inv(A), np.hstack((Jc0[:, ii], Jcf[:, ii])))

            t = 0
            for j in range(nb_sample_interpolate):

                q[j, ii] = (
                    a[0]
                    + a[1] * t
                    + a[2] * t**2
                    + a[3] * t**3
                    + a[4] * t**4
                    + a[5] * t**5
                )
                dq[j, ii] = (
                    a[1]
                    + 2 * a[2] * t
                    + 3 * a[3] * t**2
                    + 4 * a[4] * t**3
                    + 5 * a[5] * t**4
                )
                ddq[j, ii] = (
                    +2 * a[2] + 6 * a[3] * t + 12 * a[4] * t**2 + 20 * a[5] * t**3
                )
                t = t + param["ts"]

    else:
        nq = 1
        tf = 15 * np.abs(Jcf[0] - Jc0[0]) / (8 * vmax)
        tf = np.max(tf)

        nb_sample_interpolate = int(tf / param["ts"]) + 1
        q = np.zeros(nb_sample_interpolate)
        dq = np.zeros(nb_sample_interpolate)
        ddq = np.zeros(nb_sample_interpolate)

        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )

        a = np.matmul(np.linalg.inv(A), np.hstack((Jc0, Jcf)))

        t = 0
        for j in range(nb_sample_interpolate):

            q[j] = (
                a[0]
                + a[1] * t
                + a[2] * t**2
                + a[3] * t**3
                + a[4] * t**4
                + a[5] * t**5
            )
            dq[j] = (
                a[1]
                + 2 * a[2] * t
                + 3 * a[3] * t**2
                + 4 * a[4] * t**3
                + 5 * a[5] * t**4
            )
            ddq[j] = +2 * a[2] + 6 * a[3] * t + 12 * a[4] * t**2 + 20 * a[5] * t**3
            t = t + param["ts"]

    return q, dq, ddq


def generate_fourier_traj(coeff, model, param):
    """This function generates Nnsample joints' position,velocity, acceleration using
    Fourier series.
    Input: 	coeff: values of the Fourier series coefficients
            model: pinocchio model
            param: parameters (NbSample, sampling time)
    Output: q, dq, ddq: joint's position, velocity, acceleration as time series"""
    qi = np.empty((model.nq))
    dqi = np.empty((model.nq))
    ddqi = np.empty((model.nq))

    t = 0
    for i in range(param["nb_samples"]):
        for j in range(model.nq):
            qi[j] = coeff[0, j]
            dqi[j] = 0
            ddqi[j] = 0
            for nc in range(param["nb_harmonics"]):
                w = 2 * (nc + 1) * param["freq"] * np.pi
                qi[j] = (
                    qi[j]
                    + coeff[1 + nc * 2, j] * np.cos(w * t)
                    + coeff[2 + nc * 2, j] * np.sin(w * t)
                )
                dqi[j] = (
                    dqi[j]
                    - coeff[1 + nc * 2, j] * w * np.sin(w * t)
                    + coeff[2 + nc * 2, j] * w * np.cos(w * t)
                )
                ddqi[j] = (
                    ddqi[j]
                    - coeff[1 + nc * 2, j] * w * w * np.cos(w * t)
                    - coeff[2 + nc * 2, j] * w * w * np.sin(w * t)
                )

        t = round(t + param["ts"], 3)

        if i == 0:

            q = np.copy(qi)
            dq = np.copy(dqi)
            ddq = np.copy(ddqi)

        else:

            q = np.vstack((q, qi))
            dq = np.vstack((dq, dqi))
            ddq = np.vstack((ddq, ddqi))

    return q, dq, ddq


def generate_random_traj(model, param):
    """This function generates N random values for joints' position,velocity,
    acceleration.
    Input: 	N: number of samples
                    nq: length of q, nv : length of v
                    mlow and mhigh: the bound for random function
    Output: q, v, a: joint's position, velocity, acceleration"""
    q = np.empty((1, model.nq))
    v = np.empty((1, model.nv))
    a = np.empty((1, model.nv))

    for i in range(param["NbSample"]):
        q = np.vstack(
            (
                q,
                np.random.uniform(
                    low=param["mlow"], high=param["mhigh"], size=(model.nq,)
                ),
            )
        )
        v = np.vstack(
            (
                v,
                np.random.uniform(
                    low=param["mlow"], high=param["mhigh"], size=(model.nv,)
                ),
            )
        )
        a = np.vstack(
            (
                a,
                np.random.uniform(
                    low=param["mlow"], high=param["mhigh"], size=(model.nv,)
                ),
            )
        )
    return q, v, a


# %% OEM optimisation process
class generate_OEM_problem(object):
    def __init__(self, robot, param):

        self.param = param
        self.robot = robot
        self.model = robot.model
        self.ind_joint = 0

    def objective(self, x):
        # callback for objective

        if self.param["is_static_regressor"]:
            q = np.zeros((self.param["nb_postures"], self.model.nq))
            dq = np.zeros((self.param["nb_postures"], self.model.nq))
            ddq = np.zeros((self.param["nb_postures"], self.model.nq))

            kk = 0  # to be vectorized
            for ii in range(self.model.nq):
                for jj in range(self.param["nb_postures"]):
                    q[jj, ii] = x[kk]
                    kk = kk + 1

        else:
            coeff = np.reshape(x, (2 * self.param["nb_harmonics"] + 1, self.model.nq))
            q, dq, ddq = generate_fourier_traj(coeff, self.model, self.param)

        W = build_regressor_basic(self.robot, q, dq, ddq, self.param)
        W = W[:, self.param["idx_base"]]  # remove all non interesting columns

        # here will be conditions for different cost functions of the litterature

        # Normalize the columns of the regressor matrix [Khalil et al. 91]
        norm_col = np.zeros(len(self.param["idx_base"]))

        if self.param["iter"] == 1:
            W_norm = W
            for ii in range(len(self.param["idx_base"])):

                norm_col[ii] = np.linalg.norm(W[:, ii])
                W_norm[:, ii] = W[:, ii]  # /norm_col[ii]

        else:
            W_all = np.concatenate((self.param["W_prev"], W), axis=0)
            W_norm = W_all  # to get the right size
            for i in range(len(self.param["idx_base_param"])):
                norm_col[i] = np.linalg.norm(W_all)
                W_norm[i] = W_all[i]  # /norm_col[ii]

        J = np.linalg.cond(W_norm)

        return J

    # this needs to be improved
    def constraint_0(self, x):

        if not self.param["is_static_regressor"]:
            coeff = np.reshape(x, (2 * self.param["nb_harmonics"] + 1, self.model.nq))
            q, dq, ddq = generate_fourier_traj(coeff, self.model, self.param)
        else:
            q = np.zeros((self.param["nb_postures"], self.model.nq))
            q[:, 0] = x[0 : self.param["nb_postures"]]

        return np.array(np.amax(np.abs(q[:, 0])))

    def constraint_1(self, x):

        if not self.param["is_static_regressor"]:
            coeff = np.reshape(x, (2 * self.param["nb_harmonics"] + 1, self.model.nq))
            q, dq, ddq = generate_fourier_traj(coeff, self.model, self.param)
        else:
            q = np.zeros((self.param["nb_postures"], self.model.nq))
            q[:, 1] = x[self.param["nb_postures"] - 1 : -1]
        return np.array(np.amax(np.abs(q[:, 1])))

    def constraints(self, x):
        # callback for constraints

        cte0 = self.constraint_0(x)
        cte1 = self.constraint_1(x)

        return [cte0, cte1]

    def gradient(self, x):
        # callback for gradient

        G = approx_fprime(x, self.objective, self.param["eps_gradient"])

        return G

    def jacobian(self, x):
        # callback for jacobian of constraints
        jac = approx_fprime(x, self.constraint_0, self.param["eps_gradient"])
        jac = np.concatenate(
            (jac, approx_fprime(x, self.constraint_1, self.param["eps_gradient"])),
            axis=0,
        )

        return jac

    # def hessian(self, x, lagrange, obj_factor):
    #    return False  # we will use quasi-newton approaches to use hessian-info

    # progress callback
    """def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))"""


def generate_optimal_excitation_motion(robot, param):

    # ## Generation of Optimal Excitation Motions inspired by [Swevers, 1997] and
    # [Bonnet 2016]
    """This function generates continuous optimal exciting trajectories, i.e. joints'
    position,velocity, acceleration.
    Input: 	robot: robot's description including model and data
            param: all parmaeters required to run the OEM generation defined in
            parameters_setting.py
    Output: q, v, a: joint's position, velocity, acceleration"""

    model = robot.model
    nq = model.nq

    # constraint bounds for joint limitations and joint torques (not from urdf as data
    # are not available)

    vmax = (
        model.velocityLimit
    )  # in rad/s used to the polynomail inetrpolation between OEM or OEP

    if param["is_static_regressor"]:

        # Note I did not manage to remove these constriants from IPOPT SHOULD BE
        # INVESTIGATED
        # cl=cu=None
        cl = model.lowerPositionLimit  # lower joint limits
        cu = model.upperPositionLimit  # upper joint limits

        # To be tuned based on the famoeus rule of thumbs that we need 10 times more
        # measurement than parameters
        nb_postures = np.round(10 * len(param["idx_base_static"]) / nq)
        param["nb_postures"] = int(nb_postures)
        for ii in range(nq):
            if ii == 0:

                x0 = np.random.uniform(
                    low=model.lowerPositionLimit[ii],
                    high=model.upperPositionLimit[ii],
                    size=(int(nq * nb_postures)),
                )  # To be tuned
                lb = np.full(
                    int(nq * nb_postures), model.lowerPositionLimit[ii]
                )  # lower joint limits
                ub = np.full(
                    int(nq * nb_postures), model.upperPositionLimit[ii]
                )  # upper joint limits

            else:

                np.append(
                    x0,
                    np.random.uniform(
                        low=model.lowerPositionLimit[ii],
                        high=model.upperPositionLimit[ii],
                        size=(int(nq * nb_postures)),
                    ),
                )
                np.append(
                    lb, np.full(int(nq * nb_postures), model.lowerPositionLimit[ii])
                )
                np.append(
                    ub, np.full(int(nq * nb_postures), model.upperPositionLimit[ii])
                )
    else:
        lb = ub = None
        cl = model.lowerPositionLimit  # lower joint limits
        cu = model.upperPositionLimit  # upper joint limits

        if param["is_fourier_series"]:
            coeff = 0.01 * np.ones(
                (2 * param["nb_harmonics"] + 1, nq)
            )  # + np.random.uniform(
            # low=-0.01, high=0.01, size=((2*param['NbH']+1,model.nq)))
            coeff[0, 0] = np.pi / 2  # Joint offset
            coeff[0, 1] = np.pi / 2  # Joint offset
            x0 = np.reshape(coeff, ((2 * param["nb_harmonics"] + 1) * nq))

    jc0 = np.zeros((3, nq))  # initial joint configuration pos, vel, acc
    jc0[0, :] = param["q0"]  # set to model neutral jc

    q_all = np.empty((0, nq))
    dq_all = np.empty((0, nq))
    ddq_all = np.empty((0, nq))

    ts_traj = param["ts"]

    for iter in range(param["nb_iter_OEM"]):
        param["iter"] = iter + 1
        print("Generate motion phase ", param["iter"], "/", param["nb_iter_OEM"])
        # starttime = time.time()

        if not param["is_static_regressor"]:
            param["ts"] = 1 / 25  # down sample for optimization to speed up the process
            param["nb_samples"] = int(param["tf"] / param["ts"])
            coeff = 0.05 * np.ones(
                (2 * param["nb_harmonics"] + 1, nq)
            ) + np.random.uniform(
                low=-0.05, high=0.05, size=((2 * param["nb_harmonics"] + 1, nq))
            )
            x0 = np.reshape(coeff, ((2 * param["nb_harmonics"] + 1) * nq))

        nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=generate_OEM_problem(robot, param),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )
        # need to use limited-memory here as we did not give a valid hessian-callback
        # nlp.addOption(b'hessian_approximation', b'limited-memory')
        nlp.addOption("max_iter", 100)
        nlp.addOption("constr_viol_tol", 1e-4)
        nlp.addOption("tol", 1e-4)  # to be tuned
        nlp.addOption("print_level", 0)

        # MAX_ITER = 10
        # RUN_OPTIM = 1

        """while RUN_OPTIM==1:
            x_opt, info = nlp.solve(x0)
            RUN_OPTIM=0

            # check if the solution is feasible when reach max iter otherwise
            # re-optimise
            # from initial final with more iteration
            for i in range(len(info['g'])):

                if ((info['g'][i]>cu[i]) or (info['g'][i]<cl[i])):
                    print(" Non feasible solution ", info['g'])
                    MAX_ITER=MAX_ITER+30
                    nlp.addOption('max_iter', MAX_ITER)
                    x0=x_opt
                    RUN_OPTIM=1
        print(RUN_OPTIM)
                print("Cost function value:", info['obj_val'])
    """
        x_opt = x0

        if not param["is_static_regressor"]:
            param["ts"] = ts_traj  # resample at 1/Ts frequency
            param["nb_samples"] = int(param["tf"] / param["ts"])

            coeff = np.reshape(x_opt, (2 * param["nb_harmonics"] + 1, model.nq))
            q_opt, dq_opt, ddq_opt = generate_fourier_traj(coeff, model, param)
        else:
            q_opt = np.zeros((param["nb_postures"], model.nq))
            dq_opt = np.zeros((param["nb_postures"], model.nq))
            ddq_opt = np.zeros((param["nb_postures"], model.nq))

            kk = 0  # to be vectorized
            for ii in range(nq):
                for jj in range(int(nb_postures)):
                    q_opt[jj, ii] = x_opt[kk]
                    kk = kk + 1

        """fig, ax = plt.subplots(3)
        ax[0].plot(q_opt)
        ax[1].plot(dq_opt)
        ax[2].plot(ddq_opt)
        plt.show() """
        # print( np.array( np.amax(np.abs(q_opt[:,0])) ))
        # print( np.array( np.amax(np.abs(q_opt[:,1])) ))

        # generate a trajectory to link the OEM

        # plt.show()

        if param["is_static_regressor"]:

            for ii in range(int(nb_postures)):

                q, dq, ddq = generate_quinticpoly_traj(
                    jc0,
                    np.array([q_opt[ii, :], dq_opt[ii, :], ddq_opt[ii, :]]),
                    vmax,
                    [],
                    model,
                    param,
                )
                jc0 = np.array(
                    (q_opt[ii, :], dq_opt[ii, :], ddq_opt[ii, :])
                )  # save final config for next posture

                q_all = np.append(q_all, q[:, :], axis=0)

                q_all = np.append(
                    q_all,
                    np.full((int(param["time_static"] / param["ts"]), 2), q_opt[ii, :]),
                    axis=0,
                )
                dq_all = np.append(dq_all, dq[:, :], axis=0)
                dq_all = np.append(
                    dq_all,
                    np.full((int(param["time_static"] / param["ts"]), 2), [0, 0]),
                    axis=0,
                )

                ddq_all = np.append(ddq_all, ddq[:, :], axis=0)
                ddq_all = np.append(
                    ddq_all,
                    np.full((int(param["time_static"] / param["ts"]), 2), [0, 0]),
                    axis=0,
                )
        else:
            q, dq, ddq = generate_quinticpoly_traj(
                jc0,
                np.array([q_opt[0, :], dq_opt[0, :], ddq_opt[0, :]]),
                vmax,
                [],
                model,
                param,
            )
            jc0 = np.array(
                (q_opt[-1, :], dq_opt[-1, :], ddq_opt[-1, :])
            )  # save final config for next time

            """fig, ax = plt.subplots(3)
            ax[0].plot(q_opt)
            ax[1].plot(dq_opt)
            ax[2].plot(ddq_opt)"""

            q_all = np.append(q_all, q[:, :], axis=0)
            q_all = np.append(q_all, q_opt[1:-2, :], axis=0)

            dq_all = np.append(dq_all, dq[:, :], axis=0)
            dq_all = np.append(dq_all, dq_opt[1:-2, :], axis=0)

            ddq_all = np.append(ddq_all, ddq[:, :], axis=0)
            ddq_all = np.append(ddq_all, ddq_opt[1:-2, :], axis=0)

        W = build_regressor_basic(robot, q_opt, dq_opt, ddq_opt, param)
        W = W[:, param["idx_base"]]

        # norm_col = np.linalg.norm(W)

        if param["iter"] == 1:
            param["W_prev"] = W  # [:,param['idx_base']]
        else:
            param["W_prev"] = np.concatenate((param["W_prev"], W), axis=0)

        # print(param['W_prev'].shape)

    return q_all, dq_all, ddq_all
