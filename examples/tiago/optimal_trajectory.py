from datetime import datetime
import numpy as np
from numpy import pi
import cyipopt
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.function_base import gradient
from scipy.optimize.nonlin import Jacobian
import scipy.sparse as sparse
from scipy.optimize import approx_fprime
import os
from os.path import dirname, join, abspath
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import numdifftools as nd
import hppfcl
import pinocchio as pin
import csv
import yaml

import ndcurves
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import *
from figaroh.tools.qrdecomposition import *
from figaroh.tools.randomdata import *
from figaroh.tools.robotcollisions import *
from figaroh.meshcat_viewer_wrapper import MeshcatVisualizer
from simplified_colission_model import build_tiago_simplified
from cubic_spline import *
import time

# HELPER FUNCTIONS TO OBTAIN BASE REGRESSOR (BASE REG)


def get_idx_from_random(Ns, robot, q, v, a):
    """ Sole purpose is to get index of eliminate std param in W to
        produce W_e and then get index of independent params in W_e
        TODO: remove redundant computation
    """
    W = build_regressor_basic(Ns, robot, q, v, a)
    params_std = robot.get_standard_parameters()
    idx_e_, par_r_ = get_index_eliminate(W, params_std, tol_e=0.001)
    W_e_ = build_regressor_reduced(W, idx_e_)
    idx_base_ = get_baseIndex(W_e_, par_r_)
    return idx_e_, idx_base_


def build_W_b(Ns, robot, q, v, a, idx_e_, idx_base_, W_stack=None):
    """ Given index of eliminate std params and independent params,
        now build base regressor for given data
        TODO: put idx_e and idx_base into param dict
    """
    W = build_regressor_basic(Ns, robot, q, v, a)
    W_e_ = build_regressor_reduced(W, idx_e_)
    W_b_ = build_baseRegressor(W_e_, idx_base_)
    list_idx = []
    for k in range(len(idx_act_joints)):
        list_idx.extend(
            list(range(idx_act_joints[k]*Ns, (idx_act_joints[k]+1)*Ns)))
    # W_b = W_b_[list_idx, :]
    W_b = W_b_

    # stack the computed base reg below the previous base reg
    if isinstance(W_stack, np.ndarray):
        W_b = np.vstack((W_stack, W_b))
        print("regressor is stacked!")
    return W_b

# IPOPT PROBLEM FORMULATION FUNCTIONS


def objective_func(Ns, X, vel_wps, acc_wps, wp_init, W_stack_=None):
    """ This functions computes the condition number of correspondent
        base regressor from computed trajectory. The trajectory is a 
        cubic spline which is initiated by waypoints pos/vel/acc.
        Hence, search variables are set to waypoints pos/vel/acc.
        Input:  Ns: (int) number of sample points
                X: (list) search variables - waypoints pos
                wp_init: (ndarray) the starting waypoint
                vel_wps, acc_wps: (ndarray),(ndarray) vel and acc at 
                waypoints
                W_stack: (ndarray) previous base reg
        Output: condition number value of the stacked base reg
    """
    # add the start waypoint and re-arrange waypoints
    X = np.array(X)
    wps_X = np.reshape(X, (n_wps-1, len(active_joints)))
    wps = np.vstack((wp_init, wps_X))
    wps = wps.transpose()

    # create timestamps at waypoints (t_s: travel time bw 2 wps)
    tps = np.matrix(
        [t_s * i for i in range(n_wps)]).transpose()

    # create full profile
    t_f, p_f, v_f, a_f = CB_f.get_full_config(
        freq, tps, wps, vel_wps, acc_wps)

    # get stacked base reg
    W_b = build_W_b(Ns, robot, p_f, v_f, a_f, idx_e,
                    idx_b, W_stack=W_stack_)

    return np.linalg.cond(W_b)


def get_constraints_all_samples(Ns,  X, vel_wps, acc_wps,
                                wp_init, vel_wp_init, acc_wp_init):
    """ Concatenate constraints into one vector:
            - joint angle (pos) constraints at waypoints
            - velocity constraints on all sample points
            - effort (joint torque/force) constraints on all sample points
            - auto-collision pairs from simplified collision model

    """
    # add the start waypoint and re-arrange waypoints
    X = np.array(X)
    wps_X = np.reshape(X, (n_wps-1, len(active_joints)))
    wps = np.vstack((wp_init, wps_X))
    wps = wps.transpose()

    # create timestamps at waypoints (t_s: travel time bw 2 wps)
    tps = np.matrix(
        [t_s * i for i in range(n_wps)]).transpose()

    # create full profile
    t_f, p_f, v_f, a_f = CB_f.get_full_config(
        freq, tps, wps, vel_wps, acc_wps)

    # compute joint effort given full profile
    tau = get_torque_rand(p_f.shape[0], robot, p_f, v_f, a_f)
    # print(" samples:\n", t_f.shape, p_f.shape)

    ## equality constraints at starting points
    # q_init = p_f[0, idx_act_joints]
    # dq_init = v_f[0, idx_act_joints]
    # ddq_init = a_f[0, idx_act_joints]
    # init_constraints = np.concatenate((q_init, dq_init, ddq_init), axis=None)
    # print("init constraints shape: ", init_constraints.shape)

    # pos constraints at waypoints
    idx_waypoints = []
    time_points = np.array([[s * t_s] for s in range(1, n_wps)])
    for i in range(t_f.shape[0]):
        if t_f[i, 0] in time_points:
            idx_waypoints.append(i)
    q_constraints = p_f[idx_waypoints, :]
    q_constraints = q_constraints[:, idx_act_joints]

    # vel constraints at all samples
    v_constraints = v_f[:, idx_act_joints]

    # effort constraints at all samples
    tau_constraints = np.zeros((Ns, len(idx_act_joints)))
    for k in range(len(idx_act_joints)):
        tau_constraints[:, k] = tau[range(
            idx_act_joints[k]*Ns, (idx_act_joints[k]+1)*Ns)]

    # auto collision constraints for all pairs
    collision = CollisionWrapper(robot=robot, viz=None)
    dist_all = []
    for j in idx_waypoints:
        collision.computeCollisions(p_f[j, :])
        dist_all = np.append(dist_all, collision.getDistances())
    dist_all = np.asarray(dist_all)

    # combine altogether
    constr_vec = np.concatenate(
        (q_constraints, v_constraints, tau_constraints, dist_all), axis=None
    )
    return constr_vec


def get_bounds(n_wps):
    """ Set boundaries for search variables
    """
    lb = []
    ub = []
    for i in range(1, n_wps):
        lb.extend(lower_q_lim)
        ub.extend(upper_q_lim)
    return lb, ub


def get_constr_value(n_wps, Ns, wp_init, vel_wp_init, acc_wp_init):
    """ Set limit values for all constraints
    """
    cl = []
    cu = []

    # equality constraints value for starting waypoint
    # cl_init = []
    # cl_init.extend(wp_init)
    # cl_init.extend(vel_wp_init)
    # cl_init.extend(acc_wp_init)
    # cu_init = cl_init

    # inequality constraints values of pos
    cl_pos = []
    cu_pos = []
    for i in range(1, n_wps):
        cl_pos.extend(lower_q_lim)
        cu_pos.extend(upper_q_lim)
    print("cl_pos shape: ", len(cl_pos))

    # inequality constraints values of vel
    cl_vel = []
    cu_vel = []
    for j in range(Ns):  # Ns: number of total samples on traj
        cl_vel.extend(lower_dq_lim)
        cu_vel.extend(upper_dq_lim)
    print("cl_vel shape: ", len(cl_vel))

    # inequality constraints values of effort
    cl_eff = []
    cu_eff = []
    for j in range(Ns):  # Ns: number of total samples on traj
        cl_eff.extend(lower_eff_lim)
        cu_eff.extend(upper_eff_lim)
    print("cl_eff shape: ", len(cl_eff))

    # inequality constraints values of self collision

    n_cols = len(robot.geom_model.collisionPairs)
    cl_cols = [0.01] * n_cols * (n_wps - 1) # 1 cm margin 
    cu_cols = [2 * 1e19] * n_cols * (n_wps - 1) # no limit on max distance

    cl = cl_pos + cl_vel + cl_eff + cl_cols
    cu = cu_pos + cu_vel + cu_eff + cu_cols

    print("constraints shape: ", len(cl))
    return cl, cu


class Problem_cond_Wb:
    def __init__(self, Ns, vel_wps, acc_wps,
                 wp_init, vel_wp_init, acc_wp_init, W_stack, stop_flag):
        self.W_stack = W_stack
        self.wp_init = wp_init
        self.vel_wp_init = vel_wp_init
        self.acc_wp_init = acc_wp_init
        self.vel_wps = vel_wps
        self.acc_wps = acc_wps
        self.stop_flag = stop_flag

    def objective(self, X):
        return objective_func(
            Ns, X, self.vel_wps, self.acc_wps, self.wp_init, W_stack_=self.W_stack
        )

    def gradient(self, X):
        def obj_f(x): return self.objective(x)
        grad_obj = nd.Gradient(obj_f)(X)
        return grad_obj

    def constraints(self, X):
        constr_vec = get_constraints_all_samples(Ns,  X, self.vel_wps, self.acc_wps,
                                                 self.wp_init, self.vel_wp_init, self.acc_wp_init)
        return constr_vec

    def jacobian(self, X):
        def f(x): return self.constraints(x)
        jac = nd.Jacobian(f)(X)
        return jac

    def hessian(self, X, lagrange, obj_factor):
        return False

    def intermediate(
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
        ls_trials,
    ):

        iter_num.append(iter_count)
        list_obj_value.append(obj_value)
        if self.stop_flag:
            return False


def add_options_nlp(nlp):
    # add options
    nlp.add_option(b"hessian_approximation", b"limited-memory")
    nlp.add_option(b"tol", 1e-1)
    nlp.add_option(b"max_iter", 100)
    nlp.add_option("acceptable_tol", 1e-1)
    nlp.add_option(b"warm_start_init_point", b"yes")
    # nlp.add_option('acceptable_iter',1)
    nlp.add_option(b"acceptable_obj_change_tol", 1e-1)
    nlp.add_option(b"print_level", 5)
    # nlp.add_option(b"check_derivatives_for_naninf", b"yes")
    # nlp.add_option(b"evaluate_orig_obj_at_resto_trial", b"no")


start = time.time()

# 1/ Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago_no_hand.urdf",
    package_dirs = package_dirs,
    # isFext=True  # add free-flyer joint at base
)
active_joints = ["torso_lift_joint",
                 "arm_1_joint",
                 "arm_2_joint",
                 "arm_3_joint",
                 "arm_4_joint",
                 "arm_5_joint",
                 "arm_6_joint",
                 "arm_7_joint"]

# TODO: specify soft_lim and soft_lim_pool to individual joint
soft_lim = 0.05  # discount from max and min of limit for all samples

# soft_lim_pool = 0.1 # discount for pool
soft_lim_pool = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
                        


n_wps = 2  # number of waypoints for spline trajectory include the destination

# step 2: load simplified collsion model
build_tiago_simplified(robot)

# step 3: build regressor for tiago on random data
n_wps_r = 100
freq_r = 100
CB = CubicSpline(robot, n_wps_r, active_joints)
WP = WaypointsGeneration(robot, n_wps_r, active_joints)
WP.gen_rand_pool(soft_lim_pool)

# generate random waypoints
wps_r, vel_wps_r, acc_wps_r = WP.gen_rand_wp(soft_lim)

# generate timepoints
tps_r = np.matrix([0.5*i for i in range(n_wps_r)]).transpose()

# get full config traj
t_r, p_r, v_r, a_r = CB.get_full_config(
    freq_r, tps_r, wps_r, vel_wps_r, acc_wps_r)

# get index essential and base params columns: idx_e, idx_b
idx_e, idx_b = get_idx_from_random(p_r.shape[0], robot, p_r, v_r, a_r)
print("number of base params: ", len(idx_b))

# step 4: define boundaries of search vars, and constraints
idx_act_joints = CB.active_joints
upper_q_lim = CB.upper_q - soft_lim*abs(CB.upper_q-CB.lower_q)
lower_q_lim = CB.lower_q + soft_lim*abs(CB.upper_q-CB.lower_q)
upper_dq_lim = CB.upper_dq - soft_lim*abs(CB.upper_dq - CB.lower_dq)
lower_dq_lim = CB.lower_dq + soft_lim*abs(CB.upper_dq - CB.lower_dq)
upper_eff_lim = CB.upper_effort - soft_lim * \
    abs(CB.upper_effort - CB.lower_effort)
lower_eff_lim = CB.lower_effort + soft_lim * \
    abs(CB.upper_effort - CB.lower_effort)

# step 5: initialize 1st waypoint
wp_init = np.zeros(len(idx_act_joints))
vel_wp_init = np.zeros(len(idx_act_joints))
acc_wp_init = np.zeros(len(idx_act_joints))
W_stack = None

# step 6: path to save file
dt = datetime.now()
current_time = dt.strftime("%d_%b_%Y_%H%M") 
path_save_bp = join(
    dirname(dirname(str(abspath(__file__)))),
    f"tiago/data/tiago_stacking_{current_time}.csv")

# step 7: for loop in reps times

# stop condition for iterative stacking
stop_count = 0
tol_stop = 0.1
last_obj = 0
reps = 5  # stacks
freq = 10  # Hz
t_s = 10
stop_flag = False
fig_cond = plt.figure()
with open(path_save_bp, "w") as output_file:
    w = csv.writer(output_file)
    first_row = ["%d" % i for i in range(15+12+12)]
    first_row.insert(0, 't')
    w.writerow(first_row)
    pos_dict = []  # list(dist) to dump on yaml

    CB_f = CubicSpline(robot, n_wps, active_joints)
    WP_f = WaypointsGeneration(robot, n_wps, active_joints)
    for i in range(reps):
        list_obj_value = []
        iter_num = []
        is_constr_violated = True
        count = 0
        WP_f.gen_rand_pool(soft_lim_pool)
        # generate feasible initial guess
        while is_constr_violated:
            count += 1
            print("----------","run %s " % count, "----------")
            wps, vel_wps, acc_wps = WP_f.gen_rand_wp(
                soft_lim, wp_init, vel_wp_init, acc_wp_init)
            tps = np.matrix(
                [t_s * i for i in range(n_wps)]).transpose()
            t_f, p_f, v_f, a_f = CB_f.get_full_config(
                freq, tps, wps, vel_wps, acc_wps)
            # print(t_f.shape, p_f.shape, v_f.shape, a_f.shape)
            tau = get_torque_rand(p_f.shape[0], robot, p_f, v_f, a_f)
            # ATTENTION: joint torque specially arranged!
            tau = np.reshape(tau, (v_f.shape[1], v_f.shape[0])).transpose()
            is_constr_violated = CB_f.check_cfg_constraints(p_f, v_f, tau)

        # reshape wps to a vector of search variable
        X0 = wps[:, range(1, n_wps)]
        X0 = np.reshape(X0.transpose(),
                        ((len(active_joints)*(n_wps-1),))).tolist()
        print("shape of initial guess: ", len(X0))
        Ns = p_f.shape[0]
        lb, ub = get_bounds(n_wps)
        cl, cu = get_constr_value(n_wps, Ns, wp_init, vel_wp_init, acc_wp_init)

# + construct ipopt class nlp object:
#     /objective function: only take search variable as input
#     f(search varible)
#     = > generate cubic spline(search varible)
#     = > gen full config
#     = > build base regressor Wb
#     = > cond(Wb)
#     /gradient: only take search variable as, do gradient descent on objective function
#     /constraints: only take search variable as input
#     f(search varible)
#     = > generate cubic spline(search varible)
#     = > gen full config: array of q, v, a on all sample points
#     = > get joint torque: array of tau on all sample points
#     = > construct collision object -> compute distance on all pairs(simplified) at waypoints: array of collision pair dist
#     /jacobian: only take search variable as input
#     /hessian: only take search variable as input
#     /intermediate
        nlp = cyipopt.Problem(
            n=len(X0),
            m=len(cl),
            problem_obj=Problem_cond_Wb(Ns, vel_wps, acc_wps,
                                        wp_init, vel_wp_init, acc_wp_init, W_stack, stop_flag),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,)
        add_options_nlp(nlp)
        # result
        X_opt, infor = nlp.solve(X0)
        print(X_opt)
        print(infor["status"], infor["status_msg"])

        X_fin = np.array(X_opt)
        wps_X = np.reshape(X_fin, (n_wps-1, len(active_joints)))
        wps = np.vstack((wp_init, wps_X))
        wps = wps.transpose()
        tps = np.matrix(
            [t_s * i for i in range(n_wps)]).transpose()
        t_f, p_f, v_f, a_f = CB_f.get_full_config(
            freq, tps, wps, vel_wps, acc_wps)
        tau = get_torque_rand(p_f.shape[0], robot, p_f, v_f, a_f)
        tau = np.reshape(tau, (v_f.shape[1], v_f.shape[0])).transpose()
        is_constr_violated = CB_f.check_cfg_constraints(p_f, v_f, tau)
        if is_constr_violated:
            print("optimal solution is constraint-violated")

            # write to yaml file
        p_f = np.around(p_f, 4)
        v_f = np.around(v_f, 4)
        a_f = np.around(a_f, 4)
        t_f = t_f + i*t_s
        # for j in range(t_f.shape[0]):
        # CB.plot_spline(t_f, p_f, v_f, a_f)
        if i == 0:
            for j in range(p_f.shape[0]):
                a = p_f[j, :].tolist()
                a.insert(0, t_f[j, 0])
                b = v_f[j, :].tolist()
                a.extend(b)
                c = a_f[j, :].tolist()
                a.extend(c)
                w.writerow(a[k] for k in range(len(a)))
                pos_dict.append(
                    {'positions': p_f[j, idx_act_joints].tolist(),
                     'time_from_start': t_f[j, 0].item()})

        else:
            for j in range(1, p_f.shape[0]):
                a = p_f[j, :].tolist()
                a.insert(0, t_f[j, 0])
                b = v_f[j, :].tolist()
                a.extend(b)
                c = a_f[j, :].tolist()
                a.extend(c)
                w.writerow(a[k] for k in range(len(a)))
                pos_dict.append(
                    {'positions': p_f[j, idx_act_joints].tolist(),
                     'time_from_start': t_f[j, 0].item()})

        # last sampple of n-th stack is the initial sample of n+1-th stack
        wp_init = p_f[-1, idx_act_joints]
        vel_wp_init = v_f[-1, idx_act_joints]
        acc_wp_init = a_f[-1, idx_act_joints]
        W_stack = build_W_b(Ns, robot, p_f, v_f, a_f, idx_e,
                            idx_b, W_stack=W_stack)
        plt.plot(iter_num, list_obj_value, label="repeat %d" % (i + 1))
        print("repeat %d" % (i + 1))
        print("---------------------------------------")
        cur_obj = list_obj_value[-1]
        if abs((last_obj-cur_obj)/cur_obj) < tol_stop:
            stop_count += 1
        else:
            stop_count = 0
        if stop_count == 3:
            stop_flag = True
            break


dt = datetime.now()
current_time = dt.strftime("%d_%b_%Y_%H%M")
path_save_yaml = join(
    dirname(dirname(str(abspath(__file__)))),
    f"identification_toolbox/src/thanh/tiago_stacking_{current_time}.yaml")

# with open(path_save_yaml, 'w') as f:
#     data = yaml.dump(pos_dict, f, default_flow_style=None)

plt.title("Evolution of condition number of base regressor over iteration")
plt.ylabel("Cond(Wb)")
plt.xlabel("Iteration counts")
plt.legend(loc="upper right")
plt.grid()
plt.yscale("log")
plt.savefig(join(
    dirname(dirname(str(abspath(__file__)))),
    f"identification_toolbox/src/thanh/tiago_stacking_{current_time}.png"))
plt.show()
# + add options to nlp object
# + got optimal solution
# + generate full config = > constraints check
# = > ok

# + write file
# + create new 1st waypoint for the next stack
# + stack the newly generated regressor to the previous ones
# + if stop condition is not reached: repeat
print("RUNTIME IS ", start - time.time(), "(secs)")