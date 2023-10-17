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

import numpy as np
import cyipopt
import os
from matplotlib import pyplot as plt
import time
import numdifftools as nd
import yaml
from yaml.loader import SafeLoader
import pprint

from figaroh.tools.robot import Robot
from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
    get_index_eliminate,
)
from figaroh.tools.qrdecomposition import (
    get_baseIndex,
    build_baseRegressor,
)
from figaroh.tools.randomdata import get_torque_rand
from figaroh.tools.robotcollisions import CollisionWrapper
from simplified_colission_model import build_tiago_simplified
from cubic_spline import CubicSpline, WaypointsGeneration
from figaroh.identification.identification_tools import get_param_from_yaml

# HELPER FUNCTIONS TO OBTAIN BASE REGRESSOR (BASE REG)


def get_idx_from_random(robot, q, v, a, param):
    """Sole purpose is to get index of eliminate std param in W to
    produce W_e and then get index of independent params in W_e
    TODO: remove redundant computation
    """
    W = build_regressor_basic(robot, q, v, a, param)
    params_std = robot.get_standard_parameters(param)
    idx_e_, par_r_ = get_index_eliminate(W, params_std, tol_e=0.001)
    W_e_ = build_regressor_reduced(W, idx_e_)
    idx_base_ = get_baseIndex(W_e_, par_r_)
    return idx_e_, idx_base_


def build_W_b(robot, q, v, a, param, idx_e_, idx_base_, W_stack=None):
    """Given index of eliminate std params and independent params,
    now build base regressor for given data
    TODO: put idx_e and idx_base into param dict
    """
    W = build_regressor_basic(robot, q, v, a, param)
    W_e_ = build_regressor_reduced(W, idx_e_)
    W_b_ = build_baseRegressor(W_e_, idx_base_)
    # list_idx = []
    # for k in range(len(idx_act_joints)):
    #     list_idx.extend(
    #         list(range(idx_act_joints[k]*Ns, (idx_act_joints[k]+1)*Ns)))
    # W_b = W_b_[list_idx, :]
    W_b = W_b_

    # stack the computed base reg below the previous base reg
    if isinstance(W_stack, np.ndarray):
        W_b = np.vstack((W_stack, W_b))
    return W_b


def get_idx_b_cubic(robot, param, active_joints):
    """Find base parameters for cubic spline trajectory"""
    n_wps_r = 100
    freq_r = 100
    CB_r = CubicSpline(robot, n_wps_r, active_joints)
    WP_r = WaypointsGeneration(robot, n_wps_r, active_joints)
    WP_r.gen_rand_pool(soft_lim_pool)

    # generate random waypoints
    wps_r, vel_wps_r, acc_wps_r = WP_r.gen_rand_wp()

    # generate timepoints
    tps_r = np.matrix([0.5 * i for i in range(n_wps_r)]).transpose()

    # get full config traj
    t_r, p_r, v_r, a_r = CB_r.get_full_config(
        freq_r, tps_r, wps_r, vel_wps_r, acc_wps_r
    )

    # get index essential and base params columns: idx_e, idx_b
    idx_e, idx_b = get_idx_from_random(robot, p_r, v_r, a_r, param)
    print("number of base params: ", len(idx_b))
    return idx_e, idx_b


# IPOPT PROBLEM FORMULATION FUNCTIONS


def objective_func(
    X, params_settings, opt_cb, tps, vel_wps, acc_wps, wp_init, W_stack_=None
):
    """This functions computes the condition number of correspondent
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
    wps_X = np.reshape(X, (n_wps - 1, len(active_joints)))
    wps = np.vstack((wp_init, wps_X))
    wps = wps.transpose()

    # create full profile
    t_f, p_f, v_f, a_f = CB.get_full_config(freq, tps, wps, vel_wps, acc_wps)
    opt_cb["t_f"] = t_f
    opt_cb["p_f"] = p_f
    opt_cb["v_f"] = v_f
    opt_cb["a_f"] = a_f

    # get stacked base reg
    W_b = build_W_b(
        robot, p_f, v_f, a_f, params_settings, idx_e, idx_b, W_stack=W_stack_
    )

    return np.linalg.cond(W_b)


def get_constraints_all_samples(Ns, X, opt_cb, tps, vel_wps, acc_wps, wp_init):
    """Concatenate constraints into one vector:
    - joint angle (pos) constraints at waypoints
    - velocity constraints on all sample points
    - effort (joint torque/force) constraints on all sample points
    - auto-collision pairs from simplified collision model

    """
    # add the start waypoint and re-arrange waypoints
    X = np.array(X)
    wps_X = np.reshape(X, (n_wps - 1, len(active_joints)))
    wps = np.vstack((wp_init, wps_X))
    wps = wps.transpose()

    # create full profile
    t_f, p_f, v_f, a_f = CB.get_full_config(freq, tps, wps, vel_wps, acc_wps)

    # compute joint effort given full profile
    tau = get_torque_rand(p_f.shape[0], robot, p_f, v_f, a_f, params_settings)

    # pos constraints at waypoints
    idx_waypoints = []
    # time_points = np.array([[s * t_s] for s in range(1, n_wps)])
    time_points = tps[range(1, n_wps), :]
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
        tau_constraints[:, k] = tau[
            range(idx_act_joints[k] * Ns, (idx_act_joints[k] + 1) * Ns)
        ]

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


def get_bounds(CB, n_wps):
    """Set boundaries for search variables"""
    lb = []
    ub = []
    for i in range(1, n_wps):
        lb.extend(CB.lower_q)
        ub.extend(CB.upper_q)
    print("shape of search var bounds:", len(lb), len(ub))
    return lb, ub


def get_constr_value(robot, CB, n_wps, Ns):
    """Set limit values for all constraints"""
    cl = []
    cu = []

    # inequality constraints values of pos
    cl_pos = []
    cu_pos = []
    for i in range(1, n_wps):
        cl_pos.extend(CB.lower_q)
        cu_pos.extend(CB.upper_q)
    print("cl_pos shape: ", len(cl_pos))

    # inequality constraints values of vel
    cl_vel = []
    cu_vel = []
    for j in range(Ns):  # Ns: number of total samples on traj
        cl_vel.extend(CB.lower_dq)
        cu_vel.extend(CB.upper_dq)
    print("cl_vel shape: ", len(cl_vel))

    # inequality constraints values of effort
    cl_eff = []
    cu_eff = []
    for j in range(Ns):  # Ns: number of total samples on traj
        cl_eff.extend(CB.lower_effort)
        cu_eff.extend(CB.upper_effort)
    print("cl_eff shape: ", len(cl_eff))

    # inequality constraints values of self collision

    n_cols = len(robot.geom_model.collisionPairs)
    cl_cols = [0.01] * n_cols * (n_wps - 1)  # 1 cm margin
    cu_cols = [2 * 1e19] * n_cols * (n_wps - 1)  # no limit on max distance

    cl = cl_pos + cl_vel + cl_eff + cl_cols
    cu = cu_pos + cu_vel + cu_eff + cu_cols

    print("constraints bounds lower shape: ", len(cl))
    print("constraints bounds lower shape: ", len(cu))

    return cl, cu


class Problem_cond_Wb:
    def __init__(
        self,
        Ns,
        params_settings,
        opt_cb,
        tps,
        vel_wps,
        acc_wps,
        wp_init,
        vel_wp_init,
        acc_wp_init,
        W_stack,
        stop_flag,
    ):
        self.W_stack = W_stack  # update every stacking repeat
        # init waypoint of current stack = end waypoint of prev stack
        self.wp_init = wp_init 
        self.vel_wp_init = vel_wp_init
        self.acc_wp_init = acc_wp_init
        self.tps = tps  # timestamp at waypoints increasing over stacking
        self.vel_wps = vel_wps  # velocity at waypoints
        self.acc_wps = acc_wps  # acceleration at waypoints
        self.stop_flag = stop_flag  # optimization stopping flag
        self.opt_cb = opt_cb  # optimal cubic spline stored

    def gen_cb(self, X):
        # add the start waypoint and re-arrange waypoints
        X = np.array(X)
        wps_X = np.reshape(X, (n_wps - 1, len(active_joints)))
        wps = np.vstack((wp_init, wps_X))
        wps = wps.transpose()

        # create full profile
        t_f, p_f, v_f, a_f = CB.get_full_config(
            freq, tps, wps, vel_wps, acc_wps
        )

        # compute joint effort given full profile
        tau = get_torque_rand(
            p_f.shape[0], robot, p_f, v_f, a_f, params_settings
        )

        self.opt_cb["t_f"] = t_f
        self.opt_cb["p_f"] = p_f
        self.opt_cb["v_f"] = v_f
        self.opt_cb["a_f"] = a_f
        self.opt_cb["tau_f"] = tau

    def objective(self, X):
        return objective_func(
            X,
            params_settings,
            self.opt_cb,
            self.tps,
            self.vel_wps,
            self.acc_wps,
            self.wp_init,
            W_stack_=self.W_stack,
        )

    def gradient(self, X):
        def obj_f(x):
            return self.objective(x)

        grad_obj = nd.Gradient(obj_f)(X)
        return grad_obj

    def constraints(self, X):
        constr_vec = get_constraints_all_samples(
            Ns,
            X,
            self.opt_cb,
            self.tps,
            self.vel_wps,
            self.acc_wps,
            self.wp_init,
        )
        return constr_vec

    def jacobian(self, X):
        def f(x):
            return self.constraints(x)

        jac = nd.Jacobian(f)(X)
        print("constraint jacobian shape: ", jac.shape)
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
    nlp.add_option(b"check_derivatives_for_naninf", b"yes")
    # nlp.add_option(b"evaluate_orig_obj_at_resto_trial", b"no")


start = time.time()

# 1/ Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv("ROS_PACKAGE_PATH")
package_dirs = ros_package_path.split(":")
robot_dir = package_dirs[0] + "/example-robot-data/robots"
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago_no_hand.urdf",
    package_dirs=package_dirs,
    # isFext=True  # add free-flyer joint at base
)
active_joints = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
]

# TODO: specify soft_lim and soft_lim_pool to individual joint
soft_lim = 0.05  # discount from max and min of limit for all samples

# soft_lim_pool = 0.1 # discount for pool
soft_lim_pool = np.array(
    [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ]
)

# step 2: load simplified collsion model
robot = build_tiago_simplified(robot)
# load standard parameters
with open("examples/tiago/config/tiago_config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)
identif_data = config["identification"]
params_settings = get_param_from_yaml(robot, identif_data)

# step 3: get indices of base parameters specifically for cubic spline
idx_e, idx_b = get_idx_b_cubic(robot, params_settings, active_joints)


# stop condition for iterative stacking
n_wps = 2  # number of waypoints for spline trajectory include the destination
stop_count = 0
tol_stop = 0.1
last_obj = 0
stack_reps = 5  # stacks
freq = 100  # Hz
t_s = 2
stop_flag = False
fig_cond = plt.figure()

T_F = []
P_F = []
V_F = []
A_F = []

CB = CubicSpline(robot, n_wps, active_joints, soft_lim)
WP = WaypointsGeneration(robot, n_wps, active_joints, soft_lim)
WP.gen_rand_pool(3 * soft_lim_pool)
# step 4: define boundaries of search vars, and constraints
idx_act_joints = CB.active_joints

# step 5: initialize 1st waypoint
wp_init = np.zeros(len(idx_act_joints))
vel_wp_init = np.zeros(len(idx_act_joints))
acc_wp_init = np.zeros(len(idx_act_joints))
for idx in range(len(idx_act_joints)):
    wp_init[idx] = np.random.choice(WP.pool_q[:, idx], 1)
W_stack = None

# step 6: path to save file
# dt = datetime.now()
# current_time = dt.strftime("%d_%b_%Y_%H%M")
# path_save_bp = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"tiago/data/tiago_stacking_{current_time}.csv")

# with open(path_save_bp, "w") as output_file:
# w = csv.writer(output_file)
# first_row = ["%d" % i for i in range(15+12+12)]
# first_row.insert(0, 't')
# w.writerow(first_row)
# pos_dict = []  # list(dist) to dump on yaml


# step 7: for loop in stack_reps times
WP.gen_rand_pool(soft_lim_pool)
for s_rep in range(stack_reps):
    list_obj_value = []
    iter_num = []
    is_constr_violated = True
    count = 0
    print("AT THE BEGINNING OF SEGMENT %s : " % (s_rep + 1), wp_init)
    # generate feasible initial guess
    while is_constr_violated:
        count += 1
        print("----------", "run %s " % count, "----------")
        wps, vel_wps, acc_wps = WP.gen_rand_wp(
            wp_init, vel_wp_init, acc_wp_init
        )
        # wps, vel_wps, acc_wps = WP.gen_equal_wp(
        # wp_init, vel_wp_init, acc_wp_init)
        tps = (
            t_s * s_rep
            + np.matrix([t_s * i_wp for i_wp in range(n_wps)]).transpose()
        )

        t_i, p_i, v_i, a_i = CB.get_full_config(
            freq, tps, wps, vel_wps, acc_wps
        )
        tau_i = get_torque_rand(
            p_i.shape[0], robot, p_i, v_i, a_i, params_settings
        )
        # ATTENTION: joint torque specially arranged!
        tau_i = np.reshape(tau_i, (v_i.shape[1], v_i.shape[0])).transpose()
        is_constr_violated = CB.check_cfg_constraints(p_i, v_i, tau_i)
        if count > 1000:
            break
    # reshape wps to a vector of search variable
    X0 = wps[:, range(1, n_wps)]
    X0 = np.reshape(
        X0.transpose(), ((len(active_joints) * (n_wps - 1),))
    ).tolist()

    print("1st waypoint at the beginning: ", wp_init)
    print("next waypoint(s) (initial guess): ", X0)
    Ns = p_i.shape[0]

    # set search bounds
    lb, ub = get_bounds(CB, n_wps)

    # set constraints bounds
    cl, cu = get_constr_value(robot, CB, n_wps, Ns)

    # optimal segment trajectory
    opt_cb = {
        "t_f": None,
        "p_f": None,
        "v_f": None,
        "a_f": None,
        "tau_f": None,
    }

    # ipopt problem formulation
    nlp = cyipopt.Problem(
        n=len(X0),
        m=len(cl),
        problem_obj=Problem_cond_Wb(
            Ns,
            params_settings,
            opt_cb,
            tps,
            vel_wps,
            acc_wps,
            wp_init,
            vel_wp_init,
            acc_wp_init,
            W_stack,
            stop_flag,
        ),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    add_options_nlp(nlp)

    # ipopt result
    X_opt, infor = nlp.solve(X0)
    wps_X = np.reshape(np.array(X_opt), (n_wps - 1, len(active_joints)))
    print("1st waypoint at solution: ", opt_cb["p_f"][0, idx_act_joints])
    print("next waypoint(s) (solution): ", wps_X)
    print("###################################################")
    print("SOLUTION INFOR: ", infor["status"], infor["status_msg"])

    if infor["status"] in [-1, 0, 1]:
        # code 0: optimal solution found
        # code 1: acceptable solved
        # code -1: maximum iteration reached
        T_F.append(opt_cb["t_f"])
        P_F.append(opt_cb["p_f"][:, idx_act_joints])
        V_F.append(opt_cb["v_f"][:, idx_act_joints])
        A_F.append(opt_cb["a_f"][:, idx_act_joints])
        # check contraints violation for solution segment trajectory

        # if generated trajectory violated constraints, break
        tau = get_torque_rand(
            opt_cb["p_f"].shape[0],
            robot,
            opt_cb["p_f"],
            opt_cb["v_f"],
            opt_cb["a_f"],
            params_settings,
        )
        tau = np.reshape(
            tau, (opt_cb["v_f"].shape[1], opt_cb["v_f"].shape[0])
        ).transpose()
        is_constr_violated = CB.check_cfg_constraints(
            opt_cb["p_f"], opt_cb["v_f"], tau
        )
        if is_constr_violated:
            print("Constrainted VIOLATED!")
            stop_flag = True
            break

        # reinitialize for next stacking
        wp_init = wps_X[-1, :]
        W_stack = build_W_b(
            robot,
            opt_cb["p_f"],
            opt_cb["v_f"],
            opt_cb["a_f"],
            params_settings,
            idx_e,
            idx_b,
            W_stack=W_stack,
        )
        print("regressor is stacked with size: ", W_stack.shape)

        if infor["status"] == 0 or infor["status"] == 1:
            print("iter %s of stacking SUCCEEDED!" % (s_rep + 1))
            plt.plot(iter_num, list_obj_value, label="repeat %d" % (s_rep + 1))

        elif infor["status"] == -1:
            print("iter %s of stacking REACHED MAX ITER!" % (s_rep + 1))
            plt.plot(iter_num, list_obj_value, label="repeat %d" % (s_rep + 1))
        print("AT THE END OF THE SEGMENT %s" % (s_rep + 1), wp_init)

        # conditional break stacking if does not improve
        cur_obj = list_obj_value[-1]
        if abs((last_obj - cur_obj) / cur_obj) < tol_stop:
            stop_count += 1
        else:
            stop_count = 0

        if stop_count == 3:
            stop_flag = True
            print(
                "Optimizing stops because \
                bjective func value does not improve over %s iterations"
                % stop_count
            )
            break
    else:
        print("repeat of stacking %d FAILED! by not converging" % (s_rep + 1))
        stop_flag = True
        break
    # # write to yaml file

    # p_f = np.around(p_f, 4)
    # v_f = np.around(v_f, 4)
    # a_f = np.around(a_f, 4)
    # t_f = t_f + i*t_s
    # # for j in range(t_f.shape[0]):
    # # CB.plot_spline(t_f, p_f, v_f, a_f)
    # # if i == 0:
    # #     for j in range(p_f.shape[0]):
    # #         a = p_f[j, :].tolist()
    # #         a.insert(0, t_f[j, 0])
    # #         b = v_f[j, :].tolist()
    # #         a.extend(b)
    # #         c = a_f[j, :].tolist()
    # #         a.extend(c)
    # #         w.writerow(a[k] for k in range(len(a)))
    # #         pos_dict.append(
    # #             {'positions': p_f[j, idx_act_joints].tolist(),
    # #              'time_from_start': t_f[j, 0].item()})

    # # else:
    # #     for j in range(1, p_f.shape[0]):
    # #         a = p_f[j, :].tolist()
    # #         a.insert(0, t_f[j, 0])
    # #         b = v_f[j, :].tolist()
    # #         a.extend(b)
    # #         c = a_f[j, :].tolist()
    # #         a.extend(c)
    # #         w.writerow(a[k] for k in range(len(a)))
    # #         pos_dict.append(
    # #             {'positions': p_f[j, idx_act_joints].tolist(),
    # #              'time_from_start': t_f[j, 0].item()})

    # add plot of one segment
    print(
        "#########################END of %s-th OPTIMIZATION###################"
        % (s_rep + 1)
    )

print("RUNTIME IS ", start - time.time(), "(secs)")

# write to yaml file

# path_save_yaml = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"tiago/data/tiago_stacking_{current_time}.yaml")

# with open(path_save_yaml, 'w') as f:
#     data = yaml.dump(pos_dict, f, default_flow_style=None)

# plotting results
plt.title("Evolution of condition number of base regressor over iteration")
plt.ylabel("Cond(Wb)")
plt.xlabel("Iteration counts")
plt.legend(loc="upper right")
plt.grid()
# plt.yscale("log")
plt.show()
# current_time = datetime.now().strftime("%d_%b_%Y_%H%M")
# plt.savefig(join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"tiago/data/tiago_ipopt_evo_{current_time}.png"))

# plot trajectory
fig_cb, ax_cb = plt.subplots(len(idx_act_joints), 3, sharex=True)
for jj in range(len(T_F)):
    for ii in range(len(idx_act_joints)):
        ax_cb[ii, 0].plot(T_F[jj], P_F[jj][:, ii])

        ax_cb[ii, 1].plot(T_F[jj], V_F[jj][:, ii])

        ax_cb[ii, 2].plot(T_F[jj], A_F[jj][:, ii])

plt.show()

# current_time = datetime.now().strftime("%d_%b_%Y_%H%M")
# plt.savefig(join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"tiago/data/tiago_opt_cubic_{current_time}.png"))

"""

# + construct ipopt class nlp object:
#     /objective function: only take search variable as input
#     f(search varible)
#     = > generate cubic spline(search varible)
#     = > gen full config
#     = > build base regressor Wb
#     = > cond(Wb)
#     /gradient: only take search variable as, do gradient descent on
                objective function
#     /constraints: only take search variable as input
#     f(search varible)
#     = > generate cubic spline(search varible)
#     = > gen full config: array of q, v, a on all sample points
#     = > get joint torque: array of tau on all sample points
#     = > construct collision object -> compute distance on all pairs
        (simplified) at waypoints: array of collision pair dist
#     /jacobian: only take search variable as input
#     /hessian: only take search variable as input
#     /intermediate
# + add options to nlp object
# + got optimal solution
# + generate full config = > constraints check
# = > ok

# + write file
# + create new 1st waypoint for the next stack
# + stack the newly generated regressor to the previous ones
# + if stop condition is not reached: repeat


IPOPT return code meanings
public final static int SOLVE_SUCCEEDED = 0;
public final static int ACCEPTABLE_LEVEL = 1;
public final static int INFEASIBLE_PROBLEM = 2;
public final static int SEARCH_DIRECTION_TOO_SMALL = 3;
public final static int DIVERGING_ITERATES = 4;
public final static int USER_REQUESTED_STOP = 5;
public final static int ITERATION_EXCEEDED = -1;
public final static int RESTORATION_FAILED = -2;
public final static int ERROR_IN_STEP_COMPUTATION = -3;
public final static int CPUTIME_EXCEEDED = -4;
public final static int WALLTIME_EXCEEDED = -5;           
public final static int NOT_ENOUGH_DEGREES_OF_FRE = -10;
public final static int INVALID_PROBLEM_DEFINITION = -11;
public final static int INVALID_OPTION = -12;
public final static int INVALID_NUMBER_DETECTED = -13;
public final static int UNRECOVERABLE_EXCEPTION = -100;
public final static int NON_IPOPT_EXCEPTION = -101;
public final static int INSUFFICIENT_MEMORY = -102;
public final static int INTERNAL_ERROR = -199;
"""
