# from ndcurves import piecewise,  ndcurves.exact_cubic, ndcurves.curve_constraints
import ndcurves
import numpy as np
from matplotlib import pyplot as plt
from figaroh.tools.randomdata import get_torque_rand
from figaroh.tools.robot import Robot
import icecream as ic
import eigenpy
import os
eigenpy.switchToNumpyArray()

# This script inherits class robot with its all attrs/funcs, receives
# infor of activated joints, then generate feasible cubic splines which
# respects pos/vel/effort/self-collision.


class CubicSpline:
    def __init__(self,
                 robot,
                 num_waypoints: int,
                 active_joints: list):

        self.robot = robot
        self.rmodel = self.robot.model
        self.num_waypoints = num_waypoints
        self.active_joints = [(self.rmodel.getJointId(i) - 1)
                              for i in active_joints]
        self.dim = (len(self.active_joints), self.num_waypoints)

    # joint limits on active joints
        self.upper_q = self.rmodel.upperPositionLimit[self.active_joints]
        self.lower_q = self.rmodel.lowerPositionLimit[self.active_joints]
        self.upper_dq = self.rmodel.velocityLimit[self.active_joints]
        self.lower_dq = -self.rmodel.velocityLimit[self.active_joints]
        self.upper_effort = self.rmodel.effortLimit[self.active_joints]
        self.lower_effort = -self.rmodel.effortLimit[self.active_joints]

     # joint limits on active joints with soft limit on both limit ends
        if soft_lim >0:
            self.upper_q = self.upper_q - soft_lim*abs(self.upper_q-self.lower_q)
            self.lower_q = self.lower_q + soft_lim*abs(self.upper_q-self.lower_q)
            self.upper_dq = self.upper_dq - soft_lim*abs(self.upper_dq - self.lower_dq)
            self.lower_dq = self.lower_dq + soft_lim*abs(self.upper_dq - self.lower_dq)
            self.upper_effort = self.upper_effort - soft_lim * \
                abs(self.upper_effort - self.lower_effort)
            self.lower_effort = self.lower_effort + soft_lim * \
                abs(self.upper_effort - self.lower_effort)


    def get_active_config(self,
                          freq: int,
                          time_points: np.ndarray,
                          waypoints: np.ndarray,
                          vel_waypoints=None,
                          acc_waypoints=None):
        """ Generate cubic splines on active joints
        """
        # dimensions
        assert (
            self.dim == waypoints.shape), "(Pos) Check size \
                                        (num_active_joints,num_waypoints)!"
        self.pc = ndcurves.piecewise() # set piecewise object to join segments

        # C_2 continuous at waypoints
        if vel_waypoints is not None and acc_waypoints is not None:
            # dimensions
            assert (
                self.dim == vel_waypoints.shape), "(Vel) Check size\
                                        (num_active_joints, num_waypoints)!"
            assert (
                self.dim == acc_waypoints.shape), "(Acc) Check size\
                                        (num_active_joints, num_waypoints)!"

            # make exact cubic WITH constraints on vel and acc on both ends
            for i in range(self.num_waypoints-1):
                self.c = ndcurves.curve_constraints()
                self.c.init_vel = np.matrix(vel_waypoints[:, i]).transpose()
                self.c.end_vel = np.matrix(vel_waypoints[:, i+1]).transpose()
                self.c.init_acc = np.matrix(acc_waypoints[:, i]).transpose()
                self.c.end_acc = np.matrix(acc_waypoints[:, i+1]).transpose()
                ec = ndcurves.exact_cubic(waypoints[:, range(i, i+2)],
                                          time_points[range(i, i+2), 0], self.c)
                self.pc.append(ec)

        # make exact cubic WITHOUT constraints on vel and acc on both ends
        else:
            for i in range(self.num_waypoints - 1):
                ec = ndcurves.exact_cubic(
                    waypoints[:, range(i, i+2)], time_points[range(i, i+2), 0])
                self.pc.append(ec)

        # time step
        self.delta_t = 1/freq
        # total travel time
        self.T = self.pc.max() - self.pc.min()
        # get number sample points from generated trajectory
        self.N = int(self.T/self.delta_t) + 1
        # create time stamps on all sample points
        self.t = self.pc.min() + \
            np.matrix([i*self.delta_t for i in range(self.N)]).transpose()

        # compute derivatives to obtain pos/vel/acc on all samples (bad)
        self.q_act = np.array([self.pc(self.t[i, 0]) for i in range(self.N)],
                              dtype="float")
        self.dq_act = np.array([self.pc.derivate(self.t[i, 0], 1)
                                for i in range(self.N)], dtype="float")
        self.ddq_act = np.array([self.pc.derivate(self.t[i, 0], 2)
                                 for i in range(self.N)], dtype="float")
        t, p_act, v_act, a_act = self.t, self.q_act, self.dq_act, self.ddq_act

        return t, p_act, v_act, a_act

    def get_full_config(self,
                        freq: int,
                        time_points: np.ndarray,
                        waypoints: np.ndarray,
                        vel_waypoints=None,
                        acc_waypoints=None):
        """ Fill active joints profiles to full joint configuration
        """
        t, p_act, v_act, a_act = self.get_active_config(freq,
                                                        time_points,
                                                        waypoints,
                                                        vel_waypoints,
                                                        acc_waypoints)
        # create array of zero configuration times N samples
        self.q_full = np.array([self.robot.q0] * self.N)
        self.dq_full = np.array([self.robot.v0] * self.N)
        self.ddq_full = np.array([self.robot.v0] * self.N)

        # fill in trajectory with active joints values
        self.q_full[:, self.active_joints] = p_act
        self.dq_full[:, self.active_joints] = v_act
        self.ddq_full[:, self.active_joints] = a_act

        p_full, v_full, a_full = self.q_full, self.dq_full, self.ddq_full
        return t, p_full, v_full, a_full

    def check_cfg_constraints(self, q, v=None, tau=None, soft_lim=0):
        """ Check pos/vel/effort constraints on generated trajectory
        """
        __isViolated = False
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                delta_lim = soft_lim * \
                    abs(self.rmodel.upperPositionLimit[j] -
                        self.rmodel.lowerPositionLimit[j])
                if q[i, j] > self.rmodel.upperPositionLimit[j] - delta_lim:
                    # print("Joint %d upper limit violated!" % j)
                    __isViolated_pos = True

                elif q[i, j] < self.rmodel.lowerPositionLimit[j] + delta_lim:
                    # print("Joint %d lower limit violated!" % j)
                    __isViolated_pos = True
                else:
                    __isViolated_pos = False
                __isViolated = __isViolated or __isViolated_pos
                # print(__isViolated)
        if v is not None:
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    if abs(v[i, j]) > (1-soft_lim)*abs(self.rmodel.velocityLimit[j]):
                        # print("Velocity joint %d limits violated!" % j)
                        __isViolated_vel = True
                    else:
                        __isViolated_vel = False
                    __isViolated = __isViolated or __isViolated_vel
                # print(__isViolated)
        if tau is not None:
            for i in range(tau.shape[0]):
                for j in range(tau.shape[1]):
                    if abs(tau[i, j]) > (1-soft_lim)*abs(self.rmodel.effortLimit[j]):
                        # print("Effort joint %d limits violated!" % j)
                        __isViolated_eff = True
                    else:
                        __isViolated_eff = False
                    __isViolated = __isViolated or __isViolated_eff
                    # print(__isViolated)
        if not __isViolated:
            print("Succeeded to generate waypoints for  a feasible initial cubic spline")
        else:
            print("Failed to generate a feasible cubic spline")        
        return __isViolated

    def check_self_collision(self):
        __isViolated = False

        return __isViolated

    def plot_spline(self, t, p, v, a):
        q = p[:, self.active_joints]
        dq = v[:, self.active_joints]
        ddq = a[:, self.active_joints]

        for i in range(q.shape[1]):
            plot = plt.figure(i)
            plt.plot(t[:, 0], q[:, i], color='r', label='pos')
            plt.plot(t[:, 0], dq[:, i], color='b', label='vel')
            plt.plot(t[:, 0], ddq[:, i], color='g',label='acc')
            plt.title('joint %s' % i)
            plt.legend()
            plt.grid()
        plt.show()


class WaypointsGeneration(CubicSpline):
    """ Generate waypoints specific for cubic spline
    """
    def __init__(self,
                 robot,
                 num_waypoints: int,
                 active_joints: list):
        super().__init__(robot, num_waypoints, active_joints)
        self.n_set = 10
        self.pool_q = np.zeros((self.n_set, len(self.active_joints)))
        self.pool_dq = np.zeros((self.n_set, len(self.active_joints)))
        self.pool_ddq = np.zeros((self.n_set, len(self.active_joints)))

    def gen_rand_pool(self, soft_limit_pool=0):
        """ Generate a uniformly distributed waypoint pool of pos/vel/acc over
            a specific range
        """
        assert np.array(soft_limit_pool).shape == (3,len(self.active_joints)), \
            "input a vector of soft limit pool with a shape of (3, len(activejoints)"
        lim_q = soft_limit_pool[0, :]
        lim_dq = soft_limit_pool[1, :]
        lim_ddq = soft_limit_pool[2, :]

        new_upper_q = np.zeros_like(self.upper_q)
        new_lower_q = np.zeros_like(self.lower_q)
        new_upper_dq = np.zeros_like(self.upper_dq)
        new_lower_dq = np.zeros_like(self.lower_dq)
        new_upper_ddq = np.zeros_like(self.upper_dq)
        new_lower_ddq = np.zeros_like(self.lower_dq)

        for i in range(len(self.active_joints)):
            new_upper_q[i] = self.upper_q[i] - lim_q[i] * \
                abs(self.upper_q[i] - self.lower_q[i])
            new_lower_q[i] = self.lower_q[i] + lim_q[i] * \
                abs(self.upper_q[i] - self.lower_q[i])

            new_upper_dq[i] = self.upper_dq[i] - lim_dq[i] * \
                abs(self.upper_dq[i] - self.lower_dq[i])
            new_lower_dq[i] = self.lower_dq[i] + lim_dq[i] * \
                abs(self.upper_dq[i] - self.lower_dq[i])

            k = 1.5  # take accel limits as k tims of vel limts 
            new_upper_ddq[i] = k*(self.upper_dq[i] - lim_ddq[i] * 
                abs(self.upper_dq[i] - self.lower_dq[i]))
            new_lower_ddq[i] = k*(self.lower_dq[i] + lim_ddq[i] * 
                abs(self.upper_dq[i] - self.lower_dq[i]))

            # new_upper_effort = self.upper_effort - soft_limit * \
            #     abs(self.upper_effort - self.lower_effort)
            # new_lower_effort = self.lower_effort - soft_limit * \
            #     abs(self.upper_effort - self.lower_effort)

        for i in range(len(self.active_joints)):
            print(  "joint %s" % i,
                    "pos limit range: ", new_upper_q[i]- new_lower_q[i],
                    "vel limit:", abs(new_lower_dq[i]),
                    "min required travel time: ", (new_upper_q[i]- new_lower_q[i])/abs(new_lower_dq[i]))
            step_q = (new_upper_q[i] - new_lower_q[i])/(self.n_set - 1) 
            self.pool_q[:, i] = np.array(
                [new_lower_q[i] + j*step_q for j in range(self.n_set)])

            step_dq = (new_upper_dq[i] - new_lower_dq[i])/(self.n_set - 1)
            self.pool_dq[:, i] = np.array(
                [new_lower_dq[i] + j*step_dq for j in range(self.n_set)])

            step_ddq = (new_upper_ddq[i] - new_lower_ddq[i])/(self.n_set - 1)
            self.pool_ddq[:, i] = np.array(
                [new_lower_ddq[i] + j*step_ddq for j in range(self.n_set)])
        # return self.pool_q, self.pool_dq, self.pool_ddq

    def gen_rand_wp(self, soft_limit=0,
                    wp_init=None, vel_wp_init=None, acc_wp_init=None,
                    vel_set_zero=True, acc_set_zero=True):
        """ Generate waypoint pos/vel/acc which randomly pick from waypoint
            pool
            Or, set vel and/or acc at waypoints to be zero
        """
        # self.gen_rand_pool(soft_limit)
        q_rand = np.zeros((self.num_waypoints, len(self.active_joints)))
        dq_rand = np.zeros((self.num_waypoints, len(self.active_joints)))
        ddq_rand = np.zeros((self.num_waypoints, len(self.active_joints)))

        if wp_init is not None:
            q_rand[0, :] = wp_init
            for i in range(len(self.active_joints)):
                q_rand[range(1, self.num_waypoints), i] = np.random.choice(
                    self.pool_q[:, i], self.num_waypoints-1)
        else:
            for i in range(len(self.active_joints)):
                q_rand[:, i] = np.random.choice(
                    self.pool_q[:, i], self.num_waypoints)

        if vel_wp_init is not None:
            dq_rand[0, :] = vel_wp_init
            if not vel_set_zero:
                for i in range(len(self.active_joints)):
                    dq_rand[range(1, self.num_waypoints), i] = np.random.choice(
                        self.pool_dq[:, i], self.num_waypoints-1)
            else:
                dq_rand[range(1, self.num_waypoints), :] = np.zeros((self.num_waypoints-1, len(self.active_joints)))
        else:
            if not vel_set_zero:
                for i in range(len(self.active_joints)):
                    dq_rand[:, i] = np.random.choice(
                        self.pool_dq[:, i], self.num_waypoints)
            else:
                dq_rand = np.zeros((self.num_waypoints, len(self.active_joints)))


        if vel_wp_init is not None:
            ddq_rand[0, :] = vel_wp_init
            if not acc_set_zero:
                for i in range(len(self.active_joints)):
                    ddq_rand[range(1, self.num_waypoints), i] = np.random.choice(
                        self.pool_ddq[:, i], self.num_waypoints-1)
            else:
                ddq_rand[range(1, self.num_waypoints), :] = np.zeros((self.num_waypoints-1, len(self.active_joints)))
        else:
            if not acc_set_zero:
                for i in range(len(self.active_joints)):
                    ddq_rand[:, i] = np.random.choice(
                        self.pool_ddq[:, i], self.num_waypoints)
            else:
                ddq_rand = np.zeros((self.num_waypoints, len(self.active_joints)))
        # print("pool for waypoints generation: ", self.pool_q)
        return q_rand.transpose(), dq_rand.transpose(), ddq_rand.transpose()

    def get_feasible_wp(self):
        pass

    def get_rand_tp():
        pass


def main():
    
    # 1/ Load robot model and create a dictionary containing reserved constants
    ros_package_path = os.getenv('ROS_PACKAGE_PATH')
    package_dirs = ros_package_path.split(':')
    robot_dir = package_dirs[0] + "/example-robot-data/robots"
    robot = Robot(
                robot_dir + "/tiago_description/robots/tiago_no_hand.urdf",
                package_dirs = package_dirs,
                # isFext=True  # add free-flyer joint at base
                )
    num_waypoints = 2
    active_joints = ["torso_lift_joint",
                     "arm_1_joint",
                     "arm_2_joint",
                     "arm_3_joint",
                     "arm_4_joint",
                     "arm_5_joint",
                     "arm_6_joint",
                     "arm_7_joint"]
    CB = CubicSpline(robot, num_waypoints, active_joints)

    f = 50
    T1 = 0.0
    T2 = 30
    # T3 = 5.0
    soft_lim = 0.1
    soft_lim_pool = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    WP = WaypointsGeneration(robot, num_waypoints, active_joints)
    isViolated = True  # constraint flag
    WP.gen_rand_pool(soft_lim_pool)
    count = 0
    while isViolated:
        count += 1
        print("----------","run %s " % count, "----------")
        wps, vel_wps, acc_wps = WP.gen_rand_wp(soft_lim)
        # print(WP.pool_q[0, 0])
        time_points = np.matrix([T1, T2]).transpose()
        t, p_act, v_act, a_act = CB.get_full_config(
            f, time_points, wps, vel_waypoints=vel_wps, acc_waypoints=acc_wps)
        tau = get_torque_rand(p_act.shape[0], robot, p_act, v_act, a_act)
        tau = np.reshape(tau, (v_act.shape[1], v_act.shape[0])).transpose()
        isViolated = CB.check_cfg_constraints(p_act, v_act, tau)


    CB.plot_spline(t, p_act, v_act, a_act)
    print(t.shape)

if __name__ == '__main__':
    main()
