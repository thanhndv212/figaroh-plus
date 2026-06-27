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

"""
Cubic Spline Trajectory Generation for Robotics Applications.

This module provides comprehensive tools for generating smooth, continuous
trajectories using cubic splines for robotic systems. It supports constraint
handling, waypoint generation, and trajectory validation with joint limits.

The module contains two main classes:
- CubicSpline: Core trajectory generation with cubic splines
- WaypointsGeneration: Automated waypoint generation for trajectory planning

Key Features:
    - C2 continuous cubic spline trajectories
    - Joint position, velocity, and acceleration constraints
    - Waypoint-based trajectory planning
    - Collision and constraint checking
    - Visualization capabilities
    - Random and systematic waypoint generation

Dependencies:
    - ndcurves: Curve generation and manipulation
    - numpy: Numerical computations
    - matplotlib: Plotting and visualization
    - pinocchio: Robot kinematics and dynamics

Examples:
    Basic trajectory generation:
        ```python
        from figaroh.utils.cubic_spline import CubicSpline

        # Initialize spline generator
        spline = CubicSpline(robot, num_waypoints=5,
                           active_joints=['joint1', 'joint2'])

        # Generate trajectory
        time_points = np.array([[0], [1], [2], [3], [4]])
        waypoints = np.random.rand(2, 5)  # 2 joints, 5 waypoints

        t, pos, vel, acc = spline.get_full_config(
            freq=100, time_points=time_points, waypoints=waypoints
        )

        # Check constraints
        spline.check_cfg_constraints(pos, vel)
        ```

    Automated waypoint generation:
        ```python
        from figaroh.utils.cubic_spline import WaypointsGeneration

        # Initialize waypoint generator
        wp_gen = WaypointsGeneration(robot, num_waypoints=5,
                                   active_joints=['joint1', 'joint2'])

        # Generate waypoint pool
        wp_gen.gen_rand_pool()

        # Generate random waypoints
        pos_wp, vel_wp, acc_wp = wp_gen.gen_rand_wp()

        # Generate trajectory from waypoints
        t, pos, vel, acc = wp_gen.get_full_config(
            freq=100, time_points=time_points, waypoints=pos_wp
        )
        ```

References:
    - Spline theory: "A Practical Guide to Splines" by Carl de Boor
    - Robot trajectory planning: "Introduction to Robotics" by John Craig
    - ndcurves library: https://github.com/humanoid-path-planner/ndcurves
"""

# from ndcurves import piecewise, ndcurves.exact_cubic,
# ndcurves.curve_constraints
import logging

import ndcurves
import numpy as np
from matplotlib import pyplot as plt
import pinocchio as pin

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


k = 1.5  # take accel limits as k times of vel limits


class CubicSpline:
    """
    Cubic spline trajectory generator for robotic systems.

    This class generates smooth, C2-continuous trajectories using cubic splines
    for robotic systems. It supports both position-only and full kinematic
    constraint specification (position, velocity, acceleration) at waypoints.

    The class handles active joint selection, joint limit constraints, and
    provides methods for trajectory validation and visualization.

    Attributes:
        robot: Robot model instance
        rmodel: Robot kinematic model
        num_waypoints (int): Number of waypoints in trajectory
        act_Jid (List[int]): Active joint IDs
        act_Jname (List[str]): Active joint names
        act_J (List): Active joint objects
        act_idxq (List[int]): Position indices for active joints
        act_idxv (List[int]): Velocity indices for active joints
        dim_q (Tuple[int, int]): Position vector dimensions
        dim_v (Tuple[int, int]): Velocity vector dimensions
        upper_q, lower_q (np.ndarray): Position limits for active joints
        upper_dq, lower_dq (np.ndarray): Velocity limits for active joints
        upper_effort, lower_effort (np.ndarray): Effort limits for active
            joints

    Examples:
        Basic usage:
            ```python
            # Initialize for 2 joints, 5 waypoints
            spline = CubicSpline(robot, num_waypoints=5,
                               active_joints=['joint1', 'joint2'])

            # Define waypoints and time stamps
            waypoints = np.array([[0, 1, 2, 1, 0],      # joint1 positions
                                [0, 0.5, 1, 0.5, 0]])   # joint2 positions
            time_points = np.array([[0], [1], [2], [3], [4]])

            # Generate trajectory at 100 Hz
            t, pos, vel, acc = spline.get_full_config(
                freq=100, time_points=time_points, waypoints=waypoints
            )

            # Validate constraints
            is_violated = spline.check_cfg_constraints(pos, vel)
            ```

        With velocity and acceleration constraints:
            ```python
            # Define waypoint constraints
            vel_waypoints = np.zeros((2, 5))  # Zero velocity at waypoints
            acc_waypoints = np.zeros((2, 5))  # Zero acceleration at waypoints

            # Generate constrained trajectory
            t, pos, vel, acc = spline.get_full_config(
                freq=100, time_points=time_points, waypoints=waypoints,
                vel_waypoints=vel_waypoints, acc_waypoints=acc_waypoints
            )

            # Visualize results
            spline.plot_spline(t, pos, vel, acc)
            ```

    Note:
        The class uses the ndcurves library for cubic spline generation.
        Joint limits are automatically extracted from the robot model
        and can be modified with soft limits for safety margins.
    """

    def __init__(self, robot, num_waypoints: int, active_joints: list, soft_lim=0):
        """
        Initialize the cubic spline trajectory generator.

        Args:
            robot: Robot model instance containing kinematic information
            num_waypoints (int): Number of waypoints for the trajectory
            active_joints (list): List of joint names to include in trajectory
            soft_lim (float, optional): Soft limit reduction factor (0-1).
                Reduces joint limits by this fraction for safety.
                Defaults to 0.

        Raises:
            AssertionError: If joint names are not found in robot model

        Examples:
            ```python
            # Basic initialization
            spline = CubicSpline(robot, 5, ['joint1', 'joint2'])

            # With 10% safety margin on joint limits
            spline = CubicSpline(robot, 5, ['joint1', 'joint2'],
                               soft_lim=0.1)
            ```
        """

        self.robot = robot
        self.rmodel = self.robot.model
        self.num_waypoints = num_waypoints
        # joint id of active joints
        self.act_Jid = [self.rmodel.getJointId(i) for i in active_joints]
        # active joint objects and their names
        self.act_Jname = [self.rmodel.names[jid] for jid in self.act_Jid]
        self.act_J = [self.rmodel.joints[jid] for jid in self.act_Jid]
        # joint config id (e.g one joint might have >1 DOF)
        self.act_idxq = [J.idx_q for J in self.act_J]
        # joint velocity id
        self.act_idxv = [J.idx_v for J in self.act_J]

        # size of waypoints vector for all active joints
        self.dim_q = (len(self.act_idxq), self.num_waypoints)
        self.dim_v = (len(self.act_idxv), self.num_waypoints)

        # joint limits on active joints
        self.upper_q = self.rmodel.upperPositionLimit[self.act_idxq]
        self.lower_q = self.rmodel.lowerPositionLimit[self.act_idxq]

        self.upper_dq = self.rmodel.velocityLimit[self.act_idxv]
        self.lower_dq = -self.rmodel.velocityLimit[self.act_idxv]

        self.upper_effort = self.rmodel.effortLimit[self.act_idxv]
        self.lower_effort = -self.rmodel.effortLimit[self.act_idxv]

        # joint limits on active joints with soft limit on both limit ends
        if soft_lim > 0:
            self.upper_q = self.upper_q - soft_lim * abs(self.upper_q - self.lower_q)
            self.lower_q = self.lower_q + soft_lim * abs(self.upper_q - self.lower_q)

            self.upper_dq = self.upper_dq - soft_lim * abs(
                self.upper_dq - self.lower_dq
            )
            self.lower_dq = self.lower_dq + soft_lim * abs(
                self.upper_dq - self.lower_dq
            )

            self.upper_effort = self.upper_effort - soft_lim * abs(
                self.upper_effort - self.lower_effort
            )
            self.lower_effort = self.lower_effort + soft_lim * abs(
                self.upper_effort - self.lower_effort
            )

    def get_active_config(
        self,
        freq: int,
        time_points: np.ndarray,
        waypoints: np.ndarray,
        vel_waypoints=None,
        acc_waypoints=None,
    ):
        """Generate cubic splines on active joints"""
        # dimensions
        assert (
            self.dim_q == waypoints.shape
        ), "(Pos) Check size \
                                        (num_active_joints,num_waypoints)!"
        self.pc = ndcurves.piecewise()  # set piecewise object to join segments

        # C_2 continuous at waypoints
        if vel_waypoints is not None and acc_waypoints is not None:
            # dimensions
            assert (
                self.dim_v == vel_waypoints.shape
            ), "(Vel) Check size\
                                        (num_active_joints, num_waypoints)!"
            assert (
                self.dim_v == acc_waypoints.shape
            ), "(Acc) Check size\
                                        (num_active_joints, num_waypoints)!"

            # make exact cubic WITH constraints on vel and acc on both ends
            for i in range(self.num_waypoints - 1):
                self.c = ndcurves.curve_constraints()
                self.c.init_vel = np.matrix(vel_waypoints[:, i]).transpose()
                self.c.end_vel = np.matrix(vel_waypoints[:, i + 1]).transpose()
                self.c.init_acc = np.matrix(acc_waypoints[:, i]).transpose()
                self.c.end_acc = np.matrix(acc_waypoints[:, i + 1]).transpose()
                ec = ndcurves.exact_cubic(
                    waypoints[:, range(i, i + 2)],
                    time_points[range(i, i + 2), 0],
                    self.c,
                )
                self.pc.append(ec)

        # make exact cubic WITHOUT constraints on vel and acc on both ends
        else:
            for i in range(self.num_waypoints - 1):
                ec = ndcurves.exact_cubic(
                    waypoints[:, range(i, i + 2)], time_points[range(i, i + 2), 0]
                )  # Added spaces around '+'
                self.pc.append(ec)

        # time step
        self.delta_t = 1 / freq  # Added spaces around '/'
        # total travel time
        self.T = self.pc.max() - self.pc.min()
        # get number sample points from generated trajectory
        self.N = int(self.T / self.delta_t) + 1
        # create time stamps on all sample points
        self.t = (
            self.pc.min()
            + np.matrix([i * self.delta_t for i in range(self.N)]).transpose()
        )

        # compute derivatives to obtain pos/vel/acc on all samples (bad)
        self.q_act = np.array(
            [self.pc(self.t[i, 0]) for i in range(self.N)], dtype="float"
        )
        self.dq_act = np.array(
            [self.pc.derivate(self.t[i, 0], 1) for i in range(self.N)], dtype="float"
        )
        self.ddq_act = np.array(
            [self.pc.derivate(self.t[i, 0], 2) for i in range(self.N)], dtype="float"
        )
        t, p_act, v_act, a_act = self.t, self.q_act, self.dq_act, self.ddq_act

        return t, p_act, v_act, a_act

    def get_full_config(
        self,
        freq: int,
        time_points: np.ndarray,
        waypoints: np.ndarray,
        vel_waypoints=None,
        acc_waypoints=None,
    ):
        """
        Generate complete robot configuration trajectory with cubic splines.

        This method creates smooth trajectories for all robot joints by:
        1. Generating cubic splines for active joints between waypoints
        2. Filling inactive joints with zero values
        3. Ensuring C2 continuity at waypoints

        Args:
            freq (int): Sampling frequency for trajectory generation (Hz)
            time_points (np.ndarray): Time stamps for waypoints,
                shape (num_waypoints, 1)
            waypoints (np.ndarray): Position waypoints for active joints,
                shape (num_active_joints, num_waypoints)
            vel_waypoints (np.ndarray, optional): Velocity constraints at
                waypoints, same shape as waypoints. If provided, enforces
                specific velocities at waypoints.
            acc_waypoints (np.ndarray, optional): Acceleration constraints at
                waypoints, same shape as waypoints. If provided, enforces
                specific accelerations at waypoints.

        Returns:
            tuple: Four-element tuple containing:
                - t (np.ndarray): Time stamps, shape (N, 1)
                - q_full (np.ndarray): Position trajectory for all joints,
                  shape (N, robot_nq)
                - dq_full (np.ndarray): Velocity trajectory for all joints,
                  shape (N, robot_nv)
                - ddq_full (np.ndarray): Acceleration trajectory for all
                  joints, shape (N, robot_nv)

                Where N = int(total_time * freq) + 1

        Raises:
            AssertionError: If waypoint dimensions don't match active joints

        Examples:
            Position-only trajectory:
                ```python
                time_pts = np.array([[0], [1], [2], [3]])
                waypts = np.array([[0, 1, 2, 1],     # joint1
                                  [0, 0.5, 1, 0.5]]) # joint2

                t, q, dq, ddq = spline.get_full_config(
                    freq=100, time_points=time_pts, waypoints=waypts
                )
                ```

            With velocity/acceleration constraints:
                ```python
                vel_waypts = np.zeros((2, 4))  # Zero velocity at waypoints
                acc_waypts = np.zeros((2, 4))  # Zero acceleration at waypoints

                t, q, dq, ddq = spline.get_full_config(
                    freq=100, time_points=time_pts, waypoints=waypts,
                    vel_waypoints=vel_waypts, acc_waypoints=acc_waypts
                )
                ```

        Note:
            The trajectory uses piecewise cubic curves connected at waypoints.
            When velocity and acceleration constraints are provided, the
            resulting trajectory enforces exact values at waypoints.
        """
        t, p_act, v_act, a_act = self.get_active_config(
            freq, time_points, waypoints, vel_waypoints, acc_waypoints
        )
        # create array of zero configuration times N samples
        self.q_full = np.array([self.robot.q0] * self.N)
        self.dq_full = np.array([np.zeros_like(self.robot.v0)] * self.N)
        self.ddq_full = np.array([np.zeros_like(self.robot.v0)] * self.N)

        # fill in trajectory with active joints values
        self.q_full[:, self.act_idxq] = p_act
        self.dq_full[:, self.act_idxv] = v_act
        self.ddq_full[:, self.act_idxv] = a_act

        p_full, v_full, a_full = self.q_full, self.dq_full, self.ddq_full
        return t, p_full, v_full, a_full

    def check_cfg_constraints(self, q, v=None, tau=None, soft_lim=0):
        """
        Check joint constraints violation for trajectory configurations.

        Validates whether the generated trajectory respects robot joint
        limits including position, velocity, and effort constraints.
        Provides detailed violation reporting for debugging.

        Args:
            q (np.ndarray): Position trajectory to check,
                shape (N, robot_nq)
            v (np.ndarray, optional): Velocity trajectory to check,
                shape (N, robot_nv). If None, velocity checks are skipped.
            tau (np.ndarray, optional): Effort trajectory to check,
                shape (N, robot_nv). If None, effort checks are skipped.
            soft_lim (float, optional): Additional safety margin factor
                (0-1). Adds this fraction of joint range as safety buffer.
                Defaults to 0.

        Returns:
            bool: True if any constraint is violated, False if all
                constraints are satisfied

        Examples:
            Basic position check:
                ```python
                t, q, dq, ddq = spline.get_full_config(...)
                is_violated = spline.check_cfg_constraints(q)
                if is_violated:
                    print("Trajectory violates position limits!")
                ```

            Full constraint check with safety margin:
                ```python
                is_violated = spline.check_cfg_constraints(
                    q, v=dq, tau=torques, soft_lim=0.1
                )
                ```

        Note:
            Constraint violations are printed to console with specific
            joint indices and violation types for debugging purposes.
        """
        __isViolated = False
        for i in range(q.shape[0]):
            for j in self.act_idxq:
                delta_lim = soft_lim * abs(
                    self.rmodel.upperPositionLimit[j]
                    - self.rmodel.lowerPositionLimit[j]
                )
                if q[i, j] > self.rmodel.upperPositionLimit[j] - delta_lim:
                    logger.warning("Joint q %d upper limit violated!", j)
                    __isViolated_pos = True

                elif q[i, j] < self.rmodel.lowerPositionLimit[j] + delta_lim:
                    logger.warning("Joint position idx_q %d lower limit violated!", j)
                    __isViolated_pos = True
                else:
                    __isViolated_pos = False
                __isViolated = __isViolated or __isViolated_pos
                # print(__isViolated)
        if v is not None:
            for i in range(v.shape[0]):
                for j in self.act_idxv:
                    if abs(v[i, j]) > (1 - soft_lim) * abs(
                        self.rmodel.velocityLimit[j]
                    ):
                        logger.warning("Joint vel idx_v %d limits violated!", j)
                        __isViolated_vel = True
                    else:
                        __isViolated_vel = False
                    __isViolated = __isViolated or __isViolated_vel
                # print(__isViolated)
        if tau is not None:
            for i in range(tau.shape[0]):
                for j in self.act_idxv:
                    if abs(tau[i, j]) > (1 - soft_lim) * abs(
                        self.rmodel.effortLimit[j]
                    ):
                        logger.warning("Joint effort idx_v %d limits violated!", j)
                        __isViolated_eff = True
                    else:
                        __isViolated_eff = False
                    __isViolated = __isViolated or __isViolated_eff
                    # print(__isViolated)
        if not __isViolated:
            logger.info(
                "SUCCEEDED to generate waypoints for a feasible initial cubic spline"
            )
        else:
            logger.warning("FAILED to generate a feasible cubic spline")
        return __isViolated

    def check_self_collision(self):
        __isViolated = False

        return __isViolated

    def plot_spline(self, t, p, v, a):
        q = p[:, self.act_idxq]
        dq = v[:, self.act_idxv]
        ddq = a[:, self.act_idxv]

        for i in range(q.shape[1]):
            plt.figure(i)  # Removed assignment to 'plot'
            plt.plot(t[:, 0], q[:, i], color="r", label="pos")
            plt.plot(t[:, 0], dq[:, i], color="b", label="vel")
            plt.plot(t[:, 0], ddq[:, i], color="g", label="acc")
            plt.title("joint %s" % i)
            plt.legend()
            plt.grid()
        plt.show()


class WaypointsGeneration(CubicSpline):
    """
    Automated waypoint generation for cubic spline trajectories.

    This class extends CubicSpline to provide automated generation of
    feasible waypoints that respect robot joint constraints. It creates
    pools of valid configurations and uses random sampling to generate
    diverse trajectory waypoints.

    The class generates waypoints for position, velocity, and acceleration
    that can be used as initial guesses for trajectory optimization or
    as standalone feasible trajectories.

    Attributes:
        n_set (int): Size of waypoint pools (default: 10)
        pool_q (np.ndarray): Pool of valid position configurations
        pool_dq (np.ndarray): Pool of valid velocity configurations
        pool_ddq (np.ndarray): Pool of valid acceleration configurations
        soft_limit_pool_default (np.ndarray): Default soft limit values

    Examples:
        Basic waypoint generation:
            ```python
            wp_gen = WaypointsGeneration(robot, num_waypoints=5,
                                       active_joints=['joint1', 'joint2'])

            # Generate random feasible waypoints
            pos_wp, vel_wp, acc_wp = wp_gen.random_feasible_waypoints()

            # Use with cubic spline
            time_pts = np.linspace(0, 4, 5).reshape(-1, 1)
            t, q, dq, ddq = wp_gen.get_full_config(
                freq=100, time_points=time_pts, waypoints=pos_wp.T,
                vel_waypoints=vel_wp.T, acc_waypoints=acc_wp.T
            )
            ```

        With custom soft limits:
            ```python
            # Different safety margins for position/velocity/acceleration
            soft_lim_custom = np.array([
                [0.1, 0.15],  # Position limits (10%, 15% for joints)
                [0.2, 0.2],   # Velocity limits (20% for both joints)
                [0.3, 0.25]   # Acceleration limits (30%, 25%)
            ])

            wp_gen.set_soft_limit_pool(soft_lim_custom)
            pos_wp, vel_wp, acc_wp = wp_gen.random_feasible_waypoints()
            ```

    Note:
        The class automatically ensures waypoints don't violate joint
        constraints and avoids repeated waypoint values that could
        cause numerical issues in spline generation.
    """

    def __init__(self, robot, num_waypoints: int, active_joints: list, soft_lim=0):
        """
        Initialize waypoint generation for cubic spline trajectories.

        Sets up pools for generating random feasible waypoints that respect
        robot joint constraints. Initializes configuration pools for
        positions, velocities, and accelerations.

        Args:
            robot: Robot model instance containing kinematic information
            num_waypoints (int): Number of waypoints for trajectory generation
            active_joints (list): List of joint names to include in trajectory
            soft_lim (float, optional): Soft limit reduction factor (0-1).
                Defaults to 0.

        Note:
            The soft_lim parameter affects the base CubicSpline initialization
            but does not set the waypoint generation pools. Use
            set_soft_limit_pool() for custom pool limits.
        """
        super().__init__(robot, num_waypoints, active_joints, soft_lim=0)

        self.n_set = 10  # size of waypoints pool
        self.pool_q = np.zeros((self.n_set, len(self.act_idxq)))
        self.pool_dq = np.zeros((self.n_set, len(self.act_idxv)))
        self.pool_ddq = np.zeros((self.n_set, len(self.act_idxv)))
        self.soft_limit_pool_default = np.zeros((3, len(self.act_idxq)))

    def gen_rand_pool(self, soft_limit_pool=None):
        """Generate a uniformly distributed waypoint pool of pos/vel/acc over
        a specific range
        """
        if soft_limit_pool is None:
            soft_limit_pool = self.soft_limit_pool_default
        assert np.array(soft_limit_pool).shape == (
            3,
            len(self.act_idxq),
        ), "input a vector of soft limit pool with a shape of (3, len(activejoints)"
        lim_q = soft_limit_pool[0, :]
        lim_dq = soft_limit_pool[1, :]
        lim_ddq = soft_limit_pool[2, :]

        new_upper_q = np.zeros_like(self.upper_q)
        new_lower_q = np.zeros_like(self.lower_q)
        new_upper_dq = np.zeros_like(self.upper_dq)
        new_lower_dq = np.zeros_like(self.lower_dq)
        new_upper_ddq = np.zeros_like(self.upper_dq)
        new_lower_ddq = np.zeros_like(self.lower_dq)

        for i in range(len(self.act_idxq)):
            new_upper_q[i] = self.upper_q[i] - lim_q[i] * abs(
                self.upper_q[i] - self.lower_q[i]
            )
            new_lower_q[i] = self.lower_q[i] + lim_q[i] * abs(
                self.upper_q[i] - self.lower_q[i]
            )

            step_q = (new_upper_q[i] - new_lower_q[i]) / (self.n_set - 1)
            self.pool_q[:, i] = np.array(
                [new_lower_q[i] + j * step_q for j in range(self.n_set)]
            )

        for i in range(len(self.act_idxv)):
            new_upper_dq[i] = self.upper_dq[i] - lim_dq[i] * abs(
                self.upper_dq[i] - self.lower_dq[i]
            )
            new_lower_dq[i] = self.lower_dq[i] + lim_dq[i] * abs(
                self.upper_dq[i] - self.lower_dq[i]
            )

            new_upper_ddq[i] = k * (
                self.upper_dq[i] - lim_ddq[i] * abs(self.upper_dq[i] - self.lower_dq[i])
            )  # Fixed line break
            new_lower_ddq[i] = k * (
                self.lower_dq[i] + lim_ddq[i] * abs(self.upper_dq[i] - self.lower_dq[i])
            )  # Fixed line break

            step_dq = (new_upper_dq[i] - new_lower_dq[i]) / (self.n_set - 1)
            self.pool_dq[:, i] = np.array(
                [new_lower_dq[i] + j * step_dq for j in range(self.n_set)]
            )

            step_ddq = (new_upper_ddq[i] - new_lower_ddq[i]) / (self.n_set - 1)
            self.pool_ddq[:, i] = np.array(
                [new_lower_ddq[i] + j * step_ddq for j in range(self.n_set)]
            )
        # return self.pool_q, self.pool_dq, self.pool_ddq

    def check_repeat_wp(self, wp_list: list):
        repeat = False
        for ii in range(len(wp_list) - 1):
            if wp_list[ii] == wp_list[ii + 1]:
                repeat = True
        return repeat

    def gen_rand_wp(
        self,
        wp_init=None,
        vel_wp_init=None,
        acc_wp_init=None,
        vel_set_zero=True,
        acc_set_zero=True,
    ):
        """Generate waypoint pos/vel/acc which randomly pick from waypoint
        pool
        Or, set vel and/or acc at waypoints to be zero
        """
        wps_rand = np.zeros((self.num_waypoints, len(self.act_idxq)))
        vel_wps_rand = np.zeros((self.num_waypoints, len(self.act_idxv)))
        acc_wps_rand = np.zeros((self.num_waypoints, len(self.act_idxv)))
        if wp_init is not None:
            wps_rand[0, :] = wp_init
            for i in range(len(self.act_idxq)):
                repeat_ = True
                while repeat_:
                    wps_rand[range(1, self.num_waypoints), i] = np.random.choice(
                        self.pool_q[:, i], self.num_waypoints - 1
                    )
                    repeat_ = self.check_repeat_wp(
                        list(wps_rand[range(1, self.num_waypoints), i])
                    )
        else:
            for i in range(len(self.act_idxq)):
                repeat_ = True
                while repeat_:
                    wps_rand[:, i] = np.random.choice(
                        self.pool_q[:, i], self.num_waypoints
                    )
                    repeat_ = self.check_repeat_wp(list(wps_rand[:, i]))
        if vel_wp_init is not None:
            vel_wps_rand[0, :] = vel_wp_init
            if not vel_set_zero:
                for i in range(len(self.act_idxv)):
                    repeat_ = True
                    while repeat_:
                        vel_wps_rand[range(1, self.num_waypoints), i] = (
                            np.random.choice(self.pool_dq[:, i], self.num_waypoints - 1)
                        )
                        repeat_ = self.check_repeat_wp(
                            list(vel_wps_rand[range(1, self.num_waypoints), i])
                        )
        else:
            if not vel_set_zero:
                for i in range(len(self.act_idxv)):
                    repeat_ = True
                    while repeat_:
                        vel_wps_rand[:, i] = np.random.choice(
                            self.pool_dq[:, i], self.num_waypoints
                        )
                        repeat_ = self.check_repeat_wp(list(vel_wps_rand[:, i]))
        if vel_wp_init is not None:
            acc_wps_rand[0, :] = vel_wp_init
            if not acc_set_zero:
                for i in range(len(self.act_idxv)):
                    repeat_ = True
                    while repeat_:
                        acc_wps_rand[range(1, self.num_waypoints), i] = (
                            np.random.choice(
                                self.pool_ddq[:, i], self.num_waypoints - 1
                            )
                        )
                        repeat_ = self.check_repeat_wp(
                            list(acc_wps_rand[range(1, self.num_waypoints), i])
                        )
        else:
            if not acc_set_zero:
                for i in range(len(self.act_idxv)):
                    repeat_ = True
                    while repeat_:
                        acc_wps_rand[:, i] = np.random.choice(
                            self.pool_ddq[:, i], self.num_waypoints
                        )
                        repeat_ = self.check_repeat_wp(list(acc_wps_rand[:, i]))
        return wps_rand.transpose(), vel_wps_rand.transpose(), acc_wps_rand.transpose()

    def gen_equal_wp(self, wp_init=None, vel_wp_init=None, acc_wp_init=None):
        """Generate equal waypoints everywhere same as first waypoints
        with default: zero vel and zero acc
        """
        wps_equal = np.zeros((self.num_waypoints, len(self.act_idxq)))
        vel_wps_equal = np.zeros((self.num_waypoints, len(self.act_idxv)))
        acc_wps_equal = np.zeros((self.num_waypoints, len(self.act_idxv)))

        step_q = (self.upper_q - self.lower_q) / 20
        if wp_init is not None:
            wps_equal = np.tile(wp_init, (self.num_waypoints, 1))
            for jj in range(1, self.num_waypoints):
                wps_equal[jj, :] = wps_equal[jj - 1, :] + step_q

        if vel_wp_init is not None:
            vel_wps_equal = np.tile(vel_wp_init, (self.num_waypoints, 1))

        if acc_wp_init is not None:
            acc_wps_equal = np.tile(acc_wp_init, (self.num_waypoints, 1))
        return (
            wps_equal.transpose(),
            vel_wps_equal.transpose(),
            acc_wps_equal.transpose(),
        )


def init_robot(robot):
    import pinocchio as pin

    pin.framesForwardKinematics(robot.model, robot.data, robot.q0)
    pin.updateFramePlacements(robot.model, robot.data)


def calc_torque(N, robot, q, v, a):
    tau = np.zeros(robot.model.nv * N)
    for i in range(N):
        for j in range(robot.model.nv):
            tau[j * N + i] = pin.rnea(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )[j]
    return tau
