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


def generate_waypoints(N, robot, mlow, mhigh):
    """Generate random values for joint positions, velocities, accelerations.

    Args:
        N: Number of samples
        robot: Robot model object
        mlow: Lower bound for random values
        mhigh: Upper bound for random values

    Returns:
        tuple: (q, v, a) joint position, velocity, acceleration arrays
    """
    q = np.empty((1, robot.model.nq))
    v = np.empty((1, robot.model.nv))
    a = np.empty((1, robot.model.nv))
    for i in range(N):
        q = np.vstack((
            q,
            np.random.uniform(low=mlow, high=mhigh, size=(robot.model.nq,))
        ))
        v = np.vstack((
            v,
            np.random.uniform(low=mlow, high=mhigh, size=(robot.model.nv,))
        ))
        a = np.vstack((
            a,
            np.random.uniform(low=mlow, high=mhigh, size=(robot.model.nv,))
        ))
    return q, v, a


# TODO: generalize determine number of joints after the base link
def generate_waypoints_fext(N, robot, mlow, mhigh):
    """Generate random values for joints with external forces.

    Args:
        N: Number of samples
        robot: Robot model object
        mlow: Lower bound for random values
        mhigh: Upper bound for random values

    Returns:
        tuple: (q, v, a) joint position, velocity, acceleration arrays
    """
    nq = robot.model.nq
    nv = robot.model.nv
    q0 = robot.q0[:7]
    v0 = np.zeros(6)
    a0 = np.zeros(6)
    q = np.empty((1, nq))
    v = np.empty((1, nv))
    a = np.empty((1, nv))
    for i in range(N):
        q_ = np.append(
            q0,
            np.random.uniform(low=mlow, high=mhigh, size=(nq - 7,))
        )
        q = np.vstack((q, q_))

        v_ = np.append(
            v0,
            np.random.uniform(low=mlow, high=mhigh, size=(nv - 6,))
        )
        v = np.vstack((v, v_))

        a_ = np.append(
            a0,
            np.random.uniform(low=mlow, high=mhigh, size=(nv - 6,))
        )
        a = np.vstack((a, a_))
    return q, v, a


def get_torque_rand(N, robot, q, v, a, param):
    """Calculate random torque values including various dynamic effects.

    Args:
        N: Number of samples
        robot: Robot model object
        q: Joint positions array
        v: Joint velocities array
        a: Joint accelerations array
        param: Dictionary of dynamic parameters

    Returns:
        array: Joint torques array
    """
    tau = np.zeros(robot.model.nv * N)
    for i in range(N):
        for j in range(robot.model.nv):
            tau[j * N + i] = pin.rnea(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :])[j]
    if param["has_friction"]:
        for i in range(N):
            for j in range(robot.model.nv):
                fv_term = v[i, j] * param["fv"][j]
                fs_term = np.sign(v[i, j]) * param["fs"][j]
                tau[j * N + i] += fv_term + fs_term
    if param["has_actuator_inertia"]:
        for i in range(N):
            for j in range(robot.model.nv):
                tau[j * N + i] += param["Ia"][j] * a[i, j]
    if param["has_joint_offset"]:
        for i in range(N):
            for j in range(robot.model.nv):
                tau[j * N + i] += param["off"][j]
    if param["has_coupled_wrist"]:
        for i in range(N):
            for j in range(robot.model.nv):
                if j == robot.model.nv - 2:
                    tau[j * N + i] += (
                        param["Iam6"] * v[i, robot.model.nv - 1]
                        + param["fvm6"] * v[i, robot.model.nv - 1]
                        + param["fsm6"]
                        * np.sign(
                            v[i, robot.model.nv - 2] + v[i, robot.model.nv - 1]
                        )
                    )
                if j == robot.model.nv - 1:
                    tau[j * N + i] += (
                        param["Iam6"] * v[i, robot.model.nv - 2]
                        + param["fvm6"] * v[i, robot.model.nv - 2]
                        + param["fsm6"]
                        * np.sign(
                            v[i, robot.model.nv - 2] + v[i, robot.model.nv - 1]
                        )
                    )
    return tau


# TODO: finish complete identification on any robot
