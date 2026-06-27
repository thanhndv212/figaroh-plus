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
    has_backend = hasattr(robot, "backend")
    nq = robot.backend.nq if has_backend else robot.model.nq
    nv = robot.backend.nv if has_backend else robot.model.nv
    q = np.empty((1, nq))
    v = np.empty((1, nv))
    a = np.empty((1, nv))
    for i in range(N):
        q = np.vstack((q, np.random.uniform(low=mlow, high=mhigh, size=(nq,))))
        v = np.vstack((v, np.random.uniform(low=mlow, high=mhigh, size=(nv,))))
        a = np.vstack((a, np.random.uniform(low=mlow, high=mhigh, size=(nv,))))
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
    has_backend = hasattr(robot, "backend")
    nq = robot.backend.nq if has_backend else robot.model.nq
    nv = robot.backend.nv if has_backend else robot.model.nv
    q0 = robot.q0[:7]
    v0 = np.zeros(6)
    a0 = np.zeros(6)
    q = np.empty((1, nq))
    v = np.empty((1, nv))
    a = np.empty((1, nv))
    for i in range(N):
        q_ = np.append(q0, np.random.uniform(low=mlow, high=mhigh, size=(nq - 7,)))
        q = np.vstack((q, q_))

        v_ = np.append(v0, np.random.uniform(low=mlow, high=mhigh, size=(nv - 6,)))
        v = np.vstack((v, v_))

        a_ = np.append(a0, np.random.uniform(low=mlow, high=mhigh, size=(nv - 6,)))
        a = np.vstack((a, a_))
    return q, v, a


def get_torque_rand(N, robot, q, v, a, identif_config):
    """Calculate random torque values including various dynamic effects.

    Args:
        N: Number of samples
        robot: Robot model object
        q: Joint positions array
        v: Joint velocities array
        a: Joint accelerations array
        identif_config: Dictionary of dynamic parameters

    Returns:
        array: Joint torques array
    """
    has_backend = hasattr(robot, "backend")
    nv = robot.backend.nv if has_backend else robot.model.nv
    tau = np.zeros(nv * N)
    for i in range(N):
        if has_backend:
            tau_vec = robot.backend.compute_inverse_dynamics(q[i, :], v[i, :], a[i, :])
        else:
            tau_vec = pin.rnea(robot.model, robot.data, q[i, :], v[i, :], a[i, :])
        for j in range(nv):
            tau[j * N + i] = tau_vec[j]
    if identif_config["has_friction"]:
        for i in range(N):
            for j in range(nv):
                fv_term = v[i, j] * identif_config["fv"][j]
                fs_term = np.sign(v[i, j]) * identif_config["fs"][j]
                tau[j * N + i] += fv_term + fs_term
    if identif_config["has_actuator_inertia"]:
        for i in range(N):
            for j in range(nv):
                tau[j * N + i] += identif_config["Ia"][j] * a[i, j]
    if identif_config["has_joint_offset"]:
        for i in range(N):
            for j in range(nv):
                tau[j * N + i] += identif_config["off"][j]
    if identif_config["has_coupled_wrist"]:
        for i in range(N):
            for j in range(nv):
                if j == nv - 2:
                    tau[j * N + i] += (
                        identif_config["Iam6"] * v[i, nv - 1]
                        + identif_config["fvm6"] * v[i, nv - 1]
                        + identif_config["fsm6"] * np.sign(v[i, nv - 2] + v[i, nv - 1])
                    )
                if j == nv - 1:
                    tau[j * N + i] += (
                        identif_config["Iam6"] * v[i, nv - 2]
                        + identif_config["fvm6"] * v[i, nv - 2]
                        + identif_config["fsm6"] * np.sign(v[i, nv - 2] + v[i, nv - 1])
                    )
    return tau


# TODO: finish complete identification on any robot
