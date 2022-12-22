import pinocchio as pin
import numpy as np


def generate_waypoints(N, robot, mlow, mhigh):
    """This function generates N random values for joints' position,velocity, acceleration.
    Input:  N: number of samples
                    nq: length of q, nv : length of v
                    mlow and mhigh: the bound for random function
    Output: q, v, a: joint's position, velocity, acceleration"""
    q = np.empty((1, robot.model.nq))
    v = np.empty((1, robot.model.nv))
    a = np.empty((1, robot.model.nv))
    for i in range(N):
        q = np.vstack(
            (q, np.random.uniform(low=mlow, high=mhigh, size=(robot.model.nq,)))
        )
        v = np.vstack(
            (v, np.random.uniform(low=mlow, high=mhigh, size=(robot.model.nv,)))
        )
        a = np.vstack(
            (a, np.random.uniform(low=mlow, high=mhigh, size=(robot.model.nv,)))
        )
    return q, v, a


# TODO: generalize determine number of joints after the base link
def generate_waypoints_fext(N, robot, mlow, mhigh):
    """This function generates N random values for joints' position,velocity, acceleration.
    Input:  N: number of samples
                    nq: length of q, nv : length of v
                    mlow and mhigh: the bound for random function
    Output: q, v, a: joint's position, velocity, acceleration"""
    nq = robot.model.nq
    nv = robot.model.nv
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


def get_torque_rand(N, robot, q, v, a):
    tau = np.zeros(robot.model.nv * N)
    for i in range(N):
        for j in range(robot.model.nv):
            tau[j * N + i] = pin.rnea(
                robot.model, robot.data, q[i, :], v[i, :], a[i, :]
            )[j]
    if robot.isFrictionincld:
        for i in range(N):
            for j in range(robot.model.nv):
                tau[j * N + i] += v[i, j] * robot.fv[j] + np.sign(v[i, j]) * robot.fs[j]
    if robot.isActuator_inertia:
        for i in range(N):
            for j in range(robot.model.nv):
                tau[j * N + i] += robot.Ia[j] * a[i, j]
    if robot.isOffset:
        for i in range(N):
            for j in range(robot.model.nv):
                tau[j * N + i] += robot.off[j]
    if robot.isCoupling:

        for i in range(N):
            for j in range(robot.model.nv):
                if j == robot.model.nv - 2:
                    tau[j * N + i] += (
                        robot.Iam6 * v[i, robot.model.nv - 1]
                        + robot.fvm6 * v[i, robot.model.nv - 1]
                        + robot.fsm6
                        * np.sign(v[i, robot.model.nv - 2] + v[i, robot.model.nv - 1])
                    )
                if j == robot.model.nv - 1:
                    tau[j * N + i] += (
                        robot.Iam6 * v[i, robot.model.nv - 2]
                        + robot.fvm6 * v[i, robot.model.nv - 2]
                        + robot.fsm6
                        * np.sign(v[i, robot.model.nv - 2] + v[i, robot.model.nv - 1])
                    )
    return tau


# TODO: finsish complete identification on any robot
