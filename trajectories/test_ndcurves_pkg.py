import numpy as np
from matplotlib import pyplot as plt
from ndcurves import piecewise, cubic_hermite_spline
import trapezoidal
import eigenpy

eigenpy.switchToNumpyArray()


njoints = 3
nwaypoints = 4
acc = np.random.uniform(-0.2, 0.2, size=(nwaypoints - 1, njoints))
delta_t1 = np.random.randint(100, 300, size=(nwaypoints - 1, njoints))
delta_t2 = np.random.randint(300, 400, size=(nwaypoints - 1, njoints))
time_points = np.array([[0.0], [10.0], [20.0], [30.0]])
Nf = np.full(nwaypoints - 1, 1000, dtype=int)
traj = trapezoidal.Trapezoidal(
    njoints=njoints,
    nwaypoints=nwaypoints,
    acc=acc,
    delta_t1=delta_t1,
    delta_t2=delta_t2,
    Nf=Nf,
)
q, qd, qdd = traj.Trapezoidal()

points = np.array([[q[0][i * 1000] for i in range(nwaypoints)]])


##################################################

# N = 7
# points = np.array(np.random.rand(1,N))
points_derivative = np.array(np.random.rand(1, nwaypoints))
points_second_derivative = np.array(np.zeros((1, nwaypoints)))
# time_points = np.array(np.random.rand(1, N)).T
time_points.sort(0)
step = 3000
print(points)


# piece wise from list polynomial with 2nd_derivatives = zeros[]
def pw_polynomial():
    pol = piecewise.FromPointsList(
        points, points_derivative, points_second_derivative, time_points
    )
    waypoints = np.array(
        [
            (
                i / step * (pol.max() - pol.min()) + pol.min(),
                pol(i / step * (pol.max() - pol.min()) + pol.min())[0],
            )
            for i in range(step + 1)
        ]
    )
    vels = np.array(
        [
            (
                i / step * (pol.max() - pol.min()) + pol.min(),
                pol.derivate(i / step * (pol.max() - pol.min()) + pol.min(), 1),
            )
            for i in range(step + 1)
        ],
        dtype="float",
    )
    accs = np.array(
        [
            (
                i / step * (pol.max() - pol.min()) + pol.min(),
                pol.derivate(i / step * (pol.max() - pol.min()) + pol.min(), 2),
            )
            for i in range(step + 1)
        ],
        dtype="float",
    )

    return waypoints, vels, accs


# piece wise with cubic hermite curve
def pw_chs():
    pol = piecewise()

    for i in range(nwaypoints - 1):
        p = np.array([points[0][i : (i + 2)]])
        v = np.array([points_derivative[0][i : (i + 2)]])
        t = time_points[i : (i + 2)]
        print(p)
        # print(v)
        # print(t)
        a = cubic_hermite_spline(p, v, t)
        pol.append(a)
    waypoints = np.array(
        [
            (
                i / step * (pol.max() - pol.min()) + pol.min(),
                pol(i / step * (pol.max() - pol.min()) + pol.min())[0],
            )
            for i in range(step + 1)
        ]
    )
    vels = np.array(
        [
            (
                i / step * (pol.max() - pol.min()) + pol.min(),
                pol.derivate(i / step * (pol.max() - pol.min()) + pol.min(), 1),
            )
            for i in range(step + 1)
        ],
        dtype="float",
    )
    accs = np.array(
        [
            (
                i / step * (pol.max() - pol.min()) + pol.min(),
                pol.derivate(i / step * (pol.max() - pol.min()) + pol.min(), 2),
            )
            for i in range(step + 1)
        ],
        dtype="float",
    )
    return waypoints, vels, accs


def trap():
    pass


waypoints1, vels1, accs1 = pw_chs()
waypoints, vels, accs = pw_polynomial()
fig = plt.figure()
axs = fig.subplots(3, 1)

axs[0].plot(waypoints[:, 0], waypoints[:, 1], color="b", label="piecewise_polynomial")
axs[0].plot(waypoints1[:, 0], waypoints1[:, 1], color="g", label="cubic spline")
axs[0].plot(waypoints[:, 0], q[0][:], color="r", label="trapezoidal")
axs[0].legend(loc="upper left")
axs[0].set_ylabel("q")
axs[0].scatter(time_points, points)
axs[0].grid()

axs[1].plot(vels[:, 0], vels[:, 1], color="b", label="piecewise_polynomial")
axs[1].plot(vels1[:, 0], vels1[:, 1], color="g", label="cubic spline")
axs[1].plot(waypoints[:, 0], qd[0][:], color="r", label="trapezoidal")

axs[1].set_ylabel("dq")
axs[1].scatter(time_points, points_derivative)
axs[1].grid()

axs[2].plot(accs[:, 0], accs[:, 1], color="b", label="piecewise_polynomial")
axs[2].plot(accs1[:, 0], accs1[:, 1], color="g", label="cubic spline")
axs[2].plot(waypoints[:, 0], qdd[0][:], color="r", label="trapezoidal")

axs[2].set_ylabel("ddq")
axs[2].scatter(time_points, points_second_derivative)
axs[2].grid()
axs[2].set_xlabel("time")

plt.show(block=True)
