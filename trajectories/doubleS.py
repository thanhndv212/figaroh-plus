import numpy as np
import matplotlib.pyplot as plt


class doubleS:
    def __init__(self, t_s, q0, qf, dq0, dqf, v_max, a_max, j_max):
        # time step
        self.ts = t_s
        # given arbitrarilly
        self.q0 = q0
        self.qf = qf
        self.dq0 = dq0
        self.dqf = dqf
        # assume in all cases to be zeros
        self.dqq0 = 0
        self.ddqf = 0
        # constraints
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.v_min = -v_max
        self.a_min = -a_max
        self.j_min = -j_max
        # run time
        self.T = 0
        self.T_a = 0
        self.T_v = 0
        self.T_d = 0
        self.Tj1 = 0
        self.Tj2 = 0
        # paramters of double S
        self.sigma = 1
        # self.v_lima = 0
        # self.a_lima = 0
        # self.v_limd = 0
        # self.a_limd = 0

    def get_doubleS(self):
        # general double S curves (multiple points)
        # calculat waypoints, vels, accs in seven segments
        ts = self.ts
        T_a = np.round(self.T_a, 3)
        T_d = np.round(self.T_d, 3)
        T_v = np.round(self.T_v, 3)
        Tj1 = np.round(self.Tj1, 3)
        Tj2 = np.round(self.Tj2, 3)
        T = T_a + T_d + T_v
        a_lima = np.round(self.j_max * self.Tj1, 3)
        a_limd = np.round(-self.j_max * self.Tj2, 3)
        v_lima = np.round(self.dq0 + (self.T_a - self.Tj1) * a_lima, 3)
        v_limd = np.round(self.dqf - (self.T_d - self.Tj2) * a_limd, 3)
        N_a = int(T_a / ts)
        N_d = int(T_d / ts)
        N_v = int(T_v / ts)
        N = N_a + N_v + N_d

        Nj1 = int(Tj1 / ts)
        Nj2 = int(Tj2 / ts)
        print("No. of sample points: ", N, N_a, N_d, N_v, Nj1, Nj2)

        # positions, velocities, acceleration, jerks and tine steps.
        q = np.zeros(N)
        dq = np.zeros(N)
        ddq = np.zeros(N)
        d3q = np.zeros(N)
        rt = np.zeros(N)

        # phase 1: acceleration
        for i in range(Nj1):
            t = i * ts
            rt[i] = t
            q[i] = self.q0 + self.dq0 * t + self.j_max * np.power(t, 3) / 6
            dq[i] = self.dq0 + self.j_max * np.power(t, 2) / 2
            ddq[i] = self.j_max * t
            d3q[i] = self.j_max

        for i in range(Nj1, N_a - Nj1):
            t = i * ts
            rt[i] = t
            q[i] = (
                self.q0
                + self.dq0 * t
                + (a_lima / 6) * (3 * np.power(t, 2) - 3 * Tj1 * t + Tj1 * Tj1)
            )
            dq[i] = self.dq0 + a_lima * (t - Tj1 / 2)
            ddq[i] = a_lima
            d3q[i] = 0

        for i in range(N_a - Nj1, N_a):
            t = i * ts
            rt[i] = t
            q[i] = (
                self.q0
                + (v_lima + self.dq0) * T_a / 2
                - v_lima * (T_a - t)
                - (self.j_min / 6) * np.power((T_a - t), 3)
            )
            dq[i] = v_lima + self.j_min * np.power((T_a - t), 2) / 2
            ddq[i] = -self.j_min * (T_a - t)
            d3q[i] = self.j_min

        # phase 2: constant velocity
        for i in range(N_a, N_a + N_v):
            t = i * ts
            rt[i] = t
            q[i] = self.q0 + (v_lima + self.dq0) * T_a / 2 + v_lima * (t - T_a)
            dq[i] = v_lima
            ddq[i] = 0
            d3q[i] = 0

        # phase 3: decelartion
        for i in range(N - N_d, N - N_d + Nj2):
            t = i * ts
            rt[i] = t
            q[i] = (
                self.qf
                - (v_limd + self.dqf) * T_d / 2
                + v_limd * (t - T + T_d)
                - (self.j_max / 6) * np.power((t - T + T_d), 3)
            )
            dq[i] = v_limd - self.j_max * np.power((t - T + T_d), 2) / 2
            ddq[i] = -self.j_max * (t - T + T_d)
            d3q[i] = self.j_min

        for i in range(N - N_d + Nj2, N - Nj2):
            t = i * ts
            rt[i] = t
            q[i] = (
                self.qf
                - (v_limd + self.dqf) * T_d / 2
                + v_limd * (t - T + T_d)
                + (a_limd / 6)
                * (3 * np.power((t - T + T_d), 2) - 3 * Tj2 * (t - T + T_d) + Tj2 * Tj2)
            )
            dq[i] = v_limd + a_limd * (t - T + T_d - Tj2 / 2)
            ddq[i] = a_limd
            d3q[i] = 0

        for i in range(N - Nj2, N):
            t = i * ts
            rt[i] = t
            q[i] = (
                self.qf - (self.dqf) * (T - t) - (self.j_max / 6) * np.power((T - t), 3)
            )
            dq[i] = self.dqf + self.j_max * np.power((T - t), 2) / 2
            ddq[i] = -self.j_max * (T - t)
            d3q[i] = self.j_max

        # segment1
        self.q = q
        self.dq = dq
        self.ddq = ddq
        self.d3q = d3q
        self.rt = rt
        self.finalize()
        return self.rt, self.q, self.dq, self.ddq, self.d3q

    def initilize(self):
        self.sigma = np.sign(self.qf - self.q0)
        sigma = self.sigma
        self.q0 = self.q0 * sigma
        self.qf = self.qf * sigma
        self.dq0 = self.dq0 * sigma
        self.dqf = self.dqf * sigma

        v_max = self.v_max
        a_max = self.a_max
        j_max = self.j_max
        v_min = self.v_min
        a_min = self.a_min
        j_min = self.j_min

        self.v_max = (sigma + 1) * v_max / 2 + (sigma - 1) * v_min / 2
        self.v_min = (sigma + 1) * v_min / 2 + (sigma - 1) * v_min / 2

        self.a_max = (sigma + 1) * a_max / 2 + (sigma - 1) * a_min / 2
        self.a_min = (sigma + 1) * a_min / 2 + (sigma - 1) * a_max / 2

        self.j_max = (sigma + 1) * j_max / 2 + (sigma - 1) * j_min / 2
        self.j_min = (sigma + 1) * j_min / 2 + (sigma - 1) * j_max / 2

    def finalize(self):
        self.q = self.sigma * self.q
        self.dq = self.sigma * self.dq
        self.ddq = self.sigma * self.ddq

        return self.q, self.dq, self.ddq

    def check_feasiable(self):
        # check the feasibility of the profile given a fixed run time, in 02 cases
        # defaullt False
        Tj_lim = np.array(
            [self.a_max / self.j_max, np.sqrt(abs(self.dqf - self.dq0) / self.j_max)]
        )
        Tj_min = Tj_lim.min(axis=0)
        h_lim = np.array(
            [
                (self.dq0 + self.qf) * Tj_min,
                1
                / 2
                * (self.dq0 + self.qf)
                * (Tj_min + abs(self.dqf - self.dq0) / self.a_max),
            ]
        )
        h = self.qf - self.q0
        check_pass1 = True
        check_pass2 = True
        if Tj_min == Tj_lim[0]:
            if h < h_lim[0]:
                check_pass1 = False
                print("Failed at the 1st check")
            else:
                print("pass1")

        elif Tj_min < Tj_lim[0]:
            if h < h_lim[1]:
                check_pass2 = False
                print("Failed at the 2nd check")
            else:
                print("pass2")

        if check_pass1 and check_pass2:
            return True
        else:
            return False

    def check_reached_v_max(self):
        # run check if v_lim can reach v_max
        # default False
        if self.T_v > 0:
            return True
        else:
            return False

        pass

    def check_reached_a_max(self):
        if (self.v_max - self.dq0) * self.j_max < self.a_max * self.a_max:
            return False
        else:
            return True

    def check_reached_a_min(self):
        if (self.v_max - self.dqf) * self.j_max < self.a_min * self.a_min:
            return False
        else:
            return True
        # run check if a_lim can reach a_max

    def getParams_amax_vmax(self):
        reached_a_max = self.check_reached_a_max()
        reached_a_min = self.check_reached_a_min()

        Tj1_lim = np.array(
            [self.a_max / self.j_max, np.sqrt(abs(self.v_max - self.dq0) / self.j_max)]
        )
        Ta_lim = np.array(
            [Tj1_lim[0] + (self.v_max - self.dq0) / self.a_max, 2 * Tj1_lim[1]]
        )

        Tj2_lim = np.array(
            [self.a_max / self.j_max, np.sqrt(abs(self.v_max - self.dqf) / self.j_max)]
        )
        Td_lim = np.array(
            [Tj2_lim[0] + (self.v_max - self.dqf) / self.a_max, 2 * Tj2_lim[1]]
        )
        # assuming v_max are all reached, calculate Ta, Td, Tj1, Tj2 depeniding on cases
        # whether a_max and a_min are reached or not
        if reached_a_max:
            self.Tj1 = Tj1_lim[0]
            self.T_a = Ta_lim[0]
            if reached_a_min:
                self.Tj2 = Tj2_lim[0]
                self.T_d = Td_lim[0]
            else:
                self.Tj2 = Tj2_lim[1]
                self.T_d = Td_lim[1]

        else:
            self.Tj1 = Tj1_lim[1]
            self.T_a = Ta_lim[1]
            if reached_a_min:
                self.Tj2 = Tj2_lim[0]
                self.T_d = Td_lim[0]
            else:
                self.Tj2 = Tj2_lim[1]
                self.T_d = Td_lim[1]
        self.T_v = (
            (self.qf - self.q0) / self.v_max
            - (self.T_a / 2) * (1 + self.dq0 / self.v_max)
            - (self.T_d / 2) * (1 + self.dqf / self.v_max)
        )
        return self.T_a, self.T_d, self.T_v, self.Tj1, self.Tj2

    def getParams_amax_only(self, a_max):
        # v-max is not reached, a_max is reached, meaining no constant velocity phase
        # Tv = 0, but assuming 2 phases acceleration and deceleration
        self.Tj1 = a_max / self.j_max
        self.Tj2 = a_max / self.j_max
        delta = (
            np.power(a_max, 4) / np.power(self.j_max, 2)
            + 2 * (np.power(self.dq0, 2) + np.power(self.dqf, 2))
            + a_max
            * (
                4 * (self.qf - self.q0)
                - 2 * (a_max / self.j_max) * (self.dq0 + self.dqf)
            )
        )
        self.T_a = (1 / (2 * a_max)) * (
            np.power(a_max, 2) / self.j_max - 2 * self.dq0 + np.sqrt(delta)
        )
        self.T_d = (1 / (2 * a_max)) * (
            np.power(a_max, 2) / self.j_max - 2 * self.dqf + np.sqrt(delta)
        )

        return self.T_a, self.T_d, self.Tj1, self.Tj2

    def getParams_onephase(self, T_a, T_d, a_max):
        # v_max is not reached, a_max is either reached or not, but only one phase is
        # either acceleration or deceleration, depending whether Ta <0 or Td < 0
        if T_a == 0:
            self.T_a = 0
            self.Tj1 = 0
            if self.dq0 < self.dqf:
                print("v0 is supposed to be larger than v1")
            else:
                self.T_d = 2 * (self.qf - self.q0) / (self.dqf + self.dq0)
                beta = self.j_max * (
                    self.j_max * np.power(self.qf - self.q0, 2)
                    + np.power(self.dqf + self.dq0, 2) * (self.dqf - self.dq0)
                )
                self.Tj2 = (self.j_max * (self.qf - self.q0) - np.sqrt(beta)) / (
                    self.j_max * (self.dqf + self.dq0)
                )
        elif T_d == 0:
            self.T_d = 0
            self.Tj2 = 0
            if self.dq0 > self.dqf:
                print("v0 is supposed to be smaller than v1")
            else:
                self.Ta = 2 * (self.qf - self.q0) / (self.dqf + self.dq0)
                beta = self.j_max * (
                    self.j_max * np.power((self.qf - self.q0), 2)
                    - np.power((self.dqf + self.dq0), 2) * (self.dqf - self.dq0)
                )
                self.Tj1 = (self.j_max * (self.qf - self.q0) - np.sqrt(beta)) / (
                    self.j_max * (self.dqf + self.dq0)
                )
        else:
            print("error, either Ta or Td supposes to be possitive")

    def getCoeff_PTP(self):
        # get coeff in  04 cases
        # self.initilize()
        T_a, T_d, T_v, Tj1, Tj2 = self.getParams_amax_vmax()
        if T_v > 0:
            self.T_a = T_a
            self.T_d = T_d
            self.T_v = T_v
            self.Tj1 = Tj1
            self.Tj2 = Tj2
            # return T_a, T_d, T_v, Tj1, Tj2
        else:
            self.T_v = 0
            both_reached = False
            a_max_recursive = self.a_max
            gamma = 0.95
            while not both_reached:
                T_a, T_d, Tj1, Tj2 = self.getParams_amax_only(a_max_recursive)
                if T_a <= 0:
                    T_a = 0
                    self.getParams_onephase(T_a, T_d, a_max_recursive)
                    break
                elif T_d <= 0:
                    T_d = 0
                    self.getParams_onephase(T_a, T_d, a_max_recursive)
                    break
                else:
                    if T_a > 2 * Tj1 and T_d > 2 * Tj2:
                        both_reached = True
                    else:
                        both_reached = False
                a_max_recursive = gamma * a_max_recursive

        print(
            "T_a, T_d, T_v, Tj1, Tj2: ",
            self.T_a,
            self.T_d,
            self.T_v,
            self.Tj1,
            self.Tj2,
        )

    def plot_doubleS(self, Scurve_color=None, fig=None):
        # if Scurve_color == None:
        # 	Scurve_color = 'blue'
        # if axs.shape[0] == 0:
        axs = []
        for i in range(4):
            ind = i + 1
            axs.append(fig.add_subplot(4, 1, ind))
        axs[0].plot(self.rt, self.q, color=Scurve_color)
        axs[0].set_ylabel("Positions, q")
        axs[0].grid()

        axs[1].plot(self.rt, self.dq, color=Scurve_color)
        axs[1].set_ylabel("Velocities, dq")
        axs[1].axhline(y=self.v_max, color="black", linestyle="dashed")
        axs[1].axhline(y=self.v_min, color="black", linestyle="dashed")
        axs[1].grid()

        axs[2].plot(self.rt, self.ddq, color=Scurve_color)
        axs[2].set_ylabel("Accelerations, dqq")
        axs[2].axhline(y=self.a_max, color="black", linestyle="dashed")
        axs[2].axhline(y=self.a_min, color="black", linestyle="dashed")
        axs[2].grid()

        axs[3].plot(self.rt, self.d3q, color=Scurve_color)
        axs[3].set_ylabel("Jerks, d3q")
        axs[3].set_xlabel("Time (s)")
        axs[3].axhline(y=self.j_max, color="black", linestyle="dashed")
        axs[3].axhline(y=self.j_min, color="black", linestyle="dashed")
        axs[3].grid()
        # plt.show(block=True)


def main():
    fig1 = plt.figure(figsize=(6, 12))
    isFeasible = False
    q0 = 0  # initial position
    qf = 0.35  # final position
    dq0 = 0  # initial velocity
    v_max = 0.07 / 5  # maximum velocity
    dqf = 0  # maximum velocity
    a_max = v_max * 0.5  # maximum acceleration
    j_max = a_max * 2  # maximum jerk
    t_s = 0.001  # time step

    # initiate a double S curve planner object
    traj = doubleS(t_s, q0, qf, dq0, dqf, v_max, a_max, j_max)

    # initialize defined conditions
    traj.initilize()

    # check feasibility of defined conditions
    isFeasible = traj.check_feasiable()
    if isFeasible:
        traj.getCoeff_PTP()
        traj.get_doubleS()
        traj.plot_doubleS("blue", fig1)
        print("double S curves is ready!")
    else:
        print("not possible to create a double S velocity profile given configuration!")
    plt.show(block=True)


if __name__ == "__main__":
    main()
