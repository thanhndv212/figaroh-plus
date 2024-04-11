import numpy as np
from os.path import dirname, join, abspath, isfile
from matplotlib import pyplot as plt
import pinocchio as pin
from tiago_utils.tiago_tools import load_robot
from figaroh.calibration.calibration_tools import rank_in_configuration
from tiago_utils.backlash.calib_inspect import get_q_arm, calc_vel_acc
from scipy.optimize import least_squares
from tiago_utils.backlash.polynomial_fitting import create_poly
import plotly.graph_objects as go
from tiago_utils.backlash.polynomial_fitting import SurfaceFitting

rad_deg = 180 / np.pi

params = {
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.labelpad": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(params)


class InspectJoint:
    def __init__(self, tiago, joint_name, folder, sample_range):
        self.tiago = tiago
        self.folder = folder
        self.joint_inspected = joint_name
        self.load_inspection()
        # self.compute_entities(sample_range)
        # self.plot_inspection()

    def load_inspection(self):
        # directory path
        bags = "/home/thanhndv212/Downloads/experiment_data"
        if ".csv" in self.folder:
            SOURCE = "PLOTJUGGLER"
            path_to_values = bags + self.folder
            path_to_names = None
        else:
            path_to_names = bags + self.folder + "/introspection_datanames.csv"
            path_to_values = (
                bags + self.folder + "/introspection_datavalues.csv"
            )
            SOURCE = None
        print(path_to_names, path_to_values)

        (
            t_res,
            self.f_res,
            self.joint_names,
            self.q_abs_res,
            self.q_pos_res,
        ) = get_q_arm(self.tiago, path_to_values, path_to_names, SOURCE=SOURCE)

    def compute_entities(self, sample_range=None):
        # indexing joint_inspected
        self.joint_inspected_id = self.tiago.model.getJointId(
            self.joint_inspected
        )
        self.joint_inspected_idq = rank_in_configuration(
            self.tiago.model, self.joint_inspected
        )
        self.joint_inspected_idv = self.tiago.model.joints[
            self.tiago.model.getJointId(self.joint_inspected)
        ].idx_v

        # sampling range
        if sample_range is None:
            sample_range = range(10, self.q_abs_res.shape[0] - 10)
        self.nsample = len(sample_range)

        # finite difference
        self.q_arm, self.dq_arm, self.ddq_arm = calc_vel_acc(
            self.tiago,
            self.q_pos_res,
            sample_range,
            self.joint_names,
            self.f_res,
            f_cutoff=2,
        )
        self.qabs_arm, self.dqabs_arm, self.ddqabs_arm = calc_vel_acc(
            self.tiago,
            self.q_abs_res,
            sample_range,
            self.joint_names,
            self.f_res,
            f_cutoff=2,
        )

        # extract and compute on joint_inspected
        self.joint_inspected_abs = self.qabs_arm[:, self.joint_inspected_idq]
        self.joint_inspected_rel = self.q_arm[:, self.joint_inspected_idq]
        self.encoder_diff = self.joint_inspected_rel - self.joint_inspected_abs
        self.dq_joint_inspected = self.dq_arm[:, self.joint_inspected_idv]
        self.dqabs_joint_inspected = self.dqabs_arm[
            :, self.joint_inspected_idv
        ]
        self.ddq_joint_inspected = self.ddq_arm[:, self.joint_inspected_idv]
        self.ddqbs_joint_inspected = self.ddqabs_arm[
            :, self.joint_inspected_idv
        ]
        self.tau_joint_inspected = np.zeros((self.nsample, 6))
        for i in range(self.nsample):
            pin.computeCentroidalMomentumTimeVariation(
                self.tiago.model,
                self.tiago.data,
                self.q_arm[i, :],
                self.dq_arm[i, :],
                self.ddq_arm[i, :],
            )
            self.tau_joint_inspected[i, :] = self.tiago.data.f[
                self.joint_inspected_id
            ].vector

        # compute joint torque
        self.tau_g = np.zeros(self.nsample)  # gravity torque
        self.tau_b = np.zeros(self.nsample)  # non-linear dynamic torque
        self.tau = np.zeros(self.nsample)  # joint torque
        for i in range(self.nsample):
            # joint torque
            pin.rnea(
                self.tiago.model,
                self.tiago.data,
                self.q_arm[i, :],
                self.dq_arm[i, :],
                self.ddq_arm[i, :],
            )
            self.tau[i] = self.tiago.data.tau[self.joint_inspected_idv]

            # gravity effects
            pin.computeGeneralizedGravity(
                self.tiago.model, self.tiago.data, self.q_arm[i, :]
            )
            self.tau_g[i] = self.tiago.data.g[self.joint_inspected_idv]

            # non-linear elements
            pin.nonLinearEffects(
                self.tiago.model,
                self.tiago.data,
                self.q_arm[i, :],
                self.dq_arm[i, :],
            )
            self.tau_b[i] = self.tiago.data.nle[self.joint_inspected_idv]

    def plot_inspection(self):
        # %matplotlib
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(
            np.arange(self.nsample),
            self.tau_joint_inspected[:, 5],
            # (self.tau-self.tau_b),
            label="impulse signal",
            color="black",
            linestyle="--",
        )
        ax[0].plot(
            np.arange(self.nsample),
            self.tau,
            label="joint torque",
            color="black",
        )
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(
            np.arange(self.nsample),
            rad_deg * self.encoder_diff,
            label="encoder readings difference",
            color="blue",
        )
        ax[1].grid()
        ax[1].legend()

        ax[2].plot(
            np.arange(self.nsample),
            rad_deg * self.joint_inspected_abs,
            label="absolute encoder",
            color="green",
        )
        ax[2].plot(
            np.arange(self.nsample),
            rad_deg * self.joint_inspected_rel,
            label="relative encoder",
            color="red",
        )
        ax[2].grid()
        ax[2].legend()

        ax[3].plot(
            np.arange(self.nsample),
            rad_deg * self.dq_joint_inspected,
            label="velocities",
            color="green",
        )
        ax[3].plot(
            np.arange(self.nsample),
            rad_deg * self.dqabs_joint_inspected,
            label="abs velocities",
            color="green",
            linestyle="--",
        )
        ax[3].grid()
        ax[3].legend()
        fig.suptitle(self.folder)

    def plot_hysteresis(self):
        plt.plot(
            rad_deg * self.joint_inspected_rel,
            rad_deg * self.joint_inspected_abs,
        )
        plt.xlabel("relative encoder")
        plt.ylabel("absolute encoder")
        plt.grid()

    def plot_torque_joint_diff(self):
        fig_jt = plt.figure()
        plt.plot(
            self.tau,
            rad_deg * self.encoder_diff,
            label="torque against encoder difference",
            color="black",
        )
        plt.ylabel("encoder difference [deg]")
        plt.xlabel("joint torque [N.m]")
        plt.grid()

    def plot_allInOne(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample")
        ax1.set_ylabel("deg", color="blue")
        ax1.plot(
            np.arange(self.nsample),
            rad_deg * self.encoder_diff,
            label="$\Delta \\theta$",
            color="blue",
        )
        ax1.plot(
            np.arange(self.nsample),
            self.tau_joint_inspected[:, 5],
            # self.tau,
            label="impulse signal",
            color="black",
            linestyle="--",
        )
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        ax2 = (
            ax1.twinx()
        )  # instantiate a second axes that shares the same x-axis

        color = "green"
        ax2.set_ylabel(
            "deg/s", color=color
        )  # we already handled the x-label with ax1
        ax2.plot(
            np.arange(self.nsample),
            rad_deg * self.dq_joint_inspected,
            label="$\dot{ \\theta }_M$",
            color="green",
        )
        ax2.plot(
            np.arange(self.nsample),
            rad_deg * self.dqabs_joint_inspected,
            label="$\dot{\\theta}_L$",
            color="green",
            linestyle="--",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid()
        ax2.legend(loc="lower left")
        plt.show()

    def play_motion(self, q_, dt=0.01):
        import sys
        import time
        from pinocchio.visualize import GepettoVisualizer

        viz = GepettoVisualizer(
            model=self.tiago.model,
            collision_model=self.tiago.collision_model,
            visual_model=self.tiago.visual_model,
        )
        try:
            viz.initViewer()
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install gepetto-viewer"
            )
            print(err)
            sys.exit(0)

        try:
            viz.loadViewerModel("pinocchio")
        except AttributeError as err:
            print(
                "Error while loading the viewer model. It seems you should start gepetto-viewer"
            )
            print(err)
            sys.exit(0)

        time.sleep(3)
        viz.display(self.tiago.q0)
        if q_.ndim == 1:
            viz.display(q_)
        else:
            for i in range(q_.shape[0]):
                viz.display(q_[i])
                time.sleep(dt)


def read_result(files):
    import pandas as pd

    data_dir = "/home/thanhndv212/develop/figaroh/examples/tiago/data/media"
    list_order = []
    list_rms = []
    list_r2 = []
    list_coeff = []
    for file in files:
        result_data = pd.read_csv(data_dir + file)
        # data_header = ["order", "rms", "r2", "coeficient and rho"]
        list_order = list_order + [
            str(order) for order in result_data.loc[:, "order"].values
        ]
        list_rms = list_rms + list(result_data.loc[:, "rms"].values)
        list_r2 = list_r2 + list(result_data.loc[:, "r2"].values)
        list_coeff = list_coeff + list(
            result_data.loc[:, "coeficient and rho"].values
        )

    # print(list({tuple(sorted(i)): i for i in result_data.loc[:, "order"].values}.values()))
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("rmse [deg]", color="black")
    ax1.set_xlabel("polynomial order of fitting function", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.bar(
        list_order,
        list_rms,
        color="gray",
    )
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.scatter(
        list_order,
        list_r2,
        linestyle="--",
        color="blue",
    )
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylabel("R-squared score", color="blue")
    plt.suptitle(file[file.find("backlash/") :])
    plt.show()


def write_result(SF: SurfaceFitting):
    import csv

    with open("surface_fitting_{}.csv".format("0202"), "w") as ofile:
        w = csv.writer(ofile)
        w.writerow(["order", "rms", "r2", "coeficient and rho"])
        w.writerow([SF.co_order, SF.rmse, SF.r2_score, SF.bl_solve.x])


def plot_torque_joint_diff(self, title, fig=None, ax1=None, ax2=None):
    if isinstance(fig, plt.Figure) and isinstance(ax1, plt.Axes):
        fig = fig
        ax1 = ax1
    else:
        fig, ax1 = plt.subplots()
    ax1.set_ylabel("joint arm 6 configuration [deg]", color="black")
    # ax1.set_xlabel("encoder difference [deg]", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    # ax1.plot(
    #     rad_deg * self.encoder_diff,
    #     rad_deg * self.joint_inspected_rel,
    #     # color="black",
    # )
    ax1.plot(
        self.tau,
        rad_deg * self.joint_inspected_rel,
        color="black",
    )
    ax1.grid()
    if isinstance(ax2, plt.Axes):
        ax2 = ax2
    else:
        ax2 = ax1.twinx()

    ax2.plot(
        self.tau,
        rad_deg * self.encoder_diff,
        # rad_deg * self.encoder_diff,
        linestyle="--",
        color="green",
    )
    ax2.tick_params(axis="y", labelcolor="blue")
    # ax2.set_xlabel("torque by gravity effect [N.m]", color="blue")
    ax2.set_ylabel("encoder difference [deg]", color="blue")

    # ax2.set_ylabel("torque by gravity effect [N.m]", color="blue")

    plt.suptitle(title)
    return fig, ax1, ax2


def surface_fitting(IJ, co_order):
    if isinstance(IJ, list):
        qd_ = IJ[0].joint_inspected_rel
        tau_ = IJ[0].tau
        signal_bl = IJ[0].dq_joint_inspected
        ed = IJ[0].encoder_diff

        for j in range(1, len(IJ)):
            qd_ = np.append(qd_, IJ[j].joint_inspected_rel)
            tau_ = np.append(tau_, IJ[j].tau)
            signal_bl = np.append(signal_bl, IJ[j].dq_joint_inspected)
            ed = np.append(ed, IJ[j].encoder_diff)
        qd = np.array([qd_]).T
        tau = np.array([tau_]).T

    else:
        qd_ = IJ.joint_inspected_rel
        tau_ = IJ.tau
        signal_bl = IJ.dq_joint_inspected
        ed = IJ.encoder_diff
        # signal_bl = IJ.tau_joint_inspected[:, 5]
        # signal_bl = IJ.tau
        qd = np.array([qd_]).T
        tau = np.array([tau_]).T

    A_r = create_poly(qd, tau, co_order)
    A_l = create_poly(qd, tau, co_order)
    rho_0 = 100
    C_0 = [0] * int((A_r.shape[1] + A_l.shape[1]))
    bl = np.zeros_like(qd_)

    def calc_bl(x):
        assert len(x) == int(A_r.shape[1] + A_l.shape[1])
        C_r = np.dot(A_r, np.array([x[: A_r.shape[1]]]).T)
        C_l = np.dot(A_l, np.array([x[A_r.shape[1] :]]).T)
        for jj in range(qd.shape[0]):
            bl[jj] = (C_r[jj]) * (1 / (1 + np.exp(-rho_0 * signal_bl[jj]))) + (
                C_l[jj]
            ) * (1 - (1 / (1 + np.exp(-rho_0 * signal_bl[jj]))))
        return bl

    def cost_func(x):
        return calc_bl(x) - ed

    bl_solve = least_squares(
        cost_func,
        C_0,
        method="lm",
        verbose=1,
        args=(),
    )

    def rmse(y, y_pred):
        return np.sqrt(sum((y - y_pred) ** 2) / y.shape[0])

    def r2_score(y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    rmse = rmse(ed, calc_bl(bl_solve.x))
    r2_score = r2_score(ed, calc_bl(bl_solve.x))
    print("Polynomial order: {}".format(co_order))
    print(
        "RMS error of the fitting is {} [deg] and r2 score is {} ".format(
            rmse, r2_score
        )
    )
    return [co_order, bl_solve, rmse, r2_score]


def test_arm6(IJ_h):
    q = IJ_h.qabs_arm[0]
    pin.forwardKinematics(IJ_h.tiago.model, IJ_h.tiago.data, q)
    pin.updateFramePlacements(IJ_h.tiago.model, IJ_h.tiago.data)
    rot_arm6 = IJ_h.tiago.data.oMi[IJ_h.joint_inspected_id].rotation
    arm5 = "arm_5_joint"
    arm5_jointid = IJ_h.tiago.model.getJointId(arm5)
    arm5_idxq = IJ_h.tiago.model.joints[arm5_jointid].idx_q
    (arm5_l, arm5_u) = (
        IJ_h.tiago.model.lowerPositionLimit[arm5_idxq],
        IJ_h.tiago.model.upperPositionLimit[arm5_idxq],
    )
    n_test = 11
    q_test = np.zeros((n_test, len(q)))
    for ii in range(n_test):
        q_test[ii, :] = q
        q_test[ii, arm5_idxq] = arm5_l + ii * (arm5_u - arm5_l) / (n_test - 1)
    print(len(IJ_h.tiago.model.lowerPositionLimit))
    print(IJ_h.tiago.model.njoints)
    print(arm5_l, arm5_u)
    print(q_test.shape)
    print(rad_deg * pin.rpy.matrixToRpy(rot_arm6))
    print(rot_arm6)
    print(rad_deg * np.arccos(rot_arm6[2, 2]))

    IJ_h.play_motion(q_test, 0.5)

    # find no gravity configuration
    arm4_idxq = IJ_h.tiago.model.joints[
        IJ_h.tiago.model.getJointId("arm_4_joint")
    ].idx_q

    def f(x):
        q[arm4_idxq] = x
        pin.forwardKinematics(IJ_h.tiago.model, IJ_h.tiago.data, q)
        pin.updateFramePlacements(IJ_h.tiago.model, IJ_h.tiago.data)
        rot_arm6 = IJ_h.tiago.data.oMi[IJ_h.joint_inspected_id].rotation
        return rot_arm6[2, 2] + 1

    qarm4_0 = q[arm4_idxq]

    solve_qarm4 = least_squares(
        f,
        qarm4_0,
        method="lm",
        verbose=1,
        args=(),
    )
    print(solve_qarm4.x, qarm4_0)
    q[arm4_idxq] = solve_qarm4.x
    n_test = 11
    q_test = np.zeros((n_test, len(q)))
    for ii in range(n_test):
        q_test[ii, :] = q
        q_test[ii, arm5_idxq] = arm5_l + ii * (arm5_u - arm5_l) / (n_test - 1)


def main():
    pass


if __name__ == "__main__":
    # joint_name = "arm_1_joint"
    # sample_range = None
    # orders = [5, 5]

    # tiago = load_robot(
    #     abspath("urdf/tiago_48_schunk.urdf"),
    #     load_by_urdf=True,
    # )
    # # folder_sin = "/backlash/20230916/creps_calib_vicon.csv"
    # folder_sin = "/backlash/20230916/sinus_amp3_period10_bottom.csv"
    # IJ = InspectJoint(tiago, joint_name, folder_sin, sample_range)
    # joint_names = [
    #     "arm_1_joint",
    #     "arm_2_joint",
    #     "arm_3_joint",
    #     "arm_4_joint",
    #     "arm_5_joint",
    #     "arm_6_joint",
    #     "arm_7_joint",
    # ]
    # models = []
    # SFs = []
    # for joint_name_ in joint_names:
    #     IJ.joint_inspected = joint_name_
    #     IJ.compute_entities()
    #     SF = SurfaceFitting(IJ, orders)
    #     models.append(SF.solve().x)
    #     SFs.append(SF)
    #     # SF.plot_3dregression()
    # SFs[0].plot_3dregression()

    # arm6
    joint_name = "arm_1_joint"
    sample_range = None
    tiago_hey5 = load_robot(
        abspath("urdf/tiago_48_hey5.urdf"),
        load_by_urdf=True,
    )
    folder_arm6 = [
        "/backlash/20240202/arm5_minus_1_57.csv",
        "/backlash/20240202/arm5_minus_1_2.csv",
        "/backlash/20240202/arm5_minus_0_8_2.csv",
        "/backlash/20240202/arm5_minus_0_8.csv",
        "/backlash/20240202/arm5_minus_0_5_2.csv",
        "/backlash/20240202/arm5_minus_0_5.csv",
        "/backlash/20240202/arm5_minus_0_1.csv",
        "/backlash/20240202/arm5_1_2.csv",
        "/backlash/20240202/arm5_0.csv",
        "/backlash/20240202/arm5_0_8.csv",
        "/backlash/20240202/arm5_0_5.csv",
        "/backlash/20240202/arm5_0_4.csv",
        "/backlash/20240202/arm5_0_2.csv",
    ]
    IJS = []
    for fa6 in folder_arm6:
        IJ_arm6 = InspectJoint(tiago_hey5, "arm_6_joint", fa6, sample_range)
        IJ_arm6.compute_entities()
        IJS.append(IJ_arm6)
    orders = [
        [x, y]
        for n in range(1, 5 + 1)
        for y in range(1, n + 1)
        for x in range(1, n + 1)
        if x + y == n]
    SF_arm6 = SurfaceFitting(IJS, orders)
    SF_arm6.solve()
    SFs.append(SF_arm6)

    # import pandas as pd
    # df = pd.read_csv(abspath('data/calibration/backlash/result/surface_fitting_rho100_0202.csv'))
    # fig, ax1 = plt.subplots()
    # %matplotlib
    # ax2 = ax1.twinx()
    # ax1.plot(list(df['order']), np.array(df['rms']), color='black', marker='+', label='rmse')
    # ax2.plot(list(df['order']), np.array(df['r2']), color='black', marker='o', label='R-squared')
    # ax1.set_xticklabels(list(df['order']), fontsize=8, rotation=90)
    # ax1.set_ylabel('RMSE of $arm\_6$[rad]')
    # ax2.set_ylabel('R-Squared value')
    # ax1.set_xlabel('Order of polynomial function')
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='lower left')

    # # %matplotlib
    # data = []
    # data_ed = []

    # for sf in SFs:
    #     data.append(sf.ed - sf.calc_bl(sf.bl_solve.x))
    #     data_ed.append(sf.ed)

    # params = {
    #     "axes.labelsize": 24,
    #     "axes.titlesize": 24,
    #     "axes.labelsize": 24,
    #     "xtick.labelsize": 24,
    #     "ytick.labelsize": 24,
    #     "legend.fontsize": 24,
    #     "axes.labelpad": 4,
    #     "axes.spines.top": False,
    #     "axes.spines.right": False,
    # }
    # plt.rcParams.update(params)
    # joints = [
    #     "arm_1",
    #     "arm_2",
    #     "arm_3",
    #     "arm_4",
    #     "arm_5",
    #     "arm_6",
    #     "arm_7",
    # ]
    # bp1 = plt.boxplot(
    #     data[:5] + [data[7]] + [data[6]],
    #     positions=2 * np.arange(1, 8),
    #     showfliers=False,
    #     notch=True,
    #     sym="k+",
    #     patch_artist=True,
    # )
    # plt.setp(bp1["boxes"], facecolor="red")

    # bp2 = plt.boxplot(
    #     data_ed[:5] + [data_ed[7]] + [data_ed[6]],
    #     positions=2 * np.arange(1, 8) - 1,
    #     showfliers=False,
    #     notch=True,
    #     sym="k+",
    #     patch_artist=True,
    # )
    # plt.setp(bp2["boxes"], facecolor="blue")

    # plt.xticks(2 * np.arange(1, 8) - 0.5, joints)
    # plt.xticks(rotation=45)
    # plt.ylabel("Two Encoders Measure Difference [rad]")
    # # plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['No compensation', 'Compensated with backlash model'])
