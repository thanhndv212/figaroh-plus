import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.optimize import least_squares
import plotly.graph_objects as go

rad_deg = 180 / np.pi


def generateData(n=30):
    # similar to peaks() function in MATLAB
    g = np.linspace(-3.0, 3.0, n)
    X, Y = np.meshgrid(g, g)
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    Z = (
        3 * (1 - X) ** 2 * np.exp(-(X**2) - (Y + 1) ** 2)
        - 10 * (X / 5 - X**3 - Y**5) * np.exp(-(X**2) - Y**2)
        - 1 / 3 * np.exp(-((X + 1) ** 2) - Y**2)
    )
    return X, Y, Z


def exp2model(e):
    # C[i] * X^n * Y^m
    model = " + ".join(
        [
            f"C[{i}]"
            + ("*" if x > 0 or y > 0 else "")
            + (f"X^{x}" if x > 1 else "X" if x == 1 else "")
            + ("*" if x > 0 and y > 0 else "")
            + (f"Y^{y}" if y > 1 else "Y" if y == 1 else "")
            for i, (x, y) in enumerate(e)
        ]
    )
    return model


def names2model(names):
    # C[i] * X^n * Y^m
    return " + ".join(
        [f"C[{i}]*{n.replace(' ','*')}" for i, n in enumerate(names)]
    )


def create_poly(X, Y, co_order):
    # create a design matrix of a polynomial
    if isinstance(co_order, list):
        order = min(co_order)
        s_order = np.asarray([range(min(co_order) + 1, max(co_order) + 1)])
    elif isinstance(co_order, int):
        order = co_order
    e = [
        (x, y)
        for n in range(0, order + 1)
        for y in range(0, n + 1)
        for x in range(0, n + 1)
        if x + y == n
    ]
    eX = np.asarray([[x] for x, _ in e]).T
    eY = np.asarray([[y] for _, y in e]).T
    # best-fit polynomial surface
    A = (X**eX) * (Y**eY)
    if isinstance(co_order, list):
        S = X**s_order
        A = np.c_[A, S]
    return A


def polyfit(X, Y, Z, co_order):
    # calculate exponents of design matrix
    # e = [(x,y) for x in range(0,order+1) for y in range(0,order-x+1)]
    if isinstance(co_order, list):
        order = min(co_order)
        s_order = np.asarray([range(min(co_order) + 1, max(co_order))])
    elif isinstance(order, int):
        order = co_order
    e = [
        (x, y)
        for n in range(0, order + 1)
        for y in range(0, n + 1)
        for x in range(0, n + 1)
        if x + y == n
    ]
    eX = np.asarray([[x] for x, _ in e]).T
    eY = np.asarray([[y] for _, y in e]).T

    # best-fit polynomial surface
    A = (X**eX) * (Y**eY)
    if isinstance(co_order, list):
        S = X**s_order
        A = np.c_[A, S]
    C, resid, _, _ = lstsq(A, Z)  # coefficients

    # calculate R-squared from residual error
    r2 = 1 - resid[0] / (Z.size * Z.var())

    # print summary
    print(f"data = {Z.size}x3")
    print(f"model = {exp2model(e)}")
    print(
        f"\n{len(C)} coefficients =\n{C}",
    )
    print(f"R2 = {r2}")

    # uniform grid covering the domain of the data
    XX, YY = np.meshgrid(
        np.linspace(X.min(), X.max(), 20), np.linspace(Y.min(), Y.max(), 20)
    )

    # evaluate model on grid
    A = (XX.reshape(-1, 1) ** eX) * (YY.reshape(-1, 1) ** eY)
    if isinstance(co_order, list):
        S = XX.reshape(-1, 1) ** s_order
        A = np.c_[A, S]
    ZZ = np.dot(A, C).reshape(XX.shape)

    # plot points and fitted surface
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(X, Y, Z, c="r", s=2)
    ax.plot_surface(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=1,
        alpha=0.2,
        linewidth=0.5,
        edgecolor="b",
    )
    ax.axis("tight")
    ax.view_init(azim=-60.0, elev=30.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def scikitfit(X, Y, Z, order):
    # best-fit polynomial surface
    model = make_pipeline(
        PolynomialFeatures(degree=order), LinearRegression(fit_intercept=False)
    )
    model.fit(np.c_[X, Y], Z)

    m = names2model(model[0].get_feature_names_out(["X", "Y"]))
    C = model[1].coef_.T  # coefficients
    r2 = model.score(np.c_[X, Y], Z)  # R-squared

    # print summary
    print(f"data = {Z.size}x3")
    print(f"model = {m}")
    print(f"coefficients =\n{C}")
    print(f"R2 = {r2}")

    # uniform grid covering the domain of the data
    XX, YY = np.meshgrid(
        np.linspace(X.min(), X.max(), 20), np.linspace(Y.min(), Y.max(), 20)
    )

    # evaluate model on grid
    ZZ = model.predict(np.c_[XX.flatten(), YY.flatten()]).reshape(XX.shape)

    # plot points and fitted surface
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(X, Y, Z, c="r", s=2)
    ax.plot_surface(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=1,
        alpha=0.2,
        linewidth=0.5,
        edgecolor="b",
    )
    ax.axis("tight")
    ax.view_init(azim=-60.0, elev=30.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


class SurfaceFitting:
    def __init__(self, IJ, co_order):
        self.IJ_ = IJ
        self.co_order = co_order

        if isinstance(self.IJ_, list):
            # self.IJ_.joint_inspected = self.IJ_[0].joint_inspected
            qd_ = self.IJ_[0].joint_inspected_rel
            tau_ = self.IJ_[0].tau_g
            signal_bl = self.IJ_[0].dq_joint_inspected
            ed = self.IJ_[0].encoder_diff

            for j in range(1, len(self.IJ_)):
                qd_ = np.append(qd_, self.IJ_[j].joint_inspected_rel)
                tau_ = np.append(tau_, self.IJ_[j].tau_g)
                signal_bl = np.append(
                    signal_bl, self.IJ_[j].dq_joint_inspected
                )
                ed = np.append(ed, self.IJ_[j].encoder_diff)
            qd = np.array([qd_]).T
            tau = np.array([tau_]).T

        else:
            qd_ = self.IJ_.joint_inspected_rel
            tau_ = self.IJ_.tau_g
            signal_bl = self.IJ_.dq_joint_inspected
            ed = self.IJ_.encoder_diff
            # signal_bl = self.IJ_.tau_joint_inspected[:, 5]
            # signal_bl = self.IJ_.tau
            qd = np.array([qd_]).T
            tau = np.array([tau_]).T

        A_r = create_poly(qd, tau, co_order)
        A_l = create_poly(qd, tau, co_order)
        print("class SF A shape", A_r.shape)
        self.rho_0 = 100
        C_0 = [0] * int((A_r.shape[1] + A_l.shape[1]))

        self.qd_ = qd_
        self.tau_ = tau_
        self.qd = qd
        self.tau = tau
        self.C_0 = C_0
        self.A_r = A_r
        self.A_l = A_l
        self.signal_bl = signal_bl
        self.ed = ed

    def calc_bl(self, x):
        bl = np.zeros_like(self.qd_)
        assert len(x) == int(self.A_r.shape[1] + self.A_l.shape[1])
        C_r = np.dot(self.A_r, np.array([x[: self.A_r.shape[1]]]).T)
        C_l = np.dot(self.A_l, np.array([x[self.A_r.shape[1] :]]).T)
        for jj in range(self.qd.shape[0]):
            bl[jj] = (C_r[jj]) * (
                1 / (1 + np.exp(-self.rho_0 * (self.signal_bl[jj])))
            ) + (C_l[jj]) * (
                1 - (1 / (1 + np.exp(-self.rho_0 * (self.signal_bl[jj]))))
            )
        return bl

    def rmse(self, y, y_pred):
        return np.sqrt(sum((y - y_pred) ** 2) / y.shape[0])

    def r2_score(self, y, y_pred):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def solve(self):
        def cost_func(x):
            return self.calc_bl(x) - self.ed

        self.bl_solve = least_squares(
            cost_func,
            self.C_0,
            method="lm",
            verbose=1,
            args=(),
        )

        self.rmse = self.rmse(self.ed, self.calc_bl(self.bl_solve.x))
        self.r2_score = self.r2_score(self.ed, self.calc_bl(self.bl_solve.x))
        print("Polynomial order: {}".format(self.co_order))
        print(
            "RMS error of the fitting is {} [deg] and r2 score is {} ".format(
                self.rmse, self.r2_score
            )
        )
        return self.bl_solve

    def plot_2dregression(self):
        if not isinstance(self.IJ_, list):
            plt.plot(
                self.qd_,
                self.calc_bl(self.bl_solve.x),
                color="black",
                linestyle="--",
                label="predicted joint angle difference",
            )
            plt.plot(
                self.qd_,
                self.ed,
                color="blue",
                label="measured joint angle difference",
            )
            plt.xlabel("joint angle measures by relative encoder[deg]")
            plt.ylabel("joint angle measures difference[deg]")
            plt.grid()
        else:
            pass

    def plot_3dregression(self, renderer="browser", save_=False):
        marker_data = go.Scatter3d(
            # x=rad_deg * self.qd_,
            x=self.qd_,
            y=self.tau_,
            # z=rad_deg * self.ed,
            z=self.ed,
            marker=go.scatter3d.Marker(size=2),
            opacity=0.8,
            mode="markers",
            name="Encoder difference measures",
        )
        fig = go.Figure(
            data=marker_data,
        )
        fig.add_scatter3d(
            # x=rad_deg * self.qd_,
            x=self.qd_,
            y=self.tau_,
            # z=rad_deg * self.calc_bl(self.bl_solve.x),
            z=self.calc_bl(self.bl_solve.x),
            marker=go.scatter3d.Marker(size=2),
            opacity=0.5,
            mode="markers",
            name="Predicted encoder difference by surface fitting",
        )
        fig.update_layout(showlegend=True)

        # fig.update_layout(
        #     scene=dict(
        #         xaxis_title="Joint Angle [rad]",
        #         yaxis_title="Joint Torque [N.m]",
        #         zaxis_title="Two Encoders Measure Difference [rad]",
        #     ),
        #     margin=dict(r=10, l=10, b=10, t=10)
        # )
        # fig.update_xaxes(
        #     title_text="joint angle measure (deg)",
        #     title_font={"size": 25},
        #     tickfont=dict(family="Rockwell", size=20),
        #     gridcolor="black",
        # )
        fig.update_layout(
            width=1400,
            height=1400,
            # title_text="Polynomial of order {}, rms = {}, R2 score = {}".format(
            #     self.IJ_.joint_inspected + "_" + str(self.co_order),
            #     round(self.rmse, 4),
            #     round(self.r2_score, 4),
            # ),
        )
        # grid
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    gridcolor="black",
                    showbackground=True,
                ),
                yaxis=dict(
                    gridcolor="black",
                    showbackground=True,
                ),
                zaxis=dict(
                    gridcolor="black",
                    showbackground=True,
                ),
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    nticks=5,
                    tickfont=dict(
                        color="black",
                        size=24,
                    ),
                    title_font=dict(color="black", size=5),
                ),
                yaxis=dict(
                    nticks=5,
                    tickfont=dict(
                        color="black",
                        size=24,
                    ),
                    title_font=dict(color="black", size=5),
                ),
                zaxis=dict(
                    nticks=5,
                    tickfont=dict(
                        color="black",
                        size=24,
                    ),
                    title_font=dict(color="black", size=5),
                ),
            ),
            margin=dict(r=30, l=30, b=30, t=10)
        )
        fig.show(renderer=renderer)
        if save_:
            fig.write_html(
                "/home/thanhndv212/develop/figaroh/examples/tiago/data/calibration/backlash/media/{}.html".format(
                    self.IJ_.joint_inspected + "_" + str(self.co_order)
                )
            )


# # generate some random 3-dim points
# X, Y, Z = generateData()

# # 1=linear, 2=quadratic, 3=cubic, ..., nth degree
# co_order = [3, 5]

# polyfit(X, Y, Z, co_order)
# # scikitfit(X, Y, Z, co_order)
