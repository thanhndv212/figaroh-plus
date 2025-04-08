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
from os.path import abspath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from scipy import signal
from figaroh.identification.identification_tools import get_param_from_yaml
from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
    get_index_eliminate,
)
from figaroh.tools.qrdecomposition import double_QR
from figaroh.identification.identification_tools import relative_stdev
from utils.tiago_tools import load_robot
from datetime import datetime
import csv


def load_config(file_path):
    """Load configuration from YAML file."""
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def load_csv_data(params_settings):
    """Load and process CSV data."""
    ts = pd.read_csv(
        abspath(params_settings["pos_data"]), usecols=[0]
    ).to_numpy()
    pos = pd.read_csv(abspath(params_settings["pos_data"]))
    vel = pd.read_csv(abspath(params_settings["vel_data"]))
    eff = pd.read_csv(abspath(params_settings["torque_data"]))

    cols = {"pos": [], "vel": [], "eff": []}
    for jn in params_settings["active_joints"]:
        cols["pos"].extend([col for col in pos.columns if jn in col])
        cols["vel"].extend([col for col in vel.columns if jn in col])
        cols["eff"].extend([col for col in eff.columns if jn in col])

    q = pos[cols["pos"]].to_numpy()
    dq = vel[cols["vel"]].to_numpy()
    tau = eff[cols["eff"]].to_numpy()

    return ts, q, dq, tau


def apply_filters(t, q, dq, nbutter=4, f_butter=2, med_fil=5, f_sample=100):
    """Apply median and lowpass filters to position and velocity data."""
    b1, b2 = signal.butter(nbutter, f_butter / (f_sample / 2), "low")

    q_filtered = np.zeros(q.shape)
    dq_filtered = np.zeros(dq.shape)

    for j in range(dq.shape[1]):
        q_med = signal.medfilt(q[:, j], med_fil)
        q_filtered[:, j] = signal.filtfilt(
            b1,
            b2,
            q_med,
            padtype="odd",
            padlen=3 * (max(len(b1), len(b2)) - 1),
        )

        dq_med = signal.medfilt(dq[:, j], med_fil)
        dq_filtered[:, j] = signal.filtfilt(
            b1,
            b2,
            dq_med,
            padtype="odd",
            padlen=3 * (max(len(b1), len(b2)) - 1),
        )

    return q_filtered, dq_filtered


def estimate_acceleration(t, dq_filtered):
    """Estimate acceleration from filtered velocity."""
    return np.array(
        [
            np.gradient(dq_filtered[:, j]) / np.gradient(t[:, 0])
            for j in range(dq_filtered.shape[1])
        ]
    ).T


def build_full_configuration(robot, q_f, dq_f, ddq_f, params_settings, N_):
    """Build full configuration arrays for position, velocity, and acceleration."""
    p = np.tile(robot.q0, (N_, 1))
    v = np.tile(robot.v0, (N_, 1))
    a = np.tile(robot.v0, (N_, 1))

    p[:, params_settings["act_idxq"]] = q_f
    v[:, params_settings["act_idxv"]] = dq_f
    a[:, params_settings["act_idxv"]] = ddq_f

    return p, v, a


def truncate_data(t, q, dq, tau, n_i, n_f):
    t, q, dq, tau = t[n_i:n_f], q[n_i:n_f], dq[n_i:n_f], tau[n_i:n_f]
    return t, q, dq, tau


def process_torque_data(tau, params_settings, robot):
    """Process torque data with reduction ratios and motor constants."""

    pin.computeSubtreeMasses(robot.model, robot.data)
    for i, joint_name in enumerate(params_settings["active_joints"]):
        if joint_name == "torso_lift_joint":
            tau[:, i] = (
                params_settings["reduction_ratio"][joint_name]
                * params_settings["kmotor"][joint_name]
                * tau[:, i]
                + 9.81 * robot.data.mass[robot.model.getJointId(joint_name)]
            )
        else:
            tau[:, i] = (
                params_settings["reduction_ratio"][joint_name]
                * params_settings["kmotor"][joint_name]
                * tau[:, i]
            )

    return tau


def decimate_data(t, tau, W_e, params_settings, Ntotal):
    """Decimate data for each joint."""

    #######separate link-by-link and parallel decimate########
    # joint torque
    tau_dec = []
    for i in range(len(params_settings["act_idxv"])):
        tau_dec.append(signal.decimate(tau[:, i], q=10, zero_phase=True))

    tau_rf = tau_dec[0]
    for i in range(1, len(tau_dec)):
        tau_rf = np.append(tau_rf, tau_dec[i])

    # regressor
    W_list = []  # list of sub regressor for each joitnt
    for i in range(len(params_settings["act_idxv"])):
        W_dec = []
        for j in range(W_e.shape[1]):
            W_dec.append(
                signal.decimate(
                    W_e[
                        range(
                            params_settings["act_idxv"][i] * Ntotal,
                            (params_settings["act_idxv"][i] + 1) * Ntotal,
                        ),
                        j,
                    ],
                    q=10,
                    zero_phase=True,
                )
            )

        W_temp = np.zeros((W_dec[0].shape[0], len(W_dec)))
        for i in range(len(W_dec)):
            W_temp[:, i] = W_dec[i]
        W_list.append(W_temp)

    # rejoining sub  regresosrs into one complete regressor
    W_rf = np.zeros((tau_rf.shape[0], W_list[0].shape[1]))
    for i in range(len(W_list)):
        W_rf[
            range(i * W_list[i].shape[0], (i + 1) * W_list[i].shape[0]), :
        ] = W_list[i]

    # time
    t_dec = signal.decimate(t[:, 0], q=10, zero_phase=True)

    return t_dec, tau_dec, tau_rf, W_rf


def plot_torque(t, t_dec, tau_dec, tau_base, tau_ref, params_settings):
    """Plot comparison of estimated and measured torques."""
    fig, axs = plt.subplots(
        len(params_settings["active_joints"]),
        1,
        figsize=(12, 3 * len(params_settings["active_joints"])),
    )
    plt.rcParams.update({"font.size": 12, "grid.color": "lightgrey"})

    for i, ax in enumerate(axs):
        ax.plot(
            t_dec, tau_dec[i], color="red", label="Measured" if i == 0 else ""
        )
        ax.plot(
            t_dec,
            tau_base[i * len(t_dec) : (i + 1) * len(t_dec)],
            color="green",
            label="Identified model" if i == 0 else "",
        )
        # ax.plot(
        #     t,
        #     tau_ref[i*len(t):(i+1)*len(t)],
        #     color="blue",
        #     linestyle="--",
        #     label="Initial model" if i == 0 else ""
        # )

        ax.set_ylabel(
            f"{'Torso' if i == 0 else f'Arm {i}'} [{'N' if i == 0 else 'N.m'}]",
            rotation="horizontal",
            ha="right",
        )
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(labelbottom=i == len(axs) - 1)

        if i == len(axs) - 1:
            ax.set_xlabel("Time [s]")

    axs[0].legend(loc="upper right")
    plt.tight_layout()
    plt.show()


class TiagoIdentification:
    def __init__(self, robot, config_file):
        self._robot = robot
        self.model = self._robot.model
        self.data = self._robot.data
        self.load_param(config_file)

    def solve(
        self,
        truncate=True,
        decimate=True,
        plotting=True,
        save_params=False,
    ):
        self.process_data(truncate=truncate)
        self.calc_full_regressor()
        self.calc_baseparam(
            decimate=decimate, plotting=plotting, save_params=save_params
        )

    def load_param(self, config_file, setting_type="identification"):
        """
        Load the calibration parameters from the yaml file and add undefined
        parameters.
        """
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)
        self.params_settings = get_param_from_yaml(
            self._robot, config[setting_type]
        )

    def process_data(self, truncate=True):
        """Load and process data"""
        t_, q_, dq_, tau_ = load_csv_data(self.params_settings)

        # Truncate data
        if truncate:
            n_i, n_f = 921, 6791
            t_, q_, dq_, tau_ = truncate_data(t_, q_, dq_, tau_, n_i, n_f)

        # Apply filters and estimate acceleration
        q_filtered_, dq_filtered_ = apply_filters(t_, q_, dq_)
        ddq_filtered_ = estimate_acceleration(t_, dq_filtered_)

        # after processing, total number of samples
        self.Nsample_ = q_filtered_.shape[0]

        # Build full configuration
        p_, v_, a_ = build_full_configuration(
            self._robot,
            q_filtered_,
            dq_filtered_,
            ddq_filtered_,
            self.params_settings,
            self.Nsample_,
        )

        # Process torque data
        tau_prcd = process_torque_data(tau_, self.params_settings, self._robot)
        self.processed_data = {
            "t": t_,
            "p": p_,
            "v": v_,
            "a": a_,
            "tau": tau_prcd,
        }

    def calc_full_regressor(self):
        # Build regressor
        self.W = build_regressor_basic(
            self._robot,
            self.processed_data["p"],
            self.processed_data["v"],
            self.processed_data["a"],
            self.params_settings,
        )
        self.standard_parameter = self._robot.get_standard_parameters(
            self.params_settings
        )
        # joint torque estimated from p,v,a with std params
        phi_ref = np.array(list(self.standard_parameter.values()))
        tau_ref = np.dot(self.W, phi_ref)
        self.tau_ref = tau_ref[
            range(len(self.params_settings["act_idxv"]) * self.Nsample_)
        ]

    def calc_baseparam(self, decimate=True, plotting=True, save_params=False):
        # Eliminate zero columns
        idx_e_, active_parameter_ = get_index_eliminate(
            self.W, self.standard_parameter, tol_e=0.001
        )
        W_e_ = build_regressor_reduced(self.W, idx_e_)

        # remove zero-crossing data
        if decimate:
            t_dec, tau_dec, tau_rf, W_rf = decimate_data(
                self.processed_data["t"],
                self.processed_data["tau"],
                W_e_,
                self.params_settings,
                self.Nsample_,
            )
        else:
            tau_rf = self.processed_data["tau"]
            W_rf = W_e_
        # calculate base parameters
        W_b, bp_dict, base_parameter, phi_b, phi_std = double_QR(
            tau_rf, W_rf, active_parameter_, self.standard_parameter
        )
        rmse = np.linalg.norm(tau_rf - np.dot(W_b, phi_b)) / np.sqrt(
            tau_rf.shape[0]
        )
        std_xr_ols = relative_stdev(W_b, phi_b, tau_rf)

        self.result = {
            "base regressor": W_b,
            "base parameters": bp_dict,
            "condition number": np.linalg.cond(W_b),
            "rmse norm (N/m)": rmse,
            "torque estimated": np.dot(W_b, phi_b),
            "std dev of estimated param": std_xr_ols,
        }

        def save_to_csv(file_path=None):
            dt = datetime.now()
            _time = dt.strftime("%d_%b_%Y_%H%M")
            if file_path is None:
                bp_csv = abspath(
                    f"data/identification/dynamic/{self.model.name}_bp_{_time}.csv"
                )
            else:
                bp_csv = file_path + f"{self.model.name}_bp_{time}.csv"
            with open(bp_csv, "w") as output_file:
                w = csv.writer(output_file)
                for i in range(len(base_parameter)):
                    w.writerow(
                        [base_parameter[i], phi_b[i], 100 * std_xr_ols[i]]
                    )

        if save_params:
            save_to_csv()

        if plotting:
            plot_torque(
                self.processed_data["t"],
                t_dec,
                tau_dec,
                self.result["torque estimated"],
                self.tau_ref,
                self.params_settings,
            )


def main():
    robot = load_robot(abspath("urdf/tiago_48_schunk.urdf"), load_by_urdf=True)

    TiagoIden = TiagoIdentification(robot, "config/tiago_config.yaml")

    # define additional parameters excluded from yaml files
    ps = TiagoIden.params_settings
    ps["reduction_ratio"] = {
        "torso_lift_joint": 1,
        "arm_1_joint": 100,
        "arm_2_joint": 100,
        "arm_3_joint": 100,
        "arm_4_joint": 100,
        "arm_5_joint": 336,
        "arm_6_joint": 336,
        "arm_7_joint": 336,
    }
    ps["kmotor"] = {
        "torso_lift_joint": 1,
        "arm_1_joint": 0.136,
        "arm_2_joint": 0.136,
        "arm_3_joint": -0.087,
        "arm_4_joint": -0.087,
        "arm_5_joint": -0.0613,
        "arm_6_joint": -0.0613,
        "arm_7_joint": -0.0613,
    }

    ps["active_joints"] = [
        "torso_lift_joint",
        "arm_1_joint",
        "arm_2_joint",
        "arm_3_joint",
        "arm_4_joint",
        "arm_5_joint",
        "arm_6_joint",
        "arm_7_joint",
    ]

    # joint id of active joints
    ps["act_Jid"] = [
        TiagoIden.model.getJointId(i) for i in ps["active_joints"]
    ]
    # active joint objects
    ps["act_J"] = [TiagoIden.model.joints[jid] for jid in ps["act_Jid"]]
    # joint config id (e.g one joint might have >1 DOF)
    ps["act_idxq"] = [J.idx_q for J in ps["act_J"]]
    # joint velocity id
    ps["act_idxv"] = [J.idx_v for J in ps["act_J"]]
    # dataset path
    ps["pos_data"] = "data/identification/dynamic/tiago_position.csv"
    ps["vel_data"] = "data/identification/dynamic/tiago_velocity.csv"
    ps["torque_data"] = "data/identification/dynamic/tiago_effort.csv"

    TiagoIden.solve(
        truncate=True,
        decimate=True,
        plotting=True,
        save_params=False,
    )
    for key, value in TiagoIden.result.items():
        print(f"{key} : {value}")


if __name__ == "__main__":
    main()
