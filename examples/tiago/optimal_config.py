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

import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import picos as pc
import argparse
from figaroh.calibration.calibration_tools import (
    calculate_base_kinematics_regressor,
    rank_in_configuration,
)
from tiago_utils.tiago_tools import load_robot, TiagoCalibration
from optimization_utils import rearrange_rb, sub_info_matrix, Detmax, SOCP


class TiagoOptimalCalibration(TiagoCalibration):
    """Optimal calibration class for TIAGo robot."""

    def __init__(self, robot, config_file):
        super().__init__(robot, config_file)
        self._sampleConfigs_file = self.param["sample_configs_file"]
        self.minNbChosen = self.calculate_min_nb_chosen()

    def calculate_min_nb_chosen(self):
        """Calculate the minimum number of configurations to be chosen."""
        if self.param["calib_model"] == "full_params":
            return (
                int(
                    len(self.param["actJoint_idx"])
                    * 6
                    / self.param["calibration_index"]
                )
                + 1
            )
        elif self.param["calib_model"] == "joint_offset":
            return (
                int(
                    len(self.param["actJoint_idx"])
                    / self.param["calibration_index"]
                )
                + 1
            )
        else:
            raise ValueError("Calibration model not supported.")

    def initialize(self):
        """Initialize the calibration process."""
        self.load_data_set()
        self.calculate_regressor()
        self.calculate_detroot_whole()

    def solve(self, file_name=None):
        """Solve the optimal calibration problem."""
        self.calculate_optimal_configurations()
        self.write_to_file(name_=file_name)
        self.plot()

    def load_data_set(self):
        """Load the dataset from file."""
        if "csv" in self._sampleConfigs_file:
            super().load_data_set()
        elif "yaml" in self._sampleConfigs_file:
            with open(self._sampleConfigs_file, "r") as file:
                self._configs = yaml.load(file, Loader=SafeLoader)

            q_jointNames = self._configs["calibration_joint_names"]
            q_jointConfigs = np.array(
                self._configs["calibration_joint_configurations"]
            ).T

            df = pd.DataFrame.from_dict(dict(zip(q_jointNames, q_jointConfigs)))

            q = np.zeros([len(df), self._robot.q0.shape[0]])
            for i in range(len(df)):
                for j, name in enumerate(q_jointNames):
                    jointidx = rank_in_configuration(self.model, name)
                    q[i, jointidx] = df[name][i]
            self.q_measured = q

            self.param["NbSample"] = self.q_measured.shape[0]
        else:
            raise ValueError("Data file format not supported.")

    def calculate_regressor(self):
        """Calculate the kinematic regressor."""
        (
            Rrand_b,
            R_b,
            R_e,
            paramsrand_base,
            paramsrand_e,
        ) = calculate_base_kinematics_regressor(
            self.q_measured, self.model, self.data, self.param
        )
        for ii, param_b in enumerate(paramsrand_base):
            print(ii + 1, param_b)

        self.R_rearr = rearrange_rb(R_b, self.param)
        self._subX_list, self._subX_dict = sub_info_matrix(
            self.R_rearr, self.param
        )
        return True

    def calculate_detroot_whole(self):
        """Calculate the determinant root of the whole matrix."""
        assert self.calculate_regressor(), "Calculate regressor first."
        M_whole = np.matmul(self.R_rearr.T, self.R_rearr)
        self.detroot_whole = pc.DetRootN(M_whole)
        print("detrootn of whole matrix:", self.detroot_whole)

    def calculate_optimal_configurations(self):
        """Calculate the optimal configurations for calibration."""
        assert self.calculate_regressor(), "Calculate regressor first."

        prev_time = time.time()
        SOCP_algo = SOCP(self._subX_dict, self.param)
        self.w_list, self.w_dict_sort = SOCP_algo.solve()
        solve_time = time.time() - prev_time
        print("solve time of socp: ", solve_time)

        self.eps_opt = 1e-5
        chosen_config = [
            i
            for i in self.w_dict_sort.keys()
            if self.w_dict_sort[i] > self.eps_opt
        ]

        assert (
            len(chosen_config) >= self.minNbChosen
        ), "Infeasible design, try to increase NbSample."

        print(len(chosen_config), "configs are chosen: ", chosen_config)
        self.nb_chosen = len(chosen_config)
        opt_ids = chosen_config
        opt_configs_values = [
            self._configs["calibration_joint_configurations"][opt_id]
            for opt_id in opt_ids
        ]
        self.opt_configs = self._configs.copy()
        self.opt_configs["calibration_joint_configurations"] = list(
            opt_configs_values
        )
        return True

    def write_to_file(self, name_=None):
        """Write the optimal configurations to a file."""
        assert (
            self.calculate_optimal_configurations()
        ), "Calculate optimal configurations first."
        path_save = (
            "data/calibration/optimal_configuration/eye_hand/tiago_optimal_configurations.yaml"
            if name_ is None
            else f"data/calibration/optimal_configuration/eye_hand/{name_}"
        )
        with open(path_save, "w") as stream:
            try:
                yaml.dump(
                    self.opt_configs,
                    stream,
                    sort_keys=False,
                    default_flow_style=True,
                )
            except yaml.YAMLError as exc:
                print(exc)
        return True

    def plot(self):
        """Plot the results of the optimization."""
        det_root_list = []
        n_key_list = []

        for nbc in range(self.minNbChosen, self.param["NbSample"] + 1):
            n_key = list(self.w_dict_sort.keys())[:nbc]
            n_key_list.append(n_key)
            M_i = pc.sum(
                self.w_dict_sort[i] * self._subX_list[i] for i in n_key
            )
            det_root_list.append(pc.DetRootN(M_i))

        fig, ax = plt.subplots(2)

        ratio = self.detroot_whole / det_root_list[-1]
        plot_range = self.param["NbSample"] - self.minNbChosen
        ax[0].set_ylabel("D-optimality criterion", fontsize=20)
        ax[0].tick_params(axis="y", labelsize=18)
        ax[0].plot(ratio * np.array(det_root_list[:plot_range]))
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].grid(True, linestyle="--")
        ax[0].legend(fontsize=18)

        ax[1].set_ylabel("Weight values (log)", fontsize=20)
        ax[1].set_xlabel("Data sample", fontsize=20)
        ax[1].tick_params(axis="both", labelsize=18)
        ax[1].tick_params(axis="y", labelrotation=30)
        ax[1].scatter(
            np.arange(len(list(self.w_dict_sort.values()))),
            list(self.w_dict_sort.values()),
        )
        ax[1].set_yscale("log")
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].grid(True, linestyle="--")
        plt.show()

        return True


def main():
    """Main function to run the optimal calibration."""
    parser = argparse.ArgumentParser(
        description="parse calibration setups", add_help=False
    )
    parser.add_argument(
        "-e", "--end_effector", default="hey5", dest="end_effector"
    )
    parser.add_argument(
        "-u", "--load_by_urdf", default=True, dest="load_by_urdf"
    )
    args = parser.parse_args()

    tiago = load_robot(
        f"data/urdf/tiago_48_{args.end_effector}.urdf", load_by_urdf=True
    )

    tiago_optcalib = TiagoOptimalCalibration(
        tiago, f"config/tiago_config_{args.end_effector}.yaml"
    )
    tiago_optcalib.initialize()
    tiago_optcalib.solve(
        file_name=f"tiago_optimal_configurations_{args.end_effector}.yaml"
    )


if __name__ == "__main__":
    main()
