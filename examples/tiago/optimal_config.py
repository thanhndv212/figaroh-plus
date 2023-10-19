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
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import picos as pc
import pandas as pd
from figaroh.calibration.calibration_tools import (
    calculate_base_kinematics_regressor,
    rank_in_configuration,
)
from tiago_tools import load_robot, TiagoCalibration


def rearrange_rb(R_b, param):
    """Rearrange the kinematic regressor by sample numbered order.

    Args:
        R_b (numpy.ndarray): The kinematic regressor.
        param (dict): The dictionary containing calibration parameters.

    Returns:
        numpy.ndarray: The rearranged kinematic regressor.
    """
    Rb_rearr = np.empty_like(R_b)
    for i in range(param["calibration_index"]):
        for j in range(param["NbSample"]):
            Rb_rearr[j * param["calibration_index"] + i, :] = R_b[
                i * param["NbSample"] + j
            ]
    return Rb_rearr


def sub_info_matrix(R, param):
    """Return a list of sub information matrices (product of transpose of
    regressor and regressor) which corresponds to each data sample.

    Input:  R (numpy.ndarray): The kinematic regressor.
            param (dict): The dictionary containing calibration parameters.

    Output: tuple: A tuple containing a list of sub information matrices and a
            dictionary mapping sample indices to sub information matrices.
    """
    subX_list = []
    idex = param["calibration_index"]
    for it in range(param["NbSample"]):
        sub_R = R[it * idex : (it * idex + idex), :]
        subX = np.matmul(sub_R.T, sub_R)
        subX_list.append(subX)
    subX_dict = dict(zip(np.arange(param["NbSample"]), subX_list))
    return subX_list, subX_dict


class Detmax:
    def __init__(self, candidate_pool, NbChosen):
        """Initialize the Detmax class.

        Input:  candidate_pool (dict): Pool of candidate samples.
                NbChosen (int): The number of chosen samples.
        """
        self.pool = candidate_pool
        self.nd = NbChosen
        self.cur_set = []
        self.fail_set = []
        self.opt_set = []
        self.opt_critD = []

    def get_critD(self, set):
        """Given a list of indices in the candidate pool, output the n-th
        squared determinant of information matrix constructed by the given
        list.

        Input:
            set (list): A list of indices in the candidate pool.

        Output:
            float: The n-th squared determinant of the information matrix.
        """
        infor_mat = 0
        for idx in set:
            assert (
                idx in self.pool.keys()
            ), "chosen sample not in candidate pool"
            infor_mat += self.pool[idx]
        return float(pc.DetRootN(infor_mat))

    def main_algo(self):
        """Implement the main algorithm for maximizing the determinant of the
        information matrix.

        Returns:
            A list of optimal squared determinants of the information matrix.
        """
        pool_idx = tuple(self.pool.keys())

        # Initialize a random set
        cur_set = random.sample(pool_idx, self.nd)
        updated_pool = list(set(pool_idx) - set(self.cur_set))

        # Adding samples from remaining pool: k = 1
        opt_k = updated_pool[0]
        opt_critD = self.get_critD(cur_set)
        # init_set = set(cur_set)
        # fin_set = set([])
        rm_j = cur_set[0]

        while opt_k != rm_j:
            # add
            for k in updated_pool:
                cur_set.append(k)
                cur_critD = self.get_critD(cur_set)
                if opt_critD < cur_critD:
                    opt_critD = cur_critD
                    opt_k = k
                cur_set.remove(k)
            cur_set.append(opt_k)
            opt_critD = self.get_critD(cur_set)

            # remove
            delta_critD = opt_critD
            rm_j = cur_set[0]
            for j in cur_set:
                rm_set = cur_set.copy()
                rm_set.remove(j)
                cur_delta_critD = opt_critD - self.get_critD(rm_set)

                if cur_delta_critD < delta_critD:
                    delta_critD = cur_delta_critD
                    rm_j = j
            cur_set.remove(rm_j)
            opt_critD = self.get_critD(cur_set)
            # fin_set = set(cur_set)

            self.opt_critD.append(opt_critD)
        return self.opt_critD


class SOCP:
    def __init__(self, subX_dict, param):
        """Initialize the SOCP class.

        Input:  subX_dict (dict):  sub information matrices.
                param (dict): calibration settings.
        """
        self.pool = subX_dict
        self.param = param
        self.problem = pc.Problem()
        self.w = pc.RealVariable("w", self.param["NbSample"], lower=0)
        self.t = pc.RealVariable("t", 1)

    def add_constraints(self):
        """Add constraints to the optimization problem."""
        Mw = pc.sum(
            self.w[i] * self.pool[i] for i in range(self.param["NbSample"])
        )
        self.problem.add_constraint(1 | self.w <= 1)
        self.problem.add_constraint(self.t <= pc.DetRootN(Mw))

    def set_objective(self):
        """Set the optimization objective."""
        self.problem.set_objective("max", self.t)

    def solve(self):
        """Solve the optimization problem and return the solution.

        Returns:
            tuple: A tuple containing a list of solution values and a sorted
            dictionary of the solution.
        """
        self.add_constraints()
        self.set_objective()
        self.solution = self.problem.solve(solver="cvxopt")

        w_list = []
        for i in range(self.w.dim):
            w_list.append(float(self.w.value[i]))
        print("sum of all element in vector solution: ", sum(w_list))

        # Convert to dict
        w_dict = dict(zip(np.arange(self.param["NbSample"]), w_list))
        w_dict_sort = dict(
            reversed(sorted(w_dict.items(), key=lambda item: item[1]))
        )
        return w_list, w_dict_sort


class TiagoOptimalCalibration(TiagoCalibration):
    """
    Generate optimal configurations for calibration.
    """

    def __init__(self, robot, config_file):
        super().__init__(robot, config_file)
        self._sampleConfigs_file = self.param["sample_configs_file"]
        if self.param["calib_model"] == "full_params":
            self.minNbChosen = (
                int(
                    len(self.param["actJoint_idx"])
                    * 6
                    / self.param["calibration_index"]
                )
                + 1
            )
        elif self.param["calib_model"] == "joint_offset":
            self.minNbChosen = (
                int(
                    len(self.param["actJoint_idx"])
                    / self.param["calibration_index"]
                )
                + 1
            )
        else:
            assert False, "Calibration model not supported."

    def initialize(self):
        """
        Initialize the generation process.
        """
        self.load_data_set()
        self.calculate_regressor()
        self.calculate_detroot_whole()

    def solve(self):
        """
        Solve the optimization problem.
        """
        self.calculate_optimal_configurations()
        self.write_to_file()
        self.plot()

    def load_data_set(self):
        """
        Load data from yaml file.
        """
        if "csv" in self._sampleConfigs_file:
            self.load_data_set()
        elif "yaml" in self._sampleConfigs_file:
            with open(self._sampleConfigs_file, "r") as file:
                self._configs = yaml.load(file, Loader=SafeLoader)

            q_jointNames = self._configs["calibration_joint_names"]
            q_jointConfigs = np.array(
                self._configs["calibration_joint_configurations"]
            ).T

            df = pd.DataFrame.from_dict(
                dict(zip(q_jointNames, q_jointConfigs))
            )

            q = np.zeros([len(df), self._robot.q0.shape[0]])
            for i in range(len(df)):
                for j, name in enumerate(q_jointNames):
                    jointidx = rank_in_configuration(self.model, name)
                    q[i, jointidx] = df[name][i]
            self.q_measured = q

            # update number of samples
            self.param["NbSample"] = self.q_measured.shape[0]
        else:
            assert False, "Data file format not supported."

    def calculate_regressor(self):
        """
        Calculate regressor.
        """
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

        # Rearrange the kinematic regressor by sample numbered order
        self.R_rearr = rearrange_rb(R_b, self.param)
        subX_list, subX_dict = sub_info_matrix(self.R_rearr, self.param)
        self._subX_dict = subX_dict
        self._subX_list = subX_list
        return True

    def calculate_detroot_whole(self):
        """
        Calculate detrootn of whole matrix
        """
        assert self.calculate_regressor(), "Calculate regressor first."
        M_whole = np.matmul(self.R_rearr.T, self.R_rearr)
        self.detroot_whole = pc.DetRootN(M_whole)
        print("detrootn of whole matrix:", self.detroot_whole)

    def calculate_optimal_configurations(self):
        """
        Calculate optimal configurations.
        """
        assert self.calculate_regressor(), "Calculate regressor first."

        # Picos optimization (A-optimality, C-optimality, D-optimality)
        prev_time = time.time()
        SOCP_algo = SOCP(self._subX_dict, self.param)
        self.w_list, self.w_dict_sort = SOCP_algo.solve()
        solve_time = time.time() - prev_time
        print("solve time of socp: ", solve_time)

        # Select optimal config based on values of weight
        self.eps_opt = 1e-5
        chosen_config = []
        for i in list(self.w_dict_sort.keys()):
            if self.w_dict_sort[i] > self.eps_opt:
                chosen_config.append(i)

        assert (
            len(chosen_config) >= self.minNbChosen
        ), "Infeasible design, try to increase NbSample."

        print(len(chosen_config), "configs are chosen: ", chosen_config)
        self.nb_chosen = len(chosen_config)
        opt_ids = chosen_config
        opt_configs_values = []
        for opt_id in opt_ids:
            opt_configs_values.append(
                self._configs["calibration_joint_configurations"][opt_id]
            )
        self.opt_configs = self._configs.copy()
        self.opt_configs["calibration_joint_configurations"] = list(
            opt_configs_values
        )
        return True

    def write_to_file(self, name_=None):
        """
        Write optimal configurations to file.
        """
        assert (
            self.calculate_optimal_configurations()
        ), "Calculate optimal configurations first."
        if name_ is None:
            name_ = "data/tiago_optimal_configurations.yaml"
        with open(name_, "w") as stream:
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
        # Plotting
        det_root_list = []
        n_key_list = []

        # Calculate det_root_list and n_key_list
        for nbc in range(self.minNbChosen, self.param["NbSample"] + 1):
            n_key = list(self.w_dict_sort.keys())[0:nbc]
            n_key_list.append(n_key)
            M_i = pc.sum(
                self.w_dict_sort[i] * self._subX_list[i] for i in n_key
            )
            det_root_list.append(pc.DetRootN(M_i))

        # Create subplots
        fig, ax = plt.subplots(2)

        # Plot D-optimality criterion
        ratio = self.detroot_whole / det_root_list[-1]
        plot_range = self.param["NbSample"] - self.minNbChosen
        ax[0].set_ylabel("D-optimality criterion", fontsize=20)
        ax[0].tick_params(axis="y", labelsize=18)
        ax[0].plot(ratio * np.array(det_root_list[:plot_range]))
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].grid(True, linestyle="--")
        ax[0].legend(fontsize=18)

        # Plot quality of estimation
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
    robot = load_robot("data/tiago_hey5.urdf")
    tiago_optcalib = TiagoOptimalCalibration(robot, "config/tiago_config.yaml")
    tiago_optcalib.initialize()
    tiago_optcalib.solve()


if __name__ == "__main__":
    main()
