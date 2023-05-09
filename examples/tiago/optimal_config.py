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
import sys
from os.path import abspath, dirname, join
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
import pprint
import picos as pc
import pandas as pd

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from figaroh.calibration.calibration_tools import (
    get_param_from_yaml,
    calculate_base_kinematics_regressor,
    rank_in_configuration)
from figaroh.tools.robot import Robot


def rearrange_rb(R_b, param):
    """Rearrange the kinematic regressor by sample numbered order.
    
    Args:
        R_b (numpy.ndarray): The kinematic regressor.
        param (dict): The dictionary containing calibration parameters.
    
    Returns:
        numpy.ndarray: The rearranged kinematic regressor.
    """
    Rb_rearr = np.empty_like(R_b)
    for i in range(param['calibration_index']):
        for j in range(param['NbSample']):
            Rb_rearr[j*param['calibration_index'] + i, :] = R_b[i*param['NbSample'] + j]
    return Rb_rearr


def sub_info_matrix(R, param):
    """Return a list of sub information matrices (product of transpose of regressor and regressor)
    which corresponds to each data sample.
    
    Args:
        R (numpy.ndarray): The kinematic regressor.
        param (dict): The dictionary containing calibration parameters.
    
    Returns:
        tuple: A tuple containing a list of sub information matrices and a dictionary mapping sample indices to sub information matrices.
    """
    subX_list = []
    idex = param['calibration_index']
    for it in range(param['NbSample']):
        sub_R = R[it*idex:(it*idex+idex), :]
        subX = np.matmul(sub_R.T, sub_R)
        subX_list.append(subX)
    subX_dict = dict(zip(np.arange(param['NbSample']), subX_list))
    return subX_list, subX_dict


class Detmax():
    def __init__(self, candidate_pool, NbChosen):
        """Initialize the Detmax class.
        
        Args:
            candidate_pool (dict): The dictionary containing candidate samples.
            NbChosen (int): The number of chosen samples from the candidate pool.
        """
        self.pool = candidate_pool
        self.nd = NbChosen
        self.cur_set = []
        self.fail_set = []
        self.opt_set = []
        self.opt_critD = []

    def get_critD(self, set):
        """Given a list of indices in the candidate pool, output the n-th squared determinant
        of information matrix constructed by the given list.
        
        Args:
            set (list): A list of indices in the candidate pool.
        
        Returns:
            float: The n-th squared determinant of the information matrix.
        """
        infor_mat = 0
        for idx in set:
            assert idx in self.pool.keys(), "chosen sample not in candidate pool"
            infor_mat += self.pool[idx]
        return float(pc.DetRootN(infor_mat))

    def main_algo(self):
        """Implement the main algorithm for maximizing the determinant of the information matrix.
        
        Returns:
            list: A list of optimal squared determinants of the information matrix.
        """
        pool_idx = tuple(self.pool.keys())

        # Initialize a random set
        cur_set = random.sample(pool_idx, self.nd)
        updated_pool = list(set(pool_idx) - set(self.cur_set))

        # Adding samples from remaining pool: k = 1
        opt_k = updated_pool[0]
        opt_critD = self.get_critD(cur_set)
        init_set = set(cur_set)
        fin_set = set([])
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
            fin_set = set(cur_set)

            self.opt_critD.append(opt_critD)
        return self.opt_critD


class SOCP():
    def __init__(self, subX_dict, param):
        """Initialize the SOCP class.
        
        Args:
            subX_dict (dict): The dictionary containing sub information matrices.
            param (dict): The dictionary containing calibration parameters.
        """
        self.pool = subX_dict
        self.param = param
        self.problem = pc.Problem()
        self.w = pc.RealVariable('w', self.param['NbSample'], lower=0)
        self.t = pc.RealVariable('t', 1)

    def add_constraints(self):
        """Add constraints to the optimization problem."""
        Mw = pc.sum(self.w[i]*self.pool[i] for i in range(self.param['NbSample']))
        wgt_cons = self.problem.add_constraint(1|self.w <= 1)
        det_root_cons = self.problem.add_constraint(self.t <= pc.DetRootN(Mw))

    def set_objective(self):
        """Set the optimization objective."""
        self.problem.set_objective('max', self.t)

    def solve(self):
        """Solve the optimization problem and return the solution.
        
        Returns:
            tuple: A tuple containing a list of solution values and a sorted dictionary of the solution.
        """
        self.add_constraints()
        self.set_objective()
        self.solution = self.problem.solve(solver='cvxopt')

        w_list = []
        for i in range(self.w.dim):
            w_list.append(float(self.w.value[i]))
        print("sum of all element in vector solution: ", sum(w_list))

        # Convert to dict
        w_dict = dict(zip(np.arange(self.param['NbSample']), w_list))
        w_dict_sort = dict(reversed(sorted(w_dict.items(), key=lambda item: item[1])))
        return w_list, w_dict_sort


# Load robot model and create a dictionary containing reserved constants
# Specify the path to tiago model
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
robot_dir = package_dirs[0] + "/example-robot-data/robots"

# Create a robot object with the specified URDF file and package directories
robot = Robot(
    robot_dir + "/tiago_description/robots/tiago.urdf",
    package_dirs=package_dirs,
)
model = robot.model
data = robot.data

# Load calibration configuration from YAML file
with open('config/tiago_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config['calibration'])
calib_data = config['calibration']
param = get_param_from_yaml(robot, calib_data)

# Read sample configuration pool from file, otherwise random configs are generated
q = []
with open('data/tiago_calibration_joint_configurations_500_pmb2_hey5.yaml', 'r') as file:
    configs = yaml.load(file, Loader=SafeLoader)

q_jointNames = configs['calibration_joint_names']
q_jointConfigs = np.array(configs['calibration_joint_configurations']).T
df = pd.DataFrame.from_dict(dict(zip(q_jointNames, q_jointConfigs)))
q = np.zeros([len(df), robot.q0.shape[0]])
for i in range(len(df)):
    for j, name in enumerate(q_jointNames):
        jointidx = rank_in_configuration(model, name)
        q[i, jointidx] = df[name][i]

# Calculate regressor and individual information matrix
Rrand_b, R_b, R_e, paramsrand_base, paramsrand_e = calculate_base_kinematics_regressor(
    q, model, data, param)
R_rearr = rearrange_rb(R_b, param)
subX_list, subX_dict = sub_info_matrix(R_rearr, param)
for ii, param_b in enumerate(paramsrand_base):
    print(ii+1, param_b)

# Find optimal combination of data samples from a candidate pool (combinatorial optimization)
# Required minimum number of configurations
if param['calib_model'] == 'full_params':
    NbChosen = int(len(param['actJoint_idx'])*6/param['calibration_index']) + 1
elif param['calib_model'] == 'joint_offset':
    NbChosen = int(len(param['actJoint_idx'])/param['calibration_index']) + 1

# Picos optimization (A-optimality, C-optimality, D-optimality)
prev_time = time.time()
M_whole = np.matmul(R_rearr.T, R_rearr)
det_root_whole = pc.DetRootN(M_whole)
print('detrootn of whole matrix:', det_root_whole)

SOCP_algo = SOCP(subX_dict, param)
w_list, w_dict_sort = SOCP_algo.solve()
solve_time = time.time() - prev_time
print('solve time of socp: ', solve_time)

min_NbChosen = NbChosen

# Select optimal config based on values of weight
eps = 1e-5
chosen_config = []
for i in list(w_dict_sort.keys()):
    if w_dict_sort[i] > eps:
        chosen_config.append(i)
if len(chosen_config) < min_NbChosen:
    print("Infeasible design")
else: 
    print(len(chosen_config), "configs are chosen: ", chosen_config)

# Plotting
det_root_list = []
n_key_list = []

# Calculate det_root_list and n_key_list
for nbc in range(min_NbChosen, param["NbSample"] + 1):
    n_key = list(w_dict_sort.keys())[0:nbc]
    n_key_list.append(n_key)
    M_i = pc.sum(w_dict_sort[i] * subX_list[i] for i in n_key)
    det_root_list.append(pc.DetRootN(M_i))

idx_subList = range(len(det_root_list))

# Create subplots
fig, ax = plt.subplots(2)

# Plot D-optimality criterion
ratio = det_root_whole / det_root_list[-1]
plot_range = param["NbSample"] - NbChosen
ax[0].set_ylabel('D-optimality criterion', fontsize=20)
ax[0].tick_params(axis='y', labelsize=18)
ax[0].plot(ratio * np.array(det_root_list[:plot_range]))
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].grid(True, linestyle='--')
ax[0].legend(fontsize=18)

# Plot quality of estimation
ax[1].set_ylabel('Weight values (log)', fontsize=20)
ax[1].set_xlabel('Data sample', fontsize=20)
ax[1].tick_params(axis='both', labelsize=18)
ax[1].tick_params(axis='y', labelrotation=30)
ax[1].scatter(np.arange(len(list(w_dict_sort.values()))), list(w_dict_sort.values()))
ax[1].set_yscale("log")
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].grid(True, linestyle='--')

# DETMAX optimization
for i in range(1):
    prev_time = time.time()
    DM = Detmax(subX_dict, NbChosen)
    y = DM.main_algo()
    x = np.arange(len(y))
    plt.plot(y)
    plt.scatter(x[-1], y[-1], c='red', marker="^")
    print("solve time of detmax: ", time.time() - prev_time)
    NbChosen += 10

# Plot criterion value for D optimality
plt.scatter(x[-1], DM.get_critD(list(w_dict_sort.keys())[:NbChosen]), color='g', marker="^")
plt.ylabel("criterion value - D optimality")
plt.xlabel("iteration")
plt.grid()
plt.show()
