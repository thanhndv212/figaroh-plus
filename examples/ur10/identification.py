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
import csv
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import (
    build_regressor_basic,
    get_index_eliminate,
    build_regressor_reduced,
)
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import (
    get_param_from_yaml,
    calculate_first_second_order_differentiation,
    base_param_from_standard,
    calculate_standard_parameters,
)
import matplotlib.pyplot as plt
import pprint
import yaml
from yaml.loader import SafeLoader

# 1/ Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv("ROS_PACKAGE_PATH")
package_dirs = ros_package_path.split(":")

robot = Robot(
    "data/robot.urdf",
    package_dirs=package_dirs
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

with open("config/ur10_config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)

identif_data = config["identification"]
params_settings = get_param_from_yaml(robot, identif_data)
print(params_settings)

params_standard_u = robot.get_standard_parameters(params_settings)
print(params_standard_u)

# Print out the placement of each joint of the kinematic tree
print("\nJoint placements:")
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))

# generate a list containing the full set of standard parameters
params_standard = robot.get_standard_parameters(params_settings)

# 1. First we build the structural base identification model, i.e. the one that can
# be observed, using random samples

q_rand = np.random.uniform(
    low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nq)
)

dq_rand = np.random.uniform(
    low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nv)
)

ddq_rand = np.random.uniform(
    low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nv)
)

W = build_regressor_basic(robot, q_rand, dq_rand, ddq_rand, params_settings)

# remove zero cols and build a zero columns free regressor matrix
idx_e, params_r = get_index_eliminate(W, params_standard, 1e-6)
W_e = build_regressor_reduced(W, idx_e)

# Calulate the base regressor matrix, the base regroupings equations params_base and
# get the idx_base, ie. the index of base parameters in the initial regressor matrix
_, params_base, idx_base = get_baseParams(W_e, params_r, params_standard)

print("The structural base parameters are: ")
for ii in range(len(params_base)):
    print(params_base[ii])

# simulating a sinus trajectory on joints shoulder_lift_joint, elbow_joint, wrist_2_joint

nb_samples = 100

q = np.zeros((nb_samples, model.nq))

with open("data/identification_q_simulation.csv", "r") as f:
    csvreader = csv.reader(f)
    ii = 0
    for row in csvreader:
        if ii == 0:
            print("Row zero")
        else:
            q[ii - 1, :] = np.array(
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ]
            )
        ii += 1

q, dq, ddq = calculate_first_second_order_differentiation(model, q, params_settings)

W = build_regressor_basic(robot, q, dq, ddq, params_settings)
# select only the columns of the regressor corresponding to the structural base
# parameters
W_base = W[:, idx_base]
print("When using all trajectories the cond num is", int(np.linalg.cond(W_base)))

# simulation of the measured joint torques
tau_noised = np.empty(len(q) * model.nq)

with open("data/identification_tau_simulation.csv", "r") as f:
    csvreader = csv.reader(f)
    ii = 0
    for row in csvreader:
        if ii == 0:
            print("Row zero")
        else:
            tau_temp = np.array(
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ]
            )
            tau_noised[ii - 1] = tau_temp[0]
            tau_noised[len(q) + ii - 1] = tau_temp[1]
            tau_noised[2 * len(q) + ii - 1] = tau_temp[2]
            tau_noised[3 * len(q) + ii - 1] = tau_temp[3]
            tau_noised[4 * len(q) + ii - 1] = tau_temp[4]
            tau_noised[5 * len(q) + ii - 1] = tau_temp[5]
        ii += 1

# Least-square identification process
phi_base = np.matmul(np.linalg.pinv(W_base), tau_noised)

phi_base_real = base_param_from_standard(params_standard, params_base)

tau_identif = W_base @ phi_base

plt.plot(tau_noised, label="simulated+noised")
plt.plot(tau_identif, label="identified")
plt.legend()
plt.show()

COM_max = np.ones((6 * 3, 1))  # subject to be more adapted
COM_min = -np.ones((6 * 3, 1))

phi_standard, phi_ref = calculate_standard_parameters(
    robot, W, tau_noised, COM_max, COM_min, params_settings
)

print(phi_standard)
print(phi_ref)

plt.plot(phi_standard, label="SIP Identified")
plt.plot(phi_ref, label="SIP URDF")
plt.legend()
plt.show()

# TODO : adapt constraints on COM, verify SIP, modify the model with SIP ?
