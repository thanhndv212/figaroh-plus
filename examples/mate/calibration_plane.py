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


from utils.mate_tools import MateCalibration, load_robot
from pinocchio.visualize import GepettoVisualizer
import time
from figaroh.calibration.calibration_tools import (
    update_forward_kinematics,
    get_LMvariables,
)
import numpy as np
from scipy.optimize import least_squares

# load_by_urdf = False, load robot from rospy.get_param(/robot_description)
robot = load_robot(
    "urdf/mate.urdf", load_by_urdf=True
)
model = robot.model
data = robot.data


mate_xyz = MateCalibration(robot, "config/mate.yaml", del_list=[])
mate_xyz.create_param_list()
params_list = mate_xyz.param["param_name"]

mate_x = MateCalibration(robot, "config/mate_x.yaml", del_list=[])
mate_y = MateCalibration(robot, "config/mate_y.yaml", del_list=[])
mate_z = MateCalibration(robot, "config/mate_z.yaml", del_list=[])

# # load data file and determine parameters to be calibrated
mate_x.load_data_set()
mate_y.load_data_set()
mate_z.load_data_set()

q = []
pee = []

for mate_ in [mate_x, mate_y, mate_z]:
    q.append(mate_.q_measured)
    pee.append(mate_.PEE_measured)
    mate_.param['param_name'] = params_list


def cost_function(var, mate_x, mate_y, mate_z):
    """
    Cost function for the optimization problem.
    """
    # coeff_ = self.param["coeff_regularize"]
    PEEe_x = update_forward_kinematics(
        model, data, var, mate_x.q_measured, mate_x.param
    )
    PEEe_y = update_forward_kinematics(
        model, data, var, mate_y.q_measured, mate_y.param
    )
    PEEe_z = update_forward_kinematics(
        model, data, var, mate_z.q_measured, mate_z.param
    )
    res_vect = np.append(
        (mate_x.PEE_measured - PEEe_x), (mate_y.PEE_measured - PEEe_y)
    )
    res_vect = np.append(res_vect, (mate_z.PEE_measured - PEEe_z))
    # res_vect = np.append(
    #     (self.PEE_measured - PEEe),
    #     np.sqrt(coeff_)
    #     * var[6 : -self.param["NbMarkers"] * self.param["calibration_index"]],
    # )
    return res_vect


def solve_optimisation():

    # set initial guess
    _var_0, _ = get_LMvariables(mate_xyz.param, mode=0)

    # define solver parameters
    iterate = True
    iter_max = 10
    count = 0
    res = _var_0

    # while count < iter_max and iterate:
    print("*" * 50)
    print(
        "{} iter guess".format(count),
        dict(zip(mate_xyz.param["param_name"], list(_var_0))),
    )

    # define solver
    LM_solve = least_squares(
        cost_function,
        _var_0,
        method="lm",
        verbose=1,
        args=(mate_x, mate_y, mate_z),
    )

    # solution
    res = LM_solve.x
    
    # rmse = np.sqrt(np.mean((_PEEe_sol - self.PEE_measured) ** 2))
    # mae = np.mean(np.abs(_PEEe_sol - self.PEE_measured))

    print("solution of calibrated parameters: ")
    for x_i, xname in enumerate(mate_xyz.param["param_name"]):
        print(x_i + 1, xname, list(res)[x_i])
    # print("position root-mean-squared error of end-effector: ", rmse)
    # print("position mean absolute error of end-effector: ", mae)
    print("optimality: ", LM_solve.optimality)


solve_optimisation()