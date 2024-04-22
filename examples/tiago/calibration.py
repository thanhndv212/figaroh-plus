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

from tiago_tools import TiagoCalibration, load_robot, write_to_xacro
from figaroh.calibration.calibration_tools import update_forward_kinematics
from scipy.optimize import least_squares
import numpy as np

# import argparse


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="parse calibration setups", add_help=False
#     )
#     parser.add_argument(
#         "-e", "--end_effector", default="hey5", dest="end_effector"
#     )
#     parser.add_argument(
#         "-u", "--load_by_urdf", default=True, dest="load_by_urdf"
#     )
#     args = parser.parse_args()

#     return args


# args = parse_args()

# # load_by_urdf = False, load robot from rospy.get_param(/robot_description)
# tiago = load_robot(
#     "data/urdf/tiago_48_{}.urdf".format(args.end_effector), load_by_urdf=True
# )

# # create a calibration object from config file
# # del_list=[(0, 1)], 0: numbered marker, 1: numbered sample will be removed
# tiago_calib = TiagoCalibration(
#     tiago, "config/tiago_config_{}.yaml".format(args.end_effector), del_list=[]
# )

# # load data file and determine parameters to be calibrated
# tiago_calib.initialize()

# # solve least_squares estimation
# tiago_calib.solve()

# write_to_xacro(
#     tiago_calib,
#     file_name="tiago_master_calibration_{}.yaml".format(args.end_effector),
#     file_type="yaml",
# )
# tiago_calib.plot(lvl=1)

tiago = load_robot("data/urdf/tiago_48_hey5.urdf", load_by_urdf=False)
TAGcalib_cen = TiagoCalibration(tiago, "config/tiago_config_hey5_center.yaml")
TAGcalib_cen.initialize()
TAGcalib_cen.solve()
var_cen0 = TAGcalib_cen.LM_result.x

var0_common = var_cen0[
    0 : -(
        TAGcalib_cen.param["NbMarkers"]
        * TAGcalib_cen.param["calibration_index"]
    )
]
var0_cenPose = var_cen0[
    -TAGcalib_cen.param["NbMarkers"]
    * TAGcalib_cen.param["calibration_index"] :
]

var0 = np.append(var0_common, var0_cenPose)


# cost function for the optimization problem
def cost_function(var):
    """Combine two cost functions for the optimization problem."""
    var_common = var[
        0 : -(
            TAGcalib_cen.param["NbMarkers"]
            * TAGcalib_cen.param["calibration_index"]
        )
    ]
    var_cenPose = var[
        -(
            TAGcalib_cen.param["NbMarkers"]
            * TAGcalib_cen.param["calibration_index"]
        ) :
    ]

    # calib_center error function for the optimization problem
    var_cen = np.append(var_common, var_cenPose)
    # coeff_cen = TAGcalib_cen.param["coeff_regularize"]
    PEEe_cen = update_forward_kinematics(
        TAGcalib_cen.model,
        TAGcalib_cen.data,
        var_cen,
        TAGcalib_cen.q_measured,
        TAGcalib_cen.param,
    )
    res_vect = TAGcalib_cen.PEE_measured - PEEe_cen

    # chessboard orientation
    cb_orient = np.array([0, -np.pi / 2, -np.pi / 2])
    res_vect = np.append(
        res_vect,
        0.01 * TAGcalib_cen.param["NbSample"] * (var_cenPose[-3:] - cb_orient),
    )

    # torso to zero
    # res_vect = np.append(
    #     res_vect,
    #     TAGcalib_cen.param["NbSample"]
    #     * var[
    #         TAGcalib_cen.param["param_name"].index(
    #             "offsetPZ_torso_lift_joint"
    #         )
    #     ],
    # )

    # head joints to close to zero
    res_vect = np.append(
        res_vect,
        0.1
        * TAGcalib_cen.param["NbSample"]
        * var[TAGcalib_cen.param["param_name"].index("offsetRZ_head_1_joint")],
    )
    res_vect = np.append(
        res_vect,
        0.1
        * TAGcalib_cen.param["NbSample"]
        * var[TAGcalib_cen.param["param_name"].index("offsetRZ_head_2_joint")],
    )
    # camera pose to close to initial pose
    cam_pose = np.array([0.0908, 0.08, 0.0, -np.pi / 2, 0.0, 0.0])
    res_vect = np.append(
        res_vect, 1 * TAGcalib_cen.param["NbSample"] * (var[0:6] - cam_pose)
    )
    # arm1, arm2, arm3 to zero
    arm123_coeff = 0.5
    res_vect = np.append(
        res_vect,
        arm123_coeff
        * TAGcalib_cen.param["NbSample"]
        * (
            var[TAGcalib_cen.param["param_name"].index("offsetRZ_arm_1_joint")]
            - 0.01
        ),
    )
    res_vect = np.append(
        res_vect,
        arm123_coeff
        * TAGcalib_cen.param["NbSample"]
        * (
            var[TAGcalib_cen.param["param_name"].index("offsetRZ_arm_2_joint")]
            - 0.005
        ),
    )
    res_vect = np.append(
        res_vect,
        arm123_coeff
        * TAGcalib_cen.param["NbSample"]
        * var[TAGcalib_cen.param["param_name"].index("offsetRZ_arm_3_joint")],
    )

    return res_vect


# define solver parameters
del_list_ = []
res = var0
# outlier_eps = self.param["outlier_eps"]
print("*" * 50)
# define solver
LM_solve = least_squares(
    cost_function,
    var0,
    method="lm",
    verbose=1,
    args=(),
)

# solution
res = LM_solve.x
res_cen = res


def print_solution(TAGcalib, res_):
    print("solution of calibrated parameters: ")
    for x_i, xname in enumerate(TAGcalib.param["param_name"]):
        print(x_i + 1, xname, list(res_)[x_i])


print("--------------------------------")
print("chessboard center")

PEEe_censol = update_forward_kinematics(
    TAGcalib_cen.model,
    TAGcalib_cen.data,
    res_cen,
    TAGcalib_cen.q_measured,
    TAGcalib_cen.param,
)
# TAGcalib_cen.calc_errors(PEEe_censol)
print("--------------------------------")
param_cen = [float(np.round(res_i_, 6)) for res_i_ in res_cen]
TAGcalib_cen.calibrated_param = dict(
    zip(TAGcalib_cen.param["param_name"], param_cen)
)
print_solution(TAGcalib_cen, param_cen)
write_to_xacro(
    TAGcalib_cen,
    file_name="tiago_master_calibration.yaml",
    file_type="yaml",
)
