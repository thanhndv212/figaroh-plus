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

import os
from os.path import dirname, join, abspath
import numpy as np

# update estimated parameters to xacro file for left hand

torso_list = [0, 1, 2, 3, 4, 5]
arm1_list = [6, 7, 8, 11]
arm2_list = [13, 16]
arm3_list = [19, 22]
arm4_list = [24, 27]
arm5_list = [30, 33]
arm6_list = [36, 39]
arm7_list = [43, 46]  # include phiz7
total_list = [torso_list, arm1_list, arm2_list, arm3_list, arm4_list, arm5_list,
              arm6_list, arm7_list]

zero_list = []
for i in range(len(total_list)):
    zero_list = [*zero_list, *total_list[i]]

def update_parameters(model, res, param):

    param_list = np.zeros((param['NbJoint'], 6))

    # torso all zeros

    # arm 1
    param_list[1, 3] = res[6]
    param_list[1, 4] = res[7]

    # arm 2
    param_list[2, 0] = res[8]
    param_list[2, 2] = res[9]
    param_list[2, 3] = res[10]
    param_list[2, 5] = res[11]

    # arm 3
    param_list[3, 0] = res[12]
    param_list[3, 2] = res[13]
    param_list[3, 3] = res[14]
    param_list[3, 5] = res[15]

    # arm 4
    param_list[4, 1] = res[16]
    param_list[4, 2] = res[17]
    param_list[4, 4] = res[18]
    param_list[4, 5] = res[19]

    # arm 5
    param_list[5, 1] = res[20]
    param_list[5, 2] = res[21]
    param_list[5, 4] = res[22]
    param_list[5, 5] = res[23]

    # arm 6
    param_list[6, 1] = res[24]
    param_list[6, 2] = res[25]
    param_list[6, 4] = res[26]
    param_list[6, 5] = res[27]

    # arm 7
    param_list[7, 0] = res[28]
    param_list[7, 2] = res[29]
    param_list[7, 3] = res[30]
    param_list[7, 5] = res[31]

    joint_names = [name for i, name in enumerate(model.names)]
    offset_name = ['_x_offset', '_y_offset', '_z_offset', '_roll_offset',
                '_pitch_offset', '_yaw_offset']
    path_save_xacro = join(
        dirname(dirname(str(abspath(__file__)))),
        f"data/offset.xacro")
    with open(path_save_xacro, "w") as output_file:
        for i in range(param['NbJoint']):
            for j in range(6):
                update_name = joint_names[i+1] + offset_name[j]
                update_value = param_list[i, j]
                update_line = "<xacro:property name=\"{}\" value=\"{}\" / >".format(
                    update_name, update_value)
                output_file.write(update_line)
                output_file.write('\n')
    path_save_yaml = join(
        dirname(dirname(str(abspath(__file__)))),
        f"data/offset.yaml")
    with open(path_save_yaml, "w") as output_file:
        for i in range(param['NbJoint']):
            for j in range(6):
                update_name = joint_names[i+1] + offset_name[j]
                update_value = param_list[i, j]
                update_line = "{}: {}".format(
                    update_name, update_value)
                output_file.write(update_line)
                output_file.write('\n')
############################################################
# # path_save_ep = join(
# #     dirname(dirname(str(abspath(__file__)))),
# #     f"data/talos/0403_estimation_result.csv")
# # with open(path_save_ep, "w") as output_file:
# #     w = csv.writer(output_file)
# #     for i in range(nvars):
# #         w.writerow(
# #             [
# #                 params_name[i],
# #                 LM_solve.x[i],
# #                 std_dev[i],
# #                 std_pctg[i]
# #             ]
# #         )