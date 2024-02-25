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

from tiago_utils.robot_tools import (
    RobotCalibration,
    load_robot,
    write_to_xacro,
)
import argparse
from os.path import abspath


# load_by_urdf, otherwise from rospy.get_param(/robot_description)
tiago = load_robot(abspath("urdf/tiago_48_schunk.urdf"), load_by_urdf=True)

# del_list=[(0, 1)], 0: numbered marker, 1: numbered sample will be removed
tiago_calib = RobotCalibration(
    tiago, abspath("config/tiago_config_mocap_vicon.yaml"), del_list=[]
)

tiago_calib.initialize()
torso_idxq = tiago_calib.param["config_idx"][0]
tiago_calib.solve()
tiago_calib.plot(lvl=1)
# load absolute encoder values
tiago_abs = RobotCalibration(
    tiago, abspath("config/tiago_config_mocap_vicon_abs.yaml"), del_list=[]
)
tiago_abs.initialize()

# replace relative encoder values of torso to absolute one
# tiago_abs.q_measured[:, torso_idxq] = tiago_calib.q_measured[:, torso_idxq]
# tiago_calib.param["param_name"].remove('d_pz_arm_2_joint')
tiago_abs.solve()
tiago_abs.plot(lvl=1)

# write_to_xacro(
#     tiago_calib,
#     file_name="tiago_master_calibration_{}.yaml".format(args.end_effector),
#     file_type="yaml",
# )
