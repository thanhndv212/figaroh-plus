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

from utils.tiago_tools import TiagoCalibration, load_robot, write_to_xacro
import argparse


# load_by_urdf = False, load robot from rospy.get_param(/robot_description)
tiago = load_robot("urdf/tiago_48_schunk.urdf", load_by_urdf=True)

# create a calibration object from config file
# del_list=[(0, 1)], 0: numbered marker, 1: numbered sample will be removed
tiago_calib = TiagoCalibration(tiago, "config/tiago_config.yaml", del_list=[])
tiago_calib.param["known_baseframe"] = False
tiago_calib.param["known_tipframe"] = False
# load data file and determine parameters to be calibrated
tiago_calib.initialize()
print(tiago_calib.param["param_name"])
# solve least_squares estimation
tiago_calib.solve()

# tiago_calib.plot()
