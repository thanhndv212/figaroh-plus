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
import argparse


def parse_args():
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

    return args


args = parse_args()

# load_by_urdf = False, load robot from rospy.get_param(/robot_description)
tiago = load_robot(
    "data/urdf/tiago_48_{}.urdf".format(args.end_effector), load_by_urdf=True
)

# create a calibration object from config file
# del_list=[(0, 1)], 0: numbered marker, 1: numbered sample will be removed
tiago_calib = TiagoCalibration(
    tiago, "config/tiago_config_{}.yaml".format(args.end_effector), del_list=[]
)

# load data file and determine parameters to be calibrated
tiago_calib.initialize()

# solve least_squares estimation
tiago_calib.solve()
write_to_xacro(
    tiago_calib,
    file_name="tiago_master_calibration_{}.yaml".format(args.end_effector),
    file_type="yaml",
)
tiago_calib.plot(lvl=1)
