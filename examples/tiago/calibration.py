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
# if load_by_urdf=True, make sure to create a correct urdf for the robot
tiago = load_robot("data/urdf/tiago_48_hey5.urdf", load_by_urdf=False)
# del_list=[(0, 1)], 0: numbered marker, 1: numbered sample will be removed
tiago_calib = TiagoCalibration(tiago, "config/tiago_config.yaml", del_list=[])
tiago_calib.initialize()
tiago_calib.solve()
tiago_calib.plot()
write_to_xacro(
    tiago_calib, file_name="tiago_master_calibration.yaml", file_type="yaml"
)
