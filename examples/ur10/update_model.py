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

from math import pi
import numpy as np
import yaml
from string import Template
import pinocchio
from pinocchio import SE3, Quaternion

# Conversion from names in the kinematics parameter file to joint names
paramToJoint = {
    'shoulder'  : 'shoulder_pan_joint',
    'upper_arm' : 'shoulder_lift_joint',
    'forearm'   : 'elbow_joint',
    'wrist_1'   : 'wrist_1_joint',
    'wrist_2'   : 'wrist_2_joint',
    'wrist_3'   : 'wrist_3_joint'
}
# Conversion from joint names to names in the kinematics parameter file
jointToParam = {
    'shoulder_pan_joint'  : 'shoulder',
    'shoulder_lift_joint' : 'upper_arm',
    'elbow_joint'         : 'forearm',
    'wrist_1'             : 'wrist_1',
    'wrist_2'             : 'wrist_2',
    'wrist_3'             : 'wrist_3'
}

axes = {'x': 'd_px', 'y': 'd_py', 'z': 'd_pz',
        'roll': 'd_phix', 'pitch': 'd_phiy', 'yaw': 'd_phiz'}

# Order the names as in output file
ordered_names = ['shoulder', 'upper_arm', 'forearm', 'wrist_1', 'wrist_2',
                 'wrist_3']
template = """    x: $x
    y: $y
    z: $z
    roll: $roll
    pitch: $pitch
    yaw: $yaw
"""
##
#  Update kinematic parameter file for ur10 robot
#  \param f_input path to the file containing the current parameters,
#  \param f_output path to the file where updated parameters will be written,
#  \param var calibration parameters after least square minimization,
#  \param param variable containing the name of the parameters in var.
#
#  shoulder  -> shoulder_pan_joint
#  upper_arm -> shoulder_lift_joint
#  forearm   -> elbow_joint
#  wrist_1   -> wrist_1_joint
#  wrist_2   -> wrist_2_joint
#  wrist_3   -> wrist_3_joint
#  
def update_parameters(f_input, f_output, var, param):
    # read kinematic parameters
    with open(f_input, 'r') as f:
        kinematics_params = yaml.load(f, Loader=yaml.SafeLoader)
        for pname in paramToJoint.keys():
            # Do not change pose of first robot axis
            if pname == 'shoulder': continue
            for axis, n in axes.items():
                try:
                    id = param['param_name'].index(n + '_' +
                                                   paramToJoint[pname], 0)
                    offset = var[id]
                except ValueError as exc:
                    offset = 0
                kinematics_params['kinematics'][pname][axis] += offset
    t = Template(template)
    output="kinematics:\n"
    for name in ordered_names:
        output += (f'  {name}:\n')
        output += t.substitute(kinematics_params['kinematics'][name])
    output +='  hash:\n'
    # Write camera parameters
    # optical frame in wrist_3_joint as computed by calibration procedure
    jMop = SE3(translation = var[-6:-3],
              rotation = pinocchio.rpy.rpyToMatrix(var[-3:]))
    # d435_mount_link in wrist_3_joint
    jMm = SE3(translation = np.array([0.,0.,0.]),
              rotation = pinocchio.rpy.rpyToMatrix(np.array([0.,0.,pi])))
    # optical frame in ref_camera_link
    cMop = SE3(translation = np.array([0.011, 0.033, 0.013]),
               rotation = pinocchio.rpy.rpyToMatrix(np.array([-pi/2,0,-pi/2])))
    # mMc = mMj * jMop * opMc
    mMc = jMm.inverse() * jMop * cMop.inverse()
    xyz = mMc.translation
    rpy = pinocchio.rpy.matrixToRpy(mMc.rotation)
    kinematics_params['camera']['x'] = xyz[0]
    kinematics_params['camera']['y'] = xyz[1]
    kinematics_params['camera']['z'] = xyz[2]
    kinematics_params['camera']['roll'] = rpy[0]
    kinematics_params['camera']['pitch'] = rpy[1]
    kinematics_params['camera']['yaw'] = rpy[2]
    output += '\n# Pose of ref_camera_link in ur10e_d435_mount_link\n'
    output += 'camera:\n'
    output += t.substitute(kinematics_params['camera'])
    with open(f_output, 'w') as f:
        f.write(output)
