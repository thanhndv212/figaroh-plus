# Copyright 2022 CNRS
#
# Author: Florent Lamiraux
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import yaml
from string import Template

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
    with open(f_output, 'w') as f:
        f.write(output)
