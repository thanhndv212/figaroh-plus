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


class Measurement:
    """Class for handling different types of measurements."""

    def __init__(self, name, joint, frame, type, value):
        """Initialize measurement.

        Args:
            name: Name of measurement
            joint: Joint where measurement is expressed 
            frame: Closest frame from measurement
            type: Type of measurement (SE3, wrench, current)
            value: 6D measurement value wrt joint placement
        """
        self.name = name
        self.joint = joint
        self.frame = frame
        self.type = type
        if self.type == "SE3":
            self.SE3_value = pin.SE3(
                pin.rpyToMatrix(np.array([value[3], value[4], value[5]])),
                np.array([value[0], value[1], value[2]]),
            )
        elif self.type == "wrench":
            self.wrench_value = pin.Force(value)
        elif self.type == "current":
            self.current_value = value
        else:
            raise TypeError("The type of your measurement is not valid")

    def add_SE3_measurement(self, model):
        """Add SE3 measurement to model.

        Args:
            model: Pinocchio model to add measurement to

        Returns:
            tuple: Updated model and data
        """
        if self.type == "SE3":
            self.frame_index = model.addFrame(
                pin.Frame(
                    self.name,
                    model.getJointId(self.joint),
                    model.getFrameId(self.frame),
                    self.SE3_value,
                    pin.OP_FRAME,
                ),
                "False",
            )
        data = model.createData()
        return model, data
