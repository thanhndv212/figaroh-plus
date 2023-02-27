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
    def __init__(self,name,joint,frame,type,value):
        """
        Input : name (str) : the name of the measurement
                joint (str) : the joint in which the measurement is expressed
                frame (str) : the closest frame from the measurement
                type (str) : type of measurement, choice between SE3, wrench or current
                value (6D array) : value of the measurement. Suppose that value is given wrt the joint placement
                      (what if current ?)
        """
        self.name = name
        self.joint = joint
        self.frame = frame
        self.type = type
        if self.type == 'SE3':
            self.SE3_value = pin.SE3(pin.rpyToMatrix(np.array([value[3],value[4],value[5]])),np.array([value[0],value[1],value[2]]))
        elif self.type == 'wrench':
            self.wrench_value = pin.Force(value)
        elif self.type == 'current':
            self.current_value = value
        else : 
            raise TypeError("The type of your measurement is not valid")

    def add_SE3_measurement(self,model):
        """ Adds the SE3 measurement to a given model
        """
        
        if self.type == 'SE3':
            self.frame_index=model.addFrame(pin.Frame(self.name,model.getJointId(self.joint),model.getFrameId(self.frame),self.SE3_value,pin.OP_FRAME),'False')
        data = model.createData()
        return model,data

