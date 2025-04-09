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

from sys import argv
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer


class Robot(RobotWrapper):
    """Robot class extending Pinocchio's RobotWrapper with additional features."""

    def __init__(
        self,
        robot_urdf,
        package_dirs,
        isFext=False,
        freeflyer_ori=None,
    ):
        """Initialize robot model from URDF.
        
        Args:
            robot_urdf: Path to URDF file
            package_dirs: Package directories for mesh files
            isFext: Whether to add floating base joint
            freeflyer_ori: Optional orientation for floating base
        """
        # Intrinsic dynamic parameter names 
        self.params_name = (
            "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz",
            "mx", "my", "mz", "m"
        )

        self.isFext = isFext
        self.robot_urdf = robot_urdf

        # Initialize robot model
        if not isFext:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs)
        else:
            self.initFromURDF(
                robot_urdf,
                package_dirs=package_dirs, 
                root_joint=pin.JointModelFreeFlyer()
            )

        # Set floating base parameters if provided
        if freeflyer_ori is not None and isFext:
            joint_id = self.model.getJointId("root_joint")
            self.model.jointPlacements[joint_id].rotation = freeflyer_ori
            
            # Update position limits
            ub = self.model.upperPositionLimit
            lb = self.model.lowerPositionLimit
            ub[:7] = 1
            lb[:7] = -1
            self.model.upperPositionLimit = ub
            self.model.lowerPositionLimit = lb
            self.data = self.model.createData()

        self.geom_model = self.collision_model

    def get_standard_parameters(self, param):
        """Get standard inertial parameters from URDF model.
        
        Args:
            param: Dictionary of parameter settings

        Returns:
            dict: Parameter names mapped to values
        """
        model = self.model
        phi = []
        params = []

        params_name = (
            "Ixx",
            "Ixy",
            "Ixz",
            "Iyy",
            "Iyz",
            "Izz",
            "mx",
            "my",
            "mz",
            "m",
        )

        # Change order of values in phi['m', 'mx','my','mz','Ixx','Ixy','Iyy',
        # 'Ixz', 'Iyz', 'Izz'] - from Pinocchio
        # Corresponding to params_name ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz',
        # 'mx','my','mz','m']

        for i in range(1, len(model.inertias)):
            P = model.inertias[i].toDynamicParameters()
            P_mod = np.zeros(P.shape[0])
            P_mod[9] = P[0]  # m
            P_mod[8] = P[3]  # mz
            P_mod[7] = P[2]  # my
            P_mod[6] = P[1]  # mx
            P_mod[5] = P[9]  # Izz
            P_mod[4] = P[8]  # Iyz
            P_mod[3] = P[6]  # Iyy
            P_mod[2] = P[7]  # Ixz
            P_mod[1] = P[5]  # Ixy
            P_mod[0] = P[4]  # Ixx
            for j in params_name:
                params.append(j + str(i))
            for k in P_mod:
                phi.append(k)

            params.extend(["Ia" + str(i)])
            params.extend(["fv" + str(i), "fs" + str(i)])
            params.extend(["off" + str(i)])

            if param["has_actuator_inertia"]:
                try:
                    phi.extend([param["Ia"][i - 1]])
                except Exception as e:
                    print("Warning: ", "has_actuator_inertia_%d" % i, e)
                    phi.extend([0])
            else:
                phi.extend([0])
            if param["has_friction"]:
                try:
                    phi.extend([param["fv"][i - 1], param["fs"][i - 1]])
                except Exception as e:
                    print("Warning: ", "has_friction_%d" % i, e)
                    phi.extend([0, 0])
            else:
                phi.extend([0, 0])
            if param["has_joint_offset"]:
                try:
                    phi.extend([param["off"][i - 1]])
                except Exception as e:
                    print("Warning: ", "has_joint_offset_%d" % i, e)
                    phi.extend([0])
            else:
                phi.extend([0])

        params_std = dict(zip(params, phi))
        return params_std

    def display_q0(self):
        """Display robot in initial configuration.
        
        Uses either Gepetto (-g) or Meshcat (-m) visualizer based on 
        command line argument.
        """
        VISUALIZER = None
        if len(argv) > 1:
            opt = argv[1]
            if opt == "-g":
                VISUALIZER = GepettoVisualizer
            elif opt == "-m":
                VISUALIZER = MeshcatVisualizer

        if VISUALIZER:
            self.setVisualizer(VISUALIZER())
            self.initViewer()
            self.loadViewerModel(self.robot_urdf)
            q = self.q0

            self.display(q)
