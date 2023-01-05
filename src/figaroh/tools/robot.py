import sys
import os
import time
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
import pinocchio as pin
from os.path import dirname, join, abspath
from sys import argv
import numpy as np
import hppfcl


class Robot(RobotWrapper):
    def __init__(
        self,
        robot_urdf,
        package_dirs,
        isFext=False,
        isActuator_inertia=False,
        isFrictionincld=False,
        isOffset=False,
        isCoupling=False,
    ):
        # super().__init__()

        # intrinsic dynamic parameter names
        self.params_name = (
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

        # defining conditions
        self.isFext = isFext
        self.isActuator_inertia = isActuator_inertia
        self.isFrictionincld = isFrictionincld
        self.isOffset = isOffset
        self.isCoupling = isCoupling

        # folder location
        self.robot_urdf = robot_urdf

        # manual input addtiional parameters
        # example TX40
        # self.fv = (8.05e0, 5.53e0, 1.97e0, 1.11e0, 1.86e0, 6.5e-1)
        # self.fs = (7.14e0, 8.26e0, 6.34e0, 2.48e0, 3.03e0, 2.82e-1)
        # self.Ia = (3.62e-1, 3.62e-1, 9.88e-2, 3.13e-2, 4.68e-2, 1.05e-2)
        # self.off = (3.92e-1, 1.37e0, 3.26e-1, -1.02e-1, -2.88e-2, 1.27e-1)
        # self.Iam6 = 9.64e-3
        # self.fvm6 = 6.16e-1
        # self.fsm6 = 1.95e0
        # self.N1 = 32
        # self.N2 = 32
        # self.N3 = 45
        # self.N4 = -48
        # self.N5 = 45
        # self.N6 = 32
        # self.qd_lim = 0.01 * \
        #     np.array([287, 287, 430, 410, 320, 700]) * np.pi / 180
        # self.ratio_essential = 30

        # initializing robot's models
        if not isFext:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs)
        else:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs,
                              root_joint=pin.JointModelFreeFlyer())

        # self.geom_model = pin.buildGeomFromUrdf(
        #     self.model, robot_urdf, geom_type=pin.GeometryType.COLLISION,
        #     package_dirs = package_dirs
        # )

        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model

    def get_standard_parameters(self):
        """This function prints out the standard inertial parameters defined in urdf model.
        Output: params_std: a dictionary of parameter names and their values"""
        phi = []
        params = []
        # change order of values in phi['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz'] - from pinoccchio
        # corresponding to params_name ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
        for i in range(1, self.model.njoints):
            P = self.model.inertias[i].toDynamicParameters()
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
            for j in self.params_name:
                if not self.isFext:
                    params.append(j + str(i))
                else:
                    params.append(j + str(i - 1))
            for k in P_mod:
                phi.append(k)
            if self.isActuator_inertia:
                phi.extend([self.Ia[i - 1]])
                params.extend(["Ia" + str(i)])
            if self.isFrictionincld:
                phi.extend([self.fv[i - 1], self.fs[i - 1]])
                params.extend(["fv" + str(i), "fs" + str(i)])
            if self.isOffset:
                phi.extend([self.off[i - 1]])
                params.extend(["off" + str(i)])
        if self.isCoupling:
            phi.extend([self.Iam6, self.fvm6, self.fsm6])
            params.extend(["Iam6", "fvm6", "fsm6"])
        params_std = dict(zip(params, phi))
        return params_std

    def get_standard_parameters_v2(self, param):
    
        model=self.model
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

        # change order of values in phi['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz'] - from pinoccchio
        # corresponding to params_name ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']
        for i in range(len(model.inertias)):
            P =  model.inertias[i].toDynamicParameters()
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
                if not param['is_external_wrench']:#self.isFext:
                    params.append(j + str(i))
                else:
                    params.append(j + str(i-1))
            for k in P_mod:
                phi.append(k)
            #if param['hasActuatorInertia']:
                # phi.extend([self.Ia[i - 1]])
                # params.extend(["Ia" + str(i)])
            if param['has_friction']:
                phi.extend([param['fv'][i-1], param['fv'][i-1]])
                params.extend(["fv" + str(i), "fs" + str(i)])
                # phi.extend([self.fv[i - 1], self.fs[i - 1]])
                # params.extend(["fv" + str(i), "fs" + str(i)])
            #if param['hasJointOffset']:
                #phi.extend([self.off[i - 1]])
                #params.extend(["off" + str(i)])
        #if param['hasCoupledWrist']:#self.isCoupling:
            #phi.extend([self.Iam6, self.fvm6, self.fsm6])
            #params.extend(["Iam6", "fvm6", "fsm6"])
        if param["external_wrench_offsets"]:
            phi.extend([param['OFFX'],param['OFFY'],param['OFFZ']])
            params.extend(["OFFX","OFFY","OFFZ"])
        params_std = dict(zip(params, phi))
        return params_std

    def display_q0(self):
        """If you want to visualize the robot in this example,
        you can choose which visualizer to employ
        by specifying an option from the command line:
        GepettoVisualizer: -g
        MeshcatVisualizer: -m"""
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


def Capsule(name, joint, radius, length, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    # They should be capsules ... but hppfcl current version is buggy with Capsules...
    # hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Cylinder(radius, length)
    geom = pin.GeometryObject(name, joint, hppgeom, placement)
    geom.meshColor = np.array(color)
    return geom


def Box(name, joint, x, y, z, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    # They should be capsules ... but hppfcl current version is buggy with Capsules...
    # hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Box(x, y, z)
    geom = pin.GeometryObject(name, joint, hppgeom, placement)
    geom.meshColor = np.array(color)
    return geom


def Sphere(name, joint, radius, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    hppgeom = hppfcl.Sphere(radius)
    geom = pin.GeometryObject(name, joint, hppgeom, placement)
    geom.meshColor = np.array(color)
    return geom


if __name__ == "__main__":

    # robot = Robot("2DOF_description/urdf", "2DOF_description.urdf")

    # robot = Robot("anymal_b_simple_description/robots", "anymal.urdf")
    # robot = Robot("anymal_b_simple_description/robots", "anymal-kinova.urdf")

    # robot = Robot("double_pendulum_description/urdf", "double_pendulum.urdf")

    # robot = Robot("hector_description/robots", "quadrotor_base.urdf")

    # robot = Robot("hyq_description/robots", "hyq_no_sensors.urdf")

    # robot = Robot("icub_description/robots", "icub_reduced.urdf")
    # robot = Robot("icub_description/robots", "icub.urdf")
    # robot = Robot("kinova_description/robots", "kinova.urdf")

    # robot = Robot("SC_3DOF/urdf", "3DOF.urdf")

    # robot = Robot("solo_description/robots", "solo.urdf")
    # robot = Robot("solo_description/robots", "solo12.urdf")

    # robot = Robot("staubli_tx40_description/urdf", "tx40_mdh_modified.urdf")

    # robot = Robot("ur_description/urdf", "ur5_robot.urdf")
    robot = Robot("tiago_description/robots", "tiago_no_hand.urdf")
    model = robot.model
    data = robot.data
    # print(model)
    # # collision_model = robot.collision_model
    # visual_model = robot.visual_model
    # lower_pos_lim = model.lowerPositionLimit
    # upper_pos_lim = model.upperPositionLimit
    # print(lower_pos_lim)
    # print(upper_pos_lim)

    # geom_model = robot.geom_model
    # xyz_1 = np.array([-0.028, 0, -0.01])
    # xyz_2 = np.array([0.005, 0, -0.35])
    # xyz_3 = np.array([0, 0, 0.20])
    # xyz_4 = np.array([0.005, 0, -0.05])
    # xyz_5 = np.array([-0.03, 0.09, 0])

    # geom_model.addGeometryObject(
    #     Box("torso_up_box", 1, 0.28, 0.35, 0.2, pin.SE3(np.eye(3), xyz_1))
    # )
    # visual_model.addGeometryObject(
    #     Box("torso_up_box", 1, 0.28, 0.35, 0.2, pin.SE3(np.eye(3), xyz_1))
    # )
    # geom_model.addGeometryObject(
    #     Box("torso_low_box", 1, 0.221, 0.26, 0.35, pin.SE3(np.eye(3), xyz_2))
    # )
    # visual_model.addGeometryObject(
    #     Box("torso_low_box", 1, 0.221, 0.26, 0.35, pin.SE3(np.eye(3), xyz_2))
    # )
    # geom_model.addGeometryObject(
    #     Capsule("base_cap", 0, 0.3, 0.25, pin.SE3(np.eye(3), xyz_3))
    # )
    # visual_model.addGeometryObject(
    #     Capsule("base_cap", 0, 0.3, 0.25, pin.SE3(np.eye(3), xyz_3))
    # )
    # # geom_model.addGeometryObject(
    # #     Box("head_box", 9, 0.1, 0.14, 0.1, pin.SE3(np.eye(3), xyz_4))
    # # )
    # # visual_model.addGeometryObject(
    # #     Box("head_box", 9, 0.1, 0.14, 0.1, pin.SE3(np.eye(3), xyz_4))
    # # )
    # geom_model.addGeometryObject(
    #     Capsule("head_cap", 10, 0.17, 0.25, pin.SE3(np.eye(3), xyz_5))
    # )
    # visual_model.addGeometryObject(
    #     Capsule("head_cap", 10, 0.17, 0.25, pin.SE3(np.eye(3), xyz_5))
    # )

    # # for k in range(len(geom_model.geometryObjects)):
    # #     print("object number %d" % k, geom_model.geometryObjects[k].name)

    # arm_link_names = [
    #     "arm_4_link_0",
    #     "arm_5_link_0",
    #     "arm_6_link_0",
    #     "wrist_ft_link_0",
    #     "wrist_ft_tool_link_0",
    # ]
    # arm_link_ids = [geom_model.getGeometryId(k) for k in arm_link_names]
    # mask_link_names = [
    #     "torso_up_box",
    #     "torso_low_box",
    #     "base_cap",
    #     "head_cap",
    # ]
    # mask_link_ids = [geom_model.getGeometryId(k) for k in mask_link_names]
    # for i in mask_link_ids:
    #     for j in arm_link_ids:
    #         geom_model.addCollisionPair(pin.CollisionPair(i, j))

    # # geom_model.addCollisionPair(pin.CollisionPair(30, 19))
    # geom_data = geom_model.createData()

    # q = robot.q0
    # q[4] = 0
    # q[3] = 0
    # q[2] = -0.7
    # q[1] = 1.57079633
    # pin.computeCollisions(model, data, geom_model, geom_data, q, False)

    # # print(pin.computeDistance(geom_model, geom_data, 0).min_distance)
    # for i in range(len(geom_model.collisionPairs)):
    #     print(len(geom_model.collisionPairs))
    #     print(geom_model.collisionPairs[i])
    # # for k in range(len(model.referenceConfigurations)):
    # # print(model.referenceConfigurations[k])
    # # print(model.referenceConfigurations)

    # pin.forwardKinematics(model, data, q)
    # p = robot.data.oMi[7].translation
    robot.display_q0()

    # active_joints = ["torso_lift_joint",
    #                  "arm_1_joint",
    #                  "arm_2_joint",
    #                  "arm_3_joint",
    #                  "arm_4_joint",
    #                  "arm_5_joint",
    #                  "arm_6_joint",
    #                  "arm_7_joint"]

    # act_idx = [model.getJointId(i) for i in active_joints]
    # print(act_idx)

    # # Create data structures
    # collision_model = robot.collision_model
    # visual_model = robot.visual_model
    # viz = MeshcatVisualizer(
    #     model=model, collision_model=collision_model, visual_model=visual_model
    # )
    # try:
    #     viz.initViewer(open=True)
    # except ImportError as err:
    #     print("error initializing, install meshcat")
    #     print(err)
    #     sys.exit(0)
    # viz.loadViewerModel(rootNodeName="robot")
    # time.sleep(3)

    # viz.display(q)
    # print("--------------------")
