import cyipopt
from scipy.optimize import approx_fprime

import numpy as np
import pinocchio as pin


def Rquat(x, y, z, w):
    q = pin.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix()


def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()


def make_markers_dict(PMarkers, markers_name, timestamp, joints_width=None):
    values = [timestamp]
    names = ["time"]
    for ii in range(len(PMarkers)):
        values.append(PMarkers[ii])
        names.append(markers_name[ii])
    if joints_width is not None:
        names.append("LKNEW")
        values.append(joints_width[0])
        names.append("RKNEW")
        values.append(joints_width[1])
        names.append("LANKW")
        values.append(joints_width[2])
        names.append("RANKW")
        values.append(joints_width[3])
        names.append("LELBW")
        values.append(joints_width[4])
        names.append("RELBW")
        values.append(joints_width[5])
    return dict(zip(names, values))


def compute_joint_centers(DMarkers):
    names = [
        "PELC",
        "THOC",
        "LHIPC",
        "LKNEC",
        "LANKC",
        "RHIPC",
        "RKNEC",
        "RANKC",
        "LSHOC",
        "LELBC",
        "LWRC",
        "RSHOC",
        "RELBC",
        "RWRC",
        "HC",
    ]
    positions = []
    ASIS_ASIS = np.linalg.norm(DMarkers["RASI"] - DMarkers["LASI"])
    offset_lasis = np.array([-0.22 * ASIS_ASIS, -0.14 * ASIS_ASIS, -0.3 * ASIS_ASIS])
    offset_rasis = np.array([-0.22 * ASIS_ASIS, 0.14 * ASIS_ASIS, -0.3 * ASIS_ASIS])
    LSHO_RSHO = np.linalg.norm(DMarkers["RSHO"] - DMarkers["LSHO"])
    offset_rshoulder = np.array([0, 0, -0.17 * LSHO_RSHO])
    offset_lshoulder = np.array([0, 0, -0.17 * LSHO_RSHO])
    positions.append(
        (
            (DMarkers["RASI"] + DMarkers["LASI"]) / 2
            + (DMarkers["RPSI"] + DMarkers["LPSI"]) / 2
        )
        / 2
    )  # PELC
    # positions.append(DMarkers['CLAV']+np.array([-0.05,0,0]))#THOC
    positions.append(DMarkers["C7"] + np.array([0.05, 0, 0]))  # THOC
    positions.append(DMarkers["LASI"] + offset_lasis)  # LHIPC
    positions.append(
        DMarkers["LKNE"] + np.array([0, -DMarkers["LKNEW"] / 2, 0])
    )  # LKNEC
    positions.append(
        DMarkers["LANK"] + np.array([0, -DMarkers["LANKW"] / 2, 0])
    )  # LANKC
    positions.append(DMarkers["RASI"] + offset_rasis)  # RHIPC
    positions.append(
        DMarkers["RKNE"] + np.array([0, DMarkers["RKNEW"] / 2, 0])
    )  # RKNEC
    positions.append(
        DMarkers["RANK"] + np.array([0, DMarkers["RANKW"] / 2, 0])
    )  # RANKC
    positions.append(DMarkers["LSHO"] + offset_lshoulder)  # LSHOC
    positions.append(
        DMarkers["LELB"] + np.array([DMarkers["LELBW"] / 2, 0, 0])
    )  # LELBC
    positions.append((DMarkers["LWRA"] + DMarkers["LWRB"]) / 2)  # LWRC
    positions.append(DMarkers["RSHO"] + offset_rshoulder)  # RSHOC
    positions.append(DMarkers["RELB"] + np.array([DMarkers["RELBW"] / 2, 0, 0]))  # RELB
    positions.append((DMarkers["RWRA"] + DMarkers["RWRB"]) / 2)  # RWRC
    positions.append(
        (
            (DMarkers["LFHD"] + DMarkers["RFHD"]) / 2
            + (DMarkers["LBHD"] + DMarkers["RBHD"]) / 2
        )
        / 2
    )  # HC
    return dict(zip(names, positions))


def scale_human_model(model, DMarkers):

    # Get the joint to scale ids

    IDX_JLANKZ = model.getJointId("left_ankle_Z")
    IDX_JRANKZ = model.getJointId("right_ankle_Z")
    IDX_JRKNEE = model.getJointId("right_knee")
    IDX_JLKNEE = model.getJointId("left_knee")
    IDX_JLELBZ = model.getJointId("left_elbow_Z")
    IDX_JRELBZ = model.getJointId("right_elbow_Z")
    IDX_JLWRZ = model.getJointId("left_wrist_Z")
    IDX_JRWRZ = model.getJointId("right_wrist_Z")
    IDX_JLSHOY = model.getJointId("left_shoulder_Y")
    IDX_JRSHOY = model.getJointId("right_shoulder_Y")

    # Retrieve the segments lengths from measures

    l_fem_lenght = np.linalg.norm(DMarkers["LKNEC"] - DMarkers["LHIPC"])
    l_tib_lenght = np.linalg.norm(DMarkers["LANKC"] - DMarkers["LKNEC"])
    l_forearm_lenght = np.linalg.norm(DMarkers["LWRC"] - DMarkers["LELBC"])
    l_upperarm_lenght = np.linalg.norm(DMarkers["LELBC"] - DMarkers["LSHOC"])

    r_fem_lenght = np.linalg.norm(DMarkers["RKNEC"] - DMarkers["RHIPC"])
    r_tib_lenght = np.linalg.norm(DMarkers["RANKC"] - DMarkers["RKNEC"])
    r_forearm_lenght = np.linalg.norm(DMarkers["RWRC"] - DMarkers["RELBC"])
    r_upperarm_lenght = np.linalg.norm(DMarkers["RELBC"] - DMarkers["RSHOC"])

    l_trunk = np.linalg.norm(DMarkers["LSHOC"] - DMarkers["THOC"])
    r_trunk = np.linalg.norm(DMarkers["RSHOC"] - DMarkers["THOC"])

    # KNEES
    model.jointPlacements[IDX_JRKNEE].translation = np.array([0, -r_fem_lenght, 0])
    model.jointPlacements[IDX_JLKNEE].translation = np.array([0, -l_fem_lenght, 0])

    # ELBOWS
    model.jointPlacements[IDX_JRELBZ].translation = np.array([0, -r_upperarm_lenght, 0])
    model.jointPlacements[IDX_JLELBZ].translation = np.array([0, -l_upperarm_lenght, 0])

    # ANKLES
    model.jointPlacements[IDX_JRANKZ].translation = np.array([0, -r_tib_lenght, 0])
    model.jointPlacements[IDX_JLANKZ].translation = np.array([0, -l_tib_lenght, 0])

    # WRISTS
    model.jointPlacements[IDX_JRWRZ].translation = np.array([0, -r_forearm_lenght, 0])
    model.jointPlacements[IDX_JLWRZ].translation = np.array([0, -l_forearm_lenght, 0])

    # SHOULDERS
    model.jointPlacements[IDX_JRSHOY].translation[2] = r_trunk
    model.jointPlacements[IDX_JLSHOY].translation[2] = -l_trunk

    return model


class calibration(object):
    def __init__(self, robot, DMarkers):

        self.DMarkers = DMarkers
        self.robot = robot
        self.model = robot.model
        self.data = robot.data

    def objective(self, x):
        # callback for objective
        q_tPose = np.zeros((43,))
        q_tPose[20] = -np.pi / 2
        q_tPose[31] = -np.pi / 2

        Goal = np.empty(shape=[0, 3])

        for cle, valeur in self.DMarkers.items():
            Goal = np.concatenate((Goal, np.reshape(np.array(valeur), (1, 3))), axis=0)

        ids = []  # retrieve the index of joints of interest to fit
        ids.append(self.model.getJointId("root_joint"))  # PELC
        ids.append(self.model.getJointId("middle_cervical_Z"))  # THOC
        ids.append(self.model.getJointId("left_hip_Z"))  # LHIPC
        ids.append(self.model.getJointId("left_knee"))  # LKNEC
        ids.append(self.model.getJointId("left_ankle_Z"))  # LANKC
        ids.append(self.model.getJointId("right_hip_Z"))  # RHIPC
        ids.append(self.model.getJointId("right_knee"))  # RKNEC
        ids.append(self.model.getJointId("right_ankle_Z"))  # RANKC
        ids.append(self.model.getJointId("left_shoulder_Y"))  # LSHOC
        ids.append(self.model.getJointId("left_elbow_Z"))  # LELBC
        ids.append(self.model.getJointId("left_wrist_Z"))  # LWRC
        ids.append(self.model.getJointId("right_shoulder_Y"))  # RSHOC
        ids.append(self.model.getJointId("right_elbow_Z"))  # RELBC
        ids.append(self.model.getJointId("right_wrist_Z"))  # RWRC

        pin.forwardKinematics(self.model, self.data, x)

        Mposition_joints = np.empty(shape=[0, 3])

        for ii in range(len(ids)):
            Mposition_joints = np.concatenate(
                (
                    Mposition_joints,
                    np.reshape(self.data.oMi[ids[ii]].translation, (1, 3)),
                ),
                axis=0,
            )

        Mposition_joints = np.concatenate(
            (
                Mposition_joints,
                np.reshape(self.data.oMi[ids[1]].translation + [0, 0, 0.25], (1, 3)),
            ),
            axis=0,
        )  # HC

        J = np.sum((Goal - Mposition_joints) ** 2) + 1e-2 * np.sum((q_tPose - x) ** 2)

        return J

    def constraints(self, x):
        """Returns the constraints."""
        return np.linalg.norm(
            [x[3], x[4], x[5], x[6]]
        )  # norm of the freeflyer quaternion equal to 1

    def gradient(self, x):
        # callback for gradient

        G = approx_fprime(x, self.objective, 1e-5)

        return G

    def jacobian(self, x):
        # callback for jacobian of constraints
        jac = approx_fprime(x, self.constraints, 1e-5)

        return jac


def calibrate_human_model(robot, DMarkers, q0):

    lb = robot.model.lowerPositionLimit  # lower joint limits
    ub = robot.model.upperPositionLimit  # upper joint limits
    cl = cu = [1]

    nlp = cyipopt.Problem(
        n=len(q0),
        m=len(cl),
        problem_obj=calibration(robot, DMarkers),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    nlp.add_option("tol", 1e-2)
    nlp.add_option("print_level", 0)
    q_opt, info = nlp.solve(q0)

    return q_opt


def get_local_markers(model, data, q0, markers_global):
    local_values = []
    names = [
        "LFHD",
        "RFHD",
        "LBHD",
        "RBHD",
        "C7",
        "T10",
        "CLAV",
        "STRN",
        "RBAK",
        "LSHO",
        "LELB",
        "LWRA",
        "LWRB",
        "LFIN",
        "RSHO",
        "RELB",
        "RWRA",
        "RWRB",
        "RFIN",
        "LASI",
        "RASI",
        "LPSI",
        "RPSI",
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        "LHEE",
        "LTOE",
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        "RHEE",
        "RTOE",
    ]
    pin.forwardKinematics(model, data, q0)
    oMi = dict(zip(model.names, data.oMi))
    local_values.append(
        oMi["middle_cervical_Z"].inverse().rotation
        @ (markers_global["LFHD"] - oMi["middle_cervical_Z"].translation)
    )
    local_values.append(
        oMi["middle_cervical_Z"].inverse().rotation
        @ (markers_global["RFHD"] - oMi["middle_cervical_Z"].translation)
    )
    local_values.append(
        oMi["middle_cervical_Z"].inverse().rotation
        @ (markers_global["LBHD"] - oMi["middle_cervical_Z"].translation)
    )
    local_values.append(
        oMi["middle_cervical_Z"].inverse().rotation
        @ (markers_global["RBHD"] - oMi["middle_cervical_Z"].translation)
    )
    local_values.append(
        oMi["middle_thoracic_Z"].inverse().rotation
        @ (markers_global["C7"] - oMi["middle_thoracic_Z"].translation)
    )
    local_values.append(
        oMi["root_joint"].inverse().rotation
        @ (markers_global["T10"] - oMi["root_joint"].translation)
    )
    local_values.append(
        oMi["middle_thoracic_Z"].inverse().rotation
        @ (markers_global["CLAV"] - oMi["middle_thoracic_Z"].translation)
    )
    local_values.append(
        oMi["middle_thoracic_Z"].inverse().rotation
        @ (markers_global["STRN"] - oMi["middle_thoracic_Z"].translation)
    )
    local_values.append(
        oMi["middle_thoracic_Z"].inverse().rotation
        @ (markers_global["RBAK"] - oMi["middle_thoracic_Z"].translation)
    )
    local_values.append(
        oMi["left_shoulder_Y"].inverse().rotation
        @ (markers_global["LSHO"] - oMi["left_shoulder_Y"].translation)
    )
    local_values.append(
        oMi["left_elbow_Z"].inverse().rotation
        @ (markers_global["LELB"] - oMi["left_elbow_Z"].translation)
    )
    local_values.append(
        oMi["left_wrist_Z"].inverse().rotation
        @ (markers_global["LWRA"] - oMi["left_wrist_Z"].translation)
    )
    local_values.append(
        oMi["left_wrist_Z"].inverse().rotation
        @ (markers_global["LWRB"] - oMi["left_wrist_Z"].translation)
    )
    local_values.append(
        oMi["left_wrist_Z"].inverse().rotation
        @ (markers_global["LFIN"] - oMi["left_wrist_Z"].translation)
    )
    local_values.append(
        oMi["right_shoulder_Y"].inverse().rotation
        @ (markers_global["RSHO"] - oMi["right_shoulder_Y"].translation)
    )
    local_values.append(
        oMi["right_elbow_Z"].inverse().rotation
        @ (markers_global["RELB"] - oMi["right_elbow_Z"].translation)
    )
    local_values.append(
        oMi["right_wrist_Z"].inverse().rotation
        @ (markers_global["RWRA"] - oMi["right_wrist_Z"].translation)
    )
    local_values.append(
        oMi["right_wrist_Z"].inverse().rotation
        @ (markers_global["RWRB"] - oMi["right_wrist_Z"].translation)
    )
    local_values.append(
        oMi["right_wrist_Z"].inverse().rotation
        @ (markers_global["RFIN"] - oMi["right_wrist_Z"].translation)
    )
    local_values.append(
        oMi["root_joint"].inverse().rotation
        @ (markers_global["LASI"] - oMi["root_joint"].translation)
    )
    local_values.append(
        oMi["root_joint"].inverse().rotation
        @ (markers_global["RASI"] - oMi["root_joint"].translation)
    )
    local_values.append(
        oMi["root_joint"].inverse().rotation
        @ (markers_global["LPSI"] - oMi["root_joint"].translation)
    )
    local_values.append(
        oMi["root_joint"].inverse().rotation
        @ (markers_global["RPSI"] - oMi["root_joint"].translation)
    )
    local_values.append(
        oMi["left_hip_Z"].inverse().rotation
        @ (markers_global["LTHI"] - oMi["left_hip_Z"].translation)
    )
    local_values.append(
        oMi["left_knee"].inverse().rotation
        @ (markers_global["LKNE"] - oMi["left_knee"].translation)
    )
    local_values.append(
        oMi["left_knee"].inverse().rotation
        @ (markers_global["LTIB"] - oMi["left_knee"].translation)
    )
    local_values.append(
        oMi["left_ankle_Z"].inverse().rotation
        @ (markers_global["LANK"] - oMi["left_ankle_Z"].translation)
    )
    local_values.append(
        oMi["left_ankle_Z"].inverse().rotation
        @ (markers_global["LHEE"] - oMi["left_ankle_Z"].translation)
    )
    local_values.append(
        oMi["left_ankle_Z"].inverse().rotation
        @ (markers_global["LTOE"] - oMi["left_ankle_Z"].translation)
    )
    local_values.append(
        oMi["right_hip_Z"].inverse().rotation
        @ (markers_global["RTHI"] - oMi["right_hip_Z"].translation)
    )
    local_values.append(
        oMi["right_knee"].inverse().rotation
        @ (markers_global["RKNE"] - oMi["right_knee"].translation)
    )
    local_values.append(
        oMi["right_knee"].inverse().rotation
        @ (markers_global["RTIB"] - oMi["right_knee"].translation)
    )
    local_values.append(
        oMi["right_ankle_Z"].inverse().rotation
        @ (markers_global["RANK"] - oMi["right_ankle_Z"].translation)
    )
    local_values.append(
        oMi["right_ankle_Z"].inverse().rotation
        @ (markers_global["RHEE"] - oMi["right_ankle_Z"].translation)
    )
    local_values.append(
        oMi["right_ankle_Z"].inverse().rotation
        @ (markers_global["RTOE"] - oMi["right_ankle_Z"].translation)
    )
    return dict(zip(names, local_values))


def markers_local_for_df(markers_local):
    names = [
        "LFHDx",
        "LFHDy",
        "LFHDz",
        "RFHDx",
        "RFHDy",
        "RFHDz",
        "LBHDx",
        "LBHDy",
        "LBHDz",
        "RBHDx",
        "RBHDy",
        "RBHDz",
        "C7x",
        "C7y",
        "C7z",
        "T10x",
        "T10y",
        "T10z",
        "CLAVx",
        "CLAVy",
        "CLAVz",
        "STRNx",
        "STRNy",
        "STRNz",
        "RBAKx",
        "RBAKy",
        "RBAKz",
        "LSHOx",
        "LSHOy",
        "LSHOz",
        "LELBx",
        "LELBy",
        "LELBz",
        "LWRAx",
        "LWRAy",
        "LWRAz",
        "LWRBx",
        "LWRBy",
        "LWRBz",
        "LFINx",
        "LFINy",
        "LFINz",
        "RSHOx",
        "RSHOy",
        "RSHOz",
        "RELBx",
        "RELBy",
        "RELBz",
        "RWRAx",
        "RWRAy",
        "RWRAz",
        "RWRBx",
        "RWRBy",
        "RWRBz",
        "RFINx",
        "RFINy",
        "RFINz",
        "LASIx",
        "LASIy",
        "LASIz",
        "RASIx",
        "RASIy",
        "RASIz",
        "LPSIx",
        "LPSIy",
        "LPSIz",
        "RPSIx",
        "RPSIy",
        "RPSIz",
        "LTHIx",
        "LTHIy",
        "LTHIz",
        "LKNEx",
        "LKNEy",
        "LKNEz",
        "LTIBx",
        "LTIBy",
        "LTIBz",
        "LANKx",
        "LANKy",
        "LANKz",
        "LHEEx",
        "LHEEy",
        "LHEEz",
        "LTOEx",
        "LTOEy",
        "LTOEz",
        "RTHIx",
        "RTHIy",
        "RTHIz",
        "RKNEx",
        "RKNEy",
        "RKNEz",
        "RTIBx",
        "RTIBy",
        "RTIBz",
        "RANKx",
        "RANKy",
        "RANKz",
        "RHEEx",
        "RHEEy",
        "RHEEz",
        "RTOEx",
        "RTOEy",
        "RTOEz",
    ]
    values = []
    for ii in [
        "LFHD",
        "RFHD",
        "LBHD",
        "RBHD",
        "C7",
        "T10",
        "CLAV",
        "STRN",
        "RBAK",
        "LSHO",
        "LELB",
        "LWRA",
        "LWRB",
        "LFIN",
        "RSHO",
        "RELB",
        "RWRA",
        "RWRB",
        "RFIN",
        "LASI",
        "RASI",
        "LPSI",
        "RPSI",
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        "LHEE",
        "LTOE",
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        "RHEE",
        "RTOE",
    ]:
        values.append(markers_local[ii][0])
        values.append(markers_local[ii][1])
        values.append(markers_local[ii][2])
    return dict(zip(names, values))


def mean_local_markers(mean):
    names = [
        "LFHD",
        "RFHD",
        "LBHD",
        "RBHD",
        "C7",
        "T10",
        "CLAV",
        "STRN",
        "RBAK",
        "LSHO",
        "LELB",
        "LWRA",
        "LWRB",
        "LFIN",
        "RSHO",
        "RELB",
        "RWRA",
        "RWRB",
        "RFIN",
        "LASI",
        "RASI",
        "LPSI",
        "RPSI",
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        "LHEE",
        "LTOE",
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        "RHEE",
        "RTOE",
    ]
    values = []
    values.append(np.array([mean["LFHDx"], mean["LFHDy"], mean["LFHDz"]]))
    values.append(np.array([mean["RFHDx"], mean["RFHDy"], mean["RFHDz"]]))
    values.append(np.array([mean["LBHDx"], mean["LBHDy"], mean["LBHDz"]]))
    values.append(np.array([mean["RBHDx"], mean["RBHDy"], mean["RBHDz"]]))
    values.append(np.array([mean["C7x"], mean["C7y"], mean["C7z"]]))
    values.append(np.array([mean["T10x"], mean["T10y"], mean["T10z"]]))
    values.append(np.array([mean["CLAVx"], mean["CLAVy"], mean["CLAVz"]]))
    values.append(np.array([mean["STRNx"], mean["STRNy"], mean["STRNz"]]))
    values.append(np.array([mean["RBAKx"], mean["RBAKy"], mean["RBAKz"]]))
    values.append(np.array([mean["LSHOx"], mean["LSHOy"], mean["LSHOz"]]))
    values.append(np.array([mean["LELBx"], mean["LELBy"], mean["LELBz"]]))
    values.append(np.array([mean["LWRAx"], mean["LWRAy"], mean["LWRAz"]]))
    values.append(np.array([mean["LWRBx"], mean["LWRBy"], mean["LWRBz"]]))
    values.append(np.array([mean["LFINx"], mean["LFINy"], mean["LFINz"]]))
    values.append(np.array([mean["RSHOx"], mean["RSHOy"], mean["RSHOz"]]))
    values.append(np.array([mean["RELBx"], mean["RELBy"], mean["RELBz"]]))
    values.append(np.array([mean["RWRAx"], mean["RWRAy"], mean["RWRAz"]]))
    values.append(np.array([mean["RWRBx"], mean["RWRBy"], mean["RWRBz"]]))
    values.append(np.array([mean["RFINx"], mean["RFINy"], mean["RFINz"]]))
    values.append(np.array([mean["LASIx"], mean["LASIy"], mean["LASIz"]]))
    values.append(np.array([mean["RASIx"], mean["RASIy"], mean["RASIz"]]))
    values.append(np.array([mean["LPSIx"], mean["LPSIy"], mean["LPSIz"]]))
    values.append(np.array([mean["RPSIx"], mean["RPSIy"], mean["RPSIz"]]))
    values.append(np.array([mean["LTHIx"], mean["LTHIy"], mean["LTHIz"]]))
    values.append(np.array([mean["LKNEx"], mean["LKNEy"], mean["LKNEz"]]))
    values.append(np.array([mean["LTIBx"], mean["LTIBy"], mean["LTIBz"]]))
    values.append(np.array([mean["LANKx"], mean["LANKy"], mean["LANKz"]]))
    values.append(np.array([mean["LHEEx"], mean["LHEEy"], mean["LHEEz"]]))
    values.append(np.array([mean["LTOEx"], mean["LTOEy"], mean["LTOEz"]]))
    values.append(np.array([mean["RTHIx"], mean["RTHIy"], mean["RTHIz"]]))
    values.append(np.array([mean["RKNEx"], mean["RKNEy"], mean["RKNEz"]]))
    values.append(np.array([mean["RTIBx"], mean["RTIBy"], mean["RTIBz"]]))
    values.append(np.array([mean["RANKx"], mean["RANKy"], mean["RANKz"]]))
    values.append(np.array([mean["RHEEx"], mean["RHEEy"], mean["RHEEz"]]))
    values.append(np.array([mean["RTOEx"], mean["RTOEy"], mean["RTOEz"]]))
    return dict(zip(names, values))
