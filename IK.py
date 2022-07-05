import numpy as np


import numpy as np
import pinocchio as pin
from os.path import dirname, join, abspath

from parameters_settings import params_settings
from pinocchio.visualize import GepettoVisualizer
from tools.robot import load_model
from pinocchio.utils import *

import quadprog
import cyipopt
from scipy.optimize import approx_fprime


def Rquat(x, y, z, w):
    q = pin.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix()


def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()


def add_markers_frames(model, data, markers_local):
    i_markers = []

    # Get the index of frames of interest
    IDX_LF = model.getFrameId("left_foot")
    IDX_LH = model.getFrameId("left_hand")
    IDX_LLA = model.getFrameId("left_lowerarm")
    IDX_LLL = model.getFrameId("left_lowerleg")
    IDX_LUA = model.getFrameId("left_upperarm")
    IDX_LUL = model.getFrameId("left_upperleg")
    IDX_MH = model.getFrameId("middle_head")
    IDX_MP = model.getFrameId("middle_pelvis")
    IDX_MT = model.getFrameId("middle_thorax")
    IDX_RF = model.getFrameId("right_foot")
    IDX_RH = model.getFrameId("right_hand")
    IDX_RLA = model.getFrameId("right_lowerarm")
    IDX_RLL = model.getFrameId("right_lowerleg")
    IDX_RUA = model.getFrameId("right_upperarm")
    IDX_RUL = model.getFrameId("right_upperleg")

    # Get the index of joints of interest

    IDX_JLF = model.getJointId("left_ankle_Z")
    IDX_JLH = model.getJointId("left_wrist_Z")
    IDX_JLLA = model.getJointId("left_elbow_Z")
    IDX_JLLL = model.getJointId("left_knee")
    IDX_JLUA = model.getJointId("left_shoulder_Y")
    IDX_JLUL = model.getJointId("left_hip_Y")
    IDX_JMH = model.getJointId("middle_cervical_Z")
    IDX_JMP = model.getJointId("root_joint")
    IDX_JMT = model.getJointId("middle_thoracic_Z")
    IDX_JRF = model.getJointId("right_ankle_Z")
    IDX_JRH = model.getJointId("right_wrist_Z")
    IDX_JRLA = model.getJointId("right_elbow_Z")
    IDX_JRLL = model.getJointId("right_knee")
    IDX_JRUA = model.getJointId("right_shoulder_Y")
    IDX_JRUL = model.getJointId("right_hip_Z")

    # 35 MARKERS FRAMES TO ADD

    inertia = pin.Inertia.Zero()

    F1 = pin.Frame(
        "marker1",
        IDX_JMP,
        IDX_MP,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LASI"][0],
                    markers_local["LASI"][1],
                    markers_local["LASI"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i1 = model.addFrame(F1, False)
    i_markers.append(i1)
    F2 = pin.Frame(
        "marker2",
        IDX_JMP,
        IDX_MP,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RASI"][0],
                    markers_local["RASI"][1],
                    markers_local["RASI"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i2 = model.addFrame(F2, False)
    i_markers.append(i2)
    F3 = pin.Frame(
        "marker3",
        IDX_JMP,
        IDX_MP,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LPSI"][0],
                    markers_local["LPSI"][1],
                    markers_local["LPSI"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i3 = model.addFrame(F3, False)
    i_markers.append(i3)
    F4 = pin.Frame(
        "marker4",
        IDX_JMP,
        IDX_MP,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RPSI"][0],
                    markers_local["RPSI"][1],
                    markers_local["RPSI"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i4 = model.addFrame(F4, False)
    i_markers.append(i4)

    F5 = pin.Frame(
        "marker5",
        IDX_JLUL,
        IDX_LUL,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LTHI"][0],
                    markers_local["LTHI"][1],
                    markers_local["LTHI"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i5 = model.addFrame(F5, False)
    i_markers.append(i5)
    F6 = pin.Frame(
        "marker6",
        IDX_JLLL,
        IDX_LLL,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LKNE"][0],
                    markers_local["LKNE"][1],
                    markers_local["LKNE"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i6 = model.addFrame(F6, False)
    i_markers.append(i6)
    F7 = pin.Frame(
        "marker7",
        IDX_JLLL,
        IDX_LLL,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LTIB"][0],
                    markers_local["LTIB"][1],
                    markers_local["LTIB"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i7 = model.addFrame(F7, False)
    i_markers.append(i7)
    F8 = pin.Frame(
        "marker8",
        IDX_JLF,
        IDX_LF,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LANK"][0],
                    markers_local["LANK"][1],
                    markers_local["LANK"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i8 = model.addFrame(F8, False)
    i_markers.append(i8)
    F9 = pin.Frame(
        "marker9",
        IDX_JLF,
        IDX_LF,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LHEE"][0],
                    markers_local["LHEE"][1],
                    markers_local["LHEE"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i9 = model.addFrame(F9, False)
    i_markers.append(i9)
    F10 = pin.Frame(
        "marker10",
        IDX_JLF,
        IDX_LF,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LTOE"][0],
                    markers_local["LTOE"][1],
                    markers_local["LTOE"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i10 = model.addFrame(F10, False)
    i_markers.append(i10)

    F11 = pin.Frame(
        "marker11",
        IDX_JRUL,
        IDX_RUL,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RTHI"][0],
                    markers_local["RTHI"][1],
                    markers_local["RTHI"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i11 = model.addFrame(F11, False)
    i_markers.append(i11)
    F12 = pin.Frame(
        "marker12",
        IDX_JRLL,
        IDX_RLL,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RKNE"][0],
                    markers_local["RKNE"][1],
                    markers_local["RKNE"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i12 = model.addFrame(F12, False)
    i_markers.append(i12)
    F13 = pin.Frame(
        "marker13",
        IDX_JRLL,
        IDX_RLL,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RTIB"][0],
                    markers_local["RTIB"][1],
                    markers_local["RTIB"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i13 = model.addFrame(F13, False)
    i_markers.append(i13)
    F14 = pin.Frame(
        "marker14",
        IDX_JRF,
        IDX_RF,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RANK"][0],
                    markers_local["RANK"][1],
                    markers_local["RANK"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i14 = model.addFrame(F14, False)
    i_markers.append(i14)
    F15 = pin.Frame(
        "marker15",
        IDX_JRF,
        IDX_RF,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RHEE"][0],
                    markers_local["RHEE"][1],
                    markers_local["RHEE"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i15 = model.addFrame(F15, False)
    i_markers.append(i15)
    F16 = pin.Frame(
        "marker16",
        IDX_JRF,
        IDX_RF,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RTOE"][0],
                    markers_local["RTOE"][1],
                    markers_local["RTOE"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i16 = model.addFrame(F16, False)
    i_markers.append(i16)

    F17 = pin.Frame(
        "marker17",
        IDX_JMH,
        IDX_MH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LFHD"][0],
                    markers_local["LFHD"][1],
                    markers_local["LFHD"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i17 = model.addFrame(F17, False)
    i_markers.append(i17)
    F18 = pin.Frame(
        "marker18",
        IDX_JMH,
        IDX_MH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RFHD"][0],
                    markers_local["RFHD"][1],
                    markers_local["RFHD"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i18 = model.addFrame(F18, False)
    i_markers.append(i18)
    F19 = pin.Frame(
        "marker19",
        IDX_JMH,
        IDX_MH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LBHD"][0],
                    markers_local["LBHD"][1],
                    markers_local["LBHD"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i19 = model.addFrame(F19, False)
    i_markers.append(i19)
    F20 = pin.Frame(
        "marker20",
        IDX_JMH,
        IDX_MH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RBHD"][0],
                    markers_local["RBHD"][1],
                    markers_local["RBHD"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i20 = model.addFrame(F20, False)
    i_markers.append(i20)

    F21 = pin.Frame(
        "marker21",
        IDX_JMT,
        IDX_MT,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [markers_local["C7"][0], markers_local["C7"][1], markers_local["C7"][2]]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i21 = model.addFrame(F21, False)
    i_markers.append(i21)
    F22 = pin.Frame(
        "marker22",
        IDX_JMP,
        IDX_MP,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["T10"][0],
                    markers_local["T10"][1],
                    markers_local["T10"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i22 = model.addFrame(F22, False)
    i_markers.append(i22)
    F23 = pin.Frame(
        "marker23",
        IDX_JMT,
        IDX_MT,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["CLAV"][0],
                    markers_local["CLAV"][1],
                    markers_local["CLAV"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i23 = model.addFrame(F23, False)
    i_markers.append(i23)
    F24 = pin.Frame(
        "marker24",
        IDX_JMT,
        IDX_MT,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["STRN"][0],
                    markers_local["STRN"][1],
                    markers_local["STRN"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i24 = model.addFrame(F24, False)
    i_markers.append(i24)
    F25 = pin.Frame(
        "marker25",
        IDX_JMT,
        IDX_MT,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RBAK"][0],
                    markers_local["RBAK"][1],
                    markers_local["RBAK"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i25 = model.addFrame(F25, False)
    i_markers.append(i25)

    F26 = pin.Frame(
        "marker26",
        IDX_JLUA,
        IDX_LUA,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LSHO"][0],
                    markers_local["LSHO"][1],
                    markers_local["LSHO"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i26 = model.addFrame(F26, False)
    i_markers.append(i26)
    F27 = pin.Frame(
        "marker27",
        IDX_JLLA,
        IDX_LLA,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LELB"][0],
                    markers_local["LELB"][1],
                    markers_local["LELB"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i27 = model.addFrame(F27, False)
    i_markers.append(i27)
    F28 = pin.Frame(
        "marker28",
        IDX_JLH,
        IDX_LH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LWRA"][0],
                    markers_local["LWRA"][1],
                    markers_local["LWRA"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i28 = model.addFrame(F28, False)
    i_markers.append(i28)
    F29 = pin.Frame(
        "marker29",
        IDX_JLH,
        IDX_LH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LWRB"][0],
                    markers_local["LWRB"][1],
                    markers_local["LWRB"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i29 = model.addFrame(F29, False)
    i_markers.append(i29)
    F30 = pin.Frame(
        "marker30",
        IDX_JLH,
        IDX_LH,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["LFIN"][0],
                    markers_local["LFIN"][1],
                    markers_local["LFIN"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i30 = model.addFrame(F30, False)
    i_markers.append(i30)

    F31 = pin.Frame(
        "marker31",
        IDX_JRUA,
        IDX_RUA,
        pin.SE3(
            Rquat(1, 0, 0, 0),
            np.matrix(
                [
                    markers_local["RSHO"][0],
                    markers_local["RSHO"][1],
                    markers_local["RSHO"][2],
                ]
            ).T,
        ),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i31 = model.addFrame(F31, False)
    i_markers.append(i31)
    F32 = pin.Frame(
        "marker32",
        IDX_JRLA,
        IDX_RLA,
        pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local["RELB"]).T),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i32 = model.addFrame(F32, False)
    i_markers.append(i32)
    F33 = pin.Frame(
        "marker33",
        IDX_JRH,
        IDX_RH,
        pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local["RWRA"]).T),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i33 = model.addFrame(F33, False)
    i_markers.append(i33)
    F34 = pin.Frame(
        "marker34",
        IDX_JRH,
        IDX_RH,
        pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local["RWRB"]).T),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i34 = model.addFrame(F34, False)
    i_markers.append(i34)
    F35 = pin.Frame(
        "marker35",
        IDX_JRH,
        IDX_RH,
        pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local["RFIN"]).T),
        pin.FrameType.OP_FRAME,
        inertia,
    )
    i35 = model.addFrame(F35, False)
    i_markers.append(i35)

    data = model.createData()

    return data, i_markers


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = 0.5 * (P + P.T) + np.eye(P.shape[0]) * (
        1e-5
    )  # make sure P is symmetric, pos,def
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def IK_markers_human_quadprog(model, data, DMarkers, i_markers, q0):

    Goal = np.empty(shape=[0, 3])

    for ii in [
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
    ]:
        Goal = np.concatenate(
            (Goal, np.reshape(np.array(DMarkers[ii]), (1, 3))), axis=0
        )
        Mposition_goal_temp = pin.SE3(
            Rquat(1, 0, 0, 0), np.matrix(np.array(DMarkers[ii])).T
        )
        # viz.viewer.gui.addXYZaxis('world/framegoal'+str(ii), [1, 0., 0., 1.], 0.0115, 0.15)
        # place(viz,'world/framegoal'+str(ii), Mposition_goal_temp)

    pin.framesForwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    Mposition_markers = []
    for ii in range(len(i_markers)):
        Mposition_markers.append(data.oMf[i_markers[ii]].translation)

    K_ii = 0.5

    P = np.zeros((len(q0) - 1, len(q0) - 1))
    q = np.zeros((len(q0) - 1,))
    G = np.zeros((len(q0) - 1, len(q0) - 1))
    h = np.zeros((len(q0) - 1, 1))

    dt = 1
    damping = 0.001

    rmse = np.sqrt(np.mean((np.array(Goal) - np.array(Mposition_markers)) ** 2))

    ### ----------------------------------------------------------------------------- ###
    ### OPTIMIZATION PROBLEM

    while rmse > 0.015:
        # print(rmse)
        pin.forwardKinematics(model, data, q0)  # Compute joint placements
        pin.updateFramePlacements(
            model, data
        )  # Also compute operational frame placements

        for ii in range(len(i_markers)):
            Mposition_markers[ii] = data.oMf[i_markers[ii]].translation
            # viz.viewer.gui.addXYZaxis('world/framemarkers'+str(ii), [0., 0., 1., 1.], 0.0115, 0.05)
            # place(viz,'world/framemarkers'+str(ii), pin.SE3(Rquat(1, 0, 0, 0), Mposition_markers[ii]))

            v_ii = (np.array(Goal[ii]) - np.array(Mposition_markers[ii])) / dt
            mu_ii = damping * np.matmul(v_ii, v_ii)

            J_ii = pin.computeFrameJacobian(
                model, data, q0, i_markers[ii], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_ii_reduced = J_ii[:3, :]

            P_ii = np.matmul(J_ii_reduced.T, J_ii_reduced) + mu_ii * np.eye(len(q0) - 1)
            P += P_ii

            q_ii = np.matmul(-K_ii * v_ii.T, J_ii_reduced)
            q += q_ii

        dq = quadprog_solve_qp(P, q, G, h.reshape((len(q0) - 1,)))

        q0 = pin.integrate(model, q0, dq * dt)
        q0 = pin.normalize(model, q0)
        # viz.display(q0)

        rmse = np.sqrt(np.mean((np.array(Goal) - np.array(Mposition_markers)) ** 2))

    return q0


def IK_markers_human(viz, model, data, DMarkers, i_markers, q0):

    Goal = np.empty(shape=[0, 3])

    for ii in [
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
    ]:
        Goal = np.concatenate(
            (Goal, np.reshape(np.array(DMarkers[ii]), (1, 3))), axis=0
        )
        Mposition_goal_temp = pin.SE3(
            Rquat(1, 0, 0, 0), np.matrix(np.array(DMarkers[ii])).T
        )
        viz.viewer.gui.addXYZaxis(
            "world/framegoal" + str(ii), [1, 0.0, 0.0, 1.0], 0.0115, 0.15
        )
        place(viz, "world/framegoal" + str(ii), Mposition_goal_temp)

    pin.framesForwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    Mposition_markers = np.empty(shape=[0, 3])

    for ii in range(len(i_markers)):
        Mposition_markers = np.concatenate(
            (
                Mposition_markers,
                np.reshape(data.oMf[i_markers[ii]].translation, (1, 3)),
            ),
            axis=0,
        )

    rmse = np.sqrt(np.mean((np.array(Goal) - np.array(Mposition_markers)) ** 2))

    dt = 0.5
    dq = np.zeros(len(q0) - 1)
    J = np.empty(shape=[0, len(q0) - 1])
    p = np.array([])

    ### ----------------------------------------------------------------------------- ###
    ### OPTIMIZATION PROBLEM

    while rmse > 0.02:
        # print(rmse)
        pin.forwardKinematics(model, data, q0)  # Compute joint placements
        pin.updateFramePlacements(
            model, data
        )  # Also compute operational frame placements

        for ii in range(len(i_markers)):
            Mposition_markers[ii] = data.oMf[i_markers[ii]].translation
            viz.viewer.gui.addXYZaxis(
                "world/framemarkers" + str(ii), [0.0, 0.0, 1.0, 1.0], 0.0115, 0.05
            )
            place(
                viz,
                "world/framemarkers" + str(ii),
                pin.SE3(Rquat(1, 0, 0, 0), Mposition_markers[ii]),
            )

            p_ii = np.array(Goal[ii]) - np.array(Mposition_markers[ii])
            p = np.concatenate((p, p_ii), axis=0)

            J_ii = pin.computeFrameJacobian(
                model, data, q0, i_markers[ii], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_ii = J_ii[:3, :]

            J = np.concatenate((J, J_ii), axis=0)

        dq = np.linalg.pinv(J) @ p
        q0 = pin.integrate(model, q0, dq * dt)
        q0 = pin.normalize(model, q0)
        viz.display(q0)

        rmse = np.sqrt(np.mean((np.array(Goal) - np.array(Mposition_markers)) ** 2))
    return q0


class initialisation(object):
    def __init__(self, model, data, DMarkers, i_markers):

        self.DMarkers = DMarkers
        self.i_markers = i_markers
        self.model = model
        self.data = data

    def objective(self, x):
        # callback for objective

        Goal = np.empty(shape=[0, 3])

        for ii in [
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
        ]:
            Goal = np.concatenate(
                (Goal, np.reshape(np.array(self.DMarkers[ii]), (1, 3))), axis=0
            )

        pin.forwardKinematics(self.model, self.data, x)
        pin.updateFramePlacements(self.model, self.data)

        Mposition_markers = []
        for ii in range(len(self.i_markers)):
            Mposition_markers.append(self.data.oMf[self.i_markers[ii]].translation)

        J = np.sum((Goal - Mposition_markers) ** 2)

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


def init_human_model(model, data, DMarkers, i_markers, q0):

    lb = model.lowerPositionLimit  # lower joint limits
    ub = model.upperPositionLimit  # upper joint limits
    cl = cu = [1]

    nlp = cyipopt.Problem(
        n=len(q0),
        m=len(cl),
        problem_obj=initialisation(model, data, DMarkers, i_markers),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    nlp.add_option("tol", 1e-3)
    nlp.add_option("print_level", 0)
    q_opt, info = nlp.solve(q0)

    return q_opt
