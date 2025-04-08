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

import numpy as np
import pinocchio as pin


def place(viz, name, M):
    """This function places in the gui a coordinate system at the location provided in M.
    Input: viz (viz) a robot visualiser (such as gepetto-viewer)
           name (str) the name of the object coordinate system
           M (se3) homogenous transformation matrix
    Output: (void) places the object at the desired location
    """
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(M))


def display_COM(model, data, viz, q, IDX):
    """This function displays the positions of the center of mass of each link in the
    gui (the radius is porportionnal to the ratio mass_link/global mass)
    Input : model: (model) a pinocchio robot model
            data: (data) a pinocchio robot data
            viz: (viz) a pinocchio visualizer (such as gepetto_viewer, etc ...), could
            be configured by a command line
            q: (ndarray) an angular position vector configuration
            IDX: (list) list of the frames of interest indices (should be given in the
            order of the kinematic tree)
    Output : Void : display in the gui the centers of mass of the desired frames
    """
    pin.forwardKinematics(model, data, q)
    pin.computeSubtreeMasses(model, data)
    pin.centerOfMass(model, data, q, True)
    rgbt = [1.0, 0.0, 0.0, 1.0]  # red, green, blue, transparency
    ballID = []
    for ii in range(len(IDX)):
        lii = np.linalg.norm(
            data.oMf[IDX[ii + 1]].translation - data.oMf[IDX[ii]].translation
        )
        Placement = data.oMf[IDX[ii]]
        ballID.append("world/ball_" + str(ii))
        radius = lii * data.mass[ii] / data.mass[0]  # mass ratio
        Placement.translation = +data.com[model.frames[IDX[ii]].parent - 1]
        viz.viewer.gui.addSphere(ballID[ii], radius, rgbt)
        place(viz, ballID[ii], Placement)


def display_axes(model, data, viz, q):
    """This function displays the axes for each body joints.
    Input: model (model) a pinocchio robot model
           data (data) a pinocchio robot data
           viz (viz) a robot visualiser (such as gepetto-viewer)
           q (ndarray) an articular robot position
    Output: (void) returns in the gui the different axes
    """
    ids = []
    M = []
    names = model.names
    axis = []

    # Get the joints id and creates the axis
    for ii in range(len(names)):
        # if model.getJointId(names[ii])==0 or model.getJointId(names[ii])==1 or model.getJointId(names[ii])==11 or model.getJointId(names[ii])==12 or model.getJointId(names[ii])==8 or model.getJointId(names[ii])==9 or model.getJointId(names[ii])==10:
        ids.append(model.getJointId(names[ii]))
        axis.append("world/link_" + str(ii))
        viz.viewer.gui.addXYZaxis(axis[ii], [1.0, 0.0, 0.0, 1.0], 0.01, 0.15)

    # compute the forward kinematics w.r.t the q
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Get the homogenous matrices for each frame and place the axis
    for ii in range(len(ids)):
        M.append(data.oMi[ids[ii]])
        place(viz, axis[ii], M[ii])


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with
    vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def display_force(viz, phi, M_se3):
    """Displays the force phi (expressed in the M_se3 frame) in the M_se3 frame and the
    visualiser viz given
    Input: viz (viz) a pinocchio visualiser
           phi (Force Tpl) a pinocchio force 6D vector
           M_se3 (SE3) the se3 object relating the transformation matrix in which we
           want the force to be displayed
    Output: (void) Displays the force in the gui
    """
    M_se3_temp = M_se3
    color = [1, 1, 0, 1]
    radius = 0.01
    phi = phi.se3Action(M_se3)
    force = [phi.linear[0], phi.linear[1], phi.linear[2]]
    lenght = np.linalg.norm(force)*1e-3
    Rot = rotation_matrix_from_vectors([1, 0, 0], phi.linear)   # addArrow in pinocchio is always along the x_axis so we have to project the x_axis on the direction of the force vector for display purposes # noqa
    M_se3_temp.rotation = Rot
    viz.viewer.gui.addArrow("world/arrow", radius, lenght, color)
    place(viz, "world/arrow", M_se3_temp)


def display_bounding_boxes(viz, model, data, q, COM_min, COM_max, IDX):
    """Displays the bounding boxes for the COM as given as inputs for the optimisation
    problem for the estimation of the SIP
    Inputs: viz (viz) a pinocchio visualiser
            model (model) a pinocchio robot model
            data (data) a pinocchio data model
            q (ndarray) a articular configuration
            COM_min (ndarray) shape : (3*number of segments,1) regroups all the lower
            boundaries (in x, in y, in z) for the COM estimation
            COM_max (ndarray) shape : (3*number of segments,1) regroups all the upper
            boundaries (in x, in y, in z) for the COM estimation
            IDX (list) : a list of the frames of interest
    Output: (void) displays the bounding boxes in the gui
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q, True)
    for ii in range(len(IDX)):
        M = data.oMf[IDX[ii]]
        M.translation += data.com[model.frames[IDX[ii]].parent - 1]
        size_x = COM_max[3 * ii] - COM_min[3 * ii]
        size_y = COM_max[3 * ii + 1] - COM_min[3 * ii + 1]
        size_z = COM_max[3 * ii + 2] - COM_min[3 * ii + 2]
        viz.viewer.gui.addBox(
            "world/box" + str(ii), size_x, size_y, size_z, [0.5, 0, 0.5, 0.5]
        )
        place(viz, "world/box" + str(ii), M)


def display_joints(viz,model,data,q_ii):
    pin.forwardKinematics(model,data,q_ii)
    pin.updateFramePlacements(model,data)
    for ii in range(model.nv):
        joint_pos = data.oMi[ii].translation
        joint_ori = pin.Quaternion(data.oMi[ii].rotation)
        viz.viewer.gui.addXYZaxis('world/joint'+str(ii), [1.0, 0.0, 0.0, 1.0], 0.01, 0.15)
        viz.viewer.gui.applyConfiguration('world/joint'+str(ii),[joint_pos[0],joint_pos[1],joint_pos[2],joint_ori[0],joint_ori[1],joint_ori[2],joint_ori[3]])
        viz.viewer.gui.refresh()
