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


def place(viz, name: str, M: pin.SE3) -> None:
    """Place coordinate system in GUI visualization.
    
    Args:
        viz: Robot visualizer (e.g. gepetto-viewer)
        name: Name of coordinate system object
        M: Homogeneous transformation matrix
    """
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(M))


def display_COM(model: pin.Model, data: pin.Data, viz, q: np.ndarray, IDX: list) -> None:
    """Display center of mass positions for each link.
    
    Args:
        model: Pinocchio robot model
        data: Pinocchio robot data
        viz: Robot visualizer
        q: Joint configuration vector
        IDX: List of frame indices in kinematic tree order
    """
    pin.forwardKinematics(model, data, q)
    pin.computeSubtreeMasses(model, data)
    pin.centerOfMass(model, data, q, True)
    rgbt = [1.0, 0.0, 0.0, 1.0]  # red, green, blue, transparency
    ball_ids = []
    for i in range(len(IDX)):
        link_length = np.linalg.norm(
            data.oMf[IDX[i + 1]].translation - data.oMf[IDX[i]].translation
        )
        placement = data.oMf[IDX[i]]
        ball_ids.append("world/ball_" + str(i))
        radius = link_length * data.mass[i] / data.mass[0]  # mass ratio
        placement.translation = data.com[model.frames[IDX[i]].parent - 1]
        viz.viewer.gui.addSphere(ball_ids[i], radius, rgbt)
        place(viz, ball_ids[i], placement)


def display_axes(model: pin.Model, data: pin.Data, viz, q: np.ndarray) -> None:
    """Display coordinate axes for each joint.
    
    Args:
        model: Pinocchio robot model
        data: Pinocchio robot data 
        viz: Robot visualizer
        q: Joint configuration vector
    """
    ids = []
    matrices = []
    names = model.names
    axes = []

    # Get the joints id and create the axes
    for i in range(len(names)):
        ids.append(model.getJointId(names[i]))
        axes.append("world/link_" + str(i))
        viz.viewer.gui.addXYZaxis(axes[i], [1.0, 0.0, 0.0, 1.0], 0.01, 0.15)

    # Compute the forward kinematics w.r.t the q
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Get the homogeneous matrices for each frame and place the axes
    for i in range(len(ids)):
        matrices.append(data.oMi[ids[i]])
        place(viz, axes[i], matrices[i])


def rotation_matrix_from_vectors(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> np.ndarray:
    """Find rotation matrix aligning vec1 to vec2.
    
    Args:
        vec1: Source 3D vector
        vec2: Destination 3D vector
        
    Returns:
        ndarray: 3x3 rotation matrix that aligns vec1 with vec2
    """
    # Normalize vectors
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]
    ])

    rotation_matrix = (np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2)))
    return rotation_matrix


def display_force(viz, phi: pin.Force, M_se3: pin.SE3) -> None:
    """Display force vector in visualization.
    
    Args:
        viz: Robot visualizer
        phi: 6D force vector in M_se3 frame
        M_se3: SE3 transformation for force display
    """
    M_se3_temp = M_se3
    color = [1, 1, 0, 1]
    radius = 0.01
    
    phi = phi.se3Action(M_se3)
    force = [phi.linear[0], phi.linear[1], phi.linear[2]]
    length = np.linalg.norm(force) * 1e-3
    
    # Project x-axis onto force direction for display
    Rot = rotation_matrix_from_vectors([1, 0, 0], phi.linear)
    M_se3_temp.rotation = Rot
    
    viz.viewer.gui.addArrow("world/arrow", radius, length, color)
    place(viz, "world/arrow", M_se3_temp)


def display_bounding_boxes(
    viz, 
    model: pin.Model, 
    data: pin.Data, 
    q: np.ndarray,
    COM_min: np.ndarray, 
    COM_max: np.ndarray, 
    IDX: list
) -> None:
    """Display COM bounding boxes for optimization.
    
    Args:
        viz: Robot visualizer
        model: Pinocchio robot model
        data: Pinocchio robot data 
        q: Joint configuration vector
        COM_min: Min COM boundaries per segment (3*num_segments)
        COM_max: Max COM boundaries per segment (3*num_segments) 
        IDX: List of frame indices
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.centerOfMass(model, data, q, True)
    
    for i in range(len(IDX)):
        M = data.oMf[IDX[i]]
        M.translation += data.com[model.frames[IDX[i]].parent - 1]
        
        size_x = COM_max[3 * i] - COM_min[3 * i]
        size_y = COM_max[3 * i + 1] - COM_min[3 * i + 1]
        size_z = COM_max[3 * i + 2] - COM_min[3 * i + 2]
        
        box_name = f"world/box{i}"
        viz.viewer.gui.addBox(
            box_name,
            size_x,
            size_y, 
            size_z,
            [0.5, 0, 0.5, 0.5]
        )
        place(viz, box_name, M)


def display_joints(viz, model: pin.Model, data: pin.Data, q: np.ndarray) -> None:
    """Display joint frames in visualization.
    
    Args:
        viz: Robot visualizer
        model: Pinocchio robot model
        data: Pinocchio robot data
        q: Joint configuration vector
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    for i in range(model.nv):
        joint_pos = data.oMi[i].translation
        joint_ori = pin.Quaternion(data.oMi[i].rotation)
        joint_name = f"world/joint{i}"
        
        viz.viewer.gui.addXYZaxis(
            joint_name, 
            [1.0, 0.0, 0.0, 1.0],
            0.01,
            0.15
        )
        
        config = [
            joint_pos[0], joint_pos[1], joint_pos[2],
            joint_ori[0], joint_ori[1], joint_ori[2], joint_ori[3]
        ]
        viz.viewer.gui.applyConfiguration(joint_name, config)
        viz.viewer.gui.refresh()
