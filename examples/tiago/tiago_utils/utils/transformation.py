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

from figaroh.tools.robot import Robot
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
import pinocchio as pin

tm = UrdfTransformManager()
filename = "/home/thanhndv212/Cooking/figaroh/scripts/models/others/robots/\
    talos_data/robots/talos_reduced.urdf"
data_dir = "/home/thanhndv212/Cooking/figaroh/scripts/models/others/robots/\
    talos_data/meshes"
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)

robot = Robot(
    "talos_data/robots",
    "talos_reduced.urdf",
    # "tiago_description/robots",
    # "tiago_no_hand_mod.urdf",
    # "canopies_description/robots",
    # "canopies_arm.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

joint_names = model.names.tolist()[1:]
joint_angles = pin.randomConfiguration(model)

frame_names = []
for i in range(1, int(model.nframes)):
    if "joint" not in model.frames[i].name:
        if "base_link" in model.frames[i].name:
            frame_names.append(model.frames[i].name)
        if "gripper_right_base_link" in model.frames[i].name:
            frame_names.append(model.frames[i].name)
        if "arm_right" in model.frames[i].name:
            frame_names.append(model.frames[i].name)
        if "torso" in model.frames[i].name:
            frame_names.append(model.frames[i].name)

for name, angle in zip(joint_names, joint_angles):
    tm.set_joint(name, angle)
ax = tm.plot_frames_in("talos", whitelist=frame_names, s=0.15, show_name=False)
ax = tm.plot_frames_in(
    "talos",
    ax=ax,
    whitelist=["base_link", "gripper_right_base_link"],
    s=0.15,
    show_name=True,
)

ax = tm.plot_connections_in(
    "talos",
    ax=ax,
    whitelist=frame_names,
)
ax.set_xlim((-0.2, 0.8))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((-0.2, 0.8))
plt.show()
