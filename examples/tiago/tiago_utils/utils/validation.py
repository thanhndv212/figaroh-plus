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

from tiago_tools import (
    load_robot,
    model_todict,
    compare_model,
)
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="parse calibration setups", add_help=False
    )
    parser.add_argument(
        "-e", "--end_effector", default="hey5", dest="end_effector"
    )
    parser.add_argument(
        "-u", "--load_by_urdf", default=True, dest="load_by_urdf"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    tiago_target = load_robot(
        "data/urdf/tiago_48_{}.urdf".format(args.end_effector),
        load_by_urdf=args.load_by_urdf,
    )

    tiago_urdf = load_robot(
        "data/urdf/tiago_48_{}.urdf".format(args.end_effector),
        load_by_urdf=True,
    )
    compare_model(
        model_todict(tiago_target.model), model_todict(tiago_urdf.model)
    )


def joint_inspect(robot_model, joint_name):
    assert joint_name in robot_model.names, "{} does not exists in {}.".format(
        joint_name, robot_model.name
    )
    joint = {}
    joint_id = robot_model.getJointId(joint_name)
    joint["name"] = joint_name
    joint["detail"] = robot_model.joints[joint_id]
    joint["lower_limit"] = robot_model.lowerPositionLimit[
        joint["detail"].idx_q
    ]
    joint["upper_limit"] = robot_model.upperPositionLimit[
        joint["detail"].idx_q
    ]
    joint["parent_id"] = robot_model.parents[joint_id]
    joint["parent"] = robot_model.names[robot_model.parents[joint_id]]
    joint["relative_placement"] = robot_model.jointPlacements[joint_id]
    joint["inertia"] = robot_model.inertias[joint_id]
    joint["subtree"] = [
        robot_model.names[x] for x in robot_model.subtrees[joint_id]
    ]
    joint["support"] = [
        robot_model.names[y] for y in robot_model.supports[joint_id]
    ]
    return joint


if __name__ == "__main__":
    import numpy as np
    import yaml

    tiago = load_robot(
        "data/urdf/tiago_48_hey5.urdf",
        load_by_urdf=True,
    )
    nsample = 20
    configs = {}
    joint_names = [
        "torso_lift_joint",
        "arm_1_joint",
        "arm_2_joint",
        "arm_3_joint",
        "arm_4_joint",
        "arm_5_joint",
        "arm_6_joint",
        "arm_7_joint",
    ]
    configs["calibration_joint_names"] = joint_names
    torso_ref = 0.30
    arm1_ref = 1.61
    arm2_ref = 0.04
    arm3_ref = -3.14
    arm4_ref = 0.12
    arm5_ref = 0.04
    arm7_ref = -1.58
    arm6 = joint_inspect(tiago.model, "arm_6_joint")
    arm6_configs = []
    soft_lm = arm6["lower_limit"] + 0.05
    soft_um = arm6["upper_limit"] - 0.05
    configs_values = []
    for ii in range(nsample):
        arm6_ref = float(soft_lm + ii * (soft_um - soft_lm) / (nsample - 1))
        configs_values.append(
            [
                torso_ref,
                arm1_ref,
                arm2_ref,
                arm3_ref,
                arm4_ref,
                arm5_ref,
                arm6_ref,
                arm7_ref,
            ]
        )
    for jj in range(nsample):
        arm6_ref = float(soft_um - jj * (soft_um - soft_lm) / (nsample - 1))
        configs_values.append(
            [
                torso_ref,
                arm1_ref,
                arm2_ref,
                arm3_ref,
                arm4_ref,
                arm5_ref,
                arm6_ref,
                arm7_ref,
            ]
        )
    configs["calibration_joint_configurations"] = configs_values
    path_save = "data/optimal_configs/arm6_inspect.yaml"
    with open(path_save, "w") as file_output:
        try:
            yaml.dump(
                configs,
                file_output,
                sort_keys=False,
                default_flow_style=True,
            )
        except yaml.YAMLError as exc:
            print(exc)
