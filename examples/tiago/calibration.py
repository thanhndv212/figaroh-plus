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

# Import necessary modules
import argparse
from tiago_utils.robot_tools import (
    RobotCalibration,
    load_robot,
    write_to_xacro
)
from os.path import abspath


def run_calibration(
    urdf_path,
    config_path,
    remove_param=None,
    plot_level=1,
    save_calibration=False,
    end_effector="schunk",
):
    """
    Perform the calibration process for the TIAGo robot.

    Args:
    urdf_path (str): Path to the URDF file of the robot.
    config_path (str): Path to the configuration file for calibration.
    remove_param (str, optional): Parameter to remove from calibration. 
        Defaults to None.
    plot_level (int, optional): Level of detail for the calibration plot. 
        Defaults to 1.
    save_calibration (bool, optional): Whether to save the calibration results. 
        Defaults to False.
    end_effector (str, optional): Type of end effector. Defaults to "schunk".

    Returns:
    None
    """
    # Load the TIAGo robot model
    tiago = load_robot(
        abspath(urdf_path),
        isFext=True,
        load_by_urdf=True,
    )

    # Initialize the calibration object with the robot model and configuration
    tiago_calib = RobotCalibration(
        tiago, abspath(config_path)
    )
    tiago_calib.initialize()

    # Remove a specific parameter from calibration if specified
    if remove_param:
        tiago_calib.param["param_name"].remove(remove_param)

    # Perform the calibration
    tiago_calib.solve()

    # Plot the calibration results
    tiago_calib.plot(lvl=plot_level)

    # Save the calibration results if requested
    if save_calibration:
        write_to_xacro(
            tiago_calib,
            file_name=f"tiago_master_calibration_{end_effector}.yaml",
            file_type="yaml",
        )


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="TIAGo robot calibration script"
    )
    parser.add_argument(
        "--urdf_path",
        default="urdf/tiago_48_schunk.urdf",
        help="Path to URDF file",
    )
    parser.add_argument(
        "--config_path",
        default="config/tiago_config_mocap_vicon.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--remove_param",
        help="Parameter to remove from calibration",
    )
    parser.add_argument(
        "--plot_level",
        type=int,
        default=1,
        help="Plot level for calibration results",
    )
    parser.add_argument(
        "--save_calibration",
        action="store_true",
        help="Save calibration results",
    )
    parser.add_argument(
        "--end_effector",
        default="schunk",
        help="End effector type",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the calibration with the provided arguments
    run_calibration(
        args.urdf_path,
        args.config_path,
        args.remove_param,
        args.plot_level,
        args.save_calibration,
        args.end_effector,
    )

# The following code is commented out as in the original file
# This section appears to be for loading absolute encoder values and performing
# additional calibration. It's kept here for reference or potential future use

# load absolute encoder values
# torso_idxq = tiago_calib.param["config_idx"][0]
# tiago_abs = RobotCalibration(
#     tiago, abspath("config/tiago_config_mocap_vicon_abs.yaml"), del_list=[]
# )
# tiago_abs.initialize()
# replace relative encoder values of torso to absolute one
# tiago_abs.q_measured[:, torso_idxq] = tiago_calib.q_measured[:, torso_idxq]
# tiago_abs.solve()
# tiago_abs.plot(lvl=1)
