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


from os.path import abspath
from scipy.optimize import least_squares
import numpy as np
import yaml
from yaml.loader import SafeLoader

from figaroh.calibration.calibration_tools import (
    get_param_from_yaml,
    add_pee_name,
    load_data,
    calculate_base_kinematics_regressor,
    update_forward_kinematics,
    get_LMvariables,
)
from figaroh.tools.robot import Robot
import matplotlib.pyplot as plt


class TiagoCalibration:
    def __init__(self, robot, config_file):
        self._robot = robot
        self.model = self._robot.model
        self.data = self._robot.data

        self.load_param(config_file)
        self.nvars = len(self.param["param_name"])

        self._data_path = abspath(self.param["data_file"])
        self.STATUS = "NOT CALIBRATED"

    def initialize(self):
        self.load_data_set()
        self.create_param_list()

    def solve(self):
        self.solve_optimisation()
        self.calc_stddev()
        if self.param['PLOT']:
            self.plot()

    def plot(self):
        self.plot_errors_distribution()
        self.plot_3d_poses()
        self.plot_joint_configurations()
        plt.show()

    def load_param(self, config_file, setting_type="calibration"):
        """
        Load the calibration parameters from the yaml file.
        """
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)
        calib_data = config[setting_type]
        self.param = get_param_from_yaml(self._robot, calib_data)

    def create_param_list(self, q=None):
        """
        Create the list of parameters to be calibrated.
        """
        if q is None:
            q_ = []
        else:
            q_ = q
        (
            Rrand_b,
            R_b,
            R_e,
            paramsrand_base,
            paramsrand_e,
        ) = calculate_base_kinematics_regressor(
            q_, self.model, self.data, self.param
        )

        # Add markers name to param['param_name']
        add_pee_name(self.param)

    def load_data_set(self):
        """
        Load the data set from the data file.
        """
        self.PEE_measured, self.q_measured = load_data(
            self._data_path, self.model, self.param
        )

    def cost_function(self, var):
        """
        Cost function for the optimization problem.
        """
        coeff_ = self.param["coeff_regularize"]
        PEEe = update_forward_kinematics(
            self.model, self.data, var, self.q_measured, self.param
        )
        res_vect = np.append(
            (self.PEE_measured - PEEe),
            np.sqrt(coeff_)
            * var[
                6 : -self.param["NbMarkers"] * self.param["calibration_index"]
            ],
        )
        return res_vect

    def solve_optimisation(self):
        """
        Solve the optimization problem.
        """

        # set initial guess
        _var_0, _ = get_LMvariables(self.param, mode=0)
        _var_0[0:6] = np.array(self.param["camera_pose"])
        _var_0[-self.param["calibration_index"] :] = np.array(
            self.param["tip_pose"]
        )[: self.param["calibration_index"]]
        self._var_0 = _var_0

        # define solver parameters
        iterate = True
        iter_max = 10
        count = 0
        del_list = []
        res = _var_0
        outlier_eps = 0.05

        while count < iter_max and iterate:
            print("*" * 50)
            print(
                "{} iter guess".format(count),
                dict(zip(self.param["param_name"], list(_var_0))),
            )

            # define solver
            LM_solve = least_squares(
                self.cost_function,
                _var_0,
                method="lm",
                verbose=1,
                args=(),
            )

            # solution
            res = LM_solve.x
            _PEEe_sol = update_forward_kinematics(
                self.model, self.data, res, self.q_measured, self.param
            )
            rmse = np.sqrt(np.mean((_PEEe_sol - self.PEE_measured) ** 2))
            mae = np.mean(np.abs(_PEEe_sol - self.PEE_measured))

            print("solution of calibrated parameters: ")
            for x_i, xname in enumerate(self.param["param_name"]):
                print(x_i + 1, xname, list(res)[x_i])
            print("position root-mean-squared error of end-effector: ", rmse)
            print("position mean absolute error of end-effector: ", mae)
            print("optimality: ", LM_solve.optimality)

            # check for unrealistic values
            delta_PEE = _PEEe_sol - self.PEE_measured
            PEE_xyz = delta_PEE.reshape(
                (
                    self.param["NbMarkers"] * self.param["calibration_index"],
                    self.param["NbSample"],
                )
            )
            PEE_dist = np.zeros(
                (self.param["NbMarkers"], self.param["NbSample"])
            )
            for i in range(self.param["NbMarkers"]):
                for j in range(self.param["NbSample"]):
                    PEE_dist[i, j] = np.sqrt(
                        PEE_xyz[i * 3, j] ** 2
                        + PEE_xyz[i * 3 + 1, j] ** 2
                        + PEE_xyz[i * 3 + 2, j] ** 2
                    )
            for i in range(self.param["NbMarkers"]):
                for k in range(self.param["NbSample"]):
                    if PEE_dist[i, k] > outlier_eps:
                        del_list.append((i, k))
            print(
                "indices of samples with >{} m deviation: ".format(
                    outlier_eps
                ),
                del_list,
            )

            # reset iteration with outliers removal
            if len(del_list) > 0 and count < iter_max:
                self.PEEm_LM, q_LM = load_data(
                    self._data_path, self.model, self.param, del_list
                )
                self.param["NbSample"] = q_LM.shape[0]
                count += 1
                _var_0 = res + np.random.normal(0, 0.01, size=res.shape)
            else:
                iterate = False
        self._PEE_dist = PEE_dist
        self.calibrated_param = dict(zip(self.param["param_name"], list(res)))
        self.LM_result = LM_solve
        self.rmse = rmse
        self.mae = mae
        if self.LM_result.success:
            self.STATUS = "CALIBRATED"

    def get_pose_from_measure(self, _res):
        """
        Get the pose of the robot given a set of parameters.
        """
        return update_forward_kinematics(
            self.model, self.data, _res, self.q_measured, self.param
        )

    def calc_stddev(self):
        """
        Calculate the standard deviation of the calibrated parameters.
        """
        assert self.STATUS == "CALIBRATED", "Calibration not performed yet"
        sigma_ro_sq = (self.LM_result.cost**2) / (
            self.param["NbSample"] * self.param["calibration_index"]
            - self.nvars
        )
        J = self.LM_result.jac
        C_param = sigma_ro_sq * np.linalg.pinv(np.dot(J.T, J))
        std_dev = []
        std_pctg = []
        for i_ in range(self.nvars):
            std_dev.append(np.sqrt(C_param[i_, i_]))
            std_pctg.append(
                abs(np.sqrt(C_param[i_, i_]) / self.LM_result.x[i_])
            )
        self.std_dev = std_dev
        self.std_pctg = std_pctg

    def plot_errors_distribution(self):
        """
        Plot the distribution of the errors.
        """
        assert self.STATUS == "CALIBRATED", "Calibration not performed yet"

        fig1, ax1 = plt.subplots(self.param["NbMarkers"], 1)
        colors = ["blue", "red", "yellow", "purple"]

        if self.param["NbMarkers"] == 1:
            ax1.bar(np.arange(self.param["NbSample"]), self._PEE_dist[0, :])
            ax1.set_xlabel("Sample", fontsize=25)
            ax1.set_ylabel("Error (meter)", fontsize=30)
            ax1.tick_params(axis="both", labelsize=30)
            ax1.grid()
        else:
            for i in range(self.param["NbMarkers"]):
                ax1[i].bar(
                    np.arange(self.param["NbSample"]),
                    self._PEE_dist[i, :],
                    color=colors[i],
                )
                ax1[i].set_xlabel("Sample", fontsize=25)
                ax1[i].set_ylabel(
                    "Error of marker %s (meter)" % (i + 1), fontsize=25
                )
                ax1[i].tick_params(axis="both", labelsize=30)
                ax1[i].grid()

    def plot_3d_poses(self, INCLUDE_UNCALIB=True):
        """
        Plot the 3D poses of the robot.
        """
        assert self.STATUS == "CALIBRATED", "Calibration not performed yet"

        fig2 = plt.figure(2)
        fig2.suptitle(
            "Visualization of estimated poses and measured pose in Cartesian"
        )
        ax2 = fig2.add_subplot(111, projection="3d")
        PEEm_LM2d = self.PEE_measured.reshape(
            (
                self.param["NbMarkers"] * self.param["calibration_index"],
                self.param["NbSample"],
            )
        )
        PEEe_sol = self.get_pose_from_measure(self.LM_result.x)
        PEEe_sol2d = PEEe_sol.reshape(
            (
                self.param["NbMarkers"] * self.param["calibration_index"],
                self.param["NbSample"],
            )
        )
        PEEe_uncalib = self.get_pose_from_measure(self._var_0)
        PEEe_uncalib2d = PEEe_uncalib.reshape(
            (
                self.param["NbMarkers"] * self.param["calibration_index"],
                self.param["NbSample"],
            )
        )
        for i in range(self.param["NbMarkers"]):
            ax2.scatter3D(
                PEEm_LM2d[i * 3, :],
                PEEm_LM2d[i * 3 + 1, :],
                PEEm_LM2d[i * 3 + 2, :],
                marker="^",
                color="blue",
                label="Measured",
            )
            ax2.scatter3D(
                PEEe_sol2d[i * 3, :],
                PEEe_sol2d[i * 3 + 1, :],
                PEEe_sol2d[i * 3 + 2, :],
                marker="o",
                color="red",
                label="Estimated",
            )
            if INCLUDE_UNCALIB:
                ax2.scatter3D(
                    PEEe_uncalib2d[i * 3, :],
                    PEEe_uncalib2d[i * 3 + 1, :],
                    PEEe_uncalib2d[i * 3 + 2, :],
                    marker="x",
                    color="green",
                    label="Uncalibrated",
                )
            for j in range(self.param["NbSample"]):
                ax2.plot3D(
                    [PEEm_LM2d[i * 3, j], PEEe_sol2d[i * 3, j]],
                    [PEEm_LM2d[i * 3 + 1, j], PEEe_sol2d[i * 3 + 1, j]],
                    [PEEm_LM2d[i * 3 + 2, j], PEEe_sol2d[i * 3 + 2, j]],
                    color="red",
                )
                if INCLUDE_UNCALIB:
                    ax2.plot3D(
                        [PEEm_LM2d[i * 3, j], PEEe_uncalib2d[i * 3, j]],
                        [
                            PEEm_LM2d[i * 3 + 1, j],
                            PEEe_uncalib2d[i * 3 + 1, j],
                        ],
                        [
                            PEEm_LM2d[i * 3 + 2, j],
                            PEEe_uncalib2d[i * 3 + 2, j],
                        ],
                        color="green",
                    )
        ax2.set_xlabel("X - front (meter)")
        ax2.set_ylabel("Y - side (meter)")
        ax2.set_zlabel("Z - height (meter)")
        ax2.grid()
        ax2.legend()

    def plot_joint_configurations(self):
        """
        Joint configurations within range bound.
        """
        fig4 = plt.figure()
        fig4.suptitle("Joint configurations with joint bounds")
        ax4 = fig4.add_subplot(111, projection="3d")
        lb = ub = []
        for j in self.param["config_idx"]:
            lb = np.append(lb, self.model.lowerPositionLimit[j])
            ub = np.append(ub, self.model.upperPositionLimit[j])
        q_actJoint = self.q_measured[:, self.param["config_idx"]]
        sample_range = np.arange(self.param["NbSample"])
        for i in range(len(self.param["actJoint_idx"])):
            ax4.scatter3D(q_actJoint[:, i], sample_range, i)
        for i in range(len(self.param["actJoint_idx"])):
            ax4.plot(
                [lb[i], ub[i]], [sample_range[0], sample_range[0]], [i, i]
            )
            ax4.set_xlabel("Angle (rad)")
            ax4.set_ylabel("Sample")
            ax4.set_zlabel("Joint")
            ax4.grid()


def load_robot(robot_urdf, package_dirs=None, isFext=False, load_by_urdf=True):
    """
    Load the robot model from the URDF file.
    """
    import rospy
    import pinocchio
    import rospkg

    if load_by_urdf:
        package_dirs = rospkg.RosPack().get_path("tiago_description")
        # robot_urdf = "data/tiago_hey5.urdf"
        robot = Robot(
            robot_urdf,
            package_dirs=package_dirs,
            isFext=isFext,
        )
    else:
        robot_xml = rospy.get_param("robot_description")
        if isFext:
            robot = pinocchio.buildModelFromXML(
                robot_xml, root_joint=pinocchio.JointModelFreeFlyer()
            )
        else:
            robot = pinocchio.buildModelFromXML(robot_xml, root_joint=None)
    return robot


def write_to_xacro(tiago_calib, file_type="yaml"):
    """
    Write calibration result to xacro file.
    """
    assert tiago_calib.STATUS == "CALIBRATED", "Calibration not performed yet"
    model = tiago_calib.model
    calib_result = tiago_calib.calibrated_param
    param = tiago_calib.param

    calibration_parameters = {}
    calibration_parameters["camera_position_x"] = calib_result["base_px"]
    calibration_parameters["camera_position_y"] = calib_result["base_py"]
    calibration_parameters["camera_position_z"] = calib_result["base_pz"]
    calibration_parameters["camera_orientation_r"] = calib_result["base_phix"]
    calibration_parameters["camera_orientation_p"] = calib_result["base_phiy"]
    calibration_parameters["camera_orientation_y"] = calib_result["base_phiz"]

    for idx in param["actJoint_idx"]:
        joint = model.names[idx]
        for key in calib_result.keys():
            if joint in key:
                calibration_parameters[joint + "_offset"] = calib_result[key]

    if file_type == "xacro":
        path_save_xacro = abspath(
            "data/tiago_master_calibration_{}.xacro".format(param["NbSample"])
        )

        with open(path_save_xacro, "w") as output_file:
            for parameter in calibration_parameters.keys():
                update_name = parameter
                update_value = calibration_parameters[parameter]
                update_line = (
                    '<xacro:property name="{}" value="{}" / >'.format(
                        update_name, update_value
                    )
                )
                output_file.write(update_line)
                output_file.write("\n")

    elif file_type == "yaml":
        path_save_yaml = abspath(
            "data/tiago_master_calibration_{}.yaml".format(param["NbSample"])
        )
        with open(path_save_yaml, "w") as output_file:
            for parameter in calibration_parameters.keys():
                update_name = parameter
                update_value = calibration_parameters[parameter]
                update_line = "{}:{}".format(update_name, update_value)
                output_file.write(update_line)
                output_file.write("\n")


def main():
    tiago = load_robot("data/tiago_hey5.urdf")
    tiago_calib = TiagoCalibration(tiago, "config/tiago_config.yaml")
    tiago_calib.initialize()
    tiago_calib.solve()
    tiago_calib.plot()
    write_to_xacro(tiago_calib, file_type="yaml")
    return 0


if __name__ == "__main__":
    main()