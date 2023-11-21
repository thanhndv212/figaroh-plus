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
import pinocchio as pin


class TiagoCalibration:
    def __init__(self, robot, config_file, del_list=[]):
        self._robot = robot
        self.model = self._robot.model
        self.data = self._robot.data
        self.del_list_ = del_list
        self.param = None
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
        if self.param["PLOT"]:
            self.plot()

    def plot(self, lvl=1):
        if lvl == 1:
            self.plot_errors_distribution()
        if lvl > 1:
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
        return True

    def load_calibration_param(self, param_file):
        with open(param_file, "r") as param_file:
            param_dict_ = yaml.load(param_file, Loader=SafeLoader)
        assert len(self.param["param_name"]) == len(
            param_dict_
        ), "The loaded param list does not match calibration config."
        self.var_ = np.zeros(len(self.param["param_name"]))
        updated_var_ = []
        for i_, name_ in enumerate(self.param["param_name"]):
            assert name_ == list(param_dict_.keys())[i_]
            self.var_[i_] = list(param_dict_.values())[i_]
            updated_var_.append(name_)
        assert len(updated_var_) == len(self.var_), "Not all param imported."
        self.LOAD_NEWMODEL = True

    def validate_model(self):
        assert (
            self.LOAD_NEWMODEL
        ), "Call load_calibration_param() to load model parameters first."
        pee_valid_ = self.get_pose_from_measure(self.var_)
        self.calc_errors(pee_valid_)

    def calc_errors(self, pee_):
        PEE_errors = pee_ - self.PEE_measured
        if self.param["measurability"][0:3] == [True, True, True]:
            rmse_pos = np.sqrt(
                np.mean((PEE_errors[0 : self.param["NbSample"] * 3]) ** 2)
            )
            mae_pos = np.mean(
                np.abs(PEE_errors[0 : self.param["NbSample"] * 3])
            )
            print(
                "position root-mean-squared error of end-effector: ", rmse_pos
            )
            print("position mean absolute error of end-effector: ", mae_pos)
            if self.param["measurability"][3:6] == [True, True, True]:
                rmse_rot = np.sqrt(
                    np.mean((PEE_errors[self.param["NbSample"] * 3 :]) ** 2)
                )
                mae_rot = np.mean(
                    np.abs(PEE_errors[self.param["NbSample"] * 3 :])
                )
                print(
                    "rotation root-mean-squared error of end-effector: ",
                    rmse_rot,
                )
                print(
                    "rotation mean absolute error of end-effector: ", mae_rot
                )
                return [rmse_pos, mae_pos, rmse_rot, mae_rot]
            else:
                return [rmse_pos, mae_pos]

    def load_data_set(self):
        """
        Load the data set from the data file.
        """
        self.PEE_measured, self.q_measured = load_data(
            self._data_path, self.model, self.param, self.del_list_
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
        del_list_ = []
        res = _var_0
        outlier_eps = self.param["outlier_eps"]

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

            print("solution of calibrated parameters: ")
            for x_i, xname in enumerate(self.param["param_name"]):
                print(x_i + 1, xname, list(res)[x_i])

            self.calc_errors(_PEEe_sol)
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
                        del_list_.append((i, k))
            print(
                "indices of samples with >{} m deviation: ".format(
                    outlier_eps
                ),
                del_list_,
            )

            # reset iteration with outliers removal
            if len(del_list_) > 0 and count < iter_max:
                self.PEE_measured, self.q_measured = load_data(
                    self._data_path,
                    self.model,
                    self.param,
                    self.del_list_ + del_list_,
                )
                self.param["NbSample"] = self.q_measured.shape[0]
                count += 1
                _var_0 = res + np.random.normal(0, 0.01, size=res.shape)
            else:
                iterate = False
        self._PEE_dist = PEE_dist
        param_values_ = [float(res_i_) for res_i_ in res]
        self.calibrated_param = dict(
            zip(self.param["param_name"], param_values_)
        )
        self.LM_result = LM_solve
        if self.LM_result.success:
            self.STATUS = "CALIBRATED"

    def get_pose_from_measure(self, res_):
        """
        Get the pose of the robot given a set of parameters.
        """
        return update_forward_kinematics(
            self.model, self.data, res_, self.q_measured, self.param
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
        fig1.suptitle("Distribution of the distance errors")
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

        fig2 = plt.figure()
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
    import pinocchio
    import rospkg

    if load_by_urdf:
        # import os
        # ros_package_path = os.getenv('ROS_PACKAGE_PATH')
        # package_dirs = ros_package_path.split(':')
        package_dirs = rospkg.RosPack().get_path("tiago_description")
        robot = Robot(
            robot_urdf,
            package_dirs=package_dirs,
            isFext=isFext,
        )
        print("Robot model is loaded from " + package_dirs)

    else:
        import rospy
        from pinocchio.robot_wrapper import RobotWrapper

        robot_xml = rospy.get_param("robot_description")
        if isFext:
            robot = RobotWrapper(
                pinocchio.buildModelFromXML(
                    robot_xml, root_joint=pinocchio.JointModelFreeFlyer()
                )
            )
        else:
            robot = pinocchio.buildModelFromXML(robot_xml, root_joint=None)
    return robot


def write_to_xacro(tiago_calib, file_name=None, file_type="yaml"):
    """
    Write calibration result to xacro file.
    """
    assert tiago_calib.STATUS == "CALIBRATED", "Calibration not performed yet"
    model = tiago_calib.model
    calib_result = tiago_calib.calibrated_param
    param = tiago_calib.param

    calibration_parameters = {}
    calibration_parameters["camera_position_x"] = float(
        calib_result["base_px"]
    )
    calibration_parameters["camera_position_y"] = float(
        calib_result["base_py"]
    )
    calibration_parameters["camera_position_z"] = float(
        calib_result["base_pz"]
    )
    calibration_parameters["camera_orientation_r"] = float(
        calib_result["base_phix"]
    )
    calibration_parameters["camera_orientation_p"] = float(
        calib_result["base_phiy"]
    )
    calibration_parameters["camera_orientation_y"] = float(
        calib_result["base_phiz"]
    )

    for idx in param["actJoint_idx"]:
        joint = model.names[idx]
        for key in calib_result.keys():
            if joint in key and "torso" not in key:
                calibration_parameters[joint + "_offset"] = float(
                    calib_result[key]
                )
    if tiago_calib.param["measurability"][0:3] == [True, True, True]:
        calibration_parameters["tip_position_x"] = float(
            calib_result["pEEx_1"]
        )
        calibration_parameters["tip_position_y"] = float(
            calib_result["pEEy_1"]
        )
        calibration_parameters["tip_position_z"] = float(
            calib_result["pEEz_1"]
        )
    if tiago_calib.param["measurability"][3:6] == [True, True, True]:
        calibration_parameters["tip_orientation_r"] = float(
            calib_result["phiEEx_1"]
        )
        calibration_parameters["tip_orientation_p"] = float(
            calib_result["phiEEy_1"]
        )
        calibration_parameters["tip_orientation_y"] = float(
            calib_result["phiEEz_1"]
        )


    if file_type == "xacro":
        if file_name is None:
            path_save_xacro = abspath(
                "data/calibration_paramters/tiago_master_calibration_{}.xacro".format(
                    param["NbSample"]
                )
            )
        else:
            path_save_xacro = abspath(
                "data/calibration_parameters/" + file_name
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
        if file_name is None:
            path_save_yaml = abspath(
                "data/calibration_parameters/tiago_master_calibration_{}.yaml".format(
                    param["NbSample"]
                )
            )
        else:
            path_save_yaml = abspath(
                "data/calibration_parameters/" + file_name
            )
        with open(path_save_yaml, "w") as output_file:
            # for parameter in calibration_parameters.keys():
            #     update_name = parameter
            #     update_value = calibration_parameters[parameter]
            #     update_line = "{}:{}".format(update_name, update_value)
            #     output_file.write(update_line)
            #     output_file.write("\n")
            try:
                yaml.dump(
                    calibration_parameters,
                    output_file,
                    sort_keys=False,
                    default_flow_style=False,
                )
            except yaml.YAMLError as exc:
                print(exc)


def model_todict(robot_model):
    """
    Convert the robot model jointPlacements in dictionary format.
    """
    model_dict = {}
    for jj, jname in enumerate(robot_model.names):
        model_dict[jname] = [
            round(num, 6)
            for num in robot_model.jointPlacements[jj].translation.tolist()
            + pin.rpy.matrixToRpy(
                robot_model.jointPlacements[jj].rotation
            ).tolist()
        ]
    return model_dict


def save_model(robot_model, model_name):
    """
    Save the robot model in dictionary format.
    """
    model_dict = model_todict(robot_model)
    with open(
        "data/calibration_parameters/{}.yaml".format(model_name), "w"
    ) as output_file:
        try:
            yaml.dump(
                model_dict,
                output_file,
                sort_keys=False,
                default_flow_style=True,
            )
        except yaml.YAMLError as exc:
            print(exc)


def compare_model(model_dict1, model_dict2):
    """
    Compare two robot models in dictionary format.
    """
    for jname in model_dict1.keys():
        assert jname in model_dict2.keys()
        assert len(model_dict1[jname]) == len(model_dict2[jname])
        for i in range(len(model_dict1[jname])):
            assert model_dict1[jname][i] == model_dict2[jname][i]
    print("Models are equal")


def main():
    return 0


if __name__ == "__main__":
    tiago = load_robot("data/urdf/tiago_48_hey5.urdf")
    TAGcalib_cen = TiagoCalibration(
        tiago, "config/tiago_config_hey5_center.yaml"
    )
    TAGcalib_cen.initialize()
    TAGcalib_cen.solve()

    TAGcalib_tl = TiagoCalibration(
        tiago, "config/tiago_config_hey5_topleft.yaml"
    )
    TAGcalib_tl.initialize()
    TAGcalib_tl.solve()

    var_cen0 = TAGcalib_cen.LM_result.x
    var_tl0 = TAGcalib_tl.LM_result.x

    var0_common = var_cen0[
        0 : -(
            TAGcalib_cen.param["NbMarkers"]
            * TAGcalib_cen.param["calibration_index"]
        )
    ]
    var0_cenPose = var_cen0[
        -TAGcalib_cen.param["NbMarkers"]
        * TAGcalib_cen.param["calibration_index"] :
    ]
    var0_tlPose = var_tl0[
        -TAGcalib_tl.param["NbMarkers"]
        * TAGcalib_tl.param["calibration_index"] :
    ]
    var0 = np.append(var0_common, var0_cenPose)
    var0 = np.append(var0, var0_tlPose)

    # cost function for the optimization problem
    def cost_function(var):
        """Combine two cost functions for the optimization problem."""
        var_common = var[
            0 : -(
                TAGcalib_cen.param["NbMarkers"]
                * TAGcalib_cen.param["calibration_index"]
                + TAGcalib_tl.param["NbMarkers"]
                * TAGcalib_tl.param["calibration_index"]
            )
        ]
        var_cenPose = var[
            -(
                TAGcalib_cen.param["NbMarkers"]
                * TAGcalib_cen.param["calibration_index"]
                + TAGcalib_tl.param["NbMarkers"]
                * TAGcalib_tl.param["calibration_index"]
            ) : -TAGcalib_tl.param["NbMarkers"]
            * TAGcalib_tl.param["calibration_index"]
        ]
        var_tlPose = var[
            -TAGcalib_tl.param["NbMarkers"]
            * TAGcalib_tl.param["calibration_index"] :
        ]
        # calib_center error function for the optimization problem
        var_cen = np.append(var_common, var_cenPose)
        # coeff_cen = TAGcalib_cen.param["coeff_regularize"]
        PEEe_cen = update_forward_kinematics(
            TAGcalib_cen.model,
            TAGcalib_cen.data,
            var_cen,
            TAGcalib_cen.q_measured,
            TAGcalib_cen.param,
        )
        res_vect = TAGcalib_cen.PEE_measured - PEEe_cen

        # calib_topleft error function for the optimization problem
        var_tl = np.append(var_common, var_tlPose)
        # coeff_tl = TAGcalib_tl.param["coeff_regularize"]
        PEEe_tl = update_forward_kinematics(
            TAGcalib_tl.model,
            TAGcalib_tl.data,
            var_tl,
            TAGcalib_tl.q_measured,
            TAGcalib_tl.param,
        )
        res_vect = np.append(res_vect, (TAGcalib_tl.PEE_measured - PEEe_tl))
        # res_vect = np.append(
        #     res_vect,
        #     np.sqrt(coeff_tl)
        #     * var_tl[
        #         6 : -TAGcalib_tl.param["NbMarkers"]
        #         * TAGcalib_tl.param["calibration_index"]
        #     ],
        # )

        # chessboard center displacement
        displ_vect = var_tlPose - var_cenPose
        displ_ref = np.array([0.01, 0, -0.02, 0, 0, 0])
        res_vect = np.append(
            res_vect,
            0.01 * TAGcalib_cen.param["NbSample"] * (displ_vect - displ_ref),
        )

        # chessboard orientation
        cb_orient = np.array([0, -np.pi / 2, -np.pi / 2])
        res_vect = np.append(
            res_vect,
            0.01
            * TAGcalib_cen.param["NbSample"]
            * (var_cenPose[-3:] - cb_orient),
        )
        res_vect = np.append(
            res_vect,
            0.01
            * TAGcalib_cen.param["NbSample"]
            * (var_tlPose[-3:] - cb_orient),
        )

        # torso to zero
        res_vect = np.append(
            res_vect,
            TAGcalib_cen.param["NbSample"]
            * var[
                TAGcalib_cen.param["param_name"].index(
                    "offsetPZ_torso_lift_joint"
                )
            ],
        )

        # head joints to close to zero
        res_vect = np.append(
            res_vect,
            0.1
            * TAGcalib_cen.param["NbSample"]
            * var[
                TAGcalib_cen.param["param_name"].index("offsetRZ_head_1_joint")
            ],
        )
        res_vect = np.append(
            res_vect,
            0.1
            * TAGcalib_cen.param["NbSample"]
            * var[
                TAGcalib_cen.param["param_name"].index("offsetRZ_head_2_joint")
            ],
        )
        # camera pose to close to initial pose
        cam_pose = np.array([0.0908, 0.08, 0.0, -np.pi/2, 0.0, 0.0])
        res_vect = np.append(
            res_vect,
            1
            * TAGcalib_cen.param["NbSample"]
            * (var[0:6] - cam_pose)
        )
        # arm1, arm2, arm3 to zero
        arm123_coeff = 0.5
        res_vect = np.append(
            res_vect,
            arm123_coeff
            * TAGcalib_cen.param["NbSample"]
            * (var[
                TAGcalib_cen.param["param_name"].index("offsetRZ_arm_1_joint")
            ] - 0.01),
        )
        res_vect = np.append(
            res_vect,
            arm123_coeff
            * TAGcalib_cen.param["NbSample"]
            * (var[
                TAGcalib_cen.param["param_name"].index("offsetRZ_arm_2_joint")
            ] - 0.005),
        )
        res_vect = np.append(
            res_vect,
            arm123_coeff
            * TAGcalib_cen.param["NbSample"]
            * var[
                TAGcalib_cen.param["param_name"].index("offsetRZ_arm_3_joint")
            ],
        )
        # arm5
        # res_vect = np.append(
        #     res_vect,
        #     10
        #     * TAGcalib_cen.param["NbSample"]
        #     * var[
        #         TAGcalib_cen.param["param_name"].index("offsetRZ_arm_5_joint")
        #     ] + 0.2,
        # )
        return res_vect

    # define solver parameters
    del_list_ = []
    res = var0
    # outlier_eps = self.param["outlier_eps"]
    print("*" * 50)
    # define solver
    LM_solve = least_squares(
        cost_function,
        var0,
        method="lm",
        verbose=1,
        args=(),
    )

    # solution
    res = LM_solve.x
    res_cen = res[
        0 : -TAGcalib_tl.param["NbMarkers"]
        * TAGcalib_tl.param["calibration_index"]
    ]
    res_tl = np.append(
        res[
            0 : -(
                TAGcalib_cen.param["NbMarkers"]
                * TAGcalib_cen.param["calibration_index"]
                + TAGcalib_tl.param["NbMarkers"]
                * TAGcalib_tl.param["calibration_index"]
            )
        ],
        res[
            -TAGcalib_tl.param["NbMarkers"]
            * TAGcalib_tl.param["calibration_index"] :
        ],
    )

    def print_solution(TAGcalib, res_):
        print("solution of calibrated parameters: ")
        for x_i, xname in enumerate(TAGcalib.param["param_name"]):
            print(x_i + 1, xname, list(res_)[x_i])

    print("--------------------------------")
    print("chessboard center")
    print_solution(TAGcalib_cen, res_cen)

    PEEe_censol = update_forward_kinematics(
        TAGcalib_cen.model,
        TAGcalib_cen.data,
        res_cen,
        TAGcalib_cen.q_measured,
        TAGcalib_cen.param,
    )
    TAGcalib_cen.calc_errors(PEEe_censol)
    print("--------------------------------")
    print("chessboard center")
    print_solution(TAGcalib_tl, res_tl)

    PEEe_tlsol = update_forward_kinematics(
        TAGcalib_tl.model,
        TAGcalib_tl.data,
        res_tl,
        TAGcalib_tl.q_measured,
        TAGcalib_tl.param,
    )
    TAGcalib_tl.calc_errors(PEEe_tlsol)
    print("--------------------------------")
    print("relative displacement of cb poses:", res_tl[-6:] - res_cen[-6:])
    param_cen = [float(res_i_) for res_i_ in res_cen]
    TAGcalib_cen.calibrated_param = dict(
        zip(TAGcalib_cen.param["param_name"], param_cen)
    )
    param_tl = [float(res_i_) for res_i_ in res_tl]
    TAGcalib_tl.calibrated_param = dict(
        zip(TAGcalib_tl.param["param_name"], param_tl)
    )
    write_to_xacro(
        TAGcalib_cen,
        file_name="tiago_master_calibration_cbcenter.yaml",
        file_type="yaml",
    )
