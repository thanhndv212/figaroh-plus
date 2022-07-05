# this file is set all the parameters required for the generation of the Optimal
# Exciting Motions (OEM)

# from pickle import FALSE
import numpy as np


"""
params_settings = {
    #### parameters related to the identification model
    "is_external_wrench": False,
    "is_joint_torques": True,
    # forces and torques you desire under the format : Fx,Fy,Fz,Mx,My,Mz
    # (NB : answer 'All' gives the regressor for Fx,Fy,Fz,Mx,My,Mz
    "force_torque": [ "All" ],
    "has_friction": True,
    "has_joint_offset": False,
    "has_actuator_inertia": False,
    "has_coupled_wrist": False,
    "K_1": 2,  # default values of drive gains in case they are used
    "K_2": 2,  # default values of drive gains in case they are used
    # Use this flag to work only with the static regressor (segment's masses and COM)
    "is_static_regressor": True,
    # Use this flag to work only with the inertial regressor (segment's inertia)
    "is_inertia_regressor": True,
    # Use this flag to specify the index of the body on which you have set the
    # additional mass (for the loaded part)
    "which_body_loaded": 2,
    #### defaut values to be set in case the URDF is incomplete
    # default value for joint limits in cas the URDF does not have the info
    "q_lim_def": 1.57,
    "dq_lim_def": 5,  # in rad.s-1
    "ddq_lim_def": 20,  # in rad.s-2
    "tau_lim_def": 4,  # in N.m
    #### parameters related to all exciting trajectories
    "ts": 1 / 400,  # Sampling frequency of the trajectories to be recorded
    #### parameters related to the data elaboration (filtering, etc...)
    "cut_off_frequency_butterworth": 100,
    #### parameters related to the trapezoidal trajectory generation
    "nb_repet_trap": 2,  # number of repetition of the trapezoidal motions
    "trapez_vel_steps": 10,  # Velocity step in % of the max velocity
    #### parameters related to the OEM generation
    "tf": 1,  # duration of one OEM
    "nb_iter_OEM": 1,  # number of OEM to be optimized
    "nb_harmonics": 4,  # number of harmonics of the fourier serie
    "freq": 1,  # frequency of the fourier serie coefficient
    "q_safety": 0.08,  # value in radian (=5deg) to remove from the actual joint limits
    "eps_gradient": 1e-5,  # numerical gradient step
    "is_fourier_series": True,  # use Fourier series for trajectory generation
    "time_static": 2,  # in seconds the time for each static pose
    #### parameters related to the Totl least square identification
    "mass_load": 3.0,
    # move all joint simultaneously when using quintic polynomial interpolation
    "sync_joint_motion": False,
    #### parameters related to  animation/saving data
    "ANIMATE": False,  # plot flag for gepetto-viewer
    "SAVE_FILE": True,
}
params_settings["nb_samples"] = int(params_settings["tf"] / params_settings["ts"])
"""


def get_params_settings():
    params_settings = {
        # ### parameters related to the identification model
        "is_external_wrench": False,
        "is_joint_torques": True,
        # forces and torques you desire under the format :
        # Fx,Fy,Fz,Mx,My,Mz (NB : answer 'All' gives the regressor for Fx,Fy,Fz,Mx,My,Mz
        "force_torque": ["All"],
        "has_friction": True,
        "has_joint_offset": False,
        "has_actuator_inertia": False,
        "has_coupled_wrist": False,
        "K_1": 2,  # default values of drive gains in case they are used
        "K_2": 2,  # default values of drive gains in case they are used
        # Use this flag to work only with the static regressor (segment's masses and
        # COM)
        "is_static_regressor": True,
        # Use this flag to work only with the inertial regressor (segment's inertia)
        "is_inertia_regressor": True,
        # Use this flag to specify the index of the body on which you have set the
        # additional mass (for the loaded part)
        "which_body_loaded": 2,
        # ### defaut values to be set in case the URDF is incomplete
        # default value for joint limits in cas the URDF does not have the info
        "q_lim_def": 1.57,
        "dq_lim_def": 5,  # in rad.s-1
        "ddq_lim_def": 20,  # in rad.s-2
        "tau_lim_def": 4,  # in N.m
        # ### parameters related to all exciting trajectories
        # Sampling time of the trajectories to be/that have been recorded
        "ts": 1 / 400,
        # ### parameters related to the data elaboration (filtering, etc...)
        "cut_off_frequency_butterworth": 100,
        # ### parameters related to the trapezoidal trajectory generation
        "nb_repet_trap": 2,  # number of repetition of the trapezoidal motions
        "trapez_vel_steps": 10,  # Velocity step in % of the max velocity
        # ### parameters related to the OEM generation
        "tf": 1,  # duration of one OEM
        "nb_iter_OEM": 1,  # number of OEM to be optimized
        "nb_harmonics": 4,  # number of harmonics of the fourier serie
        "freq": 1,  # frequency of the fourier serie coefficient
        # value in radian (=5deg) to remove from the actual joint limits
        "q_safety": 0.08,
        "eps_gradient": 1e-5,  # numerical gradient step
        "is_fourier_series": True,  # use Fourier series for trajectory generation
        "time_static": 2,  # in seconds the time for each static pose
        # ### parameters related to the Totl least square identification
        "mass_load": 3.0,
        # move all joint simultaneously when using quintic polynomial interpolation
        "sync_joint_motion": False,
        # ### parameters related to  animation/saving data
        "ANIMATE": False,  # plot flag for gepetto-viewer
        "SAVE_FILE": True,
    }
    params_settings["nb_samples"] = int(params_settings["tf"] / params_settings["ts"])
    return params_settings


def set_missing_params_setting(robot, params_settings):

    diff_limit = np.setdiff1d(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    # upper and lower joint limits are the same so use defaut values for all joints
    if not diff_limit:
        print("No joint limits. Set default values")
        for ii in range(robot.model.nq):
            robot.model.lowerPositionLimit[ii] = -params_settings["q_lim_def"]
            robot.model.upperPositionLimit[ii] = params_settings["q_lim_def"]

    if np.sum(robot.model.velocityLimit) == 0:
        print("No velocity limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.velocityLimit[ii] = params_settings["dq_lim_def"]

    # maybe we need to check an other field somethnibg weird here
    if np.sum(robot.model.velocityLimit) == 0:
        print("No joint torque limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.effortLimit[ii] = -params_settings["tau_lim_def"]

    accelerationLimit = np.zeros(robot.model.nq)
    for ii in range(robot.model.nq):
        # accelerationLimit to be consistent with PIN naming
        accelerationLimit[ii] = params_settings["ddq_lim_def"]
    params_settings["accelerationLimit"] = accelerationLimit
    # print(model.accelerationLimit)

    if params_settings["has_friction"]:

        for ii in range(robot.model.nv):
            if ii == 0:
                # default values of the joint viscous friction in case they are used
                fv = [(ii + 1) / 10]
                # default value of the joint static friction in case they are used
                fs = [(ii + 1) / 10]
            else:
                fv.append((ii + 1) / 10)
                fs.append((ii + 1) / 10)

        params_settings["fv"] = fv
        params_settings["fs"] = fs

    return params_settings


# isFrictionincld = False
# if len(argv) > 1:
#    if argv[1] == '-f':
#        isFrictionincld = True

# fv = 0.05
# fc = 0.01
