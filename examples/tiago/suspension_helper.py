import pinocchio as pin

from matplotlib import pyplot as plt
import numpy as np
from figaroh.calibration.calibration_tools import cartesian_to_SE3


def calc_floatingbase_pose(
    base_marker_name: str,
    marker_data: dict,
    marker_base: dict,
    Mref=pin.SE3.Identity(),
):
    """Calculate floatingbase pose w.r.t a fixed frame from measures expressed
    in mocap frame.

    Args:
        marker_data (dict): vicon marker data
        marker_base (dict): SE3 from a marker to a fixed base
        base_marker_name (str): base marker name
        Mref (_type_, optional): Defaults to pin.SE3.Identity().

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray : position, rpy, quaternion
    """

    Mmarker_floatingbase = cartesian_to_SE3(marker_base[base_marker_name])
    [base_trans, base_rot] = marker_data[base_marker_name]

    n_ = len(base_rot)
    xyz_u = np.zeros((n_, 3))
    rpy_u = np.zeros((n_, 3))
    quat_u = np.zeros((n_, 4))

    for i in range(n_):
        SE3_floatingbase = (
            Mref
            * pin.SE3(base_rot[i], base_trans[i, :])
            * Mmarker_floatingbase
        )

        xyz_u[i, :] = SE3_floatingbase.translation
        rpy_u[i, :] = pin.rpy.matrixToRpy(SE3_floatingbase.rotation)
        quat_u[i, :] = pin.Quaternion(SE3_floatingbase.rotation).coeffs()

    return xyz_u, rpy_u, quat_u


def estimate_fixed_base(
    base_marker_name: str,
    marker_data: dict,
    marker_fixedbase: dict,
    stationary_range: range,
    Mref=pin.SE3.Identity(),
):
    """Estimate fixed base frame of robot expressed in a fixed frame

    Args:
        marker_data (dict): vicon marker data
        marker_fixedbase (dict): SE3 from a marker to a fixed base
        base_marker_name (str): base marker name
        stationary_range (range): range to extract data
        Mref (SE3, optional): Defaults to pin.SE3.Identity() as mocap frame.

    Returns:
        SE3: transformatiom SE3
    """

    Mmarker_fixedbase = cartesian_to_SE3(marker_fixedbase[base_marker_name])
    [base_trans, base_rot] = marker_data[base_marker_name]

    ne_ = len(stationary_range)
    xyz_u_ = np.zeros((ne_, 3))
    rpy_u_ = np.zeros((ne_, 3))

    for j, i in enumerate(stationary_range):
        SE3_fixdebase = (
            pin.SE3(base_rot[i], base_trans[i, :]) * Mmarker_fixedbase
        )
        xyz_u_[j, :] = SE3_fixdebase.translation
        rpy_u_[j, :] = pin.rpy.matrixToRpy(SE3_fixdebase.rotation)

    xyz_mean = np.mean(xyz_u_, axis=0)
    rpy_mean = np.mean(rpy_u_, axis=0)

    return cartesian_to_SE3(np.append(xyz_mean, rpy_mean))


def compute_estimated_pee(
    endframe_name: str,
    base_marker_name: str,
    marker_fixedbase: dict,
    Mmocap_fixedbase,
    Mendframe_marker,
    mocap_range_: range,
    q_arm: np.ndarray,
    marker_data,
    tiago_fb,
):
    """_summary_

    Args:
        endframe_name (str): name of the end of kinematic chain
        base_marker_name (str): name of base marker measured
        marker_fixedbase (dict): fixed transformation from marker to fixed base
        Mmocap_fixedbase (SE3): transf. from mocap to fixed base
        Mendframe_marker (SE3): transf. from end of kin.chain to last marker
        mocap_range_ (range): range of selected mocap data
        q_arm (np.ndarray): encoder data within 'encoder_range'

    Returns:
        numpy.ndarray, numpy.ndarray: estimate of last marker in mocap frame
    """
    assert len(q_arm) == len(
        mocap_range_
    ), "joint configuration range is not matched with mocap range."

    model = tiago_fb.model
    data = tiago_fb.data

    endframe_id = model.getFrameId(endframe_name)
    Mmarker_fixedbase = cartesian_to_SE3(marker_fixedbase[base_marker_name])
    [base_trans, base_rot] = marker_data[base_marker_name]

    nc_ = len(mocap_range_)
    pee_est = np.zeros((nc_, 3))
    peews_est = np.zeros((nc_, 3))
    phiee_est = np.zeros((nc_, 3))
    phieews_est = np.zeros((nc_, 3))

    for jj, ii in enumerate(mocap_range_):
        pin.framesForwardKinematics(model, data, q_arm[jj])
        pin.updateFramePlacements(model, data)

        Mmocap_floatingbase = (
            pin.SE3(base_rot[ii], base_trans[ii, :]) * Mmarker_fixedbase
        )

        pee_SE3 = Mmocap_fixedbase * data.oMf[endframe_id] * Mendframe_marker

        peews_SE3 = (
            Mmocap_floatingbase * data.oMf[endframe_id] * Mendframe_marker
        )

        pee_est[jj, :] = pee_SE3.translation
        peews_est[jj, :] = peews_SE3.translation
        phiee_est[jj, :] = pin.rpy.matrixToRpy(pee_SE3.rotation)
        phieews_est[jj, :] = pin.rpy.matrixToRpy(peews_SE3.rotation)
    return (
        pee_est,
        peews_est,
        phiee_est,
        phieews_est,
    )


def plot_compare_suspension(
    pee_est=None,
    peews_est=None,
    gripper3_pos=None,
    lookup_marker="gripper",
    plot_err=True,
):
    fig, ax = plt.subplots(3, 1)
    t_tf2 = []
    t_tf = []
    if pee_est is not None:
        t_tf = np.arange(len(pee_est))
        ax[0].plot(
            t_tf,
            peews_est[:, 0],
            color="red",
            label="estimated with suspension added",
        )
        ax[1].plot(
            t_tf,
            peews_est[:, 1],
            color="blue",
            label="estimated with suspension added",
        )
        ax[2].plot(
            t_tf,
            peews_est[:, 2],
            color="green",
            label="estimated with suspension added",
        )
    if peews_est is not None:
        t_tf = np.arange(len(peews_est))
        ax[0].plot(
            t_tf,
            pee_est[:, 0],
            color="red",
            linestyle="dotted",
            label="estimated without suspension added",
        )
        ax[1].plot(
            t_tf,
            pee_est[:, 1],
            color="blue",
            linestyle="dotted",
            label="estimated without suspension added",
        )
        ax[2].plot(
            t_tf,
            pee_est[:, 2],
            color="green",
            linestyle="dotted",
            label="estimated without suspension added",
        )
    if gripper3_pos is not None:
        t_tf2 = np.arange(len(gripper3_pos))
        ax[0].plot(
            t_tf2,
            gripper3_pos[:, 0],
            color="red",
            label="measured",
            linestyle="--",
        )
        ax[1].plot(
            t_tf2,
            gripper3_pos[:, 1],
            color="blue",
            label="measured",
            linestyle="--",
        )
        ax[2].plot(
            t_tf2,
            gripper3_pos[:, 2],
            color="green",
            label="measured",
            linestyle="--",
        )

    ax3 = ax[0].twinx()
    ax4 = ax[1].twinx()
    ax5 = ax[2].twinx()
    if plot_err:
        if pee_est is not None and gripper3_pos is not None:
            ax3.bar(
                t_tf,
                pee_est[:, 0] - gripper3_pos[:, 0],
                color="black",
                label="errors - x axis",
                alpha=0.3,
            )
            ax4.bar(
                t_tf,
                pee_est[:, 1] - gripper3_pos[:, 1],
                color="black",
                label="errors - without susspension added",
                alpha=0.3,
            )
            ax5.bar(
                t_tf,
                pee_est[:, 2] - gripper3_pos[:, 2],
                color="black",
                label="errors - z axis",
                alpha=0.3,
            )
        if peews_est is not None and gripper3_pos is not None:

            ax3.bar(
                t_tf,
                peews_est[:, 0] - gripper3_pos[:, 0],
                color="red",
                label="errors - x axis",
                alpha=0.3,
            )
            ax4.bar(
                t_tf,
                peews_est[:, 1] - gripper3_pos[:, 1],
                color="blue",
                label="errors - with susspension added",
                alpha=0.3,
            )
            ax5.bar(
                t_tf,
                peews_est[:, 2] - gripper3_pos[:, 2],
                color="green",
                label="errors - z axis",
                alpha=0.3,
            )
    ax4.legend()
    ax[0].legend()
    ax[0].set_ylabel("x component (meter)")
    ax[1].set_ylabel("y component (meter)")
    ax[2].set_ylabel("z component (meter)")
    ax[2].set_xlabel("sample")
    fig.suptitle("Marker Position of {}".format(lookup_marker))


# fixed frame:
# mocap frame = marker position measures
# forceplate frame = force and moment measures
# fixed base frame = base_footprint (free-flyer joint) at stationary state

# dynamic frame:
# floating base frame = base_footprint in dynamic state
# marker_shoulder frame = marker on shoulder
# marker_base frame = marker on base
# marker_gripper frame = marker on gripper

# [mocap frame] ---- <marker base frame> ---(4) Mmarker_floatingbase---><floating base frame>
#     |        \
#     |                  \
# (1) Mmocap_forceplate                \
#     |                                 (2) Mmocap_fixedbase
#     |                                             \
#     |                                                         \
# [forceplate frame]---(3) Mforceplate_fixedbase----->[fixed base frame or robot frame]


# (1) forceplate - mocap reference frame transformation
forceplate_frame_rot = np.array([[-1, 0.0, 0.0], [0, 1.0, 0], [0, 0, -1]])
forceplate_frame_trans = np.array([0.9, 0.45, 0.0])

Mmocap_forceplate = pin.SE3(forceplate_frame_rot, forceplate_frame_trans)
Mforceplate_mocap = Mmocap_forceplate.inverse()


# (2.1) mocap - fixedbase footprint
Mmocap_fixedfootprint = cartesian_to_SE3(
    np.array(
        [
            9.21473783e-01,
            5.16372267e-01,
            -5.44391042e-03,
            1.42087717e-02,
            7.31715478e-04,
            -1.61396925e00,
        ]
    )
)
Mfixedfootprint_mocap = Mmocap_fixedfootprint.inverse()

# (2.2) mocap - fixedbase baselink
Mmocap_fixedbaselink = cartesian_to_SE3(
    np.array(
        [
            9.20782862e-01,
            5.17310886e-01,
            9.31457902e-02,
            1.42957502e-02,
            6.76446542e-04,
            -1.61388066e00,
        ]
    )
)
Mfixedbaselink_mocap = Mmocap_fixedbaselink.inverse()


# (3.1) forceplate - robot fixed baselink frame transformation
Mforceplate_fixedfootprint = Mforceplate_mocap * Mmocap_fixedfootprint
Mfixedfootprint_forceplate = Mforceplate_fixedfootprint.inverse()

# (3.2) forceplate - robot fixed baselink frame transformation
Mforceplate_fixedbaselink = Mforceplate_mocap * Mmocap_fixedbaselink
Mfixedbaselink_forceplate = Mforceplate_fixedbaselink.inverse()

# (4.1) marker on the base - robot floating base_footprint transformation
marker_footprint = dict()
marker_footprint["base2"] = np.array(
    [
        0.01905164792882577,
        -0.20057504109760418,
        -0.3148863380453684,
        0.006911212803801684,
        0.009815807728356198,
        0.053830497405014326,
    ]
)
Mmarker_footprint = cartesian_to_SE3(marker_footprint["base2"])
Mfootprint_marker = Mmarker_footprint.inverse()

# (4.2) marker on the base - robot floating baselink transformation
marker_baselink = dict()
marker_baselink["base2"] = np.array(
    [
        0.01904,
        -0.200587,
        -0.21628975,
        0.006999,
        0.00976118669378908,
        0.0539194899,
    ]
)
Mmarker_baselink = cartesian_to_SE3(marker_baselink["base2"])
Mbaselink_marker = Mmarker_baselink.inverse()

# (5) gripper_tool_link to gripper 3 marker
Mwrist_gripper3 = cartesian_to_SE3(
    np.array(
        [
            -0.02686200922255708,
            -0.00031620763696974614,
            -0.1514577985136796,
            0,
            0,
            0,
        ]
    )
)

# (6) torso_lift_link to shoulder_1 marker
Mshoulder_torso = cartesian_to_SE3(
    np.array(
        [
            0.14574771716972612,
            0.1581171862116727,
            -0.0176625098292798,
            -0.005016136500979825,
            0.006322745940971755,
            0.027530310085705736,
        ]
    )
)
Mtorso_shoulder = Mshoulder_torso.inverse()


# pu.plot_SE3(pin.SE3.Identity())

# pu.plot_SE3(estimate_fixed_base(marker_datas[0], marker_baselink, "base2", range(50, 1000)), "baselink")

# pu.plot_SE3(pin.SE3(marker_datas[0]["base2"][1][0], marker_datas[0]["base2"][0][0]), "base2")
# pu.plot_SE3(estimate_fixed_base(marker_datas[0], marker_footprint, "base2", range(50, 1000)), "footprint")
# pu.plot_markertf(marker_datas[0]["base2"][0])


# suspension parameters
#          kx_t        cx_t        nx_t         ky_t      cy_t        ny_t       kz_t         cz_t       nz_t       kx_r       cx_r        nx_r        ky_r       cy_r        ny_r       kz_r       cz_r       nz_r
# --  ---------  ----------  ----------  -----------  --------  ----------  ---------  -----------  ---------  ---------  ---------  ----------  ----------  ---------  ----------  ---------  ---------  ---------
#  0  91263.4    7187.43     1970.79     -27238.1     -2851.04  1806.82      16388.9   -16388.4      -819.378   21650.5    1133.13     67675.8    80305.9     8926.74     50.8998    -260.265   447.576    -399.304
#  1    127.719   267.252       8.04827  -39443.4     -3194.98  2622.34      86622.2   -20589.9     -1200.81   -62228.8   -7850.97   -194697        418.258    213.365    17.3418   -2432.65    599.876   -3718.61
#  2  52671.8    3613.77     1115.57      -7695.64    -3023.68   550.307     -1483.19     -23.5074   -721.664   -4761.33   -750.434   -14914.3   -62217.7     1235      -105.027     -855.957  -101.744   -1290.59
#  3  54036.3    3907.27     1165.37     -36138.3     -2808.73  2395.21     -79753.9    36607.7      -294.753   27258      2731.74     85219.5   109910       5088.58     71.7775    -493.16    339.607    -755.116
#  4  34966.6    5105.72      760.618    -13162       -3121.36   874.002     70327.8   -37055.6     -1110.23     1559.76    793.889     4856.42  -69723.8    -5487.41    -89.9673   -1235.36   -153.613   -1904.32
#  5  48221.9    5380.84     1046.39         34.3674  -1988.18    -6.93978   30547       5601.91     -895.475    1119.28   -804.706     3479.44   -6453.46    -658.964    -1.18849   -988.419  -110.341   -1523.42
#  6  22555      1997.53      481.692    -16801.3     -4771.86  1158.74     -32730      39314.3      -551.004   -6545.91   -983.792   -20505.2   -64815.4    -4677.83    -98.4621    -787.036    18.8089  -1197.57
#  7  17381      5048.8       375.14     -16107.4     -5146.4   1074.4      -11837.6   -18622.4      -665.209   -7079.95    114.099   -22172.5   -40501.4    -1890.06    -18.4334    -818.05   -217.77    -1247.36
#  8  -6866.63      5.25113  -141.999    -38935.3     -3659.72  2590.46      66368.4   -61263.6     -1091.1    -72395.1   13068.3    -226491       3274.71    -970.053    20.8176   -4067.72   2494.25    -6217.23
