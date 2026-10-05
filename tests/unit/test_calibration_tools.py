"""Tests for calc_updated_fkm (figaroh.calibration.calibration_tools).

These are regression tests for a merge that folded the elasticity and
camera-ref-frame support of the (now-deleted, previously dead-code)
``update_forward_kinematics`` into ``calc_updated_fkm``, fixing four bugs
found in the process:

1. The base/camera transform (``bMo``) was computed but never composed
   into the final pose.
2. Elasticity indexing (``xyz_rpy[elas_id + 3]``) went out of bounds for
   any rotary joint.
3. The elasticity match loop reused a stale ``key`` left over from an
   unrelated earlier loop instead of iterating its own.
4. The per-sample pose write was gated on a parameter-count bookkeeping
   variable that accumulated across the whole sample loop instead of
   resetting per sample, silently zeroing out later samples.

Each test below is named after the behavior/bug it guards.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.calibration_tools import (
    apply_joint_offset,
    calc_updated_fkm,
    update_joint_placement,
    get_rel_transform,
    calculate_base_kinematics_regressor,
)
from figaroh.tools.qrdecomposition import redistribute_min_norm


def _base_calib_config(**overrides):
    config = {
        "calib_model": "full_params",
        "base_to_ref_frame": None,
        "ref_frame": None,
        "non_geom": False,
        "NbMarkers": 1,
        "NbSample": 1,
    }
    config.update(overrides)
    return config


class TestBaseFrameComposition:
    """calc_updated_fkm must compose wMo into the final pose (bug #1)."""

    def test_unknown_baseframe_shifts_position(self, temp_urdf):
        model = pin.buildModelFromUrdf(temp_urdf)
        data = model.createData()
        j1 = model.getJointId("joint1")

        param_name = [
            "base_px",
            "base_py",
            "base_pz",
            "base_phix",
            "base_phiy",
            "base_phiz",
            "d_px_joint1",
            "d_py_joint1",
            "d_pz_joint1",
            "d_phix_joint1",
            "d_phiy_joint1",
            "d_phiz_joint1",
        ]
        calib_config = _base_calib_config(
            start_frame="base_link",
            end_frame="link1",
            actJoint_idx=[j1],
            measurability=[True] * 6,
            calibration_index=6,
            param_name=param_name,
        )
        q = np.zeros((1, model.nq))

        var_zero = np.zeros(len(param_name))
        pee_zero = calc_updated_fkm(model, data, var_zero, q, calib_config)

        var_shifted = var_zero.copy()
        var_shifted[0] = 0.1  # base_px
        pee_shifted = calc_updated_fkm(model, data, var_shifted, q, calib_config)

        # a pure world-frame translation offset must shift position by
        # exactly that amount and leave orientation untouched
        assert pee_shifted[0] == pytest.approx(pee_zero[0] + 0.1)
        assert pee_shifted[1:] == pytest.approx(pee_zero[1:])

    def test_camera_ref_frame_affects_output(self, two_joint_urdf):
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        j1 = model.getJointId("joint1")
        j2 = model.getJointId("joint2")

        param_name = [
            "base_px",
            "base_py",
            "base_pz",
            "base_phix",
            "base_phiy",
            "base_phiz",
        ]
        calib_config = _base_calib_config(
            start_frame="base_link",
            end_frame="link2",
            base_to_ref_frame="link1",
            ref_frame="link1",
            actJoint_idx=[j1, j2],
            measurability=[True] * 6,
            calibration_index=6,
            param_name=param_name,
        )
        q = np.zeros((1, model.nq))

        pee_identity_anchor = calc_updated_fkm(
            model, data, np.zeros(6), q, calib_config
        )
        var_anchor_offset = np.zeros(6)
        var_anchor_offset[0] = 0.1  # base_px -> perturbs the estimated anchor
        pee_offset_anchor = calc_updated_fkm(
            model, data, var_anchor_offset, q, calib_config
        )

        # under the pre-fix bug, bMo was computed and discarded, so this
        # parameter would have had *no* effect on the output at all
        assert not np.allclose(pee_identity_anchor, pee_offset_anchor)


class TestElasticity:
    """non_geom=True: gravity-torque-driven per-joint deflection."""

    def test_single_joint_matches_manual_deflection(self, temp_urdf):
        model = pin.buildModelFromUrdf(temp_urdf)
        data = model.createData()
        j1 = model.getJointId("joint1")

        param_name = ["k_RZ_joint1"]
        compliance = 0.05
        calib_config = _base_calib_config(
            start_frame="base_link",
            end_frame="link1",
            actJoint_idx=[j1],
            measurability=[True] * 6,
            calibration_index=6,
            param_name=param_name,
            non_geom=True,
        )
        q = np.array([[0.3]])

        # must not raise (pre-fix: IndexError from xyz_rpy[elas_id + 3])
        pee = calc_updated_fkm(model, data, np.array([compliance]), q, calib_config)

        # independently compute the expected deflected pose
        tau = pin.computeGeneralizedGravity(model, data, q[0, :])
        tau_j = tau[j1 - 1]
        deflected = model.copy()
        xyz_rpy = np.zeros(6)
        xyz_rpy[5] = compliance * tau_j  # k_RZ -> about joint z, ELAS_TPL index 5
        deflected = apply_joint_offset(deflected, j1, xyz_rpy)
        ddata = deflected.createData()
        pin.framesForwardKinematics(deflected, ddata, q[0, :])
        pin.updateFramePlacements(deflected, ddata)
        expected_T = get_rel_transform(deflected, ddata, "base_link", "link1")
        expected = np.concatenate(
            [expected_T.translation, pin.rpy.matrixToRpy(expected_T.rotation)]
        )

        assert pee == pytest.approx(expected, abs=1e-9)

    def test_multi_joint_deflections_are_independent(self, two_joint_urdf):
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        j1 = model.getJointId("joint1")
        j2 = model.getJointId("joint2")

        param_name = ["k_RZ_joint1", "k_RY_joint2"]
        c1, c2 = 0.05, -0.03
        calib_config = _base_calib_config(
            start_frame="base_link",
            end_frame="link2",
            actJoint_idx=[j1, j2],
            measurability=[True] * 6,
            calibration_index=6,
            param_name=param_name,
            non_geom=True,
        )
        q = np.array([[0.2, -0.4]])

        pee = calc_updated_fkm(model, data, np.array([c1, c2]), q, calib_config)

        tau = pin.computeGeneralizedGravity(model, data, q[0, :])
        deflected = model.copy()
        xyz_rpy1 = np.zeros(6)
        xyz_rpy1[5] = c1 * tau[j1 - 1]  # k_RZ
        deflected = apply_joint_offset(deflected, j1, xyz_rpy1)
        xyz_rpy2 = np.zeros(6)
        xyz_rpy2[4] = c2 * tau[j2 - 1]  # k_RY -> about joint y, ELAS_TPL index 4
        deflected = apply_joint_offset(deflected, j2, xyz_rpy2)
        ddata = deflected.createData()
        pin.framesForwardKinematics(deflected, ddata, q[0, :])
        pin.updateFramePlacements(deflected, ddata)
        expected_T = get_rel_transform(deflected, ddata, "base_link", "link2")
        expected = np.concatenate(
            [expected_T.translation, pin.rpy.matrixToRpy(expected_T.rotation)]
        )

        assert pee == pytest.approx(expected, abs=1e-9)

    def test_many_samples_all_populated(self, two_joint_urdf):
        """Every sample's pose must be written, not just the first few.

        Pre-fix, ``updated_params`` accumulated across the sample loop
        (never reset) while the pose write was gated on
        ``len(updated_params) < len(param_dict)`` -- with only one
        elastic parameter, that gate flips false after the first couple
        of samples and every later PEE row is silently left at zero.

        Uses joint1 (X-axis) with full 6-DOF measurability: joint1's
        placement translation is zero in this fixture, so a rotational
        deflection there only shows up in orientation, not position.
        """
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        j1 = model.getJointId("joint1")

        param_name = ["k_RX_joint1"]
        compliance = 0.05
        n_samples = 10
        calib_config = _base_calib_config(
            start_frame="base_link",
            end_frame="link1",
            actJoint_idx=[j1],
            measurability=[True] * 6,
            calibration_index=6,
            param_name=param_name,
            non_geom=True,
            NbSample=n_samples,
        )
        # q2 must be nonzero: at q2=0 every link's CoM sits on joint1's own
        # rotation axis (X), which is invariant under a rotation about that
        # same axis, making joint1's gravity torque identically zero for
        # any q1 -- a degenerate, not a bug, but not what this test wants.
        q1 = np.linspace(-1.0, 1.0, n_samples)
        q = np.column_stack([q1, np.full(n_samples, 0.4)])

        # PEE is flattened "C" from a (calibration_index, NbSample) array,
        # i.e. DOF-major then sample -- reshape(6, N).T to get (N, 6).
        pee_deflected = (
            calc_updated_fkm(model, data, np.array([compliance]), q, calib_config)
            .reshape(6, n_samples)
            .T
        )
        pee_undeflected = (
            calc_updated_fkm(model, data, np.array([0.0]), q, calib_config)
            .reshape(6, n_samples)
            .T
        )

        # every sample must show the deflection's effect (roll, elas_id=3);
        # under the accumulation bug, later samples silently kept whatever
        # PEE was initialized to (zero) instead of being written at all
        for deflected, undeflected in zip(pee_deflected, pee_undeflected):
            assert deflected[3] != pytest.approx(undeflected[3], abs=1e-9)


TIAGO_OFFSET_JOINTS = [
    ("torso_lift_joint", "offsetPZ"),
    ("arm_1_joint", "offsetRZ"),
    ("arm_2_joint", "offsetRZ"),
    ("arm_3_joint", "offsetRZ"),
    ("arm_4_joint", "offsetRZ"),
    ("arm_5_joint", "offsetRZ"),
    ("arm_6_joint", "offsetRZ"),
    ("arm_7_joint", "offsetRZ"),
]


class TestJointOffset:
    """joint_offset parameters are joint-configuration offsets (#101).

    TIAGo's arm_2..arm_7 placements rotate the joint axis away from the
    parent's z (arm_4 and arm_5 sit at pitch -pi/2), so adding the offset to
    the placement's parent-frame yaw is not ``q + offset``.
    """

    @pytest.mark.parametrize("joint_name, prefix", TIAGO_OFFSET_JOINTS)
    def test_offset_equals_shifted_configuration(self, tiago_model, joint_name, prefix):
        model = tiago_model.copy()
        data = model.createData()
        jid = model.getJointId(joint_name)
        act = [model.getJointId(j) for j, _ in TIAGO_OFFSET_JOINTS]
        offset = 0.05
        rng = np.random.default_rng(0)
        q = np.array([pin.randomConfiguration(model) for _ in range(5)])
        calib_config = _base_calib_config(
            calib_model="joint_offset",
            start_frame="base_link",
            end_frame="arm_tool_link",
            actJoint_idx=act,
            measurability=[True] * 6,
            calibration_index=6,
            NbSample=len(q),
            param_name=[f"{prefix}_{joint_name}"],
        )
        q[:, model.joints[jid].idx_q] = rng.uniform(0.0, 0.2, len(q))

        pee = calc_updated_fkm(model, data, np.array([offset]), q, calib_config)

        q_shift = q.copy()
        q_shift[:, model.joints[jid].idx_q] += offset
        expected = calc_updated_fkm(model, data, np.zeros(1), q_shift, calib_config)
        pee, expected = pee.reshape(6, -1), expected.reshape(6, -1)
        assert pee[:3] == pytest.approx(expected[:3], abs=1e-9)
        rot_err = [
            pin.log3(
                pin.rpy.rpyToMatrix(pee[3:, i]).T @ pin.rpy.rpyToMatrix(expected[3:, i])
            )
            for i in range(len(q))
        ]
        assert np.abs(rot_err).max() < 1e-9

    def test_model_placements_restored_exactly(self, tiago_model):
        model = tiago_model.copy()
        data = model.createData()
        act = [model.getJointId(j) for j, _ in TIAGO_OFFSET_JOINTS]
        before = [model.jointPlacements[j].copy() for j in act]
        names = [f"{p}_{j}" for j, p in TIAGO_OFFSET_JOINTS]
        calib_config = _base_calib_config(
            calib_model="joint_offset",
            start_frame="base_link",
            end_frame="arm_tool_link",
            actJoint_idx=act,
            measurability=[True] * 6,
            calibration_index=6,
            param_name=names,
        )
        q = pin.neutral(model)[None, :]
        calc_updated_fkm(
            model, data, np.linspace(-0.1, 0.1, len(names)), q, calib_config
        )
        for j, placement in zip(act, before):
            assert model.jointPlacements[j].isApprox(placement, 0.0)

    def test_full_params_keep_parent_frame_placement_errors(self, tiago_model):
        """d_phiz is a placement error in the parent frame, not q + offset."""
        model = tiago_model.copy()
        data = model.createData()
        jid = model.getJointId("arm_2_joint")
        q = pin.neutral(model)[None, :]
        calib_config = _base_calib_config(
            calib_model="full_params",
            start_frame="base_link",
            end_frame="arm_tool_link",
            actJoint_idx=[jid],
            measurability=[True] * 6,
            calibration_index=6,
            param_name=["d_phiz_arm_2_joint"],
        )
        pee = calc_updated_fkm(model, data, np.array([0.05]), q, calib_config)

        perturbed = update_joint_placement(
            model.copy(), jid, np.array([0, 0, 0, 0, 0, 0.05])
        )
        pdata = perturbed.createData()
        pin.framesForwardKinematics(perturbed, pdata, q[0])
        expected_T = get_rel_transform(perturbed, pdata, "base_link", "arm_tool_link")
        assert pee[:3] == pytest.approx(expected_T.translation, abs=1e-9)


class TestFullParamsConvention:
    """full_params errors use the kinematic regressor's convention (#110).

    Parameter selection and the base-mapping matrix come from
    ``computeFrameKinematicRegressor(..., LOCAL)``; the fitted FK must move
    the tool exactly along those columns, including where the nominal
    placement is rotated (TIAGo arm_1..arm_7).
    """

    @pytest.mark.parametrize(
        "joint_name", ["arm_1_joint", "arm_2_joint", "arm_4_joint", "arm_5_joint"]
    )
    def test_fk_derivative_matches_regressor_column(self, tiago_model, joint_name):
        model = tiago_model.copy()
        data = model.createData()
        fid = model.getFrameId("arm_tool_link")
        jid = model.getJointId(joint_name)
        h = 1e-7
        for _ in range(5):
            q = np.clip(pin.randomConfiguration(model), -1.0, 1.0)
            pin.framesForwardKinematics(model, data, q)
            R = pin.computeFrameKinematicRegressor(model, data, fid, pin.LOCAL)
            oMf = data.oMf[fid].copy()
            for axis in range(6):
                delta = np.zeros(6)
                delta[axis] = h
                m = update_joint_placement(model.copy(), jid, delta)
                d = m.createData()
                pin.framesForwardKinematics(m, d, q)
                fd = pin.log6(oMf.actInv(d.oMf[fid])).vector / h
                np.testing.assert_allclose(
                    fd, R[:, 6 * (jid - 1) + axis], atol=1e-5, err_msg=str(axis)
                )

    def test_rotation_is_a_rotation_vector_in_the_joint_frame(self, tiago_model):
        model = tiago_model.copy()
        jid = model.getJointId("arm_4_joint")
        M0 = model.jointPlacements[jid].copy()
        xyz_rpy = np.array([0.001, -0.002, 0.003, 0.02, -0.01, 0.03])
        update_joint_placement(model, jid, xyz_rpy)
        expected = M0 * pin.SE3(pin.exp3(xyz_rpy[3:]), xyz_rpy[:3])
        assert model.jointPlacements[jid].isApprox(expected, 1e-12)


class TestMultiMarkerGuard:
    def test_raises_instead_of_silently_falling_back(self, temp_urdf):
        model = pin.buildModelFromUrdf(temp_urdf)
        data = model.createData()
        j1 = model.getJointId("joint1")

        calib_config = _base_calib_config(
            start_frame="base_link",
            end_frame="link1",
            actJoint_idx=[j1],
            measurability=[True, False, False, False, False, False],
            calibration_index=1,
            param_name=["d_px_joint1"],
            NbMarkers=2,
        )
        q = np.zeros((1, model.nq))

        with pytest.raises(NotImplementedError):
            calc_updated_fkm(model, data, np.zeros(1), q, calib_config)


def _two_joint_full_params_config(model, **overrides):
    j1 = model.getJointId("joint1")
    j2 = model.getJointId("joint2")
    config = _base_calib_config(
        start_frame="base_link",
        end_frame="link2",
        actJoint_idx=[j1, j2],
        measurability=[True] * 6,
        calibration_index=6,
        free_flyer=False,
        NbSample=8,
        q0=np.zeros(model.nq),
        config_idx=np.arange(model.nq),
        param_name=[],
    )
    config.update(overrides)
    return config


class TestBaseMappingSideChannel:
    """calculate_base_kinematics_regressor stashes the structural
    base-mapping matrix (phi_base = M @ theta_r) into calib_config so
    callers can redistribute a fitted base-parameter vector back onto the
    full standard-parameter set instead of leaving eliminated members
    implicitly at 0 -- see BaseCalibration.redistribute_parameters and
    qrdecomposition.redistribute_min_norm.

    two_joint_urdf's joint2 (axis Y) sits on a pure X-offset from joint1
    (axis X), with nothing decoupling their X-translation/X-rotation: this
    reliably reduces 12 candidate params (6/joint) to 10 base params,
    eliminating d_px_joint2/d_phix_joint2 as *exact* duplicates of
    d_px_joint1/d_phix_joint1 (coefficient 1.0) -- a small, deterministic,
    real instance of the redistribution problem, not a synthetic stand-in.
    """

    def test_base_mapping_keys_present_with_consistent_shapes(self, two_joint_urdf):
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        calib_config = _two_joint_full_params_config(model)

        calculate_base_kinematics_regressor([], model, data, calib_config)

        M = calib_config["base_mapping_matrix"]
        full_names = calib_config["base_mapping_param_names"]
        row_names = calib_config["base_mapping_row_names"]
        start, end = calib_config["base_mapping_slice"]

        assert M.shape == (len(row_names), len(full_names))
        assert end - start == len(row_names)
        assert calib_config["param_name"][start:end] == row_names
        # This fixture is known to have a genuine reduction (not full rank).
        assert M.shape[0] < M.shape[1]

    def test_redistribution_recovers_the_known_duplicate_pair(self, two_joint_urdf):
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        calib_config = _two_joint_full_params_config(model)

        calculate_base_kinematics_regressor([], model, data, calib_config)
        M = calib_config["base_mapping_matrix"]
        full_names = calib_config["base_mapping_param_names"]

        # d_px_joint2/d_phix_joint2 are eliminated -- absent from param_name,
        # implicitly 0 under today's deploy.
        assert "d_px_joint2" not in calib_config["param_name"]
        assert "d_phix_joint2" not in calib_config["param_name"]

        rng = np.random.default_rng(42)
        phi_base = rng.normal(size=M.shape[0])
        theta_full = redistribute_min_norm(M, phi_base)
        values = dict(zip(full_names, theta_full))

        # Round-trips exactly: same predictions as the base-only fit.
        np.testing.assert_allclose(M @ theta_full, phi_base, rtol=1e-8, atol=1e-10)

        # The known exact duplicates get equal (not one-hot: 100%/0%) credit.
        assert values["d_px_joint2"] != 0.0
        assert values["d_px_joint2"] == pytest.approx(values["d_px_joint1"])
        assert values["d_phix_joint2"] != 0.0
        assert values["d_phix_joint2"] == pytest.approx(values["d_phix_joint1"])
