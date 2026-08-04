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
    calc_updated_fkm,
    update_joint_placement,
    get_rel_transform,
)


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
        xyz_rpy[5] = compliance * tau_j  # k_RZ -> yaw, ELAS_TPL index 5
        deflected = update_joint_placement(deflected, j1, xyz_rpy)
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
        deflected = update_joint_placement(deflected, j1, xyz_rpy1)
        xyz_rpy2 = np.zeros(6)
        xyz_rpy2[4] = c2 * tau[j2 - 1]  # k_RY -> pitch, ELAS_TPL index 4
        deflected = update_joint_placement(deflected, j2, xyz_rpy2)
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
