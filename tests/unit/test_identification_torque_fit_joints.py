"""Torque rows of untrusted joints left out of the fit (figaroh-examples#68).

``torque_fit_joints`` keeps every active joint's kinematics in the regressor
but fits and scores only the named joints' effort. Synthetic arm with exact
inverse-dynamics torques; the wrist effort is then corrupted, so only the
arm joints' effort is trustworthy.
"""

import os
import sys

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from test_data_contract_wiring import N, _Ident, _trajectory  # noqa: E402

from figaroh.data import TrajectoryData  # noqa: E402
from figaroh.identification.config import _extract_problem_config  # noqa: E402

ARM = ["shoulder1_joint", "shoulder2_joint", "shoulder3_joint", "elbow_joint"]
WRIST = ["wrist1_joint", "wrist2_joint"]


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


def _corrupted_wrist(model):
    """Exact torques, except the wrist effort: wrong scale plus an offset."""
    traj = _trajectory(model)
    effort = np.array(traj.effort)
    effort[:, 4:] = 0.3 * effort[:, 4:] + 0.5
    return TrajectoryData(
        t=traj.t,
        joint_names=traj.joint_names,
        q=traj.q,
        dq=traj.dq,
        ddq=traj.ddq,
        effort=effort,
        effort_kind=traj.effort_kind,
        effort_unit=traj.effort_unit,
    )


def _solve(model, traj, fit_joints=None):
    ident = _Ident(model)
    if fit_joints is not None:
        ident.identif_config["torque_fit_joints"] = fit_joints
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.solve(decimate=False, plotting=False)
    return ident


def test_default_fits_every_active_joint(model):
    traj = _trajectory(model)
    default = _solve(model, traj)
    named = _solve(model, traj, fit_joints=ARM + WRIST)
    assert default.fit_joints() == ARM + WRIST
    np.testing.assert_array_equal(named.phi_base, default.phi_base)
    assert default.qr_rows[0].shape == (6 * N,)


def test_untrusted_effort_is_neither_fitted_nor_scored(model):
    traj = _corrupted_wrist(model)
    fitted_all = _solve(model, traj)
    arm_only = _solve(model, traj, fit_joints=ARM)

    # only the arm's rows enter the regression; the wrist's motion still does
    assert arm_only.qr_rows[0].shape == (4 * N,)
    assert arm_only.qr_rows[1].shape[1] == fitted_all.qr_rows[1].shape[1]

    val_all = fitted_all._compute_validation_metrics()
    val_arm = arm_only._compute_validation_metrics()
    assert list(val_arm["per_joint"]) == ARM
    assert val_arm["joint_names"] == ARM
    assert val_arm["not_fitted_joints"] == WRIST
    assert list(val_arm["tau_measured_per_joint"]) == ARM
    assert "not_fitted_joints" not in val_all

    # the corrupted wrist rows pull the fit off the exact arm torques
    for joint in ARM:
        exact = val_arm["per_joint"][joint]["rmse_identified"]
        polluted = val_all["per_joint"][joint]["rmse_identified"]
        assert exact < 0.1 * polluted

    stats = arm_only._compute_per_joint_stats()
    assert stats["joint_names"] == ARM


def test_unknown_or_empty_fit_joints_are_refused(model):
    ident = _Ident(model)
    ident.identif_config["torque_fit_joints"] = ["shoulder1_joint", "gripper"]
    with pytest.raises(ValueError, match="gripper"):
        ident.fit_joints()
    ident.identif_config["torque_fit_joints"] = []
    with pytest.raises(ValueError, match="empty"):
        ident.fit_joints()


def test_fit_joints_read_from_problem():
    cfg = {}
    _extract_problem_config(cfg, {"torque_fit_joints": ARM})
    assert cfg["torque_fit_joints"] == ARM
    _extract_problem_config(cfg, {})
    assert cfg["torque_fit_joints"] is None
    with pytest.raises(ValueError, match="list"):
        _extract_problem_config(cfg, {"torque_fit_joints": "elbow_joint"})
