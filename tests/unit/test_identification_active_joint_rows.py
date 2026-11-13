"""Active-joint row selection when the active joints are not a prefix.

The regressor has one block of rows per model joint. When only some joints
are active, and they are not the first ones, every step that pairs the
regressor with the measured effort has to pick the same blocks: the
decimated path, the undecimated path and the reference torque. Synthetic
six-joint arm with joints 0, 1, 3 and 5 active; the others stay at rest.
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

from test_data_contract_wiring import FS, N, _Ident  # noqa: E402

from figaroh.data import JOINT_TORQUE, TrajectoryData  # noqa: E402

ACTIVE = [0, 1, 3, 5]


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


def _names(model, idx):
    return [model.names[j + 1] for j in idx]


def _trajectory(model, active):
    """Exact inverse-dynamics effort of the active joints; the rest at rest."""
    rng = np.random.default_rng(1)
    t = np.arange(N) / FS
    freqs = rng.uniform(0.2, 0.8, (3, len(active)))
    amps = rng.uniform(0.3, 1.0, (3, len(active)))
    w = 2 * np.pi * freqs
    q, dq, ddq = (np.zeros((N, model.nv)) for _ in range(3))
    for a, wk in zip(amps, w):
        q[:, active] += a * np.sin(wk * t[:, None])
        dq[:, active] += a * wk * np.cos(wk * t[:, None])
        ddq[:, active] += -a * wk**2 * np.sin(wk * t[:, None])
    data = model.createData()
    tau = np.stack([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(N)])
    return TrajectoryData(
        t=t,
        joint_names=_names(model, active),
        q=q[:, active],
        dq=dq[:, active],
        ddq=ddq[:, active],
        effort=tau[:, active],
        effort_kind=JOINT_TORQUE,
        effort_unit="N·m",
    )


def _ident(model, active, decimate=None):
    ident = _Ident(model)
    ident.identif_config.update(
        act_idxv=list(active),
        act_idxq=list(active),
        active_joints=_names(model, active),
    )
    ident.trajectory_to_return = _trajectory(model, active)
    ident.initialize()
    if decimate is not None:
        ident.solve(decimate=decimate, decimation_factor=4, plotting=False)
    return ident


def test_row_slices_follow_the_active_joint_indices(model):
    ident = _ident(model, ACTIVE)
    assert ident._active_joint_rows(10) == [
        slice(0, 10),
        slice(10, 20),
        slice(30, 40),
        slice(50, 60),
    ]


def _expected_rows(active):
    """Row indices written out by hand, independent of the library helper."""
    return np.concatenate([np.arange(j * N, (j + 1) * N) for j in active])


def _relative_rms(residual, reference):
    return np.sqrt(np.mean(residual**2) / np.mean(reference**2))


def test_reference_torque_is_the_active_joints_torque(model):
    ident = _ident(model, ACTIVE)
    # inverse dynamics on the processed motion, active joints, joint-major
    data = model.createData()
    q, dq, ddq = (
        ident.processed_data[k] for k in ("positions", "velocities", "accelerations")
    )
    tau = np.stack([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(N)])
    assert ident.tau_ref.shape == (len(ACTIVE) * N,)
    np.testing.assert_allclose(ident.tau_ref, tau[:, ACTIVE].T.flatten(), atol=1e-8)


def test_undecimated_solve_uses_the_active_joint_rows(model):
    ident = _ident(model, ACTIVE, decimate=False)
    tau, regressor = ident.qr_rows
    np.testing.assert_array_equal(tau, ident.processed_data["torques"].T.flatten())
    np.testing.assert_array_equal(
        regressor, ident.regressor_reduced[_expected_rows(ACTIVE)]
    )
    # rows and effort belong to the same joints: the fit explains the effort
    residual = ident.dynamic_regressor_base @ ident.phi_base - tau
    assert _relative_rms(residual, tau) < 1e-2


def test_decimated_and_undecimated_paths_select_the_same_joints(model):
    full = _ident(model, ACTIVE, decimate=False)
    decimated = _ident(model, ACTIVE, decimate=True)
    assert list(full.params_base) == list(decimated.params_base)
    assert decimated.qr_rows[1].shape[0] == len(ACTIVE) * (N // 4)
    # the decimated estimate explains the undecimated rows as well
    tau, _ = full.qr_rows
    residual = full.dynamic_regressor_base @ decimated.phi_base - tau
    assert _relative_rms(residual, tau) < 1e-2


def test_full_active_set_is_unchanged(model):
    every = list(range(model.nv))
    ident = _ident(model, every, decimate=False)
    _, regressor = ident.qr_rows
    np.testing.assert_array_equal(regressor, ident.regressor_reduced)
    np.testing.assert_array_equal(
        ident.tau_ref,
        ident.dynamic_regressor @ np.array(list(ident.standard_parameter.values())),
    )
