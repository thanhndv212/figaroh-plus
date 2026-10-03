"""Resource regressions behind the optimal-trajectory examples.

The TIAGo and UR10 ``optimal_trajectory.py`` examples were killed by memory
exhaustion (a full m x m Q from pivoted QR on a tall regressor) and exceeded
their time budget (Richardson-extrapolated gradients, nv-times redundant RNEA
calls).  These tests pin the cheaper behaviour without allocating anything
large.
"""

from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
from scipy import linalg as scipy_linalg

from figaroh.tools import qrdecomposition
from figaroh.tools.qrdecomposition import QR_pivoting, QRDecomposer, get_baseIndex
from figaroh.utils.cubic_spline import calc_torque


def _rank_deficient_regressor(m=60, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((m, 5))
    # Column 5 = col 0 + 2 * col 1, column 6 = -col 3: rank 5 of 7.
    return np.column_stack([W, W[:, 0] + 2 * W[:, 1], -W[:, 3]])


@pytest.fixture
def qr_modes(monkeypatch):
    """Record the ``mode`` of every pivoted scipy QR call in qrdecomposition."""
    modes = []

    def recording_qr(a, *args, **kwargs):
        if kwargs.get("pivoting"):
            modes.append(kwargs.get("mode", "full"))
        return scipy_linalg.qr(a, *args, **kwargs)

    monkeypatch.setattr(
        qrdecomposition,
        "linalg",
        SimpleNamespace(qr=recording_qr),
    )
    return modes


def test_pivoted_qr_never_builds_full_q(qr_modes):
    W = _rank_deficient_regressor()
    params = [f"p{i}" for i in range(W.shape[1])]
    tau = W @ np.arange(1.0, 8.0)

    get_baseIndex(W, params)
    QR_pivoting(tau, W, params)
    QRDecomposer().decompose(W, params, tau=tau, method="pivoting")
    QRDecomposer().get_base_mapping_matrix_pivoting(W, params)

    assert len(qr_modes) == 4
    assert "full" not in qr_modes


def test_base_index_matches_full_mode_qr():
    W = _rank_deficient_regressor()
    params = [f"p{i}" for i in range(W.shape[1])]

    _, R_full, P_full = scipy_linalg.qr(W, pivoting=True)
    rank = QRDecomposer()._find_rank(R_full)
    expected = tuple(sorted(P_full[:rank].tolist()))

    assert get_baseIndex(W, params) == expected
    assert len(expected) == 5


def test_calc_torque_matches_per_joint_rnea():
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    robot = SimpleNamespace(model=model, data=data)
    rng = np.random.default_rng(1)
    N = 7
    q = np.array([pin.randomConfiguration(model) for _ in range(N)])
    v = rng.standard_normal((N, model.nv))
    a = rng.standard_normal((N, model.nv))

    tau = calc_torque(N, robot, q, v, a)

    expected = np.zeros(model.nv * N)
    for i in range(N):
        tau_i = pin.rnea(model, data, q[i], v[i], a[i])
        for j in range(model.nv):
            expected[j * N + i] = tau_i[j]
    np.testing.assert_allclose(tau, expected)
