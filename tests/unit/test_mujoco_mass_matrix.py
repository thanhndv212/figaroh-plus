"""Mass-matrix regression across the pre/post-3.10 MuJoCo APIs."""

import numpy as np
import pytest

from figaroh.backends.mujoco import MUJOCO_AVAILABLE, MuJoCoBackend
from figaroh.backends.pinocchio import PinocchioBackend


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
def test_mass_matrix_matches_pinocchio_and_returns_independent_results(two_joint_urdf):
    """A changed configuration must neither crash nor overwrite an earlier result."""
    backend = MuJoCoBackend(two_joint_urdf)
    reference = PinocchioBackend(two_joint_urdf)
    q_first = np.array([0.2, -0.4])
    first = backend.compute_mass_matrix(q_first)
    expected_first = reference.compute_mass_matrix(q_first).copy()
    np.testing.assert_allclose(first, expected_first, rtol=1e-10, atol=1e-12)

    q_second = np.array([-0.8, 0.7])
    second = backend.compute_mass_matrix(q_second)
    np.testing.assert_allclose(
        second, reference.compute_mass_matrix(q_second), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(first, expected_first, rtol=1e-10, atol=1e-12)
    assert not np.shares_memory(first, second)
    assert np.all(np.linalg.eigvalsh(second) > 0)
