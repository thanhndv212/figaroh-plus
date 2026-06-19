"""Cross-backend validation suite.

Verifies that PinocchioBackend and MuJoCoBackend produce consistent
dynamics computation results on the same robot model, and benchmarks
relative performance.
"""

import pytest
import numpy as np
import time

pytest.importorskip("mujoco")

from figaroh.backends.pinocchio import PinocchioBackend
from figaroh.backends.mujoco import MuJoCoBackend
from figaroh.backends import list_backends


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def backends(temp_urdf):
    """Create both backends from the same URDF."""
    pin_backend = PinocchioBackend(temp_urdf)
    mj_backend = MuJoCoBackend(temp_urdf)
    return pin_backend, mj_backend


@pytest.fixture
def random_state(backends):
    """Generate random q, v, a for testing."""
    pin_backend, _ = backends
    np.random.seed(42)
    q = np.random.uniform(-1.0, 1.0, pin_backend.nq)
    v = np.random.uniform(-0.5, 0.5, pin_backend.nv)
    a = np.random.uniform(-2.0, 2.0, pin_backend.nv)
    tau = np.random.uniform(-5.0, 5.0, pin_backend.nv)
    return q, v, a, tau


# ============================================================================
# Numerical consistency tests
# ============================================================================


class TestCrossBackendDynamics:
    """Compare dynamics computations across backends."""

    def test_nq_nv_match(self, backends):
        """Both backends have the same nq and nv."""
        pin_backend, mj_backend = backends
        assert pin_backend.nq == mj_backend.nq
        assert pin_backend.nv == mj_backend.nv

    def test_mass_matrix_consistency(self, backends, random_state):
        """Mass matrices match across backends."""
        pin_backend, mj_backend = backends
        q, _, _, _ = random_state

        M_pin = pin_backend.compute_mass_matrix(q)
        M_mj = mj_backend.compute_mass_matrix(q)

        assert M_pin.shape == M_mj.shape
        np.testing.assert_allclose(M_pin, M_mj, atol=1e-10)

    def test_gravity_vector_consistency(self, backends, random_state):
        """Gravity vectors match across backends."""
        pin_backend, mj_backend = backends
        q, _, _, _ = random_state

        g_pin = pin_backend.compute_gravity_vector(q)
        g_mj = mj_backend.compute_gravity_vector(q)

        assert g_pin.shape == g_mj.shape
        np.testing.assert_allclose(g_pin, g_mj, atol=1e-10)

    def test_inverse_dynamics_consistency(self, backends, random_state):
        """Inverse dynamics torques match across backends."""
        pin_backend, mj_backend = backends
        q, v, a, _ = random_state

        tau_pin = pin_backend.compute_inverse_dynamics(q, v, a)
        tau_mj = mj_backend.compute_inverse_dynamics(q, v, a)

        assert tau_pin.shape == tau_mj.shape
        np.testing.assert_allclose(tau_pin, tau_mj, atol=1e-10)

    def test_forward_dynamics_consistency(self, backends, random_state):
        """Forward dynamics accelerations match across backends."""
        pin_backend, mj_backend = backends
        q, v, _, tau = random_state

        a_pin = pin_backend.compute_forward_dynamics(q, v, tau)
        a_mj = mj_backend.compute_forward_dynamics(q, v, tau)

        assert a_pin.shape == a_mj.shape
        np.testing.assert_allclose(a_pin, a_mj, atol=1e-10)

    def test_regressor_consistency(self, backends, random_state):
        """Regressor matrices match exactly across backends.

        MuJoCoBackend delegates to Pinocchio's analytical regressor,
        so results should be exactly identical.
        """
        pin_backend, mj_backend = backends
        q, v, a, _ = random_state

        W_pin = pin_backend.compute_regressor(q, v, a)
        W_mj = mj_backend.compute_regressor(q, v, a)

        assert W_pin.shape == W_mj.shape
        np.testing.assert_allclose(W_pin, W_mj, atol=1e-10)

    def test_forward_kinematics_structure(self, backends, random_state):
        """Forward kinematics returns comparable results."""
        pin_backend, mj_backend = backends
        q, _, _, _ = random_state

        fk_pin = pin_backend.compute_forward_kinematics(q)
        fk_mj = mj_backend.compute_forward_kinematics(q)

        # MuJoCo absorbs base_link into world, so frame naming may differ.
        # Check that both return dicts with the expected keys per frame.
        assert isinstance(fk_pin, dict)
        assert isinstance(fk_mj, dict)
        assert len(fk_pin) > 0
        assert len(fk_mj) > 0

        # Both should have link1
        assert "link1" in fk_mj

        # Check structure matches
        for name in fk_mj:
            for key in ("position", "orientation", "transformation"):
                assert key in fk_mj[name], f"{name} missing {key}"

    def test_jacobian_consistency(self, backends, random_state):
        """Jacobian shapes match across backends."""
        pin_backend, mj_backend = backends
        q, _, _, _ = random_state

        # Both backends should have a link1 frame
        J_pin = pin_backend.compute_jacobian(q, "link1")
        J_mj = mj_backend.compute_jacobian(q, "link1")

        assert J_pin.shape == J_mj.shape
        assert J_pin.shape == (6, pin_backend.nv)


# ============================================================================
# End-to-end identification consistency
# ============================================================================


class TestCrossBackendIdentification:
    """Verify identification-consistent results across backends."""

    def test_regressor_produces_same_identification(self, backends, random_state):
        """Both backends produce regressors yielding the same parameters."""
        pin_backend, mj_backend = backends
        q, v, a, _ = random_state

        # Compute regressors
        W_pin = pin_backend.compute_regressor(q, v, a)
        W_mj = mj_backend.compute_regressor(q, v, a)

        # Should be identical (MuJoCo delegates to Pinocchio)
        np.testing.assert_allclose(W_pin, W_mj, atol=1e-10)

        # Compute torques from both backends
        tau_pin = pin_backend.compute_inverse_dynamics(q, v, a)
        tau_mj = mj_backend.compute_inverse_dynamics(q, v, a)

        # Torques should match
        np.testing.assert_allclose(tau_pin, tau_mj, atol=1e-10)

        # Solve for parameters: theta = pinv(W) @ tau
        theta_pin = np.linalg.lstsq(W_pin, tau_pin, rcond=None)[0]
        theta_mj = np.linalg.lstsq(W_mj, tau_mj, rcond=None)[0]

        # Parameters should be identical
        np.testing.assert_allclose(theta_pin, theta_mj, atol=1e-10)


# ============================================================================
# Performance benchmarks
# ============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmarks comparing Pinocchio and MuJoCo backends."""

    @pytest.fixture
    def perf_backends(self, temp_urdf):
        """Create backends for performance testing."""
        return PinocchioBackend(temp_urdf), MuJoCoBackend(temp_urdf)

    def test_mass_matrix_performance(self, perf_backends):
        """Benchmark mass matrix computation."""
        pin_backend, mj_backend = perf_backends
        q = np.zeros(pin_backend.nq)
        N = 1000

        # Warm up
        for _ in range(10):
            pin_backend.compute_mass_matrix(q)
            mj_backend.compute_mass_matrix(q)

        # Benchmark Pinocchio
        start = time.perf_counter()
        for _ in range(N):
            pin_backend.compute_mass_matrix(q)
        pin_time = time.perf_counter() - start

        # Benchmark MuJoCo
        start = time.perf_counter()
        for _ in range(N):
            mj_backend.compute_mass_matrix(q)
        mj_time = time.perf_counter() - start

        print(f"\nMass matrix ({N} iterations):")
        print(f"  Pinocchio: {pin_time*1000:.1f}ms ({pin_time/N*1000:.3f}ms/call)")
        print(f"  MuJoCo:    {mj_time*1000:.1f}ms ({mj_time/N*1000:.3f}ms/call)")
        print(f"  Ratio:     {mj_time/pin_time:.2f}x")

        # No assertion on speed — just report
        assert True

    def test_gravity_performance(self, perf_backends):
        """Benchmark gravity vector computation."""
        pin_backend, mj_backend = perf_backends
        q = np.zeros(pin_backend.nq)
        N = 1000

        for _ in range(10):
            pin_backend.compute_gravity_vector(q)
            mj_backend.compute_gravity_vector(q)

        start = time.perf_counter()
        for _ in range(N):
            pin_backend.compute_gravity_vector(q)
        pin_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(N):
            mj_backend.compute_gravity_vector(q)
        mj_time = time.perf_counter() - start

        print(f"\nGravity vector ({N} iterations):")
        print(f"  Pinocchio: {pin_time*1000:.1f}ms ({pin_time/N*1000:.3f}ms/call)")
        print(f"  MuJoCo:    {mj_time*1000:.1f}ms ({mj_time/N*1000:.3f}ms/call)")
        print(f"  Ratio:     {mj_time/pin_time:.2f}x")

    def test_inverse_dynamics_performance(self, perf_backends):
        """Benchmark inverse dynamics computation."""
        pin_backend, mj_backend = perf_backends
        np.random.seed(42)
        q = np.random.uniform(-1, 1, pin_backend.nq)
        v = np.random.uniform(-0.5, 0.5, pin_backend.nv)
        a = np.random.uniform(-2, 2, pin_backend.nv)
        N = 1000

        for _ in range(10):
            pin_backend.compute_inverse_dynamics(q, v, a)
            mj_backend.compute_inverse_dynamics(q, v, a)

        start = time.perf_counter()
        for _ in range(N):
            pin_backend.compute_inverse_dynamics(q, v, a)
        pin_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(N):
            mj_backend.compute_inverse_dynamics(q, v, a)
        mj_time = time.perf_counter() - start

        print(f"\nInverse dynamics ({N} iterations):")
        print(f"  Pinocchio: {pin_time*1000:.1f}ms ({pin_time/N*1000:.3f}ms/call)")
        print(f"  MuJoCo:    {mj_time*1000:.1f}ms ({mj_time/N*1000:.3f}ms/call)")
        print(f"  Ratio:     {mj_time/pin_time:.2f}x")

    def test_regressor_performance(self, perf_backends):
        """Benchmark regressor computation."""
        pin_backend, mj_backend = perf_backends
        np.random.seed(42)
        q = np.random.uniform(-1, 1, pin_backend.nq)
        v = np.random.uniform(-0.5, 0.5, pin_backend.nv)
        a = np.random.uniform(-2, 2, pin_backend.nv)
        N = 100

        for _ in range(5):
            pin_backend.compute_regressor(q, v, a)
            mj_backend.compute_regressor(q, v, a)

        start = time.perf_counter()
        for _ in range(N):
            pin_backend.compute_regressor(q, v, a)
        pin_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(N):
            mj_backend.compute_regressor(q, v, a)
        mj_time = time.perf_counter() - start

        print(f"\nRegressor ({N} iterations):")
        print(f"  Pinocchio: {pin_time*1000:.1f}ms ({pin_time/N*1000:.3f}ms/call)")
        print(f"  MuJoCo:    {mj_time*1000:.1f}ms ({mj_time/N*1000:.3f}ms/call)")
        print(f"  Ratio:     {mj_time/pin_time:.2f}x")


# ============================================================================
# Backend availability tests
# ============================================================================


class TestBackendAvailability:
    """Verify backend availability reporting."""

    def test_list_backends_correct(self):
        """list_backends returns correct availability status."""
        available = list_backends()
        assert available["pinocchio"] is True
        assert available["mujoco"] is True
        assert available["genesis"] is False
        assert available["isaacsim"] is False
