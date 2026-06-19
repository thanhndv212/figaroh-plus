"""Test suite for dynamics backends.

Tests cover the factory interface, PinocchioBackend implementation,
numerical correctness against direct Pinocchio calls, kinematics,
optional methods, and interface conformance.
"""

import pytest
import numpy as np
import pinocchio as pin
from figaroh.backends import get_backend, list_backends, get_backend_info
from figaroh.backends.base import DynamicsBackend
from figaroh.backends.pinocchio import PinocchioBackend, PINOCCHIO_AVAILABLE


# ============================================================================
# Tests for the factory interface
# ============================================================================


class TestFactory:
    """Test the backend factory functions."""

    def test_get_backend_pinocchio(self, temp_urdf):
        """get_backend('pinocchio') returns a PinocchioBackend instance."""
        backend = get_backend("pinocchio", model_path=temp_urdf)
        assert isinstance(backend, PinocchioBackend)
        assert isinstance(backend, DynamicsBackend)

    def test_get_backend_invalid(self, temp_urdf):
        """get_backend with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            get_backend("nonexistent", model_path=temp_urdf)

    def test_get_backend_no_model_path(self):
        """get_backend without model_path raises ValueError."""
        with pytest.raises(ValueError, match="model_path is required"):
            get_backend("pinocchio")

    def test_list_backends(self):
        """list_backends returns dict with pinocchio key = True."""
        backends = list_backends()
        assert isinstance(backends, dict)
        assert "pinocchio" in backends
        assert backends["pinocchio"] is True

    def test_get_backend_info_pinocchio(self):
        """get_backend_info('pinocchio') returns correct info dict."""
        info = get_backend_info("pinocchio")
        assert info["name"] == "Pinocchio"
        assert "urdf" in info["formats"]
        assert info["available"] is True

    def test_get_backend_info_invalid(self):
        """get_backend_info with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_info("nonexistent")


# ============================================================================
# Tests for PinocchioBackend initialization
# ============================================================================


class TestPinocchioBackend:
    """Test PinocchioBackend creation and basic properties."""

    def test_init(self, temp_urdf):
        """PinocchioBackend creates instance with correct dimensions."""
        backend = PinocchioBackend(temp_urdf)
        assert backend.nq == 1
        assert backend.nv == 1
        assert backend.model_format == "urdf"
        assert backend.model_path == temp_urdf

    def test_init_invalid_path(self):
        """PinocchioBackend raises RuntimeError for nonexistent file."""
        with pytest.raises(RuntimeError, match="Failed to load model"):
            PinocchioBackend("/nonexistent/path.urdf")

    def test_repr(self, temp_urdf):
        """repr contains class name and model format."""
        backend = PinocchioBackend(temp_urdf)
        rep = repr(backend)
        assert "PinocchioBackend" in rep
        assert "urdf" in rep
        assert "nq=1" in rep
        assert "nv=1" in rep

    def test_context_manager(self, temp_urdf):
        """Context manager works."""
        with PinocchioBackend(temp_urdf) as b:
            assert b.nq == 1
            assert b.nv == 1

    def test_get_model_object(self, temp_urdf):
        """get_model_object returns the pin.Model."""
        backend = PinocchioBackend(temp_urdf)
        model = backend.get_model_object()
        assert isinstance(model, pin.Model)


# ============================================================================
# Numerical correctness: compare PinocchioBackend vs. direct Pinocchio calls
# ============================================================================


class TestPinocchioBackendDynamics:
    """Compare PinocchioBackend dynamics against direct pin.* calls."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_urdf):
        """Create backend and reference model/data."""
        np.random.seed(42)
        self.backend = PinocchioBackend(temp_urdf)
        self.ref_model = pin.buildModelFromUrdf(temp_urdf)
        self.ref_data = self.ref_model.createData()
        # Single-joint robot: nq = nv = 1
        self.q = np.random.uniform(-1.0, 1.0, (self.ref_model.nq,))
        self.v = np.random.uniform(-0.5, 0.5, (self.ref_model.nv,))
        self.a = np.random.uniform(-2.0, 2.0, (self.ref_model.nv,))
        self.tau = np.random.uniform(-5.0, 5.0, (self.ref_model.nv,))

    def test_mass_matrix(self):
        """Mass matrix matches direct pin.crba call."""
        actual = self.backend.compute_mass_matrix(self.q)
        expected = pin.crba(self.ref_model, self.ref_data, self.q)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_coriolis_matrix(self):
        """Coriolis matrix matches direct pin.computeCoriolisMatrix call."""
        actual = self.backend.compute_coriolis_matrix(self.q, self.v)
        expected = pin.computeCoriolisMatrix(
            self.ref_model, self.ref_data, self.q, self.v
        )
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_gravity_vector(self):
        """Gravity vector matches direct pin.computeGeneralizedGravity call."""
        actual = self.backend.compute_gravity_vector(self.q)
        expected = pin.computeGeneralizedGravity(
            self.ref_model, self.ref_data, self.q
        )
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_inverse_dynamics(self):
        """Inverse dynamics matches direct pin.rnea call."""
        actual = self.backend.compute_inverse_dynamics(self.q, self.v, self.a)
        expected = pin.rnea(self.ref_model, self.ref_data, self.q, self.v, self.a)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_forward_dynamics(self):
        """Forward dynamics matches direct pin.aba call."""
        actual = self.backend.compute_forward_dynamics(self.q, self.v, self.tau)
        expected = pin.aba(self.ref_model, self.ref_data, self.q, self.v, self.tau)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_regressor(self):
        """Regressor matches direct pin.computeJointTorqueRegressor call."""
        actual = self.backend.compute_regressor(self.q, self.v, self.a)
        expected = pin.computeJointTorqueRegressor(
            self.ref_model, self.ref_data, self.q, self.v, self.a
        )
        # Ensure expected is 2D (Pinocchio may return 1D for nv=1)
        if expected.ndim == 1:
            expected = expected.reshape(self.ref_model.nv, -1)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        # Check shape: [nv, n_params]
        assert actual.shape[0] == self.ref_model.nv
        assert actual.ndim == 2


# ============================================================================
# Kinematics tests
# ============================================================================


class TestPinocchioBackendKinematics:
    """Test forward kinematics and Jacobian computations."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_urdf):
        """Create backend and reference model/data."""
        np.random.seed(42)
        self.backend = PinocchioBackend(temp_urdf)
        self.ref_model = pin.buildModelFromUrdf(temp_urdf)
        self.ref_data = self.ref_model.createData()
        self.q = np.random.uniform(-1.0, 1.0, (self.ref_model.nq,))

    def test_forward_kinematics(self):
        """Forward kinematics returns dict with required keys per frame."""
        fk = self.backend.compute_forward_kinematics(self.q)
        assert isinstance(fk, dict)
        assert len(fk) > 0
        for frame_name, data in fk.items():
            assert "position" in data
            assert "orientation" in data
            assert "transformation" in data
            assert data["position"].shape == (3,)
            assert data["orientation"].shape == (3, 3)
            assert data["transformation"].shape == (4, 4)

    def test_forward_kinematics_known_frame(self, temp_urdf):
        """Forward kinematics returns correct frame names from URDF."""
        backend = PinocchioBackend(temp_urdf)
        fk = backend.compute_forward_kinematics(np.zeros(backend.nq))
        assert "base_link" in fk
        assert "link1" in fk

    def test_jacobian(self):
        """Jacobian for 'link1' frame returns correct shape and matches direct call."""
        # Build reference via direct Pinocchio (returns [6, nv])
        frame_id = self.ref_model.getFrameId("link1")
        J_ref = pin.computeFrameJacobian(
            self.ref_model, self.ref_data, self.q, frame_id, pin.LOCAL
        )
        J_ref = J_ref.reshape(6, self.ref_model.nv)

        actual = self.backend.compute_jacobian(self.q, "link1")
        assert actual.shape == (6, self.ref_model.nv)
        np.testing.assert_allclose(actual, J_ref, atol=1e-10)

    def test_jacobian_invalid_frame(self):
        """Jacobian raises ValueError for nonexistent frame."""
        with pytest.raises(ValueError, match="not found"):
            self.backend.compute_jacobian(self.q, "nonexistent_frame")


# ============================================================================
# Optional method tests
# ============================================================================


class TestPinocchioBackendOptional:
    """Test the optional interface methods."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_urdf):
        """Create backend and reference model/data."""
        np.random.seed(42)
        self.backend = PinocchioBackend(temp_urdf)
        self.ref_model = pin.buildModelFromUrdf(temp_urdf)
        self.ref_data = self.ref_model.createData()

    def test_get_joint_names(self):
        """get_joint_names returns list including 'joint1'."""
        names = self.backend.get_joint_names()
        assert isinstance(names, list)
        assert "joint1" in names

    def test_get_frame_names(self):
        """get_frame_names returns list of frame names."""
        names = self.backend.get_frame_names()
        assert isinstance(names, list)
        assert "base_link" in names
        assert "link1" in names

    def test_get_inertias(self):
        """get_inertias returns list of inertia objects."""
        inertias = self.backend.get_inertias()
        assert isinstance(inertias, list)
        # Number of inertias equals number of bodies
        assert len(inertias) == self.ref_model.nbodies

    def test_get_frame_id_valid(self):
        """get_frame_id returns int for valid frame names."""
        base_id = self.backend.get_frame_id("base_link")
        link_id = self.backend.get_frame_id("link1")
        assert isinstance(base_id, int)
        assert isinstance(link_id, int)
        assert base_id >= 0
        assert link_id >= 0

    def test_get_frame_id_invalid(self):
        """get_frame_id raises ValueError for invalid frame."""
        with pytest.raises(ValueError, match="not found"):
            self.backend.get_frame_id("nonexistent_frame")

    def test_compute_difference(self):
        """compute_difference matches direct pin.difference call."""
        q1 = np.random.uniform(-1.0, 1.0, (self.ref_model.nq,))
        q2 = np.random.uniform(-1.0, 1.0, (self.ref_model.nq,))
        actual = self.backend.compute_difference(q1, q2)
        expected = pin.difference(self.ref_model, q1, q2)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        assert actual.shape == (self.ref_model.nv,)

    def test_compute_integrate(self):
        """compute_integrate matches direct pin.integrate call."""
        q = np.random.uniform(-1.0, 1.0, (self.ref_model.nq,))
        v = np.random.uniform(-0.5, 0.5, (self.ref_model.nv,))
        actual = self.backend.compute_integrate(q, v)
        expected = pin.integrate(self.ref_model, q, v)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        assert actual.shape == (self.ref_model.nq,)

    def test_random_configuration(self):
        """random_configuration returns array of shape (nq,)."""
        q_rand = self.backend.random_configuration()
        assert isinstance(q_rand, np.ndarray)
        assert q_rand.shape == (self.ref_model.nq,)

    def test_get_model_object(self):
        """get_model_object returns the underlying pin.Model."""
        model_obj = self.backend.get_model_object()
        assert isinstance(model_obj, pin.Model)
        assert model_obj.nq == self.ref_model.nq
        assert model_obj.nv == self.ref_model.nv


# ============================================================================
# Interface conformance
# ============================================================================


class TestInterfaceConformance:
    """Verify PinocchioBackend conforms to the DynamicsBackend interface."""

    def test_is_dynamics_backend(self, temp_urdf):
        """PinocchioBackend is a DynamicsBackend."""
        assert isinstance(PinocchioBackend(temp_urdf), DynamicsBackend)

    def test_abstract_methods_implemented(self, temp_urdf):
        """All abstract methods are implemented (not raising NotImplementedError)."""
        backend = PinocchioBackend(temp_urdf)
        np.random.seed(123)
        q = np.random.uniform(-1.0, 1.0, (backend.nq,))
        v = np.random.uniform(-0.5, 0.5, (backend.nv,))
        a = np.random.uniform(-2.0, 2.0, (backend.nv,))
        tau = np.random.uniform(-5.0, 5.0, (backend.nv,))

        # All abstract methods should execute without raising NotImplementedError
        backend.compute_mass_matrix(q)
        backend.compute_coriolis_matrix(q, v)
        backend.compute_gravity_vector(q)
        backend.compute_forward_kinematics(q)
        backend.compute_jacobian(q, "link1")
        backend.compute_regressor(q, v, a)
        backend.compute_inverse_dynamics(q, v, a)
        backend.compute_forward_dynamics(q, v, tau)
