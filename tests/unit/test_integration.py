"""Test suite for the high-level integration API."""

import pytest
import numpy as np
import tempfile
import os

from figaroh.integration import RobotIdentificationSystem, IdentificationResult
from figaroh.integration.api import _SimpleIdentification


# ============================================================================
# Tests for RobotIdentificationSystem
# ============================================================================


class TestRobotIdentificationSystem:
    """Test system creation and basic properties."""

    def test_from_urdf(self, temp_urdf):
        """from_urdf creates instance with correct dimensions."""
        system = RobotIdentificationSystem.from_urdf(temp_urdf)
        assert system.nq == 1
        assert system.nv == 1
        assert system.backend_name == "pinocchio"

    def test_from_urdf_with_backend(self, temp_urdf):
        """from_urdf with explicit backend works."""
        system = RobotIdentificationSystem.from_urdf(
            temp_urdf, backend="pinocchio"
        )
        assert system.backend_name == "pinocchio"

    def test_from_urdf_invalid_path(self):
        """from_urdf with invalid path raises error."""
        with pytest.raises(Exception):
            RobotIdentificationSystem.from_urdf("/nonexistent/robot.urdf")

    def test_repr(self, temp_urdf):
        """repr contains class name and backend."""
        system = RobotIdentificationSystem.from_urdf(temp_urdf)
        rep = repr(system)
        assert "RobotIdentificationSystem" in rep
        assert "pinocchio" in rep

    def test_backend_property(self, temp_urdf):
        """backend property returns a DynamicsBackend."""
        system = RobotIdentificationSystem.from_urdf(temp_urdf)
        backend = system.backend
        from figaroh.backends.base import DynamicsBackend
        assert isinstance(backend, DynamicsBackend)

    def test_nq_nv(self, temp_urdf):
        """nq and nv match robot dimensions."""
        system = RobotIdentificationSystem.from_urdf(temp_urdf)
        assert system.nq == system.robot.model.nq
        assert system.nv == system.robot.model.nv

    def test_from_mjcf_not_implemented(self):
        """from_mjcf raises NotImplementedError with helpful message."""
        with pytest.raises(NotImplementedError, match="from_mjcf|MJCF"):
            RobotIdentificationSystem.from_mjcf("robot.xml")

    def test_init_with_robot(self, temp_urdf):
        """Direct init with a robot object works."""
        from figaroh.tools.robot import Robot
        robot = Robot(temp_urdf, package_dirs=os.path.dirname(temp_urdf))
        system = RobotIdentificationSystem(robot)
        assert system.backend_name == "pinocchio"
        assert system.nq == 1
        assert system.nv == 1


# ============================================================================
# Tests for IdentificationResult
# ============================================================================


class TestIdentificationResult:
    """Test the IdentificationResult dataclass."""

    def test_default_values(self):
        """Default values are None or appropriate defaults."""
        result = IdentificationResult()
        assert result.phi_base is None
        assert result.params_base is None
        assert result.phi_standard is None
        assert result.rms_error is None
        assert result.correlation is None
        assert result.backend == ""
        assert result.model_path == ""
        assert result.config is None
        assert result.raw is None

    def test_with_values(self):
        """All fields are stored correctly."""
        result = IdentificationResult(
            phi_base=np.array([1.0, 2.0]),
            params_base=["m1", "Ixx1"],
            phi_standard=np.array([0.5, 0.3]),
            rms_error=0.05,
            correlation=0.99,
            backend="pinocchio",
            model_path="/path/to/robot.urdf",
            config={"key": "value"},
            raw={"extra": "data"},
        )
        np.testing.assert_array_equal(result.phi_base, [1.0, 2.0])
        assert result.params_base == ["m1", "Ixx1"]
        np.testing.assert_array_equal(result.phi_standard, [0.5, 0.3])
        assert result.rms_error == 0.05
        assert result.correlation == 0.99
        assert result.backend == "pinocchio"
        assert result.model_path == "/path/to/robot.urdf"
        assert result.config == {"key": "value"}
        assert result.raw == {"extra": "data"}


# ============================================================================
# Tests for identify_parameters (data-related errors)
# ============================================================================


class TestIdentifyParameters:
    """Test identify_parameters error cases."""

    def test_identify_no_data_dir(self, temp_urdf):
        """identify_parameters raises ValueError when no data_dir in config."""
        # Create minimal config without data_dir
        config = {"identification": {"ts": 0.01}}
        system = RobotIdentificationSystem.from_urdf(temp_urdf)

        with pytest.raises(ValueError, match="data_dir|No data source"):
            system.identify_parameters(config=config)

    def test_identify_invalid_config_file(self, temp_urdf):
        """identify_parameters with nonexistent config file raises error."""
        system = RobotIdentificationSystem.from_urdf(temp_urdf)
        with pytest.raises(FileNotFoundError):
            system.identify_parameters(config="/nonexistent/config.yaml")

    def _make_identif_config(self, tmpdir_path):
        """Create a minimal identification config dict with proper structure."""
        return {
            "robot_params": [{
                "q_lim_def": [],
                "dq_lim_def": [],
                "fv": [],
                "fs": [],
                "Ia": [],
                "offset": [],
                "Iam6": 0.0,
                "fvm6": 0.0,
                "fsm6": 0.0,
                "reduction_ratio": [],
                "ratio_essential": [],
            }],
            "problem_params": [{
                "is_external_wrench": False,
                "is_joint_torques": True,
                "force_torque": [],
                "external_wrench_offsets": [],
                "has_friction": False,
                "has_actuator_inertia": False,
                "has_joint_offset": False,
                "has_coupled_wrist": False,
            }],
            "processing_params": [{
                "ts": 0.01,
                "cut_off_frequency_butterworth": 10.0,
            }],
            "tls_params": [{
                "mass_load": 0.0,
                "which_body_loaded": 0,
            }],
            "data_dir": tmpdir_path,
        }

    def test_identify_with_data_dir_but_no_files(self, temp_urdf):
        """identify_parameters with data_dir but no CSV files raises error."""
        # Create a temporary directory without data files
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                "identification": self._make_identif_config(tmpdir),
            }

            # Write config to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False
            ) as f:
                import yaml
                yaml.dump(config_data, f)
                config_path = f.name

            try:
                system = RobotIdentificationSystem.from_urdf(temp_urdf)
                with pytest.raises(FileNotFoundError, match="Data file"):
                    system.identify_parameters(config=config_path)
            finally:
                os.unlink(config_path)

    def test_identify_with_data_dir_override(self, temp_urdf):
        """identify_parameters with data_dir override but no files raises error."""
        # Create config without data_dir
        base_cfg = self._make_identif_config(None)
        del base_cfg["data_dir"]
        config_data = {"identification": base_cfg}

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name

        # Create temp data dir without CSV files
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                system = RobotIdentificationSystem.from_urdf(temp_urdf)
                with pytest.raises(FileNotFoundError, match="Data file"):
                    system.identify_parameters(
                        config=config_path,
                        data_dir=tmpdir,
                    )
            finally:
                os.unlink(config_path)


# ============================================================================
# Tests for imports
# ============================================================================


class TestImports:
    """Test that all symbols import correctly."""

    def test_import_all(self):
        """All expected symbols are exported from integration package."""
        from figaroh.integration import RobotIdentificationSystem
        from figaroh.integration import IdentificationResult
        assert RobotIdentificationSystem is not None
        assert IdentificationResult is not None
