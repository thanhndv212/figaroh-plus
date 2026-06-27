"""Shared test fixtures and configuration."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"


@pytest.fixture
def temp_urdf():
    """Create a temporary simple URDF file for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>
</robot>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_content)
        temp_path = f.name

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def sample_robot_data():
    """Generate sample robot trajectory data."""
    np.random.seed(42)  # Reproducible tests
    N = 10
    nq, nv = 2, 2

    q = np.random.uniform(-1, 1, (N, nq))
    v = np.random.uniform(-0.5, 0.5, (N, nv))
    a = np.random.uniform(-2, 2, (N, nv))
    tau = np.random.uniform(-10, 10, N * nv)

    return {"q": q, "v": v, "a": a, "tau": tau, "N": N, "nq": nq, "nv": nv}


@pytest.fixture
def regressor_params():
    """Standard regressor parameter configurations."""
    return {
        "basic": {
            "is_joint_torques": True,
            "has_friction": False,
            "has_actuator_inertia": False,
            "has_joint_offset": False,
        },
        "with_friction": {
            "is_joint_torques": True,
            "has_friction": True,
            "has_actuator_inertia": False,
            "has_joint_offset": False,
        },
        "full_features": {
            "is_joint_torques": True,
            "has_friction": True,
            "has_actuator_inertia": True,
            "has_joint_offset": True,
        },
        "external_wrench": {
            "is_external_wrench": True,
            "force_torque": ["Fx", "Fy", "Fz"],
        },
    }


# ── TIAGo test fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tiago_urdf_path() -> Path:
    """Path to the TIAGo URDF file in the test fixtures."""
    path = FIXTURES / "tiago" / "tiago.urdf"
    if not path.exists():
        pytest.skip(f"TIAGo URDF not found: {path}")
    return path


@pytest.fixture(scope="session")
def tiago_model(tiago_urdf_path):
    """Load the TIAGo URDF and return a Pinocchio model."""
    import pinocchio as pin

    try:
        model = pin.buildModelFromUrdf(str(tiago_urdf_path))
    except Exception as e:
        pytest.skip(f"Failed to load TIAGo model: {e}")
    return model


@pytest.fixture(scope="session")
def tiago_data_dir() -> Path:
    """Path to the TIAGo test data directory."""
    path = FIXTURES / "tiago" / "data"
    if not path.exists():
        pytest.skip(f"TIAGo data dir not found: {path}")
    return path


@pytest.fixture(scope="module")
def tiago_eye_hand_csv(tiago_data_dir) -> Path:
    """Path to a reference eye-hand calibration CSV file."""
    csv_path = (
        tiago_data_dir
        / "eye_hand_calibration_recorded_data_48c_hey5_cb_center.csv"
    )
    if not csv_path.exists():
        pytest.skip(f"Eye-hand CSV not found: {csv_path}")
    return csv_path


@pytest.fixture(scope="module")
def tiago_calib_params(tiago_data_dir) -> Path:
    """Path to TIAGo calibration parameters YAML."""
    yaml_path = tiago_data_dir / "calibration_parameters" / "zero_model_tiago.yaml"
    if not yaml_path.exists():
        pytest.skip(f"Calibration params not found: {yaml_path}")
    return yaml_path


@pytest.fixture(scope="module")
def tiago_suspension_vicon(tiago_data_dir) -> Path:
    """Path to a reference Vicon suspension calibration CSV."""
    csv_path = tiago_data_dir / "vicon_calibration_gripper1_base1.csv"
    if not csv_path.exists():
        pytest.skip(f"Vicon CSV not found: {csv_path}")
    return csv_path
