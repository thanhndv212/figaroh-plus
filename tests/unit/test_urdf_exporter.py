"""Tests for URDF exporter — numerical validation + viser visualization.

Uses the :mod:`figaroh.tools.export_validation` API for FK comparison and
viser-based visual overlay.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from figaroh.tools.urdf_exporter import export_urdf
    from figaroh.tools.export_validation import URDFComparison
except ImportError as e:
    if "urdf_exporter" in str(e) or "export_validation" in str(e):
        export_urdf = None  # will trigger skip in setup
        URDFComparison = None
    else:
        raise

# Path to the inline pendulum URDF fixture
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PENDULUM_URDF = str(FIXTURES_DIR / "pendulum.urdf")


# ── XML helpers (for parameter-routing tests only) ──────────────


def _get_joints(doc):
    return doc.findall(".//joint")


def _extract_joint_origin(urdf_path: str, joint_name: str):
    """Return (xyz_str, rpy_str) for a joint's origin element."""
    import xml.etree.ElementTree as ET
    doc = ET.parse(urdf_path)
    for joint in _get_joints(doc):
        name = joint.get("name")
        if name == joint_name:
            origin = joint.find("origin")
            if origin is not None:
                return origin.get("xyz"), origin.get("rpy")
    return None, None


def _extract_link_mass(urdf_path: str, link_name: str):
    """Return mass value string for a link."""
    import xml.etree.ElementTree as ET
    doc = ET.parse(urdf_path)
    for link in doc.findall(".//link"):
        name = link.get("name")
        if name == link_name:
            inertial = link.find("inertial")
            if inertial is not None:
                mass = inertial.find("mass")
                if mass is not None:
                    return mass.get("value")
    return None


def _extract_joint_dynamics(urdf_path: str, joint_name: str):
    """Return (damping_str, friction_str) for a joint."""
    import xml.etree.ElementTree as ET
    doc = ET.parse(urdf_path)
    for joint in _get_joints(doc):
        name = joint.get("name")
        if name == joint_name:
            dyn = joint.find("dynamics")
            if dyn is not None:
                return dyn.get("damping"), dyn.get("friction")
    return None, None


# ── Tests ───────────────────────────────────────────────────────


class TestURDFExporterNumerical:
    """CI-safe FK comparison between original and exported model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if export_urdf is None or URDFComparison is None:
            pytest.skip("figaroh.tools.urdf_exporter not yet implemented")
        self.nominal = PENDULUM_URDF
        self.tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
        self.output = self.tmp.name
        self.tmp.close()

        self.params = {
            "d_px_joint2": 0.05,
            "d_phiz_joint2": 0.1,
            "offsetRX_joint1": 0.25,
            "m_link1": 2.5,
            "fv_joint1": 0.2,
            "base_pz": 0.1,
        }

    def teardown_method(self):
        if hasattr(self, "output") and os.path.exists(self.output):
            os.unlink(self.output)

    # ── Trajectory tracking (via URDFComparison) ──

    def test_trajectory_position_rmse(self):
        """100 random configs → RMSE position error ≈ d_px magnitude."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.trajectory_errors(n_samples=100)
        # d_px=0.05 + base_pz=0.1 → combined rmse ≈ sqrt(0.05²+0.1²) ≈ 0.112
        assert err.rmse_position < 0.15, f"RMSE pos too high: {err.rmse_position}"
        assert err.rmse_position > 0.001, "Changes not reflected in FK"

    def test_trajectory_orientation_rmse(self):
        """100 random configs → RMSE orientation error ≈ d_phiz magnitude."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.trajectory_errors(n_samples=100)
        assert err.rmse_orientation < 0.15, f"RMSE orient too high: {err.rmse_orientation}"
        assert err.rmse_orientation > 0.001, "Orientation changes not reflected"

    def test_trajectory_max_error_within_bounds(self):
        """Max single-point error does not wildly exceed parameter magnitudes."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.trajectory_errors(n_samples=100)
        # Combined base_pz + d_px + d_phiz: shouldn't exceed ~2× combined
        assert err.max_position < 0.3, f"Max pos err excessive: {err.max_position}"
        assert err.max_orientation < 0.3, f"Max orient err excessive: {err.max_orientation}"

    def test_zero_params_identity(self):
        """Empty params → exported URDF produces identical FK."""
        modified = export_urdf(self.nominal, {}, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.trajectory_errors(n_samples=50)
        assert err.rmse_position < 1e-10, f"Identity drift: {err.rmse_position}"
        assert err.rmse_orientation < 1e-10

    def test_default_output_path(self):
        """Omitting output_path writes to <stem>_modified.urdf beside nominal."""
        modified = export_urdf(self.nominal, {"m_link1": 3.0})
        expected_stem = self.nominal.replace(".urdf", "_modified.urdf")
        assert modified == expected_stem or modified.endswith("_modified.urdf")
        assert os.path.exists(modified), f"Modified URDF not found at {modified}"
        os.unlink(modified)

    # ── Static configurations (via URDFComparison) ──

    def test_static_home_config(self):
        """Home config: pose delta magnitude ≈ applied d_px=0.05."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        q_home = np.array([0.0, 0.0])
        poses = comp.static_poses(poses=[q_home])
        pos_mag = poses[0].position_error_mm / 1000  # back to meters
        assert pos_mag > 0.04, f"Expected position delta ~0.05, got {pos_mag}"
        assert pos_mag < 0.5, f"Position delta implausibly large: {pos_mag}"

    def test_static_configs_produce_different_deltas(self):
        """Different configs produce different FK deltas."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        q0 = np.array([0.0, 0.0])
        q1 = np.array([np.pi / 4, np.pi / 3])
        poses = comp.static_poses(poses=[q0, q1])
        twist0 = poses[0].pose_delta.twist
        twist1 = poses[1].pose_delta.twist
        assert np.linalg.norm(twist0 - twist1) > 1e-6, \
            "Pose delta should change with joint angle"

    def test_static_joint_limits(self):
        """Near joint limits: FK still computes without NaN."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        q_limits = np.array([3.14, -3.14])
        poses = comp.static_poses(poses=[q_limits])
        d = poses[0].pose_delta
        assert np.all(np.isfinite(d.translation))
        assert np.all(np.isfinite(d.rotation))

    def test_static_origin_dir(self):
        """Joint origin increments match applied params."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        xyz, rpy = _extract_joint_origin(modified, "joint2")
        orig_xyz, orig_rpy = _extract_joint_origin(self.nominal, "joint2")
        assert xyz is not None and orig_xyz is not None
        x_vals = [float(v) for v in xyz.split()]
        ox_vals = [float(v) for v in orig_xyz.split()]
        assert abs(x_vals[0] - ox_vals[0] - 0.05) < 1e-6
        if rpy and orig_rpy:
            r_vals = [float(v) for v in rpy.split()]
            or_vals = [float(v) for v in orig_rpy.split()]
            assert abs(r_vals[2] - or_vals[2] - 0.1) < 1e-6

    # ── Parameter name routing (XML-level) ──

    def test_additive_params_change_placement(self):
        """d_px params change joint origin, not mass."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        mod_xyz, _ = _extract_joint_origin(modified, "joint2")
        nom_xyz, _ = _extract_joint_origin(self.nominal, "joint2")
        assert mod_xyz != nom_xyz, "Placement XML not changed"

    def test_absolute_params_change_mass(self):
        """m_ params change link mass exactly (not additive)."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        mass = _extract_link_mass(modified, "link1")
        assert mass == "2.5", f"Expected exact mass 2.5, got {mass}"

    def test_absolute_params_preserve_other_links(self):
        """m_link1 override → link2 mass untouched."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        mass2 = _extract_link_mass(modified, "link2")
        orig_mass2 = _extract_link_mass(self.nominal, "link2")
        assert mass2 == orig_mass2, "link2 mass changed when it shouldn't"

    def test_dynamics_absolute_replace(self):
        """fv_ params replace dynamics damping exactly."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        damping, _ = _extract_joint_dynamics(modified, "joint1")
        assert damping == "0.2", f"Expected exact damping 0.2, got {damping}"

    def test_unknown_param_raises(self):
        """Unrecognized parameter → ValueError with helpful message."""
        with pytest.raises(ValueError, match="foobar_joint1"):
            export_urdf(self.nominal, {"foobar_joint1": 1.0}, output_path=self.output)

    def test_mixed_param_types_produce_correct_xml(self):
        """Mixed additive + absolute params produce expected combined result."""
        mixed_params = {
            "d_py_joint2": -0.03,
            "m_link1": 0.5,
            "fv_joint2": 0.08,
        }
        modified = export_urdf(self.nominal, mixed_params, output_path=self.output)
        xyz, _ = _extract_joint_origin(modified, "joint2")
        y = float(xyz.split()[1])
        assert abs(y - (-0.03)) < 1e-6, f"d_py not applied: y={y}"
        mass1 = _extract_link_mass(modified, "link1")
        assert mass1 == "0.5", f"Mass not replaced: {mass1}"
        damp, friction = _extract_joint_dynamics(modified, "joint2")
        assert damp == "0.08", f"Damping not replaced: {damp}"
        orig_damp, orig_friction = _extract_joint_dynamics(self.nominal, "joint2")
        assert friction == orig_friction, "Friction changed when not in params"


class TestURDFExporterVisual:
    """Interactive viser-based overlay visualization. Not for CI.

    Run with:  FIGAROH_TEST_VIZ=1 pytest tests/unit/test_urdf_exporter.py
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        if export_urdf is None or URDFComparison is None:
            pytest.skip("figaroh.tools.urdf_exporter not yet implemented")
        if not os.environ.get("FIGAROH_TEST_VIZ") and "--viz" not in " ".join(sys.argv):
            pytest.skip("Visual test: set FIGAROH_TEST_VIZ=1 or pass --viz")
        self.nominal = PENDULUM_URDF
        self.tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
        self.output = self.tmp.name
        self.tmp.close()
        self.params = {
            "d_px_joint2": 0.05,
            "d_phiz_joint2": 0.1,
            "offsetRX_joint1": 0.25,
            "m_link1": 2.5,
            "fv_joint1": 0.2,
            "base_pz": 0.1,
        }

    def teardown_method(self):
        if hasattr(self, "output") and os.path.exists(self.output):
            os.unlink(self.output)

    def test_overlay_both_models(self):
        """Original (blue) + Modified (red) overlaid via URDFComparison."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        comp.show_overlay(duration=5.0)

    def test_trajectory_animation(self):
        """Animate through configs, tracing EE paths via URDFComparison."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        comp.show_trajectory_animation(n_configs=50, duration=10.0)

    def test_static_config_grid(self):
        """5×4 grid of configs with error labels via URDFComparison."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        comp.show_static_grid(duration=15.0)
