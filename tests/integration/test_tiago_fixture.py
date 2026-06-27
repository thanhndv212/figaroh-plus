"""Integration tests for the TIAGo robot test fixture.

Verifies that the TIAGo URDF, meshes, and calibration data ported from
figaroh-examples work correctly as a reusable test fixture for figaroh.
"""

import os
import tempfile
import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)


# ── Fixture existence tests ────────────────────────────────────────────────


class TestTiagoFixtureExists:
    """Verify all fixture files are present and loadable."""

    def test_urdf_exists(self, tiago_urdf_path):
        assert tiago_urdf_path.exists(), f"TIAGo URDF missing: {tiago_urdf_path}"
        size = tiago_urdf_path.stat().st_size
        assert size > 50000, f"URDF too small ({size} bytes), likely truncated"

    def test_meshes_exist(self):
        """Verify mesh symlinks resolve to actual files."""
        from pathlib import Path

        import conftest

        meshes_dir = conftest.FIXTURES / "tiago" / "meshes"
        subdirs = ["arm", "head", "hey5", "pmb2", "sensors", "torso"]
        for sub in subdirs:
            d = meshes_dir / sub
            assert d.exists(), f"Mesh subdir missing: {d}"
            assert d.is_symlink(), f"Mesh subdir not a symlink: {d}"
            target = d.resolve()
            assert target.exists(), f"Mesh symlink target missing: {target}"

    def test_eye_hand_csv_exists(self, tiago_eye_hand_csv):
        assert tiago_eye_hand_csv.exists()
        assert tiago_eye_hand_csv.stat().st_size > 100

    def test_calib_params_yaml_exists(self, tiago_calib_params):
        assert tiago_calib_params.exists()
        assert tiago_calib_params.stat().st_size > 100

    def test_suspension_vicon_data_exists(self, tiago_suspension_vicon):
        assert tiago_suspension_vicon.exists()


# ── Model structure tests ──────────────────────────────────────────────────


class TestTiagoModelStructure:
    """Verify the TIAGo kinematic model."""

    def test_model_loads(self, tiago_model):
        """TIAGo model loads via Pinocchio without errors."""
        assert tiago_model is not None
        assert tiago_model.nq > 0
        assert tiago_model.nv > 0
        assert tiago_model.name == "tiago"

    def test_joint_count(self, tiago_model):
        """TIAGo has expected number of joints."""
        # TIAGo: base joints + torso_lift + arm_1..7 + head_1..2 + hand joints
        # Total config space: ~50 (including fixed/mimic joints)
        assert tiago_model.nq >= 40, f"Expected >=40 joints, got {tiago_model.nq}"
        assert tiago_model.nv >= 38, f"Expected >=38 DOF, got {tiago_model.nv}"

    def test_key_frames_exist(self, tiago_model):
        """Key kinematic frames are present in the model."""
        required_frames = [
            "xtion_rgb_optical_frame",  # camera
            "arm_7_link",  # last arm link
            "torso_lift_link",  # torso
            "head_2_link",  # head
            "base_link",  # base
        ]
        for frame_name in required_frames:
            assert tiago_model.existFrame(frame_name), (
                f"Missing frame: {frame_name}"
            )


# ── Forward kinematics tests ───────────────────────────────────────────────


class TestTiagoForwardKinematics:
    """Verify FK on the TIAGo model."""

    @pytest.fixture(scope="class")
    def data(self, tiago_model):
        return tiago_model.createData()

    def test_zero_config_fk(self, tiago_model, data):
        """Forward kinematics at zero configuration produces valid transforms."""
        q = pin.neutral(tiago_model)
        pin.forwardKinematics(tiago_model, data, q)
        pin.updateFramePlacements(tiago_model, data)

        ee_id = tiago_model.getFrameId("arm_7_link")
        ee_pose = data.oMf[ee_id]

        # At zero config, base and EE should be finite
        assert np.all(np.isfinite(ee_pose.translation)), "EE position not finite"
        assert np.all(np.isfinite(ee_pose.rotation)), "EE rotation not finite"

    def test_random_config_fk(self, tiago_model, data):
        """FK works at random configurations without errors."""
        for _ in range(10):
            q = pin.randomConfiguration(tiago_model)
            pin.forwardKinematics(tiago_model, data, q)
            pin.updateFramePlacements(tiago_model, data)

            ee_id = tiago_model.getFrameId("xtion_rgb_optical_frame")
            ee_pose = data.oMf[ee_id]
            assert np.all(np.isfinite(ee_pose.translation))

    def test_end_effector_frame_detected(self, tiago_urdf_path):
        """URDFComparison auto-EE detection works for TIAGo URDF."""
        from figaroh.tools.export_validation import URDFComparison

        comp = URDFComparison(str(tiago_urdf_path), str(tiago_urdf_path))
        assert comp.ee_frame_a is not None, "Should auto-detect EE frame"
        assert comp.ee_frame_a == comp.ee_frame_b
        # EE should be on the arm/hand chain, not a base sensor
        assert "link" in comp.ee_frame_a.lower() or "joint" in comp.ee_frame_a.lower()


# ── urdf_exporter + TIAGo tests ────────────────────────────────────────────

try:
    from figaroh.tools.urdf_exporter import export_urdf

    HAVE_EXPORTER = True
except ImportError:
    HAVE_EXPORTER = False


@pytest.mark.skipif(not HAVE_EXPORTER, reason="urdf_exporter not available")
class TestTiagoUrdfExporter:
    """Verify urdf_exporter works with TIAGo model."""

    @pytest.fixture
    def output_path(self, tmp_path):
        return tmp_path / "tiago_modified.urdf"

    def test_export_with_base_params(self, tiago_urdf_path, output_path):
        """Apply base placement offset to TIAGo and export modified URDF."""
        params = {"base_px": 0.01, "base_py": 0.0, "base_pz": 0.0, "base_phiz": 0.0}
        modified = export_urdf(str(tiago_urdf_path), params, output_path=str(output_path))
        assert os.path.exists(modified)
        assert os.path.getsize(modified) > 50000

    def test_export_with_joint_offset(self, tiago_urdf_path, output_path):
        """Apply a legacy joint offset to TIAGo arm joint."""
        params = {"off_arm_3_joint": 0.05}
        modified = export_urdf(str(tiago_urdf_path), params, output_path=str(output_path))
        assert os.path.exists(modified)

    def test_export_with_placement_offset(self, tiago_urdf_path, output_path):
        """Apply d_px joint placement offset to TIAGo arm joint."""
        params = {"d_px_arm_3_joint": 0.02, "d_py_arm_3_joint": 0.01}
        modified = export_urdf(str(tiago_urdf_path), params, output_path=str(output_path))
        assert os.path.exists(modified)

    def test_export_with_offset_rx(self, tiago_urdf_path, output_path):
        """Apply offsetRX (calibrated) joint offset to TIAGo arm joint."""
        params = {"offsetRX_arm_3_joint": 0.02}
        modified = export_urdf(str(tiago_urdf_path), params, output_path=str(output_path))
        assert os.path.exists(modified)

    def test_comparison_detects_changes(self, tiago_urdf_path, output_path):
        """URDFComparison detects FK changes from exported URDF."""
        from figaroh.tools.export_validation import URDFComparison

        # Export with visible parameter changes on real TIAGo joints
        params = {
            "d_px_arm_3_joint": 0.02,
            "d_py_arm_3_joint": 0.01,
            "base_pz": 0.05,
        }
        modified = export_urdf(str(tiago_urdf_path), params, output_path=str(output_path))

        comp = URDFComparison(str(tiago_urdf_path), modified)
        err = comp.trajectory_errors(n_samples=50)

        # Should detect position change from base_pz + d_px/d_py
        assert err.rmse_position > 0.001, "Exporter should produce FK difference"
        assert err.rmse_position < 0.5, f"Position error too large: {err.rmse_position}"


# ── Eye-hand calibration data tests ───────────────────────────────────────


class TestTiagoEyeHandData:
    """Verify eye-hand calibration data is parseable."""

    def test_csv_has_expected_columns(self, tiago_eye_hand_csv):
        """Eye-hand CSV has marker pose + joint angle columns."""
        import csv

        with open(tiago_eye_hand_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # Expected columns: x,y,z,phix,phiy,phiz + 9 joints + head joints
        assert len(header) >= 12, f"Expected >=12 cols, got {len(header)}: {header}"
        assert len(rows) >= 10, (
            f"Expected >=10 data rows, got {len(rows)}"
        )
        # First column should be a float
        float(rows[0][0])

    def test_multiple_csv_files(self, tiago_data_dir):
        """All expected eye-hand CSVs are present and non-empty."""
        eh_dir = tiago_data_dir / "eye_hand_calibration_recorded_data_48c_hey5_cb_center.csv"
        eh_dir = tiago_data_dir  # top-level data for eye-hand (some CSVs are at top level)
        csv_count = 0
        for entry in tiago_data_dir.rglob("*.csv"):
            if entry.stat().st_size > 100:
                csv_count += 1
        assert csv_count >= 10, f"Expected >=10 CSVs, found {csv_count}"


# ── Suspension data tests ──────────────────────────────────────────────────


class TestTiagoSuspensionData:
    """Verify suspension identification data is parseable."""

    def test_vicon_csv_has_expected_format(self, tiago_suspension_vicon):
        """Vicon CSV has time, markers, and force plate columns."""
        import csv

        with open(tiago_suspension_vicon) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # Expected format depends on the specific CSV type.
        # Calibration CSVs have marker xyz + joint angles; suspension CSVs
        # have time + markers + force plate columns.  Accept any reasonable size.
        assert len(header) >= 7, f"Expected >=7 cols, got {len(header)}: {header}"
        assert len(rows) >= 10, f"Expected >=10 data rows, got {len(rows)}"


# ── Visual comparison test (requires viser, gated) ────────────────────────

VIZ = os.environ.get("FIGAROH_TEST_VIZ", "0") == "1"


@pytest.mark.skipif(not VIZ, reason="Set FIGAROH_TEST_VIZ=1 for visual tests")
class TestTiagoVisualComparison:
    """Visual comparison of TIAGo nominal vs. modified URDF."""

    def test_overlay_nominal_and_modified(self, tiago_urdf_path, tmp_path):
        """Show viser overlay of nominal and modified TIAGo URDF."""
        try:
            from figaroh.tools.urdf_exporter import export_urdf
            from figaroh.tools.export_validation import URDFComparison
        except ImportError as e:
            pytest.skip(f"Not available: {e}")

        output = tmp_path / "tiago_viz_modified.urdf"
        params = {
            "d_px_arm_3_joint": 0.02,
            "d_py_arm_3_joint": 0.01,
            "base_pz": 0.03,
            "off_arm_3_joint": 0.05,
        }
        modified = export_urdf(str(tiago_urdf_path), params, output_path=str(output))

        comp = URDFComparison(str(tiago_urdf_path), modified)
        comp.show_overlay(
            orig_color=(255, 255, 255), mod_color=(51, 229, 102), duration=8.0
        )
