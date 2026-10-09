"""Tests for URDF exporter — numerical validation + viser visualization.

Uses the :mod:`figaroh.tools.export_validation` API for FK comparison and
viser-based visual overlay.
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
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

        # No joint offset here: the magnitude checks below assume only the
        # d_px/d_phiz placement errors move FK. Joint offsets are checked by
        # test_joint_offset_reloads_as_shifted_configuration.
        self.params = {
            "d_px_joint2": 0.05,
            "d_phiz_joint2": 0.1,
            "m_link1": 2.5,
            "fv_joint1": 0.2,
        }

    def teardown_method(self):
        if hasattr(self, "output") and os.path.exists(self.output):
            os.unlink(self.output)

    # ── Trajectory tracking (via URDFComparison) ──

    def test_trajectory_position_rmse(self):
        """100 random configs → RMSE position error ≈ d_px magnitude."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.fk_consistency_check(n_samples=100)
        # d_px=0.05 → rmse ≈ 0.05
        assert err.rmse_position < 0.15, f"RMSE pos too high: {err.rmse_position}"
        assert err.rmse_position > 0.001, "Changes not reflected in FK"

    def test_trajectory_orientation_rmse(self):
        """100 random configs → RMSE orientation error ≈ d_phiz magnitude."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.fk_consistency_check(n_samples=100)
        assert (
            err.rmse_orientation < 0.15
        ), f"RMSE orient too high: {err.rmse_orientation}"
        assert err.rmse_orientation > 0.001, "Orientation changes not reflected"

    def test_trajectory_max_error_within_bounds(self):
        """Max single-point error does not wildly exceed parameter magnitudes."""
        modified = export_urdf(self.nominal, self.params, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.fk_consistency_check(n_samples=100)
        # d_px=0.05 + d_phiz=0.1 → shouldn't exceed ~2× combined
        assert err.max_position < 0.3, f"Max pos err excessive: {err.max_position}"
        assert (
            err.max_orientation < 0.3
        ), f"Max orient err excessive: {err.max_orientation}"

    def test_zero_params_identity(self):
        """Empty params → exported URDF produces identical FK."""
        modified = export_urdf(self.nominal, {}, output_path=self.output)
        comp = URDFComparison(self.nominal, modified)
        err = comp.fk_consistency_check(n_samples=50)
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
        assert (
            np.linalg.norm(twist0 - twist1) > 1e-6
        ), "Pose delta should change with joint angle"

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


@pytest.mark.parametrize(
    "calibration_type, frame",
    [("mocap", "arm_7_link"), ("eye_hand", "head_2_link"), (None, "<frame_name>")],
)
def test_frame_settings_doc_names_the_ee_parameter(caplog, calibration_type, frame):
    """The customisation hint names a real EE parameter, not a literal "%s"."""
    from figaroh.tools.urdf_exporter import frame_settings_doc

    with caplog.at_level("INFO", logger="figaroh.tools.urdf_exporter"):
        frame_settings_doc(calibration_type=calibration_type, verbose=True)

    assert f"pass e.g. pEEx_{frame} = <value>" in caplog.text
    assert "%s" not in caplog.text


TIAGO_URDF = FIXTURES_DIR / "tiago" / "tiago.urdf"


@pytest.mark.skipif(not TIAGO_URDF.exists(), reason="TIAGo fixture missing")
@pytest.mark.parametrize(
    "param, joint",
    [
        ("offsetRZ_arm_2_joint", "arm_2_joint"),
        ("offsetRZ_arm_4_joint", "arm_4_joint"),  # origin at pitch -pi/2
        ("offsetRZ_arm_5_joint", "arm_5_joint"),  # origin at pitch -pi/2
        ("offsetPZ_torso_lift_joint", "torso_lift_joint"),
    ],
)
def test_joint_offset_reloads_as_shifted_configuration(tmp_path, param, joint):
    """An exported joint offset reloads as ``q + offset`` (#101).

    Pinocchio ignores ``<calibration rising>``, so the offset must be written
    into the joint origin, about the joint's own axis.
    """
    import pinocchio as pin

    offset = 0.05
    out = tmp_path / "tiago_offset.urdf"
    export_urdf(str(TIAGO_URDF), {param: offset}, output_path=str(out))

    nominal = pin.buildModelFromUrdf(str(TIAGO_URDF))
    exported = pin.buildModelFromUrdf(str(out))
    nd, ed = nominal.createData(), exported.createData()
    fid = nominal.getFrameId("arm_tool_link")
    idx_q = nominal.joints[nominal.getJointId(joint)].idx_q
    for _ in range(10):
        q = pin.randomConfiguration(nominal)
        q[idx_q] = np.clip(q[idx_q], -1.0, 0.2)
        pin.framesForwardKinematics(exported, ed, q)
        q_shift = q.copy()
        q_shift[idx_q] += offset
        pin.framesForwardKinematics(nominal, nd, q_shift)
        diff = pin.log6(nd.oMf[fid].inverse() * ed.oMf[fid]).vector
        # exporter writes 6 significant digits
        assert np.abs(diff).max() < 1e-5


@pytest.mark.parametrize(
    "rpy",
    [
        [0.3, -0.4, 1.2],
        [-1.5708, -1.5707963267948966, 0.0],
        [0.0, -1.5707963267948966, 0.0],
        [0.2, 1.5707963267948966, -0.7],
    ],
)
def test_rpy_matrix_round_trip(rpy):
    """URDF rpy helpers invert each other, including at pitch = +-pi/2."""
    from figaroh.tools.urdf_exporter import _matrix_to_rpy, _rpy_to_matrix

    rot = _rpy_to_matrix(rpy)
    assert _rpy_to_matrix(_matrix_to_rpy(rot)) == pytest.approx(rot, abs=1e-12)


def test_full_params_placement_reloads_as_calibrated_model(tmp_path):
    """Exported d_* corrections reload to the calibrated FK (#110).

    The fit applies a joint's six values together in the joint frame
    (``update_joint_placement``); the exporter must write the same origin,
    including at TIAGo's rotated placements (arm_4, arm_5 at pitch -pi/2).
    """
    import pinocchio as pin

    from figaroh.calibration.calibration_tools import update_joint_placement

    corrections = {
        "arm_2_joint": [0.002, -0.001, 0.003, 0.01, -0.02, 0.015],
        "arm_4_joint": [-0.001, 0.002, 0.0, 0.03, 0.01, -0.02],
        "arm_5_joint": [0.0, 0.0, 0.002, -0.015, 0.0, 0.04],
    }
    names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
    params = {
        f"{n}_{joint}": v
        for joint, values in corrections.items()
        for n, v in zip(names, values)
    }
    out = tmp_path / "tiago_full_params.urdf"
    export_urdf(str(TIAGO_URDF), params, output_path=str(out))

    calibrated = pin.buildModelFromUrdf(str(TIAGO_URDF))
    for joint, values in corrections.items():
        update_joint_placement(calibrated, calibrated.getJointId(joint), values)
    exported = pin.buildModelFromUrdf(str(out))
    cd, ed = calibrated.createData(), exported.createData()
    fid = calibrated.getFrameId("arm_tool_link")
    for _ in range(10):
        q = pin.randomConfiguration(calibrated)
        pin.framesForwardKinematics(calibrated, cd, q)
        pin.framesForwardKinematics(exported, ed, q)
        diff = pin.log6(cd.oMf[fid].inverse() * ed.oMf[fid]).vector
        # exporter writes 6 significant digits
        assert np.abs(diff).max() < 1e-5


def test_transmission_joint_is_not_mistaken_for_the_robot_joint(tmp_path):
    """A <transmission> listed first must not receive the correction (#114)."""
    import xml.etree.ElementTree as ET

    urdf = tmp_path / "with_transmission.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="arm">
  <transmission name="t1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="m1"><mechanicalReduction>1</mechanicalReduction></actuator>
  </transmission>
  <link name="base"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""")
    out = export_urdf(
        str(urdf), {"d_px_joint1": 0.01}, output_path=str(tmp_path / "out.urdf")
    )
    root = ET.parse(out).getroot()
    robot_joint = root.find("joint[@name='joint1']")
    assert float(robot_joint.find("origin").get("xyz").split()[0]) == pytest.approx(
        0.11
    )
    assert root.find("transmission/joint/origin") is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"offsetRZ_no_such_joint": 0.01}, "no_such_joint"),
        ({"d_px_no_such_joint": 0.01}, "no_such_joint"),
        ({"m_no_such_link": 1.0}, "no_such_link"),
        ({"Ixx_arm_2_link": 0.1}, "partial inertial set"),
        ({"m_arm_2_link": 1.0, "mx_arm_2_link": 0.1}, "partial inertial set"),
        ({"off_arm_2_joint": 0.1}, "legacy"),
        ({"offsetRZ_arm_2_joint": 0.01, "d_px_arm_2_joint": 0.001}, "both"),
    ],
)
def test_unsupported_mappings_are_rejected(tmp_path, params, match):
    """A correction the exporter cannot write raises and writes nothing,
    instead of a URDF that silently differs from the estimate (#62)."""
    out = tmp_path / "out.urdf"
    with pytest.raises(ValueError, match=match):
        export_urdf(str(TIAGO_URDF), params, output_path=str(out))
    assert not out.exists()


# ── Standard inertial parameters (#60) ──────────────────────────


def _feasible_p10(mass, com, principal, rpy):
    """A physically consistent ``toDynamicParameters()`` vector built from
    mass, centre of mass and a rotated principal tensor, so the fixture does
    not depend on any optimizer."""
    import pinocchio as pin

    from figaroh.tools.urdf_exporter import _rpy_to_matrix

    rot = _rpy_to_matrix(rpy)
    inertia = rot @ np.diag(principal) @ rot.T
    return pin.Inertia(mass, np.asarray(com, float), inertia).toDynamicParameters()


def _inertial_params(target, p10):
    keys = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]
    return {f"{k}_{target}": float(v) for k, v in zip(keys, p10)}


def _assert_reloads_as(urdf, exported_path, inertials):
    """Reloading gives the intended per-joint inertias, physical verdicts
    and inverse dynamics. ``inertials`` maps joint name to p10."""
    import pinocchio as pin

    from figaroh.identification.physical_consistency import check_p10_feasibility

    expected = pin.buildModelFromUrdf(str(urdf))
    for joint, p10 in inertials.items():
        expected.inertias[expected.getJointId(joint)] = (
            pin.Inertia.FromDynamicParameters(np.asarray(p10))
        )
    exported = pin.buildModelFromUrdf(str(exported_path))

    for joint, p10 in inertials.items():
        got = exported.inertias[exported.getJointId(joint)]
        assert got.toDynamicParameters() == pytest.approx(p10, abs=1e-10)
        want = pin.Inertia.FromDynamicParameters(np.asarray(p10))
        assert got.mass == pytest.approx(want.mass, abs=1e-12)
        assert got.lever == pytest.approx(want.lever, abs=1e-10)
        assert got.inertia == pytest.approx(want.inertia, abs=1e-10)
        assert check_p10_feasibility(got.toDynamicParameters()).status == "feasible"

    ed, xd = expected.createData(), exported.createData()
    rng = np.random.default_rng(60)
    for _ in range(20):
        q = pin.randomConfiguration(expected)
        v = rng.standard_normal(expected.nv)
        a = rng.standard_normal(expected.nv)
        tau_e = pin.rnea(expected, ed, q, v, a)
        tau_x = pin.rnea(exported, xd, q, v, a)
        assert tau_x == pytest.approx(tau_e, abs=1e-9)


def test_inertial_set_reloads_with_intended_dynamics(tmp_path):
    """A full set per link, by link or by joint name, reloads as the
    intended inertia and inverse dynamics; the nominal file is untouched."""
    p1 = _feasible_p10(2.5, [0.03, -0.02, 0.45], [0.2, 0.19, 0.03], [0.3, -0.2, 0.5])
    p2 = _feasible_p10(0.8, [-0.01, 0.04, 0.6], [0.05, 0.06, 0.015], [0.0, 0.4, -1.1])
    params = {**_inertial_params("link1", p1), **_inertial_params("joint2", p2)}
    params["fv_joint1"] = 0.2
    nominal_bytes = Path(PENDULUM_URDF).read_bytes()

    out = tmp_path / "pendulum_inertial.urdf"
    export_urdf(PENDULUM_URDF, params, output_path=str(out))

    assert Path(PENDULUM_URDF).read_bytes() == nominal_bytes
    _assert_reloads_as(PENDULUM_URDF, out, {"joint1": p1, "joint2": p2})


def test_inertial_set_keeps_existing_rpy(tmp_path):
    """An existing ``<inertial><origin rpy>`` is kept; the tensor is written
    in its axes, so the reloaded inertia is unchanged by the choice."""
    import xml.etree.ElementTree as ET

    rpy = "0.4 -0.3 1.2"
    nominal = tmp_path / "pendulum_rpy.urdf"
    inertial_origin = 'xyz="0 0 0.5" rpy="{}"/>\n      <inertia'
    nominal.write_text(
        Path(PENDULUM_URDF)
        .read_text()
        .replace(inertial_origin.format("0 0 0"), inertial_origin.format(rpy))
    )
    assert nominal.read_text().count(f'rpy="{rpy}"') == 2
    p2 = _feasible_p10(1.3, [0.02, 0.01, 0.55], [0.09, 0.08, 0.02], [0.2, 0.1, 0.3])

    out = tmp_path / "out.urdf"
    export_urdf(str(nominal), _inertial_params("link2", p2), output_path=str(out))

    origin = ET.parse(out).getroot().find("link[@name='link2']/inertial/origin")
    assert origin.get("rpy") == rpy
    _assert_reloads_as(nominal, out, {"joint2": p2})


@pytest.mark.skipif(not TIAGO_URDF.exists(), reason="TIAGo fixture missing")
def test_inertial_set_on_tiago_joint_target(tmp_path):
    """Identification keys inertials by Pinocchio joint; a joint whose child
    link carries the whole body exports and reloads exactly."""
    p10 = _feasible_p10(1.9, [0.01, -0.03, 0.08], [0.012, 0.01, 0.006], [0.1, 0.2, 0.3])
    out = tmp_path / "tiago_inertial.urdf"
    export_urdf(
        str(TIAGO_URDF), _inertial_params("arm_3_joint", p10), output_path=str(out)
    )
    _assert_reloads_as(TIAGO_URDF, out, {"arm_3_joint": p10})


@pytest.mark.skipif(not TIAGO_URDF.exists(), reason="TIAGo fixture missing")
@pytest.mark.parametrize(
    "target, match",
    [
        ("arm_7_joint", "fixed-attached links"),  # wrist FT sensor and hand
        ("arm_tool_joint", "fixed joint"),
    ],
)
def test_inertial_set_on_merged_body_is_refused(tmp_path, target, match):
    """Pinocchio merges fixed-attached links into a joint's body; that
    estimate does not belong to one URDF link."""
    p10 = _feasible_p10(1.0, [0.0, 0.0, 0.05], [0.01, 0.01, 0.005], [0.0, 0.0, 0.0])
    out = tmp_path / "out.urdf"
    with pytest.raises(ValueError, match=match):
        export_urdf(
            str(TIAGO_URDF), _inertial_params(target, p10), output_path=str(out)
        )
    assert not out.exists()


def test_infeasible_inertial_set_is_refused_unless_allowed(tmp_path, caplog):
    """The physical verdict gates the export by default and is logged."""
    import pinocchio as pin

    p10 = _feasible_p10(1.0, [0.0, 0.0, 0.5], [0.1, 0.1, 0.01], [0.0, 0.0, 0.0])
    p10[9] = 0.5  # Izz beyond Ixx + Iyy about the centre of mass
    params = _inertial_params("link1", p10)
    out = tmp_path / "out.urdf"

    with pytest.raises(ValueError, match="not physically consistent"):
        export_urdf(PENDULUM_URDF, params, output_path=str(out))
    assert not out.exists()

    with caplog.at_level("WARNING", logger="figaroh.tools.urdf_exporter"):
        export_urdf(PENDULUM_URDF, params, output_path=str(out), allow_infeasible=True)
    assert "link1" in caplog.text and "allow_infeasible" in caplog.text
    reloaded = pin.buildModelFromUrdf(str(out)).inertias[1].toDynamicParameters()
    assert reloaded == pytest.approx(p10, abs=1e-10)


def test_psd_eig_tol_admits_a_marginal_inertial_set(tmp_path):
    """A set accepted upstream at a looser tolerance is not refused here."""
    from figaroh.identification.physical_consistency import check_p10_feasibility

    # flat body (pseudo-inertia eigenvalue 0) pushed just past it
    p10 = _feasible_p10(1.0, [0.0, 0.0, 0.0], [0.1, 0.1, 0.2], [0.0, 0.0, 0.0])
    p10[9] += 2e-9
    min_eig = check_p10_feasibility(p10, psd_eig_tol=-1.0).min_eig
    assert -1e-8 < min_eig < -1e-10
    params = _inertial_params("link1", p10)
    out = tmp_path / "out.urdf"

    with pytest.raises(ValueError, match="not physically consistent"):
        export_urdf(PENDULUM_URDF, params, output_path=str(out))
    export_urdf(PENDULUM_URDF, params, output_path=str(out), psd_eig_tol=-1e-8)
    assert out.exists()


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_inertial_set_without_positive_mass_is_refused(tmp_path, mass):
    p10 = np.zeros(10)
    p10[0] = mass
    with pytest.raises(ValueError, match="not positive"):
        export_urdf(
            PENDULUM_URDF,
            _inertial_params("link1", p10),
            output_path=str(tmp_path / "out.urdf"),
        )


def test_mass_only_keeps_urdf_centre_of_mass_and_tensor(tmp_path):
    """``m_`` alone stays a mass override, by link or by joint name."""
    import xml.etree.ElementTree as ET

    out = tmp_path / "out.urdf"
    export_urdf(PENDULUM_URDF, {"m_joint2": 3.0}, output_path=str(out))
    link = ET.parse(out).getroot().find("link[@name='link2']/inertial")
    assert link.find("mass").get("value") == "3"
    assert link.find("origin").get("xyz") == "0 0 0.5"
    assert link.find("inertia").get("izz") == "0.01"


def test_same_link_from_link_and_joint_targets_is_refused(tmp_path):
    with pytest.raises(ValueError, match="several targets"):
        export_urdf(
            PENDULUM_URDF,
            {"m_link2": 1.0, "m_joint2": 2.0},
            output_path=str(tmp_path / "out.urdf"),
        )


# ── merged bodies: merged_bodies="subtract_fixed" (#61) ──


def _body_p10(urdf, joint):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    return np.asarray(model.inertias[model.getJointId(joint)].toDynamicParameters())


def _link_inertial(path, link):
    el = ET.parse(path).getroot().find(f"link[@name='{link}']/inertial")
    return ET.tostring(el)


@pytest.mark.skipif(not TIAGO_URDF.exists(), reason="TIAGo fixture missing")
def test_subtract_fixed_reloads_as_the_identified_body(tmp_path):
    """The whole body of arm_7_joint reloads as the identified set while the
    fixed-attached links keep their CAD inertials."""
    p10 = 1.05 * _body_p10(TIAGO_URDF, "arm_7_joint")
    out = tmp_path / "out.urdf"
    export_urdf(
        str(TIAGO_URDF),
        _inertial_params("arm_7_joint", p10),
        output_path=str(out),
        merged_bodies="subtract_fixed",
    )
    np.testing.assert_allclose(_body_p10(out, "arm_7_joint"), p10, rtol=0, atol=1e-10)
    nominal_doc = ET.parse(TIAGO_URDF).getroot()
    merged = [
        j.find("child").get("link")
        for j in nominal_doc.findall("joint")
        if j.get("type") == "fixed"
        and j.find("child").get("link") != "arm_7_link"
        and j.find("child").get("link")
        in {"wrist_ft_link", "wrist_ft_tool_link", "hand_link"}
    ]
    assert merged  # the wrist sensor and hand are really attached
    for link in merged:
        assert _link_inertial(out, link) == _link_inertial(TIAGO_URDF, link)
    assert _link_inertial(out, "arm_7_link") != _link_inertial(TIAGO_URDF, "arm_7_link")


@pytest.mark.skipif(not TIAGO_URDF.exists(), reason="TIAGo fixture missing")
def test_subtract_fixed_refuses_an_infeasible_remainder(tmp_path):
    # a body lighter than the fixed links it contains leaves a negative child
    p10 = 0.01 * _body_p10(TIAGO_URDF, "arm_7_joint")
    out = tmp_path / "out.urdf"
    with pytest.raises(ValueError, match="not positive|not physically consistent"):
        export_urdf(
            str(TIAGO_URDF),
            _inertial_params("arm_7_joint", p10),
            output_path=str(out),
            merged_bodies="subtract_fixed",
        )
    assert not out.exists()


@pytest.mark.skipif(not TIAGO_URDF.exists(), reason="TIAGo fixture missing")
def test_subtract_fixed_needs_all_ten_parameters(tmp_path):
    with pytest.raises(ValueError, match="lone mass"):
        export_urdf(
            str(TIAGO_URDF),
            {"m_arm_7_joint": 2.0},
            output_path=str(tmp_path / "o.urdf"),
            merged_bodies="subtract_fixed",
        )


def test_subtract_fixed_is_a_noop_for_a_plain_body(tmp_path):
    p10 = _feasible_p10(1.0, [0.0, 0.0, 0.05], [0.01, 0.01, 0.005], [0.0, 0.0, 0.0])
    a, b = tmp_path / "a.urdf", tmp_path / "b.urdf"
    export_urdf(PENDULUM_URDF, _inertial_params("link1", p10), output_path=str(a))
    export_urdf(
        PENDULUM_URDF,
        _inertial_params("link1", p10),
        output_path=str(b),
        merged_bodies="subtract_fixed",
    )
    assert a.read_text() == b.read_text()


def test_unknown_merged_bodies_policy_raises(tmp_path):
    with pytest.raises(ValueError, match="merged_bodies"):
        export_urdf(PENDULUM_URDF, {}, merged_bodies="merge")
