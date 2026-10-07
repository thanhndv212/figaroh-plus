"""Tests for figaroh.tools.geometric_calibration_export.

Builds a minimal BaseCalibration stand-in (BaseCalibration.__new__,
bypassing __init__'s robot/config-file requirements — same pattern as
test_base_calibration_redistribution.py) with a fixed
redistribute_parameters() output, since this module only consumes that
method's return value.
"""

import numpy as np
import pinocchio as pin
import pytest
import yaml

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.tools.geometric_calibration_export import (
    _pal_joint_name,
    build_geometric_calibration,
    export_geometric_calibration_yaml,
)


def _bare_calibration(redistributed, calib_config):
    calib = BaseCalibration.__new__(BaseCalibration)
    calib.redistribute_parameters = lambda: redistributed
    calib.calib_config = calib_config
    return calib


class TestPalJointName:
    def test_strips_joint_suffix(self):
        assert _pal_joint_name("arm_right_2_joint") == "arm_right_2"

    def test_passes_through_without_suffix(self):
        assert _pal_joint_name("arm_right_2") == "arm_right_2"

    def test_only_strips_trailing_suffix(self):
        assert _pal_joint_name("joint_arm_1_joint") == "joint_arm_1"


class TestBuildGeometricCalibration:
    def test_maps_axes_to_pal_suffixes(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
            "d_py_arm_1_joint": {"value": 0.002, "std_dev": 0.0001},
            "d_pz_arm_1_joint": {"value": 0.003, "std_dev": 0.0001},
            "d_phix_arm_1_joint": {"value": 0.01, "std_dev": 0.001},
            "d_phiy_arm_1_joint": {"value": 0.02, "std_dev": 0.001},
            "d_phiz_arm_1_joint": {"value": 0.03, "std_dev": 0.001},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        # identity placement: translation passes through, the rotation
        # vector is written as the RPY of exp3(d_phi) (#110)
        rpy = pin.rpy.matrixToRpy(pin.exp3(np.array([0.01, 0.02, 0.03])))
        assert gc == pytest.approx(
            {
                "arm_1_dx": 0.001,
                "arm_1_dy": 0.002,
                "arm_1_dz": 0.003,
                "arm_1_droll": rpy[0],
                "arm_1_dpitch": rpy[1],
                "arm_1_dyaw": rpy[2],
            },
            abs=1e-12,
        )

    def test_rotated_placement_converts_to_origin_deltas(self):
        """d_* act in the joint frame; PAL keys are origin xyz/rpy deltas."""
        model = pin.Model()
        placement = pin.SE3(pin.rpy.rpyToMatrix(0.0, -np.pi / 2, 0.3), np.zeros(3))
        model.addJoint(0, pin.JointModelRZ(), placement, "arm_4_joint")
        redistributed = {
            "d_px_arm_4_joint": {"value": 0.002, "std_dev": 1e-4},
            "d_phiz_arm_4_joint": {"value": 0.01, "std_dev": 1e-3},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        calib.model = model

        gc = build_geometric_calibration(calib)["robot_state_publisher"][
            "geometric_calibration"
        ]

        corrected = placement * pin.SE3(
            pin.exp3(np.array([0, 0, 0.01])), np.array([0.002, 0, 0])
        )
        xyz = np.array([gc.get(f"arm_4_{k}", 0.0) for k in ("dx", "dy", "dz")])
        rpy0 = pin.rpy.matrixToRpy(placement.rotation)
        rpy = rpy0 + [gc.get(f"arm_4_{k}", 0.0) for k in ("droll", "dpitch", "dyaw")]
        np.testing.assert_allclose(xyz, corrected.translation, atol=1e-12)
        np.testing.assert_allclose(
            pin.rpy.rpyToMatrix(rpy), corrected.rotation, atol=1e-9
        )
        # the joint-frame x translation lands on the parent's z axis
        assert abs(gc["arm_4_dz"]) == pytest.approx(0.002)

    def test_exports_kinematic_corrections_only(self):
        """d_* placements and joint offsets (calib_model='joint_offset',
        #123) are exported; elasticity and unknown names are not."""
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
            "offsetRX_arm_2_joint": {"value": 0.5, "std_dev": 0.01},
            "k_RZ_arm_3_joint": {"value": 0.05, "std_dev": 0.001},
            "not_a_known_param": {"value": 1.0, "std_dev": 0.1},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == pytest.approx({"arm_1_dx": 0.001, "arm_2_droll": 0.5})

    def test_keeps_lifted_first_joint_when_baseframe_unknown(self):
        """The lift (redistribute_parameters, #111) holds the base-frame rows
        at 0, so first-joint values it returns are joint corrections, not
        base-frame quantities: nothing is excluded on that basis. (Before
        #111 the first six base rows were dropped here instead.)"""
        redistributed = {
            "d_px_torso_lift_joint": {"value": 0.01, "std_dev": 0.001},
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
        }
        calib_config = {
            "known_baseframe": False,
            "base_mapping_row_names": ["d_px_torso_lift_joint", "d_px_arm_1_joint"],
        }
        calib = _bare_calibration(redistributed, calib_config)

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"torso_lift_dx": 0.01, "arm_1_dx": 0.001}

    def test_includes_first_joint_when_baseframe_known(self):
        """known_baseframe=True (or absent/default) means there's no
        co-estimated base transform to merge with -- nothing should be
        excluded on that basis."""
        redistributed = {
            "d_px_torso_lift_joint": {"value": 0.01, "std_dev": 0.001},
        }
        calib_config = {
            "known_baseframe": True,
            "base_mapping_row_names": ["d_px_torso_lift_joint"],
        }
        calib = _bare_calibration(redistributed, calib_config)

        result = build_geometric_calibration(calib)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"torso_lift_dx": 0.01}

    def test_min_sigma_filters_low_confidence_parameters(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.01, "std_dev": 0.001},  # 10 sigma
            "d_py_arm_1_joint": {"value": 0.001, "std_dev": 0.002},  # 0.5 sigma
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib, min_sigma=2.0)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.01}

    def test_min_sigma_none_includes_everything(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 10.0},  # tiny sigma
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib, min_sigma=None)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.001}

    def test_min_sigma_handles_zero_std_dev(self):
        """A parameter with exactly-zero std_dev is treated as maximally
        confident (sigma=inf), not a divide-by-zero crash."""
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})

        result = build_geometric_calibration(calib, min_sigma=100.0)
        gc = result["robot_state_publisher"]["geometric_calibration"]

        assert gc == {"arm_1_dx": 0.001}


class TestExportGeometricCalibrationYaml:
    def test_writes_matching_pal_structure(self, tmp_path):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        out_file = tmp_path / "master_calibration.yaml"

        returned_path = export_geometric_calibration_yaml(calib, str(out_file))

        assert returned_path == str(out_file)
        assert out_file.exists()
        loaded = yaml.safe_load(out_file.read_text())
        assert loaded == build_geometric_calibration(calib)
        assert (
            loaded["robot_state_publisher"]["geometric_calibration"]["arm_1_dx"]
            == 0.001
        )

    def test_writes_header_comment(self, tmp_path):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.001, "std_dev": 0.0001},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        out_file = tmp_path / "master_calibration.yaml"

        export_geometric_calibration_yaml(
            calib, str(out_file), header_comment="48 samples, RMSE 8.99mm"
        )

        text = out_file.read_text()
        assert text.startswith("# 48 samples, RMSE 8.99mm\n")
        # Still valid YAML despite the leading comment.
        assert yaml.safe_load(text) is not None

    def test_forwards_min_sigma(self, tmp_path):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.01, "std_dev": 0.001},
            "d_py_arm_1_joint": {"value": 0.001, "std_dev": 0.002},
        }
        calib = _bare_calibration(redistributed, {"known_baseframe": True})
        out_file = tmp_path / "master_calibration_conservative.yaml"

        export_geometric_calibration_yaml(calib, str(out_file), min_sigma=2.0)

        loaded = yaml.safe_load(out_file.read_text())
        gc = loaded["robot_state_publisher"]["geometric_calibration"]
        assert gc == {"arm_1_dx": 0.01}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def apply_pal_yaml(nominal, geometric_calibration, output):
    """Apply PAL keys as additive URDF origin xyz/rpy deltas.

    Written independently of the exporter: what robot_state_publisher is
    taken to do with ``<joint>_<dx|dy|dz|droll|dpitch|dyaw>`` keys.
    """
    import xml.etree.ElementTree as ET

    axes = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    tree = ET.parse(str(nominal))
    joints = {j.get("name"): j for j in tree.getroot().findall("joint")}
    for key, value in geometric_calibration.items():
        joint, axis = key.rsplit("_", 1)
        elem = joints.get(f"{joint}_joint", joints.get(joint))
        assert elem is not None, key
        origin = elem.find("origin")
        if origin is None:
            origin = ET.SubElement(elem, "origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        i = axes.index(axis)
        if i < 3:
            xyz[i] += value
        else:
            rpy[i - 3] += value
        origin.set("xyz", " ".join(repr(v) for v in xyz))
        origin.set("rpy", " ".join(repr(v) for v in rpy))
    tree.write(str(output))
    return output


_ROTATED_MOUNT = """<?xml version="1.0"?>
<robot name="arm">
  <link name="world"/>
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <link name="tool"/>
  <joint name="mount" type="fixed">
    <parent link="world"/><child link="base"/>
    <origin xyz="0 0 0.9" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="shoulder_joint" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0.1 0 0.2" rpy="0.3 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow_joint" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0.4 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="tool_joint" type="fixed">
    <parent link="l2"/><child link="tool"/>
    <origin xyz="0.3 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>
"""


@pytest.mark.parametrize(
    "corrections",
    [
        {
            "d_px_shoulder_joint": 0.003,
            "d_py_shoulder_joint": -0.002,
            "d_phix_shoulder_joint": 0.01,
            "d_phiz_shoulder_joint": -0.02,
        },
        {"offsetRZ_shoulder_joint": 0.03, "offsetRY_elbow_joint": -0.02},
    ],
    ids=["full_params", "joint_offset"],
)
def test_yaml_applied_to_urdf_reproduces_corrected_fk(tmp_path, corrections):
    """Behind a rotated fixed joint the deltas are taken against the URDF
    origin, not Pinocchio's merged placement (#123)."""
    from figaroh.calibration.calibration_tools import (
        apply_joint_offset,
        update_joint_placement,
    )
    from figaroh.tools.urdf_exporter import _parse_param_name

    nominal = tmp_path / "arm.urdf"
    nominal.write_text(_ROTATED_MOUNT)
    model = pin.buildModelFromUrdf(str(nominal))
    calib = _bare_calibration(
        {n: {"value": v, "std_dev": 1e-4} for n, v in corrections.items()},
        {"known_baseframe": True},
    )
    calib.model = model
    gc = build_geometric_calibration(calib)["robot_state_publisher"][
        "geometric_calibration"
    ]

    expected = model.copy()
    per_joint = {}
    for name, value in corrections.items():
        _, joint, idx, _ = _parse_param_name(name)
        per_joint.setdefault(joint, np.zeros(6))[idx] = value
    for joint, values in per_joint.items():
        jid = expected.getJointId(joint)
        if next(iter(corrections)).startswith("d_"):
            update_joint_placement(expected, jid, values)
        else:
            apply_joint_offset(expected, jid, values)
    reloaded = pin.buildModelFromUrdf(
        str(apply_pal_yaml(nominal, gc, tmp_path / "pal.urdf"))
    )
    fid = model.getFrameId("tool")
    ed, rd = expected.createData(), reloaded.createData()
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(-2, 2, model.nq)
        pin.framesForwardKinematics(expected, ed, q)
        pin.framesForwardKinematics(reloaded, rd, q)
        diff = pin.log6(ed.oMf[fid].inverse() * rd.oMf[fid]).vector
        assert np.abs(diff).max() < 1e-12


@pytest.mark.parametrize(
    "rpy",
    ["0.4 1.5707963267948966 0.2", "0.3 -1.5708 0", "0.1 2.0 -0.5"],
    ids=["gimbal-lock", "near-lock", "pitch-beyond-pi/2"],
)
def test_rpy_deltas_are_exact_for_the_origin_as_written(tmp_path, rpy):
    """PAL adds the deltas to the rpy written in the URDF; at or near
    pitch +-pi/2, or beyond, that triplet is not the one Pinocchio
    decomposes, so the URDF text is the reference (#123)."""
    from figaroh.calibration.calibration_tools import update_joint_placement

    nominal = tmp_path / "arm.urdf"
    nominal.write_text(_ROTATED_MOUNT.replace('rpy="0.3 0 0"', f'rpy="{rpy}"'))
    model = pin.buildModelFromUrdf(str(nominal))
    values = [0.003, -0.002, 0.001, 0.01, -0.02, 0.015]
    names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
    calib = _bare_calibration(
        {
            f"{n}_shoulder_joint": {"value": v, "std_dev": 1e-4}
            for n, v in zip(names, values)
        },
        {"known_baseframe": True},
    )
    calib.model = model
    gc = build_geometric_calibration(calib, nominal_urdf=str(nominal))[
        "robot_state_publisher"
    ]["geometric_calibration"]

    expected = model.copy()
    update_joint_placement(expected, expected.getJointId("shoulder_joint"), values)
    reloaded = pin.buildModelFromUrdf(
        str(apply_pal_yaml(nominal, gc, tmp_path / "pal.urdf"))
    )
    fid = model.getFrameId("tool")
    ed, rd = expected.createData(), reloaded.createData()
    for q in np.random.default_rng(0).uniform(-2, 2, (10, model.nq)):
        pin.framesForwardKinematics(expected, ed, q)
        pin.framesForwardKinematics(reloaded, rd, q)
        diff = pin.log6(ed.oMf[fid].inverse() * rd.oMf[fid]).vector
        assert np.abs(diff).max() < 1e-12


def test_nominal_urdf_missing_joint_raises(tmp_path):
    nominal = tmp_path / "arm.urdf"
    nominal.write_text(_ROTATED_MOUNT)
    calib = _bare_calibration(
        {"d_px_no_such_joint": {"value": 1e-3, "std_dev": 1e-4}},
        {"known_baseframe": True},
    )
    with pytest.raises(ValueError, match="no_such_joint"):
        build_geometric_calibration(calib, nominal_urdf=str(nominal))


@pytest.mark.parametrize(
    "rpy, switches",
    [
        # pitch -pi/2, joint axis not the shared roll/yaw axis: small delta
        ([-1.57079632679, -1.57079632679, 0.0], False),
        ([1.57079632679, -1.57079632679, -1.57079632679], False),
        # joint axis is the shared axis: no small rpy delta exists
        ([0.0, -1.57079632679, 0.0], True),
    ],
)
def test_rpy_delta_switches_branch_only_when_needed(rpy, switches):
    """An offset about the joint axis (TIAGo arm_4..arm_6 origins): the
    delta is the small one when it exists (#123)."""
    from figaroh.tools.geometric_calibration_export import _rpy_delta

    rpy = np.array(rpy)
    for offset in (0.003, -0.02):
        target = pin.rpy.rpyToMatrix(rpy) @ pin.rpy.rpyToMatrix(
            np.array([0.0, 0.0, offset])
        )
        d = _rpy_delta(rpy, target)
        exact = pin.rpy.rpyToMatrix(rpy + d).T @ target
        assert np.linalg.norm(pin.log3(exact)) < 1e-12
        if switches:
            assert abs(abs(d[0]) - np.pi / 2) < 1e-6
        else:
            assert np.abs(d).max() == pytest.approx(abs(offset), abs=1e-9)
