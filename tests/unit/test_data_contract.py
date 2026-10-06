"""Data contract types (docs/decisions/data-result-contract.md, #55).

Legacy parity: each type converts to and from today's dictionaries and CSV
loader exactly. Effort kinds, masks and protocols behave as the decision
record states.
"""

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.config import unified_to_legacy_config
from figaroh.calibration.data_loader import load_data
from figaroh.data import (
    JOINT_FORCE,
    JOINT_TORQUE,
    MOTOR_CURRENT,
    DataSource,
    PoseObservations,
    Protocol,
    Session,
    TrajectoryData,
    file_sha256,
)

ARM = ["torso_lift_joint"] + [f"arm_{i}_joint" for i in range(1, 8)]


def _raw(n=50, nj=len(ARM), seed=0):
    rng = np.random.default_rng(seed)
    return {
        "timestamps": np.arange(n).reshape(-1, 1) * 0.01,
        "positions": rng.normal(size=(n, nj)),
        "velocities": rng.normal(size=(n, nj)),
        "accelerations": None,
        "torques": rng.normal(size=(n, nj)),
    }


def _units():
    # torso is prismatic: a force in N; the arm joints are revolute
    return [JOINT_FORCE] + [JOINT_TORQUE] * 7, ["N"] + ["N·m"] * 7


# ── TrajectoryData ──


def test_trajectory_round_trips_the_legacy_dict():
    raw = _raw()
    kinds, units = _units()
    traj = TrajectoryData.from_legacy(raw, ARM, effort_kind=kinds, effort_unit=units)
    back = traj.to_legacy()
    for key in raw:
        if raw[key] is None:
            assert back[key] is None
        else:
            np.testing.assert_array_equal(back[key], raw[key])
    assert traj.origin == {"q": "measured", "dq": "measured", "ddq": "absent"}
    assert traj.mask.all()


def test_stacking_is_joint_major_like_the_regressor():
    raw = _raw()
    kinds, units = _units()
    traj = TrajectoryData.from_legacy(raw, ARM, effort_kind=kinds, effort_unit=units)
    np.testing.assert_array_equal(traj.stacked_effort(), raw["torques"].T.flatten())
    mask = np.ones(50, bool)
    mask[[3, 7]] = False
    rows = traj.with_mask(mask).stacked_mask()
    assert rows.shape == (50 * len(ARM),)
    # sample 3 of every joint is excluded, in the joint-major layout
    assert not rows[3] and not rows[50 + 3] and rows[4]


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda r: r.update(timestamps=r["timestamps"][::-1]), "increasing"),
        (lambda r: r.update(positions=r["positions"][:, :3]), "shape"),
        (lambda r: r["torques"].__setitem__((0, 0), np.nan), "non-finite"),
    ],
)
def test_trajectory_rejects_bad_arrays(change, match):
    raw = _raw()
    change(raw)
    kinds, units = _units()
    with pytest.raises(ValueError, match=match):
        TrajectoryData.from_legacy(raw, ARM, effort_kind=kinds, effort_unit=units)


def test_trajectory_rejects_bad_clock_and_mask():
    kinds, units = _units()
    with pytest.raises(ValueError, match="clock"):
        TrajectoryData.from_legacy(
            _raw(), ARM, effort_kind=kinds, effort_unit=units, clock="guessed"
        )
    traj = TrajectoryData.from_legacy(_raw(), ARM, effort_kind=kinds, effort_unit=units)
    with pytest.raises(ValueError, match="mask"):
        traj.with_mask(np.ones(3))


def test_effort_kinds_against_the_model(tiago_model):
    """Joint torque on revolute, joint force (N) on the prismatic torso;
    a motor current only with a declared drive gain."""
    kinds, units = _units()
    traj = TrajectoryData.from_legacy(_raw(), ARM, effort_kind=kinds, effort_unit=units)
    traj.check_effort(tiago_model)

    as_torque = TrajectoryData.from_legacy(
        _raw(), ARM, effort_kind=JOINT_TORQUE, effort_unit="N·m"
    )
    with pytest.raises(ValueError, match="torso_lift_joint \\(prismatic\\)"):
        as_torque.check_effort(tiago_model)

    current = TrajectoryData.from_legacy(
        _raw(),
        ARM,
        effort_kind=[JOINT_FORCE] + [MOTOR_CURRENT] * 7,
        effort_unit=["N"] + ["A"] * 7,
    )
    with pytest.raises(ValueError, match="arm_1_joint .*drive gain"):
        current.check_effort(tiago_model)
    current.check_effort(tiago_model, drive_gain_joints=ARM[1:])

    wrong_unit = TrajectoryData.from_legacy(
        _raw(), ARM, effort_kind=kinds, effort_unit=["N"] + ["mN·m"] * 7
    )
    with pytest.raises(ValueError, match="arm_1_joint"):
        wrong_unit.check_effort(tiago_model)


def test_conversion_keeps_the_recorded_signal(tiago_model):
    raw = _raw()
    recorded = TrajectoryData.from_legacy(
        raw, ARM, effort_kind=MOTOR_CURRENT, effort_unit="A"
    )
    scale = np.linspace(1.0, 2.0, len(ARM))
    offset = np.r_[9.81 * 20.0, np.zeros(7)]
    kinds, units = _units()
    joint = recorded.converted(
        scale,
        offset,
        effort_kind=kinds,
        effort_unit=units,
        description="x reduction_ratio x kmotor; + m g (torso)",
    )
    np.testing.assert_allclose(joint.effort, raw["torques"] * scale + offset)
    np.testing.assert_array_equal(joint.effort_raw, raw["torques"])
    assert joint.effort_raw_kind == (MOTOR_CURRENT,) * len(ARM)
    assert "kmotor" in joint.effort_conversion
    joint.check_effort(tiago_model)


# ── PoseObservations ──


@pytest.fixture
def calib_config(tiago_model):
    class _Robot:
        model = tiago_model
        data = tiago_model.createData()
        q0 = pin.neutral(tiago_model)

    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {"calibration_level": "joint_offset"},
        "measurements": {
            "markers": [
                {
                    "reference_joint": "arm_7_joint",
                    "measurable_dof": [True] * 3 + [False] * 3,
                }
            ]
        },
        "data": {"source_file": "unused.csv"},
    }
    return unified_to_legacy_config(_Robot(), unified)


@pytest.mark.parametrize("del_list", [[], [0, 5, 6]])
def test_observations_reproduce_load_data(
    tiago_model, tiago_data_dir, calib_config, del_list
):
    path = tiago_data_dir / "vicon_calibration_gripper1_shoulder.csv"
    legacy_cfg = copy.deepcopy(calib_config)
    pee, q = load_data(str(path), tiago_model, legacy_cfg, list(del_list))

    before = copy.deepcopy(calib_config)
    obs = PoseObservations.from_csv(
        path,
        tiago_model,
        calib_config,
        del_list=del_list,
        frame="vicon:world",
        registered_to="universe",
    )
    pee2, q2 = obs.to_legacy(tiago_model, calib_config)

    np.testing.assert_array_equal(pee2, pee)
    np.testing.assert_array_equal(q2, q)
    assert obs.n_samples == len(pd.read_csv(path))  # masked, not deleted
    assert (~obs.mask).sum() == len(del_list)
    # neither the config nor q0 is modified
    assert calib_config.keys() == before.keys()
    assert calib_config.get("NbSample") == before.get("NbSample")
    np.testing.assert_array_equal(calib_config["q0"], before["q0"])
    assert obs.source.files and obs.source.adapter


def test_missing_columns_are_listed(tiago_model, calib_config, tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"x1": [0.0], "arm_1_joint": [0.0]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="y1.*torso_lift_joint"):
        PoseObservations.from_csv(path, tiago_model, calib_config)


def test_constraint_only_observations(tiago_model, calib_config):
    """TALOS contact: no measured values, a declared constraint."""
    obs = PoseObservations(
        joint_names=ARM,
        q=np.zeros((4, len(ARM))),
        constraint="contact_gap_zero",
        session=np.array(["a", "a", "b", "b"], dtype=object),
    )
    assert list(obs.session) == ["a", "a", "b", "b"]
    with pytest.raises(ValueError, match="constraint-only"):
        obs.to_legacy(tiago_model, calib_config)
    with pytest.raises(ValueError, match="values or a constraint"):
        PoseObservations(joint_names=ARM, q=np.zeros((4, len(ARM))))


def test_joint_order_must_match_the_calibration(tiago_model, calib_config):
    obs = PoseObservations(
        joint_names=ARM[::-1],
        q=np.zeros((2, len(ARM))),
        values=np.zeros((2, 1, 6)),
        point_names=("1",),
        measurability=np.array([[True] * 3 + [False] * 3]),
    )
    with pytest.raises(ValueError, match="joint order"):
        obs.to_legacy(tiago_model, calib_config)


# ── Protocol ──


def _write_protocol(tmp_path, digest):
    data = tmp_path / "session.csv"
    data.write_text("x1\n0.0\n")
    manifest = tmp_path / "protocol.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "version": 1,
                "sessions": [
                    {
                        "id": "s1",
                        "role": "training",
                        "files": {"session.csv": digest(data)},
                    },
                    {"id": "s2", "role": "validation", "files": {}},
                ],
            }
        )
    )
    return manifest, data


def test_protocol_roles_and_hashes(tmp_path):
    manifest, data = _write_protocol(tmp_path, file_sha256)
    protocol = Protocol.load(manifest)
    assert [s.id for s in protocol.with_role("training")] == ["s1"]
    assert protocol.session("s2").role == "validation"
    protocol.verify(root=tmp_path)

    data.write_text("x1\n0.1\n")  # the recording changed under the protocol
    with pytest.raises(ValueError, match="s1: session.csv does not match"):
        protocol.verify(root=tmp_path)
    data.unlink()
    with pytest.raises(ValueError, match="missing"):
        protocol.verify(root=tmp_path)


def test_protocol_rejects_duplicate_sessions(tmp_path):
    manifest = tmp_path / "p.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "d",
                "version": 1,
                "sessions": [{"id": "s", "role": "a"}, {"id": "s", "role": "b"}],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        Protocol.load(manifest)


def test_session_identity_carries_no_role(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("a\n1\n")
    source = DataSource.from_files(
        [path], adapter="test", session=Session("2021-11-30-1544", "2021-11-30")
    )
    assert source.files[str(path)] == file_sha256(path)
    assert not hasattr(source.session, "role")


# ── sample indices and point selection (#131) ──


def test_sample_index_defaults_and_validation():
    kinds, units = _units()
    traj = TrajectoryData.from_legacy(_raw(), ARM, effort_kind=kinds, effort_unit=units)
    np.testing.assert_array_equal(traj.sample_index, np.arange(50))
    shifted = TrajectoryData.from_legacy(
        _raw(),
        ARM,
        effort_kind=kinds,
        effort_unit=units,
        sample_index=np.arange(50) + 921,
    )
    assert shifted.sample_index[0] == 921
    for bad in (np.arange(49), np.arange(50)[::-1], np.arange(50) * 0.5):
        with pytest.raises(ValueError, match="sample_index"):
            TrajectoryData.from_legacy(
                _raw(), ARM, effort_kind=kinds, effort_unit=units, sample_index=bad
            )
    with pytest.raises(ValueError, match="sample_index"):
        PoseObservations(
            joint_names=ARM,
            q=np.zeros((3, len(ARM))),
            constraint="gap",
            sample_index=np.array([0, 0, 1]),
        )


def test_csv_rows_survive_masking(tiago_model, tiago_data_dir, calib_config):
    path = tiago_data_dir / "vicon_calibration_gripper1_shoulder.csv"
    obs = PoseObservations.from_csv(path, tiago_model, calib_config, del_list=[2, 7])
    np.testing.assert_array_equal(obs.sample_index, np.arange(obs.n_samples))
    # the excluded postures are named by their CSV rows
    assert list(obs.sample_index[~obs.mask]) == [2, 7]


def test_select_points(tiago_model, calib_config):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 4, 6))
    meas = np.tile([True] * 3 + [False] * 3, (4, 1))
    obs = PoseObservations(
        joint_names=[tiago_model.names[j] for j in calib_config["actJoint_idx"]],
        q=rng.normal(size=(5, len(calib_config["actJoint_idx"]))),
        values=values,
        point_names=("BL", "BR", "TR", "TL"),
        measurability=meas,
        frame="qualisys:base_frame",
    )
    bl = obs.select_points(["BL"])
    assert bl.point_names == ("BL",) and bl.frame == obs.frame
    np.testing.assert_array_equal(bl.values[:, 0, :3], values[:, 0, :3])
    pee, _ = bl.to_legacy(tiago_model, calib_config)
    np.testing.assert_array_equal(pee, values[:, 0, :3].T.flatten())
    with pytest.raises(KeyError, match="XX"):
        obs.select_points(["XX"])


def test_zero_joints_are_refused():
    """An adapter whose joint list resolved empty must fail, not produce an
    empty dataset (found with TIAGo's legacy config, examples#17)."""
    raw = _raw()
    raw = {
        k: (v[:, :0] if k != "timestamps" and v is not None else v)
        for k, v in raw.items()
    }
    with pytest.raises(ValueError, match="at least one joint"):
        TrajectoryData.from_legacy(raw, [], effort_kind=[], effort_unit=[])
    with pytest.raises(ValueError, match="at least one joint"):
        PoseObservations(joint_names=[], q=np.zeros((2, 0)), constraint="gap")
