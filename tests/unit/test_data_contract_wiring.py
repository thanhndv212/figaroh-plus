"""Data contract in the base classes (#55, part 2).

Identification accepts a TrajectoryData: its effort is checked, its mask
leaves samples out after filtering, and every step is recorded as a stage.
Calibration reads its CSV through PoseObservations and records stages too.
"""

import json
import types

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.data import JOINT_TORQUE, MOTOR_CURRENT, TrajectoryData
from figaroh.identification.base_identification import BaseIdentification
from figaroh.tools.stages import SCHEMA_VERSION, StageResult, record_stage

N = 400
FS = 100.0


def _trajectory(model, kind=JOINT_TORQUE, unit="N·m", mask=None):
    """Smooth excitation with exact inverse-dynamics torques."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / FS
    freqs = rng.uniform(0.2, 0.8, (3, model.nv))
    amps = rng.uniform(0.3, 1.0, (3, model.nv))
    q = sum(a * np.sin(2 * np.pi * f * t[:, None]) for a, f in zip(amps, freqs))
    dq = sum(
        a * 2 * np.pi * f * np.cos(2 * np.pi * f * t[:, None])
        for a, f in zip(amps, freqs)
    )
    ddq = sum(
        -a * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * t[:, None])
        for a, f in zip(amps, freqs)
    )
    data = model.createData()
    tau = np.stack([pin.rnea(model, data, q[i], dq[i], ddq[i]) for i in range(N)])
    names = [model.names[j] for j in range(1, model.njoints)]
    return TrajectoryData(
        t=t,
        joint_names=names,
        q=q,
        dq=dq,
        ddq=ddq,
        effort=tau,
        effort_kind=kind,
        effort_unit=unit,
        mask=mask,
    )


class _Ident(BaseIdentification):
    trajectory_to_return = None

    def __init__(self, model):
        robot = types.SimpleNamespace(
            model=model,
            data=model.createData(),
            q0=pin.neutral(model),
            v0=np.zeros(model.nv),
        )
        # BaseIdentification.__init__ reads a config file; set what the
        # pipeline uses directly instead
        self.robot, self.model, self.data = robot, model, robot.data
        nv = model.nv
        self.identif_config = {
            "has_friction": False,
            "has_actuator_inertia": False,
            "has_joint_offset": False,
            "is_joint_torques": True,
            "is_external_wrench": False,
            "act_idxv": list(range(nv)),
            "act_idxq": list(range(model.nq)),
            "active_joints": [model.names[j] for j in range(1, model.njoints)],
        }
        self.filter_config = {
            "differentiation_method": "gradient",
            "filter_params": {
                "nbutter": 4,
                "f_butter": 10,
                "med_fil": 1,
                "f_sample": FS,
            },
        }
        for attr in (
            "dynamic_regressor",
            "standard_parameter",
            "additional_parameters",
            "custom_parameters",
            "params_base",
            "dynamic_regressor_base",
            "phi_base",
            "rms_error",
            "correlation",
            "processed_data",
            "result",
            "num_samples",
            "tau_ref",
            "tau_identif",
            "tau_noised",
            "_val_processed_data",
            "_val_num_samples",
            "_idx_eliminated",
            "_base_indices",
            "trajectory",
            "_sample_mask",
            "_val_trajectory",
            "_val_sample_mask",
        ):
            setattr(self, attr, None)
        self._val_available = False
        self._decimate_used = False
        self.stages = []

    def load_trajectory_data(self, data_source=None):
        return self.trajectory_to_return

    def _calculate_base_parameters(self, tau, regressor, active):
        self.qr_rows = (np.array(tau), np.array(regressor))  # rows entering QR
        return super()._calculate_base_parameters(tau, regressor, active)


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


def _solve(model, traj, decimate=False):
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.solve(decimate=decimate, decimation_factor=4, plotting=False)
    return ident


def test_trajectory_runs_and_records_stages(model):
    ident = _solve(model, _trajectory(model))
    stages = {s.stage: s for s in ident.stages}
    assert stages["data"].status == "ok"
    assert "TrajectoryData" in stages["data"].reason
    assert stages["fit"].status == "ok"
    assert stages["fit"].metrics["effort_rmse"]["unit"].startswith("N·m")
    # no held-out trajectory: the training fallback is a stage, not a log line
    assert stages["validation"].status == "fallback"
    assert [s["stage"] for s in ident.result["stages"]] == ["data", "fit", "validation"]
    assert ident.result["effort rmse"] == ident.result["rmse norm (N/m)"]


def test_drive_side_effort_is_refused(model):
    ident = _Ident(model)
    ident.trajectory_to_return = _trajectory(model, kind=MOTOR_CURRENT, unit="A")
    with pytest.raises(ValueError, match="drive gain"):
        ident.initialize()


def test_trajectory_with_a_torque_override_is_refused(model):
    class _Converting(_Ident):
        def process_torque_data(self, **kwargs):
            return self.raw_data["torques"] * 2.0

    ident = _Converting(model)
    ident.trajectory_to_return = _trajectory(model)
    with pytest.raises(TypeError, match="converted"):
        ident.initialize()


def _rows(ident):
    return ident.qr_rows


@pytest.mark.parametrize("decimate", [False, True])
def test_mask_applies_after_filtering(model, decimate):
    """Masking samples gives the rows of the full, filtered signal minus the
    masked ones, not the rows of a signal cut before filtering."""
    mask = np.ones(N, bool)
    mask[100:140] = False
    full = _solve(model, _trajectory(model), decimate)
    masked = _solve(model, _trajectory(model, mask=mask), decimate)

    factor = 4 if decimate else 1
    keep = np.tile(mask[::factor], model.nv)
    tau_full, W_full = _rows(full)
    tau_masked, W_masked = _rows(masked)
    np.testing.assert_allclose(tau_masked, tau_full[keep])
    np.testing.assert_allclose(W_masked, W_full[keep])
    assert masked.stages[0].metrics["masked_samples"]["value"] == 40

    # cutting the samples before filtering gives different regressor rows:
    # the filters and derivatives then run across the gap (torques are not
    # filtered by the base class, so they agree)
    cut = _trajectory(model)
    cut_traj = TrajectoryData(
        t=cut.t[mask],
        joint_names=cut.joint_names,
        q=cut.q[mask],
        dq=cut.dq[mask],
        ddq=cut.ddq[mask],
        effort=cut.effort[mask],
        effort_kind=JOINT_TORQUE,
        effort_unit="N·m",
    )
    if not decimate:
        cut_ident = _solve(model, cut_traj, decimate)
        _, W_cut = _rows(cut_ident)
        assert W_cut.shape == W_masked.shape
        assert not np.allclose(W_cut, W_masked)


def test_legacy_dict_still_works(model):
    ident = _Ident(model)
    ident.trajectory_to_return = _trajectory(model).to_legacy()
    ident.initialize()
    ident.solve(decimate=False, plotting=False)
    assert ident.trajectory is None
    assert ident.stages[0].reason == "legacy dict"


# ── stage records ──


def test_stage_records_replace_and_order():
    obj = types.SimpleNamespace()
    record_stage(obj, "validation", "fallback", "no held-out data")
    record_stage(obj, "data", "ok", metrics={"samples": 3})
    record_stage(obj, "data", "failed", "second load")
    assert [(s.stage, s.status) for s in obj.stages] == [
        ("data", "failed"),
        ("validation", "fallback"),
    ]
    assert obj.stages[0].metrics == {}
    with pytest.raises(ValueError, match="stage"):
        StageResult("plotting", "ok")
    with pytest.raises(ValueError, match="status"):
        StageResult("fit", "maybe")


def test_verification_export_carries_schema_and_stages(model, tmp_path):
    ident = _solve(model, _trajectory(model))
    path = ident.export_verification_report(
        output_path=str(tmp_path / "v.json"), scope="execution"
    )
    data = json.loads(open(path).read())
    assert data["schema_version"] == SCHEMA_VERSION
    assert [s["stage"] for s in data["stages"]] == ["data", "fit", "validation"]


# ── calibration ──


def test_calibration_reads_observations_and_records_stages(tiago_model, tmp_path):
    import copy

    import pandas as pd

    from figaroh.calibration.base_calibration import BaseCalibration
    from figaroh.calibration.calibration_tools import (
        calc_updated_fkm,
        random_joint_configuration,
    )
    from figaroh.calibration.config import unified_to_legacy_config
    from figaroh.utils.error_handling import CalibrationError

    class _Robot:
        model = tiago_model
        data = tiago_model.createData()
        q0 = pin.neutral(tiago_model)

    class _Calib(BaseCalibration):
        def cost_function(self, var):
            return (
                calc_updated_fkm(
                    self.model, self.data, var, self.q_measured, self.calib_config
                )
                - self.PEE_measured
            )

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
    cfg = unified_to_legacy_config(_Robot(), unified)
    cfg.update(known_baseframe=False, known_tipframe=False)
    cfg["coeff_regularize"] = 0.0
    rng = np.random.default_rng(0)
    n = 30
    q = np.array([random_joint_configuration(tiago_model, rng) for _ in range(n)])
    joints = [tiago_model.names[j] for j in cfg["actJoint_idx"]]
    pee = calc_updated_fkm(
        tiago_model,
        tiago_model.createData(),
        np.zeros(2),
        q,
        dict(copy.deepcopy(cfg), param_name=["base_px", "pEEx_1"], NbSample=n),
    ).reshape(3, n)
    df = pd.DataFrame({"x1": pee[0], "y1": pee[1], "z1": pee[2]})
    for j in joints:
        df[j] = q[:, tiago_model.joints[tiago_model.getJointId(j)].idx_q]
    path = tmp_path / "postures.csv"
    df.to_csv(path, index=False)

    calib = _Calib.__new__(_Calib)
    calib.model, calib.data, calib.calib_config = (
        tiago_model,
        tiago_model.createData(),
        cfg,
    )
    calib._data_path = str(path)
    calib.del_list_ = [3, 4]
    calib.stages, calib.observations, calib._val_available = [], None, False
    calib.initialize()
    calib.solve(plotting=False, enable_logging=False)

    assert calib.observations.n_samples == n  # masked, not deleted
    assert calib.calib_config["NbSample"] == n - 2 == len(calib.q_measured)
    stages = {s.stage: s for s in calib.stages}
    assert stages["data"].metrics["masked_samples"]["value"] == 2
    assert stages["fit"].status == "ok"
    assert stages["validation"].status == "fallback"
    assert [s["stage"] for s in calib.results_data["stages"]] == [
        "data",
        "fit",
        "validation",
    ]

    # an export the URDF cannot carry is a failed export stage
    calib.calib_config = dict(calib.calib_config)
    calib.calib_config["param_name"] = list(cfg["param_name"]) + ["k_RZ_arm_2_joint"]
    calib.var_ = np.append(calib.var_, 1e-3)
    with pytest.raises(CalibrationError):
        calib.joint_corrections(lift=False)
    assert {s.stage: s.status for s in calib.stages}["export"] == "failed"
