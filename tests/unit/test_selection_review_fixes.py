"""Regression tests for the #61 code-review findings."""

import os
import sys

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from test_data_contract_wiring import _Ident, _trajectory  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


@pytest.fixture(scope="module")
def traj(model):
    return _trajectory(model)


def _run(model, traj, **cfg):
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(cfg)
    ident.solve(decimate=False, plotting=False)
    return ident


def test_max_seconds_is_mapped_to_picos_timelimit(model, traj):
    pytest.importorskip("picos")
    ident = _run(
        model,
        traj,
        select_stage="physical_fit",
        physical_fit={"max_seconds": 60},
    )
    assert ident.selected.accepted, ident.selected.reason


# -- export_urdf: only identified joints are written --

from pathlib import Path  # noqa: E402

from figaroh.identification.selection import SelectedEstimate  # noqa: E402

PENDULUM = str(Path(__file__).resolve().parent.parent / "fixtures" / "pendulum.urdf")


def _nominal_selection(ident):
    values = np.array(list(ident.standard_parameter.values()), dtype=float)
    ident.selected = SelectedEstimate(
        stage="physical_fit",
        requested="physical_fit",
        status="accepted",
        reason="test",
        space="standard",
        names=list(ident.standard_parameter.keys()),
        values=values,
    )


def test_export_urdf_with_a_floating_base_model(tmp_path):
    m = pin.buildModelFromUrdf(PENDULUM, pin.JointModelFreeFlyer())
    ident = _Ident(m)
    ident.initialize_standard_parameters()
    ident._idx_eliminated = []
    _nominal_selection(ident)
    out = ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))
    assert pin.buildModelFromUrdf(out).nv == 2


MERGED = """<?xml version="1.0"?>
<robot name="merged">
  <link name="world"/>
  <joint name="joint1" type="revolute">
    <parent link="world"/><child link="link1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial><mass value="1"/><origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="joint2" type="revolute">
    <origin xyz="0 0 1"/>
    <parent link="link1"/><child link="link2"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial><mass value="1"/><origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="fix" type="fixed">
    <origin xyz="0 0 0.5"/><parent link="link2"/><child link="tool"/>
  </joint>
  <link name="tool">
    <inertial><mass value="0.5"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
</robot>
"""


def test_non_identified_merged_body_joint_does_not_block_export(tmp_path):
    urdf = tmp_path / "merged.urdf"
    urdf.write_text(MERGED)
    m = pin.buildModelFromUrdf(str(urdf))
    ident = _Ident(m)
    ident.initialize_standard_parameters()
    names = list(ident.standard_parameter.keys())
    ident._idx_eliminated = [i for i, n in enumerate(names) if n.endswith("_joint2")]
    _nominal_selection(ident)
    out = ident.export_urdf(str(urdf), output_path=str(tmp_path / "o.urdf"))
    assert pin.buildModelFromUrdf(out).nv == 2


# -- solve_with_custom_solver refreshes the rows the selection uses --


def test_custom_solver_after_solve_uses_the_new_rows(model, traj):
    pytest.importorskip("picos")
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(select_stage="physical_fit")
    ident.solve(decimate=True, decimation_factor=4, plotting=False)
    decimated_rows = ident._solve_W.shape[0]
    ident.solve_with_custom_solver(decimate=False)
    assert ident._solve_W.shape[0] == len(ident.tau_noised) != decimated_rows
    assert ident._solve_tau.shape[0] == len(ident.tau_noised)
    assert ident.selected.accepted, ident.selected.reason
    assert ident._wls_row_weight is None and ident._recon_result is None


def test_custom_solver_without_a_prior_solve_does_not_fail(model, traj):
    pytest.importorskip("picos")
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(select_stage="physical_fit")
    ident.solve_with_custom_solver(decimate=False)
    assert ident.selected.accepted, ident.selected.reason


# -- export_urdf: every failure is recorded, a file that cannot load is removed --


@pytest.fixture(scope="module")
def pend():
    return pin.buildModelFromUrdf(PENDULUM)


def _export_ident(pend):
    pytest.importorskip("picos")
    ident = _run(pend, _trajectory(pend), select_stage="physical_fit")
    assert ident.selected.accepted
    return ident


def _export_stage(ident):
    return {s.stage: s for s in ident.stages}["export"]


def test_export_records_a_missing_nominal_file(pend, tmp_path):
    ident = _export_ident(pend)
    with pytest.raises(FileNotFoundError):
        ident.export_urdf(
            str(tmp_path / "missing.urdf"), output_path=str(tmp_path / "o.urdf")
        )
    st = _export_stage(ident)
    assert st.status == "failed" and "FileNotFoundError" in st.reason


def test_export_removes_the_file_when_the_reload_fails(pend, tmp_path, monkeypatch):
    ident = _export_ident(pend)

    def boom(*a, **k):
        raise RuntimeError("cannot load")

    monkeypatch.setattr(pin, "buildModelFromUrdf", boom)
    out = tmp_path / "o.urdf"
    with pytest.raises(RuntimeError, match="cannot load"):
        ident.export_urdf(PENDULUM, output_path=str(out))
    assert not out.exists()
    st = _export_stage(ident)
    assert st.status == "failed" and "RuntimeError" in st.reason


# -- the export stage record follows the run --


def test_result_stages_follow_the_export_record(pend, tmp_path):
    ident = _export_ident(pend)
    assert "export" not in [d["stage"] for d in ident.result["stages"]]
    ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))
    exp = [d for d in ident.result["stages"] if d["stage"] == "export"]
    assert len(exp) == 1 and exp[0]["status"] == "ok"


def test_a_new_solve_drops_the_old_export_record(pend, tmp_path):
    ident = _export_ident(pend)
    ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))
    ident.solve(decimate=False, plotting=False)
    assert "export" not in [s.stage for s in ident.stages]
    assert "export" not in [d["stage"] for d in ident.result["stages"]]
    ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o2.urdf"))
    ident.solve_with_custom_solver(decimate=False)
    assert "export" not in [s.stage for s in ident.stages]


# -- scoped_verification: a fact may carry its own failure reason --


def _fact_checks(facts):
    from figaroh.tools._report_common import scoped_verification

    v = scoped_verification({}, {}, "execution", {}, False, [], facts=facts)
    return {c.name: c for c in v.checks}


def test_tuple_fact_uses_its_reason_and_plain_facts_keep_theirs():
    checks = _fact_checks(
        {
            "tuple_bad": (False, "because of x"),
            "tuple_ok": (True, "unused"),
            "plain_bad": False,
            "plain_ok": True,
            "unknown": None,
        }
    )
    assert checks["tuple_bad"].status == "fail"
    assert checks["tuple_bad"].reason == "because of x"
    assert checks["tuple_ok"].status == "pass" and checks["tuple_ok"].reason == ""
    assert checks["plain_bad"].reason == (
        "Numerical dimensions are missing or inconsistent"
    )
    assert checks["plain_ok"].status == "pass"
    assert checks["unknown"].status == "not_evaluated"


def test_rejected_selection_reason_reaches_the_verdict(model, traj, monkeypatch):
    pytest.importorskip("picos")
    from figaroh.identification import physical_fit as pf

    monkeypatch.setattr(
        pf,
        "_solve_core",
        lambda *a, **k: {"status": "error", "exception": "boom", "runtime_s": 0.0},
    )
    ident = _run(model, traj, select_stage="physical_fit")
    check = {c.name: c for c in ident.verify(scope="execution").checks}[
        "selected_stage_accepted"
    ]
    assert check.status == "fail"
    assert "physical_fit" in check.reason and "cvxopt: error" in check.reason


# -- verify and the reports speak for the selected estimate --


@pytest.fixture(scope="module")
def worse_selected(model, traj):
    """A physical fit pulled to the nominal model: worse than the base fit."""
    pytest.importorskip("picos")
    ident = _run(
        model,
        traj,
        select_stage="physical_fit",
        physical_fit={"prior_weight": 1e4},
    )
    assert ident.selected.accepted, ident.selected.reason
    return ident


def test_verify_judges_the_selected_estimate_not_the_base_fit(worse_selected):
    ident = worse_selected
    base = float(ident.result["rmse norm (N/m)"])
    chosen = float(ident.result["selected"]["effort_rmse_fit"])
    assert chosen > 1.01 * base
    limit = {"rmse": {"threshold": (base + chosen) / 2, "comparison": "max"}}
    v = ident.verify(thresholds=limit, scope="execution")
    assert v.metrics["rmse"] == pytest.approx(chosen)
    check = {c.name: c for c in v.checks}
    assert check["rmse"].status == "fail" and not v.passed
    assert check["finite_fit_rmse"].value == 1.0


def test_default_verify_still_uses_the_base_fit(model, traj):
    ident = _run(model, traj)
    v = ident.verify(scope="execution")
    assert v.metrics["rmse"] == pytest.approx(ident.result["rmse norm (N/m)"])


def test_terminal_report_labels_the_base_fit(worse_selected, capsys):
    worse_selected.print_quality_report()
    out = capsys.readouterr().out
    assert "RMSE (base fit)" in out
    assert "Base fit residuals" in out
    assert "RMSE (physical_fit estimate)" in out
    assert "Selected estimate (physical_fit) RMSE per joint" in out


def test_default_terminal_report_is_unlabelled(model, traj, capsys):
    _run(model, traj).print_quality_report()
    out = capsys.readouterr().out
    assert "base fit" not in out.lower()
    assert "  RMSE:         " in out


def test_html_report_labels_the_base_fit(worse_selected, tmp_path):
    path = worse_selected.export_html_report(output_path=str(tmp_path / "r.html"))
    html = Path(path).read_text()
    assert "RMSE (base fit)" in html
    assert "Base fit residuals" in html
    assert "RMSE (selected estimate)" in html
    assert "RMSE per joint" in html


# -- feasibility: identified links only, mass bound within solver tolerance --


def test_reconstruction_feasibility_skips_links_without_identified_parameters(
    model, traj
):
    from figaroh.identification.selection import _link_feasibility

    ident = _run(model, traj)
    names = list(ident.standard_parameter.keys())
    joints = list(model.names[1:])
    first, second = joints[0], joints[1]
    ident._params_r_for_recon = [n for n in names if n.endswith(f"_{first}")]
    theta = np.array(list(ident.standard_parameter.values()), dtype=float)
    theta[names.index(f"m_{second}")] = 0.0  # nominal-only link, unphysical
    feas = _link_feasibility(ident, theta, 1e-6, -1e-10)
    assert set(feas) == {first}
    ident._params_r_for_recon = None  # no information: every link is judged
    assert not _link_feasibility(ident, theta, 1e-6, -1e-10)[second]["ok"]


KEYS = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]


def _sphere_p10(m):
    return np.array([m, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1])


def test_physical_fit_feasibility_accepts_a_mass_on_its_bound():
    pytest.importorskip("picos")
    from figaroh.identification import physical_fit as pf

    names = [f"{k}_j1" for k in KEYS]
    problem = pf.build_problem(
        np.eye(10),
        np.ones(10),
        names,
        ["j1"],
        prior=_sphere_p10(0.5),
        policy=pf.PhysicalPolicy(mass_min=0.5),
    )
    on_bound = pf._feasibility(problem, _sphere_p10(0.5 - 1e-12))
    assert on_bound["j1"]["ok"]
    below = pf._feasibility(problem, _sphere_p10(0.5 - 1e-6))
    assert not below["j1"]["ok"]


def test_reconstruction_feasibility_accepts_a_mass_on_its_bound(model, traj):
    from figaroh.identification.selection import _link_feasibility

    ident = _run(model, traj)
    names = list(ident.standard_parameter.keys())
    theta = np.array(list(ident.standard_parameter.values()), dtype=float)
    joint = model.names[1]
    for k, v in zip(KEYS, _sphere_p10(0.5 - 1e-12)):
        theta[names.index(f"{k}_{joint}")] = v
    assert _link_feasibility(ident, theta, 0.5, -1e-10)[joint]["ok"]
