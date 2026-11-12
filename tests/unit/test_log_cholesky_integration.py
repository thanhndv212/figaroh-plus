"""physical_fit.method: log_cholesky through the identification workflow (#25)."""

import json
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
from test_estimate_selection import _run  # noqa: E402

import figaroh.identification.log_cholesky_fit as lcf  # noqa: E402
from figaroh.data import TrajectoryData  # noqa: E402
from figaroh.identification import physical_fit as pf  # noqa: E402
from figaroh.identification.config import (  # noqa: E402
    PHYSICAL_FIT_DEFAULTS,
    _extract_selection_config,
)
from figaroh.tools.run_archive import archive_run  # noqa: E402

LC = {"method": "log_cholesky"}


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


@pytest.fixture(scope="module")
def traj(model):
    return _trajectory(model)


def _lc(**options):
    return {"select_stage": "physical_fit", "physical_fit": {**LC, **options}}


def _parse(src):
    out = {}
    _extract_selection_config(out, src)
    return out


# ── config parsing and forwarding ──


def test_method_defaults_to_direct():
    out = _parse({"physical_fit": {"enabled": True}})["physical_fit"]
    assert out["method"] == "direct" and out["log_cholesky"] is None
    assert set(out) == set(PHYSICAL_FIT_DEFAULTS)


def test_log_cholesky_options_are_kept():
    out = _parse({"physical_fit": {"method": " Log_Cholesky ", "log_cholesky": {}}})
    assert out["physical_fit"]["method"] == "log_cholesky"
    out = _parse(
        {"physical_fit": {"method": "log_cholesky", "log_cholesky": {"max_nfev": 7}}}
    )
    assert out["physical_fit"]["log_cholesky"] == {"max_nfev": 7}


@pytest.mark.parametrize(
    "pf_cfg, match",
    [
        ({"method": "gauss_newton"}, "method='gauss_newton' is unknown"),
        ({"log_cholesky": {"max_nfev": 5}}, "method is 'direct'"),
        ({**LC, "second_solver": "qics"}, "second_solver is not supported"),
        ({**LC, "max_seconds": 5.0}, "log_cholesky.max_seconds"),
        ({**LC, "log_cholesky": {"nonsense": 1}}, "unknown physical_fit.log_cholesky"),
        ({**LC, "log_cholesky": {"sdp_solver": "mosek"}}, "physical_fit.solver"),
        ({**LC, "log_cholesky": {"stop_rms": -1.0}}, "stop_rms must be positive"),
        ({**LC, "log_cholesky": {"branch": "middle"}}, "branch must be one of"),
        ({**LC, "log_cholesky": [1]}, "must be a mapping"),
    ],
)
def test_bad_settings_fail_clearly(pf_cfg, match):
    with pytest.raises(ValueError, match=match):
        _parse({"physical_fit": pf_cfg})


def _from_unified_yaml(tmp_path, model, physical_fit):
    import types

    import yaml

    from figaroh.identification.config import unified_to_legacy_identif_config
    from figaroh.utils.config_parser import UnifiedConfigParser, create_task_config

    path = tmp_path / "unified.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {"name": "arm"},
                "tasks": {
                    "identification": {
                        "select_stage": "physical_fit",
                        "physical_fit": physical_fit,
                    }
                },
            }
        )
    )
    robot = types.SimpleNamespace(model=model)
    task = create_task_config(
        robot, UnifiedConfigParser(str(path)).parse(), "identification"
    )
    return unified_to_legacy_identif_config(robot, task)


def test_unified_yaml_carries_method_and_options(tmp_path, model):
    cfg = _from_unified_yaml(
        tmp_path,
        model,
        {"method": "log_cholesky", "log_cholesky": {"stop_rms": 1e-3, "max_nfev": 50}},
    )
    assert cfg["select_stage"] == "physical_fit"
    assert cfg["physical_fit"]["method"] == "log_cholesky"
    assert cfg["physical_fit"]["log_cholesky"] == {"stop_rms": 1e-3, "max_nfev": 50}


def test_unified_yaml_rejects_contradictions(tmp_path, model):
    with pytest.raises(ValueError, match="second_solver"):
        _from_unified_yaml(
            tmp_path, model, {"method": "log_cholesky", "second_solver": "qics"}
        )


def test_programmatic_bad_config_raises_from_solve(model, traj):
    with pytest.raises(ValueError, match="is unknown"):
        _run(model, traj, **_lc(method="nope"))
    with pytest.raises(ValueError, match="unknown physical_fit keys"):
        _run(model, traj, select_stage="physical_fit", physical_fit={"bogus": 1})


def test_options_reach_the_solver(model, traj, monkeypatch):
    seen = {}
    real = lcf.solve_log_cholesky_fit

    def spy(p, **kw):
        seen["options"] = kw["options"]
        return real(p, **kw)

    monkeypatch.setattr(lcf, "solve_log_cholesky_fit", spy)
    _run(
        model,
        traj,
        **_lc(solver="cvxopt", log_cholesky={"max_nfev": 3, "repair_start": True}),
    )
    opts = seen["options"]
    assert opts.max_nfev == 3 and opts.repair_start is True
    assert opts.sdp_solver == "cvxopt"
    assert opts.stop_rms == lcf.LogCholeskyOptions().stop_rms


# ── default compatibility ──


def test_direct_result_names_its_method(model, traj):
    pytest.importorskip("picos")
    ident = _run(model, traj, select_stage="physical_fit")
    assert ident.result["physical_fit"]["method"] == "direct"
    assert "log_cholesky" not in ident.result["physical_fit"]
    assert ident.result["selected"]["stage"] == "physical_fit"
    assert ident.result["selected"]["physical_fit_method"] == "direct"
    assert ident.selected.effective_method == "direct"


# ── training data only ──


class _HeldOut(_Ident):
    """Training trajectory, plus a shorter held-out one for validation."""

    def load_trajectory_data(self, data_source=None):
        return self.heldout if data_source else self.trajectory_to_return


def test_solver_sees_training_rows_never_heldout(model, traj, monkeypatch):
    seen = {}
    real = lcf.solve_log_cholesky_fit

    def spy(p, **kw):
        seen["problem"] = p
        return real(p, **kw)

    monkeypatch.setattr(lcf, "solve_log_cholesky_fit", spy)
    n = 150
    heldout = TrajectoryData(
        t=traj.t[:n],
        joint_names=list(traj.joint_names),
        q=traj.q[:n],
        dq=traj.dq[:n],
        ddq=traj.ddq[:n],
        effort=traj.effort[:n] * 1.01,
        effort_kind=traj.effort_kind,
        effort_unit=traj.effort_unit,
    )
    ident = _HeldOut(model)
    ident.trajectory_to_return, ident.heldout = traj, heldout
    ident.identif_config["validation_data_file"] = "heldout"
    ident.initialize()
    ident.identif_config.update(_lc(log_cholesky={"branch": "interior"}))
    ident.solve(decimate=False, plotting=False)
    assert ident._val_available
    p = seen["problem"]
    np.testing.assert_array_equal(p.tau, ident._solve_tau)
    assert p.tau.size == ident._solve_tau.size != ident._val_num_samples * 6
    eliminated = set(ident._idx_eliminated or [])
    keep = [i for i in range(len(p.params_std)) if i not in eliminated]
    np.testing.assert_array_equal(p.Y[:, keep], ident._solve_W)
    assert list(p.params_std) == list(ident.standard_parameter)


# ── boundary branch: accepted, labelled everywhere ──


@pytest.fixture(scope="module")
def boundary(model, traj):
    pytest.importorskip("picos")
    return _run(model, traj, **_lc(log_cholesky={"branch": "boundary"}))


def test_boundary_branch_is_accepted_and_labelled(boundary):
    sel = boundary.selected
    assert sel.accepted and sel.stage == "physical_fit"
    assert sel.effective_method == "log_cholesky"
    res = boundary.result
    assert res["selected"]["log_cholesky_branch"] == "boundary"
    pfr = res["physical_fit"]
    assert pfr["method"] == "log_cholesky" and pfr["status"] == "accepted"
    assert pfr["solver_status"] == "optimal" and pfr["objective_value"] > 0
    assert pfr["runtime_s"] > 0 and pfr["solvers"] == ["cvxopt"]
    lc = pfr["log_cholesky"]
    assert lc["branch"] == "boundary" and lc["branch_forced"] is True
    assert lc["rank"] + lc["null_space_dim"] == lc["n_params"]
    assert lc["start_used"] is False and lc["interior"] is None
    assert "boundary branch (cvxopt SDP): optimal" in sel.reason
    json.dumps(pfr)  # serialisable as-is


def test_boundary_matches_the_direct_fit(boundary, model, traj):
    """Same objective, same SDP: the forced boundary branch is the direct fit."""
    direct = _run(model, traj, select_stage="physical_fit")
    np.testing.assert_allclose(
        boundary.selected.values, direct.selected.values, rtol=1e-6, atol=1e-8
    )


def test_boundary_prediction_validation_and_verdict(boundary):
    vm = boundary.result["validation_metrics"]
    assert vm["estimate_stage"] == "physical_fit"
    assert vm["estimate_method"] == "log_cholesky"
    got = np.concatenate([vm["tau_identified_per_joint"][j] for j in vm["joint_names"]])
    want = boundary.dynamic_regressor @ boundary.selected.values
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    stage = {s.stage: s for s in boundary.stages}["physical"]
    assert stage.status == "ok" and "log_cholesky" in stage.reason
    assert stage.metrics["runtime_s"]["unit"] == "s"
    assert "objective_value" in stage.metrics and "null_space_dim" in stage.metrics
    v = boundary.verify(scope="execution")
    assert v.passed and v.selected_stage == "physical_fit"


def test_boundary_reports_and_archive(boundary, tmp_path, capsys):
    boundary.print_quality_report()
    out = capsys.readouterr().out
    label = "physical_fit (log_cholesky, boundary branch)"
    assert f"Reported estimate: {label}" in out
    assert f"RMSE ({label} estimate)" in out
    assert "Physical fit (log_cholesky): accepted" in out
    html = open(boundary.export_html_report(output_path=str(tmp_path / "r.html")))
    assert label in html.read()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    boundary.export_verification_report(
        output_path=str(run_dir / "verdict.json"), scope="execution"
    )
    archive_run(boundary, run_dir)
    assert (run_dir / "parameters.csv").exists()
    stages = json.loads((run_dir / "stages.json").read_text())["stages"]
    physical = next(s for s in stages if s["stage"] == "physical")
    assert "log_cholesky boundary branch" in physical["reason"]
    assert "method: log_cholesky" in (run_dir / "config.snapshot.yaml").read_text()


# ── interior branch ──


def test_interior_converges_without_sdp(model, traj, monkeypatch):
    """The interior path neither imports PICOS nor calls the SDP core."""

    def no_sdp(*a, **k):
        raise AssertionError("SDP core called on the interior branch")

    monkeypatch.setattr(pf, "_solve_core", no_sdp)
    monkeypatch.setitem(sys.modules, "picos", None)  # any import raises
    # default options: the #155 stop rule holds on this model (#234)
    ident = _run(model, traj, **_lc())
    lc = ident.result["physical_fit"]["log_cholesky"]
    assert lc["branch"] == "interior" and lc["branch_forced"] is False
    assert lc["start_used"] and lc["start_source"] == "prior"
    assert lc["interior"]["stop_reason"] == "converged"
    assert lc["interior"]["identifiable_rms"] <= 1e-7
    assert ident.selected.accepted and ident.result["physical_fit"]["solvers"] == [
        "scipy"
    ]
    assert ident.result["selected"]["log_cholesky_branch"] == "interior"
    assert {s.stage: s for s in ident.stages}["physical"].metrics["nfev"]["value"] > 0


def test_interior_matches_the_direct_fit(model, traj, monkeypatch):
    """Interior optimum, same objective: the same estimate as the SDP (#234)."""
    pytest.importorskip("picos")
    seen = {}
    real = lcf.solve_log_cholesky_fit

    def spy(p, **kw):
        seen["p"] = p
        return real(p, **kw)

    monkeypatch.setattr(lcf, "solve_log_cholesky_fit", spy)
    interior = _run(model, traj, **_lc())
    assert interior.result["physical_fit"]["log_cholesky"]["branch"] == "interior"
    direct = _run(model, traj, select_stage="physical_fit")
    # Parameters along the null space are set only by the weak prior term and
    # the SDP tolerance, so compare what the data determines.
    # R from the QR of the weighted regressor: ||R d|| = ||W Y d||.
    R = seen["p"].R
    diff = R @ (interior.selected.values - direct.selected.values)
    assert np.linalg.norm(diff) / np.sqrt(seen["p"].tau.size) < 1e-5
    obj_i = interior.result["physical_fit"]["objective_value"]
    obj_d = direct.result["physical_fit"]["objective_value"]
    assert obj_i <= obj_d * (1 + 1e-6) + 1e-12


def test_non_convergence_is_a_failure_not_a_zero(model, traj, tmp_path):
    """An exhausted evaluation budget rejects the fit; nothing reads as zero."""
    ident = _run(model, traj, **_lc(log_cholesky={"max_nfev": 1}))
    pfr = ident.result["physical_fit"]
    assert pfr["status"] == "rejected" and pfr["objective_value"] is None
    assert pfr["solver_status"] == "max_nfev"
    assert pfr["log_cholesky"]["interior"]["identifiable_rms"] > 1e-7
    sel = ident.selected
    assert not sel.accepted and sel.stage == "none" and sel.values is None
    assert sel.effective_method == "log_cholesky"
    stage = {s.stage: s for s in ident.stages}["physical"]
    assert stage.status == "failed" and "objective_value" not in stage.metrics
    assert ident.result["validation_metrics"]["estimate_stage"] == "fit"
    assert "estimate_method" not in ident.result["validation_metrics"]
    assert ident.verify(scope="execution").passed is False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    archive_run(ident, run_dir)
    assert not (run_dir / "parameters.csv").exists()
    assert (run_dir / "fit_parameters.csv").exists()
    with pytest.raises(ValueError, match="rejected"):
        ident.export_urdf("unused.urdf")


def test_invalid_start_is_reported(model, traj, monkeypatch):
    """A prior the log-Cholesky map cannot encode fails unless repaired."""
    real = lcf.solve_log_cholesky_fit

    def bad_prior(p, **kw):
        import dataclasses

        prior = p.theta_prior.copy()
        # positive mass, inertia not positive definite: repairable
        prior[p.params_std.index(f"Ixx_{p.joint_names[0]}")] = -1.0
        return real(dataclasses.replace(p, theta_prior=prior), **kw)

    monkeypatch.setattr(lcf, "solve_log_cholesky_fit", bad_prior)
    opts = {"branch": "interior"}
    ident = _run(model, traj, **_lc(log_cholesky=opts))
    assert ident.result["physical_fit"]["solver_status"] == "invalid_start"
    assert "repair_start=True" in ident.selected.reason
    ident = _run(model, traj, **_lc(log_cholesky={**opts, "repair_start": True}))
    repairs = ident.result["physical_fit"]["log_cholesky"]["start_repairs"]
    assert list(repairs) == [ident.model.names[1]]
    assert repairs[ident.model.names[1]]["min_eig_after"] > 0
