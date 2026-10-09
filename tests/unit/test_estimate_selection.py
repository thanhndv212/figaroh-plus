"""select_stage: which estimate an identification run reports (#61)."""

import json
import os
import re
import sys

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from _selection_capture import capture  # noqa: E402
from test_data_contract_wiring import _Ident, _trajectory  # noqa: E402

from figaroh.identification import physical_fit as pf  # noqa: E402
from figaroh.identification.selection import select_estimate  # noqa: E402

GOLDEN = os.path.join(
    os.path.dirname(__file__), "data", "identification_default_capture.json"
)


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


# Expressed in the base-parameter basis, which QR picks differently across
# platforms and BLAS builds (#116): compared by size only. Predictions,
# errors, validation and the verdict do not depend on that choice.
BASIS_DEPENDENT = {
    "/result/base parameters",
    "/result/base parameters names",
    "/result/base parameters values",
    "/result/base regressor",
    "/result/std dev of estimated param",
    "/result/condition number",
    "/verdict/metrics/condition_number",
}


# insight sentences quote the condition number and, for poorly identified
# parameters, names expressed in the (basis-dependent) base parameters
_COND = re.compile(r"(Condition number )[-+0-9.eE]+")
_POOR = re.compile(r"(poorly identified: ).*")


def _mask(s):
    return _POOR.sub(r"\1#", _COND.sub(r"\1#", s))


def _size(x):
    if isinstance(x, dict) and "shape" in x:
        return x["shape"]
    return len(x) if isinstance(x, (dict, list)) else None


def _close(a, b, path=""):
    """Equal structure; numbers within a tight tolerance."""
    if path in BASIS_DEPENDENT:
        assert _size(a) == _size(b), path
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12), path
    elif isinstance(a, str) and isinstance(b, str):
        assert _mask(a) == _mask(b), path
    else:
        assert a == b, path


# ── default behaviour is untouched ──


@pytest.mark.parametrize("cfg", [{}, {"select_stage": "fit"}])
def test_default_and_explicit_fit_match_pre_change_capture(model, traj, cfg):
    ident = _run(model, traj, **cfg)
    with open(GOLDEN) as f:
        golden = json.load(f)
    got = capture(ident)
    assert "selected" not in got["result"]
    assert "physical_fit" not in got["result"]
    assert "estimate_stage" not in got["result"].get("validation_metrics", {})
    assert ident.selected is None
    _close(golden, got)


def test_explicit_fit_equals_default_exactly(model, traj):
    a, b = _run(model, traj), _run(model, traj, select_stage="fit")
    assert capture(a) == capture(b)


# ── physical_fit ──


@pytest.fixture(scope="module")
def pfit(model, traj):
    pytest.importorskip("picos")
    return _run(model, traj, select_stage="physical_fit")


def test_physical_fit_noise_free_is_accepted(pfit):
    sel = pfit.selected
    assert sel.accepted and sel.stage == "physical_fit" and sel.space == "standard"
    assert all(v["ok"] for v in sel.feasibility.values())
    assert pfit.result["selected"]["stage"] == "physical_fit"
    # noise-free and physical already: the constraint costs nothing
    assert pfit.result["selected"]["effort_rmse_fit"] == pytest.approx(
        pfit.rms_error, rel=1e-3
    )
    assert set(pfit.result["selected"]["effort_rmse_fit_per_joint"]) == set(
        pfit.identif_config["active_joints"]
    )
    assert {s.stage: s.status for s in pfit.stages}["physical"] == "ok"
    v = pfit.verify(scope="execution")
    assert v.passed and v.selected_stage == "physical_fit"
    # export judges the links with the tolerance they were accepted with
    assert pfit.selected.psd_eig_tol == pytest.approx(-1e-8)


def test_physical_fit_prediction_is_full_regressor_times_theta(pfit):
    vm = pfit.result["validation_metrics"]
    assert vm["estimate_stage"] == "physical_fit"
    got = np.concatenate([vm["tau_identified_per_joint"][j] for j in vm["joint_names"]])
    want = pfit.dynamic_regressor @ pfit.selected.values
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_physical_fit_phi_base_equivalent_is_diagnostic(pfit):
    sel = pfit.selected
    assert sel.phi_base_equivalent is not None
    assert sel.base_residual_rel < 1e-3
    np.testing.assert_array_equal(pfit.result["base parameters values"], pfit.phi_base)


def _break_solver(monkeypatch):
    monkeypatch.setattr(
        pf,
        "_solve_core",
        lambda *a, **k: {"status": "error", "exception": "boom", "runtime_s": 0.0},
    )


def test_physical_fit_solver_error_is_a_hard_reject(model, traj, monkeypatch, tmp_path):
    pytest.importorskip("picos")
    _break_solver(monkeypatch)
    ident = _run(model, traj, select_stage="physical_fit")
    sel = ident.selected
    assert not sel.accepted and sel.stage == "none" and sel.requested == "physical_fit"
    assert {s.stage: s.status for s in ident.stages}["physical"] == "failed"
    for scope in ("execution", "prediction"):
        v = ident.verify(scope=scope)
        assert v.passed is False
        assert v.selected_stage == "none"
        assert any(
            c.name == "selected_stage_accepted" and c.status == "fail" for c in v.checks
        )
    # fit numbers stay under the legacy keys and in validation
    assert ident.result["validation_metrics"]["estimate_stage"] == "fit"
    assert ident.result["base parameters values"] is not None


def test_second_solver_is_used_after_a_non_optimal_first(model, traj, monkeypatch):
    pytest.importorskip("picos")
    real = pf._solve_core
    calls = []

    def core(p, mode, *, solver="cvxopt", **kw):
        calls.append(solver)
        if solver == "cvxopt":
            return {"status": "error", "exception": "first", "runtime_s": 0.0}
        return real(p, mode, solver="cvxopt", **kw)  # stand-in second backend

    monkeypatch.setattr(pf, "_solve_core", core)
    ident = _run(
        model,
        traj,
        select_stage="physical_fit",
        physical_fit={
            "enabled": False,
            "solver": "cvxopt",
            "second_solver": "other",
            "mass_min": 1e-6,
            "prior_weight": 1e-6,
            "max_seconds": None,
        },
    )
    assert calls == ["cvxopt", "other"]
    sel = ident.selected
    assert sel.accepted
    assert "cvxopt: error" in sel.reason and "other: optimal" in sel.reason
    assert sel.solvers == ["cvxopt", "other"]


def test_physical_fit_enabled_does_not_change_selection(model, traj):
    pytest.importorskip("picos")
    ident = _run(model, traj, physical_fit={"enabled": True})
    assert ident.selected is None
    assert "selected" not in ident.result
    assert ident.result["physical_fit"]["status"] == "accepted"
    assert {s.stage: s.status for s in ident.stages}["physical"] == "ok"
    assert ident.verify(scope="execution").selected_stage == "fit"


# ── reconstruction ──


def _recon_cfg(method, **extra):
    return {"reconstruction": {"enabled": True, "method": method, **extra}}


def test_reconstruction_nullspace_equals_fit_when_feasible(model, traj):
    ident = _run(model, traj, select_stage="reconstruction", **_recon_cfg("nullspace"))
    sel = ident.selected
    # the nullspace representative closest to the nominal prior reproduces the
    # fit exactly; with the nominal model it is also physical
    assert sel.accepted, sel.reason
    assert sel.effective_method == "nullspace"
    vm = ident.result["validation_metrics"]
    assert vm["estimate_stage"] == "reconstruction"
    got = np.concatenate([vm["tau_identified_per_joint"][j] for j in vm["joint_names"]])
    fit = ident.dynamic_regressor_base @ ident.phi_base
    np.testing.assert_allclose(got, fit, rtol=1e-9, atol=1e-9 * np.abs(fit).max())
    assert ident.result["reconstruction"]["effective_method"] == "nullspace"


def test_reconstruction_with_an_infeasible_link_is_rejected(model, traj, monkeypatch):
    ident = _run(model, traj, select_stage="reconstruction", **_recon_cfg("nullspace"))
    # make the nominal prior unphysical for one link and re-select
    name = f"m_{model.names[1]}"
    ident.standard_parameter[name] = -5.0
    ident._apply_reconstruction_if_enabled(
        {
            "M": ident._M_matrix,
            "phi_base": ident.phi_base,
            "params_r": ident._params_r_for_recon,
        }
    )
    sel = select_estimate(ident)
    assert not sel.accepted and sel.stage == "none"
    assert model.names[1] in sel.reason or "infeasible" in sel.reason


def test_reconstruction_sdp_without_picos_is_a_fallback_reject(
    model, traj, monkeypatch
):
    monkeypatch.setitem(sys.modules, "picos", None)
    ident = _run(model, traj, select_stage="reconstruction", **_recon_cfg("sdp"))
    sel = ident.selected
    assert not sel.accepted and sel.stage == "none"
    assert "requested sdp but nullspace was used" in sel.reason
    assert {s.stage: s.status for s in ident.stages}["physical"] == "fallback"
    assert ident.verify(scope="execution").passed is False


def test_reconstruction_auto_without_picos_may_be_nullspace(model, traj, monkeypatch):
    monkeypatch.setitem(sys.modules, "picos", None)
    ident = _run(model, traj, select_stage="reconstruction", **_recon_cfg("auto"))
    assert ident.selected.effective_method == "nullspace"
    assert "swapped" not in ident.selected.reason


def test_select_estimate_rejects_unknown_stage(model, traj):
    ident = _run(model, traj)
    ident.identif_config["select_stage"] = "projected"
    with pytest.raises(ValueError):
        select_estimate(ident)
