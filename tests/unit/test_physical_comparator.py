"""Tests for the private physical-consistency comparator."""

import json
import subprocess
import sys

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
pytest.importorskip("picos")

from figaroh.identification import _physical_comparator as pcmp  # noqa: E402
from figaroh.identification._physical_comparator import (  # noqa: E402
    FixedExtras,
    PhysicalPolicy,
    build_problem,
    diagnose_exact,
    solve_direct_effort_fit,
    solve_exact_reconstruction,
    solve_per_link_projection,
)

KEYS = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]


@pytest.fixture(scope="module")
def setup():
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    rng = np.random.default_rng(3)
    N, nv = 40, model.nv
    q = np.array([pin.randomConfiguration(model) for _ in range(N)])
    v = rng.uniform(-1, 1, (N, nv))
    a = rng.uniform(-3, 3, (N, nv))
    rows = [
        np.asarray(pin.computeJointTorqueRegressor(model, data, q[i], v[i], a[i]))
        for i in range(N)
    ]
    # joint-major stacking: row j * N + i
    Y = np.vstack([np.array([rows[i][j] for i in range(N)]) for j in range(nv)])
    joints = list(model.names[1:])
    names = [f"{k}_{j}" for j in joints for k in KEYS]
    truth = np.concatenate(
        [model.inertias[i + 1].toDynamicParameters() for i in range(nv)]
    )
    prior = truth * (1 + 0.1 * rng.standard_normal(truth.size))
    return Y, names, joints, truth, prior


def _problem(setup, tau=None, **kw):
    Y, names, joints, truth, prior = setup
    tau = Y @ truth if tau is None else tau
    return build_problem(
        Y, tau, names, joints, prior=prior, theta_truth=truth, **kw
    )


def test_noise_free_all_accepted(setup):
    p = _problem(setup)
    r1 = solve_exact_reconstruction(p)
    r2 = solve_direct_effort_fit(p)
    r3 = solve_per_link_projection(p)
    for r in (r1, r2, r3):
        assert r.solver_status == "optimal", r.exception
        assert r.accepted, r.feasibility
        assert r.fallback_used is False
    assert r1.base_residual["rel"] < 1e-6
    assert r2.base_residual["rel"] < 1e-6
    json.dumps([r.as_dict() for r in (r1, r2, r3)])


def test_negated_phi_infeasible(setup):
    p = _problem(setup)
    bad = pcmp.ComparatorProblem(
        **{**p.__dict__, "phi_ols": -p.phi_ols}
    )
    recs = diagnose_exact(bad)
    d1 = [r for r in recs if r.objective == "D1:phase1"][0]
    assert d1.objective_value < 0
    assert "infeasible" in d1.notes["label"]
    assert not solve_exact_reconstruction(bad).accepted


def test_solver_exception_is_error_not_fallback(setup, monkeypatch):
    import picos

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(picos.Problem, "solve", boom)
    r = solve_exact_reconstruction(_problem(setup))
    assert r.solver_status == "error"
    assert r.theta is None
    assert r.fallback_used is False
    assert not r.accepted
    assert r.as_dict()["fallback_used"] is False


def test_independent_check_catches_wrong_theta(setup, monkeypatch):
    p = _problem(setup)
    bad = p.theta_prior.copy()
    idx = pcmp._link_indices(p)[p.joint_names[0]]
    bad[idx["m"]] = -1.0
    monkeypatch.setattr(
        pcmp,
        "_solve_core",
        lambda *a, **k: {"status": "optimal", "theta": bad, "runtime_s": 0.0},
    )
    r = solve_exact_reconstruction(p)
    assert r.solver_status == "optimal"
    assert not r.accepted
    assert not r.feasibility[p.joint_names[0]]["ok"]


def test_removed_column_still_gets_lmi(setup):
    Y, names, joints, truth, prior = setup
    Y2 = Y.copy()
    Y2[:, names.index(f"Izz_{joints[-1]}")] = 0.0
    p = build_problem(Y2, Y2 @ truth, names, joints, prior=prior)
    assert p.removed_columns
    assert np.all(p.M_full[:, list(p.removed_columns)] == 0)
    assert set(pcmp._link_indices(p)) == set(joints)
    r = solve_exact_reconstruction(p)
    assert set(r.feasibility) == set(joints)


def test_frozen_extras_subtracted(setup):
    Y, names, joints, truth, prior = setup
    k = 5
    tau = Y @ truth
    ex = FixedExtras((names[k],), (float(truth[k]),), "truth")
    p = build_problem(Y, tau, names, joints, prior=prior, extras=ex)
    assert names[k] not in p.params_std
    assert np.allclose(p.tau, tau - Y[:, k] * truth[k])
    assert p.Y.shape[1] == len(names) - 1


def test_module_not_exported():
    import figaroh.identification as ident

    assert not hasattr(ident, "_physical_comparator") or (
        "_physical_comparator" not in getattr(ident, "__all__", [])
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, figaroh.identification as i;"
            "print('figaroh.identification._physical_comparator' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "False"
